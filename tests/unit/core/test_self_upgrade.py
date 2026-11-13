"""Tests for lobster.core.self_upgrade (the ``lobster upgrade`` core)."""

import json
import subprocess  # nosec B404 # test inspects subprocess call arguments
from pathlib import Path

import pytest
from rich.console import Console

from lobster.core import self_upgrade
from lobster.core.uv_tool_env import UvToolEnvInfo

_REAL_IS_EDITABLE = self_upgrade.is_editable_install


@pytest.fixture(autouse=True)
def _not_editable(monkeypatch):
    """The dev checkout is an editable install; default tests to a normal one."""
    monkeypatch.setattr(self_upgrade, "is_editable_install", lambda: False)


def _make_console() -> Console:
    return Console(record=True, width=120, no_color=True)


def _output(console: Console) -> str:
    # clear=False: repeated assertions in one test must not drain the buffer
    # (rich >= 13 defaults export_text(clear=True)).
    return console.export_text(clear=False)


def _fake_tool_env(extras=(), with_pkgs=(), prefix=None):
    return UvToolEnvInfo(
        tool_name="lobster-ai",
        tool_extras=list(extras),
        installed_packages=["lobster-ai"] + list(with_pkgs),
        prefix=prefix or Path("/fake/prefix"),
    )


class TestDetectUpgradePlan:
    def test_uv_tool_env_builds_upgrade_command_preserving_receipt(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            "lobster.core.uv_tool_env.detect_uv_tool_env",
            lambda: _fake_tool_env(
                extras=["full", "anthropic"], with_pkgs=["some-pkg"]
            ),
        )
        monkeypatch.setattr(
            self_upgrade.shutil, "which", lambda name: "/usr/local/bin/uv"
        )

        plan = self_upgrade.detect_upgrade_plan()

        assert plan.supported is True
        assert plan.method == "uv-tool"
        assert plan.command[:4] == ["/usr/local/bin/uv", "tool", "install", "--upgrade"]
        assert "lobster-ai[anthropic,full]" in plan.command
        assert "--with" in plan.command
        assert "some-pkg" in plan.command

    def test_uv_tool_env_without_uv_is_manual(self, monkeypatch):
        monkeypatch.setattr(
            "lobster.core.uv_tool_env.detect_uv_tool_env",
            lambda: _fake_tool_env(),
        )
        monkeypatch.setattr(self_upgrade.shutil, "which", lambda name: None)

        plan = self_upgrade.detect_upgrade_plan()

        assert plan.supported is False
        assert plan.method == "manual"
        assert "uv` is not on PATH" in plan.reason

    def test_venv_with_uv_targets_running_interpreter(self, monkeypatch, tmp_path):
        monkeypatch.setattr("lobster.core.uv_tool_env.detect_uv_tool_env", lambda: None)
        monkeypatch.setattr(
            self_upgrade.shutil, "which", lambda name: "/usr/local/bin/uv"
        )
        (tmp_path / "pyvenv.cfg").write_text("home = /usr/bin\n")
        monkeypatch.setattr(self_upgrade.sys, "prefix", str(tmp_path))

        plan = self_upgrade.detect_upgrade_plan()

        assert plan.method == "uv-pip"
        assert plan.command == [
            "/usr/local/bin/uv",
            "pip",
            "install",
            "--python",
            self_upgrade.sys.executable,
            "--upgrade",
            "lobster-ai",
        ]

    def test_venv_without_uv_falls_back_to_pip(self, monkeypatch, tmp_path):
        monkeypatch.setattr("lobster.core.uv_tool_env.detect_uv_tool_env", lambda: None)
        monkeypatch.setattr(self_upgrade.shutil, "which", lambda name: None)
        (tmp_path / "pyvenv.cfg").write_text("home = /usr/bin\n")
        monkeypatch.setattr(self_upgrade.sys, "prefix", str(tmp_path))

        plan = self_upgrade.detect_upgrade_plan()

        assert plan.method == "pip"
        assert plan.command == [
            self_upgrade.sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "lobster-ai",
        ]

    def test_unknown_install_is_manual(self, monkeypatch, tmp_path):
        monkeypatch.setattr("lobster.core.uv_tool_env.detect_uv_tool_env", lambda: None)
        # No pyvenv.cfg in tmp_path: not a virtualenv.
        monkeypatch.setattr(self_upgrade.sys, "prefix", str(tmp_path))

        plan = self_upgrade.detect_upgrade_plan()

        assert plan.supported is False
        assert plan.method == "manual"
        assert "no uv tool receipt, no virtualenv" in plan.reason


class TestRunUpgrade:
    def test_dry_run_prints_command_without_running(self, monkeypatch):
        monkeypatch.setattr("lobster.core.uv_tool_env.detect_uv_tool_env", lambda: None)
        # Do not depend on the test interpreter itself running in a virtualenv.
        monkeypatch.setattr(self_upgrade, "_is_virtualenv", lambda: True)
        monkeypatch.setattr(
            self_upgrade.shutil, "which", lambda name: "/usr/local/bin/uv"
        )

        def _must_not_run(*args, **kwargs):
            raise AssertionError("subprocess.run must not be called in dry-run")

        monkeypatch.setattr(self_upgrade.subprocess, "run", _must_not_run)

        console = _make_console()
        code = self_upgrade.run_upgrade(console=console, dry_run=True)

        assert code == 0
        assert "Dry run" in _output(console)
        # Rich wraps long lines at the console width; normalize before asserting.
        assert "--upgrade lobster-ai" in " ".join(_output(console).split())

    def test_unsupported_environment_exits_1_with_guidance(self, monkeypatch, tmp_path):
        monkeypatch.setattr("lobster.core.uv_tool_env.detect_uv_tool_env", lambda: None)
        monkeypatch.setattr(self_upgrade.sys, "prefix", str(tmp_path))

        console = _make_console()
        code = self_upgrade.run_upgrade(console=console)

        assert code == 1
        assert "no uv tool receipt, no virtualenv" in _output(console)
        assert "Update manually with:" in _output(console)

    def test_success_clears_cache_and_prints_restart_notice(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr("lobster.core.uv_tool_env.detect_uv_tool_env", lambda: None)
        monkeypatch.setattr(
            self_upgrade.shutil, "which", lambda name: "/usr/local/bin/uv"
        )
        (tmp_path / "pyvenv.cfg").write_text("home = /usr/bin\n")
        monkeypatch.setattr(self_upgrade.sys, "prefix", str(tmp_path))

        result = subprocess.CompletedProcess(args=[], returncode=0)
        monkeypatch.setattr(self_upgrade.subprocess, "run", lambda *a, **kw: result)
        cache_calls = []
        monkeypatch.setattr(
            "lobster.config.version_check.clear_update_cache",
            lambda: cache_calls.append(True),
        )

        console = _make_console()
        code = self_upgrade.run_upgrade(console=console)

        assert code == 0
        assert cache_calls == [True]
        assert "upgraded" in _output(console)
        assert "Restart" in _output(console)

    def test_failure_returns_subprocess_code_and_keeps_cache(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr("lobster.core.uv_tool_env.detect_uv_tool_env", lambda: None)
        monkeypatch.setattr(
            self_upgrade.shutil, "which", lambda name: "/usr/local/bin/uv"
        )
        (tmp_path / "pyvenv.cfg").write_text("home = /usr/bin\n")
        monkeypatch.setattr(self_upgrade.sys, "prefix", str(tmp_path))

        result = subprocess.CompletedProcess(args=[], returncode=2)
        monkeypatch.setattr(self_upgrade.subprocess, "run", lambda *a, **kw: result)
        cache_calls = []
        monkeypatch.setattr(
            "lobster.config.version_check.clear_update_cache",
            lambda: cache_calls.append(True),
        )

        console = _make_console()
        code = self_upgrade.run_upgrade(console=console)

        assert code == 2
        assert cache_calls == []
        assert "Upgrade failed" in _output(console)


class TestClearUpdateCache:
    def test_clear_update_cache_removes_file(self, tmp_path, monkeypatch):
        from lobster.config import version_check

        cache_file = tmp_path / "version-check.json"
        cache_file.write_text("{}")
        monkeypatch.setattr(version_check, "_get_cache_path", lambda: cache_file)

        version_check.clear_update_cache()

        assert not cache_file.exists()

    def test_clear_update_cache_is_safe_when_missing(self, tmp_path, monkeypatch):
        from lobster.config import version_check

        monkeypatch.setattr(
            version_check, "_get_cache_path", lambda: tmp_path / "absent.json"
        )

        version_check.clear_update_cache()  # must not raise


class _FakeDist:
    def __init__(self, direct_url):
        self._direct_url = direct_url

    def read_text(self, name):
        assert name == "direct_url.json"
        return self._direct_url


class TestEditableInstall:
    def _patch_dist(self, monkeypatch, direct_url):
        from importlib import metadata

        monkeypatch.setattr(
            metadata, "distribution", lambda name: _FakeDist(direct_url)
        )

    def test_detects_editable_direct_url(self, monkeypatch):
        self._patch_dist(
            monkeypatch,
            json.dumps({"url": "file:///src", "dir_info": {"editable": True}}),
        )
        assert _REAL_IS_EDITABLE() is True

    def test_non_editable_or_missing_direct_url(self, monkeypatch):
        self._patch_dist(
            monkeypatch,
            json.dumps({"url": "file:///src", "dir_info": {"editable": False}}),
        )
        assert _REAL_IS_EDITABLE() is False
        self._patch_dist(monkeypatch, None)
        assert _REAL_IS_EDITABLE() is False
        self._patch_dist(monkeypatch, "not json")
        assert _REAL_IS_EDITABLE() is False

    def test_missing_distribution_is_not_editable(self, monkeypatch):
        from importlib import metadata

        def _raise(name):
            raise metadata.PackageNotFoundError(name)

        monkeypatch.setattr(metadata, "distribution", _raise)
        assert _REAL_IS_EDITABLE() is False

    def test_real_upgrade_refused_without_running(self, monkeypatch, tmp_path):
        monkeypatch.setattr(self_upgrade, "is_editable_install", lambda: True)
        monkeypatch.setattr("lobster.core.uv_tool_env.detect_uv_tool_env", lambda: None)
        monkeypatch.setattr(
            self_upgrade.shutil, "which", lambda name: "/usr/local/bin/uv"
        )
        (tmp_path / "pyvenv.cfg").write_text("home = /usr/bin\n")
        monkeypatch.setattr(self_upgrade.sys, "prefix", str(tmp_path))

        def _must_not_run(*args, **kwargs):
            raise AssertionError("must not upgrade an editable install")

        monkeypatch.setattr(self_upgrade.subprocess, "run", _must_not_run)

        console = _make_console()
        code = self_upgrade.run_upgrade(console=console)

        assert code == 1
        text = " ".join(_output(console).split())
        assert "editable development install" in text
        assert "git pull" in text

    def test_dry_run_prints_command_and_note(self, monkeypatch, tmp_path):
        monkeypatch.setattr(self_upgrade, "is_editable_install", lambda: True)
        monkeypatch.setattr("lobster.core.uv_tool_env.detect_uv_tool_env", lambda: None)
        monkeypatch.setattr(
            self_upgrade.shutil, "which", lambda name: "/usr/local/bin/uv"
        )
        (tmp_path / "pyvenv.cfg").write_text("home = /usr/bin\n")
        monkeypatch.setattr(self_upgrade.sys, "prefix", str(tmp_path))

        console = _make_console()
        code = self_upgrade.run_upgrade(console=console, dry_run=True)

        text = " ".join(_output(console).split())
        assert code == 0
        assert "--upgrade lobster-ai" in text
        assert "editable development install" in text
