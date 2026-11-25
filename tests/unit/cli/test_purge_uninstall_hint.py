"""``lobster purge`` ends with the uninstall command for the way Lobster was installed."""

import sys

import pytest
from typer.testing import CliRunner

from lobster.cli_internal.provider_hint import quote
from lobster.core import projects, self_upgrade
from lobster.core.self_upgrade import UpgradePlan


@pytest.fixture
def project(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("LOBSTER_WORKSPACE", raising=False)
    from lobster.config import global_config

    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / "cfg" / "lobster")
    root = tmp_path / "study"
    projects.init_project(root)
    return root


SPACED_PYTHON = "/opt/my envs/lobster/bin/python"


def purge_output(project, monkeypatch, method):
    monkeypatch.setattr(sys, "executable", SPACED_PYTHON)
    plan = UpgradePlan(method=method, command=[], display="")
    monkeypatch.setattr(self_upgrade, "detect_upgrade_plan", lambda: plan)
    from lobster.cli import app

    result = CliRunner().invoke(
        app, ["purge", "--scope", "workspace", "-w", str(project), "--force"]
    )
    assert result.exit_code == 0, result.output
    return result.output


def test_a_uv_tool_install_is_told_to_run_uv_tool_uninstall(project, monkeypatch):
    out = purge_output(project, monkeypatch, "uv-tool")

    assert "uv tool uninstall lobster-ai" in out
    assert "uv pip uninstall" not in out
    assert "pipx" not in out


def test_a_uv_venv_install_is_told_to_run_uv_pip_uninstall(project, monkeypatch):
    out = purge_output(project, monkeypatch, "uv-pip")

    # Names Lobster's own interpreter, so it works without activating the venv.
    assert f"uv pip uninstall --python {quote(SPACED_PYTHON)} lobster-ai" in out
    assert "uv tool uninstall" not in out


def test_a_pip_install_is_told_to_run_pip_uninstall(project, monkeypatch):
    out = purge_output(project, monkeypatch, "pip")

    assert f"{quote(SPACED_PYTHON)} -m pip uninstall lobster-ai" in out
    assert "uv pip uninstall" not in out
    assert "uv tool uninstall" not in out


def test_an_unknown_install_lists_each_way(project, monkeypatch):
    out = purge_output(project, monkeypatch, "manual")

    for cmd in (
        "uv tool uninstall lobster-ai",
        f"uv pip uninstall --python {quote(SPACED_PYTHON)} lobster-ai",
        f"{quote(SPACED_PYTHON)} -m pip uninstall lobster-ai",
    ):
        assert cmd in out
