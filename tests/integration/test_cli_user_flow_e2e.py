from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


def _isolated_env(tmp_path: Path, *, hide_npm_cli: bool = True) -> dict[str, str]:
    home = tmp_path / "home"
    xdg_config = tmp_path / "xdg-config"
    xdg_cache = tmp_path / "xdg-cache"
    workspace = tmp_path / "workspace"
    for path in (home, xdg_config, xdg_cache, workspace):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(xdg_config),
            "XDG_CACHE_HOME": str(xdg_cache),
            "LOBSTER_WORKSPACE": str(workspace),
            "NO_COLOR": "1",
            "PYTHONUNBUFFERED": "1",
            "TERM": "xterm-256color",
        }
    )
    env.pop("LOBSTER_CLI_BINARY", None)
    env.pop("LOBSTER_ENDPOINT", None)
    env.pop("LOBSTER_CLOUD_KEY", None)
    env.pop("OMICS_OS_API_KEY", None)

    if hide_npm_cli:
        env["PATH"] = os.pathsep.join(
            [
                str(Path(sys.executable).resolve().parent),
                "/usr/bin",
                "/bin",
                "/usr/sbin",
                "/sbin",
            ]
        )

    return env


def _run_cli(
    args: list[str],
    tmp_path: Path,
    *,
    hide_npm_cli: bool = True,
    cwd: Path | None = None,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "lobster.cli", *args],
        cwd=str(cwd or REPO_ROOT),
        env=_isolated_env(tmp_path, hide_npm_cli=hide_npm_cli),
        input="",
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _combined_output(result: subprocess.CompletedProcess[str]) -> str:
    return f"{result.stdout}\n{result.stderr}"


def _assert_no_python_traceback(result: subprocess.CompletedProcess[str]) -> None:
    output = _combined_output(result)
    assert "Traceback (most recent call last)" not in output, output


def test_local_python_cli_can_hide_global_npm_and_find_go_binary(tmp_path: Path) -> None:
    env = _isolated_env(tmp_path, hide_npm_cli=True)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from lobster.cli_internal.npm_launcher import find_npm_binary; "
                "from lobster.cli_internal.go_tui_launcher import find_tui_binary_fast; "
                "from lobster.ui.bridge.binary_finder import find_tui_binary; "
                "print(json.dumps({"
                "'npm': find_npm_binary(), "
                "'go_fast': find_tui_binary_fast(), "
                "'go_bridge': find_tui_binary()"
                "}))"
            ),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, _combined_output(result)
    payload = json.loads(result.stdout)
    assert payload["npm"] is None

    for key in ("go_fast", "go_bridge"):
        binary = Path(payload[key])
        assert binary.is_file(), payload
        assert binary.name == "lobster-tui"
        assert REPO_ROOT in binary.parents


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            ["chat", "--ui", "go", "--no-intro"],
            "Go TUI requires an interactive terminal",
        ),
        (
            ["chat", "--ui", "ink"],
            "Invalid --ui value 'ink'",
        ),
    ],
)
def test_chat_ui_non_tty_user_paths_fail_cleanly(
    args: list[str],
    expected: str,
    tmp_path: Path,
) -> None:
    result = _run_cli(args, tmp_path)

    assert result.returncode != 0
    _assert_no_python_traceback(result)
    assert expected in _combined_output(result)


def test_init_force_non_tty_empty_workspace_fails_cleanly(tmp_path: Path) -> None:
    cwd = tmp_path / "empty-project"
    cwd.mkdir()

    result = _run_cli(["init", "--force"], tmp_path, cwd=cwd, timeout=45)

    assert result.returncode != 0
    _assert_no_python_traceback(result)
    assert "stdin is not a terminal" in _combined_output(result)


def test_cloud_chat_without_npm_binary_fails_cleanly(tmp_path: Path) -> None:
    result = _run_cli(["cloud", "chat"], tmp_path, hide_npm_cli=True)

    assert result.returncode != 0
    _assert_no_python_traceback(result)
    output = _combined_output(result)
    assert "Cloud TUI not installed" in output
    assert "npm install -g @omicsos/lobster" in output


def test_cloud_chat_with_stub_npm_binary_reaches_tty_guard(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "lobster-cli"
    stub.write_text(
        "#!/bin/sh\n"
        "echo 'stub npm lobster-cli should not run before TTY guard' >&2\n"
        "exit 99\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)

    env = _isolated_env(tmp_path, hide_npm_cli=True)
    env["PATH"] = os.pathsep.join([str(bin_dir), env["PATH"]])

    result = subprocess.run(
        [sys.executable, "-m", "lobster.cli", "cloud", "chat"],
        cwd=str(REPO_ROOT),
        env=env,
        input="",
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode != 0
    _assert_no_python_traceback(result)
    output = _combined_output(result)
    assert "Cloud chat requires an interactive terminal" in output
    assert "stub npm lobster-cli should not run" not in output


def test_init_manifest_fallback_and_preflight_import_contracts() -> None:
    from lobster.cli import _preflight_provider_check
    from lobster.cli_internal.commands.heavy.init_commands import _offer_npm_cross_install
    from lobster.ui.bridge.questionary_fallback import run_questionary_init
    from lobster.ui.wizard.manifest import build_init_manifest

    manifest = build_init_manifest()

    assert len(manifest.providers) >= 1
    assert run_questionary_init is not None
    assert _offer_npm_cross_install is not None
    assert _preflight_provider_check is not None


def test_cloud_endpoint_validation_contract() -> None:
    from lobster.cli_internal.commands.light.cloud_query import (
        CloudQueryError,
        _validate_endpoint,
    )

    with pytest.raises(CloudQueryError):
        _validate_endpoint("https://evil.com")

    with pytest.raises(CloudQueryError):
        _validate_endpoint("https://app.omics-os.com/api/v1")

    _validate_endpoint("https://app.omics-os.com")
    _validate_endpoint("https://stream.omics-os.com")
    _validate_endpoint("http://localhost:8000")
