"""`lobster init` installs into the Python that runs Lobster, and says so when it fails.

Without an activated venv, a bare `uv pip install` stops with "No virtual
environment found", yet the wizard still ended with "Next: run lobster chat".
"""

import subprocess
import sys

import pytest
from rich.console import Console

from lobster.cli_internal.commands.heavy import init_commands
from lobster.cli_internal.commands.light import agent_commands


@pytest.fixture(autouse=True)
def _plain_env(monkeypatch):
    monkeypatch.setattr("lobster.core.uv_tool_env.is_uv_tool_env", lambda: False)
    monkeypatch.setattr(agent_commands, "_find_local_package", lambda _pkg: None)
    monkeypatch.setattr("shutil.which", lambda _name: "/bin/uv")
    monkeypatch.setattr(agent_commands, "FAILED_INSTALLS", [], raising=False)


def test_install_targets_the_running_interpreter(monkeypatch):
    calls = []

    def _run(cmd, **_kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(agent_commands.subprocess, "run", _run)

    ok, _ = agent_commands._uv_pip_install("chromadb>=1.0.0")

    assert ok
    assert calls == [
        ["/bin/uv", "pip", "install", "--python", sys.executable, "chromadb>=1.0.0"]
    ]


def test_a_failed_install_is_reported_instead_of_next_chat(monkeypatch, tmp_path):
    def _run(cmd, **_kwargs):
        raise subprocess.CalledProcessError(2, cmd, stderr="No virtual environment")

    monkeypatch.setattr(agent_commands.subprocess, "run", _run)
    console = Console(width=300, record=True)
    monkeypatch.setattr(init_commands, "console", console)

    ok, _ = agent_commands._uv_pip_install("lobster-research")
    init_commands._print_init_summary({"provider": "omics-os"}, tmp_path)

    text = console.export_text()
    assert not ok
    assert "lobster-research" in text
    assert "--python" in text and sys.executable in text
    assert "Next: run lobster chat" not in text


def test_a_clean_run_still_points_to_chat(monkeypatch, tmp_path):
    console = Console(width=300, record=True)
    monkeypatch.setattr(init_commands, "console", console)

    init_commands._print_init_summary({"provider": "omics-os"}, tmp_path)

    assert "Next: run lobster chat" in console.export_text()
    assert init_commands._failed_install_lines() == []


@pytest.mark.parametrize(
    ("version", "spec"), [("1.3.0", "~=1.3.0"), ("1.3.2+c09f12bb5", "~=1.3.0")]
)
def test_agent_packages_install_at_lobsters_own_minor_version(
    monkeypatch, version, spec
):
    calls = []

    def _run(cmd, **_kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(agent_commands.subprocess, "run", _run)
    monkeypatch.setattr("lobster.version.__version__", version)

    agent_commands._uv_pip_install("lobster-research")
    agent_commands._uv_pip_install("chromadb>=1.0.0")

    assert calls[0][-1] == f"lobster-research{spec}"
    assert calls[1][-1] == "chromadb>=1.0.0"


def test_the_vector_hint_keeps_the_extra_in_the_command(monkeypatch):
    # Rich read "[vector-search]" as markup and printed `uv pip install 'lobster-ai'`.
    console = Console(width=300, record=True)
    monkeypatch.setattr(init_commands, "console", console)

    init_commands._print_vector_backend_unavailable()

    assert "uv pip install 'lobster-ai[vector-search]'" in console.export_text()


def test_uninstall_targets_the_running_interpreter(monkeypatch):
    calls = []

    def _run(cmd, **_kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(agent_commands.subprocess, "run", _run)

    agent_commands._uv_pip_uninstall("lobster-research")

    assert calls[0][:5] == ["/bin/uv", "pip", "uninstall", "--python", sys.executable]
