"""`lobster init --global` writes nothing into the folder it was run from."""

import pytest
import typer
from typer.testing import CliRunner

from lobster.cli_internal.commands.heavy import init_commands
from lobster.config import credentials

pytestmark = pytest.mark.no_auto_config
KEY = "omk_global_init_test"


@pytest.fixture
def run_folder(monkeypatch, tmp_path):
    """An empty folder to run from, and a home of its own beside it."""
    from lobster.config import global_config

    home, folder = tmp_path / "home", tmp_path / "run-here"
    home.mkdir()
    folder.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(folder)
    monkeypatch.setattr(global_config, "CONFIG_DIR", home / ".config/lobster")
    monkeypatch.setattr(credentials, "CREDENTIALS_DIR", home / ".config/omics-os")
    monkeypatch.setattr(
        credentials, "CREDENTIALS_FILE", home / ".config/omics-os/credentials.json"
    )
    for name in (
        "OMICS_OS_API_KEY",
        "OMICS_OS_ENDPOINT",
        "LOBSTER_LLM_PROVIDER",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "LOBSTER_WORKSPACE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("lobster.core.uv_tool_env.is_uv_tool_env", lambda: False)

    def forbidden(*args, **kwargs):
        raise AssertionError("init asked a question or installed something")

    monkeypatch.setattr(init_commands.Prompt, "ask", forbidden)
    monkeypatch.setattr(init_commands.Confirm, "ask", forbidden)
    monkeypatch.setattr(typer, "prompt", forbidden)
    monkeypatch.setattr(init_commands, "_ensure_provider_installed", forbidden)
    monkeypatch.setattr(init_commands, "_ensure_tui_installed", forbidden)
    monkeypatch.setattr(init_commands, "_install_extended_data", forbidden)
    # The picked packages count as installed; installing is not under test.
    monkeypatch.setattr(
        init_commands,
        "_check_and_prompt_install_packages",
        lambda agents, *args, **kwargs: list(agents),
    )
    return folder


def test_global_wizard_result_writes_nothing_into_the_folder(run_folder):
    from lobster.ui.bridge.init_adapter import apply_tui_init_result

    apply_tui_init_result(
        {"provider": "omics-os", "api_key": KEY, "agents": ["research_agent"]},
        workspace_path=run_folder / ".lobster_workspace",
        env_path=run_folder / ".env",
        global_config=True,
    )
    assert sorted(p.name for p in run_folder.iterdir()) == []


def test_global_non_interactive_init_writes_nothing_into_the_folder(run_folder):
    from lobster.cli import app

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--global",
            "--non-interactive",
            "--skip-extras",
            "--skip-ssl-test",
            "--omics-os-key",
            KEY,
            "--agents",
            "research_agent",
        ],
    )
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in run_folder.iterdir()) == []
    assert "serve every folder that has no experts" in " ".join(result.output.split())


def test_global_go_wizard_writes_nothing_into_the_folder(run_folder, monkeypatch):
    from lobster.cli_internal.commands.heavy.init_commands import init_impl

    monkeypatch.setattr("sys.stdin", type("TTY", (), {"isatty": lambda self: True})())
    monkeypatch.setattr(
        "lobster.ui.bridge.binary_finder.find_tui_binary", lambda: "/fake/lobster-tui"
    )
    monkeypatch.setattr(
        "lobster.ui.bridge.go_tui_bridge.run_init_wizard",
        lambda *args, **kwargs: {
            "provider": "omics-os",
            "api_key": KEY,
            "agents": ["research_agent"],
            "cancelled": False,
        },
    )
    with pytest.raises(typer.Exit) as done:
        init_impl(
            global_config=True,
            force=False,
            non_interactive=False,
            skip_extras=True,
            skip_ssl_test=True,
            agents=None,
            preset=None,
            lamin=False,
            ui_mode="go",
        )
    assert done.value.exit_code == 0
    assert sorted(p.name for p in run_folder.iterdir()) == []


def test_global_questionary_wizard_writes_nothing_into_the_folder(
    run_folder, monkeypatch
):
    from lobster.cli_internal.commands.heavy.init_commands import init_impl

    monkeypatch.setattr("sys.stdin", type("TTY", (), {"isatty": lambda self: True})())
    monkeypatch.setattr("lobster.ui.bridge.binary_finder.find_tui_binary", lambda: None)
    monkeypatch.setattr(init_commands, "_ask_language", lambda **kwargs: None)
    monkeypatch.setattr(
        "lobster.ui.bridge.questionary_fallback.run_questionary_init",
        lambda *args, **kwargs: {
            "provider": "omics-os",
            "api_key": KEY,
            "agents": ["research_agent"],
            "cancelled": False,
        },
    )
    with pytest.raises(typer.Exit) as done:
        init_impl(
            global_config=True,
            force=False,
            non_interactive=False,
            skip_extras=True,
            skip_ssl_test=True,
            agents=None,
            preset=None,
            lamin=False,
            ui_mode="auto",
        )
    assert done.value.exit_code == 0
    assert sorted(p.name for p in run_folder.iterdir()) == []
