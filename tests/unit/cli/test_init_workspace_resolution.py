"""``lobster init`` must honour the standard workspace resolution order."""

import importlib
import json
from pathlib import Path

import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

from lobster.cli_internal.commands.heavy import init_commands
from lobster.core.config_resolver import ConfigResolver
from lobster.core.runtime.workspace import resolve_workspace

KEY = "sk-ant-synthetic-test-key"  # nosec B105 - synthetic value


@pytest.fixture
def dirs(monkeypatch, tmp_path):
    from lobster.config import credentials, global_config

    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)
    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / "global")
    monkeypatch.setattr(credentials, "CREDENTIALS_DIR", tmp_path / "creds")
    monkeypatch.setattr(
        credentials, "CREDENTIALS_FILE", tmp_path / "creds" / "credentials.json"
    )
    monkeypatch.setattr("lobster.core.uv_tool_env.is_uv_tool_env", lambda: False)
    for name in (
        "OMICS_OS_API_KEY",
        "OMICS_OS_ENDPOINT",
        "LOBSTER_LLM_PROVIDER",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "LOBSTER_WORKSPACE",
    ):
        monkeypatch.delenv(name, raising=False)
    return proj, tmp_path / "elsewhere"


def _init_non_interactive():
    from lobster.cli import app

    return CliRunner().invoke(
        app,
        [
            "init",
            "--non-interactive",
            "--skip-extras",
            "--skip-ssl-test",
            "--anthropic-key",
            KEY,
        ],
    )


def test_env_workspace_receives_config_and_cwd_keeps_env(monkeypatch, dirs):
    proj, elsewhere = dirs
    monkeypatch.setenv("LOBSTER_WORKSPACE", str(elsewhere))

    result = _init_non_interactive()

    assert result.exit_code == 0, result.output
    config = json.loads((elsewhere / "provider_config.json").read_text())
    assert config["global_provider"] == "anthropic"
    assert (proj / ".env").exists()
    assert not (elsewhere / ".env").exists()
    assert not (proj / ".lobster_workspace").exists()

    provider, _source = ConfigResolver(
        resolve_workspace(None, create=False)
    ).resolve_provider(runtime_override=None)
    assert provider == "anthropic"


def test_default_workspace_is_cwd_without_env_var(dirs):
    proj, _elsewhere = dirs

    result = _init_non_interactive()

    assert result.exit_code == 0, result.output
    assert (proj / ".lobster_workspace" / "provider_config.json").exists()
    assert (proj / ".env").exists()


def test_tui_path_uses_resolved_workspace_and_prints_summary(monkeypatch, dirs):
    proj, elsewhere = dirs
    monkeypatch.setenv("LOBSTER_WORKSPACE", str(elsewhere))
    monkeypatch.setattr(
        "sys.stdin", type("FakeTTY", (), {"isatty": lambda self: True})()
    )
    monkeypatch.setattr(
        "lobster.ui.bridge.binary_finder.find_tui_binary", lambda: "/fake/tui"
    )
    monkeypatch.setattr(
        "lobster.ui.bridge.go_tui_bridge.run_init_wizard",
        lambda *a, **k: {
            "provider": "anthropic",
            "api_key": KEY,
            "profile": "production",
            "model_id": "claude-test-model",
            "agents": [],
        },
    )
    console = Console(width=200, record=True)
    monkeypatch.setattr(init_commands, "console", console)

    with pytest.raises(typer.Exit) as exc:
        init_commands.init_impl(
            global_config=False,
            force=True,
            non_interactive=False,
            anthropic_key=None,
            bedrock_access_key=None,
            bedrock_secret_key=None,
            use_ollama=False,
            ollama_model=None,
            gemini_key=None,
            openai_key=None,
            profile=None,
            ncbi_key=None,
            cloud_key=None,
            cloud_endpoint=None,
            skip_ssl_test=False,
            ssl_verify=None,
            ssl_cert_path=None,
            agents=None,
            preset=None,
            auto_agents=False,
            agents_description=None,
            skip_docling=False,
            install_docling=False,
            install_vector_search=False,
            skip_extras=True,
            ui_mode="go",
        )

    assert exc.value.exit_code == 0
    assert (elsewhere / "provider_config.json").exists()
    assert (proj / ".env").exists()
    text = console.export_text()
    assert "anthropic (claude-test-model)" in text
    assert str(elsewhere) in text
    assert KEY not in text


def test_summary_shows_provider_model_path_and_never_the_key(monkeypatch, tmp_path):
    console = Console(width=200, record=True)
    monkeypatch.setattr(init_commands, "console", console)

    init_commands._print_init_summary(
        {"provider": "ollama", "ollama_model": "llama3", "api_key": KEY},
        tmp_path / ".lobster_workspace",
    )

    text = console.export_text()
    assert "ollama (llama3)" in text
    assert str(tmp_path / ".lobster_workspace") in text
    assert "lobster chat" in text or "cli.init.next_chat" in text
    assert KEY not in text


def test_summary_without_model_omits_parentheses(monkeypatch, tmp_path):
    console = Console(width=200, record=True)
    monkeypatch.setattr(init_commands, "console", console)

    init_commands._print_init_summary({"provider": "omics-os"}, Path(tmp_path))

    assert "omics-os (" not in console.export_text()


def test_npm_cross_install_offer_is_gone():
    assert not hasattr(init_commands, "_offer_npm_cross_install")


def test_npm_install_hint_is_built_from_package_constant():
    from lobster.cli_internal import npm_launcher

    assert npm_launcher.install_hint() == f"npm install -g {npm_launcher.NPM_PACKAGE}"
    assert npm_launcher.NPM_PACKAGE == "@omicsos/lobster"


def _init_with(flag, key, *extra):
    from lobster.cli import app

    return CliRunner().invoke(
        app,
        ["init", "--non-interactive", "--skip-extras", "--skip-ssl-test", flag, key]
        + list(extra),
    )


def test_reinit_from_other_directory_is_refused_without_force(
    monkeypatch, dirs, tmp_path
):
    proj, elsewhere = dirs
    monkeypatch.setenv("LOBSTER_WORKSPACE", str(elsewhere))
    assert _init_non_interactive().exit_code == 0
    before = (elsewhere / "provider_config.json").read_text()

    other = tmp_path / "dirB"
    other.mkdir()
    monkeypatch.chdir(other)
    result = _init_with("--openai-key", "sk-synthetic-openai-key")  # nosec B106

    assert (
        result.exit_code != 0
        or "rerun with --force" in result.output
        or ("--force" in result.output)
    )
    assert (elsewhere / "provider_config.json").read_text() == before
    assert not list(elsewhere.glob("provider_config.json.backup.*"))
    assert not (other / ".env").exists()


def test_reinit_from_other_directory_with_force_backs_up_and_writes(
    monkeypatch, dirs, tmp_path
):
    proj, elsewhere = dirs
    monkeypatch.setenv("LOBSTER_WORKSPACE", str(elsewhere))
    assert _init_non_interactive().exit_code == 0
    original = (elsewhere / "provider_config.json").read_text()

    other = tmp_path / "dirB"
    other.mkdir()
    monkeypatch.chdir(other)
    result = _init_with(
        "--openai-key", "sk-synthetic-openai-key", "--force"
    )  # nosec B106

    assert result.exit_code == 0, result.output
    backups = list(elsewhere.glob("provider_config.json.backup.*"))
    assert len(backups) == 1
    assert backups[0].read_text() == original
    config = json.loads((elsewhere / "provider_config.json").read_text())
    assert config["global_provider"] == "openai"


def test_unwritable_workspace_exits_cleanly(monkeypatch, dirs, tmp_path):
    blocker = tmp_path / "a-file"
    blocker.write_text("not a directory")
    monkeypatch.setenv("LOBSTER_WORKSPACE", str(blocker / "ws"))
    console = Console(width=1000, record=True)
    # Patch the module the CLI will import, not this file's import-time
    # reference: other tests purge lobster.cli* from sys.modules.
    live = importlib.import_module("lobster.cli_internal.commands.heavy.init_commands")
    monkeypatch.setattr(live, "console", console)

    result = _init_non_interactive()

    assert result.exit_code == 1
    output = console.export_text()
    assert str(blocker) in output
    assert "LOBSTER_WORKSPACE" in output
    assert "Traceback" not in output
    assert not isinstance(result.exception, OSError)


def test_summary_escapes_rich_markup_in_values(monkeypatch):
    console = Console(width=200, record=True)
    monkeypatch.setattr(init_commands, "console", console)

    init_commands._print_init_summary(
        {"provider": "ollama", "ollama_model": "m[/x]"}, Path("/a/[x]/ws/[/x]")
    )

    text = console.export_text()
    assert "/a/[x]/ws/[/x]" in text
    assert "ollama (m[/x])" in text
