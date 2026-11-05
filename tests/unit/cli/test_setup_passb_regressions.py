"""Installed Pass-B contracts; synthetic local credentials and receipt only."""

import importlib
import shlex

import pytest
from rich.console import Console
from typer.testing import CliRunner

from lobster.cli_internal.commands.heavy import init_commands
from lobster.config import credentials, global_config
from lobster.core import uv_tool_env

pytestmark = pytest.mark.no_auto_config
KEY = "omk_passb_synthetic"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / "global")
    monkeypatch.setattr(credentials, "CREDENTIALS_DIR", tmp_path / "profiles")
    monkeypatch.setattr(
        credentials, "CREDENTIALS_FILE", tmp_path / "profiles/credentials.json"
    )
    monkeypatch.setattr(init_commands, "console", Console(width=2000))
    for name in (
        "OMICS_OS_API_KEY",
        "OMICS_OS_ENDPOINT",
        "LOBSTER_LLM_PROVIDER",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "LOBSTER_WORKSPACE",
    ):
        monkeypatch.delenv(name, raising=False)


def invoke(*options):
    from lobster.cli import app

    return CliRunner().invoke(
        app, ["init", "--non-interactive", "--skip-extras", "--skip-ssl-test", *options]
    )


def receipt(monkeypatch, tmp_path):
    prefix = tmp_path / "tool-prefix"
    prefix.mkdir()
    (prefix / "uv-receipt.toml").write_text(
        '[tool]\nname = "lobster-ai"\nrequirements = [\n'
        '  { name = "lobster-ai", extras = ["anthropic", "full"] },\n'
        '  { name = "lobster-proteomics" },\n'
        '  { name = "custom-helper" },\n]\n'
    )
    monkeypatch.setattr(uv_tool_env.sys, "prefix", str(prefix))
    monkeypatch.setattr(uv_tool_env.shutil, "which", lambda name: "uv")


@pytest.mark.parametrize("global_mode", [False, True])
def test_saved_python_key_without_env_satisfies_repeat_init(
    global_mode, monkeypatch, tmp_path
):
    monkeypatch.setattr(uv_tool_env, "is_uv_tool_env", lambda: False)
    options = ["--global"] if global_mode else []
    first = invoke(*options, "--omics-os-key", KEY)
    assert first.exit_code == 0, first.output
    key_file = (
        global_config.get_global_credentials_path()
        if global_mode
        else tmp_path / ".env"
    )
    original = key_file.read_bytes()
    assert credentials.get_api_key() is None

    second = invoke(*options, "--use-omics-os")

    assert second.exit_code == 0, second.output
    assert "already configured for Omics-OS Cloud" in second.output
    assert key_file.read_bytes() == original
    assert KEY not in second.output


@pytest.mark.parametrize("global_mode", [False, True])
def test_explicit_blank_never_falls_back_to_saved_python_key(
    global_mode, monkeypatch, tmp_path
):
    monkeypatch.setattr(uv_tool_env, "is_uv_tool_env", lambda: False)
    options = ["--global"] if global_mode else []
    assert invoke(*options, "--omics-os-key", KEY).exit_code == 0
    key_file = (
        global_config.get_global_credentials_path()
        if global_mode
        else tmp_path / ".env"
    )
    original = key_file.read_bytes()
    result = invoke(*options, "--omics-os-key", "", "--force")
    assert result.exit_code == 1
    assert "API key is required" in result.output
    assert key_file.read_bytes() == original
    assert not list(key_file.parent.glob(key_file.name + ".backup.*"))


def test_skip_extras_suppresses_entire_tool_handoff(monkeypatch, tmp_path, capsys):
    receipt(monkeypatch, tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("skip-extras attempted a package handoff")

    monkeypatch.setattr(init_commands, "_build_uv_tool_init_command", forbidden)
    monkeypatch.setattr("rich.prompt.Confirm.ask", forbidden)
    init_commands._uv_tool_env_handoff(
        "anthropic", ["research_agent"], skip_extras=True
    )
    assert "Package installation needed" not in capsys.readouterr().out


def test_nonskip_tool_handoff_displays_receipt_preserving_command(
    monkeypatch, tmp_path, capsys
):
    receipt(monkeypatch, tmp_path)
    monkeypatch.setattr("rich.prompt.Confirm.ask", lambda *args, **kwargs: False)
    init_commands._uv_tool_env_handoff(
        "anthropic", ["research_agent"], skip_extras=False
    )
    output = capsys.readouterr().out
    assert "lobster-ai[anthropic,full]" in output
    assert "--with custom-helper" in output
    assert "--with lobster-proteomics" in output
    assert "--with lobster-research" in output
    command = init_commands._build_uv_tool_init_command("anthropic", ["research_agent"])
    assert "lobster-ai[anthropic,full]" in command
    assert set(command[command.index("lobster-ai[anthropic,full]") + 1 :]) >= {
        "custom-helper",
        "lobster-proteomics",
        "lobster-research",
    }
    assert shlex.split(shlex.join(command)) == command


def test_global_saved_key_can_configure_a_fresh_workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(uv_tool_env, "is_uv_tool_env", lambda: False)
    assert invoke("--global", "--omics-os-key", KEY).exit_code == 0
    result = invoke("--use-omics-os")
    assert result.exit_code == 0, result.output
    assert f"OMICS_OS_API_KEY={KEY}" in (tmp_path / ".env").read_text()
    assert (tmp_path / ".env").stat().st_mode & 0o777 == 0o600
    assert KEY not in result.output


@pytest.mark.parametrize("selected", ["workspace", "environment", "explicit"])
def test_force_uses_credential_priority_and_backs_up_both_artifacts(
    selected, monkeypatch, tmp_path
):
    monkeypatch.setattr(uv_tool_env, "is_uv_tool_env", lambda: False)
    assert invoke("--global", "--omics-os-key", "omk_global").exit_code == 0
    assert invoke("--omics-os-key", "omk_workspace").exit_code == 0
    env_path = tmp_path / ".env"
    provider_path = tmp_path / ".lobster_workspace/provider_config.json"
    original = {
        env_path: env_path.read_bytes(),
        provider_path: provider_path.read_bytes(),
    }
    options = ["--use-omics-os", "--force"]
    expected = "omk_workspace"
    if selected in {"environment", "explicit"}:
        monkeypatch.setenv("OMICS_OS_API_KEY", "omk_environment")
        expected = "omk_environment"
    if selected == "explicit":
        options += ["--omics-os-key", "omk_explicit"]
        expected = "omk_explicit"
    result = invoke(*options)
    assert result.exit_code == 0, result.output
    assert f"OMICS_OS_API_KEY={expected}" in env_path.read_text()
    for path, data in original.items():
        backups = list(path.parent.glob(path.name + ".backup.*"))
        assert len(backups) == 1
        assert backups[0].read_bytes() == data
    assert expected not in result.output


def test_saved_active_python_profile_precedes_local_saved_files(monkeypatch, tmp_path):
    monkeypatch.setattr(uv_tool_env, "is_uv_tool_env", lambda: False)
    assert invoke("--global", "--omics-os-key", "omk_global").exit_code == 0
    assert invoke("--omics-os-key", "omk_workspace").exit_code == 0
    monkeypatch.setattr(credentials, "get_api_key", lambda: "omk_profile")
    result = invoke("--use-omics-os", "--force")
    assert result.exit_code == 0, result.output
    assert "OMICS_OS_API_KEY=omk_profile" in (tmp_path / ".env").read_text()
    assert "omk_profile" not in result.output


def test_go_parent_does_not_offer_a_handoff_under_skip(monkeypatch, tmp_path, capsys):
    import inspect
    import sys

    import typer
    from typer.models import ParameterInfo

    from lobster.ui.bridge import binary_finder, go_tui_bridge

    # Actual parent dispatch with synthetic TTY; not actual Go/PTY qualification.
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    current_init = importlib.import_module(
        "lobster.cli_internal.commands.heavy.init_commands"
    )
    receipt(monkeypatch, tmp_path)
    monkeypatch.setattr(
        binary_finder, "find_tui_binary", lambda: "/synthetic/lobster-tui"
    )
    monkeypatch.setattr(
        go_tui_bridge,
        "run_init_wizard",
        lambda *args, **kwargs: {
            "provider": "omics-os",
            "api_key": KEY,
            "agents": ["research_agent"],
        },
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("skip-extras parent offered a handoff")

    monkeypatch.setattr(current_init, "_uv_tool_env_handoff", forbidden)
    kwargs = {
        name: (
            parameter.default.default
            if isinstance(parameter.default, ParameterInfo)
            else parameter.default
        )
        for name, parameter in inspect.signature(
            current_init.init_impl
        ).parameters.items()
    }
    kwargs.update(
        global_config=True, skip_extras=True, skip_ssl_test=True, non_interactive=False
    )
    with pytest.raises(typer.Exit) as error:
        current_init.init_impl(**kwargs)
    assert error.value.exit_code == 0
    output = capsys.readouterr().out
    assert "Package installation needed" not in output
    assert "--reinstall" in output
    assert "lobster-ai[anthropic,full]" in output
    assert KEY not in output


def test_receipt_command_builder_retains_argv_extras_and_supplements(
    monkeypatch, tmp_path
):
    receipt(monkeypatch, tmp_path)
    command = init_commands._build_uv_tool_init_command("anthropic", ["research_agent"])
    assert "lobster-ai[anthropic,full]" in command
    assert {"custom-helper", "lobster-proteomics", "lobster-research"} <= set(command)


def test_cloud_create_key_guidance_and_help_use_verified_account_route():
    from lobster.cli import app
    from lobster.config.cloud_setup import OMICS_OS_AUTH_GUIDANCE

    assert "https://app.omics-os.com/account" in OMICS_OS_AUTH_GUIDANCE
    result = CliRunner().invoke(app, ["cloud", "login", "--help"])
    assert result.exit_code == 0
    assert "app.omics-os.com/account" in result.output
    assert "settings/api-keys" not in result.output
