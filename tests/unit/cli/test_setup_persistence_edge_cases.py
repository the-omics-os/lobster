"""Setup edge contracts; synthetic credentials, no installed/live-account claims."""

import json
import shlex

import pytest
import typer
from botocore.exceptions import ClientError
from rich.console import Console
from typer.testing import CliRunner

from lobster.cli_internal.commands.heavy import init_commands
from lobster.config import credentials
from lobster.config.workspace_config import WorkspaceProviderConfig

pytestmark = pytest.mark.no_auto_config
KEY = "omk_setup_edge_test"


@pytest.fixture(autouse=True)
def isolated_setup(monkeypatch, tmp_path):
    from lobster.config import global_config

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / "global-config")
    monkeypatch.setattr(credentials, "CREDENTIALS_DIR", tmp_path / "credentials")
    monkeypatch.setattr(
        credentials, "CREDENTIALS_FILE", tmp_path / "credentials/credentials.json"
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
        raise AssertionError("Setup attempted an unapproved prompt or install")

    monkeypatch.setattr(init_commands.Prompt, "ask", forbidden)
    monkeypatch.setattr(init_commands.Confirm, "ask", forbidden)
    monkeypatch.setattr(typer, "prompt", forbidden)
    monkeypatch.setattr(init_commands, "_ensure_provider_installed", forbidden)
    monkeypatch.setattr(init_commands, "_ensure_tui_installed", forbidden)
    monkeypatch.setattr(init_commands, "_install_extended_data", forbidden)
    monkeypatch.setattr(init_commands, "_check_and_prompt_install_packages", forbidden)


def init(*options):
    from lobster.cli import app

    return CliRunner().invoke(
        app, ["init", "--non-interactive", "--skip-extras", "--skip-ssl-test", *options]
    )


@pytest.mark.parametrize("malformed", [False, True])
@pytest.mark.parametrize("force", [False, True])
def test_json_only_workspace_requires_force_and_keeps_provider_backup(
    malformed, force, tmp_path
):
    workspace = tmp_path / ".lobster_workspace"
    workspace.mkdir()
    path = workspace / "provider_config.json"
    old = "{" if malformed else json.dumps({"global_provider": "anthropic"})
    path.write_text(old)
    assert not (tmp_path / ".env").exists()
    result = init("--omics-os-key", KEY, *(["--force"] if force else []))
    if force:
        assert result.exit_code == 0, result.output
        backups = list(workspace.glob("provider_config.json.backup.*"))
        assert len(backups) == 1
        assert backups[0].read_text() == old
        assert json.loads(path.read_text())["global_provider"] == "omics-os"
    else:
        assert result.exit_code != 0, result.output
        assert "--force" in result.output
        assert path.read_text() == old
        assert not (tmp_path / ".env").exists()
        assert list(workspace.glob("provider_config.json.backup.*")) == []


@pytest.mark.parametrize("backup_fails", [False, True])
def test_cloud_force_backs_up_both_artifacts_before_provider_write(
    backup_fails, monkeypatch, tmp_path
):
    import shutil

    workspace = tmp_path / ".lobster_workspace"
    workspace.mkdir()
    config = workspace / "provider_config.json"
    config.write_text('{"global_provider":"anthropic"}')
    env = tmp_path / ".env"
    env.write_text("LOBSTER_LLM_PROVIDER=anthropic\n")
    original_copy = shutil.copy2
    original_save = WorkspaceProviderConfig.save
    events = []

    def backup(source, destination):
        events.append("backup:" + source.name)
        if backup_fails and source.name == "provider_config.json":
            raise OSError("synthetic backup refusal")
        return original_copy(source, destination)

    def save(self, path):
        events.append("write:provider")
        assert env.read_text().find("omics-os") >= 0
        return original_save(self, path)

    monkeypatch.setattr(shutil, "copy2", backup)
    monkeypatch.setattr(WorkspaceProviderConfig, "save", save)
    result = init("--omics-os-key", KEY, "--force")
    if backup_fails:
        assert result.exit_code != 0
        assert events == ["backup:.env", "backup:provider_config.json"]
        assert env.read_text() == "LOBSTER_LLM_PROVIDER=anthropic\n"
        assert json.loads(config.read_text())["global_provider"] == "anthropic"
    else:
        assert result.exit_code == 0, result.output
        assert events == [
            "backup:.env",
            "backup:provider_config.json",
            "write:provider",
        ]


def test_json_only_force_blank_key_rejects_before_backup(tmp_path):
    workspace = tmp_path / ".lobster_workspace"
    workspace.mkdir()
    config = workspace / "provider_config.json"
    config.write_text("{")
    result = init("--omics-os-key", "", "--force")
    assert result.exit_code != 0
    assert config.read_text() == "{"
    assert list(workspace.glob("*.backup.*")) == []
    assert not (tmp_path / ".env").exists()


def test_required_cloud_provider_json_failure_exits_nonzero_without_saved_claim(
    monkeypatch,
):
    def fail(*args, **kwargs):
        raise OSError("synthetic provider write failure")

    monkeypatch.setattr(WorkspaceProviderConfig, "save", fail)
    result = init("--omics-os-key", KEY)
    assert result.exit_code != 0, result.output
    assert "partial" in result.output.lower()
    assert "saved, not checked" not in result.output.lower()
    assert "Configuration saved" not in result.output


@pytest.mark.parametrize("global_mode", [False, True])
def test_noninteractive_cloud_credential_write_failure_has_partial_diagnostic(
    global_mode, monkeypatch, tmp_path
):
    import builtins

    if global_mode:

        def fail(*args, **kwargs):
            raise OSError("synthetic credential write failure")

        monkeypatch.setattr(
            "lobster.config.global_config.save_global_credentials", fail
        )
    else:
        original = builtins.open

        def guarded(path, mode="r", *args, **kwargs):
            if str(path) == str(tmp_path / ".env") and "w" in mode:
                raise OSError("synthetic credential write failure")
            return original(path, mode, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", guarded)
    result = init("--omics-os-key", KEY, *(["--global"] if global_mode else []))
    assert result.exit_code != 0
    assert "partial" in result.output.lower()
    assert "saved, not checked" not in result.output.lower()


@pytest.mark.parametrize("global_mode", [False, True])
def test_cloud_wizard_required_persistence_failure_is_terminal(
    monkeypatch, tmp_path, capsys, global_mode
):
    def fail(*args, **kwargs):
        raise OSError("synthetic required write failure")

    if global_mode:
        monkeypatch.setattr(
            "lobster.config.global_config.save_global_credentials", fail
        )
    else:
        monkeypatch.setattr(WorkspaceProviderConfig, "save", fail)
    with pytest.raises(typer.Exit) as error:
        init_commands._postprocess_tui_init_result(
            {"provider": "omics-os", "api_key": KEY, "agents": []},
            workspace_path=tmp_path / ".lobster_workspace",
            env_path=tmp_path / ".env",
            global_config=global_mode,
            skip_extras=True,
        )
    assert error.value.exit_code == 1
    output = capsys.readouterr().out
    assert "partial" in output.lower()
    assert "saved, not checked" not in output.lower()


def test_cloud_go_postwrite_failure_does_not_restart_another_wizard(
    monkeypatch, capsys
):
    import inspect
    import sys

    from typer.models import ParameterInfo

    from lobster.ui.bridge import binary_finder, go_tui_bridge

    # Direct parent dispatch with synthetic TTY; not real PTY qualification.
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        binary_finder, "find_tui_binary", lambda: "/synthetic/lobster-tui"
    )
    monkeypatch.setattr(
        go_tui_bridge,
        "run_init_wizard",
        lambda binary: {"provider": "omics-os", "api_key": KEY, "agents": []},
    )

    def fail(*args, **kwargs):
        raise OSError("synthetic global credential write failure")

    monkeypatch.setattr("lobster.config.global_config.save_global_credentials", fail)
    kwargs = {
        name: (
            parameter.default.default
            if isinstance(parameter.default, ParameterInfo)
            else parameter.default
        )
        for name, parameter in inspect.signature(
            init_commands.init_impl
        ).parameters.items()
    }
    kwargs.update(
        global_config=True, skip_extras=True, skip_ssl_test=True, non_interactive=False
    )
    with pytest.raises(typer.Exit) as error:
        init_commands.init_impl(**kwargs)
    assert error.value.exit_code == 1
    output = capsys.readouterr().out.lower()
    assert "partial" in output
    assert "falling back" not in output
    assert "configuration saved" not in output
    assert "saved, not checked" not in output


def test_uv_tool_deferred_command_preserves_existing_extras_and_with_packages(
    monkeypatch, capsys
):
    from lobster.core import uv_tool_env

    info = uv_tool_env.UvToolEnvInfo(
        tool_name="lobster-ai",
        tool_extras=["full", "anthropic"],
        installed_packages=["lobster-ai", "lobster-ml"],
    )
    monkeypatch.setattr(uv_tool_env, "detect_uv_tool_env", lambda: info)
    monkeypatch.setattr(uv_tool_env, "is_uv_tool_env", lambda: True)
    monkeypatch.setattr(init_commands, "console", Console(width=2000))
    init_commands._print_deferred_agent_install(["data_expert_agent"])
    command = capsys.readouterr().out.split("Install later: ", 1)[1].strip()
    argv = shlex.split(command)
    assert "lobster-ai[anthropic,full]" in argv
    assert "--reinstall" in argv
    assert "lobster-ml" in argv
    assert "lobster-research" in argv


@pytest.mark.parametrize("override", [None, "omics-os"])
def test_status_respects_active_workspace_and_runtime_override(override, tmp_path):
    from lobster.cli_internal.commands.heavy.display_helpers import build_status_blocks

    current = tmp_path / ".lobster_workspace"
    WorkspaceProviderConfig(global_provider="anthropic").save(current)
    study = tmp_path / "study"
    WorkspaceProviderConfig(global_provider="openai").save(study)
    blocks = build_status_blocks(
        workspace_path=study, provider_override=override, compact=True
    )
    block = next(
        b for b in blocks if b.kind == "kv" and b.data.get("title") == "Initialization"
    )
    rows = dict(block.data["rows"])
    assert rows["Provider"] == (override or "openai")
    assert rows["Configuration Source"] == (
        "runtime flag --provider" if override else "workspace config"
    )
    assert "not checked" in rows["Account Access"].lower()


@pytest.mark.parametrize("command", ["/status", "/status-panel"])
def test_slash_status_callers_forward_active_client_context(
    command, monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from lobster.cli_internal.commands.heavy import slash_commands

    study = tmp_path / "study"
    WorkspaceProviderConfig(global_provider="openai").save(study)
    client = SimpleNamespace(workspace_path=study, provider_override="omics-os")
    rendered = []
    monkeypatch.setattr(slash_commands, "_is_protocol_output", lambda output: True)
    monkeypatch.setattr(
        slash_commands,
        "_render_structured_output",
        lambda output, blocks: rendered.extend(blocks),
    )
    slash_commands._execute_command(command, client, output=SimpleNamespace())
    block = next(
        b
        for b in rendered
        if b.kind == "kv" and b.data.get("title") == "Initialization"
    )
    rows = dict(block.data["rows"])
    assert rows["Provider"] == "omics-os"
    assert rows["Configuration Source"] == "runtime flag --provider"


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize(
    "payload", [{"error": "Forbidden"}, {"error": []}, ["Forbidden"], "Forbidden"]
)
def test_malformed_gateway_error_shape_keeps_cloud_diagnosis(status, payload):
    from lobster.config.providers.gateway_bedrock_client import GatewayBedrockClient

    client = GatewayBedrockClient(
        endpoint="https://app.omics-os.com", token_fn=lambda: KEY
    )
    with pytest.raises(ClientError) as error:
        client._handle_error(status, json.dumps(payload))
    message = error.value.response["Error"]["Message"]
    assert "Omics-OS Cloud" in message
    if status == 401:
        assert "API key rejected or expired" in message
    else:
        assert "create a new" not in message.lower()
