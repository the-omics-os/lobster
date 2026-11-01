"""Local setup contracts: no browser login and no hidden machine prompts.

Keys and profiles here are synthetic. These tests do not establish live account
access, installed-wheel compatibility, or an interactive terminal acceptance run.
"""

import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import typer
from typer.testing import CliRunner

from lobster.cli_internal.commands.heavy import init_commands
from lobster.cli_internal.commands.light import cloud_commands
from lobster.config import credentials
from lobster.config.providers.omics_os_provider import OmicsOSProvider

pytestmark = pytest.mark.no_auto_config
KEY = "omk_local_setup_test"
API_KEYS_URL = "https://app.omics-os.com/account"
NPM_PACKAGE = "@omicsos/lobster"


@pytest.fixture(autouse=True)
def isolated_setup(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from lobster.config import global_config

    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / "global-config")
    monkeypatch.setattr(credentials, "CREDENTIALS_DIR", tmp_path / "credentials")
    monkeypatch.setattr(
        credentials, "CREDENTIALS_FILE", tmp_path / "credentials" / "credentials.json"
    )
    for name in (
        "OMICS_OS_API_KEY",
        "OMICS_OS_ENDPOINT",
        "LOBSTER_LLM_PROVIDER",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("lobster.core.uv_tool_env.is_uv_tool_env", lambda: False)


@pytest.fixture
def forbid_interactive_and_install(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Machine setup attempted a hidden prompt or install")

    monkeypatch.setattr(init_commands.Prompt, "ask", forbidden)
    monkeypatch.setattr(init_commands.Confirm, "ask", forbidden)
    monkeypatch.setattr(typer, "prompt", forbidden)
    monkeypatch.setattr(init_commands, "_ensure_provider_installed", forbidden)
    monkeypatch.setattr(init_commands, "_install_extended_data", forbidden)
    monkeypatch.setattr(init_commands, "_ensure_tui_installed", forbidden)
    monkeypatch.setattr(init_commands, "_check_and_prompt_install_packages", forbidden)


def invoke_init(*options):
    from lobster.cli import app

    return CliRunner().invoke(
        app,
        ["init", "--non-interactive", "--skip-extras", "--skip-ssl-test", *options],
    )


def assert_supported_guidance(output):
    assert "app.omics-os.com" in output
    assert NPM_PACKAGE in output
    assert "cloud login --device" in output


def assert_cloud_config(tmp_path, expected_key):
    config = json.loads(
        (tmp_path / ".lobster_workspace" / "provider_config.json").read_text()
    )
    # Configuration is checked through its actual persisted public schema.
    assert config["global_provider"] == "omics-os"
    text = (tmp_path / ".env").read_text()
    assert "LOBSTER_LLM_PROVIDER=omics-os" in text
    assert f"OMICS_OS_API_KEY={expected_key}" in text
    assert KEY not in str(config)


@pytest.mark.parametrize("route", ["explicit", "environment", "saved-profile"])
def test_machine_cloud_setup_saves_actual_provider_without_prompts(
    route, monkeypatch, tmp_path, forbid_interactive_and_install
):
    if route == "explicit":
        options = ["--omics-os-key", KEY]
    else:
        options = ["--use-omics-os"]
        if route == "environment":
            monkeypatch.setenv("OMICS_OS_API_KEY", KEY)
        else:
            credentials.save_credentials(
                {
                    "auth_mode": "api_key",
                    "api_key": KEY,
                    "endpoint": "https://app.omics-os.com",
                }
            )
    result = invoke_init(*options)
    assert result.exit_code == 0, result.output
    assert_cloud_config(tmp_path, KEY)
    assert KEY not in result.output


@pytest.mark.parametrize("key", [None, "", "   "])
def test_machine_cloud_missing_key_fails_clearly_without_prompts_or_config(
    key, monkeypatch, tmp_path, forbid_interactive_and_install
):
    if key is not None:
        monkeypatch.setenv("OMICS_OS_API_KEY", key)
    result = invoke_init("--use-omics-os")
    assert result.exit_code != 0
    assert "API key" in result.output
    assert "--omics-os-key" in result.output
    assert not (tmp_path / ".env").exists()
    assert not (tmp_path / ".lobster_workspace" / "provider_config.json").exists()


def test_machine_explicit_blank_key_does_not_fall_back_to_environment(
    monkeypatch, tmp_path, forbid_interactive_and_install
):
    monkeypatch.setenv("OMICS_OS_API_KEY", KEY)
    result = invoke_init("--omics-os-key", "   ")
    assert result.exit_code != 0
    assert "API key" in result.output
    assert not (tmp_path / ".env").exists()


@pytest.mark.parametrize("key", [KEY, "  " + KEY + "  "])
def test_machine_explicit_key_is_trimmed(key, tmp_path, forbid_interactive_and_install):
    result = invoke_init("--omics-os-key", key)
    assert result.exit_code == 0, result.output
    assert_cloud_config(tmp_path, KEY)


def test_machine_multiple_providers_preserve_anthropic_priority(
    tmp_path, forbid_interactive_and_install
):
    result = invoke_init("--anthropic-key", "sk-ant-unit-test", "--omics-os-key", KEY)
    assert result.exit_code == 0, result.output
    config = json.loads(
        (tmp_path / ".lobster_workspace" / "provider_config.json").read_text()
    )
    assert config["global_provider"] == "anthropic"
    text = (tmp_path / ".env").read_text()
    assert "ANTHROPIC_API_KEY=sk-ant-unit-test" in text
    assert "LOBSTER_LLM_PROVIDER=omics-os" not in text
    assert "Warning" in result.output


def test_legacy_cloud_key_alone_does_not_select_gateway_provider(
    tmp_path, forbid_interactive_and_install
):
    result = invoke_init("--cloud-key", "legacy_premium_key")
    assert result.exit_code != 0
    assert not (tmp_path / ".env").exists()


def test_env_only_local_gateway_configuration_still_works_without_init(
    monkeypatch, tmp_path
):
    from lobster.core.config_resolver import ConfigResolver

    monkeypatch.setenv("LOBSTER_LLM_PROVIDER", "omics-os")
    monkeypatch.setenv("OMICS_OS_API_KEY", KEY)
    ConfigResolver.reset_instance()
    try:
        resolver = ConfigResolver.get_instance(workspace_path=None)
        provider, source = resolver.resolve_provider()
        assert provider == "omics-os"
        assert "environment" in source.lower()
        assert OmicsOSProvider().is_configured()
        assert credentials.get_api_key() == KEY
        assert not (tmp_path / ".env").exists()
    finally:
        ConfigResolver.reset_instance()


def test_python_login_without_key_only_redirects_without_browser_or_listener(
    monkeypatch, tmp_path
):
    import http.server
    import webbrowser

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported browser/listener/prompt was attempted")

    monkeypatch.setattr(webbrowser, "open", forbidden)
    monkeypatch.setattr(http.server, "HTTPServer", forbidden)
    monkeypatch.setattr(typer, "prompt", forbidden)
    result = CliRunner().invoke(cloud_commands.cloud_app, ["login"])
    assert result.exit_code != 0
    assert_supported_guidance(result.output)
    assert "--api-key" in result.output
    assert not credentials.CREDENTIALS_FILE.exists()


def test_init_login_helper_returns_false_with_supported_guidance_only(
    monkeypatch, capsys
):
    import http.server
    import webbrowser

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported browser/listener was attempted")

    monkeypatch.setattr(webbrowser, "open", forbidden)
    monkeypatch.setattr(http.server, "HTTPServer", forbidden)
    assert cloud_commands.attempt_login_for_init() is False
    assert_supported_guidance(capsys.readouterr().out)
    assert not credentials.CREDENTIALS_FILE.exists()


def test_provider_help_explains_supported_local_and_device_paths():
    help_text = OmicsOSProvider().get_configuration_help()
    assert_supported_guidance(help_text)
    assert "OMICS_OS_API_KEY" in help_text
    assert "--api-key" in help_text


def test_provider_missing_key_error_does_not_offer_bare_browser_login():
    with pytest.raises(ValueError) as error:
        OmicsOSProvider().create_chat_model(
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        )
    assert_supported_guidance(str(error.value))
    assert "OMICS_OS_API_KEY" in str(error.value)


def test_manual_api_key_login_preserves_v2_profiles_and_uses_allowed_endpoint(
    monkeypatch,
):
    credentials.save_credentials(
        {
            "version": 2,
            "active_profile": "default",
            "profiles": {
                "default": {"auth_mode": "api_key", "api_key": "omk_old"},
                "other": {"auth_mode": "api_key", "api_key": "omk_sibling"},
            },
        }
    )
    requests = []
    real_client = httpx.Client

    def response(request):
        requests.append(request)
        return httpx.Response(
            200, json={"tier": "free", "budget": {}, "user_id": "test-user"}
        )

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(response), **kwargs),
    )
    result = CliRunner().invoke(cloud_commands.cloud_app, ["login", "--api-key", KEY])
    assert result.exit_code == 0, result.output
    assert len(requests) == 1
    assert str(requests[0].url) == "https://app.omics-os.com/api/v1/gateway/usage"
    assert requests[0].headers["Authorization"] == f"Bearer {KEY}"
    raw = credentials.load_credentials()
    assert raw["profiles"]["default"]["api_key"] == KEY
    assert raw["profiles"]["other"]["api_key"] == "omk_sibling"
    assert KEY not in result.output


def test_manual_api_key_login_refuses_untrusted_endpoint_before_http(monkeypatch):
    monkeypatch.setenv("OMICS_OS_ENDPOINT", "https://not-an-allowed-endpoint.example")

    def forbidden(*args, **kwargs):
        raise AssertionError("Token was sent to an untrusted endpoint")

    monkeypatch.setattr(httpx, "Client", forbidden)
    result = CliRunner().invoke(cloud_commands.cloud_app, ["login", "--api-key", KEY])
    assert result.exit_code != 0
    assert "allowlist" in result.output
    assert not credentials.CREDENTIALS_FILE.exists()


def init_kwargs():
    return {
        name: (
            param.default.default
            if isinstance(param.default, typer.models.OptionInfo)
            else param.default
        )
        for name, param in inspect.signature(init_commands.init_impl).parameters.items()
    }


def test_classic_cloud_wizard_does_not_attempt_browser_before_key(
    monkeypatch, tmp_path
):
    import http.server

    calls = []

    def forbidden_listener(*args, **kwargs):
        calls.append("browser-listener")
        raise AssertionError("Classic Cloud wizard attempted unsupported browser login")

    monkeypatch.setattr(http.server, "HTTPServer", forbidden_listener)
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    answers = iter(["1", KEY])
    monkeypatch.setattr(
        init_commands.Prompt, "ask", lambda *args, **kwargs: next(answers)
    )
    monkeypatch.setattr(init_commands.Confirm, "ask", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        init_commands,
        "_perform_agent_selection_interactive",
        lambda *args, **kwargs: ([], None),
    )
    options = init_kwargs()
    options.update(ui_mode="classic", skip_extras=True, skip_ssl_test=True)
    try:
        init_commands.init_impl(**options)
    except typer.Exit as error:
        assert error.exit_code == 0
    assert calls == []
    assert_cloud_config(tmp_path, KEY)


def test_go_postprocess_empty_cloud_key_cannot_become_configured_success(
    monkeypatch, tmp_path, capsys
):
    import http.server

    def forbidden(*args, **kwargs):
        raise AssertionError("Empty Cloud setup attempted a browser or hidden prompt")

    monkeypatch.setattr(http.server, "HTTPServer", forbidden)
    monkeypatch.setattr(init_commands.Prompt, "ask", forbidden)
    with pytest.raises((typer.Exit, ValueError)):
        init_commands._postprocess_tui_init_result(
            {"provider": "omics-os", "api_key": "", "agents": []},
            workspace_path=tmp_path / ".lobster_workspace",
            env_path=tmp_path / ".env",
            global_config=False,
            skip_extras=True,
        )
    assert not (tmp_path / ".env").exists()
    assert_supported_guidance(capsys.readouterr().out)


def test_questionary_cloud_wizard_requests_key_without_browser(monkeypatch, capsys):
    import http.server
    import sys
    from types import ModuleType

    from lobster.ui.bridge.questionary_fallback import run_questionary_init
    from lobster.ui.wizard.manifest import ProviderDef, WizardManifest

    # The optional questionary distribution is not required for this adapter
    # unit test. This synthetic dialog surface is NOT an installed-UI proof.
    questionary = ModuleType("questionary")
    questionary.Choice = lambda title, **kwargs: SimpleNamespace(title=title, **kwargs)
    monkeypatch.setitem(sys.modules, "questionary", questionary)
    calls = []
    prompts = []

    def listener(*args, **kwargs):
        calls.append("browser-listener")
        raise AssertionError("Questionary Cloud wizard attempted browser login")

    def password(prompt, **kwargs):
        prompts.append(prompt)
        return SimpleNamespace(ask=lambda: KEY)

    monkeypatch.setattr(http.server, "HTTPServer", listener)
    monkeypatch.setattr(
        questionary,
        "select",
        lambda *args, **kwargs: SimpleNamespace(ask=lambda: "omics-os"),
        raising=False,
    )
    monkeypatch.setattr(questionary, "password", password, raising=False)
    monkeypatch.setattr(
        questionary,
        "checkbox",
        lambda *args, **kwargs: SimpleNamespace(ask=lambda: []),
        raising=False,
    )
    monkeypatch.setattr(
        questionary,
        "confirm",
        lambda *args, **kwargs: SimpleNamespace(ask=lambda: False),
        raising=False,
    )
    manifest = WizardManifest(
        providers=[
            ProviderDef(
                name="omics-os",
                display_name="Omics-OS Cloud",
                model_selection="managed",
            )
        ]
    )
    result = run_questionary_init(manifest)
    assert calls == []
    assert result["provider"] == "omics-os"
    assert result["api_key"] == KEY
    assert_supported_guidance(capsys.readouterr().out + " ".join(prompts))


def seed_existing_config(tmp_path, provider="anthropic", key="sk-ant-old-test"):
    env_path = tmp_path / ".env"
    env_name = "OMICS_OS_API_KEY" if provider == "omics-os" else "ANTHROPIC_API_KEY"
    text = f"LOBSTER_LLM_PROVIDER={provider}\n{env_name}={key}\n"
    env_path.write_text(text)
    workspace = tmp_path / ".lobster_workspace"
    workspace.mkdir(exist_ok=True)
    (workspace / "provider_config.json").write_text(
        json.dumps({"global_provider": provider})
    )
    return text


@pytest.mark.parametrize("force", [False, True])
def test_missing_cloud_key_is_rejected_before_existing_config_and_backup(
    force, tmp_path, forbid_interactive_and_install
):
    old = seed_existing_config(tmp_path)
    options = ["--use-omics-os"] + (["--force"] if force else [])
    result = invoke_init(*options)
    assert result.exit_code != 0
    assert "API key" in result.output
    assert (tmp_path / ".env").read_text() == old
    assert list(tmp_path.glob(".env.backup.*")) == []


def test_existing_other_provider_cloud_request_requires_force(
    monkeypatch, tmp_path, forbid_interactive_and_install
):
    old = seed_existing_config(tmp_path)
    monkeypatch.setenv("OMICS_OS_API_KEY", KEY)
    result = invoke_init("--use-omics-os")
    assert result.exit_code != 0
    assert "existing configuration found" in result.output.lower()
    assert "--force" in result.output
    assert (tmp_path / ".env").read_text() == old
    assert list(tmp_path.glob(".env.backup.*")) == []


@pytest.mark.parametrize("credential_source", ["env", "stored"])
def test_existing_cloud_configuration_with_key_is_truthfully_idempotent(
    credential_source, monkeypatch, tmp_path, forbid_interactive_and_install
):
    old = seed_existing_config(tmp_path, "omics-os", KEY)
    if credential_source == "env":
        monkeypatch.setenv("OMICS_OS_API_KEY", KEY)
        options = ["--use-omics-os"]
    else:
        options = ["--omics-os-key", KEY]
    result = invoke_init(*options)
    assert result.exit_code == 0, result.output
    assert "already configured for Omics-OS Cloud" in result.output
    assert (tmp_path / ".env").read_text() == old
    assert list(tmp_path.glob(".env.backup.*")) == []


def test_valid_force_cloud_setup_keeps_backup_before_replacement(
    monkeypatch, tmp_path, forbid_interactive_and_install
):
    old = seed_existing_config(tmp_path)
    monkeypatch.setenv("OMICS_OS_API_KEY", KEY)
    result = invoke_init("--use-omics-os", "--force")
    assert result.exit_code == 0, result.output
    backups = list(tmp_path.glob(".env.backup.*"))
    assert len(backups) == 1
    assert backups[0].read_text() == old
    assert_cloud_config(tmp_path, KEY)


def test_explicit_key_takes_precedence_when_use_flag_and_environment_are_present(
    monkeypatch, tmp_path, forbid_interactive_and_install
):
    monkeypatch.setenv("OMICS_OS_API_KEY", "omk_other_unit_key")
    result = invoke_init("--use-omics-os", "--omics-os-key", KEY)
    assert result.exit_code == 0, result.output
    assert_cloud_config(tmp_path, KEY)


def test_global_cloud_setup_saves_secure_credentials_without_cwd_env(
    tmp_path, forbid_interactive_and_install
):
    from lobster.config.global_config import (
        GlobalProviderConfig,
        get_global_credentials_path,
    )

    result = invoke_init("--global", "--omics-os-key", KEY)
    assert result.exit_code == 0, result.output
    config = json.loads(GlobalProviderConfig.get_config_path().read_text())
    assert config["default_provider"] == "omics-os"
    path = get_global_credentials_path()
    assert f"OMICS_OS_API_KEY={KEY}" in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600
    assert not (tmp_path / ".env").exists()
    assert KEY not in result.output


def test_init_cloud_key_save_does_not_claim_account_access_was_checked(
    tmp_path, forbid_interactive_and_install
):
    result = invoke_init("--omics-os-key", "x")
    assert result.exit_code == 0, result.output
    assert "not checked" in result.output.lower()
    assert_cloud_config(tmp_path, "x")


def test_interactive_init_bridge_has_no_five_minute_deadline(monkeypatch):
    from lobster.ui.bridge import go_tui_bridge

    calls = []

    def run(argv, **kwargs):
        calls.append(kwargs)
        path = Path(argv[argv.index("--result-file") + 1])
        path.write_text(json.dumps({"cancelled": True}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(go_tui_bridge.subprocess, "run", run)
    assert go_tui_bridge.run_init_wizard("/synthetic/lobster-tui") == {
        "cancelled": True
    }
    assert calls[0]["timeout"] is None


@pytest.mark.parametrize("outcome", ["success", "timeout", "interrupt"])
def test_init_bridge_restores_original_terminal_state_on_every_exit(
    outcome, monkeypatch
):
    import subprocess
    import termios

    from lobster.ui.bridge import go_tui_bridge

    original = [1, 2, 3, 4, 5, 6, [b"a", b"b"]]
    restored = []
    monkeypatch.setattr(
        "sys.stdin", SimpleNamespace(isatty=lambda: True, fileno=lambda: 9)
    )
    monkeypatch.setattr(termios, "tcgetattr", lambda fd: original)
    monkeypatch.setattr(
        termios, "tcsetattr", lambda fd, when, attrs: restored.append((fd, attrs))
    )

    def run(argv, **kwargs):
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(argv, 0.1)
        if outcome == "interrupt":
            raise KeyboardInterrupt
        Path(argv[argv.index("--result-file") + 1]).write_text(
            json.dumps({"cancelled": True})
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(go_tui_bridge.subprocess, "run", run)
    try:
        go_tui_bridge.run_init_wizard("/synthetic/lobster-tui", timeout=0.1)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        assert outcome != "success"
    assert restored == [(9, original)]


def test_gateway_unauthorized_diagnosis_is_not_misclassified_as_aws_iam():
    from botocore.exceptions import ClientError

    from lobster.cli_internal.go_tui_launcher import _friendly_error
    from lobster.config.providers.gateway_bedrock_client import GatewayBedrockClient

    client = GatewayBedrockClient(
        endpoint="https://app.omics-os.com", token_fn=lambda: KEY
    )
    with pytest.raises(ClientError) as error:
        client._handle_error(401, json.dumps({"detail": "Not authenticated"}))
    # A Bedrock-compatible transport error can contain this operation name;
    # the gateway auth identity must take precedence over substring matching.
    title, message = _friendly_error("ConverseStream: " + str(error.value))
    assert "Omics-OS Cloud API key rejected or expired" in message
    assert "app.omics-os.com" in message
    assert "IAM" not in message
    assert "Bedrock access denied" not in message


def test_gateway_tier_refusal_preserves_cloud_detail_not_iam_or_key_workaround():
    from botocore.exceptions import ClientError

    from lobster.cli_internal.go_tui_launcher import _friendly_error
    from lobster.config.providers.gateway_bedrock_client import GatewayBedrockClient

    detail = "This model is not available for your tier"
    client = GatewayBedrockClient(
        endpoint="https://app.omics-os.com", token_fn=lambda: KEY
    )
    with pytest.raises(ClientError) as error:
        client._handle_error(403, json.dumps({"detail": detail}))
    _, message = _friendly_error("ConverseStream: " + str(error.value))
    assert "Omics-OS Cloud:" in message
    assert detail in message
    assert "IAM" not in message
    assert "create a new" not in message.lower()


def test_python_cloud_login_help_describes_supported_paths_not_automatic_oauth():
    from lobster.cli import app

    result = CliRunner().invoke(app, ["cloud", "login", "--help"])
    assert result.exit_code == 0
    output = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result.output)
    assert_supported_guidance(output)
    assert "opens a browser for OAuth" not in output


def test_real_bedrock_iam_diagnosis_is_preserved():
    from botocore.exceptions import ClientError

    from lobster.cli_internal.go_tui_launcher import _friendly_error

    error = ClientError(
        {
            "Error": {
                "Code": "AccessDeniedException",
                "Message": "Not authorized for bedrock:InvokeModel",
            }
        },
        "ConverseStream",
    )
    _, message = _friendly_error(str(error))
    assert "AWS Bedrock access denied. Check IAM permissions" in message
    assert "Omics-OS" not in message


def test_skip_extras_records_selected_agents_without_install_and_shows_command(
    monkeypatch, tmp_path, capsys
):
    calls = []

    def install(agents, *args, **kwargs):
        calls.append(agents)
        return agents

    monkeypatch.setattr(init_commands, "_check_and_prompt_install_packages", install)
    result = init_commands._postprocess_tui_init_result(
        {"provider": "omics-os", "api_key": KEY, "agents": ["data_expert_agent"]},
        workspace_path=tmp_path / ".lobster_workspace",
        env_path=tmp_path / ".env",
        global_config=False,
        skip_extras=True,
    )
    assert calls == []
    assert result["agents"] == ["data_expert_agent"]
    assert (
        "data_expert_agent"
        in (tmp_path / ".lobster_workspace" / "config.toml").read_text()
    )
    output = capsys.readouterr().out
    assert "Install later: uv pip install lobster-research" in output


@pytest.mark.parametrize("configuration", ["global", "env"])
def test_status_uses_runtime_provider_resolution_without_cwd_env(
    configuration, monkeypatch, tmp_path, forbid_interactive_and_install
):
    from lobster.cli_internal.commands.heavy.display_helpers import build_status_blocks
    from lobster.core.config_resolver import ConfigResolver

    if configuration == "global":
        result = invoke_init("--global", "--omics-os-key", KEY)
        assert result.exit_code == 0, result.output
    else:
        monkeypatch.setenv("LOBSTER_LLM_PROVIDER", "omics-os")
        monkeypatch.setenv("OMICS_OS_API_KEY", KEY)
    assert not (tmp_path / ".env").exists()
    provider, source = ConfigResolver(
        tmp_path / ".lobster_workspace"
    ).resolve_provider()
    assert provider == "omics-os"
    blocks = build_status_blocks(compact=True)
    initialization = next(
        block
        for block in blocks
        if block.kind == "kv" and block.data.get("title") == "Initialization"
    )
    rows = dict(initialization.data["rows"])
    assert rows["Initialization"] == "Configured"
    assert rows["Provider"] == provider
    assert rows["Configuration Source"] == source
    assert "not checked" in rows["Account Access"].lower()
    assert not any(
        block.kind == "alert" and "not initialized" in str(block.data)
        for block in blocks
    )


def test_classic_skip_extras_keeps_agent_selection_and_prints_later_command(
    monkeypatch, tmp_path, capsys
):
    selected = ["data_expert_agent"]
    monkeypatch.setattr(
        init_commands, "_prompt_manual_agent_selection", lambda path: selected
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("--skip-extras attempted interactive agent installation")

    monkeypatch.setattr(init_commands, "_check_and_prompt_install_packages", forbidden)
    agents, preset = init_commands._perform_agent_selection_interactive(
        tmp_path / ".lobster_workspace", skip_extras=True
    )
    assert agents == selected
    assert preset is None
    assert "Install later: uv pip install lobster-research" in capsys.readouterr().out
