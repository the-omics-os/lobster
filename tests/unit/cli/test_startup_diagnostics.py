import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from rich.console import Console

from lobster.cli_internal.commands.heavy import (
    chat_commands,
    query_commands,
    session_infra,
)
from lobster.cli_internal.startup_diagnostics import (
    StartupDiagnostic,
    StartupDiagnosticError,
    classify_startup_exception,
    format_startup_diagnostic_text,
    render_startup_diagnostic_rich,
)
from lobster.core.config_resolver import ConfigurationError


def test_classify_missing_provider_package_preserves_install_hint():
    diagnostic = classify_startup_exception(
        ImportError(
            "langchain-aws package not installed. Install with: uv pip install 'lobster-ai[bedrock]'"
        ),
        provider_override="bedrock",
    )

    assert diagnostic.code == "missing_provider_package"
    assert diagnostic.title == "Missing provider package"
    assert diagnostic.detail_lines == (
        "langchain-aws package not installed. Install with: uv pip install 'lobster-ai[bedrock]'",
    )
    assert diagnostic.fix_lines == ("Run: uv pip install 'lobster-ai[bedrock]'",)


def test_classify_provider_not_configured_uses_help_as_fix_text():
    diagnostic = classify_startup_exception(
        ConfigurationError(
            "No provider configured.",
            "Long resolver help text should not leak directly into compact UI alerts.",
        ),
        workspace=Path("/tmp/project/.lobster_workspace"),
    )

    assert diagnostic.code == "provider_not_configured"
    assert diagnostic.title == "No provider configured"
    assert diagnostic.detail_lines == ()
    assert "Run: lobster init --global" in diagnostic.fix_lines
    assert "Or run: lobster init" in diagnostic.fix_lines
    assert "Or set: export LOBSTER_LLM_PROVIDER=anthropic" in diagnostic.fix_lines
    assert any(line.startswith("Valid providers: ") for line in diagnostic.fix_lines)


def test_classify_provider_not_configured_prefers_existing_workspace_hint(
    monkeypatch,
):
    monkeypatch.setattr(
        "lobster.core.config_resolver._find_existing_configs",
        lambda *args, **kwargs: [Path("/tmp/existing/.lobster_workspace")],
    )

    diagnostic = classify_startup_exception(
        ConfigurationError(
            "No provider configured.",
            "Resolver help text is available but the shared diagnostic should stay compact.",
        ),
        workspace=Path("/tmp/project/.lobster_workspace"),
    )

    assert diagnostic.fix_lines[0] == (
        "Reuse existing workspace: export LOBSTER_WORKSPACE=/tmp/existing/.lobster_workspace"
    )


def test_classify_invalid_provider_yields_structured_fix_guidance():
    diagnostic = classify_startup_exception(
        ConfigurationError(
            "Invalid provider 'bogus'. Valid providers: anthropic, bedrock, ollama"
        )
    )

    assert diagnostic.code == "invalid_provider"
    assert diagnostic.title == "Invalid provider"
    assert diagnostic.detail_lines == (
        "Invalid provider 'bogus'. Valid providers: anthropic, bedrock, ollama",
    )
    assert diagnostic.fix_lines == (
        "Use one of: anthropic, bedrock, ollama",
        "Or run: lobster init",
    )


def test_classify_missing_credentials_includes_workspace_and_export_hint(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "lobster.core.config_resolver._find_existing_configs",
        lambda *args, **kwargs: [],
    )

    diagnostic = classify_startup_exception(
        ValueError(
            "AWS credentials not found in environment. Set it with: export AWS_ACCESS_KEY_ID=test"
        ),
        workspace=tmp_path / ".lobster_workspace",
        provider_override="bedrock",
    )

    assert diagnostic.code == "missing_credentials"
    assert diagnostic.title == "Missing credentials for provider 'bedrock'"
    assert diagnostic.detail_lines == (
        "AWS credentials not found in environment. Set it with: export AWS_ACCESS_KEY_ID=test",
    )
    assert any(
        line.strip() == str(tmp_path / ".lobster_workspace" / ".env")
        for line in diagnostic.fix_lines
    )
    assert any(
        line.strip() == "export AWS_ACCESS_KEY_ID=test" for line in diagnostic.fix_lines
    )


def test_format_and_rich_render_share_same_startup_diagnostic_copy():
    diagnostic = StartupDiagnostic(
        code="missing_provider_package",
        title="Missing provider package",
        detail_lines=("langchain-aws package not installed.",),
        fix_lines=("Run: uv pip install 'lobster-ai[bedrock]'",),
    )
    console = Console(record=True, width=100)

    render_startup_diagnostic_rich(console, diagnostic)
    plain_text = format_startup_diagnostic_text(diagnostic)
    rich_text = console.export_text()

    assert "Missing provider package" in plain_text
    assert "How to fix:" in plain_text
    assert "Run: uv pip install 'lobster-ai[bedrock]'" in plain_text
    assert "Missing provider package" in rich_text
    assert "How to fix:" in rich_text
    assert "Run: uv pip install 'lobster-ai[bedrock]'" in rich_text


def test_validate_startup_passes_provider_override_to_resolver(monkeypatch, tmp_path):
    seen = {}

    class _FakeResolver:
        def resolve_provider(self, runtime_override=None):
            seen["runtime_override"] = runtime_override
            return "bedrock", "runtime flag --provider"

    monkeypatch.setattr(
        session_infra,
        "_maybe_seed_workspace_provider_config",
        lambda path: seen.setdefault("seed_path", path),
    )
    monkeypatch.setattr(
        "lobster.core.workspace.resolve_workspace",
        lambda explicit_path=None, create=True: tmp_path,
    )
    monkeypatch.setattr(
        "lobster.config.settings.settings.reload_credentials",
        lambda workspace_path: seen.setdefault("reload_path", workspace_path),
    )
    monkeypatch.setattr(
        "lobster.core.config_resolver.ConfigResolver.get_instance",
        lambda workspace_path: _FakeResolver(),
    )

    resolved = session_infra.validate_startup_or_raise_startup_diagnostic(
        workspace=tmp_path,
        provider_override="bedrock",
    )

    assert resolved == tmp_path
    assert seen["seed_path"] == tmp_path
    assert seen["reload_path"] == tmp_path
    assert seen["runtime_override"] == "bedrock"


def test_init_client_renders_startup_diagnostic_and_exits(monkeypatch):
    recorded_console = Console(record=True, width=120)
    diagnostic = StartupDiagnostic(
        code="missing_provider_package",
        title="Missing provider package",
        detail_lines=("langchain-aws package not installed.",),
        fix_lines=("Run: uv pip install 'lobster-ai[bedrock]'",),
    )

    monkeypatch.setattr(session_infra, "console", recorded_console)
    monkeypatch.setattr(
        session_infra,
        "init_client_or_raise_startup_diagnostic",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            StartupDiagnosticError(diagnostic)
        ),
    )

    with pytest.raises(typer.Exit) as exc_info:
        session_infra.init_client(provider_override="bedrock")

    assert exc_info.value.exit_code == diagnostic.exit_code
    output = recorded_console.export_text()
    assert "Missing provider package" in output
    assert "How to fix:" in output


def test_init_client_go_tui_mode_skips_terminal_callbacks(tmp_path, monkeypatch):
    seen = {}

    class _FakeDataManager:
        def __init__(self, workspace_path, console, session_dir):
            seen["workspace_path"] = workspace_path
            seen["console"] = console
            seen["session_dir"] = session_dir

    def _fake_create_local_agent_client(**kwargs):
        seen["callbacks"] = kwargs["callbacks"]
        return SimpleNamespace(profile_timings_enabled=None)

    monkeypatch.setattr(
        session_infra,
        "validate_startup_or_raise_startup_diagnostic",
        lambda workspace=None, provider_override=None: tmp_path,
    )
    monkeypatch.setattr("lobster.ui.setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "lobster.core.data_manager_v2.DataManagerV2",
        _FakeDataManager,
    )
    monkeypatch.setattr(
        session_infra,
        "_resolve_profile_timings_flag",
        lambda profile_timings: False,
    )
    monkeypatch.setattr(
        session_infra,
        "_create_local_agent_client",
        _fake_create_local_agent_client,
    )

    session_infra.set_go_tui_active(True)
    try:
        client = session_infra.init_client_or_raise_startup_diagnostic(
            workspace=tmp_path
        )
    finally:
        session_infra.set_go_tui_active(False)

    assert client.profile_timings_enabled is False
    assert seen["workspace_path"] == tmp_path
    assert seen["callbacks"] == []


def test_query_impl_json_reports_structured_startup_error(monkeypatch, capsys):
    diagnostic = StartupDiagnostic(
        code="invalid_provider",
        title="Invalid provider",
        detail_lines=("Invalid provider 'bogus'. Valid providers: anthropic, bedrock",),
        fix_lines=("Use one of: anthropic, bedrock",),
    )

    monkeypatch.setattr(
        query_commands,
        "validate_startup_or_raise_startup_diagnostic",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            StartupDiagnosticError(diagnostic)
        ),
    )

    with pytest.raises(typer.Exit) as exc_info:
        query_commands.query_impl(
            "hello",
            workspace=None,
            session_id=None,
            reasoning=False,
            verbose=False,
            debug=False,
            output=None,
            profile_timings=None,
            provider="bogus",
            model=None,
            stream=False,
            json_output=True,
        )

    assert exc_info.value.exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is False
    assert payload["error_code"] == "invalid_provider"
    assert "Invalid provider" in payload["error"]


def test_chat_impl_surfaces_startup_diagnostic_before_welcome(monkeypatch):
    diagnostic = StartupDiagnostic(
        code="missing_provider_package",
        title="Missing provider package",
        detail_lines=("langchain-aws package not installed.",),
        fix_lines=("Run: uv pip install 'lobster-ai[bedrock]'",),
    )
    recorded_console = Console(record=True, width=120)
    state = {"welcome_called": False}

    monkeypatch.setattr(chat_commands, "console", recorded_console)
    monkeypatch.setattr(
        chat_commands,
        "console_manager",
        SimpleNamespace(error_console=recorded_console),
    )
    monkeypatch.setattr(chat_commands, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        chat_commands,
        "validate_startup_or_raise_startup_diagnostic",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            StartupDiagnosticError(diagnostic)
        ),
    )
    monkeypatch.setattr(
        chat_commands,
        "display_welcome",
        lambda: state.__setitem__("welcome_called", True),
    )

    with pytest.raises(typer.Exit) as exc_info:
        chat_commands.chat_impl(
            workspace=None,
            session_id=None,
            reasoning=False,
            verbose=False,
            debug=False,
            profile_timings=None,
            provider="bedrock",
            model=None,
            stream=False,
        )

    assert exc_info.value.exit_code == 1
    assert state["welcome_called"] is False
    output = recorded_console.export_text()
    assert "Missing provider package" in output
