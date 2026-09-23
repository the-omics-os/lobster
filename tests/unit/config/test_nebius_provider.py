"""
Focused unit tests for the Nebius AI Studio provider.

Nebius is a reduced, OpenAI-wire-compatible provider: static KNOWN_MODELS
catalog, no live catalog fetch, no branding headers, no /auth/key probe.
These tests therefore assert only what the Nebius provider actually does —
they do NOT clone the OpenRouter tests (live catalog / headers / auth-key).
"""

import importlib.util
import os
from unittest.mock import MagicMock, patch

import pytest
import typer

from lobster.config.constants import VALID_PROVIDERS
from lobster.config.providers import ProviderRegistry, get_provider
from lobster.config.providers.nebius_provider import NebiusProvider

_DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"


def test_nebius_provider_name_and_abstractmethods():
    """name is 'nebius' and every ILLMProvider abstractmethod is implemented."""
    assert NebiusProvider().name == "nebius"
    # Empty frozenset proves no ABC abstractmethod is left unimplemented,
    # so instantiation/registration cannot raise TypeError.
    assert NebiusProvider.__abstractmethods__ == frozenset()


def test_nebius_provider_display_name():
    """display_name is the human-friendly Nebius label."""
    assert NebiusProvider().display_name == "Nebius AI Studio"


def test_nebius_is_configured_reflects_env():
    """is_configured() tracks NEBIUS_API_KEY presence."""
    provider = NebiusProvider()

    # Key present
    with patch.dict(os.environ, {"NEBIUS_API_KEY": "test-key"}, clear=True):
        assert provider.is_configured()

    # Key absent
    with patch.dict(os.environ, {}, clear=True):
        assert not provider.is_configured()


def test_nebius_is_available_matches_configured():
    """is_available() equals is_configured() for this cloud provider."""
    provider = NebiusProvider()
    with patch.dict(os.environ, {"NEBIUS_API_KEY": "test-key"}, clear=True):
        assert provider.is_available()
    with patch.dict(os.environ, {}, clear=True):
        assert not provider.is_available()


def test_nebius_get_default_model():
    """Default model is the canonical Qwen3 30B instruct id."""
    assert NebiusProvider().get_default_model() == _DEFAULT_MODEL


@pytest.mark.skipif(
    not importlib.util.find_spec("langchain_openai"),
    reason="langchain-openai not installed",
)
@patch("langchain_openai.ChatOpenAI")
def test_nebius_create_chat_model(mock_chat_openai):
    """create_chat_model builds ChatOpenAI with Nebius base_url, no headers, no network."""
    mock_chat_openai.return_value = MagicMock()

    with patch.dict(os.environ, {"NEBIUS_API_KEY": "dummy"}, clear=True):
        provider = get_provider("nebius")
        provider.create_chat_model(_DEFAULT_MODEL)

    mock_chat_openai.assert_called_once()
    call_kwargs = mock_chat_openai.call_args.kwargs
    assert call_kwargs["model"] == _DEFAULT_MODEL
    assert call_kwargs["base_url"] == _BASE_URL
    assert call_kwargs["api_key"] == "dummy"
    # Nebius must NOT attach OpenRouter-style branding headers.
    assert "default_headers" not in call_kwargs


def test_nebius_known_models_shape():
    """KNOWN_MODELS: exactly 20 entries, all nebius, positive input cost, single default."""
    models = NebiusProvider.KNOWN_MODELS
    assert len(models) == 20
    assert all(m.provider == "nebius" for m in models)
    assert all(m.input_cost_per_million is not None for m in models)
    assert all(m.input_cost_per_million > 0 for m in models)

    defaults = [m for m in models if m.is_default is True]
    assert len(defaults) == 1
    assert defaults[0].name == _DEFAULT_MODEL


def test_nebius_list_models_returns_known_models():
    """list_models() returns the static catalog (no network)."""
    models = NebiusProvider().list_models()
    assert len(models) == 20
    assert [m.name for m in models] == [m.name for m in NebiusProvider.KNOWN_MODELS]


def test_nebius_registered_and_valid():
    """Nebius must be reachable through lazy registry init, not just by import.

    This module imports NebiusProvider directly, and that import auto-registers
    the provider — so asserting ProviderRegistry.get("nebius") on a warm
    registry would pass even if the _provider_specs entry were missing.
    reset() clears the registry and forces _ensure_initialized() to re-resolve
    from _provider_specs, which is the real integration point.
    """
    assert "nebius" in VALID_PROVIDERS

    ProviderRegistry.reset()
    try:
        assert ProviderRegistry.get("nebius") is not None
        assert "nebius" in ProviderRegistry.get_provider_names()
    finally:
        # Leave the registry warm for the rest of the session.
        ProviderRegistry.reset()
        ProviderRegistry.get_all()


def test_nebius_unknown_model_gets_conservative_window():
    """A Nebius id absent from KNOWN_MODELS must not inherit the 200k default.

    Open-weight windows range from 8k to 1M; over-promising 200k on an
    unlisted model invites silent context overflow.
    """
    info = NebiusProvider().get_model_info("some-org/unlisted-model")
    assert info.context_window == 128_000
    assert info.provider == "nebius"


def test_nebius_pricing_reaches_unified_pricing_table():
    """KNOWN_MODELS pricing must surface in the registry-wide pricing table.

    MODEL_PRICING derives from this; a miss here means $0-cost reporting.
    """
    pricing = ProviderRegistry.get_all_models_with_pricing()
    entry = pricing.get(_DEFAULT_MODEL)
    assert entry is not None
    assert entry["input_per_million"] > 0
    assert entry["output_per_million"] > 0


# --------------------------------------------------------------------------- #
# Integration wiring — a provider is only usable if every surface knows it.
# These guard the gaps that made `nebius` selectable but not configurable.
# --------------------------------------------------------------------------- #


def test_nebius_config_helper_builds_env_vars():
    """create_nebius_config() emits the provider + key env vars."""
    from lobster.config.provider_setup import create_nebius_config

    config = create_nebius_config("  test-key  ")
    assert config.success
    assert config.provider_type == "nebius"
    assert config.env_vars["LOBSTER_LLM_PROVIDER"] == "nebius"
    assert config.env_vars["NEBIUS_API_KEY"] == "test-key"

    assert not create_nebius_config("   ").success


def test_nebius_workspace_config_has_model_field():
    """set_model_for_provider('nebius', ...) must not raise.

    Without a `nebius_model` field this raises AttributeError, which
    _create_workspace_config swallows — silently discarding the whole
    provider_config.json, including the user's provider choice.
    """
    from lobster.config.workspace_config import WorkspaceProviderConfig

    config = WorkspaceProviderConfig()
    config.set_model_for_provider("nebius", _DEFAULT_MODEL)
    assert config.nebius_model == _DEFAULT_MODEL
    assert config.get_model_for_provider("nebius") == _DEFAULT_MODEL


def test_nebius_global_config_has_default_model_field():
    """Global config exposes a nebius default-model field."""
    from lobster.config.global_config import GlobalProviderConfig

    config = GlobalProviderConfig()
    assert hasattr(config, "nebius_default_model")
    config.nebius_default_model = _DEFAULT_MODEL
    config.reset()
    assert config.nebius_default_model is None


def test_nebius_init_adapter_writes_env_and_config(tmp_path):
    """The wizard result for nebius must produce a usable .env + provider config.

    Regression: `nebius` was absent from _PROVIDER_ENV_KEYS, so this raised
    ValueError and `lobster init` could not complete for Nebius at all.
    """
    import json

    from lobster.ui.bridge.init_adapter import apply_tui_init_result

    workspace = tmp_path / ".lobster_workspace"
    env_path = tmp_path / ".env"

    apply_tui_init_result(
        {"provider": "nebius", "api_key": "test-key", "model_id": "Qwen/Qwen3-32B"},
        workspace_path=workspace,
        env_path=env_path,
    )

    assert "NEBIUS_API_KEY=test-key" in env_path.read_text()

    saved = json.loads((workspace / "provider_config.json").read_text())
    assert saved["global_provider"] == "nebius"
    assert saved["nebius_model"] == "Qwen/Qwen3-32B"


def test_nebius_present_in_init_wizard_manifest():
    """The wizard manifest must carry Nebius credentials + model catalog.

    Regression: without a _PROVIDER_CREDENTIALS entry the questionary flow
    prompted for no API key, producing an unconfigured provider.
    """
    from lobster.ui.wizard.manifest import build_init_manifest

    manifest = build_init_manifest(detect_ollama=False)
    nebius = next((p for p in manifest.providers if p.name == "nebius"), None)

    assert nebius is not None
    assert nebius.display_name == "Nebius AI Studio"
    assert nebius.model_selection == "explicit"
    assert [c.key for c in nebius.credentials] == ["NEBIUS_API_KEY"]
    assert len(nebius.models) == 20
    assert any(m.is_default for m in nebius.models)


def test_nebius_provider_package_maps_cover_nebius():
    """Auto-install maps must resolve nebius -> langchain-openai."""
    from lobster.cli_internal.commands.heavy.init_commands import (
        _PROVIDER_IMPORT_NAMES,
        _PROVIDER_PACKAGES,
    )

    assert _PROVIDER_PACKAGES["nebius"] == "langchain-openai"
    assert _PROVIDER_IMPORT_NAMES["nebius"] == "langchain_openai"


def test_nebius_validates_as_a_provider_choice():
    """--nebius-key alone must satisfy the non-interactive provider check.

    Regression: validate_provider_choice() had no has_nebius parameter, so
    `lobster init --non-interactive --nebius-key=...` aborted with
    "No provider specified".
    """
    from lobster.config.provider_setup import (
        get_provider_priority_warning,
        validate_provider_choice,
    )

    valid, error = validate_provider_choice(
        has_anthropic=False,
        has_bedrock=False,
        has_ollama=False,
        has_nebius=True,
    )
    assert valid
    assert error is None

    # Nebius alone is not a conflict; Nebius + a higher-priority provider is.
    assert (
        get_provider_priority_warning(
            has_anthropic=False,
            has_bedrock=False,
            has_ollama=False,
            has_nebius=True,
        )
        is None
    )
    warning = get_provider_priority_warning(
        has_anthropic=True,
        has_bedrock=False,
        has_ollama=False,
        has_nebius=True,
    )
    assert warning is not None and "Claude API" in warning


def test_nebius_extra_is_declared_in_pyproject():
    """LLM_PROVIDER_PACKAGES must point at extras that actually exist.

    `_build_uv_tool_init_command` passes the provider name through as a
    lobster-ai extra. uv ignores unknown extras *without failing*, so an
    undeclared extra makes the uv-tool install a silent no-op that leaves
    langchain-openai missing.

    Asserts the nebius entry by name — a loop over whatever happens to be in
    LLM_PROVIDER_PACKAGES would still pass if the entry were deleted.
    """
    import tomllib
    from pathlib import Path

    from lobster.core.component_registry import LLM_PROVIDER_PACKAGES

    pyproject_data = tomllib.loads(
        (Path(__file__).resolve().parents[3] / "pyproject.toml").read_text()
    )
    extras = pyproject_data["project"]["optional-dependencies"]

    assert LLM_PROVIDER_PACKAGES["nebius"] == ("nebius", "langchain-openai")
    assert LLM_PROVIDER_PACKAGES["openrouter"] == ("openrouter", "langchain-openai")

    # Every mapped extra must exist AND actually ship its mapped package.
    for provider, (extra, pypi_package) in LLM_PROVIDER_PACKAGES.items():
        assert extra in extras, f"{provider} maps to undeclared extra '{extra}'"
        assert any(
            req.split(">")[0].split("=")[0].split("[")[0].strip() == pypi_package
            for req in extras[extra]
        ), f"extra '{extra}' does not install {pypi_package}"

    # Both gateway extras must be reachable from the all-providers meta-extra.
    all_providers = " ".join(extras["all-providers"])
    assert "nebius" in all_providers
    assert "openrouter" in all_providers


def test_provider_icons_cover_every_provider_and_match_go_tui():
    """PROVIDER_ICONS is the shared palette — Go and Python must agree.

    The Go TUI's providerIcon() renders the same status line as the Python CLI,
    so a provider missing from either side, or mapped to a different glyph,
    shows up as an inconsistent UI.
    """
    import re
    from pathlib import Path

    from lobster.config.constants import PROVIDER_ICONS

    assert set(PROVIDER_ICONS) == set(VALID_PROVIDERS)

    go_source = (
        Path(__file__).resolve().parents[3]
        / "lobster-tui"
        / "internal"
        / "chat"
        / "model.go"
    )
    if not go_source.exists():  # pragma: no cover - Go TUI not vendored
        pytest.skip("lobster-tui sources not present")

    body = go_source.read_text().split("func providerIcon(")[1].split("\n}")[0]
    go_icons = dict(re.findall(r'case "([^"]+)":\s*\n\s*return "([^"]+)"', body))

    for provider, icon in PROVIDER_ICONS.items():
        assert go_icons.get(provider) == icon, (
            f"provider '{provider}': Python has {icon!r}, "
            f"Go has {go_icons.get(provider)!r}"
        )


def test_nebius_key_flag_is_exposed_and_forwarded():
    """`lobster init` must expose --nebius-key and forward it to init_impl.

    Regression: the flag did not exist, so Nebius was unreachable from CI/CD.
    """
    import inspect

    from lobster.cli import app
    from lobster.cli_internal.commands.heavy.init_commands import init_impl

    init_cmd = typer.main.get_command(app).commands["init"]
    flags = {opt for param in init_cmd.params for opt in param.opts}
    assert "--nebius-key" in flags

    # The flag is only useful if it reaches init_impl under the same name.
    assert "nebius_key" in inspect.signature(init_impl).parameters


def test_nebius_non_interactive_init_writes_credentials(tmp_path, monkeypatch):
    """--nebius-key must produce a usable .env + provider_config.json.

    Calls init_impl the way the CLI does. init_impl's defaults are
    ``typer.OptionInfo`` sentinels when it is invoked directly, so they are
    resolved to their real values first — passing them by hand would silently
    rot as the signature grows.
    """
    import inspect
    import io
    import json

    from rich.console import Console

    from lobster.cli_internal.commands.heavy import init_commands
    from lobster.cli_internal.commands.heavy.init_commands import init_impl

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOBSTER_WORKSPACE", str(tmp_path / ".lobster_workspace"))
    # Own console: other CLI tests can leave the shared one writing to a
    # closed stream, which would surface here as an unrelated ValueError.
    monkeypatch.setattr(init_commands, "console", Console(file=io.StringIO()))

    kwargs = {}
    for name, param in inspect.signature(init_impl).parameters.items():
        default = param.default
        if isinstance(default, typer.models.OptionInfo):
            default = default.default
        kwargs[name] = None if default is inspect.Parameter.empty else default

    kwargs.update(
        non_interactive=True,
        force=True,
        skip_extras=True,
        skip_ssl_test=True,
        nebius_key="test-nebius-key",
    )

    with pytest.raises(typer.Exit) as exc_info:
        init_impl(**kwargs)
    assert exc_info.value.exit_code == 0

    env_text = (tmp_path / ".env").read_text()
    assert "LOBSTER_LLM_PROVIDER=nebius" in env_text
    assert "NEBIUS_API_KEY=test-nebius-key" in env_text

    saved = json.loads(
        (tmp_path / ".lobster_workspace" / "provider_config.json").read_text()
    )
    assert saved["global_provider"] == "nebius"
