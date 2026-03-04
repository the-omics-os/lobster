"""Tests for OpenRouter config schema fields."""

import pytest


def test_global_config_has_openrouter_model_field():
    from lobster.config.global_config import GlobalProviderConfig
    config = GlobalProviderConfig(openrouter_default_model="anthropic/claude-sonnet-4-5")
    assert config.openrouter_default_model == "anthropic/claude-sonnet-4-5"


def test_global_config_get_model_for_openrouter():
    """base_config.get_model_for_provider() auto-resolves openrouter_default_model."""
    from lobster.config.global_config import GlobalProviderConfig
    config = GlobalProviderConfig(openrouter_default_model="openai/gpt-4o")
    assert config.get_model_for_provider("openrouter") == "openai/gpt-4o"


def test_global_config_openrouter_model_defaults_none():
    from lobster.config.global_config import GlobalProviderConfig
    config = GlobalProviderConfig()
    assert config.openrouter_default_model is None


def test_workspace_config_has_openrouter_model_field():
    from lobster.config.workspace_config import WorkspaceProviderConfig
    config = WorkspaceProviderConfig(openrouter_model="meta-llama/llama-3.3-70b-instruct")
    assert config.openrouter_model == "meta-llama/llama-3.3-70b-instruct"


def test_workspace_config_get_model_for_openrouter():
    """base_config.get_model_for_provider() auto-resolves openrouter_model."""
    from lobster.config.workspace_config import WorkspaceProviderConfig
    config = WorkspaceProviderConfig(openrouter_model="openai/gpt-4o")
    assert config.get_model_for_provider("openrouter") == "openai/gpt-4o"


def test_workspace_config_accepts_openrouter_provider():
    from lobster.config.workspace_config import WorkspaceProviderConfig
    config = WorkspaceProviderConfig(global_provider="openrouter")
    assert config.global_provider == "openrouter"


def test_global_config_roundtrip_json():
    """OpenRouter config field survives JSON serialization."""
    from lobster.config.global_config import GlobalProviderConfig
    config = GlobalProviderConfig(
        default_provider="openrouter",
        openrouter_default_model="deepseek/deepseek-r1"
    )
    json_str = config.model_dump_json()
    reloaded = GlobalProviderConfig.model_validate_json(json_str)
    assert reloaded.default_provider == "openrouter"
    assert reloaded.openrouter_default_model == "deepseek/deepseek-r1"


def test_create_openrouter_config_valid():
    from lobster.config.provider_setup import create_openrouter_config
    result = create_openrouter_config("sk-or-test-key-123")
    assert result.success is True
    assert result.provider_type == "openrouter"
    assert result.env_vars["OPENROUTER_API_KEY"] == "sk-or-test-key-123"
    assert result.env_vars["LOBSTER_LLM_PROVIDER"] == "openrouter"


def test_create_openrouter_config_strips_whitespace():
    from lobster.config.provider_setup import create_openrouter_config
    result = create_openrouter_config("  sk-or-test  ")
    assert result.env_vars["OPENROUTER_API_KEY"] == "sk-or-test"


def test_create_openrouter_config_empty_key_fails():
    from lobster.config.provider_setup import create_openrouter_config
    result = create_openrouter_config("")
    assert result.success is False
    assert result.message is not None


def test_create_openrouter_config_whitespace_only_fails():
    from lobster.config.provider_setup import create_openrouter_config
    result = create_openrouter_config("   ")
    assert result.success is False


# ---------------------------------------------------------------------------
# ConfigResolver end-to-end tests
# ---------------------------------------------------------------------------
# These tests opt out of the autouse auto_mock_provider_config fixture so they
# exercise the real ConfigResolver / ProviderRegistry code paths without any
# pre-applied mocks.  The `no_auto_config` marker is checked in conftest.py.
# ---------------------------------------------------------------------------


@pytest.fixture()
def reset_singletons():
    """Ensure ConfigResolver and ProviderRegistry singletons are clean before
    and after each e2e test, even when an assertion raises mid-test."""
    from lobster.config.providers.registry import ProviderRegistry
    from lobster.core.config_resolver import ConfigResolver

    ConfigResolver.reset_instance()
    ProviderRegistry.reset()
    try:
        yield
    finally:
        ConfigResolver.reset_instance()
        ProviderRegistry.reset()


@pytest.mark.no_auto_config
def test_config_resolver_accepts_openrouter(monkeypatch, reset_singletons):
    """ConfigResolver reads LOBSTER_LLM_PROVIDER=openrouter from env and resolves correctly."""
    from lobster.core.config_resolver import ConfigResolver

    monkeypatch.setenv("LOBSTER_LLM_PROVIDER", "openrouter")
    # workspace_path=None skips layers 2 (workspace config) and 3 (global config),
    # so resolution falls through to the env var at layer 4.
    resolver = ConfigResolver.get_instance(workspace_path=None)
    provider, source = resolver.resolve_provider()

    assert provider == "openrouter"
    assert "environment" in source.lower()


@pytest.mark.no_auto_config
def test_config_resolver_rejects_invalid_provider(monkeypatch, reset_singletons):
    """ConfigResolver raises ConfigurationError for an unrecognised provider name."""
    from lobster.core.config_resolver import ConfigResolver, ConfigurationError

    monkeypatch.setenv("LOBSTER_LLM_PROVIDER", "badprovider")
    resolver = ConfigResolver.get_instance(workspace_path=None)

    with pytest.raises(ConfigurationError) as exc_info:
        resolver.resolve_provider()

    error_message = str(exc_info.value)
    # The invalid name must be named in the error so the user knows what was wrong
    assert "badprovider" in error_message
    # The valid-providers hint must include "openrouter" so users know it is valid
    assert "openrouter" in error_message


@pytest.mark.no_auto_config
def test_llm_factory_creates_openrouter_model(monkeypatch, reset_singletons):
    """LLMFactory creates a ChatOpenAI instance routed through OpenRouter with correct params."""
    import unittest.mock as mock

    from lobster.config.llm_factory import LLMFactory

    monkeypatch.setenv("LOBSTER_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    # ChatOpenAI is imported inside create_chat_model via a local
    # `from langchain_openai import ChatOpenAI`, so patching the source
    # module attribute is the correct interception point.
    with mock.patch("langchain_openai.ChatOpenAI") as mock_chat_openai:
        mock_chat_openai.return_value = mock.MagicMock()
        LLMFactory.create_llm(
            model_config={},
            agent_name="test_agent",
        )

    # ChatOpenAI must have been called exactly once
    assert mock_chat_openai.call_count == 1
    call_kwargs = mock_chat_openai.call_args.kwargs

    # Verify the default model for the openrouter provider
    assert call_kwargs.get("model") == "anthropic/claude-sonnet-4-5"

    # Verify the OpenRouter routing base URL
    assert call_kwargs.get("base_url") == "https://openrouter.ai/api/v1"

    # Verify the Lobster AI branding headers required by OpenRouter's leaderboard
    headers = call_kwargs.get("default_headers", {})
    assert headers.get("HTTP-Referer") == "https://lobsterbio.com"
    assert headers.get("X-Title") == "Lobster AI"

    # Verify the API key is forwarded to the model
    assert call_kwargs.get("api_key") == "sk-or-test"
