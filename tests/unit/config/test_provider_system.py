"""
Unit tests for the refactored provider system.

Tests cover:
- ProviderRegistry: registration, discovery, health checks
- Individual providers: Anthropic, Bedrock, Ollama
- ConfigResolver integration with providers
- LLMFactory end-to-end flow
- Error handling for missing configuration
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lobster.config.providers import (
    ILLMProvider,
    ModelInfo,
    ProviderRegistry,
    get_provider,
)
from lobster.config.providers.registry import ProviderRegistry as Registry
from lobster.core.config_resolver import ConfigResolver, ConfigurationError

# Mark all tests in this module to skip auto_config
# These tests are specifically testing provider/config system behavior
pytestmark = pytest.mark.no_auto_config


# =============================================================================
# Provider Registry Tests
# =============================================================================


def test_provider_registry_initialization():
    """Test that providers are auto-registered on import."""
    # Reset registry for clean test
    Registry.reset()

    # Trigger initialization
    providers = ProviderRegistry.get_provider_names()

    # Should have all 4 providers registered
    assert "anthropic" in providers
    assert "bedrock" in providers
    assert "ollama" in providers
    assert "gemini" in providers
    assert len(providers) == 4


def test_get_provider():
    """Test getting individual providers by name."""
    anthropic = get_provider("anthropic")
    assert anthropic is not None
    assert anthropic.name == "anthropic"
    assert isinstance(anthropic, ILLMProvider)

    bedrock = get_provider("bedrock")
    assert bedrock is not None
    assert bedrock.name == "bedrock"

    ollama = get_provider("ollama")
    assert ollama is not None
    assert ollama.name == "ollama"

    # Non-existent provider
    unknown = get_provider("nonexistent")
    assert unknown is None


def test_provider_registry_is_registered():
    """Test checking if provider is registered."""
    assert ProviderRegistry.is_registered("anthropic")
    assert ProviderRegistry.is_registered("bedrock")
    assert ProviderRegistry.is_registered("ollama")
    assert not ProviderRegistry.is_registered("openai")


# =============================================================================
# Anthropic Provider Tests
# =============================================================================


def test_anthropic_provider_basic():
    """Test AnthropicProvider basic properties."""
    provider = get_provider("anthropic")
    assert provider.name == "anthropic"
    assert "Anthropic" in provider.display_name


def test_anthropic_provider_is_configured():
    """Test Anthropic configuration detection."""
    provider = get_provider("anthropic")

    # Without API key
    with patch.dict(os.environ, {}, clear=True):
        assert not provider.is_configured()

    # With API key
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
        assert provider.is_configured()


def test_anthropic_provider_list_models():
    """Test Anthropic model listing."""
    provider = get_provider("anthropic")
    models = provider.list_models()

    assert len(models) > 0
    assert all(isinstance(m, ModelInfo) for m in models)
    assert all(m.provider == "anthropic" for m in models)

    # Check default model exists
    default_models = [m for m in models if m.is_default]
    assert len(default_models) == 1


def test_anthropic_provider_get_default_model():
    """Test Anthropic default model selection."""
    provider = get_provider("anthropic")
    default = provider.get_default_model()

    assert default is not None
    assert "claude" in default.lower()


def test_anthropic_provider_validate_model():
    """Test Anthropic model validation."""
    provider = get_provider("anthropic")

    # Valid model
    assert provider.validate_model("claude-sonnet-4-20250514")

    # Invalid model
    assert not provider.validate_model("gpt-4")
    assert not provider.validate_model("nonexistent-model")


# =============================================================================
# Bedrock Provider Tests
# =============================================================================


def test_bedrock_provider_basic():
    """Test BedrockProvider basic properties."""
    provider = get_provider("bedrock")
    assert provider.name == "bedrock"
    assert "Bedrock" in provider.display_name or "AWS" in provider.display_name


def test_bedrock_provider_is_configured():
    """Test Bedrock configuration detection."""
    provider = get_provider("bedrock")

    # Without credentials (clear all AWS keys)
    env_clear = {k: v for k, v in os.environ.items() if not k.startswith("AWS_BEDROCK")}
    with patch.dict(os.environ, env_clear, clear=True):
        assert not provider.is_configured()

    # With only access key (still invalid)
    with patch.dict(os.environ, {"AWS_BEDROCK_ACCESS_KEY": "test"}, clear=True):
        assert not provider.is_configured()

    # With both credentials
    with patch.dict(
        os.environ,
        {
            "AWS_BEDROCK_ACCESS_KEY": "test_access",
            "AWS_BEDROCK_SECRET_ACCESS_KEY": "test_secret",
        },
        clear=True,
    ):
        assert provider.is_configured()


def test_bedrock_provider_list_models():
    """Test Bedrock model listing."""
    provider = get_provider("bedrock")
    models = provider.list_models()

    assert len(models) > 0
    assert all(isinstance(m, ModelInfo) for m in models)
    assert all(m.provider == "bedrock" for m in models)

    # Bedrock models should have "anthropic" in name
    assert all("anthropic" in m.name.lower() for m in models)


# =============================================================================
# Ollama Provider Tests
# =============================================================================


def test_ollama_provider_basic():
    """Test OllamaProvider basic properties."""
    provider = get_provider("ollama")
    assert provider.name == "ollama"
    assert "Ollama" in provider.display_name or "Local" in provider.display_name


def test_ollama_provider_is_configured():
    """Test Ollama configuration detection (always true - local)."""
    provider = get_provider("ollama")

    # Ollama is always "configured" (can use localhost)
    assert provider.is_configured()


@patch("requests.get")
def test_ollama_provider_is_available(mock_get):
    """Test Ollama availability check."""
    provider = get_provider("ollama")

    # Server running
    mock_get.return_value.status_code = 200
    assert provider.is_available()

    # Server not running
    mock_get.return_value.status_code = 404
    assert not provider.is_available()

    # Connection error
    mock_get.side_effect = Exception("Connection refused")
    assert not provider.is_available()


# =============================================================================
# Gemini Provider Tests
# =============================================================================


def test_gemini_provider_basic():
    """Test GeminiProvider basic properties."""
    provider = get_provider("gemini")
    assert provider is not None
    assert provider.name == "gemini"
    assert "Gemini" in provider.display_name


def test_gemini_provider_is_configured():
    """Test Gemini configuration detection."""
    provider = get_provider("gemini")

    # Without API key
    with patch.dict(os.environ, {}, clear=True):
        assert not provider.is_configured()

    # With GOOGLE_API_KEY
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True):
        assert provider.is_configured()

    # With GEMINI_API_KEY (fallback)
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
        assert provider.is_configured()


def test_gemini_provider_list_models():
    """Test Gemini model listing."""
    provider = get_provider("gemini")
    models = provider.list_models()

    assert len(models) >= 2
    assert all(isinstance(m, ModelInfo) for m in models)
    assert all(m.provider == "gemini" for m in models)

    # Check default model exists
    default_models = [m for m in models if m.is_default]
    assert len(default_models) == 1
    assert "gemini-3-pro" in default_models[0].name


def test_gemini_provider_get_default_model():
    """Test Gemini default model selection."""
    provider = get_provider("gemini")
    default = provider.get_default_model()

    assert default is not None
    assert default == "gemini-3-pro-preview"


def test_gemini_provider_validate_model():
    """Test Gemini model validation."""
    provider = get_provider("gemini")

    # Valid models
    assert provider.validate_model("gemini-3-pro-preview")
    assert provider.validate_model("gemini-3-flash-preview")

    # Invalid models
    assert not provider.validate_model("gpt-4")
    assert not provider.validate_model("claude-sonnet-4-20250514")


def test_provider_registry_includes_gemini():
    """Test that Gemini is registered in ProviderRegistry."""
    providers = ProviderRegistry.get_provider_names()
    assert "gemini" in providers


# =============================================================================
# ConfigResolver Integration Tests
# =============================================================================


@pytest.mark.no_auto_config
def test_config_resolver_fails_without_config(tmp_path):
    """Test that ConfigResolver raises ConfigurationError when not configured."""
    # Reset singleton to ensure clean state
    ConfigResolver.reset_instance()
    resolver = ConfigResolver.get_instance(tmp_path)

    with pytest.raises(ConfigurationError) as exc_info:
        resolver.resolve_provider()

    assert "No provider configured" in str(exc_info.value)
    assert exc_info.value.help_text is not None
    assert "lobster init" in exc_info.value.help_text


@pytest.mark.no_auto_config
def test_config_resolver_with_runtime_override(tmp_path):
    """Test that runtime override works without config file."""
    ConfigResolver.reset_instance()
    resolver = ConfigResolver.get_instance(tmp_path)

    provider, source = resolver.resolve_provider(runtime_override="anthropic")
    assert provider == "anthropic"
    assert source == "runtime flag --provider"


@pytest.mark.no_auto_config
def test_config_resolver_with_workspace_config(tmp_path):
    """Test that workspace config is respected."""
    from lobster.config.workspace_config import WorkspaceProviderConfig

    ConfigResolver.reset_instance()

    # Create workspace config
    workspace_path = tmp_path / ".lobster_workspace"
    workspace_path.mkdir()

    config = WorkspaceProviderConfig(global_provider="ollama", ollama_model="llama3:8b")
    config.save(workspace_path)

    # Resolve should use workspace config
    resolver = ConfigResolver.get_instance(workspace_path)
    provider, source = resolver.resolve_provider()

    assert provider == "ollama"
    assert source == "workspace config"


@pytest.mark.no_auto_config
def test_config_resolver_priority_order(tmp_path):
    """Test that runtime override beats workspace config."""
    from lobster.config.workspace_config import WorkspaceProviderConfig

    ConfigResolver.reset_instance()

    # Create workspace config with bedrock
    workspace_path = tmp_path / ".lobster_workspace"
    workspace_path.mkdir()

    config = WorkspaceProviderConfig(global_provider="bedrock")
    config.save(workspace_path)

    # Runtime override should win
    resolver = ConfigResolver.get_instance(workspace_path)
    provider, source = resolver.resolve_provider(runtime_override="anthropic")

    assert provider == "anthropic"
    assert source == "runtime flag --provider"


# =============================================================================
# LLMFactory Integration Tests
# =============================================================================


@pytest.mark.no_auto_config
def test_llm_factory_fails_without_config(tmp_path):
    """Test that LLMFactory raises ConfigurationError when not configured."""
    from lobster.config.llm_factory import LLMFactory

    ConfigResolver.reset_instance()

    with pytest.raises(ConfigurationError):
        LLMFactory.create_llm(
            model_config={"temperature": 0.7},
            agent_name="test_agent",
            workspace_path=tmp_path,
        )


@patch.dict(
    os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test123"}, clear=False
)
@patch("langchain_anthropic.ChatAnthropic")
def test_llm_factory_with_valid_config(mock_chat, tmp_path):
    """Test that LLMFactory creates LLM with valid configuration."""
    from lobster.config.workspace_config import WorkspaceProviderConfig
    from lobster.config.llm_factory import LLMFactory

    # Mock return value
    mock_chat.return_value = MagicMock()

    # Create workspace config
    workspace_path = tmp_path / ".lobster_workspace"
    workspace_path.mkdir()

    config = WorkspaceProviderConfig(
        global_provider="anthropic", anthropic_model="claude-sonnet-4-20250514"
    )
    config.save(workspace_path)

    llm = LLMFactory.create_llm(
        model_config={"temperature": 0.7, "max_tokens": 4096},
        agent_name="test_agent",
        workspace_path=workspace_path,
    )

    # Verify ChatAnthropic was called with correct params
    mock_chat.assert_called_once()
    call_kwargs = mock_chat.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-20250514"
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["max_tokens"] == 4096


def test_llm_factory_get_available_providers():
    """Test getting list of configured providers."""
    from lobster.config.llm_factory import LLMFactory

    # Without any API keys
    with patch.dict(os.environ, {}, clear=True):
        available = LLMFactory.get_available_providers()
        # Should be empty or only have ollama (if it's running)
        assert isinstance(available, list)

    # With Anthropic key
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
        available = LLMFactory.get_available_providers()
        assert "anthropic" in available


# =============================================================================
# End-to-End Workflow Tests
# =============================================================================


@patch.dict(
    os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test123"}, clear=False
)
@patch("langchain_anthropic.ChatAnthropic")
def test_end_to_end_provider_selection(mock_chat, tmp_path):
    """Test complete workflow from config to LLM creation."""
    from lobster.config.workspace_config import WorkspaceProviderConfig
    from lobster.config.llm_factory import create_llm

    # Mock return value
    mock_chat.return_value = MagicMock()

    # Step 1: Create workspace config
    workspace_path = tmp_path / ".lobster_workspace"
    workspace_path.mkdir()

    config = WorkspaceProviderConfig(
        global_provider="anthropic",
        anthropic_model="claude-sonnet-4-20250514",
        profile="production",
    )
    config.save(workspace_path)

    # Step 2: Create LLM via convenience function
    llm = create_llm(
        agent_name="supervisor",
        model_params={"temperature": 0.5},
        workspace_path=workspace_path,
    )

    # Verify provider was selected correctly
    mock_chat.assert_called_once()
    call_kwargs = mock_chat.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-20250514"


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False)
@patch("langchain_anthropic.ChatAnthropic")
def test_runtime_override_priority(mock_chat, tmp_path):
    """Test that runtime overrides beat workspace config."""
    from lobster.config.workspace_config import WorkspaceProviderConfig
    from lobster.config.llm_factory import create_llm

    # Mock return value
    mock_chat.return_value = MagicMock()

    # Setup workspace with bedrock
    workspace_path = tmp_path / ".lobster_workspace"
    workspace_path.mkdir()

    config = WorkspaceProviderConfig(global_provider="bedrock")
    config.save(workspace_path)

    # Override with anthropic at runtime
    llm = create_llm(
        agent_name="test",
        model_params={},
        provider_override="anthropic",  # Runtime override
        workspace_path=workspace_path,
    )

    # Should use Anthropic despite workspace config saying Bedrock
    mock_chat.assert_called_once()


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_invalid_provider_name_raises_error(tmp_path):
    """Test that invalid provider names raise ConfigurationError."""
    resolver = ConfigResolver.get_instance(tmp_path)

    with pytest.raises(ConfigurationError) as exc_info:
        resolver.resolve_provider(runtime_override="invalid_provider")

    assert "Invalid provider" in str(exc_info.value)
    assert "anthropic" in str(exc_info.value)  # Should list valid providers


def test_unregistered_provider_validation():
    """Test that Pydantic catches invalid providers at config load time."""
    from lobster.config.workspace_config import WorkspaceProviderConfig

    # Pydantic should reject invalid provider during model creation
    with pytest.raises(Exception) as exc_info:  # Pydantic ValidationError
        config = WorkspaceProviderConfig(global_provider="openai")  # Not valid yet

    # The error should mention invalid provider
    assert "openai" in str(exc_info.value).lower() or "provider" in str(exc_info.value).lower()


# =============================================================================
# Model Resolution Tests
# =============================================================================


def test_model_resolution_with_workspace_config(tmp_path):
    """Test that workspace model config is respected."""
    from lobster.config.workspace_config import WorkspaceProviderConfig

    workspace_path = tmp_path / ".lobster_workspace"
    workspace_path.mkdir()

    config = WorkspaceProviderConfig(
        global_provider="anthropic", anthropic_model="claude-opus-4-20250514"
    )
    config.save(workspace_path)

    resolver = ConfigResolver.get_instance(workspace_path)
    provider, _ = resolver.resolve_provider()
    model, source = resolver.resolve_model(provider=provider)

    assert model == "claude-opus-4-20250514"
    assert "workspace config" in source


def test_model_resolution_runtime_override(tmp_path):
    """Test that runtime model override has highest priority."""
    from lobster.config.workspace_config import WorkspaceProviderConfig

    workspace_path = tmp_path / ".lobster_workspace"
    workspace_path.mkdir()

    config = WorkspaceProviderConfig(
        global_provider="anthropic", anthropic_model="claude-sonnet-4-20250514"
    )
    config.save(workspace_path)

    resolver = ConfigResolver.get_instance(workspace_path)
    model, source = resolver.resolve_model(
        runtime_override="claude-opus-4-20250514", provider="anthropic"
    )

    assert model == "claude-opus-4-20250514"
    assert source == "runtime flag --model"


def test_model_resolution_uses_provider_default(tmp_path):
    """Test that provider default is used when no model configured."""
    from lobster.config.workspace_config import WorkspaceProviderConfig

    workspace_path = tmp_path / ".lobster_workspace"
    workspace_path.mkdir()

    # Config without specific model
    config = WorkspaceProviderConfig(global_provider="anthropic")
    config.save(workspace_path)

    resolver = ConfigResolver.get_instance(workspace_path)
    model, source = resolver.resolve_model(provider="anthropic")

    assert model is not None
    assert "provider default" in source


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


def test_llm_provider_enum_still_exists():
    """Test that LLMProvider enum exists for backward compat."""
    from lobster.config.llm_factory import LLMProvider

    assert LLMProvider.ANTHROPIC_DIRECT.value == "anthropic"
    assert LLMProvider.BEDROCK_ANTHROPIC.value == "bedrock"
    assert LLMProvider.OLLAMA.value == "ollama"
    assert LLMProvider.GEMINI.value == "gemini"


def test_create_llm_convenience_function():
    """Test that create_llm convenience function works."""
    from lobster.config.llm_factory import create_llm

    # Should raise ConfigurationError without config
    with pytest.raises(ConfigurationError):
        create_llm(agent_name="test", model_params={})
