"""
Unit tests for the refactored provider system.

Tests cover:
- ProviderRegistry: registration, discovery, health checks
- Individual providers: Anthropic, Bedrock, Ollama
- ConfigResolver integration with providers
- LLMFactory end-to-end flow
- Error handling for missing configuration
"""

import importlib.util
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

    # Should have all 6 providers registered
    assert "anthropic" in providers
    assert "bedrock" in providers
    assert "ollama" in providers
    assert "gemini" in providers
    assert "azure" in providers
    assert "openai" in providers
    assert len(providers) == 6


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
    assert ProviderRegistry.is_registered("openai")
    assert not ProviderRegistry.is_registered("nonexistent_provider")


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
# Azure Provider Tests
# =============================================================================


def test_azure_provider_basic():
    """Test AzureProvider basic properties."""
    provider = get_provider("azure")
    assert provider is not None
    assert provider.name == "azure"
    assert "Azure" in provider.display_name


def test_azure_provider_is_configured():
    """Test Azure configuration detection."""
    provider = get_provider("azure")

    # Without any credentials
    with patch.dict(os.environ, {}, clear=True):
        assert not provider.is_configured()

    # With only credential (missing endpoint)
    with patch.dict(os.environ, {"AZURE_AI_CREDENTIAL": "test-key"}, clear=True):
        assert not provider.is_configured()

    # With only endpoint (missing credential)
    with patch.dict(
        os.environ,
        {"AZURE_AI_ENDPOINT": "https://test.inference.ai.azure.com/"},
        clear=True,
    ):
        assert not provider.is_configured()

    # With both new-style env vars
    with patch.dict(
        os.environ,
        {
            "AZURE_AI_CREDENTIAL": "test-key",
            "AZURE_AI_ENDPOINT": "https://test.inference.ai.azure.com/",
        },
        clear=True,
    ):
        assert provider.is_configured()

    # With legacy Azure OpenAI env vars
    with patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        },
        clear=True,
    ):
        assert provider.is_configured()


def test_azure_provider_list_models():
    """Test Azure model listing."""
    provider = get_provider("azure")
    models = provider.list_models()

    assert len(models) >= 5  # GPT, DeepSeek, Cohere, Phi, Mistral
    assert all(isinstance(m, ModelInfo) for m in models)
    assert all(m.provider == "azure" for m in models)

    # Check default model exists
    default_models = [m for m in models if m.is_default]
    assert len(default_models) == 1
    assert "gpt-4o" in default_models[0].name

    # Verify multi-provider models exist
    model_names = [m.name for m in models]
    assert "gpt-4o" in model_names
    assert "deepseek-r1" in model_names
    assert "phi-4" in model_names


def test_azure_provider_get_default_model():
    """Test Azure default model selection."""
    provider = get_provider("azure")
    default = provider.get_default_model()

    assert default is not None
    assert default == "gpt-4o"


def test_azure_provider_validate_model():
    """Test Azure model validation."""
    provider = get_provider("azure")

    # Known models
    assert provider.validate_model("gpt-4o")
    assert provider.validate_model("deepseek-r1")
    assert provider.validate_model("phi-4")

    # Unknown models also return True (for future compatibility)
    assert provider.validate_model("future-model-2025")


@pytest.mark.skipif(
    not importlib.util.find_spec("langchain_azure_ai"),
    reason="langchain-azure-ai not installed",
)
@patch("langchain_azure_ai.chat_models.AzureAIChatCompletionsModel")
def test_azure_provider_create_chat_model(mock_azure_chat):
    """Test Azure chat model creation."""
    mock_azure_chat.return_value = MagicMock()

    with patch.dict(
        os.environ,
        {
            "AZURE_AI_CREDENTIAL": "test-key",
            "AZURE_AI_ENDPOINT": "https://test.inference.ai.azure.com/",
        },
        clear=True,
    ):
        provider = get_provider("azure")
        llm = provider.create_chat_model("gpt-4o", temperature=0.7)

        mock_azure_chat.assert_called_once()
        call_kwargs = mock_azure_chat.call_args.kwargs
        assert call_kwargs["model_name"] == "gpt-4o"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["api_version"] == "2024-05-01-preview"
        assert "endpoint" in call_kwargs
        assert "credential" in call_kwargs


@pytest.mark.skipif(
    not importlib.util.find_spec("langchain_azure_ai"),
    reason="langchain-azure-ai not installed",
)
def test_azure_provider_create_chat_model_missing_credential():
    """Test Azure chat model raises error without credential."""
    with patch.dict(
        os.environ,
        {"AZURE_AI_ENDPOINT": "https://test.inference.ai.azure.com/"},
        clear=True,
    ):
        provider = get_provider("azure")
        with pytest.raises(ValueError, match="credential not found"):
            provider.create_chat_model("gpt-4o")


@pytest.mark.skipif(
    not importlib.util.find_spec("langchain_azure_ai"),
    reason="langchain-azure-ai not installed",
)
def test_azure_provider_create_chat_model_missing_endpoint():
    """Test Azure chat model raises error without endpoint."""
    with patch.dict(os.environ, {"AZURE_AI_CREDENTIAL": "test-key"}, clear=True):
        provider = get_provider("azure")
        with pytest.raises(ValueError, match="endpoint not found"):
            provider.create_chat_model("gpt-4o")


def test_azure_provider_configuration_help():
    """Test Azure configuration help contains expected info."""
    provider = get_provider("azure")
    help_text = provider.get_configuration_help()

    assert "AZURE_AI_ENDPOINT" in help_text
    assert "AZURE_AI_CREDENTIAL" in help_text
    assert "gpt-4o" in help_text
    assert "deepseek" in help_text.lower()


def test_provider_registry_includes_azure():
    """Test that Azure is registered in ProviderRegistry."""
    providers = ProviderRegistry.get_provider_names()
    assert "azure" in providers


# =============================================================================
# OpenAI Provider Tests
# =============================================================================


def test_openai_provider_basic():
    """Test OpenAIProvider basic properties."""
    provider = get_provider("openai")
    assert provider is not None
    assert provider.name == "openai"
    assert "OpenAI" in provider.display_name


def test_openai_provider_is_configured():
    """Test OpenAI configuration detection."""
    provider = get_provider("openai")

    # Without API key
    with patch.dict(os.environ, {}, clear=True):
        assert not provider.is_configured()

    # With OPENAI_API_KEY
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
        assert provider.is_configured()


def test_openai_provider_list_models():
    """Test OpenAI model listing."""
    provider = get_provider("openai")
    models = provider.list_models()

    assert len(models) == 5  # gpt-4o, gpt-4o-mini, o1, o1-mini, o3-mini
    assert all(isinstance(m, ModelInfo) for m in models)
    assert all(m.provider == "openai" for m in models)

    # Check default model exists
    default_models = [m for m in models if m.is_default]
    assert len(default_models) == 1
    assert default_models[0].name == "gpt-4o"


def test_openai_provider_get_default_model():
    """Test OpenAI default model selection."""
    provider = get_provider("openai")
    default = provider.get_default_model()

    assert default is not None
    assert default == "gpt-4o"


def test_openai_provider_validate_model():
    """Test OpenAI model validation."""
    provider = get_provider("openai")

    # Known models
    assert provider.validate_model("gpt-4o")
    assert provider.validate_model("gpt-4o-mini")
    assert provider.validate_model("o1")
    assert provider.validate_model("o3-mini")

    # Unknown models
    assert not provider.validate_model("claude-sonnet-4-20250514")
    assert not provider.validate_model("nonexistent-model")


@pytest.mark.skipif(
    not importlib.util.find_spec("langchain_openai"),
    reason="langchain-openai not installed",
)
@patch("langchain_openai.ChatOpenAI")
def test_openai_provider_create_chat_model(mock_openai_chat):
    """Test OpenAI chat model creation."""
    mock_openai_chat.return_value = MagicMock()

    with patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "sk-test-key"},
        clear=True,
    ):
        provider = get_provider("openai")
        llm = provider.create_chat_model("gpt-4o", temperature=0.7, max_tokens=8192)

        mock_openai_chat.assert_called_once()
        call_kwargs = mock_openai_chat.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 8192
        assert call_kwargs["api_key"] == "sk-test-key"


@pytest.mark.skipif(
    not importlib.util.find_spec("langchain_openai"),
    reason="langchain-openai not installed",
)
@patch("langchain_openai.ChatOpenAI")
def test_openai_provider_create_chat_model_o1(mock_openai_chat):
    """Test that o1 reasoning models skip temperature and max_tokens."""
    mock_openai_chat.return_value = MagicMock()

    with patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "sk-test-key"},
        clear=True,
    ):
        provider = get_provider("openai")
        llm = provider.create_chat_model("o1", temperature=0.7, max_tokens=8192)

        mock_openai_chat.assert_called_once()
        call_kwargs = mock_openai_chat.call_args.kwargs
        assert call_kwargs["model"] == "o1"
        # o1 models should NOT have temperature or max_tokens
        assert "temperature" not in call_kwargs
        assert "max_tokens" not in call_kwargs


def test_openai_provider_configuration_help():
    """Test OpenAI configuration help contains expected info."""
    provider = get_provider("openai")
    help_text = provider.get_configuration_help()

    assert "OPENAI_API_KEY" in help_text
    assert "platform.openai.com" in help_text
    assert "gpt-4o" in help_text
    assert "o1" in help_text


def test_provider_registry_includes_openai():
    """Test that OpenAI is registered in ProviderRegistry."""
    providers = ProviderRegistry.get_provider_names()
    assert "openai" in providers


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


def test_llm_factory_with_valid_config(tmp_path):
    """Test that LLMFactory creates LLM with valid configuration."""
    pytest.importorskip("langchain_anthropic", reason="langchain-anthropic not installed")
    from lobster.config.llm_factory import LLMFactory
    from lobster.config.workspace_config import WorkspaceProviderConfig

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test123"}, clear=False), \
         patch("langchain_anthropic.ChatAnthropic") as mock_chat:
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


def test_end_to_end_provider_selection(tmp_path):
    """Test complete workflow from config to LLM creation."""
    pytest.importorskip("langchain_anthropic", reason="langchain-anthropic not installed")
    from lobster.config.llm_factory import create_llm
    from lobster.config.workspace_config import WorkspaceProviderConfig

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test123"}, clear=False), \
         patch("langchain_anthropic.ChatAnthropic") as mock_chat:
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


def test_runtime_override_priority(tmp_path):
    """Test that runtime overrides beat workspace config."""
    pytest.importorskip("langchain_anthropic", reason="langchain-anthropic not installed")
    from lobster.config.llm_factory import create_llm
    from lobster.config.workspace_config import WorkspaceProviderConfig

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False), \
         patch("langchain_anthropic.ChatAnthropic") as mock_chat:
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
        config = WorkspaceProviderConfig(global_provider="invalid_fake_provider")

    # The error should mention invalid provider
    assert (
        "invalid_fake_provider" in str(exc_info.value).lower()
        or "provider" in str(exc_info.value).lower()
    )


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
    assert LLMProvider.AZURE.value == "azure"
    assert LLMProvider.OPENAI.value == "openai"


def test_create_llm_convenience_function():
    """Test that create_llm convenience function works."""
    from lobster.config.llm_factory import create_llm

    # Should raise ConfigurationError without config
    with pytest.raises(ConfigurationError):
        create_llm(agent_name="test", model_params={})
