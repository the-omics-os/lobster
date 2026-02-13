"""
Unit tests for unified model service architecture.

Tests cover:
- AnthropicModelService: static model catalog
- BedrockModelService: static model catalog
- OllamaModelServiceAdapter: dynamic model discovery via HTTP
- ModelServiceFactory: provider-specific service creation
- ModelInfo: dataclass representation
"""

from unittest.mock import MagicMock, patch

import pytest

from lobster.config.model_service import (
    AnthropicModelService,
    BaseModelService,
    BedrockModelService,
    ModelInfo,
    ModelServiceFactory,
    OllamaModelServiceAdapter,
)


# =============================================================================
# ModelInfo Tests
# =============================================================================


class TestModelInfo:
    """Test ModelInfo dataclass."""

    def test_create_model_info(self):
        """Test creating ModelInfo with all fields."""
        info = ModelInfo(
            name="claude-sonnet-4-20250514",
            display_name="Claude Sonnet 4",
            description="Latest Sonnet model",
            provider="anthropic",
            context_window=200000,
            is_default=True,
        )

        assert info.name == "claude-sonnet-4-20250514"
        assert info.display_name == "Claude Sonnet 4"
        assert info.description == "Latest Sonnet model"
        assert info.provider == "anthropic"
        assert info.context_window == 200000
        assert info.is_default is True

    def test_create_model_info_minimal(self):
        """Test creating ModelInfo with minimal fields."""
        info = ModelInfo(
            name="llama3:8b",
            display_name="Llama 3 8B",
            description="Local model",
            provider="ollama",
        )

        assert info.name == "llama3:8b"
        assert info.context_window is None
        assert info.is_default is False


# =============================================================================
# AnthropicModelService Tests
# =============================================================================


class TestAnthropicModelService:
    """Test Anthropic model service with static catalog."""

    def test_list_models(self):
        """Test listing all Anthropic models."""
        service = AnthropicModelService()
        models = service.list_models()

        assert len(models) > 0
        assert all(isinstance(m, ModelInfo) for m in models)
        assert all(m.provider == "anthropic" for m in models)

    def test_list_models_returns_copy(self):
        """Test that list_models returns a copy to prevent mutation."""
        service = AnthropicModelService()
        models1 = service.list_models()
        models2 = service.list_models()

        assert models1 is not models2

    def test_get_model_info_valid(self):
        """Test getting info for valid Anthropic model."""
        service = AnthropicModelService()
        info = service.get_model_info("claude-sonnet-4-20250514")

        assert info is not None
        assert info.name == "claude-sonnet-4-20250514"
        assert info.provider == "anthropic"
        assert info.context_window == 200000

    def test_get_model_info_invalid(self):
        """Test getting info for invalid model returns None."""
        service = AnthropicModelService()
        info = service.get_model_info("nonexistent-model")

        assert info is None

    def test_validate_model_valid(self):
        """Test validating valid Anthropic model."""
        service = AnthropicModelService()

        assert service.validate_model("claude-sonnet-4-20250514") is True
        assert service.validate_model("claude-opus-4-20250514") is True
        assert service.validate_model("claude-3-5-haiku-20241022") is True

    def test_validate_model_invalid(self):
        """Test validating invalid model."""
        service = AnthropicModelService()

        assert service.validate_model("gpt-4") is False
        assert service.validate_model("llama3:8b") is False
        assert service.validate_model("nonexistent") is False

    def test_get_default_model(self):
        """Test getting default Anthropic model."""
        service = AnthropicModelService()
        default = service.get_default_model()

        assert default is not None
        assert service.validate_model(default) is True

    def test_get_model_names(self):
        """Test getting model names for tab completion."""
        service = AnthropicModelService()
        names = service.get_model_names()

        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)
        assert "claude-sonnet-4-20250514" in names

    def test_catalog_has_default_marked(self):
        """Test that exactly one model is marked as default."""
        service = AnthropicModelService()
        models = service.list_models()
        defaults = [m for m in models if m.is_default]

        assert len(defaults) == 1


# =============================================================================
# BedrockModelService Tests
# =============================================================================


class TestBedrockModelService:
    """Test Bedrock model service with static catalog."""

    def test_list_models(self):
        """Test listing all Bedrock models."""
        service = BedrockModelService()
        models = service.list_models()

        assert len(models) > 0
        assert all(isinstance(m, ModelInfo) for m in models)
        assert all(m.provider == "bedrock" for m in models)

    def test_bedrock_model_id_format(self):
        """Test Bedrock model IDs follow expected format."""
        service = BedrockModelService()
        models = service.list_models()

        for model in models:
            assert model.name.startswith("anthropic.")
            assert ":0" in model.name or model.name.endswith(":0")

    def test_get_model_info_valid(self):
        """Test getting info for valid Bedrock model."""
        service = BedrockModelService()
        info = service.get_model_info("anthropic.claude-sonnet-4-20250514-v1:0")

        assert info is not None
        assert info.name == "anthropic.claude-sonnet-4-20250514-v1:0"
        assert info.provider == "bedrock"

    def test_get_model_info_invalid(self):
        """Test getting info for invalid model returns None."""
        service = BedrockModelService()
        info = service.get_model_info("anthropic.nonexistent-model")

        assert info is None

    def test_validate_model_valid(self):
        """Test validating valid Bedrock model."""
        service = BedrockModelService()

        assert service.validate_model("anthropic.claude-sonnet-4-20250514-v1:0") is True
        assert (
            service.validate_model("anthropic.claude-3-5-sonnet-20241022-v2:0") is True
        )

    def test_validate_model_invalid(self):
        """Test validating invalid model."""
        service = BedrockModelService()

        assert service.validate_model("claude-sonnet-4-20250514") is False
        assert service.validate_model("llama3:8b") is False

    def test_get_default_model(self):
        """Test getting default Bedrock model."""
        service = BedrockModelService()
        default = service.get_default_model()

        assert default is not None
        assert service.validate_model(default) is True

    def test_catalog_has_default_marked(self):
        """Test that exactly one model is marked as default."""
        service = BedrockModelService()
        models = service.list_models()
        defaults = [m for m in models if m.is_default]

        assert len(defaults) == 1


# =============================================================================
# OllamaModelServiceAdapter Tests
# =============================================================================


class TestOllamaModelServiceAdapter:
    """Test Ollama model service adapter with mocked HTTP calls."""

    def test_init_default_url(self):
        """Test default initialization."""
        service = OllamaModelServiceAdapter()
        assert service.base_url == "http://localhost:11434"

    def test_init_custom_url(self):
        """Test initialization with custom URL."""
        service = OllamaModelServiceAdapter(base_url="http://custom:8080")
        assert service.base_url == "http://custom:8080"

    @patch("lobster.config.ollama_service.OllamaService")
    def test_list_models_success(self, mock_ollama_service):
        """Test listing Ollama models via HTTP."""
        # Mock response
        mock_model = MagicMock()
        mock_model.name = "llama3:8b-instruct"
        mock_model.description = "Llama 3 8B model"
        mock_ollama_service.list_models.return_value = [mock_model]

        service = OllamaModelServiceAdapter()
        models = service.list_models()

        assert len(models) == 1
        assert models[0].name == "llama3:8b-instruct"
        assert models[0].provider == "ollama"
        assert models[0].is_default is True  # First model is default

    @patch("lobster.config.ollama_service.OllamaService")
    def test_list_models_failure(self, mock_ollama_service):
        """Test list_models handles errors gracefully."""
        mock_ollama_service.list_models.side_effect = Exception("Connection refused")

        service = OllamaModelServiceAdapter()
        models = service.list_models()

        assert models == []

    @patch("lobster.config.ollama_service.OllamaService")
    def test_validate_model_success(self, mock_ollama_service):
        """Test validating Ollama model."""
        mock_ollama_service.validate_model.return_value = True

        service = OllamaModelServiceAdapter()
        result = service.validate_model("llama3:8b")

        assert result is True
        mock_ollama_service.validate_model.assert_called_once_with(
            "llama3:8b", "http://localhost:11434"
        )

    @patch("lobster.config.ollama_service.OllamaService")
    def test_validate_model_failure(self, mock_ollama_service):
        """Test validate_model handles errors gracefully."""
        mock_ollama_service.validate_model.side_effect = Exception("Error")

        service = OllamaModelServiceAdapter()
        result = service.validate_model("llama3:8b")

        assert result is False

    @patch("lobster.config.ollama_service.OllamaService")
    def test_get_default_model_success(self, mock_ollama_service):
        """Test getting default Ollama model."""
        mock_ollama_service.select_best_model.return_value = "llama3:70b-instruct"

        service = OllamaModelServiceAdapter()
        default = service.get_default_model()

        assert default == "llama3:70b-instruct"

    @patch("lobster.config.ollama_service.OllamaService")
    def test_get_default_model_failure(self, mock_ollama_service):
        """Test get_default_model returns fallback on error."""
        mock_ollama_service.select_best_model.side_effect = Exception("Error")

        service = OllamaModelServiceAdapter()
        default = service.get_default_model()

        assert default == "llama3:8b-instruct"

    @patch("lobster.config.ollama_service.OllamaService")
    def test_is_available_success(self, mock_ollama_service):
        """Test checking Ollama availability."""
        mock_ollama_service.is_available.return_value = True

        service = OllamaModelServiceAdapter()
        result = service.is_available()

        assert result is True

    @patch("lobster.config.ollama_service.OllamaService")
    def test_is_available_failure(self, mock_ollama_service):
        """Test is_available handles errors gracefully."""
        mock_ollama_service.is_available.side_effect = Exception("Error")

        service = OllamaModelServiceAdapter()
        result = service.is_available()

        assert result is False


# =============================================================================
# ModelServiceFactory Tests
# =============================================================================


class TestModelServiceFactory:
    """Test factory for provider-specific model services."""

    def test_get_service_anthropic(self):
        """Test getting Anthropic model service."""
        service = ModelServiceFactory.get_service("anthropic")

        assert isinstance(service, AnthropicModelService)

    def test_get_service_bedrock(self):
        """Test getting Bedrock model service."""
        service = ModelServiceFactory.get_service("bedrock")

        assert isinstance(service, BedrockModelService)

    def test_get_service_ollama(self):
        """Test getting Ollama model service."""
        service = ModelServiceFactory.get_service("ollama")

        assert isinstance(service, OllamaModelServiceAdapter)

    def test_get_service_ollama_with_kwargs(self):
        """Test getting Ollama service with custom base_url."""
        service = ModelServiceFactory.get_service(
            "ollama", base_url="http://custom:8080"
        )

        assert isinstance(service, OllamaModelServiceAdapter)
        assert service.base_url == "http://custom:8080"

    def test_get_service_invalid_provider(self):
        """Test getting service for invalid provider raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            ModelServiceFactory.get_service("invalid_provider")

        assert "Unknown provider" in str(excinfo.value)
        assert "anthropic" in str(excinfo.value)
        assert "bedrock" in str(excinfo.value)
        assert "ollama" in str(excinfo.value)

    def test_get_supported_providers(self):
        """Test listing supported providers."""
        providers = ModelServiceFactory.get_supported_providers()

        assert "anthropic" in providers
        assert "bedrock" in providers
        assert "ollama" in providers
        assert len(providers) == 3

    def test_list_all_models(self):
        """Test listing models from all providers."""
        with patch.object(OllamaModelServiceAdapter, "list_models", return_value=[]):
            models = ModelServiceFactory.list_all_models()

        # Should have Anthropic + Bedrock models (Ollama mocked empty)
        assert len(models) > 0
        providers = {m.provider for m in models}
        assert "anthropic" in providers
        assert "bedrock" in providers


# =============================================================================
# Base Class Tests
# =============================================================================


class TestBaseModelService:
    """Test BaseModelService interface."""

    def test_get_model_names_implementation(self):
        """Test default implementation of get_model_names."""
        service = AnthropicModelService()
        names = service.get_model_names()

        # Should return list of model.name from list_models
        models = service.list_models()
        expected_names = [m.name for m in models]

        assert names == expected_names


# =============================================================================
# Integration Tests
# =============================================================================


class TestModelServiceIntegration:
    """Integration tests for model service with config resolution."""

    def test_anthropic_model_catalog_consistency(self):
        """Test Anthropic models are consistent and complete."""
        service = AnthropicModelService()
        models = service.list_models()

        # All models should have required fields
        for model in models:
            assert model.name
            assert model.display_name
            assert model.description
            assert model.provider == "anthropic"
            assert model.context_window is not None

        # Should have main model families
        model_names = service.get_model_names()
        assert any("sonnet" in n for n in model_names)
        assert any("opus" in n for n in model_names)
        assert any("haiku" in n for n in model_names)

    def test_bedrock_model_catalog_consistency(self):
        """Test Bedrock models are consistent and complete."""
        service = BedrockModelService()
        models = service.list_models()

        # All models should have required fields
        for model in models:
            assert model.name
            assert model.display_name
            assert model.description
            assert model.provider == "bedrock"

        # Should have main model families
        model_names = service.get_model_names()
        assert any("sonnet" in n for n in model_names)
        assert any("opus" in n for n in model_names)
        assert any("haiku" in n for n in model_names)

    def test_factory_services_implement_interface(self):
        """Test all factory services implement BaseModelService."""
        for provider in ModelServiceFactory.get_supported_providers():
            with patch.object(
                OllamaModelServiceAdapter, "list_models", return_value=[]
            ):
                service = ModelServiceFactory.get_service(provider)

                # All required methods should exist
                assert hasattr(service, "list_models")
                assert hasattr(service, "get_model_info")
                assert hasattr(service, "validate_model")
                assert hasattr(service, "get_default_model")
                assert hasattr(service, "get_model_names")
