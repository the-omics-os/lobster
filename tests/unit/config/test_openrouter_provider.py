"""Tests for OpenRouter provider implementation."""

import os
from unittest.mock import MagicMock, patch

import pytest

from lobster.config.providers.openrouter_provider import OpenRouterProvider


@pytest.fixture(autouse=True)
def clear_model_cache():
    """Reset the class-level model cache before each test."""
    OpenRouterProvider._models_cache = None
    yield
    OpenRouterProvider._models_cache = None


@pytest.fixture
def provider():
    return OpenRouterProvider()


class TestIsConfigured:
    def test_configured_with_key(self, provider, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
        assert provider.is_configured() is True

    def test_not_configured_without_key(self, provider, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert provider.is_configured() is False

    def test_not_configured_empty_key(self, provider, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
        assert provider.is_configured() is False


class TestIsAvailable:
    def test_available_when_configured(self, provider, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
        assert provider.is_available() is True

    def test_not_available_when_not_configured(self, provider, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert provider.is_available() is False


class TestProviderIdentity:
    def test_name(self, provider):
        assert provider.name == "openrouter"

    def test_display_name(self, provider):
        assert "OpenRouter" in provider.display_name

    def test_default_model(self, provider):
        model = provider.get_default_model()
        assert "/" in model  # OpenRouter models use provider/model-name format
        assert model == "anthropic/claude-sonnet-4-5"


class TestListModels:
    def test_list_models_uses_fallback_on_network_error(self, provider, monkeypatch):
        """When network fails, returns curated fallback catalog."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        with patch("httpx.get") as mock_get:
            mock_get.side_effect = Exception("Network error")
            models = provider.list_models()

        assert len(models) >= 10  # Fallback has at least 10 models
        names = [m.name for m in models]
        assert "anthropic/claude-sonnet-4-5" in names
        assert "openai/gpt-4o" in names

    def test_list_models_parses_live_catalog(self, provider, monkeypatch):
        """When API succeeds, returns parsed live catalog."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "anthropic/claude-3-5-sonnet",
                    "name": "Claude 3.5 Sonnet",
                    "description": "Anthropic's fastest model",
                    "context_length": 200000,
                    "pricing": {"prompt": "0.000003", "completion": "0.000015"},
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            models = provider.list_models()

        assert len(models) == 1
        assert models[0].name == "anthropic/claude-3-5-sonnet"

    def test_list_models_caches_result(self, provider, monkeypatch):
        """Second call uses cache, does not re-fetch."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "openai/gpt-4o",
                    "name": "GPT-4o",
                    "description": "",
                    "context_length": 128000,
                    "pricing": {"prompt": "0.0000025", "completion": "0.000010"},
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response) as mock_get:
            provider.list_models()
            provider.list_models()

        assert mock_get.call_count == 1  # Only called once

    def test_list_models_fallback_when_httpx_missing(self, provider, monkeypatch):
        """Falls back gracefully when httpx is not installed."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        with patch.dict("sys.modules", {"httpx": None}):
            models = provider.list_models()

        assert len(models) >= 10

    def test_list_models_caches_fallback_on_failure(self, provider, monkeypatch):
        """Fallback catalog is cached so subsequent calls don't re-fetch."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        with patch("httpx.get") as mock_get:
            mock_get.side_effect = Exception("Network error")
            models1 = provider.list_models()
            models2 = provider.list_models()

        assert mock_get.call_count == 1  # Only attempted once, then cache used
        assert models1 == models2

    def test_list_models_parses_pricing(self, provider, monkeypatch):
        """Parsed live catalog models include correct pricing values."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "anthropic/claude-3-5-sonnet",
                    "name": "Claude 3.5 Sonnet",
                    "description": "Anthropic's fastest model",
                    "context_length": 200000,
                    "pricing": {"prompt": "0.000003", "completion": "0.000015"},
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            models = provider.list_models()

        assert models[0].input_cost_per_million == pytest.approx(3.0)
        assert models[0].output_cost_per_million == pytest.approx(15.0)


class TestValidateModel:
    def test_validate_with_cache_known_model(self, provider):
        """Returns True for model in populated cache."""
        from lobster.config.providers.base_provider import ModelInfo

        OpenRouterProvider._models_cache = [
            ModelInfo(
                name="anthropic/claude-sonnet-4-5",
                display_name="Claude Sonnet 4.5",
                description="",
                provider="openrouter",
            )
        ]
        assert provider.validate_model("anthropic/claude-sonnet-4-5") is True

    def test_validate_with_cache_unknown_model(self, provider):
        """Returns False for model NOT in populated cache."""
        from lobster.config.providers.base_provider import ModelInfo

        OpenRouterProvider._models_cache = [
            ModelInfo(
                name="openai/gpt-4o",
                display_name="GPT-4o",
                description="",
                provider="openrouter",
            )
        ]
        assert provider.validate_model("made-up/model") is False

    def test_validate_without_cache_accepts_nonempty(self, provider):
        """Without cache, accepts any non-empty model ID (passthrough)."""
        assert OpenRouterProvider._models_cache is None
        assert provider.validate_model("any/model-id") is True

    def test_validate_rejects_empty_string(self, provider):
        assert provider.validate_model("") is False


class TestCreateChatModel:
    def test_create_chat_model_success(self, provider, monkeypatch):
        """Creates ChatOpenAI with correct base_url and headers."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            llm = provider.create_chat_model("anthropic/claude-sonnet-4-5")

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert "HTTP-Referer" in call_kwargs["default_headers"]
        assert "lobsterbio.com" in call_kwargs["default_headers"]["HTTP-Referer"]
        assert call_kwargs["default_headers"]["X-Title"] == "Lobster AI"
        assert call_kwargs["model"] == "anthropic/claude-sonnet-4-5"

    def test_create_chat_model_no_api_key_raises(self, provider, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            provider.create_chat_model("anthropic/claude-sonnet-4-5")

    def test_create_chat_model_passes_temperature_max_tokens(
        self, provider, monkeypatch
    ):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            provider.create_chat_model(
                "openai/gpt-4o", temperature=0.5, max_tokens=2048
            )

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 2048


class TestVerifyConnection:
    def test_verify_connection_success_returns_credit_info(self, provider, monkeypatch):
        """Valid key with credit balance returns True and usage/remaining info."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "usage": 1.25,
                "limit": 10.00,
                "limit_remaining": 8.75,
                "is_free_tier": False,
            }
        }

        with patch("httpx.get", return_value=mock_response):
            ok, msg = provider.verify_connection()

        assert ok is True
        assert "$1.2500 used" in msg
        assert "$8.7500 remaining" in msg

    def test_verify_connection_invalid_key_returns_false(self, provider, monkeypatch):
        """401 response means invalid API key — returns False with actionable message."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-bad-key")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            ok, msg = provider.verify_connection()

        assert ok is False
        assert "Invalid API key" in msg
        assert "openrouter.ai/keys" in msg

    def test_verify_connection_zero_credits_returns_false(self, provider, monkeypatch):
        """Exhausted credit limit returns False with credit guidance."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "usage": 10.00,
                "limit": 10.00,
                "limit_remaining": 0,
                "is_free_tier": False,
            }
        }

        with patch("httpx.get", return_value=mock_response):
            ok, msg = provider.verify_connection()

        assert ok is False
        assert "Insufficient credits" in msg
        assert "openrouter.ai/credits" in msg

    def test_verify_connection_free_tier_shows_free_tier(self, provider, monkeypatch):
        """Free tier accounts (no limit field) display 'free tier' in the message."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "usage": 0.0,
                "limit": None,
                "limit_remaining": None,
                "is_free_tier": True,
            }
        }

        with patch("httpx.get", return_value=mock_response):
            ok, msg = provider.verify_connection()

        assert ok is True
        assert "free tier" in msg

    def test_verify_connection_httpx_unavailable_falls_back_to_inference(
        self, provider, monkeypatch
    ):
        """When httpx is not installed, falls back to inference-based verification."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock()

        with patch.dict("sys.modules", {"httpx": None}):
            with patch.object(provider, "create_chat_model", return_value=mock_llm):
                ok, msg = provider.verify_connection("anthropic/claude-sonnet-4-5")

        assert ok is True
        assert "OpenRouter" in msg
        mock_llm.invoke.assert_called_once_with("Hi")

    def test_verify_connection_no_api_key_returns_false(self, provider, monkeypatch):
        """Missing OPENROUTER_API_KEY returns False immediately without network call."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with patch("httpx.get") as mock_get:
            ok, msg = provider.verify_connection()

        assert ok is False
        assert "OPENROUTER_API_KEY" in msg
        mock_get.assert_not_called()

    def test_verify_connection_network_error_returns_false(self, provider, monkeypatch):
        """Network/unexpected errors return False with error details."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        with patch("httpx.get", side_effect=Exception("Connection refused")):
            ok, msg = provider.verify_connection()

        assert ok is False
        assert "OpenRouter verification failed" in msg


class TestGetConfigurationHelp:
    def test_configuration_help_contains_key_info(self, provider):
        help_text = provider.get_configuration_help()
        assert "OPENROUTER_API_KEY" in help_text
        assert "openrouter.ai/keys" in help_text
        assert "anthropic/claude-sonnet-4-5" in help_text

    def test_configuration_help_lists_models(self, provider):
        help_text = provider.get_configuration_help()
        assert "anthropic/" in help_text  # At least one provider/model in the list


class TestRegistryIntegration:
    def test_openrouter_in_provider_registry(self):
        """OpenRouterProvider is discoverable via ProviderRegistry."""
        from lobster.config.providers.registry import ProviderRegistry

        ProviderRegistry.reset()
        ProviderRegistry._ensure_initialized()
        assert ProviderRegistry.is_registered("openrouter")

    def test_get_provider_returns_openrouter(self):
        from lobster.config.providers import get_provider

        provider = get_provider("openrouter")
        assert provider is not None
        assert provider.name == "openrouter"

    def test_openrouter_importable_from_providers_package(self):
        """OpenRouterProvider can be imported directly from the providers package."""
        from lobster.config.providers import OpenRouterProvider

        assert OpenRouterProvider().name == "openrouter"
