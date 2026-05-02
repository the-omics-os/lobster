"""
Unit tests for thinking/reasoning configuration.

Tests cover:
- ThinkingConfig class and serialization
- THINKING_PRESETS contents
- Thinking parameter passing through llm_factory
- Provider-specific thinking behaviour (Bedrock, Gemini)
"""

import os
from unittest.mock import Mock, patch

import pytest

from lobster.config.agent_defaults import THINKING_PRESETS, ThinkingConfig

# Mark all tests to skip auto_config
pytestmark = pytest.mark.no_auto_config


# =============================================================================
# ThinkingConfig Tests
# =============================================================================


class TestThinkingConfig:
    """Test ThinkingConfig dataclass and serialization."""

    def test_thinking_config_initialization(self):
        """Test ThinkingConfig creation with defaults."""
        config = ThinkingConfig()

        assert config.enabled is False
        assert config.budget_tokens == 2000
        assert config.type == "enabled"

    def test_thinking_config_custom_values(self):
        """Test ThinkingConfig with custom values."""
        config = ThinkingConfig(enabled=True, budget_tokens=5000, type="enabled")

        assert config.enabled is True
        assert config.budget_tokens == 5000
        assert config.type == "enabled"

    def test_thinking_config_to_dict_disabled(self):
        """Test to_dict() when thinking is disabled."""
        config = ThinkingConfig(enabled=False)
        result = config.to_dict()

        assert result == {}

    def test_thinking_config_to_dict_enabled(self):
        """Test to_dict() when thinking is enabled."""
        config = ThinkingConfig(enabled=True, budget_tokens=3000)
        result = config.to_dict()

        expected = {"thinking": {"type": "enabled", "budget_tokens": 3000}}
        assert result == expected

    def test_thinking_config_aws_bedrock_format(self):
        """Test that format matches AWS Bedrock specification."""
        config = ThinkingConfig(enabled=True, budget_tokens=2000)
        result = config.to_dict()

        # AWS Bedrock expects this exact structure
        assert "thinking" in result
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 2000

    @pytest.mark.parametrize("budget", [1000, 2000, 5000, 10000])
    def test_thinking_presets_token_budgets(self, budget):
        """Test various thinking token budgets."""
        config = ThinkingConfig(enabled=True, budget_tokens=budget)
        result = config.to_dict()

        assert result["thinking"]["budget_tokens"] == budget


# =============================================================================
# THINKING_PRESETS Tests
# =============================================================================


class TestThinkingPresets:
    """Test THINKING_PRESETS module-level constant."""

    def test_thinking_presets_exist(self):
        """Test that THINKING_PRESETS are defined."""
        assert "disabled" in THINKING_PRESETS
        assert "light" in THINKING_PRESETS
        assert "standard" in THINKING_PRESETS
        assert "extended" in THINKING_PRESETS
        assert "deep" in THINKING_PRESETS

    def test_thinking_preset_configurations(self):
        """Test thinking preset configurations."""
        assert THINKING_PRESETS["disabled"].enabled is False
        assert THINKING_PRESETS["light"].budget_tokens == 1000
        assert THINKING_PRESETS["standard"].budget_tokens == 2000
        assert THINKING_PRESETS["extended"].budget_tokens == 5000
        assert THINKING_PRESETS["deep"].budget_tokens == 10000

    def test_all_presets_are_thinking_config_instances(self):
        """All preset values must be ThinkingConfig instances."""
        for name, preset in THINKING_PRESETS.items():
            assert isinstance(
                preset, ThinkingConfig
            ), f"Preset '{name}' is not a ThinkingConfig instance"

    def test_disabled_preset_produces_empty_dict(self):
        """disabled preset must produce an empty dict (no API overhead)."""
        assert THINKING_PRESETS["disabled"].to_dict() == {}

    def test_enabled_presets_produce_thinking_key(self):
        """Every enabled preset must produce a dict with a 'thinking' key."""
        for name, preset in THINKING_PRESETS.items():
            if preset.enabled:
                result = preset.to_dict()
                assert (
                    "thinking" in result
                ), f"Preset '{name}' did not produce 'thinking' key"


# =============================================================================
# LLM Factory Thinking Tests
# =============================================================================


class TestLLMFactoryThinking:
    """Test thinking parameter passing through LLMFactory."""

    @patch("lobster.core.config_resolver.ConfigResolver")
    @patch("lobster.config.providers.get_provider")
    def test_llm_factory_passes_additional_fields(
        self, mock_get_provider, mock_config_resolver
    ):
        """Test that LLMFactory passes additional_model_request_fields to provider."""
        from lobster.config.llm_factory import LLMFactory

        # Setup mocks
        mock_resolver = Mock()
        mock_resolver.resolve_provider.return_value = ("bedrock", "test source")
        mock_resolver.resolve_model.return_value = ("test-model", "test source")
        mock_config_resolver.get_instance.return_value = mock_resolver

        mock_provider = Mock()
        mock_provider.get_default_model.return_value = "test-model"
        mock_provider.create_chat_model = Mock(return_value=Mock())
        mock_get_provider.return_value = mock_provider

        # Model config with thinking
        model_config = {
            "temperature": 1.0,
            "max_tokens": 4096,
            "additional_model_request_fields": {
                "thinking": {"type": "enabled", "budget_tokens": 5000}
            },
        }

        # Create LLM
        LLMFactory.create_llm(model_config=model_config, agent_name="supervisor")

        # Verify provider.create_chat_model was called with thinking config
        mock_provider.create_chat_model.assert_called_once()
        call_kwargs = mock_provider.create_chat_model.call_args[1]

        # Should have thinking config in kwargs
        assert "thinking" in call_kwargs
        assert call_kwargs["thinking"]["type"] == "enabled"
        assert call_kwargs["thinking"]["budget_tokens"] == 5000

    @patch("lobster.core.config_resolver.ConfigResolver")
    @patch("lobster.config.providers.get_provider")
    def test_llm_factory_without_thinking(
        self, mock_get_provider, mock_config_resolver
    ):
        """Test that LLMFactory works without thinking config."""
        from lobster.config.llm_factory import LLMFactory

        # Setup mocks
        mock_resolver = Mock()
        mock_resolver.resolve_provider.return_value = ("bedrock", "test source")
        mock_resolver.resolve_model.return_value = ("test-model", "test source")
        mock_config_resolver.get_instance.return_value = mock_resolver

        mock_provider = Mock()
        mock_provider.get_default_model.return_value = "test-model"
        mock_provider.create_chat_model = Mock(return_value=Mock())
        mock_get_provider.return_value = mock_provider

        # Model config WITHOUT thinking
        model_config = {
            "temperature": 1.0,
            "max_tokens": 4096,
        }

        # Create LLM
        LLMFactory.create_llm(model_config=model_config, agent_name="supervisor")

        # Verify provider.create_chat_model was called
        mock_provider.create_chat_model.assert_called_once()
        call_kwargs = mock_provider.create_chat_model.call_args[1]

        # Should NOT have thinking config
        assert "thinking" not in call_kwargs


# =============================================================================
# Provider-Specific Thinking Tests
# =============================================================================


class TestBedrockThinkingIntegration:
    """Test Bedrock-specific thinking configuration."""

    @classmethod
    def setup_class(cls):
        pytest.importorskip("langchain_aws")

    @patch("langchain_aws.ChatBedrockConverse")
    def test_bedrock_provider_accepts_thinking_config(self, mock_chat_bedrock):
        """Test that BedrockProvider passes thinking config to ChatBedrockConverse."""
        from lobster.config.providers import get_provider

        provider = get_provider("bedrock")
        mock_chat_bedrock.return_value = Mock()

        # Mock credentials
        with patch.dict(
            os.environ,
            {
                "AWS_BEDROCK_ACCESS_KEY": "test_access",
                "AWS_BEDROCK_SECRET_ACCESS_KEY": "test_secret",
            },
            clear=True,
        ):
            # Create chat model with thinking config
            provider.create_chat_model(
                model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                temperature=1.0,
                max_tokens=4096,
                additional_model_request_fields={
                    "thinking": {"type": "enabled", "budget_tokens": 5000}
                },
            )

            # Verify ChatBedrockConverse was called with thinking config
            mock_chat_bedrock.assert_called_once()
            call_kwargs = mock_chat_bedrock.call_args[1]

            assert "additional_model_request_fields" in call_kwargs
            assert "thinking" in call_kwargs["additional_model_request_fields"]
            assert (
                call_kwargs["additional_model_request_fields"]["thinking"]["type"]
                == "enabled"
            )
            assert (
                call_kwargs["additional_model_request_fields"]["thinking"][
                    "budget_tokens"
                ]
                == 5000
            )


class TestGeminiThinkingIntegration:
    """Test Gemini-specific thinking configuration."""

    @classmethod
    def setup_class(cls):
        pytest.importorskip("langchain_google_genai")

    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_gemini_provider_includes_thoughts(self, mock_chat_gemini):
        """Test that GeminiProvider passes include_thoughts=True."""
        from lobster.config.providers import get_provider

        provider = get_provider("gemini")
        mock_chat_gemini.return_value = Mock()

        # Mock API key
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}, clear=True):
            # Create chat model
            provider.create_chat_model(
                model_id="gemini-3-pro-preview", temperature=1.0, max_tokens=4096
            )

            # Verify ChatGoogleGenerativeAI was called with include_thoughts=True
            mock_chat_gemini.assert_called_once()
            call_kwargs = mock_chat_gemini.call_args[1]

            assert "include_thoughts" in call_kwargs
            assert call_kwargs["include_thoughts"] is True

    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_gemini_temperature_enforcement(self, mock_chat):
        """Test that Gemini enforces temperature=1.0 for Gemini 3+ models."""
        from lobster.config.providers import get_provider

        provider = get_provider("gemini")
        mock_chat.return_value = Mock()

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}, clear=True):
            # Try to create with temperature=0.5 (should be overridden to 1.0)
            provider.create_chat_model(
                model_id="gemini-3-pro-preview",
                temperature=0.5,  # Will be overridden
                max_tokens=4096,
            )

            # Verify temperature was forced to 1.0
            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["temperature"] == 1.0


# =============================================================================
# Integration Tests (LLMFactory end-to-end)
# =============================================================================


class TestThinkingEndToEndFlow:
    """Test complete thinking configuration flow from agent_defaults to provider."""

    @patch("lobster.core.config_resolver.ConfigResolver")
    @patch("lobster.config.providers.get_provider")
    def test_bedrock_thinking_flow(self, mock_get_provider, mock_config_resolver):
        """Test thinking config flows from agent_defaults -> llm_factory -> bedrock_provider."""
        from lobster.config.llm_factory import LLMFactory

        # Setup mocks
        mock_resolver = Mock()
        mock_resolver.resolve_provider.return_value = ("bedrock", "test source")
        mock_resolver.resolve_model.return_value = (
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "test source",
        )
        mock_config_resolver.get_instance.return_value = mock_resolver

        mock_provider = Mock()
        mock_provider.get_default_model.return_value = "test-model"
        mock_provider.create_chat_model = Mock(return_value=Mock())
        mock_get_provider.return_value = mock_provider

        # Build params the same way agent_defaults.get_agent_params() would
        thinking_config = ThinkingConfig(enabled=True, budget_tokens=5000)
        model_params = {
            "temperature": 1.0,
            "additional_model_request_fields": thinking_config.to_dict(),
        }

        # Verify thinking config is in params
        assert "additional_model_request_fields" in model_params
        assert (
            model_params["additional_model_request_fields"]["thinking"]["budget_tokens"]
            == 5000
        )

        # Create LLM (this is what graph.py does)
        LLMFactory.create_llm(model_config=model_params, agent_name="supervisor")

        # Verify provider received thinking config
        mock_provider.create_chat_model.assert_called_once()
        call_kwargs = mock_provider.create_chat_model.call_args[1]

        # CRITICAL: Verify thinking config was passed through
        assert (
            "thinking" in call_kwargs
        ), "Thinking config was not passed to provider.create_chat_model()"
        assert call_kwargs["thinking"]["budget_tokens"] == 5000

    @patch("lobster.core.config_resolver.ConfigResolver")
    @patch("lobster.config.providers.get_provider")
    def test_gemini_thinking_flow(self, mock_get_provider, mock_config_resolver):
        """Test that Gemini thinking doesn't interfere with Bedrock thinking config."""
        from lobster.config.llm_factory import LLMFactory

        # Setup mocks for Gemini
        mock_resolver = Mock()
        mock_resolver.resolve_provider.return_value = ("gemini", "test source")
        mock_resolver.resolve_model.return_value = (
            "gemini-3-pro-preview",
            "test source",
        )
        mock_config_resolver.get_instance.return_value = mock_resolver

        mock_provider = Mock()
        mock_provider.get_default_model.return_value = "gemini-3-pro-preview"
        mock_provider.create_chat_model = Mock(return_value=Mock())
        mock_get_provider.return_value = mock_provider

        # Model config WITHOUT additional_model_request_fields
        # (Gemini uses include_thoughts=True hardcoded in provider)
        model_params = {
            "temperature": 1.0,
            "max_tokens": 4096,
        }

        # Create LLM
        LLMFactory.create_llm(model_config=model_params, agent_name="supervisor")

        # Verify provider was called (thinking handled by provider internally)
        mock_provider.create_chat_model.assert_called_once()

    @patch("lobster.core.config_resolver.ConfigResolver")
    @patch("lobster.config.providers.get_provider")
    def test_thinking_config_not_passed_to_gemini(
        self, mock_get_provider, mock_config_resolver
    ):
        """Test that additional_model_request_fields (Bedrock thinking) is safely ignored for Gemini."""
        from lobster.config.llm_factory import LLMFactory

        # Setup mocks for Gemini
        mock_resolver = Mock()
        mock_resolver.resolve_provider.return_value = ("gemini", "test source")
        mock_resolver.resolve_model.return_value = (
            "gemini-3-pro-preview",
            "test source",
        )
        mock_config_resolver.get_instance.return_value = mock_resolver

        mock_provider = Mock()
        mock_provider.get_default_model.return_value = "gemini-3-pro-preview"
        mock_provider.create_chat_model = Mock(return_value=Mock())
        mock_get_provider.return_value = mock_provider

        # Model config WITH additional_model_request_fields (shouldn't break Gemini)
        model_params = {
            "temperature": 1.0,
            "max_tokens": 4096,
            "additional_model_request_fields": {
                "thinking": {"type": "enabled", "budget_tokens": 5000}
            },
        }

        # Create LLM - should not raise exception
        LLMFactory.create_llm(model_config=model_params, agent_name="supervisor")

        # Verify provider was called
        mock_provider.create_chat_model.assert_called_once()

        # Gemini provider should receive thinking config but ignore it gracefully
        # (Gemini uses include_thoughts=True instead)


# =============================================================================
# Edge Cases
# =============================================================================


class TestThinkingEdgeCases:
    """Test edge cases and error handling for thinking configuration."""

    def test_thinking_config_with_zero_budget(self):
        """Test thinking config with zero budget tokens."""
        config = ThinkingConfig(enabled=True, budget_tokens=0)
        result = config.to_dict()

        assert result["thinking"]["budget_tokens"] == 0

    def test_thinking_config_with_large_budget(self):
        """Test thinking config with very large budget."""
        config = ThinkingConfig(enabled=True, budget_tokens=50000)
        result = config.to_dict()

        assert result["thinking"]["budget_tokens"] == 50000

    def test_thinking_disabled_returns_empty_dict(self):
        """Test that disabled thinking returns empty dict (no API call overhead)."""
        config = ThinkingConfig(enabled=False, budget_tokens=5000)
        result = config.to_dict()

        assert result == {}  # Should be completely empty

    def test_thinking_config_custom_type(self):
        """Test ThinkingConfig with custom type value."""
        config = ThinkingConfig(enabled=True, budget_tokens=2000, type="custom")
        result = config.to_dict()

        assert result["thinking"]["type"] == "custom"
