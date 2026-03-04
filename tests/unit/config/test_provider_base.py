"""Tests for the default verify_connection() implementation in ILLMProvider."""
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

from lobster.config.providers.base_provider import ILLMProvider, ModelInfo


class _MinimalProvider(ILLMProvider):
    """Minimal concrete ILLMProvider for testing base-class behaviour."""

    @property
    def name(self) -> str:
        return "minimal"

    @property
    def display_name(self) -> str:
        return "Minimal Test Provider"

    def is_configured(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def list_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(
                name="minimal/model-v1",
                display_name="Minimal Model v1",
                description="Test model",
                provider="minimal",
            )
        ]

    def get_default_model(self) -> str:
        return "minimal/model-v1"

    def create_chat_model(
        self,
        model_id: str,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError("override in tests")

    def validate_model(self, model_id: str) -> bool:
        return model_id in self.get_model_names()


@pytest.fixture
def provider():
    return _MinimalProvider()


class TestDefaultVerifyConnection:
    def test_returns_true_on_successful_inference(self, provider):
        """Default implementation returns True when LLM.invoke() succeeds."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock()

        provider.create_chat_model = MagicMock(return_value=mock_llm)

        ok, msg = provider.verify_connection()

        assert ok is True
        assert "Minimal Test Provider" in msg
        assert "connection verified" in msg
        assert "minimal/model-v1" in msg

    def test_uses_provided_model_id(self, provider):
        """When model_id is passed, that model is used instead of the default."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock()

        provider.create_chat_model = MagicMock(return_value=mock_llm)

        ok, msg = provider.verify_connection("custom/model-xyz")

        assert ok is True
        assert "custom/model-xyz" in msg
        provider.create_chat_model.assert_called_once_with(
            "custom/model-xyz", temperature=0.0, max_tokens=1
        )

    def test_returns_false_on_exception(self, provider):
        """Default implementation returns False when create_chat_model raises."""
        provider.create_chat_model = MagicMock(
            side_effect=ValueError("API key invalid")
        )

        ok, msg = provider.verify_connection()

        assert ok is False
        assert "Minimal Test Provider" in msg
        assert "connection failed" in msg

    def test_returns_false_when_invoke_raises(self, provider):
        """Default implementation returns False when llm.invoke() raises."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("Rate limit exceeded")

        provider.create_chat_model = MagicMock(return_value=mock_llm)

        ok, msg = provider.verify_connection()

        assert ok is False
        assert "connection failed" in msg

    def test_uses_max_tokens_one(self, provider):
        """The inference call uses max_tokens=1 to minimise cost."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock()

        provider.create_chat_model = MagicMock(return_value=mock_llm)
        provider.verify_connection()

        _, call_kwargs = provider.create_chat_model.call_args
        assert call_kwargs.get("max_tokens") == 1

    def test_calls_invoke_with_simple_prompt(self, provider):
        """The inference call uses a simple 'Hi' prompt."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock()

        provider.create_chat_model = MagicMock(return_value=mock_llm)
        provider.verify_connection()

        mock_llm.invoke.assert_called_once_with("Hi")
