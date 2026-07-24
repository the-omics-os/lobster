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
    """Provider is registered and present in VALID_PROVIDERS."""
    assert ProviderRegistry.get("nebius") is not None
    assert "nebius" in VALID_PROVIDERS
