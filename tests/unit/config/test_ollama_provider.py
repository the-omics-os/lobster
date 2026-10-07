"""
Unit tests for the Ollama provider's Ollama Cloud auth plumbing.

These tests verify that an ``OLLAMA_API_KEY`` is attached as a bearer token to
all five network call sites (four direct ``requests`` calls plus ``ChatOllama``
via ``client_kwargs``), and that local mode is unchanged when no key is set.

No network is used: ``requests`` and ``ChatOllama`` are mocked.
"""

import importlib.util
import os
from unittest.mock import MagicMock, patch

import pytest

from lobster.config.providers.ollama_provider import OllamaProvider


def _mock_response(status=200, payload=None):
    """Build a MagicMock requests-like response."""
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload or {}
    return r


# --------------------------------------------------------------------------- #
# Direct HTTP call sites: bearer header sent when key set, absent otherwise.
# --------------------------------------------------------------------------- #


@patch("requests.get")
def test_auth_header_sent_on_is_available(mock_get):
    """is_available() sends Authorization: Bearer when OLLAMA_API_KEY is set."""
    mock_get.return_value = _mock_response(200, {"models": []})
    with patch.dict(os.environ, {"OLLAMA_API_KEY": "secret-key"}, clear=False):
        OllamaProvider().is_available()
    assert mock_get.call_args.kwargs["headers"] == {
        "Authorization": "Bearer secret-key"
    }


@patch("requests.get")
def test_no_auth_header_on_is_available_without_key(mock_get):
    """Local mode: is_available() sends no Authorization header (headers=None)."""
    mock_get.return_value = _mock_response(200, {"models": []})
    with patch.dict(os.environ, {}, clear=True):
        OllamaProvider().is_available()
    assert mock_get.call_args.kwargs["headers"] is None


@patch("requests.get")
def test_auth_header_sent_on_fetch_models(mock_get):
    """_fetch_models() sends Authorization: Bearer when key set."""
    mock_get.return_value = _mock_response(200, {"models": []})
    with patch.dict(os.environ, {"OLLAMA_API_KEY": "secret-key"}, clear=False):
        OllamaProvider()._fetch_models()
    assert mock_get.call_args.kwargs["headers"] == {
        "Authorization": "Bearer secret-key"
    }


@patch("requests.get")
def test_no_auth_header_on_fetch_models_without_key(mock_get):
    """Local mode: _fetch_models() sends no Authorization header."""
    mock_get.return_value = _mock_response(200, {"models": []})
    with patch.dict(os.environ, {}, clear=True):
        OllamaProvider()._fetch_models()
    assert mock_get.call_args.kwargs["headers"] is None


@patch("requests.post")
def test_auth_header_sent_on_preload(mock_post):
    """preload_model() sends Authorization: Bearer when key set."""
    mock_post.return_value = _mock_response(200)
    with patch.dict(os.environ, {"OLLAMA_API_KEY": "secret-key"}, clear=False):
        OllamaProvider().preload_model("llama3:8b")
    assert mock_post.call_args.kwargs["headers"] == {
        "Authorization": "Bearer secret-key"
    }


@patch("requests.post")
def test_no_auth_header_on_preload_without_key(mock_post):
    """Local mode: preload_model() sends no Authorization header."""
    mock_post.return_value = _mock_response(200)
    with patch.dict(os.environ, {}, clear=True):
        OllamaProvider().preload_model("llama3:8b")
    assert mock_post.call_args.kwargs["headers"] is None


@patch("requests.post")
def test_auth_header_sent_on_show(mock_post):
    """_get_model_context_length() sends Authorization: Bearer when key set."""
    mock_post.return_value = _mock_response(200, {"model_info": {}})
    with patch.dict(os.environ, {"OLLAMA_API_KEY": "secret-key"}, clear=False):
        OllamaProvider()._get_model_context_length("llama3:8b")
    assert mock_post.call_args.kwargs["headers"] == {
        "Authorization": "Bearer secret-key"
    }


@patch("requests.post")
def test_no_auth_header_on_show_without_key(mock_post):
    """Local mode: _get_model_context_length() sends no Authorization header."""
    mock_post.return_value = _mock_response(200, {"model_info": {}})
    with patch.dict(os.environ, {}, clear=True):
        OllamaProvider()._get_model_context_length("llama3:8b")
    assert mock_post.call_args.kwargs["headers"] is None


# --------------------------------------------------------------------------- #
# ChatOllama call site: auth via client_kwargs (only when langchain-ollama).
# --------------------------------------------------------------------------- #

_SKIP_NO_LLM = pytest.mark.skipif(
    not importlib.util.find_spec("langchain_ollama"),
    reason="langchain-ollama not installed",
)


@_SKIP_NO_LLM
@patch("langchain_ollama.ChatOllama")
def test_create_chat_model_passes_bearer_via_client_kwargs(mock_chat):
    """create_chat_model() puts the bearer header in client_kwargs when key set.

    ChatOllama has no api_key field; auth flows through client_kwargs -> the
    underlying ollama Client. OLLAMA_NUM_CTX is set to avoid a /api/show call.
    """
    mock_chat.return_value = MagicMock()
    with patch.dict(
        os.environ,
        {"OLLAMA_API_KEY": "secret-key", "OLLAMA_NUM_CTX": "8192"},
        clear=False,
    ):
        OllamaProvider().create_chat_model("llama3:8b")
    kwargs = mock_chat.call_args.kwargs
    assert (
        kwargs["client_kwargs"]["headers"]["Authorization"]
        == "Bearer secret-key"
    )


@_SKIP_NO_LLM
@patch("langchain_ollama.ChatOllama")
def test_create_chat_model_no_client_kwargs_without_key(mock_chat):
    """Local mode: create_chat_model() adds no client_kwargs when no key set."""
    mock_chat.return_value = MagicMock()
    with patch.dict(os.environ, {"OLLAMA_NUM_CTX": "8192"}, clear=True):
        OllamaProvider().create_chat_model("llama3:8b")
    kwargs = mock_chat.call_args.kwargs
    assert "client_kwargs" not in kwargs


@_SKIP_NO_LLM
@patch("langchain_ollama.ChatOllama")
def test_explicit_api_key_kwarg_beats_env(mock_chat):
    """An explicit api_key kwarg takes precedence over OLLAMA_API_KEY."""
    mock_chat.return_value = MagicMock()
    with patch.dict(
        os.environ,
        {"OLLAMA_API_KEY": "env-key", "OLLAMA_NUM_CTX": "8192"},
        clear=False,
    ):
        OllamaProvider().create_chat_model("llama3:8b", api_key="explicit-key")
    headers = mock_chat.call_args.kwargs["client_kwargs"]["headers"]
    assert headers["Authorization"] == "Bearer explicit-key"