"""Bedrock-family providers send the model's output bound unless one is configured.

LLMFactory used to inject ``max_tokens=4096`` for every provider and the Bedrock
providers forwarded it, so each Bedrock request carried ``maxTokens=4096``. That
cap is far below the model maximum and below extended-thinking budgets
(5000/10000 tokens). Sending nothing did not help: Bedrock then applied its own
small default and cut long answers off. The default is now the model's documented
output bound (``lobster.config.output_bound``); an explicit value must still reach
the request unchanged. The bound itself is tested in ``test_output_bound.py``.
"""

import os
import sys
import types
from unittest.mock import MagicMock, Mock, patch

import pytest

# These tests exercise the real provider/factory wiring, so opt out of the
# unit-test auto-mocking of ConfigResolver and LLMFactory.
pytestmark = pytest.mark.no_auto_config

MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
BEDROCK_FAMILY = ["bedrock", "omics-os"]


@pytest.fixture(autouse=True)
def bedrock_environment(monkeypatch):
    """Fake credentials for both providers, with no ambient AWS configuration."""
    for name in list(os.environ):
        if name.startswith(("AWS_", "OMICS_OS_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("AWS_BEDROCK_ACCESS_KEY", "fake-access-key")
    monkeypatch.setenv("AWS_BEDROCK_SECRET_ACCESS_KEY", "fake-secret-key")
    monkeypatch.setenv("OMICS_OS_API_KEY", "omk_test123")


@pytest.fixture
def constructor(monkeypatch):
    """Stand-in ChatBedrockConverse, so these tests need no optional langchain-aws."""
    module = types.ModuleType("langchain_aws")
    module.ChatBedrockConverse = MagicMock(name="ChatBedrockConverse")
    monkeypatch.setitem(sys.modules, "langchain_aws", module)
    return module.ChatBedrockConverse


def _provider(name):
    from lobster.config.providers import get_provider

    return get_provider(name)


def _create_llm(model_config, provider_name):
    """Run the real LLMFactory and provider with a stubbed ConfigResolver.

    The resolver is replaced on the module so provider selection cannot be affected
    by ConfigResolver state left behind by other tests in the same process.
    """
    from lobster.config.llm_factory import LLMFactory

    resolver = Mock()
    resolver.resolve_provider.return_value = (provider_name, "test")
    with patch("lobster.core.config_resolver.ConfigResolver") as resolver_cls:
        resolver_cls.get_instance.return_value = resolver
        return LLMFactory.create_llm(model_config, agent_name="test_agent")


@pytest.mark.parametrize("provider_name", BEDROCK_FAMILY)
def test_provider_default_sends_the_model_bound(provider_name, constructor):
    _provider(provider_name).create_chat_model(MODEL)
    assert constructor.call_args.kwargs["max_tokens"] == 64000


@pytest.mark.parametrize("provider_name", BEDROCK_FAMILY)
@pytest.mark.parametrize("max_tokens", [1, 4096, 8192])
def test_provider_forwards_explicit_output_limit(
    provider_name, max_tokens, constructor
):
    _provider(provider_name).create_chat_model(MODEL, max_tokens=max_tokens)
    assert constructor.call_args.kwargs["max_tokens"] == max_tokens


@pytest.mark.parametrize("provider_name", BEDROCK_FAMILY)
def test_factory_default_sends_the_model_bound(provider_name, constructor):
    config = {"model_id": MODEL, "temperature": 0.5}
    _create_llm(config, provider_name)
    kwargs = constructor.call_args.kwargs
    assert kwargs["max_tokens"] == 64000
    assert kwargs["model_id"] == MODEL
    assert kwargs["temperature"] == 0.5


@pytest.mark.parametrize("provider_name", BEDROCK_FAMILY)
@pytest.mark.parametrize("max_tokens", [2048, 4096, 16384])
def test_factory_forwards_configured_output_limit(
    provider_name, max_tokens, constructor
):
    config = {"model_id": MODEL, "max_tokens": max_tokens}
    _create_llm(config, provider_name)
    assert constructor.call_args.kwargs["max_tokens"] == max_tokens


def test_extended_thinking_budget_is_not_paired_with_a_smaller_default_cap(constructor):
    """A 5000-token thinking budget needs a larger output limit."""
    from lobster.config.agent_defaults import THINKING_PRESETS

    thinking = THINKING_PRESETS["extended"].to_dict()["thinking"]
    assert thinking["budget_tokens"] > 4096
    config = {
        "model_id": MODEL,
        "additional_model_request_fields": {"thinking": thinking},
    }
    _create_llm(config, "bedrock")
    kwargs = constructor.call_args.kwargs
    assert kwargs["additional_model_request_fields"] == {"thinking": thinking}
    assert kwargs["max_tokens"] > thinking["budget_tokens"]


def test_gateway_model_carries_the_model_bound_by_default():
    """The real langchain-aws model sends the documented bound unless one is set."""
    pytest.importorskip("langchain_aws")
    from langchain_aws import ChatBedrockConverse

    provider = _provider("omics-os")
    default = provider.create_chat_model(MODEL)
    explicit = provider.create_chat_model(MODEL, max_tokens=2048)
    assert isinstance(default, ChatBedrockConverse)
    assert default.max_tokens == 64000
    assert explicit.max_tokens == 2048


@patch("lobster.core.config_resolver.ConfigResolver")
@patch("lobster.config.providers.get_provider")
def test_factory_forwards_max_tokens_only_when_configured(
    mock_get_provider, mock_config_resolver
):
    from lobster.config.llm_factory import LLMFactory

    resolver = Mock()
    resolver.resolve_provider.return_value = ("bedrock", "test source")
    resolver.resolve_model.return_value = ("test-model", "test source")
    mock_config_resolver.get_instance.return_value = resolver
    provider = Mock()
    provider.get_default_model.return_value = "test-model"
    mock_get_provider.return_value = provider

    LLMFactory.create_llm({"temperature": 0.3}, agent_name="supervisor")
    assert "max_tokens" not in provider.create_chat_model.call_args.kwargs
    assert provider.create_chat_model.call_args.kwargs["temperature"] == 0.3

    LLMFactory.create_llm({"max_tokens": 8192}, agent_name="supervisor")
    assert provider.create_chat_model.call_args.kwargs["max_tokens"] == 8192


def test_providers_with_their_own_default_keep_it(monkeypatch):
    """Other providers still apply their documented 4096 default."""
    pytest.importorskip("langchain_anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test123")
    config = {"model_id": "claude-sonnet-4-20250514"}
    with patch("langchain_anthropic.ChatAnthropic") as constructor:
        _create_llm(config, "anthropic")
    assert constructor.call_args.kwargs["max_tokens"] == 4096
