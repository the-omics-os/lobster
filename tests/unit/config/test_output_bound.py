"""Every Bedrock-family request carries an explicit output-token bound.

Without ``maxTokens`` Bedrock applies its own default, which is far below the
model's real maximum: a long answer ended with ``stopReason == "max_tokens"`` in
the middle of a table row. The engine now sends a bound on every call, from the
model's documented output maximum, and says so in the log when a reply is still
cut off. An explicit ``max_tokens`` (config, argument) and the
``LOBSTER_BEDROCK_MAX_TOKENS`` setting both win over the default.
"""

import logging
import os
import sys
import types
from unittest.mock import MagicMock, Mock, patch

import pytest

from lobster.config.output_bound import (
    OUTPUT_BOUND_ENV,
    OutputLimitWatch,
    default_output_bound,
    truncation_note,
)
from lobster.config.providers.bedrock_builder import build_bedrock_converse

pytestmark = pytest.mark.no_auto_config

SONNET55 = "global.anthropic.claude-sonnet-5-5"
HAIKU55 = "global.anthropic.claude-haiku-5-5"
OPUS55 = "us.anthropic.claude-opus-5-5"
SONNET45 = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
UNKNOWN = "global.example.some-new-model"

# Supervisor, supervisor-accessible agents and children (parent delegation only).
AGENTS = [
    "supervisor",
    "research_agent",
    "data_expert_agent",
    "transcriptomics_expert",
    "annotation_expert",
    "de_analysis_expert",
    "proteomics_expert",
    "proteomics_de_analysis_expert",
    "biomarker_discovery_expert",
    "genomics_expert",
    "variant_analysis_expert",
    "machine_learning_expert",
    "feature_selection_expert",
    "survival_analysis_expert",
    "metabolomics_expert",
    "drug_discovery_expert",
    "cheminformatics_expert",
    "clinical_dev_expert",
    "pharmacogenomics_expert",
    "visualization_expert_agent",
    "metadata_assistant",
    "protein_structure_visualization_expert",
]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("AWS_", "OMICS_OS_")) or name == OUTPUT_BOUND_ENV:
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


def _client_kwargs(mode):
    if mode == "gateway":
        return {"client": object(), "bedrock_client": object()}
    return {
        "region_name": "us-east-1",
        "aws_access_key_id": "fake",
        "aws_secret_access_key": "fake",
    }


# --- the default bound, by model ---------------------------------------------


@pytest.mark.parametrize("model_id", [SONNET55, HAIKU55, OPUS55])
def test_claude_55_models_use_the_catalog_ceiling(model_id):
    assert default_output_bound(model_id) == 64000


@pytest.mark.parametrize(
    "model_id",
    [
        SONNET45,
        "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "global.anthropic.claude-opus-4-5-20251101-v1:0",
    ],
)
def test_claude_45_models_use_their_documented_maximum(model_id):
    assert default_output_bound(model_id) == 64000


def test_unknown_model_gets_a_safe_bound_not_none():
    assert default_output_bound(UNKNOWN) == 8192


def test_setting_overrides_the_table(monkeypatch):
    monkeypatch.setenv(OUTPUT_BOUND_ENV, "12000")
    assert default_output_bound(SONNET55) == 12000


@pytest.mark.parametrize("value", ["", "0", "-5", "lots", "1.5"])
def test_invalid_setting_is_ignored(monkeypatch, value):
    monkeypatch.setenv(OUTPUT_BOUND_ENV, value)
    assert default_output_bound(SONNET55) == 64000


# --- the builder sends the bound ---------------------------------------------


@pytest.mark.parametrize("mode", ["gateway", "direct"])
def test_builder_sends_the_default_bound(mode, constructor):
    build_bedrock_converse(model_id=SONNET55, **_client_kwargs(mode))
    assert constructor.call_args.kwargs["max_tokens"] == 64000


@pytest.mark.parametrize("mode", ["gateway", "direct"])
def test_builder_treats_none_as_unset(mode, constructor):
    build_bedrock_converse(model_id=SONNET55, max_tokens=None, **_client_kwargs(mode))
    assert constructor.call_args.kwargs["max_tokens"] == 64000


@pytest.mark.parametrize("max_tokens", [1, 4096, 8192, 100000])
def test_explicit_max_tokens_still_wins(max_tokens, monkeypatch, constructor):
    monkeypatch.setenv(OUTPUT_BOUND_ENV, "12000")
    build_bedrock_converse(
        model_id=SONNET55, max_tokens=max_tokens, **_client_kwargs("direct")
    )
    assert constructor.call_args.kwargs["max_tokens"] == max_tokens


def test_setting_reaches_the_request(monkeypatch, constructor):
    monkeypatch.setenv(OUTPUT_BOUND_ENV, "12000")
    build_bedrock_converse(model_id=SONNET55, **_client_kwargs("direct"))
    assert constructor.call_args.kwargs["max_tokens"] == 12000


def test_bound_stays_above_a_thinking_budget(constructor):
    fields = {"thinking": {"type": "enabled", "budget_tokens": 10000}}
    build_bedrock_converse(
        model_id=UNKNOWN,
        additional_model_request_fields=fields,
        **_client_kwargs("direct"),
    )
    assert constructor.call_args.kwargs["max_tokens"] > 10000


@pytest.mark.parametrize("provider_name", ["bedrock", "omics-os"])
def test_providers_send_the_bound(provider_name, constructor):
    from lobster.config.providers import get_provider

    get_provider(provider_name).create_chat_model(SONNET55)
    assert constructor.call_args.kwargs["max_tokens"] == 64000


# --- every agent, children included, through the real factory ----------------


def _create_llm(agent_name, provider_name, model_config=None):
    from lobster.config.llm_factory import LLMFactory

    resolver = Mock()
    resolver.resolve_provider.return_value = (provider_name, "test")
    resolver.resolve_model.return_value = (SONNET55, "test")
    with patch("lobster.core.config_resolver.ConfigResolver") as resolver_cls:
        resolver_cls.get_instance.return_value = resolver
        return LLMFactory.create_llm(model_config or {}, agent_name=agent_name)


@pytest.mark.parametrize("provider_name", ["bedrock", "omics-os"])
@pytest.mark.parametrize("agent_name", AGENTS)
def test_every_agent_call_carries_max_tokens(agent_name, provider_name, constructor):
    _create_llm(agent_name, provider_name)
    assert constructor.call_args.kwargs["max_tokens"] == 64000


def test_agent_config_max_tokens_still_wins_through_the_factory(constructor):
    _create_llm("annotation_expert", "bedrock", {"max_tokens": 2048})
    assert constructor.call_args.kwargs["max_tokens"] == 2048


def test_real_model_carries_the_bound():
    pytest.importorskip("langchain_aws")
    from lobster.config.providers import get_provider

    model = get_provider("omics-os").create_chat_model(SONNET55)
    assert model.max_tokens == 64000


def test_real_model_takes_the_watch():
    pytest.importorskip("langchain_aws")
    from lobster.config.output_bound import attach_output_limit_watch
    from lobster.config.providers import get_provider

    model = get_provider("omics-os").create_chat_model(SONNET55)
    attach_output_limit_watch(model, "supervisor")
    (watch,) = [c for c in model.callbacks if isinstance(c, OutputLimitWatch)]
    assert (watch.agent_name, watch.bound) == ("supervisor", 64000)


# --- a reply that is still cut off is never silent ---------------------------


def _result(stop_key, stop):
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    message = AIMessage(content="| a | b |\n| 1 |", response_metadata={stop_key: stop})
    return LLMResult(generations=[[ChatGeneration(message=message)]])


@pytest.mark.parametrize(
    "stop_key,stop",
    [("stopReason", "max_tokens"), ("stop_reason", "max_tokens")],
)
def test_watch_warns_with_agent_and_bound(stop_key, stop, caplog):
    watch = OutputLimitWatch(agent_name="annotation_expert", bound=64000)
    with caplog.at_level(logging.WARNING, logger="lobster.config.output_bound"):
        watch.on_llm_end(_result(stop_key, stop))
    records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(records) == 1
    text = records[0].getMessage()
    assert "annotation_expert" in text and "64000" in text


@pytest.mark.parametrize("stop", ["end_turn", "tool_use", "stop_sequence"])
def test_watch_is_quiet_for_normal_stops(stop, caplog):
    watch = OutputLimitWatch(agent_name="supervisor", bound=64000)
    with caplog.at_level(logging.WARNING, logger="lobster.config.output_bound"):
        watch.on_llm_end(_result("stopReason", stop))
    assert not caplog.records


def test_factory_attaches_the_watch_with_the_agent_name(constructor):
    constructor.return_value.callbacks = None
    constructor.return_value.max_tokens = 64000
    model = _create_llm("de_analysis_expert", "bedrock")
    watches = [c for c in model.callbacks if isinstance(c, OutputLimitWatch)]
    assert len(watches) == 1
    assert watches[0].agent_name == "de_analysis_expert"
    assert watches[0].bound == 64000


def test_truncation_note_is_plain():
    note = truncation_note()
    assert "cut off" in note and "length limit" in note
