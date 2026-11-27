"""Offline synthetic callback probes; real live L1 replay is the separate T3 leaf."""

import json
import logging
from copy import deepcopy
from decimal import Decimal
from unittest.mock import patch
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from lobster.pricing import get_rates
from lobster.utils.callbacks import TokenTrackingCallback
from tests.unit.config.test_bedrock_prompt_cache import gateway  # noqa: F401

pytestmark = pytest.mark.no_auto_config

MODEL = "global.anthropic.claude-sonnet-5-5"
RUN = UUID("00000000-0000-0000-0000-000000000001")


def tracker():
    rates = get_rates("bedrock", MODEL)
    return TokenTrackingCallback(
        "synthetic",
        {
            MODEL: {
                "input_per_million": float(rates.input),
                "output_per_million": float(rates.output),
            }
        },
    )


def result(read=7, write=3, logical=20, ttl=True):
    details = {"cache_read": read, "cache_creation": write}
    if ttl:
        details["ephemeral_5m_input_tokens"] = write
    return LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="synthetic",
                        response_metadata={"model_name": MODEL},
                        usage_metadata={
                            "input_tokens": logical,
                            "output_tokens": 5,
                            "total_tokens": logical + 5,
                            "input_token_details": details,
                        },
                    )
                )
            ]
        ]
    )


def record(callback, response, provider="amazon_bedrock"):
    callback.on_chat_model_start(
        {},
        [[]],
        run_id=RUN,
        metadata=(
            {}
            if provider is None
            else {"ls_provider": provider, "ls_model_name": MODEL}
        ),
    )
    callback.on_llm_end(response, run_id=RUN)
    return callback.get_usage_summary()


def legacy_amount(logical=20):
    rates = get_rates("bedrock", MODEL)
    return (
        logical * float(rates.input) / 1_000_000 + 5 * float(rates.output) / 1_000_000
    )


def test_actual_ledger_uses_cache_rates_once_and_json_float():
    callback = tracker()
    summary = record(callback, result())
    rates = get_rates("bedrock", MODEL)
    expected = float(
        (
            10 * rates.input
            + 5 * rates.output
            + 7 * rates.cache_read
            + 3 * rates.cache_write
        )
        / Decimal(1_000_000)
    )
    invocation = callback.invocations[0]
    assert invocation.usage_complete
    assert invocation.cost_usd == expected
    assert callback.total_cost_usd == expected
    assert callback.by_agent[invocation.agent]["cost_usd"] == expected
    assert summary["invocations"][0]["cache_cost_status"] == "catalog"
    assert isinstance(summary["total_cost_usd"], float)
    assert isinstance(invocation.cost_usd, float)
    json.dumps(summary)


@pytest.mark.parametrize("provider", [None, "unknown", "omics-os"])
def test_missing_unknown_identity_uses_exact_legacy_and_one_content_free_log(
    caplog, provider
):
    callback = tracker()
    with caplog.at_level(logging.DEBUG, logger="lobster.utils.callbacks"):
        summary = record(callback, result(), provider)
    assert callback.total_cost_usd == legacy_amount()
    assert summary["invocations"][0]["cache_cost_status"] == "legacy_fallback"
    assert summary["invocations"][0]["cache_cost_reason"] == "pricing_identity_unknown"
    lines = [
        line
        for line in caplog.records
        if line.message.startswith("Cache cost fallback")
    ]
    assert len(lines) == 1
    assert len(caplog.records) == 1
    assert "synthetic" not in lines[0].message
    assert MODEL not in lines[0].message
    assert "20" not in lines[0].message
    assert "7" not in lines[0].message
    json.dumps(summary)


def test_anomaly_uses_legacy_not_negative_and_one_log(caplog):
    callback = tracker()
    with caplog.at_level(logging.WARNING, logger="lobster.utils.callbacks"):
        summary = record(callback, result(logical=9))
    assert callback.total_cost_usd == legacy_amount(9)
    assert not summary["invocations"][0]["usage_complete"]
    assert summary["invocations"][0]["cache_cost_status"] == "legacy_fallback"
    assert len(caplog.records) == 1
    assert "9" not in caplog.records[0].message
    assert "synthetic" not in caplog.records[0].message
    assert (
        len([r for r in caplog.records if r.message.startswith("Cache cost fallback")])
        == 1
    )


def test_positive_write_without_ttl_uses_legacy_and_flag():
    callback = tracker()
    summary = record(callback, result(ttl=False))
    assert callback.total_cost_usd == legacy_amount()
    assert summary["invocations"][0]["cache_cost_reason"] == "write_ttl_unreported"


def test_one_hour_write_uses_exact_legacy_and_one_flag(caplog):
    response = result()
    details = response.generations[0][0].message.usage_metadata["input_token_details"]
    details.pop("ephemeral_5m_input_tokens")
    details["ephemeral_1h_input_tokens"] = 3
    callback = tracker()
    with caplog.at_level(logging.WARNING, logger="lobster.utils.callbacks"):
        summary = record(callback, response)
    assert callback.invocations[0].cost_usd == legacy_amount()
    assert callback.total_cost_usd == legacy_amount()
    assert callback.unpriced_invocations == 0
    assert summary["cost_complete"]
    assert summary["invocations"][0]["cache_cost_reason"] == "write_ttl_unsupported"
    assert summary["invocations"][0]["cache_cost_status"] == "legacy_fallback"
    assert len(caplog.records) == 1
    assert (
        len([r for r in caplog.records if r.message.startswith("Cache cost fallback")])
        == 1
    )
    assert isinstance(summary["total_cost_usd"], float)
    json.dumps(summary)


def test_unpriced_status_only_when_legacy_itself_is_unknown(caplog):
    callback = TokenTrackingCallback("synthetic")
    with caplog.at_level(logging.DEBUG, logger="lobster.utils.callbacks"):
        summary = record(callback, result(), provider=None)
    assert callback._calculate_cost(MODEL, 20, 5) is None
    assert callback.invocations[0].cost_usd is None
    assert summary["invocations"][0]["cache_cost_status"] == "unpriced"
    assert callback.unpriced_invocations == 1
    assert len(caplog.records) == 1
    assert MODEL not in caplog.records[0].message
    json.dumps(summary)


@pytest.mark.parametrize("details", [None, {}, {"cache_read": 0, "cache_creation": 0}])
def test_no_cache_and_zero_cache_match_literal_legacy_summary(details):
    response = result(read=0, write=0, ttl=False)
    if details is None:
        response.generations[0][0].message.usage_metadata.pop("input_token_details")
    else:
        response.generations[0][0].message.usage_metadata[
            "input_token_details"
        ] = details
    actual = tracker()
    actual_summary = record(actual, response)
    assert actual.total_cost_usd == legacy_amount()
    assert isinstance(actual_summary["total_cost_usd"], float)
    assert not any(k.startswith("cache_cost") for k in actual_summary["invocations"][0])
    json.dumps(actual_summary)
    # Disable only the new seam: existing cost/usage/serialization is the control.
    from lobster.pricing.cache_cost import CacheCostDecision

    control = tracker()
    with patch(
        "lobster.pricing.cache_cost.price_cache_usage",
        return_value=CacheCostDecision("legacy", None, None, None),
    ):
        control_summary = record(control, deepcopy(response))
    for summary in (actual_summary, control_summary):
        summary["invocations"][0].pop("timestamp")
    assert json.dumps(actual_summary, sort_keys=True) == json.dumps(
        control_summary, sort_keys=True
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"provider": "amazon_bedrock"},
        {"ls_provider": "amazon_bedrock", "provider": "bedrock"},
        {"ls_provider": "unknown", "provider": "amazon_bedrock"},
    ],
)
def test_only_actual_ls_label_no_conflicting_provider_override(metadata, caplog):
    callback = tracker()
    callback.on_chat_model_start({}, [[]], run_id=RUN, metadata=metadata)
    with caplog.at_level(logging.DEBUG, logger="lobster.utils.callbacks"):
        callback.on_llm_end(result(), run_id=RUN)
    summary = callback.get_usage_summary()
    assert callback.total_cost_usd == legacy_amount()
    assert summary["invocations"][0]["cache_cost_status"] == "legacy_fallback"
    assert len(caplog.records) == 1


def test_conflicting_exact_model_identity_cannot_override_legacy():
    callback = tracker()
    callback.on_chat_model_start(
        {},
        [[]],
        run_id=RUN,
        metadata={"ls_provider": "amazon_bedrock", "ls_model_name": "unresolved-alias"},
    )
    callback.on_llm_end(result(), run_id=RUN)
    assert callback.total_cost_usd == legacy_amount()
    assert (
        callback.get_usage_summary()["invocations"][0]["cache_cost_reason"]
        == "pricing_identity_conflict"
    )


def test_reset_clears_sparse_cache_evidence_and_preserves_json_float():
    callback = tracker()
    record(callback, result())
    callback.reset()
    summary = record(callback, result(read=0, write=0, ttl=False))
    assert "cache_cost_status" not in summary["invocations"][0]
    assert isinstance(summary["total_cost_usd"], float)
    json.dumps(summary)


@pytest.mark.parametrize("path", ["direct", "gateway"])
@pytest.mark.parametrize("mode", ["invoke", "stream"])
def test_actual_sdk_provider_label_and_ledger_both_paths(
    monkeypatch, request, path, mode
):
    import socket

    from tests.unit.config.test_bedrock_prompt_cache import Runtime, model_for

    def refuse(*args, **kwargs):
        raise AssertionError("Unexpected network access")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setenv("LOBSTER_BEDROCK_PROMPT_CACHE", "1")
    if path == "gateway":
        provider, runtime, _ = request.getfixturevalue("gateway")
        runtime.tool_loop = False
        model = provider.create_chat_model(
            MODEL, endpoint="https://app.omics-os.com", cache=False
        )
    else:
        runtime = Runtime()
        model = model_for(runtime, MODEL)
    original_response = runtime.response

    def response(parameters):
        output = original_response(parameters)
        output["usage"] = {
            "inputTokens": 10,
            "outputTokens": 5,
            "totalTokens": 25,
            "cacheReadInputTokens": 7,
            "cacheWriteInputTokens": 3,
            "cacheDetails": [{"ttl": "5m", "inputTokens": 3}],
        }
        return output

    monkeypatch.setattr(runtime, "response", response)
    seen = []
    callback = tracker()
    original_start = callback.on_chat_model_start

    def start(*args, **kwargs):
        seen.append(dict(kwargs.get("metadata") or {}))
        return original_start(*args, **kwargs)

    monkeypatch.setattr(callback, "on_chat_model_start", start)
    if mode == "stream":
        list(model.stream("synthetic", config={"callbacks": [callback]}))
    else:
        model.invoke("synthetic", config={"callbacks": [callback]})
    assert seen and seen[0]["ls_provider"] == "amazon_bedrock"
    assert callback.invocations[0].usage_complete
    assert callback.invocations[0].provider == "amazon_bedrock"
    rates = get_rates("bedrock", MODEL)
    expected = float(
        (
            10 * rates.input
            + 5 * rates.output
            + 7 * rates.cache_read
            + 3 * rates.cache_write
        )
        / Decimal(1_000_000)
    )
    assert callback.total_cost_usd == expected
    assert (
        callback.get_usage_summary()["invocations"][0]["cache_cost_status"] == "catalog"
    )
    json.dumps(callback.get_usage_summary())
