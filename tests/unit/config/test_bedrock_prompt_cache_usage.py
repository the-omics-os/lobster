"""Real L1 usage-only replay through SDK/direct/gateway and the actual ledger.

This is offline replay of recorded usage, NOT another live capture/hit/savings
or deployment proof. All response content and requests below are synthetic.
"""

import json
import socket
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import pytest

from lobster.pricing import SNAPSHOT, for_context, get_rates
from lobster.pricing.usage import normalize_usage
from lobster.utils.callbacks import TokenTrackingCallback
from tests.unit.config.test_bedrock_prompt_cache import (  # noqa: F401
    Runtime,
    assert_no_unknown_cache_keys,
    gateway,
    model_for,
    points,
)

pytestmark = pytest.mark.no_auto_config
FIXTURE = Path(__file__).parents[2] / "fixtures/bedrock/cache_usage_20261027.json"
DATA = json.loads(FIXTURE.read_text())
CALLS = DATA["calls"]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("Unexpected network access")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setenv("LOBSTER_BEDROCK_PROMPT_CACHE", "1")


class RecordingCallback(TokenTrackingCallback):
    """Observe real SDK lifecycle metadata, not hand-injected provider labels."""

    def __init__(self, model_id):
        rates = get_rates("bedrock", model_id)
        super().__init__(
            "usage-only-replay",
            {
                model_id: {
                    "input_per_million": float(rates.input),
                    "output_per_million": float(rates.output),
                }
            },
        )
        self.starts = []

    def on_chat_model_start(self, *args, **kwargs):
        self.starts.append(deepcopy(kwargs.get("metadata") or {}))
        return super().on_chat_model_start(*args, **kwargs)


@pytest.fixture(params=["direct", "gateway"])
def transport(request, monkeypatch):
    # Every SDK replay consumer builds a ChatBedrockConverse (langchain_aws, an
    # optional extra absent in Fast Validation): skip them there, run them in the
    # full env. Tests that do not use this fixture stay unaffected.
    pytest.importorskip("langchain_aws")
    if request.param == "direct":
        runtime = Runtime()
        return "direct", runtime, lambda model_id: model_for(runtime, model_id)
    provider, runtime, _ = request.getfixturevalue("gateway")
    runtime.tool_loop = False

    def build(model_id):
        # Exercise real gateway alias resolution rather than injecting metadata.
        requested = (
            "claude-sonnet-4-5-20250929"
            if model_id == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
            else model_id
        )
        return provider.create_chat_model(
            requested, endpoint="https://app.omics-os.com", cache=False
        )

    return "gateway", runtime, build


def install_usage(monkeypatch, runtime, native_usage):
    original = runtime.response

    def response(parameters):
        # Usage is copied verbatim; native 5m cacheDetails are never reconstructed.
        result = original(parameters)
        result["usage"] = deepcopy(native_usage)
        return result

    monkeypatch.setattr(runtime, "response", response)


def invoke_capture(monkeypatch, transport, call):
    path, runtime, build = transport
    install_usage(monkeypatch, runtime, call["usage"])
    model = build(call["model_id"])
    callback = RecordingCallback(call["model_id"])
    if call["api_type"] == "converse_stream":
        chunks = list(
            model.stream(
                "synthetic usage-only replay", config={"callbacks": [callback]}
            )
        )
        message = chunks[0]
        for chunk in chunks[1:]:
            message += chunk
    else:
        message = model.invoke(
            "synthetic usage-only replay", config={"callbacks": [callback]}
        )
    assert runtime.requests
    found = points(runtime.requests[0])
    assert [location for location, _ in found] == ["msg0(user)[1]"]
    assert all(block == {"cachePoint": {"type": "default"}} for _, block in found)
    assert len(found) <= 4
    assert_no_unknown_cache_keys(runtime.requests[0])
    assert "cache_control" not in runtime.requests[0]
    assert "cacheDetails" not in runtime.requests[0]
    assert callback.starts[0]["ls_provider"] == "amazon_bedrock"
    assert callback.starts[0]["ls_model_name"] == call["model_id"]
    assert callback.invocations[0].requested_model == call["model_id"]
    assert callback.invocations[0].model == call["model_id"]
    assert callback.invocations[0].provider == "amazon_bedrock"
    return message, callback


def test_cache_normalizer_private_dependency_fails_loudly_if_refactored():
    from lobster.pricing import cache_cost
    from lobster.pricing.usage import _normalize_observations

    assert callable(_normalize_observations)
    assert cache_cost._normalize_observations is _normalize_observations


def test_fixture_exact_public_projection_and_real_counts():
    assert set(DATA) == {"captured_at", "source_sha256", "calls"}
    assert DATA["captured_at"] == "2026-10-27T17:22:18Z"
    assert (
        DATA["source_sha256"]
        == "4800271885184130e4b279619419027879fed8f41c815326aa8e5b6a1127d436"
    )
    assert len(CALLS) == 4
    for call in CALLS:
        assert set(call) == {"model_id", "api_type", "usage"}
        usage = call["usage"]
        assert set(usage) <= {
            "inputTokens",
            "outputTokens",
            "totalTokens",
            "cacheReadInputTokens",
            "cacheWriteInputTokens",
            "cacheDetails",
        }
        for key, value in usage.items():
            if key != "cacheDetails":
                assert type(value) is int and value >= 0
            else:
                for detail in value:
                    assert set(detail) == {"ttl", "inputTokens"}
                    assert detail["ttl"] == "5m"
                    assert type(detail["inputTokens"]) is int
        assert usage["totalTokens"] == (
            usage["inputTokens"]
            + usage["outputTokens"]
            + usage["cacheReadInputTokens"]
            + usage["cacheWriteInputTokens"]
        )
    assert [
        (c["usage"]["cacheReadInputTokens"], c["usage"]["cacheWriteInputTokens"])
        for c in CALLS
    ] == [(0, 10802), (10802, 0), (0, 7201), (0, 0)]
    assert CALLS[1]["api_type"] == "converse_stream"
    assert "cacheDetails" not in CALLS[1]["usage"]


@pytest.mark.parametrize("call", CALLS, ids=["0", "1", "2", "3"])
def test_real_usage_sdk_normalizer_exact_decomposition(monkeypatch, transport, call):
    message, callback = invoke_capture(monkeypatch, transport, call)
    native = call["usage"]
    inclusive = (
        native["inputTokens"]
        + native["cacheReadInputTokens"]
        + native["cacheWriteInputTokens"]
    )
    sdk = message.usage_metadata
    assert sdk["input_tokens"] == inclusive
    assert sdk["output_tokens"] == native["outputTokens"]
    assert sdk["total_tokens"] == native["totalTokens"]
    for metadata, raw in [(None, native), (sdk, None), (sdk, native)]:
        normalized = normalize_usage(metadata, raw, "amazon_bedrock")
        assert normalized.complete
        assert normalized.logical_input == inclusive
        assert normalized.output_incl_reasoning == native["outputTokens"]
        assert normalized.cache_read == native["cacheReadInputTokens"]
        assert normalized.cache_write == native["cacheWriteInputTokens"]
        assert (
            normalized.logical_input - normalized.cache_read - normalized.cache_write
            == native["inputTokens"]
        )
    invocation = callback.invocations[0]
    assert invocation.usage_complete
    assert invocation.logical_input_tokens == inclusive
    assert invocation.cache_read_tokens == native["cacheReadInputTokens"]
    assert invocation.cache_write_tokens == native["cacheWriteInputTokens"]
    assert invocation.logical_output_tokens == native["outputTokens"]


@pytest.mark.parametrize("call", CALLS, ids=["0", "1", "2", "3"])
def test_real_usage_actual_cache_rate_ledger_engages_both_paths(
    monkeypatch, transport, call
):
    _, callback = invoke_capture(monkeypatch, transport, call)
    native = call["usage"]
    inclusive = native["totalTokens"] - native["outputTokens"]
    rates = for_context(get_rates("bedrock", call["model_id"]), inclusive)
    expected = float(
        (
            native["inputTokens"] * rates.input
            + native["outputTokens"] * rates.output
            + native["cacheReadInputTokens"] * rates.cache_read
            + native["cacheWriteInputTokens"] * rates.cache_write
        )
        / Decimal(1_000_000)
    )
    if not (native["cacheReadInputTokens"] or native["cacheWriteInputTokens"]):
        expected = callback._calculate_cost(
            call["model_id"], inclusive, native["outputTokens"]
        )
    invocation = callback.invocations[0]
    assert invocation.cost_usd == expected
    assert callback.total_cost_usd == expected and expected > 0
    assert callback.by_agent[invocation.agent]["cost_usd"] == expected
    summary = callback.get_usage_summary()
    assert summary["total_cost_usd"] == round(expected, 4)
    assert isinstance(summary["total_cost_usd"], float)
    assert isinstance(summary["invocations"][0]["cost_usd"], float)
    json.dumps(summary)
    if native["cacheReadInputTokens"] or native["cacheWriteInputTokens"]:
        assert summary["invocations"][0]["cache_cost_status"] == "catalog"
        assert summary["invocations"][0]["cache_cost_snapshot"] == {
            "snapshot_id": SNAPSHOT.snapshot_id,
            "source_sha256": SNAPSHOT.source_sha256,
            "overrides_sha256": SNAPSHOT.overrides_sha256,
        }
        assert "cache_cost_reason" not in summary["invocations"][0]
    else:
        # Exact existing operation order on the no-cache H45 control.
        assert invocation.cost_usd == callback._calculate_cost(
            call["model_id"], inclusive, native["outputTokens"]
        )
        assert not any(
            key.startswith("cache_cost_") for key in summary["invocations"][0]
        )


# A recorded Converse usage block (Sonnet 4.5, prompt cache on, first request).
# LangChain reports this write as cache_creation 0 plus ephemeral_5m_input_tokens.
RECORDED_WRITE = {
    "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "usage": {
        "cacheDetails": [{"inputTokens": 5299, "ttl": "5m"}],
        "cacheReadInputTokenCount": 0,
        "cacheReadInputTokens": 0,
        "cacheWriteInputTokenCount": 5299,
        "cacheWriteInputTokens": 5299,
        "inputTokens": 3,
        "outputTokens": 5,
        "serverToolUsage": {},
        "totalTokens": 5307,
    },
}


@pytest.mark.parametrize("api_type", ["converse", "converse_stream"])
def test_recorded_cache_write_is_priced_at_the_write_rate(
    monkeypatch, transport, api_type
):
    call = {**RECORDED_WRITE, "api_type": api_type}
    message, callback = invoke_capture(monkeypatch, transport, call)
    details = message.usage_metadata["input_token_details"]
    assert details["cache_creation"] == 0
    assert details["ephemeral_5m_input_tokens"] == 5299
    rates = for_context(get_rates("bedrock", call["model_id"]), 5302)
    assert rates.cache_write > rates.input
    expected = float(
        (3 * rates.input + 5 * rates.output + 5299 * rates.cache_write)
        / Decimal(1_000_000)
    )
    invocation = callback.invocations[0]
    assert invocation.cache_write_tokens == 5299
    assert invocation.cost_usd == expected
    assert callback.get_usage_summary()["invocations"][0]["cache_cost_status"] == (
        "catalog"
    )
