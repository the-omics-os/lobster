"""Per-call nominal evidence must not alter legacy callback cost arithmetic."""

import json
from types import SimpleNamespace

import pytest

from lobster.utils.callbacks import TokenTrackingCallback

pytestmark = pytest.mark.no_auto_config


@pytest.fixture
def tracker():
    return TokenTrackingCallback(
        "usage", {"model": {"input_per_million": 3, "output_per_million": 15}}
    )


def raw(usage, key="usage"):
    return SimpleNamespace(
        generations=[[]], llm_output={key: usage, "model_name": "model"}
    )


def normalized(usage):
    message = SimpleNamespace(
        usage_metadata=usage, response_metadata={"model_name": "model"}
    )
    return SimpleNamespace(
        generations=[[SimpleNamespace(message=message)]], llm_output={}
    )


def test_raw_anthropic_counts_cache_once_without_repricing(tracker):
    tracker.on_llm_end(
        raw(
            {
                "input_tokens": 1000,
                "output_tokens": 100,
                "cache_read_input_tokens": 9000,
                "cache_creation_input_tokens": 2000,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 500,
                    "ephemeral_1h_input_tokens": 1500,
                },
            }
        )
    )
    inv = tracker.invocations[0]
    assert inv.input_tokens == 1000  # Existing raw counter/cost stays unchanged.
    assert inv.logical_input_tokens == 12000
    assert inv.cache_read_tokens == 9000
    assert inv.cache_write_tokens == 2000
    assert (
        inv.input_token_basis
        == "exclusive"  # nosec B105 - token-count fixture, not a credential
    )
    assert inv.usage_complete
    assert inv.cost_usd == pytest.approx(0.0045)
    assert tracker.total_cost_usd == pytest.approx(0.0045)


def test_langchain_input_and_reasoning_are_already_inclusive(tracker):
    tracker.on_llm_end(
        normalized(
            {
                "input_tokens": 12000,
                "output_tokens": 100,
                "total_tokens": 12100,
                "input_token_details": {"cache_read": 9000, "cache_creation": 2000},
                "output_token_details": {"reasoning": 40},
            }
        )
    )
    inv = tracker.invocations[0]
    assert inv.logical_input_tokens == 12000  # Not 23000.
    assert inv.output_tokens == 100  # Not 140.
    assert inv.reasoning_tokens == 40
    assert (
        inv.input_token_basis
        == "inclusive"  # nosec B105 - token-count fixture, not a credential
    )
    assert inv.usage_source == "langchain_usage_metadata"
    assert inv.cost_usd == pytest.approx(0.0375)
    assert inv.usage_complete


def test_openai_cache_and_reasoning_are_subset_details(tracker):
    tracker.on_llm_end(
        raw(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "total_tokens": 1100,
                "prompt_tokens_details": {"cached_tokens": 800},
                "completion_tokens_details": {"reasoning_tokens": 40},
            },
            "token_usage",
        )
    )
    inv = tracker.invocations[0]
    assert inv.logical_input_tokens == 1000
    assert inv.cache_read_tokens == 800
    assert inv.cache_write_tokens is None
    assert inv.reasoning_tokens == 40
    assert inv.usage_complete


def test_absent_details_are_not_invented_zero(tracker):
    tracker.on_llm_end(normalized({"input_tokens": 1000, "output_tokens": 100}))
    inv = tracker.invocations[0]
    assert inv.logical_input_tokens == 1000
    assert inv.cache_read_tokens is None
    assert inv.cache_write_tokens is None
    assert inv.reasoning_tokens is None
    assert inv.usage_complete  # Inclusive logical totals can still be known.


def test_explicit_zero_is_retained(tracker):
    tracker.on_llm_end(
        normalized(
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "input_token_details": {"cache_read": 0, "cache_creation": 0},
                "output_token_details": {"reasoning": 0},
            }
        )
    )
    inv = tracker.invocations[0]
    assert (
        inv.logical_input_tokens == inv.cache_read_tokens == inv.cache_write_tokens == 0
    )
    assert inv.reasoning_tokens == 0
    assert inv.usage_complete


@pytest.mark.parametrize("bad", [-1, True, 1.5, "4"])
def test_invalid_cache_evidence_is_incomplete_not_free(tracker, bad):
    tracker.on_llm_end(
        normalized(
            {
                "input_tokens": 1000,
                "output_tokens": 100,
                "input_token_details": {"cache_read": bad},
            }
        )
    )
    assert not tracker.invocations[0].usage_complete
    assert tracker.invocations[0].logical_input_tokens is None
    assert tracker.total_cost_usd == pytest.approx(0.0045)  # Legacy unchanged.


@pytest.mark.parametrize(
    "usage",
    [
        {
            "input_tokens": 100,
            "output_tokens": 10,
            "cache_creation_input_tokens": 20,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 5,
                "ephemeral_1h_input_tokens": 10,
            },
        },
        {
            "input_tokens": 100,
            "output_tokens": 10,
            "cache_creation_input_tokens": 20,
            "cache_write_input_tokens": 21,
        },
    ],
)
def test_conflicting_cache_aggregate_or_alias_is_incomplete(tracker, usage):
    tracker.on_llm_end(raw(usage))
    assert not tracker.invocations[0].usage_complete
    assert tracker.invocations[0].logical_input_tokens is None


def test_cache_details_cannot_exceed_inclusive_input(tracker):
    tracker.on_llm_end(
        normalized(
            {
                "input_tokens": 100,
                "output_tokens": 10,
                "input_token_details": {"cache_read": 101},
            }
        )
    )
    assert not tracker.invocations[0].usage_complete


def test_reasoning_cannot_exceed_inclusive_output(tracker):
    tracker.on_llm_end(
        normalized(
            {
                "input_tokens": 100,
                "output_tokens": 10,
                "output_token_details": {"reasoning": 11},
            }
        )
    )
    assert not tracker.invocations[0].usage_complete


def test_missing_completion_or_error_does_not_change_money(tracker):
    tracker.on_llm_end(raw({"input_tokens": 1000, "output_tokens": 100}))
    tracker.on_llm_end(SimpleNamespace(generations=[[]], llm_output={}))
    tracker.on_llm_error(RuntimeError("mock failure"))
    summary = tracker.get_usage_summary()
    assert len(tracker.invocations) == 1  # Legacy log unchanged.
    assert summary["total_cost_usd"] == 0.0045
    assert summary["missing_usage_invocations"] == 2
    assert summary["usage_complete"] is False
    # A call without usage cannot be priced: the money is unchanged but not complete.
    assert summary["cost_complete"] is False
    assert summary["attempt_coverage"] == "callback_only"


def test_multiple_generations_are_not_claimed_complete(tracker):
    result = normalized({"input_tokens": 100, "output_tokens": 10})
    result.generations.append(result.generations[0])
    tracker.on_llm_end(result)
    assert not tracker.invocations[0].usage_complete
    assert len(tracker.invocations) == 1


def test_per_run_provider_evidence_does_not_cross_parallel_calls(tracker):
    tracker._valid_agents = {"supervisor"}
    tracker.on_chat_model_start(
        {},
        [],
        run_id="a",
        metadata={"provider": "bedrock", "ls_model_name": "requested-a"},
    )
    tracker.on_chat_model_start(
        {},
        [],
        run_id="b",
        metadata={"provider": "anthropic", "ls_model_name": "requested-b"},
    )
    tracker.on_llm_end(raw({"input_tokens": 1, "output_tokens": 1}), run_id="a")
    tracker.on_llm_end(raw({"input_tokens": 1, "output_tokens": 1}), run_id="b")
    a, b = tracker.invocations
    assert (a.provider, a.requested_model) == ("bedrock", "requested-a")
    assert (b.provider, b.requested_model) == ("anthropic", "requested-b")
    assert a.provider_source == b.provider_source == "metadata"
    assert tracker._usage_context == {}


def test_pending_start_is_incomplete_without_affecting_money(tracker):
    tracker._valid_agents = {"supervisor"}
    tracker.on_chat_model_start({}, [], run_id="pending")
    summary = tracker.get_usage_summary()
    assert summary["pending_usage_invocations"] == 1
    assert not summary["usage_complete"]
    assert summary["total_cost_usd"] == 0
    tracker.on_llm_error(RuntimeError("cancelled"), run_id="pending")
    assert tracker.get_usage_summary()["pending_usage_invocations"] == 0
    assert not tracker.get_usage_summary()["usage_complete"]


def test_sdk_provider_telemetry_is_labelled_not_assumed_seller(tracker):
    tracker._valid_agents = {"supervisor"}
    tracker.on_chat_model_start(
        {}, [], run_id="a", metadata={"ls_provider": "openai", "ls_model_name": "model"}
    )
    tracker.on_llm_end(raw({"input_tokens": 1, "output_tokens": 1}), run_id="a")
    assert tracker.invocations[0].provider == "openai"
    assert tracker.invocations[0].provider_source == "ls_provider"


def test_message_level_native_usage_resolves_inconsistent_sdk_total(tracker):
    usage = {
        "input_tokens": 12000,
        "output_tokens": 100,
        "total_tokens": 1100,
        "input_token_details": {"cache_read": 9000, "cache_creation": 2000},
        "output_token_details": {"reasoning": 40},
    }
    message = SimpleNamespace(
        usage_metadata=usage,
        response_metadata={
            "model_name": "model",
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 100,
                "cache_read_input_tokens": 9000,
                "cache_creation_input_tokens": 2000,
            },
        },
    )
    result = SimpleNamespace(
        generations=[[SimpleNamespace(message=message)]], llm_output={}
    )
    tracker.on_llm_end(result)
    inv = tracker.invocations[0]
    assert inv.logical_input_tokens == 12000 and inv.logical_output_tokens == 100
    assert inv.reasoning_tokens == 40 and inv.usage_complete


def test_message_level_openai_token_usage_details_are_reconciled(tracker):
    message = SimpleNamespace(
        usage_metadata={
            "input_tokens": 1000,
            "output_tokens": 100,
            "total_tokens": 1100,
        },
        response_metadata={
            "model_name": "model",
            "token_usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "total_tokens": 1100,
                "prompt_tokens_details": {"cached_tokens": 800},
                "completion_tokens_details": {"reasoning_tokens": 40},
            },
        },
    )
    tracker.on_llm_end(
        SimpleNamespace(generations=[[SimpleNamespace(message=message)]], llm_output={})
    )
    inv = tracker.invocations[0]
    assert inv.logical_input_tokens == 1000 and inv.cache_read_tokens == 800
    assert (
        inv.logical_output_tokens == 100
        and inv.reasoning_tokens == 40
        and inv.usage_complete
    )


def test_duplicate_terminal_run_id_keeps_legacy_money_but_invalidates_coverage(tracker):
    result = raw({"input_tokens": 1000, "output_tokens": 100})
    tracker.on_llm_end(result, run_id="same")
    tracker.on_llm_end(result, run_id="same")
    assert tracker.total_cost_usd == pytest.approx(
        0.009
    )  # Deliberately legacy behavior.
    assert len(tracker.invocations) == 2
    assert tracker.invocations[1].run_id == "same"
    assert not tracker.invocations[1].usage_complete
    summary = tracker.get_usage_summary()
    assert summary["duplicate_usage_invocations"] == 1
    assert not summary["usage_complete"]
    tracker.reset()
    assert tracker.get_usage_summary()["duplicate_usage_invocations"] == 0


def test_null_exclusive_cache_not_silently_zero(tracker):
    tracker.on_llm_end(
        raw(
            {
                "input_tokens": 1000,
                "output_tokens": 100,
                "cache_read_input_tokens": None,
                "cache_creation_input_tokens": 0,
            }
        )
    )
    assert tracker.invocations[0].logical_input_tokens is None
    assert not tracker.invocations[0].usage_complete
    assert tracker.total_cost_usd == pytest.approx(0.0045)


def test_stream_chunks_merge_before_one_terminal_observation(tracker):
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGeneration, LLMResult

    first = AIMessageChunk(
        content="a",
        response_metadata={"model_name": "model"},
        usage_metadata={
            "input_tokens": 1000,
            "output_tokens": 40,
            "total_tokens": 12040,
            "input_token_details": {"cache_read": 9000, "cache_creation": 2000},
            "output_token_details": {"reasoning": 10},
        },
    )
    last = AIMessageChunk(
        content="b",
        usage_metadata={
            "input_tokens": 0,
            "output_tokens": 60,
            "total_tokens": 60,
            "input_token_details": {"cache_read": 0, "cache_creation": 0},
            "output_token_details": {"reasoning": 30},
        },
    )
    tracker.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=first + last)]]), run_id="stream"
    )
    inv = tracker.invocations[0]
    assert inv.logical_input_tokens == 12000
    assert inv.logical_output_tokens == 100
    assert inv.reasoning_tokens == 40 and inv.cache_write_tokens == 2000
    assert inv.usage_complete and len(tracker.invocations) == 1
    assert inv.cost_usd == pytest.approx(
        0.0045
    )  # Original exclusive input stays unchanged.


def test_usage_dimensions_survive_json_save_and_reset(tracker, tmp_path):
    tracker.on_llm_end(
        normalized(
            {
                "input_tokens": 100,
                "output_tokens": 10,
                "input_token_details": {"cache_read": 90},
                "total_tokens": 110,
            }
        )
    )
    tracker.save_to_workspace(tmp_path)
    summary = json.loads((tmp_path / "token_usage.json").read_text())
    inv = summary["invocations"][0]
    assert inv["cache_read_tokens"] == 90
    assert inv["logical_input_tokens"] == 100
    assert inv["reasoning_tokens"] is None
    assert summary["usage_complete"]
    tracker.on_llm_error(RuntimeError("mock"))
    tracker.reset()
    assert tracker.get_usage_summary()["missing_usage_invocations"] == 0
    assert tracker.invocations == []
