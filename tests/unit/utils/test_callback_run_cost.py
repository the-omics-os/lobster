"""``mark()`` / ``cost_since()``: the cost of one run, not of the last LLM call."""

from types import SimpleNamespace

import pytest

from lobster.utils.callbacks import TokenTrackingCallback

pytestmark = pytest.mark.no_auto_config

PRICES = {"m": {"input_per_million": 1.0, "output_per_million": 2.0}}


def call(tracker, tokens_in, tokens_out, model="m"):
    usage = {"input_tokens": tokens_in, "output_tokens": tokens_out}
    tracker.on_llm_end(
        SimpleNamespace(
            generations=[[]], llm_output={"usage": usage, "model_name": model}
        )
    )


def test_a_run_is_costed_from_the_mark_and_the_last_call_is_not_the_run():
    tracker = TokenTrackingCallback("s", PRICES)
    call(tracker, 1_000_000, 0)  # an earlier run: $1
    mark = tracker.mark()
    call(tracker, 1_000_000, 500_000)  # this run: $1 + $1
    call(tracker, 0, 0)  # a near-empty last call
    run = tracker.cost_since(mark)
    assert run["run_cost_usd"] == pytest.approx(2.0)
    assert run["run_total_tokens"] == 1_500_000
    assert run["run_cost_complete"] is True
    latest = tracker.get_latest_cost()
    assert latest["latest_cost_usd"] == 0.0  # the misleading field
    assert latest["session_total_usd"] == pytest.approx(3.0)


def test_unpriced_and_usage_less_calls_after_the_mark_make_the_run_incomplete():
    tracker = TokenTrackingCallback("s", PRICES)
    call(tracker, 10, 10, model="unknown-model")  # before the mark: not this run's
    mark = tracker.mark()
    assert tracker.cost_since(mark)["run_cost_complete"] is True
    call(tracker, 10, 10, model="unknown-model")
    tracker.on_llm_end(SimpleNamespace(generations=[[]], llm_output={}))
    run = tracker.cost_since(mark)
    assert run["run_unpriced_invocations"] == 1
    assert run["run_missing_usage_invocations"] == 1
    assert run["run_cost_complete"] is False
    assert run["run_cost_usd"] == 0.0


def test_a_reset_between_mark_and_now_is_unknown_not_free():
    tracker = TokenTrackingCallback("s", PRICES)
    call(tracker, 1_000_000, 0)
    mark = tracker.mark()
    tracker.reset()
    run = tracker.cost_since(mark)
    assert run["run_cost_usd"] == 0.0 and run["run_total_tokens"] == 0
    assert run["run_cost_complete"] is False  # unknown, not free


UNKNOWN = {
    "run_cost_usd": 0.0,
    "run_total_tokens": 0,
    "run_cost_complete": False,
    "run_unpriced_invocations": 0,
    "run_missing_usage_invocations": 0,
}


def test_a_mark_from_another_tracker_is_unknown_and_never_raises():
    first = TokenTrackingCallback("s", PRICES)
    call(first, 1_000_000, 0)
    mark = first.mark()
    restored = TokenTrackingCallback(
        "s", PRICES
    )  # the session came back on a new tracker
    call(restored, 1_000_000, 0)
    assert restored.cost_since(mark) == UNKNOWN


@pytest.mark.parametrize("bad", [None, {}, {"cost": 1.0}, "x", 3])
def test_a_malformed_mark_is_unknown_and_never_raises(bad):
    assert TokenTrackingCallback("s", PRICES).cost_since(bad) == UNKNOWN


def test_marking_again_is_independent_of_earlier_marks():
    tracker = TokenTrackingCallback("s", PRICES)
    first = tracker.mark()
    call(tracker, 1_000_000, 0)
    second = tracker.mark()  # a continued turn marks again
    call(tracker, 0, 500_000)
    assert tracker.cost_since(first)["run_cost_usd"] == pytest.approx(2.0)
    assert tracker.cost_since(second)["run_cost_usd"] == pytest.approx(1.0)
