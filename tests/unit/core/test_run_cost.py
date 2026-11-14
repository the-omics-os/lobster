"""A result's ``token_usage`` says what the run cost, not only the last call."""

from types import SimpleNamespace

import pytest

from lobster.core.runs import RUN_COST_KEYS, RunManager
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


class Client:
    """Spends what ``spend`` says during each query, then completes."""

    session_id = "s1"

    def __init__(self, tracker=None, spend=()):
        self.token_tracker = tracker
        self.spend = list(spend)

    def _play(self):
        for step in self.spend.pop(0) if self.spend else ():
            step(self)
        usage = self.token_tracker.get_latest_cost() if self.token_tracker else None
        yield {"type": "complete", "response": "ok", "token_usage": usage}

    def query(self, prompt, stream=True, cancel_event=None):
        return self._play()

    def resume_from_interrupt(self, response, stream=True, cancel_event=None):
        return self._play()


def finish(manager, prompt="go"):
    run = manager.start(prompt)
    list(run.stream())
    assert run.state == "completed"
    return run.result


def test_the_result_costs_the_run_and_keeps_the_last_call_figure_as_it_was():
    tracker = TokenTrackingCallback("s", PRICES)
    call(tracker, 1_000_000, 0)  # an earlier run: $1
    spend = [lambda c: call(c.token_tracker, 1_000_000, 500_000)]  # $1 + $1
    spend += [lambda c: call(c.token_tracker, 0, 0)]  # a near-empty last call
    result = finish(RunManager(Client(tracker, [spend])))
    usage = result["token_usage"]
    assert usage["run_cost_usd"] == pytest.approx(2.0)
    assert usage["run_total_tokens"] == 1_500_000
    assert usage["run_cost_complete"] is True
    assert usage["run_unpriced_invocations"] == 0
    assert usage["run_missing_usage_invocations"] == 0
    # What was there before is untouched: the last call, the session totals.
    assert usage["latest_cost_usd"] == 0.0
    assert usage["session_total_usd"] == pytest.approx(3.0)
    assert set(RUN_COST_KEYS) <= set(usage)


def test_a_second_run_is_costed_from_its_own_start():
    tracker = TokenTrackingCallback("s", PRICES)
    manager = RunManager(
        Client(
            tracker,
            [
                [lambda c: call(c.token_tracker, 1_000_000, 0)],
                [lambda c: call(c.token_tracker, 0, 500_000)],
            ],
        )
    )
    assert finish(manager)["token_usage"]["run_cost_usd"] == pytest.approx(1.0)
    assert finish(manager)["token_usage"]["run_cost_usd"] == pytest.approx(1.0)


def test_calls_without_price_or_usage_make_the_figure_a_lower_bound():
    tracker = TokenTrackingCallback("s", PRICES)

    def unpriced(c):
        call(c.token_tracker, 10, 10, model="unknown")

    def silent(c):
        c.token_tracker.on_llm_end(SimpleNamespace(generations=[[]], llm_output={}))

    usage = finish(RunManager(Client(tracker, [[unpriced, silent]])))["token_usage"]
    assert usage["run_cost_complete"] is False
    assert usage["run_unpriced_invocations"] == 1
    assert usage["run_missing_usage_invocations"] == 1


def test_a_run_that_resumes_after_an_answer_is_costed_from_the_first_start():
    tracker = TokenTrackingCallback("s", PRICES)
    client = Client(tracker, [[lambda c: call(c.token_tracker, 1_000_000, 0)]])
    manager = RunManager(client)
    first = manager.start("go")
    list(first.stream())
    # The answer starts a run that continues the first one.
    client.spend = [[lambda c: call(c.token_tracker, 1_000_000, 0)]]
    first.state = "needs_decision"
    manager.decisions["d"] = SimpleNamespace(
        state="pending",
        origin_run_id=first.run_id,
        request_version=1,
        interrupt_id=None,
        decision_id="d",
    )
    resumed = manager.answer("d", 1, source="model")
    list(resumed.stream())
    assert resumed.result["token_usage"]["run_cost_usd"] == pytest.approx(2.0)


def test_a_missing_tracker_leaves_the_result_without_run_keys():
    result = finish(RunManager(Client(None)))
    assert result["token_usage"] is None
    event_usage = {"input": 1}
    client = SimpleNamespace(session_id="s", token_tracker=None)
    client.query = lambda *a, **k: iter(
        [{"type": "complete", "response": "ok", "token_usage": event_usage}]
    )
    usage = finish(RunManager(client))["token_usage"]
    assert usage == {"input": 1}


def test_a_tracker_replaced_after_the_mark_leaves_the_result_without_run_keys():
    tracker = TokenTrackingCallback("s", PRICES)

    def replace(c):
        c.token_tracker = TokenTrackingCallback("s", PRICES)
        call(c.token_tracker, 1_000_000, 0)

    usage = finish(RunManager(Client(tracker, [[replace]])))["token_usage"]
    assert not set(RUN_COST_KEYS) & set(usage)
    assert usage["session_total_usd"] == pytest.approx(1.0)  # still reported


def test_a_tracker_that_cannot_cost_the_run_never_fails_it():
    tracker = TokenTrackingCallback("s", PRICES)

    def broken(_mark):
        raise RuntimeError("counters unreadable")

    tracker.cost_since = broken
    usage = finish(RunManager(Client(tracker, [[]])))["token_usage"]
    assert not set(RUN_COST_KEYS) & set(usage)
    assert "latest_cost_usd" in usage


def test_a_tracker_that_cannot_mark_never_fails_the_run():
    tracker = TokenTrackingCallback("s", PRICES)

    def broken():
        raise RuntimeError("counters unreadable")

    tracker.mark = broken
    usage = finish(RunManager(Client(tracker, [[]])))["token_usage"]
    assert not set(RUN_COST_KEYS) & set(usage)
