"""Frozen decision contract in the run API (Phase 0 A2, evidence/P0_A2_decision_contract.md)."""

import pytest

from lobster.core.runs import DecisionError, RunManager


class FakeClient:
    session_id = "s1"

    def __init__(self, *batches):
        self.batches = list(batches)
        self.resumed_with = []

    def query(self, prompt, stream=True, cancel_event=None):
        return iter(self.batches.pop(0))

    def resume_from_interrupt(self, response, stream=True, cancel_event=None):
        self.resumed_with.append(response)
        return iter(self.batches.pop(0))


COMPLETE = {"type": "complete", "response": "done", "session_id": "s1"}
SLIDER = {
    "component": "threshold_slider",
    "mode": "overlay",
    "data": {"label": "Which threshold?", "min": 0.0, "max": 1.0, "default": 0.05},
    "fallback_prompt": "Which threshold? (range: 0.0-1.0, default: 0.05)",
}


def interrupt(interrupt_id, ask_id="call_1"):
    return {
        "type": "interrupt",
        "interrupt_id": interrupt_id,
        "data": {**SLIDER, "ask_id": ask_id},
    }


def pause(manager, prompt="qc"):
    run = manager.start(prompt)
    events = list(run.stream())
    return run, events


def decision_event(events):
    return next(e for e in events if e["type"] == "decision_required")


def test_waiting_run_state_carries_the_decision_and_no_interrupt_id():
    manager = RunManager(FakeClient([interrupt("i1")]))
    run, events = pause(manager)
    state = next(e for e in events if e.get("state") == "needs_decision")
    decision = state["decision"]
    assert decision["decision_id"].startswith("dec_")
    assert decision["request_version"] == 1
    assert decision["question"] == "Which threshold?"
    assert decision["components"] == [
        {
            "n": 1,
            "component": "threshold_slider",
            "data": SLIDER["data"],
            "fallback_prompt": SLIDER["fallback_prompt"],
        }
    ]
    # The state comes before its decision event, so a client reads the version first.
    assert events.index(state) < events.index(decision_event(events))
    for event in events:
        assert "interrupt_id" not in event
        assert "ask_id" not in str(event)


@pytest.mark.parametrize("second_interrupt_id", ["i1", "i2"])
def test_resurfaced_question_keeps_its_id_and_bumps_the_version(second_interrupt_id):
    # LangGraph 1.2.16 keeps the interrupt id on a re-ask; 1.2.20 mints a new one.
    client = FakeClient([interrupt("i1")], [interrupt(second_interrupt_id)])
    manager = RunManager(client)
    _, events = pause(manager)
    first = decision_event(events)
    second_run = manager.answer(first["decision_id"], 0.5, source="human")
    second_events = list(second_run.stream())
    again = decision_event(second_events)
    assert again["decision_id"] == first["decision_id"]
    assert again["request_version"] == 2
    with pytest.raises(DecisionError) as stale:
        manager.answer(first["decision_id"], 0.2, source="human", request_version=1)
    assert stale.value.code == "rejected_stale"
    client.batches.append([COMPLETE])
    third = manager.answer(first["decision_id"], 0.2, source="human", request_version=2)
    assert list(client.resumed_with[-1]) == [second_interrupt_id]
    list(third.stream())
    assert third.state == "completed"


def test_answer_resumes_with_an_envelope_carrying_source_and_comment():
    client = FakeClient([interrupt("i1")], [COMPLETE])
    manager = RunManager(client)
    _, events = pause(manager)
    decision_id = decision_event(events)["decision_id"]
    manager.answer(
        decision_id, 0.1, source="model", request_version=1, comment="per protocol"
    )
    envelope = client.resumed_with[0]["i1"]["__lobster_decision__"]
    assert envelope == {
        "decision_id": decision_id,
        "request_version": 1,
        "answer": 0.1,
        "source": "model",
        "comment": "per protocol",
    }


def test_channel_is_carried_in_the_envelope_only_when_given():
    client = FakeClient([interrupt("i1")], [COMPLETE])
    manager = RunManager(client)
    _, events = pause(manager)
    decision_id = decision_event(events)["decision_id"]
    manager.answer(
        decision_id, 0.1, source="human", request_version=1, channel="mcp:claude-code"
    )
    envelope = client.resumed_with[0]["i1"]["__lobster_decision__"]
    assert envelope["channel"] == "mcp:claude-code"
    assert envelope["source"] == "human"


def test_channel_must_be_a_string():
    manager = RunManager(FakeClient([interrupt("i1")]))
    run, events = pause(manager)
    with pytest.raises(DecisionError) as bad:
        manager.answer(
            decision_event(events)["decision_id"], 0.1, source="human", channel=7
        )
    assert bad.value.code == "bad_answer"
    assert run.state == "needs_decision"


def test_source_outside_the_set_is_a_bad_answer():
    manager = RunManager(FakeClient([interrupt("i1")]))
    run, events = pause(manager)
    with pytest.raises(DecisionError) as bad:
        manager.answer(decision_event(events)["decision_id"], 0.1, source="robot")
    assert bad.value.code == "bad_answer"
    assert run.state == "needs_decision"


def test_duplicate_answer_is_refused_with_the_continuing_run():
    manager = RunManager(FakeClient([interrupt("i1")], [COMPLETE]))
    _, events = pause(manager)
    decision_id = decision_event(events)["decision_id"]
    second = manager.answer(decision_id, 0.1, source="human")
    with pytest.raises(DecisionError) as again:
        manager.answer(decision_id, 0.1, source="human")
    assert again.value.code == "already_answered"
    assert again.value.details == {"resumed_by": second.run_id}


def test_decision_states_are_pending_answered_cancelled():
    manager = RunManager(FakeClient([interrupt("i1")], [COMPLETE]))
    _, events = pause(manager)
    decision_id = decision_event(events)["decision_id"]
    assert manager.decisions[decision_id].state == "pending"
    manager.answer(decision_id, 0.1, source="human")
    assert manager.decisions[decision_id].state == "answered"

    other = RunManager(FakeClient([interrupt("i9", ask_id="call_9")]))
    run, events = pause(other)
    other.cancel(run.run_id, reason="decision_abandoned")
    assert other.decisions[decision_event(events)["decision_id"]].state == "cancelled"
    assert run.events[-1]["reason"] == "decision_abandoned"


def test_cancel_reason_defaults_to_cancel():
    manager = RunManager(FakeClient([interrupt("i1")]))
    run, _ = pause(manager)
    manager.cancel(run.run_id)
    assert run.events[-1]["reason"] == "cancel"


def test_stream_stopped_by_cancel_says_cancel():
    import threading

    cancel = threading.Event()
    cancel.set()
    manager = RunManager(FakeClient([]))
    run = manager.start("long", cancel_event=cancel)
    last = list(run.stream())[-1]
    assert (last["state"], last["reason"]) == ("cancelled", "cancel")


def test_head_follows_the_resumed_chain():
    manager = RunManager(FakeClient([interrupt("i1")], [COMPLETE]))
    first, events = pause(manager)
    second = manager.answer(decision_event(events)["decision_id"], 1, source="human")
    assert manager.head(first.run_id) is second
    assert manager.head(second.run_id) is second
    assert manager.head("run_unknown") is None


# --- Review of df9b0742 (reviews/20261111_P0_A2_df9b0742_review.md) ---------


def test_cancel_refuses_a_reason_outside_the_closed_set():
    manager = RunManager(FakeClient([interrupt("i1")]))
    run, _ = pause(manager)
    with pytest.raises(ValueError):
        manager.cancel(run.run_id, reason="consumer_closed")
    assert run.state == "needs_decision"


@pytest.mark.parametrize(
    "decision_id, source",
    [(["dec_x"], "human"), ({"a": 1}, "human"), (None, ["human"]), (None, {"h": 1})],
)
def test_malformed_fields_are_a_bad_answer_not_a_crash(decision_id, source):
    manager = RunManager(FakeClient([interrupt("i1")]))
    run, events = pause(manager)
    real = decision_event(events)["decision_id"]
    with pytest.raises(DecisionError) as bad:
        manager.answer(decision_id or real, 0.1, source=source)
    assert bad.value.code == "bad_answer"
    assert manager.decisions[real].state == "pending"
    assert run.state == "needs_decision"


def test_cancelled_decision_is_refused_even_if_its_run_still_waits():
    # The state a concurrent cancel leaves between its two writes.
    manager = RunManager(FakeClient([interrupt("i1")], [COMPLETE]))
    run, events = pause(manager)
    decision_id = decision_event(events)["decision_id"]
    manager.decisions[decision_id].state = "cancelled"
    with pytest.raises(DecisionError) as refused:
        manager.answer(decision_id, 0.1, source="human")
    assert refused.value.code == "rejected_cancelled"


def test_answer_and_cancel_race_has_exactly_one_winner():
    import threading

    for _ in range(50):
        manager = RunManager(FakeClient([interrupt("i1")], [COMPLETE]))
        run, events = pause(manager)
        decision_id = decision_event(events)["decision_id"]
        go = threading.Barrier(2)
        outcome = {}

        def answer():
            go.wait()
            try:
                manager.answer(decision_id, 0.1, source="human")
                outcome["answer"] = "ok"
            except DecisionError as exc:
                outcome["answer"] = exc.code

        def cancel():
            go.wait()
            outcome["cancel"] = manager.cancel(run.run_id, "decision_abandoned")

        threads = [threading.Thread(target=answer), threading.Thread(target=cancel)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        state = manager.decisions[decision_id].state
        if outcome["answer"] == "ok":
            assert (state, run.state) == ("answered", "completed")
            assert outcome["cancel"] == "already_terminal"
        else:
            assert outcome["answer"] == "rejected_cancelled"
            assert (state, run.state) == ("cancelled", "cancelled")
