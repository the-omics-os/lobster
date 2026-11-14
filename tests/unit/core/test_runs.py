"""Run API identity, event and decision rules."""

import threading

import pytest

from lobster.core.runs import (
    DecisionError,
    RunManager,
    is_run_id,
    new_run_id,
    runs_for,
)


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


COMPLETE = {"type": "complete", "response": "done", "session_id": "s1", "duration": 0.1}
INTERRUPT = {
    "type": "interrupt",
    "data": {"component": "confirm"},
    "interrupt_id": "i1",
}


def types(events):
    return [(e["type"], e.get("state")) for e in events]


def test_completed_run_ends_with_terminal_state_and_result():
    manager = RunManager(
        FakeClient([{"type": "content_delta", "delta": "hi"}, COMPLETE])
    )
    run = manager.start("hello")
    events = list(run.stream())
    assert types(events) == [
        ("run_state", "running"),
        ("content_delta", None),
        ("complete", None),
        ("run_state", "completed"),
    ]
    assert [e["seq"] for e in events] == [1, 2, 3, 4]
    assert {e["run_id"] for e in events} == {run.run_id}
    assert run.result["schema_version"] == "lobster.result/1"
    assert manager.get()["result"]["response"] == "done"


def test_decision_pauses_run_and_answer_resumes_as_new_run():
    client = FakeClient([INTERRUPT], [COMPLETE])
    manager = RunManager(client)
    first = manager.start("qc")
    events = list(first.stream())
    decision = next(e for e in events if e["type"] == "decision_required")
    assert "interrupt_id" not in decision
    assert decision["decision_id"].startswith("dec_")
    assert types(events)[-2:] == [
        ("run_state", "needs_decision"),
        ("decision_required", None),
    ]
    assert manager.get()["pending_decisions"] == [decision["decision_id"]]

    second = manager.answer(
        decision["decision_id"], {"confirmed": True}, source="human"
    )
    assert second.run_id != first.run_id
    envelope = client.resumed_with[0]["i1"]["__lobster_decision__"]
    assert envelope["answer"] == {"confirmed": True}
    assert first.state == "completed"
    assert first.events[-1] == {
        "type": "run_state",
        "state": "completed",
        "resumed_by": second.run_id,
        "run_id": first.run_id,
        "seq": len(first.events),
    }
    list(second.stream())
    assert second.state == "completed"


def test_second_answer_and_stale_answer_are_refused():
    manager = RunManager(FakeClient([INTERRUPT], [COMPLETE]))
    events = list(manager.start("qc").stream())
    decision_id = next(e for e in events if e["type"] == "decision_required")[
        "decision_id"
    ]
    with pytest.raises(DecisionError) as stale:
        manager.answer(decision_id, "x", source="human", request_version=2)
    assert stale.value.code == "rejected_stale"
    manager.answer(decision_id, "x", source="human")
    with pytest.raises(DecisionError) as again:
        manager.answer(decision_id, "y", source="human")
    assert again.value.code == "already_answered"


def test_cancel_paused_run_rejects_later_answer():
    manager = RunManager(FakeClient([INTERRUPT]))
    run = manager.start("qc")
    events = list(run.stream())
    decision_id = next(e for e in events if e["type"] == "decision_required")[
        "decision_id"
    ]
    assert manager.cancel(run.run_id) == "cancelled"
    assert manager.cancel(run.run_id) == "already_terminal"
    with pytest.raises(DecisionError) as refused:
        manager.answer(decision_id, "x", source="human")
    assert refused.value.code == "rejected_cancelled"


def test_stream_that_stops_after_cancel_is_cancelled():
    cancel = threading.Event()
    cancel.set()
    manager = RunManager(FakeClient([]))
    run = manager.start("long", cancel_event=cancel)
    assert list(run.stream())[-1]["state"] == "cancelled"


def test_error_event_fails_the_run():
    manager = RunManager(FakeClient([{"type": "error", "error": "boom"}]))
    run = manager.start("x")
    last = list(run.stream())[-1]
    assert (last["state"], last["error"]["message"]) == ("failed", "boom")


def test_events_replay_after_seq():
    manager = RunManager(FakeClient([COMPLETE]))
    run = manager.start("x")
    list(run.stream())
    assert [e["seq"] for e in manager.events(run.run_id, after_seq=1)] == [2, 3]
    assert manager.events("run_missing") == []


def test_one_manager_per_client():
    client = FakeClient()
    assert runs_for(client) is runs_for(client)


def test_seq_keeps_counting_past_the_buffer_cap(monkeypatch):
    monkeypatch.setattr("lobster.core.runs._MAX_EVENTS_PER_RUN", 3)
    deltas = [{"type": "content_delta", "delta": str(i)} for i in range(5)]
    manager = RunManager(FakeClient(deltas + [COMPLETE]))
    run = manager.start("x")
    seqs = [e["seq"] for e in run.stream()]
    assert seqs == list(range(1, len(seqs) + 1))
    # The buffer keeps its first events plus every run_state (the terminal one).
    assert len(run.events) == 4
    assert (run.events[-1]["type"], run.events[-1]["state"]) == (
        "run_state",
        "completed",
    )


def test_terminal_state_is_kept_past_the_buffer_cap_on_cancel(monkeypatch):
    monkeypatch.setattr("lobster.core.runs._MAX_EVENTS_PER_RUN", 2)
    deltas = [{"type": "content_delta", "delta": str(i)} for i in range(5)]
    interrupt = {
        "type": "interrupt",
        "interrupt_id": "i1",
        "data": {"component": "confirm"},
    }
    manager = RunManager(FakeClient(deltas + [interrupt]))
    run = manager.start("x")
    list(run.stream())
    manager.cancel(run.run_id, reason="decision_abandoned")
    assert run.events[-1]["state"] == "cancelled"
    assert run.events[-1]["reason"] == "decision_abandoned"


def test_stream_that_raised_forwards_its_terminal_state_past_the_cap(monkeypatch):
    from lobster.cli_internal.go_tui_launcher import _forward_run_end

    monkeypatch.setattr("lobster.core.runs._MAX_EVENTS_PER_RUN", 2)

    def boom():
        for i in range(5):
            yield {"type": "content_delta", "delta": str(i)}
        raise ValueError("kaput")

    client = FakeClient()
    client.query = lambda prompt, stream=True, cancel_event=None: boom()
    manager = RunManager(client)
    run = manager.start("x")
    with pytest.raises(ValueError):
        list(run.stream())
    sent = []
    _forward_run_end(manager, run, sent.append, {})
    [terminal] = sent
    assert (terminal["type"], terminal["state"]) == ("run_state", "failed")


def test_raising_source_records_error_before_failed():
    def boom():
        yield {"type": "content_delta", "delta": "a"}
        raise ValueError("kaput")

    client = FakeClient()
    client.query = lambda prompt, stream=True, cancel_event=None: boom()
    run = RunManager(client).start("x")
    with pytest.raises(ValueError):
        list(run.stream())
    error, failed = run.events[-2:]
    assert (error["type"], error["code"], error["message"], error["retryable"]) == (
        "error",
        "exception",
        "kaput",
        False,
    )
    assert (failed["state"], failed["error"]["code"]) == ("failed", "exception")
    assert failed["seq"] == error["seq"] + 1


def test_start_uses_given_run_id():
    run_id = new_run_id()
    manager = RunManager(FakeClient([COMPLETE]))
    run = manager.start("hello", run_id=run_id)
    assert run.run_id == run_id
    assert manager.runs == {run_id: run}
    assert {e["run_id"] for e in run.stream()} == {run_id}


def test_start_rejects_duplicate_run_id():
    manager = RunManager(FakeClient([COMPLETE], [COMPLETE]))
    first = manager.start("a")
    with pytest.raises(ValueError, match="duplicate"):
        manager.start("b", run_id=first.run_id)
    assert manager.runs == {first.run_id: first}


@pytest.mark.parametrize("bad", ["abc", "run_XYZ", "", "run_" + "a" * 31])
def test_start_rejects_malformed_run_id(bad):
    manager = RunManager(FakeClient([COMPLETE]))
    with pytest.raises(ValueError, match="malformed"):
        manager.start("a", run_id=bad)
    assert manager.runs == {}


def test_default_run_id_is_minted_and_valid():
    run = RunManager(FakeClient([COMPLETE])).start("a")
    assert is_run_id(run.run_id)
    assert not is_run_id("abc")


def test_start_that_raises_ends_its_run_failed():
    from lobster.core.runs import new_run_id

    client = FakeClient()

    def refused(prompt, stream=True, cancel_event=None):
        raise RuntimeError("inference context already used")

    client.query = refused
    manager = RunManager(client)
    run_id = new_run_id()
    with pytest.raises(RuntimeError):
        manager.start("x", run_id=run_id)
    run = manager.runs[run_id]
    error, failed = run.events[-2:]
    assert (error["type"], error["code"]) == ("error", "exception")
    assert (failed["state"], failed["reason"]) == ("failed", "exception")
    # The id names that one failed run; it is never silently reused.
    with pytest.raises(ValueError, match="duplicate"):
        manager.start("x", run_id=run_id)


# --- files written during a run ------------------------------------------------


class FilesDataManager:
    """The slice of DataManagerV2 that ``build_result`` reads."""

    def __init__(self, files=(), error=None):
        self.files, self.error, self.since = list(files), error, []

    def list_modalities(self):
        return []

    def files_written_since(self, since=None, **kwargs):
        self.since.append(since)
        if self.error is not None:
            raise self.error
        return self.files


def with_files(client, data_manager):
    client.data_manager = data_manager
    return client


FILE = {
    "path": "/ws/plots/a.png",
    "relative": "plots/a.png",
    "category": "plots",
    "size_bytes": 3,
    "modified": "2026-11-13T10:00:00+00:00",
}


def test_completed_result_lists_the_files_written_since_the_run_started():
    from datetime import datetime, timezone

    dm = FilesDataManager([FILE])
    manager = RunManager(with_files(FakeClient([COMPLETE]), dm), report_files=True)
    before = datetime.now(timezone.utc)
    run = manager.start("go")
    list(run.stream())
    after = datetime.now(timezone.utc)
    assert run.result["files_written"] == [FILE]
    assert "files_written_error" not in run.result
    (since,) = dm.since
    assert since.tzinfo is not None and before <= since <= after


def test_a_manager_that_was_not_asked_adds_no_file_fields():
    """agent --stdio and query --json results (pinned by golden recordings)."""
    dm = FilesDataManager([FILE])
    manager = RunManager(with_files(FakeClient([COMPLETE]), dm))
    run = manager.start("go")
    list(run.stream())
    assert not {"files_written", "files_written_error"} & set(run.result)
    assert dm.since == []


def test_a_resumed_run_counts_files_from_the_origin_runs_start():
    dm = FilesDataManager([FILE])
    client = with_files(FakeClient([INTERRUPT], [COMPLETE]), dm)
    manager = RunManager(client, report_files=True)
    first = manager.start("qc")
    events = list(first.stream())
    decision = next(e for e in events if e["type"] == "decision_required")
    second = manager.answer(
        decision["decision_id"], {"confirmed": True}, source="human"
    )
    list(second.stream())
    assert second.started_at == first.started_at
    assert dm.since == [first.started_at]
    assert second.result["files_written"] == [FILE]


def test_a_storage_error_is_reported_and_does_not_fail_the_run():
    from lobster.core.figures.errors import StorageError

    dm = FilesDataManager(error=StorageError("disk full at /secret/place"))
    manager = RunManager(with_files(FakeClient([COMPLETE]), dm), report_files=True)
    run = manager.start("go")
    list(run.stream())
    assert run.state == "completed"
    assert run.result["files_written"] is None
    error = run.result["files_written_error"]
    assert error and "\n" not in error and "/secret" not in error


def test_an_unreadable_workspace_is_reported_too():
    dm = FilesDataManager(error=PermissionError("denied /secret/place"))
    manager = RunManager(with_files(FakeClient([COMPLETE]), dm), report_files=True)
    run = manager.start("go")
    list(run.stream())
    assert run.state == "completed" and run.result["files_written"] is None
    assert "/secret" not in run.result["files_written_error"]


@pytest.mark.parametrize("data_manager", [None, object(), FilesDataManager()])
def test_no_usable_file_source_adds_no_file_fields(data_manager):
    class NotAList(FilesDataManager):
        def files_written_since(self, since=None, **kwargs):
            return "not a list"

    if data_manager is not None and hasattr(data_manager, "files"):
        data_manager = NotAList()
    manager = RunManager(
        with_files(FakeClient([COMPLETE]), data_manager), report_files=True
    )
    run = manager.start("go")
    list(run.stream())
    assert not {"files_written", "files_written_error"} & set(run.result)


def test_result_schema_describes_the_file_fields_and_requires_nothing_new():
    import json
    from pathlib import Path

    from jsonschema import Draft202012Validator, ValidationError

    path = Path(__file__).resolve().parents[3] / "lobster/schemas"
    schema = json.loads((path / "lobster.result.v1.schema.json").read_text())
    assert schema["required"] == [
        "schema_version",
        "engine",
        "session_id",
        "run_id",
        "response",
    ]
    base = {
        "schema_version": "lobster.result/1",
        "engine": {"name": "lobster-ai", "version": "1"},
        "session_id": None,
        "run_id": "run_x",
        "response": "",
    }
    validator = Draft202012Validator(schema)
    validator.validate(base)
    validator.validate({**base, "files_written": [{**FILE, "figure_ref": "f@r"}]})
    validator.validate({**base, "files_written": [{**FILE, "modified": None}]})
    validator.validate({**base, "files_written": None, "files_written_error": "x"})
    for bad in (
        {"files_written": "a.png"},
        {"files_written": [{**FILE, "size_bytes": "3"}]},
        {"files_written": [{"path": "/x"}]},
        {"files_written_error": 3},
    ):
        with pytest.raises(ValidationError):
            validator.validate({**base, **bad})
