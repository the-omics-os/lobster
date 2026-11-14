"""A paused ``lobster agent --stdio`` run waits for its decision without a timer.

The stdio session runs in-process on the ``decision`` replay script with the
launcher's decision timeout shortened, so the test proves in a second what the
real 300 s would do. The Go TUI path keeps its timeout (``lobster chat``).
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time

import pytest

from lobster.cli_internal import go_tui_launcher as launcher
from lobster.cli_internal.agent_stdio import StdioBridge
from tests.equivalence.launcher_harness import RecordingBridge, make_client

SHORT_TIMEOUT_S = 0.2


class _LineSink:
    """Text stream whose written lines become parsed messages on a queue."""

    def __init__(self):
        self.messages: "queue.Queue[dict]" = queue.Queue()
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                self.messages.put(json.loads(line))
        return len(text)

    def flush(self) -> None:
        pass


class _StdioSession:
    def __init__(self, client):
        read_fd, write_fd = os.pipe()
        self._stdin = os.fdopen(write_fd, "w", encoding="utf-8", buffering=1)
        self.client = client
        self.out = _LineSink()
        self.seen: list = []
        bridge = StdioBridge(self.out, os.fdopen(read_fd, "r", encoding="utf-8"))
        bridge.start()
        self.loop = threading.Thread(
            target=launcher._go_tui_event_loop, args=(bridge, client), daemon=True
        )
        self.loop.start()

    def write(self, msg_type: str, payload: dict) -> None:
        self._stdin.write(json.dumps({"type": msg_type, "payload": payload}) + "\n")

    def close_stdin(self) -> None:
        self._stdin.close()

    def until(self, predicate, timeout: float = 60.0) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            message = self.out.messages.get(timeout=deadline - time.monotonic())
            self.seen.append(message)
            if predicate(message):
                return message

    def quiet_for(self, seconds: float) -> list:
        """Messages that arrive within *seconds*."""
        arrived = []
        deadline = time.monotonic() + seconds
        while (left := deadline - time.monotonic()) > 0:
            try:
                arrived.append(self.out.messages.get(timeout=left))
            except queue.Empty:
                break
        self.seen.extend(arrived)
        return arrived


def _is_state(state):
    return lambda m: m["type"] == "run_state" and m["payload"]["state"] == state


@pytest.fixture
def paused(tmp_path, monkeypatch):
    """A stdio session whose run is paused on the replay's decision."""
    monkeypatch.setattr(launcher, "DECISION_TIMEOUT_S", SHORT_TIMEOUT_S)
    client = make_client(tmp_path / "workspace", monkeypatch, "decision")
    session = _StdioSession(client)
    session.write("input", {"content": "Run QC on my pbmc3k data."})
    render = session.until(lambda m: m["type"] == "component_render")
    yield session, render
    if session.loop.is_alive():
        session.close_stdin()
        session.loop.join(timeout=30)


def test_tui_path_still_cancels_a_paused_run_after_the_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "DECISION_TIMEOUT_S", SHORT_TIMEOUT_S)
    client = make_client(tmp_path / "workspace", monkeypatch, "decision")
    bridge = RecordingBridge(answers=[])
    # RecordingBridge answers with an immediate cancel; leave the prompt open.
    bridge.send = lambda msg_type, payload=None, *, msg_id=None: bridge.sent.append(
        {"type": msg_type, "payload": payload or {}}
    )

    started = time.monotonic()
    launcher._handle_user_query(bridge, client, "Run QC on my pbmc3k data.")

    assert time.monotonic() - started < 30
    assert bridge.sent[-1] == {"type": "done", "payload": {"summary": "cancelled"}}


def test_paused_stdio_run_has_no_timer_and_still_takes_the_answer(paused):
    session, render = paused
    arrived = session.quiet_for(SHORT_TIMEOUT_S * 5)
    assert arrived == [], "a paused stdio run must keep waiting for its decision"
    assert [
        m["payload"]["state"] for m in session.seen if m["type"] == "run_state"
    ] == ["running", "needs_decision"]

    session.write(
        "decision_answer",
        {
            "decision_id": render["id"],
            "request_version": 1,
            "answer": {"threshold": 0.1},
            "source": "human",
        },
    )
    completed = session.until(
        lambda m: _is_state("completed")(m) and "result" in m["payload"]
    )
    assert completed["payload"]["result"]["schema_version"] == "lobster.result/1"


def test_answer_sends_the_origin_runs_completed_state(paused):
    session, render = paused
    origin_run_id = render["run_id"]
    session.write(
        "decision_answer",
        {
            "decision_id": render["id"],
            "request_version": 1,
            "answer": {"threshold": 0.1},
            "source": "human",
        },
    )
    origin_done = session.until(_is_state("completed"))
    running = session.until(_is_state("running"))
    assert origin_done["run_id"] == origin_run_id
    assert origin_done["payload"]["run_id"] == origin_run_id
    assert origin_done["payload"]["resumed_by"] == running["payload"]["run_id"]
    assert running["payload"]["run_id"] != origin_run_id


def test_cancel_ends_a_paused_stdio_run(paused):
    session, _ = paused
    session.quiet_for(SHORT_TIMEOUT_S * 2)
    session.write("cancel", {})
    session.until(_is_state("cancelled"), timeout=10)
    session.until(lambda m: m["type"] == "done")
    session.close_stdin()
    session.loop.join(timeout=10)
    assert not session.loop.is_alive()


def test_stdin_eof_ends_a_paused_stdio_session(paused):
    session, _ = paused
    session.quiet_for(SHORT_TIMEOUT_S * 2)
    session.close_stdin()
    # The event loop returning is what makes ``run_agent_stdio`` exit 0.
    session.loop.join(timeout=10)
    assert not session.loop.is_alive()
    session.until(_is_state("cancelled"), timeout=1)


# --- Frozen decision contract ---------------------------------


def _answer(render, answer, **overrides):
    payload = {
        "decision_id": render["id"],
        "request_version": 1,
        "answer": answer,
        "source": "human",
    }
    payload.update(overrides)
    return {k: v for k, v in payload.items() if v is not _DROP}


_DROP = object()


def _alert(code):
    return lambda m: m["type"] == "alert" and m["payload"].get("code") == code


def _still_waiting(session, render):
    """The refused answer changed nothing: the real answer still completes."""
    from lobster.core.runs import runs_for

    decision = runs_for(session.client).decisions[render["id"]]
    assert decision.state == "pending"
    session.write("decision_answer", _answer(render, {"threshold": 0.1}))
    session.until(lambda m: _is_state("completed")(m) and "result" in m["payload"])


def test_needs_decision_carries_the_open_question(paused):
    session, render = paused
    waiting = next(m for m in session.seen if _is_state("needs_decision")(m))
    decision = waiting["payload"]["decision"]
    assert decision["decision_id"] == render["id"]
    assert decision["request_version"] == 1
    assert decision["question"]
    [component] = decision["components"]
    assert component["n"] == 1
    assert component["component"] == render["payload"]["component"]
    assert component["fallback_prompt"]


@pytest.mark.parametrize(
    "overrides, code",
    [
        ({"decision_id": "dec_" + "0" * 32}, "unknown_decision"),
        ({"request_version": 2}, "rejected_stale"),
        ({"source": "robot"}, "bad_answer"),
        ({"source": _DROP}, "bad_answer"),
        ({"note": "extra"}, "bad_answer"),
        ({"comment": 3}, "bad_answer"),
        ({"request_version": "1"}, "bad_answer"),
    ],
)
def test_refused_answer_leaves_the_run_waiting(paused, overrides, code):
    session, render = paused
    session.write("decision_answer", _answer(render, {"threshold": 0.1}, **overrides))
    alert = session.until(_alert(code), timeout=10)
    assert alert["payload"]["level"] == "error"
    _still_waiting(session, render)


def test_second_answer_is_already_answered_with_resumed_by(paused):
    session, render = paused
    session.write("decision_answer", _answer(render, {"threshold": 0.1}))
    running = session.until(_is_state("running"))
    session.until(lambda m: _is_state("completed")(m) and "result" in m["payload"])
    session.write("decision_answer", _answer(render, {"threshold": 0.2}))
    alert = session.until(_alert("already_answered"), timeout=10)
    assert alert["payload"]["resumed_by"] == running["payload"]["run_id"]


def test_model_answer_and_plain_text_answer_are_accepted(paused):
    session, render = paused
    session.write(
        "decision_answer",
        _answer(render, "0.1 please", source="model", comment="from the notes"),
    )
    session.until(lambda m: _is_state("completed")(m) and "result" in m["payload"])
    from lobster.core.runs import runs_for

    decision = runs_for(session.client).decisions[render["id"]]
    assert decision.state == "answered"


@pytest.mark.parametrize("how", ["cancel", "eof"])
def test_unanswered_decision_is_abandoned_and_a_late_answer_refused(paused, how):
    from lobster.core.runs import runs_for

    session, render = paused
    if how == "cancel":
        session.write("cancel", {})
    else:
        session.close_stdin()
    cancelled = session.until(_is_state("cancelled"), timeout=10)
    assert cancelled["payload"]["reason"] == "decision_abandoned"
    assert runs_for(session.client).decisions[render["id"]].state == "cancelled"
    if how == "cancel":
        session.until(lambda m: m["type"] == "done")
        session.write("decision_answer", _answer(render, {"threshold": 0.1}))
        session.until(_alert("rejected_cancelled"), timeout=10)


def test_interrupt_id_never_reaches_the_wire(paused):
    session, render = paused
    session.write("decision_answer", _answer(render, {"threshold": 0.1}))
    session.until(lambda m: _is_state("completed")(m) and "result" in m["payload"])
    wire = json.dumps(session.seen)
    assert "interrupt_id" not in wire
    assert "ask_id" not in wire
    assert "consumer_closed" not in wire


def test_tui_timeout_cancels_with_decision_abandoned(tmp_path, monkeypatch):
    from lobster.core.runs import runs_for

    monkeypatch.setattr(launcher, "DECISION_TIMEOUT_S", SHORT_TIMEOUT_S)
    client = make_client(tmp_path / "workspace", monkeypatch, "decision")
    bridge = RecordingBridge(answers=[])
    bridge.send = lambda msg_type, payload=None, *, msg_id=None: bridge.sent.append(
        {"type": msg_type, "payload": payload or {}}
    )
    launcher._handle_user_query(bridge, client, "Run QC on my pbmc3k data.")
    [run] = runs_for(client).runs.values()
    assert run.state == "cancelled"
    assert run.events[-1]["reason"] == "decision_abandoned"


def test_render_error_cancels_with_decision_unsupported(tmp_path, monkeypatch):
    from lobster.core.runs import runs_for

    client = make_client(tmp_path / "workspace", monkeypatch, "decision")
    bridge = RecordingBridge(answers=[{"action": "error"}])
    launcher._handle_user_query(bridge, client, "Run QC on my pbmc3k data.")
    [run] = runs_for(client).runs.values()
    assert run.events[-1]["reason"] == "decision_unsupported"
    # The Go TUI wire is unchanged from before the freeze: no new alert.
    assert not any(m["type"] == "alert" for m in bridge.sent)
    assert bridge.sent[-1] == {"type": "done", "payload": {"summary": "cancelled"}}


def test_stdio_render_error_says_decision_unsupported(paused):
    session, render = paused
    session._stdin.write(
        json.dumps(
            {
                "type": "component_response",
                "id": render["id"],
                "payload": {"data": {"action": "error"}},
            }
        )
        + "\n"
    )
    session.until(_alert("decision_unsupported"), timeout=10)
    cancelled = session.until(_is_state("cancelled"), timeout=10)
    assert cancelled["payload"]["reason"] == "decision_unsupported"


# --- Every run ends with exactly one terminal run_state -------------------------
# A host keys on it, not on done. It comes before done for cancelled and
# failed, and after the final done for completed (carrying result).


def _terminal_states(seen):
    return [
        m
        for m in seen
        if m["type"] == "run_state"
        and m["payload"]["state"] in ("completed", "failed", "cancelled")
    ]


def test_completed_run_state_comes_after_the_final_done(tmp_path, monkeypatch):
    client = make_client(tmp_path / "workspace", monkeypatch, "query")
    session = _StdioSession(client)
    session.write("input", {"content": "What can you do with single-cell data?"})
    completed = session.until(_is_state("completed"))
    [terminal] = _terminal_states(session.seen)
    assert terminal is completed
    assert "result" in completed["payload"]
    types = [m["type"] for m in session.seen]
    assert types.count("done") == 1
    assert types.index("done") < len(types) - 1
    assert session.seen[-1] is completed
    assert session.quiet_for(0.5) == []


def test_launcher_error_ends_the_run_failed_before_done(tmp_path, monkeypatch):
    from lobster.core.runs import runs_for

    client = make_client(tmp_path / "workspace", monkeypatch, "query")

    def broken(*args, **kwargs):
        raise RuntimeError("launcher bug")

    monkeypatch.setattr(launcher, "_forward_stream_event", broken)
    session = _StdioSession(client)
    session.write("input", {"content": "What can you do with single-cell data?"})
    session.until(lambda m: m["type"] == "done")
    [terminal] = _terminal_states(session.seen)
    assert terminal["payload"]["state"] == "failed"
    assert terminal["payload"]["reason"] == "exception"
    assert terminal["payload"]["error"]["code"] == "exception"
    assert session.seen[-1]["payload"]["summary"] == "error"
    [run] = runs_for(client).runs.values()
    assert run.state == "failed"


def test_interrupt_while_waiting_ends_the_run_cancelled_before_done(
    tmp_path, monkeypatch
):
    from lobster.core.runs import runs_for

    client = make_client(tmp_path / "workspace", monkeypatch, "decision")

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(launcher, "_handle_interrupt", interrupted)
    session = _StdioSession(client)
    session.write("input", {"content": "Run QC on my pbmc3k data."})
    session.until(
        lambda m: m["type"] == "done" and m["payload"]["summary"] != "interrupt"
    )
    [terminal] = _terminal_states(session.seen)
    assert terminal["payload"]["state"] == "cancelled"
    assert terminal["payload"]["reason"] == "cancel"
    assert session.seen[-1]["payload"]["summary"] == "cancelled"
    [decision] = runs_for(client).decisions.values()
    assert decision.state == "cancelled"
    session.write(
        "decision_answer",
        {
            "decision_id": decision.decision_id,
            "request_version": 1,
            "answer": 0.1,
            "source": "human",
        },
    )
    session.until(_alert("rejected_cancelled"), timeout=10)


def test_stream_error_is_forwarded_once(tmp_path, monkeypatch):
    client = make_client(tmp_path / "workspace", monkeypatch, "query")

    def failing(*args, **kwargs):
        raise RuntimeError("engine bug")
        yield  # pragma: no cover

    monkeypatch.setattr(client, "query", failing)
    session = _StdioSession(client)
    session.write("input", {"content": "What can you do with single-cell data?"})
    session.until(lambda m: m["type"] == "done")
    [terminal] = _terminal_states(session.seen)
    assert (terminal["payload"]["state"], terminal["payload"]["reason"]) == (
        "failed",
        "exception",
    )


def test_cancel_as_the_question_arrives_ends_the_run_cancelled(tmp_path, monkeypatch):
    from pathlib import Path

    from lobster.core.runs import RunManager, runs_for
    from scripts.validate_ndjson import load_validator

    register = RunManager._register_decision

    def register_then_cancel(self, run, event):
        decision = register(self, run, event)
        run.cancel_event.set()  # the launcher sees it at run_state needs_decision
        return decision

    monkeypatch.setattr(RunManager, "_register_decision", register_then_cancel)
    client = make_client(tmp_path / "workspace", monkeypatch, "decision")
    session = _StdioSession(client)
    session.write("input", {"content": "Run QC on my pbmc3k data."})
    session.until(lambda m: m["type"] == "done")

    schemas = Path(__file__).resolve().parents[2] / "lobster" / "schemas"
    validator = load_validator(schemas / "protocol-v1.schema.json")
    for message in session.seen:
        assert not list(validator.iter_errors(message)), message
    states = [m["payload"] for m in session.seen if m["type"] == "run_state"]
    assert all("state" in s for s in states), states
    [terminal] = _terminal_states(session.seen)
    assert (terminal["payload"]["state"], terminal["payload"]["reason"]) == (
        "cancelled",
        "cancel",
    )
    [decision] = runs_for(client).decisions.values()
    assert decision.state == "cancelled"
    session.write(
        "decision_answer",
        {
            "decision_id": decision.decision_id,
            "request_version": 1,
            "answer": 0.1,
            "source": "human",
        },
    )
    session.until(_alert("rejected_cancelled"), timeout=10)
