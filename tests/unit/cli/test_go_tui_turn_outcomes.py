"""Turn-outcome contract between the Python bridge and the Go TUI.

Every chat turn must end with exactly one terminal ``done`` message:
completed (``""``), cancelled/abandoned (``"cancelled"``) or failed
(``"error"``). A HITL interrupt only pauses the turn (``"interrupt"``) and is
not terminal. The Go TUI keys its turn lifecycle (and what it prints) off these
messages, so a missing terminal ``done`` leaves it stuck mid-turn.

The scenarios here also feed a golden fixture
(``lobster-tui/internal/chat/testdata/turn_outcomes.json``) that the Go tests
replay through the real TUI model, so the emitter and the renderer are checked
against the same recorded wire traffic.
"""

import json
import os
import threading
from pathlib import Path

import pytest

from lobster.cli_internal import go_tui_launcher

GOLDEN = (
    Path(__file__).resolve().parents[3]
    / "lobster-tui"
    / "internal"
    / "chat"
    / "testdata"
    / "turn_outcomes.json"
)

TERMINAL = {"", "cancelled", "error"}


class _Bridge:
    def __init__(self, events=None):
        self.calls = []
        self.events = list(events or [])
        self.cancel_event = None

    def send(self, msg_type, payload=None, msg_id=""):
        self.calls.append((msg_type, payload or {}, msg_id))

    def recv_event(self, timeout=None):
        return self.events.pop(0) if self.events else None

    def wire(self):
        return [{"type": t, "payload": p} for t, p, _ in self.calls]

    def dones(self):
        return [p.get("summary", "") for t, p, _ in self.calls if t == "done"]


class _Client:
    token_tracker = type("Tracker", (), {"total_tokens": 0, "total_cost": 0.0})()

    def __init__(self, first, resumed=None):
        self._first = first
        self._resumed = resumed

    def query(self, text, stream=True, cancel_event=None):
        yield from self._first(cancel_event)

    def resume_from_interrupt(self, response, stream=True, cancel_event=None):
        yield from self._resumed(cancel_event)


def _delta(text):
    return {"type": "content_delta", "delta": text}


_INTERRUPT = {
    "type": "interrupt",
    "data": {"component": "confirm", "data": {"question": "Proceed?"}},
    "interrupt_id": "i-1",
}
_CONFIRM = {"type": "confirm_response", "payload": {"confirm": True}}


def _complete_stream(cancel):
    yield _delta("ANSWER-ONE")
    yield {"type": "complete"}


def _error_after_partial(cancel):
    yield _delta("PARTIAL-ANSWER")
    yield {"type": "error", "error": "backend exploded"}


def _raises_mid_stream(cancel):
    yield _delta("PARTIAL-ANSWER")
    raise RuntimeError("stream broke")


def _silent_cancel(cancel):
    yield _delta("PARTIAL-ANSWER")
    cancel.set()  # the client stops silently after a cancellation


def _ends_without_complete(cancel):
    yield _delta("PARTIAL-ANSWER")


def _interrupt_stream(cancel):
    yield _delta("QUESTION-PART")
    yield _INTERRUPT


def _resumed_complete(cancel):
    yield _delta("RESUMED-PART")
    yield {"type": "complete"}


def _resumed_error(cancel):
    yield _delta("RESUMED-PART")
    yield {"type": "error", "error": "failed after resume"}


# name -> (client factory, bridge events, expected terminal outcome)
SCENARIOS = {
    "complete": (lambda: _Client(_complete_stream), [], ""),
    "error_after_partial": (lambda: _Client(_error_after_partial), [], "error"),
    "exception_mid_stream": (lambda: _Client(_raises_mid_stream), [], "error"),
    "silent_cancel": (lambda: _Client(_silent_cancel), [], "cancelled"),
    "ends_without_complete": (lambda: _Client(_ends_without_complete), [], "error"),
    "hitl_resume_complete": (
        lambda: _Client(_interrupt_stream, _resumed_complete),
        [_CONFIRM],
        "",
    ),
    "hitl_resume_error": (
        lambda: _Client(_interrupt_stream, _resumed_error),
        [_CONFIRM],
        "error",
    ),
    "hitl_abandoned": (lambda: _Client(_interrupt_stream), [], "cancelled"),
}


def _run(name):
    factory, events, _ = SCENARIOS[name]
    bridge = _Bridge(events)
    cancel = threading.Event()
    go_tui_launcher._handle_user_query(
        bridge, factory(), "question", cancel_event=cancel
    )
    return bridge


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_turn_settles_exactly_once(name):
    bridge = _run(name)
    dones = bridge.dones()
    terminal = [d for d in dones if d in TERMINAL]

    assert len(terminal) == 1, f"{name}: terminal done count {dones}"
    assert terminal[0] == SCENARIOS[name][2]
    # The terminal done is last; only HITL pauses may precede it.
    assert dones[-1] == terminal[0]
    assert all(d == "interrupt" for d in dones[:-1])


def test_failed_turn_reports_the_failure_before_settling():
    bridge = _run("error_after_partial")
    kinds = [t for t, _, _ in bridge.calls]

    assert "alert" in kinds
    assert kinds.index("alert") < kinds.index("done")


def test_unreported_early_end_gets_an_explicit_warning():
    bridge = _run("ends_without_complete")
    alerts = [p for t, p, _ in bridge.calls if t == "alert"]

    assert alerts and alerts[0]["level"] == "warning"
    assert bridge.dones() == ["error"]


def test_exception_after_complete_does_not_send_a_second_done(monkeypatch):
    """A failure while finishing up after ``complete`` must not re-settle."""

    def boom(client):
        raise RuntimeError("usage formatting failed")

    monkeypatch.setattr(go_tui_launcher, "_format_usage", boom)
    bridge = _run("complete")

    assert bridge.dones() == [""]


def test_keyboard_interrupt_settles_as_cancelled():
    def stream(cancel):
        yield _delta("PARTIAL-ANSWER")
        raise KeyboardInterrupt

    bridge = _Bridge()
    go_tui_launcher._handle_user_query(
        bridge, _Client(stream), "q", cancel_event=threading.Event()
    )

    assert bridge.dones() == ["cancelled"]


def test_cancel_after_complete_does_not_send_a_second_done():
    cancel = threading.Event()
    bridge = _Bridge()

    def stream(c):
        yield _delta("ANSWER-ONE")
        yield {"type": "complete"}
        cancel.set()

    go_tui_launcher._handle_user_query(
        bridge, _Client(stream), "q", cancel_event=cancel
    )

    assert bridge.dones() == [""]


def test_legacy_forwarder_without_turn_state_still_sends_done():
    """Direct callers of the forwarder keep the old one-shot behaviour."""
    bridge = _Bridge()
    go_tui_launcher._forward_stream_event(
        bridge, _Client(_complete_stream), {"type": "complete"}, 0.0
    )

    assert bridge.dones() == [""]


def _normalise(wire):
    """Drop wall-clock dependent status text; it is not part of the contract."""
    for item in wire:
        if item["type"] == "status":
            item["payload"] = {"text": "<status>"}
    return wire


def test_golden_wire_fixture_matches_the_emitter():
    """The Go replay fixture must be exactly what the emitter produces now."""
    produced = {name: _normalise(_run(name).wire()) for name in sorted(SCENARIOS)}

    if os.environ.get("LOBSTER_UPDATE_TURN_FIXTURE") == "1":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(produced, indent=2, sort_keys=True) + "\n")

    assert GOLDEN.exists(), "run once with LOBSTER_UPDATE_TURN_FIXTURE=1"
    recorded = {k: _normalise(v) for k, v in json.loads(GOLDEN.read_text()).items()}
    assert produced == recorded


# ---------------------------------------------------------------------------
# Swallowed best-effort failures
# ---------------------------------------------------------------------------


def test_note_suppressed_logs_site_and_type_only(caplog):
    secret = "AKIA-secret /Users/someone/patient_ids.csv"
    with caplog.at_level("DEBUG", logger=go_tui_launcher.logger.name):
        go_tui_launcher._note_suppressed("session save", RuntimeError(secret))

    assert [r.getMessage() for r in caplog.records] == [
        "session save failed: RuntimeError"
    ]
    assert all(r.exc_info is None for r in caplog.records)
    assert secret not in caplog.text


def test_note_suppressed_never_raises_when_logging_fails(monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("log sink gone")

    monkeypatch.setattr(go_tui_launcher.logger, "debug", broken)
    go_tui_launcher._note_suppressed("console restore", ValueError("x"))
