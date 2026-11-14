"""A ``lobster agent --stdio`` cancel is never dropped, and stdin EOF cancels a run.

In-process like ``test_stdio_decision_wait``: ``StdioBridge`` on an ``os.pipe``
stdin + ``_go_tui_event_loop`` on a replay script. The Go TUI path
(``lobster chat``) keeps its behaviour; ``tests/equivalence`` proves the bytes.
"""

from __future__ import annotations

import json
import os
import threading
import time

from lobster.cli_internal import go_tui_launcher as launcher
from lobster.cli_internal.agent_stdio import StdioBridge
from tests.equivalence.launcher_harness import load_scenario, make_client
from tests.protocol.test_stdio_decision_wait import _is_state, _LineSink, _StdioSession

# cancel.json's only model call sleeps this long; nothing interrupts it, so a
# cancel takes effect at the next graph event after it returns.
REPLAY_DELAY_S = load_scenario("cancel")["responses"][0]["delay_s"]
EOF_CANCEL_BOUND_S = REPLAY_DELAY_S + 1.0


def _drain(session) -> list:
    session.quiet_for(0.2)
    return session.seen


def test_cancel_right_after_input_is_never_dropped(tmp_path, monkeypatch):
    client = make_client(tmp_path / "workspace", monkeypatch, "query")
    read_fd, write_fd = os.pipe()
    stdin = os.fdopen(write_fd, "w", encoding="utf-8", buffering=1)
    out = _LineSink()
    bridge = StdioBridge(out, os.fdopen(read_fd, "r", encoding="utf-8"))
    bridge.cancel_event = threading.Event()  # what the loop installs
    bridge.start()
    # The reader takes both lines before the event loop sees the input: the
    # race a host hits when it cancels right after sending.
    stdin.write(json.dumps({"type": "input", "payload": {"content": "Hi"}}) + "\n")
    stdin.write(json.dumps({"type": "cancel", "payload": {}}) + "\n")
    deadline = time.monotonic() + 5
    while bridge._events.qsize() < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert bridge._events.qsize() == 2

    session = _StdioSession.__new__(_StdioSession)
    session._stdin, session.out, session.seen = stdin, out, []
    session.loop = threading.Thread(
        target=launcher._go_tui_event_loop, args=(bridge, client), daemon=True
    )
    session.loop.start()
    try:
        done = session.until(lambda m: m["type"] == "done", timeout=30)
        assert done["payload"]["summary"] == "cancelled"
        states = [
            m["payload"]["state"] for m in _drain(session) if m["type"] == "run_state"
        ]
        assert "completed" not in states
        assert states[-1] == "cancelled"
    finally:
        session.close_stdin()
        session.loop.join(timeout=30)
    assert not session.loop.is_alive()


def test_stdin_eof_during_a_run_cancels_it_then_exits(tmp_path, monkeypatch):
    client = make_client(tmp_path / "workspace", monkeypatch, "cancel")
    session = _StdioSession(client)
    session.write("input", {"content": load_scenario("cancel")["prompt"]})
    session.until(_is_state("running"))

    started = time.monotonic()
    session.close_stdin()
    session.loop.join(timeout=30)
    elapsed = time.monotonic() - started
    assert not session.loop.is_alive()

    seen = _drain(session)
    states = [m["payload"]["state"] for m in seen if m["type"] == "run_state"]
    assert states == ["running", "cancelled"]
    assert any(
        m["type"] == "done" and m["payload"]["summary"] == "cancelled" for m in seen
    )
    print(f"EOF -> loop returned in {elapsed:.2f}s (replay delay {REPLAY_DELAY_S}s)")
    assert elapsed < EOF_CANCEL_BOUND_S


def test_tui_bridge_keeps_its_cancel_and_eof_behaviour():
    # Both stdio fixes hang on StdioBridge attributes; the Go TUI bridge has
    # neither, so lobster chat drops nothing new and still ignores EOF mid-run.
    assert StdioBridge.cancel_on_eof is True
    assert not hasattr(launcher._LightBridge, "cancel_on_eof")
    assert not hasattr(launcher._LightBridge, "cancelled_before_start")
