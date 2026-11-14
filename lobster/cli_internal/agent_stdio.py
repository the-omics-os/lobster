"""``lobster agent --stdio``: Lobster's chat protocol over stdin and stdout.

A host program (a coding agent, the npm CLI, a test) launches this command and
speaks the same protocol-v1 messages the Go TUI speaks, as NDJSON:

* stdout carries only protocol messages. Each envelope adds ``seq`` (monotonic
  per process) and, while a run is active, ``run_id``. Lobster's own logs and
  any stray prints go to stderr.
* stdin accepts the Go TUI's inbound messages (``input``, ``slash_command``,
  ``component_response``, ``cancel``, ``quit``) plus ``decision_answer``
  ``{decision_id, request_version, answer, source, comment?}`` (frozen,
  Phase 0 A2; ``source`` is ``human`` or ``model``). A waiting run's
  ``run_state needs_decision`` carries ``decision`` (id, version, question,
  components) before its ``component_render``, whose ``id`` is the decision
  id. A refused answer is an ``alert`` with ``code`` (``unknown_decision``,
  ``already_answered``, ``rejected_cancelled``, ``rejected_stale``,
  ``bad_answer``) and leaves the run waiting. A cancelled run's ``reason`` is
  ``cancel``, ``decision_abandoned`` or ``decision_unsupported``.
* How the session ends, once a turn may be running:

  1. ``input``, ``quit``, EOF: the turn finishes normally (``text``,
     ``done``, then ``run_state completed``), then the process exits 0.
  2. ``input``, EOF with no ``quit``: the host is gone; the turn is cancelled
     (as ``cancel`` does), then the process exits 0.
  3. Once ``quit`` is read, a run that reaches ``needs_decision`` is cancelled
     at once (nobody can answer; stdio has no timer), reason
     ``decision_abandoned``, and the process exits 0. Nothing after ``quit``
     is read.
  4. ``cancel`` then ``quit``: cancellation is cooperative, so the run stops
     at its next step (a model call in flight is not interrupted), then the
     process exits 0.

  EOF or ``quit`` while idle exits 0 at once.

Exit codes: 0 after ``quit`` or EOF; the startup diagnostic's code when the
session cannot start; 1 on an unexpected failure.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
from typing import IO, Any, Dict, Optional

# The one place the engine spells its machine interface. Manifests carry it so
# hosts never hard-code the string.
AGENT_STDIO_COMMAND = "lobster agent --stdio"
PROTOCOL_VERSION = 1

_TERMINAL = {"completed", "failed", "cancelled"}
_EOF = {"type": "quit", "payload": {}}


class StdioBridge:
    """``_LightBridge`` interface over a pair of text streams."""

    forward_run_events = True
    # A paused run waits for its decision until ``cancel``, ``quit`` or EOF;
    # it dies with the process.
    decision_timeout = None
    # A host that closes stdin without ``quit`` is gone: EOF cancels its run
    # like ``cancel`` (inputs not started yet included), then the session ends.
    # After ``quit`` the run is left to finish.
    cancel_on_eof = True

    def __init__(self, out: IO[str], inp: IO[str]):
        self._out = out
        self._in = inp
        self._seq = 0
        self._run_id: Optional[str] = None
        self._write_lock = threading.Lock()
        self._events: "queue.Queue[dict]" = queue.Queue()
        self.cancel_event: Optional[threading.Event] = None
        # Pending-cancel latch: a cancel covers every input read before it,
        # including one the event loop has not started yet.
        self._latch = threading.Lock()
        self._inputs_read = 0
        self._inputs_taken = 0
        self._inputs_cancelled = 0
        self._reader = threading.Thread(
            target=self._read_loop, daemon=True, name="agent-stdio-reader"
        )

    def start(self) -> None:
        self._reader.start()

    def send(
        self,
        msg_type: str,
        payload: Optional[dict] = None,
        *,
        msg_id: Optional[str] = None,
    ) -> None:
        payload = payload or {}
        if msg_type == "run_state":
            self._run_id = payload.get("run_id")
        with self._write_lock:
            self._seq += 1
            envelope: Dict[str, Any] = {
                "type": msg_type,
                "version": PROTOCOL_VERSION,
                "seq": self._seq,
            }
            if self._run_id:
                envelope["run_id"] = self._run_id
            if msg_id is not None:
                envelope["id"] = msg_id
            envelope["payload"] = payload
            self._out.write(
                json.dumps(envelope, separators=(",", ":"), default=str) + "\n"
            )
            self._out.flush()
        if msg_type == "run_state" and payload.get("state") in _TERMINAL:
            self._run_id = None

    def recv_event(self, *, timeout: Optional[float] = None) -> Optional[dict]:
        try:
            event = self._events.get(timeout=timeout)
        except queue.Empty:
            return None
        if event.get("type") == "input":
            with self._latch:
                self._inputs_taken += 1
        if event.get("type") == "quit":
            # Every later reader must see the end too, not only the first: a
            # decision wait may consume the quit the event loop then needs.
            self._events.put(event)
        return event

    def cancelled_before_start(self) -> bool:
        """True when a cancel arrived for the input the event loop just took."""
        with self._latch:
            return self._inputs_taken <= self._inputs_cancelled

    def _cancel(self) -> None:
        # Latch first, then the event: the loop clears the event when it takes
        # an input and checks the latch after, so one of the two reaches it.
        with self._latch:
            self._inputs_cancelled = self._inputs_read
        if self.cancel_event is not None:
            self.cancel_event.set()

    def close(self) -> None:
        try:
            self._out.flush()
        except Exception:
            pass  # nosec B110 # stdout may already be closed by the host

    def _read_loop(self) -> None:
        for line in self._in:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self.send(
                    "alert",
                    {
                        "level": "error",
                        "message": "Invalid JSON on stdin",
                        "code": "bad_json",
                    },
                )
                continue
            if not isinstance(message, dict) or not message.get("type"):
                self.send(
                    "alert",
                    {
                        "level": "error",
                        "message": "Message needs a type",
                        "code": "bad_message",
                    },
                )
                continue
            if message["type"] == "input":
                with self._latch:
                    self._inputs_read += 1
            elif message["type"] == "cancel":
                # Cancel must reach a running turn even while the event loop
                # is busy inside it, and a turn it has not started yet.
                self._cancel()
            self._events.put(message)
            if message["type"] == "quit":
                # Nothing after quit is read, and EOF no longer cancels.
                self._events.put(_EOF)
                return
        if self.cancel_on_eof:
            self._cancel()
        self._events.put(_EOF)


def run_agent_stdio(
    *,
    workspace: Optional[str] = None,
    session_id: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    stdin: Optional[IO[str]] = None,
    stdout: Optional[IO[str]] = None,
) -> int:
    """Serve one session over stdio until ``quit`` or EOF; returns the exit code."""
    from lobster.version import __version__

    out = stdout or _claim_stdout()
    bridge = StdioBridge(out, stdin or sys.stdin)
    bridge.send(
        "handshake",
        {
            "protocol_version": PROTOCOL_VERSION,
            "engine": "lobster-ai",
            "engine_version": __version__,
            "interface": AGENT_STDIO_COMMAND,
        },
    )

    from lobster.cli_internal import go_tui_launcher as launcher
    from lobster.cli_internal.startup_diagnostics import StartupDiagnosticError

    client = None
    try:
        from lobster.ui.console_manager import get_console_manager

        get_console_manager().mute_terminal_output()
        client = launcher._open_session_client(
            bridge,
            workspace=workspace,
            session_id=session_id,
            provider=provider,
            model=model,
        )
        sid = getattr(client, "session_id", None)
        if sid:
            bridge.send("status", {"text": f"Session: {sid}"})
        bridge.send("ready", {"session_id": sid})
        bridge.start()
        launcher._go_tui_event_loop(bridge, client)
        return 0
    except StartupDiagnosticError as exc:
        launcher._emit_startup_diagnostic(bridge, exc.diagnostic)
        return int(exc.diagnostic.exit_code or 1)
    except Exception as exc:  # report, never print a traceback on stdout
        bridge.send(
            "alert", {"level": "error", "message": str(exc), "code": "internal"}
        )
        return 1
    finally:
        if client is not None:
            launcher._save_session_json_if_available(client)
        bridge.close()


def _claim_stdout() -> IO[str]:
    """Keep the real stdout for protocol messages; route fd 1 to stderr."""
    sys.stdout.flush()
    protocol_out = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return protocol_out
