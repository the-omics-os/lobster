"""Drive the Go TUI launcher's turn handler against a replay script.

The harness replaces the Go process with an in-memory bridge that records
every protocol message Python sends and answers ``component_render`` with the
script's ``answers``. The recorded transcript is what the seam must preserve:
the Go TUI only ever sees these messages.
"""

from __future__ import annotations

import json
import queue
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "protocol-v1"
REPLAY_DIR = FIXTURE_DIR / "replay"
BASELINE_DIR = Path(__file__).resolve().parent / "baseline"
SCENARIOS = ("query", "decision", "cancel", "failure")

# Values that change on every run and say nothing about behaviour.
_VOLATILE_KEYS = {
    "duration",
    "duration_ms",
    "elapsed",
    "timestamp",
    "id",
    "tool_call_id",
}
_DIGITS = re.compile(r"\d+(\.\d+)?")


class RecordingBridge:
    """In-memory stand-in for ``_LightBridge``."""

    def __init__(self, answers: List[Dict[str, Any]]):
        self.sent: List[Dict[str, Any]] = []
        self.answered: List[Dict[str, Any]] = []
        self._answers = list(answers)
        self._inbound: "queue.Queue[dict]" = queue.Queue()
        self.cancel_event: Optional[threading.Event] = None

    def send(self, msg_type: str, payload: Optional[dict] = None, *, msg_id=None):
        message = {"type": msg_type, "payload": payload or {}}
        if msg_id is not None:
            message["id"] = msg_id  # volatile: normalize() drops it
        self.sent.append(message)
        if msg_type == "component_render":
            if self._answers:
                answer = self._answers.pop(0)
                response = {
                    "type": "component_response",
                    "id": msg_id,
                    "payload": {"data": answer},
                }
                self.answered.append(response)
                self._inbound.put(response)
            else:
                self._inbound.put({"type": "cancel", "payload": {}})

    def recv_event(self, *, timeout: Optional[float] = None) -> Optional[dict]:
        try:
            return self._inbound.get(timeout=timeout)
        except queue.Empty:
            return None


def load_scenario(name: str) -> Dict[str, Any]:
    return json.loads((REPLAY_DIR / f"{name}.json").read_text(encoding="utf-8"))


def make_client(workspace: Path, monkeypatch, scenario: str):
    """Build a local AgentClient whose models replay *scenario*."""
    from lobster.config.providers.replay_provider import (
        REPLAY_ENV,
        reset_replay_scripts,
    )

    monkeypatch.setenv(REPLAY_ENV, str(REPLAY_DIR / f"{scenario}.json"))
    reset_replay_scripts()
    from lobster.core.client import AgentClient

    return AgentClient(workspace_path=str(workspace), session_id=f"fixture-{scenario}")


def run_launcher_turn(client, scenario: str) -> List[Dict[str, Any]]:
    """Run one chat turn through the Go launcher path and return its transcript."""
    bridge, _ = drive_launcher_turn(client, scenario)
    return normalize(bridge.sent)


def drive_launcher_turn(client, scenario: str):
    """Run one chat turn; returns the bridge and the turn handler's return value."""
    from lobster.cli_internal import go_tui_launcher as launcher
    from lobster.ui.callbacks.protocol_callback import ProtocolCallbackHandler

    spec = load_scenario(scenario)
    bridge = RecordingBridge(spec.get("answers", []))
    client.callbacks.append(
        ProtocolCallbackHandler(
            emit_event=lambda msg_type, payload: bridge.send(
                msg_type,
                (
                    launcher._normalize_tool_payload(payload)
                    if msg_type == "tool_execution"
                    else payload
                ),
            )
        )
    )
    cancel_event = threading.Event()
    timer = None
    if spec.get("cancel_after_s"):
        timer = threading.Timer(float(spec["cancel_after_s"]), cancel_event.set)
        timer.start()
    try:
        returned = launcher._handle_user_query(
            bridge, client, spec["prompt"], cancel_event=cancel_event
        )
    finally:
        if timer is not None:
            timer.cancel()
    return bridge, returned


def normalize(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_scrub(message) for message in messages]


def _scrub(value: Any, key: str = "") -> Any:
    if isinstance(value, dict):
        return {k: _scrub(v, k) for k, v in value.items() if k not in _VOLATILE_KEYS}
    if isinstance(value, list):
        return [_scrub(v, key) for v in value]
    if key in ("text", "summary") and isinstance(value, str):
        # Status lines and tool summaries carry durations and token counts.
        return _DIGITS.sub("#", value)
    return value


def workspace_fingerprint(client) -> Dict[str, Any]:
    """Conversation the turn left behind; files are listed by name only."""
    messages = [
        {
            "role": type(m).__name__,
            "content": m.content if isinstance(m.content, str) else str(m.content),
        }
        for m in getattr(client, "messages", [])
    ]
    return {"messages": messages}


def workspace_files(workspace: Path) -> List[str]:
    """Relative names of every file and folder the turn left in *workspace*."""
    return sorted(p.relative_to(workspace).as_posix() for p in workspace.rglob("*"))


def baseline_path(scenario: str) -> Path:
    return BASELINE_DIR / f"{scenario}.launcher.json"
