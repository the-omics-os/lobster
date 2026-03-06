#!/usr/bin/env python3
"""Deterministic smoke harness for the Go TUI protocol event-loop dispatch."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _bootstrap_repo_import() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_bootstrap_repo_import()

from lobster.cli_internal import go_tui_launcher  # noqa: E402


class _BridgeSequence:
    def __init__(self, events: List[Dict[str, Any]]) -> None:
        self._events = list(events)
        self.sent: List[Tuple[str, Dict[str, Any], str]] = []

    def recv_event(self, timeout: float | None = None) -> Dict[str, Any] | None:
        del timeout
        if self._events:
            return self._events.pop(0)
        return None

    def send(self, msg_type: str, payload: Dict[str, Any] | None = None, msg_id: str = "") -> None:
        self.sent.append((msg_type, payload or {}, msg_id))


class _FakeClient:
    def __init__(self) -> None:
        self.data_manager = type("DM", (), {"available_datasets": {"rna_reference": {}}})()


def _run_smoke() -> None:
    observed: Dict[str, List[Any]] = {"input": [], "slash": []}

    def _fake_user_query(_bridge: _BridgeSequence, _client: _FakeClient, text: str) -> None:
        observed["input"].append(text)

    def _fake_slash(_bridge: _BridgeSequence, _client: _FakeClient, cmd: str, args: str) -> None:
        observed["slash"].append((cmd, args))

    original_user_query = go_tui_launcher._handle_user_query
    original_slash_command = go_tui_launcher._handle_slash_command
    original_cwd = Path.cwd()

    try:
        go_tui_launcher._handle_user_query = _fake_user_query
        go_tui_launcher._handle_slash_command = _fake_slash

        with tempfile.TemporaryDirectory(prefix="go-tui-protocol-smoke-") as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "readme.txt").write_text("hello", encoding="utf-8")
            os.chdir(tmp_path)

            bridge = _BridgeSequence(
                [
                    {"type": "input", "payload": {"content": "hello"}},
                    {"type": "slash_command", "payload": {"command": "status", "args": ""}},
                    {
                        "type": "completion_request",
                        "id": "comp-1",
                        "payload": {"command": "/read", "prefix": "rea"},
                    },
                    {"type": "quit"},
                ]
            )
            client = _FakeClient()

            go_tui_launcher._go_tui_event_loop(bridge, client)

        assert observed["input"] == ["hello"], f"unexpected input dispatch: {observed['input']!r}"
        assert observed["slash"] == [("status", "")], (
            f"unexpected slash dispatch: {observed['slash']!r}"
        )

        completion_messages = [msg for msg in bridge.sent if msg[0] == "completion_response"]
        assert len(completion_messages) == 1, f"unexpected completion messages: {bridge.sent!r}"
        _, payload, msg_id = completion_messages[0]
        assert msg_id == "comp-1", f"unexpected completion msg id: {msg_id!r}"
        suggestions = payload.get("suggestions", [])
        assert "/read readme.txt" in suggestions, f"unexpected suggestions: {suggestions!r}"
    finally:
        os.chdir(original_cwd)
        go_tui_launcher._handle_user_query = original_user_query
        go_tui_launcher._handle_slash_command = original_slash_command


def main() -> int:
    try:
        _run_smoke()
    except Exception as exc:
        print(f"GO_TUI_PROTOCOL_SMOKE_FAIL: {exc}", file=sys.stderr)
        return 1

    print("GO_TUI_PROTOCOL_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
