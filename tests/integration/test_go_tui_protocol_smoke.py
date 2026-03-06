import os

import pytest

from lobster.cli_internal import go_tui_launcher


pytestmark = pytest.mark.integration


class _BridgeSequence:
    def __init__(self, events):
        self._events = list(events)
        self.sent = []

    def recv_event(self, timeout=None):
        if self._events:
            return self._events.pop(0)
        return None

    def send(self, msg_type, payload=None, msg_id=""):
        self.sent.append((msg_type, payload or {}, msg_id))


def test_go_tui_event_loop_smoke_dispatches_core_event_types(tmp_path, monkeypatch):
    (tmp_path / "readme.txt").write_text("hello")
    monkeypatch.chdir(tmp_path)

    observed = {"input": [], "slash": []}

    def _fake_user_query(_bridge, _client, text):
        observed["input"].append(text)

    def _fake_slash(_bridge, _client, cmd, args):
        observed["slash"].append((cmd, args))

    monkeypatch.setattr(go_tui_launcher, "_handle_user_query", _fake_user_query)
    monkeypatch.setattr(go_tui_launcher, "_handle_slash_command", _fake_slash)

    client = type(
        "Client",
        (),
        {
            "data_manager": type(
                "DM",
                (),
                {"available_datasets": {"rna_reference": {}}},
            )()
        },
    )()

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

    go_tui_launcher._go_tui_event_loop(bridge, client)

    assert observed["input"] == ["hello"]
    assert observed["slash"] == [("status", "")]

    completion_messages = [m for m in bridge.sent if m[0] == "completion_response"]
    assert len(completion_messages) == 1
    msg_type, payload, msg_id = completion_messages[0]
    assert msg_type == "completion_response"
    assert msg_id == "comp-1"
    assert any(s.startswith("/read ") for s in payload.get("suggestions", []))
