"""The Go TUI and ``lobster agent --stdio`` say when none of the project's
experts is installed."""

from types import SimpleNamespace

import pytest

from lobster.cli_internal import go_tui_launcher

pytestmark = pytest.mark.unit


class _Bridge:
    def __init__(self):
        self.sent = []

    def send(self, kind, payload):
        self.sent.append((kind, payload))


def _open(monkeypatch, uninstalled):
    from lobster.cli_internal.commands.heavy import session_infra

    client = SimpleNamespace(
        callbacks=[],
        data_manager=None,
        graph_metadata=SimpleNamespace(uninstalled_experts=uninstalled),
    )
    monkeypatch.setattr(
        session_infra, "init_client_or_raise_startup_diagnostic", lambda **_: client
    )
    monkeypatch.setattr(session_infra, "set_go_tui_active", lambda active: None)
    monkeypatch.setattr(
        go_tui_launcher, "_resolve_go_chat_session_target", lambda *a: (None, None)
    )
    monkeypatch.setattr(
        go_tui_launcher, "_resolve_active_provider_name", lambda c: "anthropic"
    )
    bridge = _Bridge()
    go_tui_launcher._open_session_client(bridge)
    return [p for kind, p in bridge.sent if kind == "alert"]


def test_the_session_says_no_expert_of_the_project_is_installed(monkeypatch):
    assert _open(monkeypatch, ["gone_expert", "lost_expert"]) == [
        {
            "level": "warning",
            "message": "None of this project's experts (gone_expert, lost_expert) "
            "are installed; run lobster agents install <package> or "
            "lobster init --force.",
        }
    ]


def test_a_session_with_its_experts_says_nothing(monkeypatch):
    assert _open(monkeypatch, []) == []
