import json
import os
from pathlib import Path

import pytest

from lobster.cli_internal import go_tui_launcher
from lobster.cli_internal.startup_diagnostics import StartupDiagnostic
from lobster.ui.console_manager import get_console_manager


class _FakeBridge:
    def __init__(self, events=None):
        self.calls = []
        self.events = list(events or [])
        self.cancel_event = None

    def send(self, msg_type, payload=None, msg_id=""):
        self.calls.append((msg_type, payload or {}, msg_id))

    def recv_event(self, timeout=None):
        if self.events:
            return self.events.pop(0)
        return None


class _FakeClient:
    def __init__(self, available_datasets=None):
        self.data_manager = type(
            "DM",
            (),
            {"available_datasets": available_datasets or {}},
        )()


class _FakeLocalProviderClient:
    def __init__(self, workspace_path, provider_override=None):
        self.workspace_path = workspace_path
        self.provider_override = provider_override
        self.data_manager = object()


class _FakeCloudClient:
    def __init__(self, workspace_path, provider_override=None):
        self.workspace_path = workspace_path
        self.provider_override = provider_override


def test_python_terminal_quarantine_suppresses_stdout_and_stderr(capfd):
    quarantine = go_tui_launcher._PythonTerminalQuarantine.activate()
    try:
        os.write(1, b"hidden-stdout\\n")
        os.write(2, b"hidden-stderr\\n")
    finally:
        quarantine.restore()

    os.write(1, b"visible-stdout\\n")
    os.write(2, b"visible-stderr\\n")

    captured = capfd.readouterr()
    combined = captured.out + captured.err
    assert "hidden-stdout" not in combined
    assert "hidden-stderr" not in combined
    assert "visible-stdout" in captured.out
    assert "visible-stderr" in captured.err


def test_console_manager_terminal_mute_suppresses_rich_console_output(capfd):
    console_manager = get_console_manager()
    muted_here = console_manager.mute_terminal_output()
    try:
        console_manager.print("hidden-console")
        console_manager.print_error("hidden-error")
    finally:
        if muted_here:
            console_manager.restore_terminal_output()

    console_manager.print("visible-console")
    console_manager.print_error("visible-error")

    captured = capfd.readouterr()
    combined = captured.out + captured.err
    assert "hidden-console" not in combined
    assert "hidden-error" not in combined
    assert "visible-console" in captured.out
    assert "visible-error" in captured.err


def test_normalize_tool_payload_preserves_tool_call_identity():
    payload = go_tui_launcher._normalize_tool_payload(
        {
            "tool": "get_dataset_metadata",
            "status": "complete",
            "agent": "research_agent",
            "summary": "0.7s",
            "tool_call_id": "run-123",
        }
    )

    assert payload == {
        "tool_name": "get_dataset_metadata",
        "event": "finish",
        "agent": "research_agent",
        "summary": "0.7s",
        "tool_call_id": "run-123",
    }


def test_completion_request_read_prefixes_command(tmp_path, monkeypatch):
    (tmp_path / "data.csv").write_text("x")
    (tmp_path / "datasets").mkdir()
    monkeypatch.chdir(tmp_path)

    bridge = _FakeBridge()
    client = _FakeClient()
    event = {
        "id": "req-1",
        "payload": {"command": "/read", "prefix": "da"},
    }

    go_tui_launcher._handle_completion_request(bridge, client, event)

    assert bridge.calls
    msg_type, payload, msg_id = bridge.calls[-1]
    assert msg_type == "completion_response"
    assert msg_id == "req-1"
    assert "/read data.csv" in payload["suggestions"]
    assert "/read datasets/" in payload["suggestions"]


def test_completion_request_quotes_paths_with_spaces(tmp_path, monkeypatch):
    (tmp_path / "my data.csv").write_text("x")
    monkeypatch.chdir(tmp_path)

    bridge = _FakeBridge()
    client = _FakeClient()
    event = {
        "id": "req-2",
        "payload": {"command": "/read", "prefix": "my"},
    }

    go_tui_launcher._handle_completion_request(bridge, client, event)

    _, payload, _ = bridge.calls[-1]
    assert '/read "my data.csv"' in payload["suggestions"]


def test_completion_request_workspace_load_includes_dataset_names():
    bridge = _FakeBridge()
    client = _FakeClient(available_datasets={"rna_reference": {}, "proteomics_v1": {}})
    event = {
        "id": "req-3",
        "payload": {"command": "/workspace load", "prefix": "rn"},
    }

    go_tui_launcher._handle_completion_request(bridge, client, event)

    _, payload, _ = bridge.calls[-1]
    assert "/workspace load rna_reference" in payload["suggestions"]
    assert all(s.startswith("/workspace load ") for s in payload["suggestions"])


def test_resolve_active_provider_name_uses_config_resolver_for_local_client(
    tmp_path, monkeypatch
):
    class _FakeResolver:
        def __init__(self, workspace_path):
            self.workspace_path = Path(workspace_path)

        def resolve_provider(self, runtime_override=None):
            assert runtime_override is None
            assert self.workspace_path == tmp_path
            return "ollama", "workspace"

    monkeypatch.setattr(
        "lobster.core.config_resolver.ConfigResolver",
        _FakeResolver,
    )

    client = _FakeLocalProviderClient(tmp_path)

    assert go_tui_launcher._resolve_active_provider_name(client) == "ollama"


def test_resolve_active_provider_name_skips_workspace_resolution_for_cloud_like_client(
    tmp_path, monkeypatch
):
    class _ExplodingResolver:
        def __init__(self, workspace_path):
            raise AssertionError("resolver should not be used for cloud clients")

    monkeypatch.setattr(
        "lobster.core.config_resolver.ConfigResolver",
        _ExplodingResolver,
    )

    client = _FakeCloudClient(tmp_path)

    assert go_tui_launcher._resolve_active_provider_name(client) == ""


def test_handle_slash_command_refreshes_provider_status_after_execution(
    tmp_path, monkeypatch
):
    class _FakeResolver:
        def __init__(self, workspace_path):
            self.workspace_path = Path(workspace_path)

        def resolve_provider(self, runtime_override=None):
            assert self.workspace_path == tmp_path
            return "ollama", "runtime"

    class _StubProtocolOutputAdapter:
        def __init__(self, emit, **kwargs):
            self.emit = emit
            self.kwargs = kwargs

    executed = {}

    def _fake_execute_command(command, client, original_command, output):
        executed["command"] = command
        executed["original_command"] = original_command
        executed["client"] = client
        executed["output"] = output

    monkeypatch.setattr(
        "lobster.core.config_resolver.ConfigResolver",
        _FakeResolver,
    )
    monkeypatch.setattr(
        "lobster.cli_internal.commands.output_adapter.ProtocolOutputAdapter",
        _StubProtocolOutputAdapter,
    )
    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.slash_commands._execute_command",
        _fake_execute_command,
    )

    bridge = _FakeBridge()
    client = _FakeLocalProviderClient(tmp_path)

    go_tui_launcher._handle_slash_command(bridge, client, "/config", "provider ollama")

    assert executed["command"] == "/config provider ollama"
    assert executed["original_command"] == "/config provider ollama"
    assert executed["client"] is client
    assert isinstance(executed["output"], _StubProtocolOutputAdapter)
    assert ("status", {"text": "Provider: ollama"}, "") in bridge.calls


def test_handle_slash_command_writes_command_history_on_summary(tmp_path, monkeypatch):
    class _StubProtocolOutputAdapter:
        def __init__(self, emit, **kwargs):
            self.emit = emit
            self.kwargs = kwargs

    history = {}

    monkeypatch.setattr(
        go_tui_launcher,
        "_emit_provider_status",
        lambda bridge, client: None,
    )
    monkeypatch.setattr(
        "lobster.cli_internal.commands.output_adapter.ProtocolOutputAdapter",
        _StubProtocolOutputAdapter,
    )
    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.slash_commands._execute_command",
        lambda *args, **kwargs: "completed",
    )

    def _fake_add_to_history(client, command, summary, is_error=False):
        history["command"] = command
        history["summary"] = summary
        history["is_error"] = is_error

    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.session_infra._add_command_to_history",
        _fake_add_to_history,
    )

    bridge = _FakeBridge()
    client = _FakeLocalProviderClient(tmp_path)

    go_tui_launcher._handle_slash_command(bridge, client, "/status", "")

    assert history == {
        "command": "/status",
        "summary": "completed",
        "is_error": False,
    }


def test_handle_slash_command_writes_error_history_on_failure(tmp_path, monkeypatch):
    class _StubProtocolOutputAdapter:
        def __init__(self, emit, **kwargs):
            self.emit = emit
            self.kwargs = kwargs

    history = {}

    monkeypatch.setattr(
        go_tui_launcher,
        "_emit_provider_status",
        lambda bridge, client: None,
    )
    monkeypatch.setattr(
        "lobster.cli_internal.commands.output_adapter.ProtocolOutputAdapter",
        _StubProtocolOutputAdapter,
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.slash_commands._execute_command",
        _boom,
    )

    def _fake_add_to_history(client, command, summary, is_error=False):
        history["command"] = command
        history["summary"] = summary
        history["is_error"] = is_error

    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.session_infra._add_command_to_history",
        _fake_add_to_history,
    )

    bridge = _FakeBridge()
    client = _FakeLocalProviderClient(tmp_path)

    go_tui_launcher._handle_slash_command(bridge, client, "/status", "")

    assert history["command"] == "/status"
    assert history["is_error"] is True
    assert "RuntimeError: boom" in history["summary"]
    assert ("alert", {"level": "error", "message": "boom"}, "") in bridge.calls


def test_go_tui_event_loop_quit_saves_session():
    bridge = _FakeBridge(events=[{"type": "quit"}])
    saved = {}

    class _Client:
        def _save_session_json(self):
            saved["called"] = True

    go_tui_launcher._go_tui_event_loop(bridge, _Client())

    assert saved["called"] is True


def test_handle_user_query_maps_context_compaction_to_info_alert(monkeypatch):
    monkeypatch.setattr(go_tui_launcher.time, "time", lambda: 100.0)

    class _QueryClient:
        def __init__(self):
            self.token_tracker = type(
                "Tracker",
                (),
                {"total_tokens": 1234, "total_cost": 0.0123},
            )()

        def query(self, text, stream=True, cancel_event=None):
            assert text == "analyze"
            assert stream is True
            yield {
                "type": "context_compaction",
                "before_count": 48,
                "after_count": 21,
                "budget_tokens": 8192,
            }
            yield {"type": "content_delta", "delta": "done"}
            yield {"type": "complete"}

    bridge = _FakeBridge()
    client = _QueryClient()

    go_tui_launcher._handle_user_query(bridge, client, "analyze")

    assert (
        "alert",
        {
            "level": "info",
            "message": (
                "Context compacted: 48->21 messages (budget 8192 tokens).\n"
                "Full delegated outputs remain available via store keys and retrieve_agent_result."
            ),
        },
        "",
    ) in bridge.calls
    assert ("text", {"content": "done", "markdown": True}, "") in bridge.calls
    assert ("done", {"summary": ""}, "") in bridge.calls


def test_handle_user_query_maps_agent_change_to_activity_transition(monkeypatch):
    monkeypatch.setattr(go_tui_launcher.time, "time", lambda: 100.0)

    class _QueryClient:
        def __init__(self):
            self.token_tracker = type(
                "Tracker",
                (),
                {"total_tokens": 12, "total_cost": 0.0012},
            )()

        def query(self, text, stream=True, cancel_event=None):
            assert text == "delegate"
            assert stream is True
            yield {
                "type": "agent_change",
                "agent": "research_agent",
                "status": "working",
            }
            yield {"type": "complete"}

    bridge = _FakeBridge()
    client = _QueryClient()

    go_tui_launcher._handle_user_query(bridge, client, "delegate")

    assert (
        "agent_transition",
        {
            "to": "research_agent",
            "kind": "activity",
            "status": "working",
            "reason": "working",
        },
        "",
    ) in bridge.calls


def test_resolve_go_chat_session_target_latest_missing_returns_none(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "lobster.core.workspace.resolve_workspace",
        lambda explicit_path=None, create=True: tmp_path,
    )
    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.session_infra.resolve_session_continuation",
        lambda workspace_path, requested_session_id: (None, "session_x", False),
    )

    session_file, resolved_id = go_tui_launcher._resolve_go_chat_session_target(
        str(tmp_path),
        "latest",
    )

    assert session_file is None
    assert resolved_id is None


def test_resolve_go_chat_session_target_latest_existing_uses_resolved_id(
    tmp_path, monkeypatch
):
    session_file = tmp_path / "session_abc.json"

    monkeypatch.setattr(
        "lobster.core.workspace.resolve_workspace",
        lambda explicit_path=None, create=True: tmp_path,
    )
    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.session_infra.resolve_session_continuation",
        lambda workspace_path, requested_session_id: (
            session_file,
            "session_abc",
            True,
        ),
    )

    resolved_file, resolved_id = go_tui_launcher._resolve_go_chat_session_target(
        str(tmp_path),
        "latest",
    )

    assert resolved_file == session_file
    assert resolved_id == "session_abc"


def test_resolve_go_chat_session_target_explicit_missing_keeps_requested_id(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "lobster.core.workspace.resolve_workspace",
        lambda explicit_path=None, create=True: tmp_path,
    )
    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.session_infra.resolve_session_continuation",
        lambda workspace_path, requested_session_id: (
            None,
            requested_session_id,
            False,
        ),
    )

    resolved_file, resolved_id = go_tui_launcher._resolve_go_chat_session_target(
        str(tmp_path),
        "my_session",
    )

    assert resolved_file is None
    assert resolved_id == "my_session"


def test_prepare_go_tui_chat_env_sets_no_intro_flag(tmp_path):
    env = go_tui_launcher._prepare_go_tui_chat_env(
        {"PATH": "/usr/bin"},
        workspace=str(tmp_path),
        provider="ollama",
        no_intro=True,
    )

    assert env["LOBSTER_TUI_PROVIDER"] == "ollama"
    assert env["LOBSTER_TUI_WORKSPACE"] == str(tmp_path)
    assert env["LOBSTER_TUI_APP_VERSION"]
    assert env["LOBSTER_TUI_NO_INTRO"] == "1"


def test_emit_startup_diagnostic_uses_protocol_alert():
    bridge = _FakeBridge()
    diagnostic = StartupDiagnostic(
        code="missing_provider_package",
        title="Missing provider package",
        detail_lines=("langchain-aws package not installed.",),
        fix_lines=("Run: uv pip install 'lobster-ai[bedrock]'",),
    )

    go_tui_launcher._emit_startup_diagnostic(bridge, diagnostic)

    assert bridge.calls == [
        ("spinner", {"active": False}, ""),
        (
            "alert",
            {
                "level": "error",
                "message": "Missing provider package\n"
                "  langchain-aws package not installed.\n\n"
                "How to fix:\n"
                "  Run: uv pip install 'lobster-ai[bedrock]'",
            },
            "",
        ),
        ("status", {"text": "Initialization failed"}, ""),
    ]


def test_await_startup_diagnostic_ack_waits_for_confirm_response():
    bridge = _FakeBridge(
        events=[
            {
                "type": "confirm_response",
                "id": go_tui_launcher._STARTUP_DIAGNOSTIC_CONFIRM_ID,
                "payload": {"confirm": True},
            }
        ]
    )

    go_tui_launcher._await_startup_diagnostic_ack(bridge)

    assert bridge.calls == [
        (
            "confirm",
            {
                "title": "Startup failed",
                "message": "Press Enter to exit.",
                "default": True,
            },
            go_tui_launcher._STARTUP_DIAGNOSTIC_CONFIRM_ID,
        )
    ]


def test_await_startup_diagnostic_ack_accepts_component_response():
    bridge = _FakeBridge(
        events=[
            {
                "type": "component_response",
                "id": go_tui_launcher._STARTUP_DIAGNOSTIC_CONFIRM_ID,
                "payload": {"data": {"confirmed": True, "action": "submit"}},
            }
        ]
    )

    go_tui_launcher._await_startup_diagnostic_ack(bridge)

    assert bridge.calls == [
        (
            "confirm",
            {
                "title": "Startup failed",
                "message": "Press Enter to exit.",
                "default": True,
            },
            go_tui_launcher._STARTUP_DIAGNOSTIC_CONFIRM_ID,
        )
    ]


def test_protocol_confirm_sends_component_render_and_returns_confirmation():
    bridge = _FakeBridge(
        events=[
            {
                "type": "component_response",
                "id": "confirm-123",
                "payload": {"data": {"confirmed": True, "action": "submit"}},
            }
        ]
    )

    original_uuid4 = go_tui_launcher.uuid.uuid4
    go_tui_launcher.uuid.uuid4 = lambda: "confirm-123"
    try:
        result = go_tui_launcher._protocol_confirm(bridge, "Clear queue?")
    finally:
        go_tui_launcher.uuid.uuid4 = original_uuid4

    assert result is True
    assert bridge.calls == [
        (
            "component_render",
            {
                "component": "confirm",
                "data": {
                    "question": "Clear queue?",
                    "default": False,
                },
            },
            "confirm-123",
        )
    ]


# ---------------------------------------------------------------------------
# Phase 3: HITL interrupt → component_render → response → resume
# ---------------------------------------------------------------------------


def test_handle_user_query_interrupt_renders_component_and_resumes(monkeypatch):
    """Full interrupt cycle: stream → interrupt → component_render → response → resume."""
    monkeypatch.setattr(go_tui_launcher.time, "time", lambda: 100.0)

    class _QueryClient:
        def __init__(self):
            self.token_tracker = type(
                "Tracker", (), {"total_tokens": 10, "total_cost": 0.001}
            )()
            self._resume_called = False

        def query(self, text, stream=True, cancel_event=None):
            # First stream: yields content then interrupt.
            yield {"type": "content_delta", "delta": "Analyzing..."}
            yield {
                "type": "interrupt",
                "data": {
                    "component": "confirm",
                    "data": {"question": "Proceed?"},
                    "fallback_prompt": "Proceed? [y/N]",
                },
                "interrupt_id": "intr-test",
            }

        def resume_from_interrupt(self, response, stream=True, cancel_event=None):
            self._resume_called = True
            assert response == {"confirmed": True}
            yield {"type": "content_delta", "delta": "Done!"}
            yield {"type": "complete"}

    client = _QueryClient()

    # Fake bridge that returns a confirm_response when polled.
    bridge = _FakeBridge(
        events=[
            {
                "type": "confirm_response",
                "payload": {"confirm": True},
            }
        ]
    )

    go_tui_launcher._handle_user_query(bridge, client, "analyze")

    # Verify component_render was sent.
    render_calls = [c for c in bridge.calls if c[0] == "component_render"]
    assert len(render_calls) == 1
    assert render_calls[0][1]["component"] == "confirm"

    # Verify content from both streams was forwarded.
    text_calls = [c for c in bridge.calls if c[0] == "text"]
    assert len(text_calls) == 2
    assert text_calls[0][1]["content"] == "Analyzing..."
    assert text_calls[1][1]["content"] == "Done!"

    # Verify resume was called.
    assert client._resume_called

    # Verify done was sent.
    done_calls = [c for c in bridge.calls if c[0] == "done"]
    assert done_calls == [
        ("done", {"summary": "interrupt"}, ""),
        ("done", {"summary": ""}, ""),
    ]


def test_handle_interrupt_quit_returns_none():
    """If user quits during interrupt, _handle_interrupt returns None."""
    bridge = _FakeBridge(events=[{"type": "quit"}])
    result = go_tui_launcher._handle_interrupt(
        bridge,
        {"data": {"component": "text_input", "fallback_prompt": "Enter:"}},
    )
    assert result is None


def test_handle_interrupt_timeout_returns_none():
    """If bridge times out, _handle_interrupt returns None."""
    bridge = _FakeBridge(events=[])  # No events → recv_event returns None.
    result = go_tui_launcher._handle_interrupt(
        bridge,
        {"data": {"component": "text_input", "fallback_prompt": "Enter:"}},
    )
    assert result is None


def test_handle_interrupt_select_response():
    """select_response is normalized to {selected, index} dict."""
    bridge = _FakeBridge(
        events=[
            {
                "type": "select_response",
                "payload": {"value": "CPM", "index": 1},
            }
        ]
    )
    result = go_tui_launcher._handle_interrupt(
        bridge,
        {"data": {"component": "select"}},
    )
    assert result == {"selected": "CPM", "index": 1}


def test_handle_interrupt_component_response_passthrough():
    """component_response passes through the data dict."""
    bridge = _FakeBridge(
        events=[
            {
                "type": "component_response",
                "payload": {"data": {"answer": "T cells"}},
            }
        ]
    )
    result = go_tui_launcher._handle_interrupt(
        bridge,
        {"data": {"component": "text_input"}},
    )
    assert result == {"answer": "T cells"}


# --- ask_user DecisionRequests (schema lobster.ask/1) ---


def _decision_request(decision_id="dec-call_1", **field_overrides):
    """A real ask_user DecisionRequest, as it arrives in an interrupt event."""
    from lobster.services.interaction.component_schemas import (
        AskUserContext,
        DecisionRequest,
    )

    field = {
        "key": "source",
        "type": "choice",
        "options": ["SRC-A", "SRC-B"],
        **field_overrides,
    }
    field = {k: v for k, v in field.items() if v is not None}  # None = leave out
    context = AskUserContext.model_validate({"fields": [field]})
    return DecisionRequest(
        decision_id=decision_id, question="Which source?", fields=context.fields
    ).to_wire()


class _GoBridge(_FakeBridge):
    """Plays the Go TUI: each scripted reply answers the latest render's id.

    A reply is (action, data), "quit", None (timeout), or ("stray", data): an
    answer carrying an old render id.
    """

    def __init__(self, replies):
        super().__init__()
        self.replies = list(replies)
        self.timeouts = []

    def recv_event(self, timeout=None):
        self.timeouts.append(timeout)
        if not self.replies:
            return None
        reply = self.replies.pop(0)
        if reply is None:
            return None
        if reply == "quit":
            return {"type": "quit"}
        renders = [c for c in self.calls if c[0] == "component_render"]
        msg_id = renders[-1][2] if renders else ""
        if reply[0] == "stray":
            msg_id, reply = "old-render-id", ("submit", reply[1])
        action, data = reply
        return {
            "type": "component_response",
            "id": msg_id,
            "payload": {"id": msg_id, "data": {**data, "action": action}},
        }

    def renders(self):
        return [
            (c[1]["component"], c[1]["data"])
            for c in self.calls
            if c[0] == "component_render"
        ]


def test_handle_user_query_decision_request_renders_choice_and_resumes(monkeypatch):
    """Full cycle: stream -> interrupt -> choice render -> answer envelope -> resume."""
    monkeypatch.setattr(go_tui_launcher.time, "time", lambda: 100.0)
    request = _decision_request()

    class _QueryClient:
        def __init__(self):
            self.token_tracker = type(
                "Tracker", (), {"total_tokens": 10, "total_cost": 0.001}
            )()
            self.resumed_with = None

        def query(self, text, stream=True, cancel_event=None):
            yield {"type": "content_delta", "delta": "Analyzing..."}
            yield {"type": "interrupt", "data": request, "interrupt_id": "intr-1"}

        def resume_from_interrupt(self, response, stream=True, cancel_event=None):
            self.resumed_with = response
            yield {"type": "content_delta", "delta": "Done!"}
            yield {"type": "complete"}

    client = _QueryClient()
    bridge = _GoBridge([("submit", {"selected": [{"id": "SRC-B"}]})])

    go_tui_launcher._handle_user_query(bridge, client, "analyze")

    assert [component for component, _ in bridge.renders()] == ["choice"]
    envelope = client.resumed_with["intr-1"]
    assert envelope["responds_to"] == {
        "decision_id": "dec-call_1",
        "request_version": 1,
    }
    assert envelope["action"] == "submit"
    assert envelope["answer"] == {"source": {"id": "SRC-B"}}

    text_calls = [c for c in bridge.calls if c[0] == "text"]
    assert [c[1]["content"] for c in text_calls] == ["Analyzing...", "Done!"]
    done_calls = [c for c in bridge.calls if c[0] == "done"]
    assert done_calls == [
        ("done", {"summary": "interrupt"}, ""),
        ("done", {"summary": ""}, ""),
    ]


def test_two_pending_questions_each_get_their_own_answer():
    """Both interrupt events are read (none dropped); answers are keyed by id."""
    first = _decision_request("dec-call_a")
    second = _decision_request(
        "dec-call_b",
        key="cutoff",
        type="number",
        view="slider",
        min=0,
        max=1,
        options=None,
    )
    bridge = _GoBridge(
        [("submit", {"selected": [{"id": "SRC-A"}]}), ("submit", {"value": 0.3})]
    )
    response = go_tui_launcher._handle_interrupts(
        bridge,
        [
            {"type": "interrupt", "data": first, "interrupt_id": "intr-a"},
            {"type": "interrupt", "data": second, "interrupt_id": "intr-b"},
        ],
    )
    assert [c for c, _ in bridge.renders()] == ["choice", "threshold_slider"]
    assert response["intr-a"]["responds_to"]["decision_id"] == "dec-call_a"
    assert response["intr-a"]["answer"] == {"source": {"id": "SRC-A"}}
    assert response["intr-b"]["responds_to"]["decision_id"] == "dec-call_b"
    assert response["intr-b"]["answer"] == {"cutoff": 0.3}


def test_stray_reply_to_an_old_render_is_ignored():
    """Bridge fix: only a reply carrying this render's id is accepted."""
    bridge = _GoBridge(
        [
            ("stray", {"selected": [{"id": "SRC-A"}]}),
            ("submit", {"selected": [{"id": "SRC-B"}]}),
        ]
    )
    response = go_tui_launcher._handle_interrupts(
        bridge, [{"data": _decision_request(), "interrupt_id": "intr-1"}]
    )
    assert response["intr-1"]["answer"] == {"source": {"id": "SRC-B"}}


def test_cancelled_question_resumes_with_a_cancel_answer():
    """Esc is an answer (the supervisor is told), not a dropped turn."""
    bridge = _GoBridge([("cancel", {})])
    response = go_tui_launcher._handle_interrupts(
        bridge, [{"data": _decision_request(), "interrupt_id": "intr-1"}]
    )
    assert response["intr-1"]["action"] == "cancel"
    assert "answer" not in response["intr-1"]


def test_decision_request_quit_returns_none():
    """If the user quits during a question, the turn ends without resuming."""
    bridge = _GoBridge(["quit"])
    assert (
        go_tui_launcher._handle_interrupts(
            bridge, [{"data": _decision_request(), "interrupt_id": "intr-1"}]
        )
        is None
    )


def test_decision_request_waits_with_no_deadline():
    """A run without an answer stays paused for as long as the process lives:
    the screen waits with no timeout, other messages (resizes) keep it
    waiting, and nothing is closed or sent for it. Only the TUI going away
    ends the wait, and then nothing is resumed. Other interrupts keep theirs."""
    assert not hasattr(go_tui_launcher, "_ANSWER_TIMEOUT_S")

    class _Resizing(_GoBridge):
        def recv_event(self, timeout=None):
            self.timeouts.append(timeout)
            if len(self.timeouts) <= 3:
                return {"type": "resize", "payload": {"width": 80, "height": 24}}
            return None  # the TUI is gone

    bridge = _Resizing([])
    response = go_tui_launcher._handle_interrupts(
        bridge, [{"data": _decision_request(), "interrupt_id": "intr-1"}]
    )
    assert bridge.timeouts == [None, None, None, None]
    assert response is None
    assert not [c for c in bridge.calls if c[0] == "component_close"]
    legacy = _GoBridge([None])
    go_tui_launcher._handle_interrupts(
        legacy, [{"data": {"component": "text_input"}, "interrupt_id": "i"}]
    )
    assert legacy.timeouts == [300]


def test_a_fault_while_asking_closes_the_screen_and_tells_the_supervisor(
    monkeypatch,
):
    """An error while the question is open resumes with a cancel naming only
    the error's type, never a silent end of turn."""
    from lobster.cli_internal import go_tui_components

    def broken(request, render):
        raise KeyError("/private/path")

    monkeypatch.setattr(go_tui_components, "collect", broken)
    bridge = _GoBridge([])
    response = go_tui_launcher._handle_interrupts(
        bridge, [{"data": _decision_request(), "interrupt_id": "intr-1"}]
    )
    answer = response["intr-1"]
    assert answer["action"] == "cancel"
    assert "(an error (KeyError))" in answer["comment"]
    assert "/private/path" not in json.dumps(answer)
    (close,) = [c for c in bridge.calls if c[0] == "component_close"]
    assert close[2] == ""  # whichever screen is open
    assert close[1]["message"] == (
        "This question was closed (an error (KeyError)); nothing was sent."
    )


def test_a_closed_tui_ends_the_wait_at_once_without_resuming():
    """When the TUI exits, the wait (which has no deadline) must not run on:
    the reader stops, the wait ends at once, and the turn ends (nobody is
    there)."""
    import time as real_time

    to_go_read, to_go_write = os.pipe()
    from_go_read, from_go_write = os.pipe()
    bridge = go_tui_launcher._LightBridge(None, to_go_write, from_go_read)
    try:
        os.close(from_go_write)  # the TUI is gone
        started = real_time.monotonic()
        response = go_tui_launcher._handle_interrupts(
            bridge, [{"data": _decision_request(), "interrupt_id": "intr-1"}]
        )
        elapsed = real_time.monotonic() - started
        assert bridge.closed.is_set()
        assert bridge.recv_event(timeout=60) is None  # every later wait too
    finally:
        bridge.close()
        os.close(to_go_read)
    assert response is None
    assert elapsed < 5
    legacy = _GoBridge([None])
    go_tui_launcher._handle_interrupts(
        legacy, [{"data": {"component": "text_input"}, "interrupt_id": "i"}]
    )
    assert legacy.timeouts == [300]


def test_component_error_is_reported_and_cancels_the_question():
    """A screen that fails is not the user's cancel: the comment says so."""
    bridge = _GoBridge([("error", {"error": "init_failed", "detail": "bad data"})])
    response = go_tui_launcher._handle_interrupts(
        bridge, [{"data": _decision_request(), "interrupt_id": "intr-1"}]
    )
    assert response["intr-1"]["action"] == "cancel"
    assert response["intr-1"]["comment"] == (
        "Not answered: this terminal can't show this question (bad data). Don't "
        "ask again with the same shape; ask in chat or with a shape it can show."
    )
    alerts = [c[1]["message"] for c in bridge.calls if c[0] == "alert"]
    assert alerts == ["Couldn't show the question: bad data"]


def test_component_error_without_detail_still_says_the_screen_failed():
    """A screen that breaks while drawing names its error code instead."""
    request = _decision_request(
        key="note", type="text", options=None, question="Anything else?"
    )
    bridge = _GoBridge([("error", {"error": "view_panic"})])
    response = go_tui_launcher._handle_interrupts(
        bridge, [{"data": request, "interrupt_id": "intr-1"}]
    )
    assert response["intr-1"]["action"] == "cancel"
    assert response["intr-1"]["comment"].startswith(
        "Not answered: this terminal can't show this question (view_panic)."
    )


def test_question_the_terminal_cannot_draw_is_cancelled_with_a_reason():
    request = _decision_request(
        key="times", type="number", count={"min": 1}, options=None
    )
    bridge = _GoBridge([])
    response = go_tui_launcher._handle_interrupts(
        bridge, [{"data": request, "interrupt_id": "intr-1"}]
    )
    assert bridge.renders() == []
    assert response["intr-1"]["action"] == "cancel"
    comment = response["intr-1"]["comment"]
    assert comment.startswith("Not answered: this terminal can't show times: a list")
    assert "Don't ask again with the same shape" in comment


def test_plate_the_go_screen_would_refuse_is_cancelled_before_rendering():
    """More than 9 options: the plate's Init would fail, so it is never sent."""
    request = _decision_request(
        key="groups",
        view="plate_map",
        for_each=["S01", "S02"],
        options=[f"g{n}" for n in range(10)],
    )
    bridge = _GoBridge([])
    response = go_tui_launcher._handle_interrupts(
        bridge, [{"data": request, "interrupt_id": "intr-1"}]
    )
    assert bridge.renders() == []
    assert response["intr-1"]["action"] == "cancel"
    assert response["intr-1"]["comment"] == (
        "Not answered: this terminal can't show groups: a plate with 10 options "
        "(at most 9). Don't ask again with the same shape; ask in chat or with a "
        "shape it can show."
    )


# --- Interrupts that are not DecisionRequests keep the _handle_interrupt path ---


def test_single_other_interrupt_resumes_bare_through_handle_interrupt():
    """One pending non-ask_user interrupt: rendered as is, bare value back."""
    payload = {"component": "text_input", "data": {"question": "Q?"}}
    bridge = _FakeBridge(
        events=[{"type": "component_response", "payload": {"data": {"answer": "x"}}}]
    )
    response = go_tui_launcher._handle_interrupts(
        bridge, [{"data": payload, "interrupt_id": "intr-1"}]
    )
    assert response == {"answer": "x"}
    renders = [c for c in bridge.calls if c[0] == "component_render"]
    assert [c[1] for c in renders] == [payload]


def test_single_other_interrupt_cancel_ends_the_turn():
    bridge = _FakeBridge(
        events=[
            {
                "type": "component_response",
                "payload": {"data": {"action": "cancel"}},
            }
        ]
    )
    assert (
        go_tui_launcher._handle_interrupts(
            bridge, [{"data": {"component": "confirm"}, "interrupt_id": "i"}]
        )
        is None
    )


def test_decision_request_and_other_interrupt_pending_together_are_keyed():
    payload = {"component": "text_input", "data": {"question": "Q?"}}
    bridge = _GoBridge(
        [("submit", {"selected": [{"id": "SRC-A"}]}), ("submit", {"answer": "x"})]
    )
    response = go_tui_launcher._handle_interrupts(
        bridge,
        [
            {"data": _decision_request(), "interrupt_id": "intr-a"},
            {"data": payload, "interrupt_id": "intr-b"},
        ],
    )
    assert response["intr-a"]["answer"] == {"source": {"id": "SRC-A"}}
    # _handle_interrupt passes the reply through as is, action included.
    assert response["intr-b"] == {"answer": "x", "action": "submit"}


def test_every_render_carries_the_whole_question_as_fallback_prompt():
    from lobster.cli_internal.go_tui_components import fallback_prompt

    request = _decision_request(
        options=["SRC-A", {"id": "SRC-C", "unavailable": "normalized"}]
    )
    bridge = _GoBridge([("submit", {"selected": [{"id": "SRC-A"}]})])
    go_tui_launcher._handle_interrupts(
        bridge, [{"data": request, "interrupt_id": "intr-1"}]
    )
    (render,) = [c[1] for c in bridge.calls if c[0] == "component_render"]
    assert render["fallback_prompt"] == fallback_prompt(request)
    assert "SRC-C (unavailable: normalized)" in render["fallback_prompt"]
    assert render["fallback_prompt"].endswith("Write your answer in your own words.")


def test_older_tui_text_box_reply_resumes_as_free_text():
    """An older TUI draws its text box for "choice" and replies {"answer"}."""
    bridge = _GoBridge([("submit", {"answer": "SRC-B, the repository one"})])
    response = go_tui_launcher._handle_interrupts(
        bridge, [{"data": _decision_request(), "interrupt_id": "intr-1"}]
    )
    envelope = response["intr-1"]
    assert envelope["action"] == "free_text"
    assert envelope["text"] == "SRC-B, the repository one"
    assert "answer" not in envelope


class _ScriptedBridge(_FakeBridge):
    """Plays the Go TUI from a script. ("reply", n, action, data) answers the
    n-th render (0-based) with that render's id, so a step can repeat an old
    render's reply; any other step is returned as the event itself."""

    def __init__(self, script):
        super().__init__()
        self.script = list(script)

    def recv_event(self, timeout=None):
        if not self.script:
            return None
        step = self.script.pop(0)
        if not (isinstance(step, tuple) and step[0] == "reply"):
            return step
        _, n, action, data = step
        msg_id = [c for c in self.calls if c[0] == "component_render"][n][2]
        payload = {"id": msg_id, "data": {**data, "action": action}}
        return {"type": "component_response", "id": msg_id, "payload": payload}


class _ResumingClient:
    """query() pauses with the first event; each resume records the response
    and pauses with the next event, until none is left (then completes)."""

    def __init__(self, *pauses):
        self.token_tracker = type(
            "Tracker", (), {"total_tokens": 10, "total_cost": 0.001}
        )()
        self.pauses = list(pauses)
        self.resumed_with = []

    def query(self, text, stream=True, cancel_event=None):
        yield self.pauses.pop(0)

    def resume_from_interrupt(self, response, stream=True, cancel_event=None):
        self.resumed_with.append(response)
        if self.pauses:
            yield self.pauses.pop(0)
        else:
            yield {"type": "complete"}


def _done(bridge):
    return [c[1]["summary"] for c in bridge.calls if c[0] == "done"]


def test_duplicate_reply_after_resume_is_ignored_by_the_next_render(monkeypatch):
    """A second copy of an answered render's reply arrives while the re-asked
    question is shown: the next render takes only a reply with its own id."""
    monkeypatch.setattr(go_tui_launcher.time, "time", lambda: 100.0)
    first = _decision_request()
    second = {
        **first,
        "request_version": 2,
        "problems": [{"path": "source", "message": "pick again"}],
    }
    client = _ResumingClient(
        {"type": "interrupt", "data": first, "interrupt_id": "intr-1"},
        {"type": "interrupt", "data": second, "interrupt_id": "intr-1"},
    )
    bridge = _ScriptedBridge(
        [
            ("reply", 0, "submit", {"selected": [{"id": "SRC-A"}]}),
            ("reply", 0, "submit", {"selected": [{"id": "SRC-A"}]}),  # duplicate
            ("reply", 1, "submit", {"selected": [{"id": "SRC-B"}]}),
        ]
    )
    go_tui_launcher._handle_user_query(bridge, client, "analyze")

    answers = [r["intr-1"] for r in client.resumed_with]
    assert [a["responds_to"]["request_version"] for a in answers] == [1, 2]
    assert [a["answer"] for a in answers] == [
        {"source": {"id": "SRC-A"}},
        {"source": {"id": "SRC-B"}},
    ]
    renders = [c for c in bridge.calls if c[0] == "component_render"]
    assert len(renders) == 2 and renders[0][2] != renders[1][2]
    assert "⚠ pick again" in renders[1][1]["data"]["details"]
    assert _done(bridge) == ["interrupt", "interrupt", ""]


def test_duplicate_reply_after_the_turn_is_ignored_by_the_main_loop():
    """A component_response that arrives when no question is shown starts
    nothing and answers nothing."""

    class _Client:
        def query(self, *args, **kwargs):
            raise AssertionError("a stray reply must not start a query")

        def _save_session_json(self):
            pass

    stray = {
        "type": "component_response",
        "id": "render-1",
        "payload": {"id": "render-1", "data": {"selected": [{"id": "SRC-A"}]}},
    }
    bridge = _FakeBridge(events=[stray, stray, {"type": "quit"}])
    go_tui_launcher._go_tui_event_loop(bridge, _Client())
    assert bridge.calls == []


@pytest.mark.parametrize("action", ["cancel", "defer"])
def test_cancel_or_defer_resumes_the_turn(action, monkeypatch):
    """Esc and ctrl+d are answers: the supervisor is told, the turn goes on."""
    monkeypatch.setattr(go_tui_launcher.time, "time", lambda: 100.0)
    client = _ResumingClient(
        {"type": "interrupt", "data": _decision_request(), "interrupt_id": "intr-1"}
    )
    bridge = _ScriptedBridge([("reply", 0, action, {})])
    go_tui_launcher._handle_user_query(bridge, client, "analyze")
    (response,) = client.resumed_with
    assert response["intr-1"]["action"] == action
    assert "answer" not in response["intr-1"]
    assert _done(bridge) == ["interrupt", ""]


def test_quit_while_a_question_is_shown_ends_the_turn(monkeypatch):
    monkeypatch.setattr(go_tui_launcher.time, "time", lambda: 100.0)
    client = _ResumingClient(
        {"type": "interrupt", "data": _decision_request(), "interrupt_id": "intr-1"}
    )
    bridge = _ScriptedBridge([{"type": "quit"}])
    go_tui_launcher._handle_user_query(bridge, client, "analyze")
    assert client.resumed_with == []
    assert _done(bridge) == ["interrupt", "cancelled"]


def test_unanswered_question_stays_open_until_the_tui_goes():
    """With a real bridge and no reply, the question stays open: nothing is
    closed, nothing resumes. When the TUI exits, the turn ends unresumed."""
    import threading
    import time as real_time

    to_go_read, to_go_write = os.pipe()  # what the bridge sends
    from_go_read, from_go_write = os.pipe()  # what "Go" would send: nothing
    bridge = go_tui_launcher._LightBridge(None, to_go_write, from_go_read)
    client = _ResumingClient(
        {"type": "interrupt", "data": _decision_request(), "interrupt_id": "intr-1"}
    )
    turn = threading.Thread(
        target=go_tui_launcher._handle_user_query,
        args=(bridge, client, "analyze"),
        daemon=True,
    )
    try:
        turn.start()
        real_time.sleep(0.5)
        still_waiting = turn.is_alive()
        os.set_blocking(to_go_read, False)
        sent = [json.loads(line) for line in os.read(to_go_read, 1 << 16).splitlines()]
        os.close(from_go_write)  # the TUI exits
        turn.join(timeout=5)
    finally:
        bridge.close()
        os.close(to_go_read)
    assert still_waiting
    assert not turn.is_alive()
    types = [m["type"] for m in sent]
    assert "component_render" in types
    assert "component_close" not in types
    assert client.resumed_with == []


def test_other_interrupt_payload_goes_through_upstream_handle_interrupt(monkeypatch):
    monkeypatch.setattr(go_tui_launcher.time, "time", lambda: 100.0)
    upstream = go_tui_launcher._handle_interrupt
    seen = []

    def spy(bridge, event):
        seen.append(event)
        return upstream(bridge, event)

    monkeypatch.setattr(go_tui_launcher, "_handle_interrupt", spy)
    legacy = {
        "type": "interrupt",
        "data": {"component": "confirm", "data": {"question": "Proceed?"}},
        "interrupt_id": "intr-1",
    }
    client = _ResumingClient(legacy)
    bridge = _FakeBridge(
        events=[{"type": "confirm_response", "payload": {"confirm": True}}]
    )
    go_tui_launcher._handle_user_query(bridge, client, "analyze")
    assert seen == [legacy]
    assert client.resumed_with == [{"confirmed": True}]  # bare, as upstream


# ---------------------------------------------------------------------------
# Cancel event bridge-level signal
# ---------------------------------------------------------------------------


def test_event_loop_wires_cancel_event_to_bridge():
    """_go_tui_event_loop assigns cancel_event to bridge so _read_loop can set it."""
    bridge = _FakeBridge(events=[{"type": "quit"}])

    class _Client:
        def _save_session_json(self):
            pass

    go_tui_launcher._go_tui_event_loop(bridge, _Client())

    # After the loop runs, bridge.cancel_event should be a threading.Event.
    import threading

    assert isinstance(bridge.cancel_event, threading.Event)


def test_bridge_cancel_event_set_bypasses_event_loop():
    """Simulate _read_loop setting cancel_event directly without event loop dispatch."""
    import threading

    cancel_event = threading.Event()

    # Simulate what _read_loop does when it reads a cancel message.
    bridge = _FakeBridge()
    bridge.cancel_event = cancel_event

    # Simulate the _read_loop inline: parse msg, enqueue, check cancel.
    msg = {"type": "cancel"}
    bridge.events.append(msg)
    if msg.get("type") == "cancel" and bridge.cancel_event is not None:
        bridge.cancel_event.set()

    assert cancel_event.is_set(), "cancel_event should be set by bridge-level signal"


def test_several_questions_without_ids_are_refused_not_merged():
    """Answers are keyed by interrupt id; without ids the second answer would
    overwrite the first, so nothing is sent (the classic UI refuses too)."""
    bridge = _GoBridge([])
    response = go_tui_launcher._handle_interrupts(
        bridge,
        [
            {"type": "interrupt", "data": _decision_request("dec-call_a")},
            {"type": "interrupt", "data": _decision_request("dec-call_b")},
        ],
    )
    assert response is None
    assert bridge.renders() == []
    alerts = [c[1] for c in bridge.calls if c[0] == "alert"]
    assert alerts == [
        {
            "level": "error",
            "message": "Several questions arrived without their ids, so their "
            "answers can't be told apart. Nothing was sent.",
        }
    ]
