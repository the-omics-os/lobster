import os
from pathlib import Path

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
