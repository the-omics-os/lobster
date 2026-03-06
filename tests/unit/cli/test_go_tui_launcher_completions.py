from pathlib import Path

from lobster.cli_internal import go_tui_launcher
from lobster.cli_internal.startup_diagnostics import StartupDiagnostic


class _FakeBridge:
    def __init__(self, events=None):
        self.calls = []
        self.events = list(events or [])

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
        def __init__(self, emit):
            self.emit = emit

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


def test_prepare_go_tui_chat_env_sets_no_intro_flag(tmp_path):
    env = go_tui_launcher._prepare_go_tui_chat_env(
        {"PATH": "/usr/bin"},
        workspace=str(tmp_path),
        provider="ollama",
        no_intro=True,
    )

    assert env["LOBSTER_TUI_PROVIDER"] == "ollama"
    assert env["LOBSTER_TUI_WORKSPACE"] == str(tmp_path)
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
