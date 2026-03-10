from types import SimpleNamespace

from rich.console import Console

from lobster.cli_internal.commands.heavy import query_commands
from lobster.cli_internal.commands.heavy import session_infra


class _Tracker:
    def get_minimal_summary(self):
        return "Session: 5.8k tokens (local)"


class _Client:
    def __init__(self, callbacks=None):
        self.session_id = "session-123"
        self.token_tracker = _Tracker()
        self.callbacks = callbacks or []


def test_compact_query_usage_summary_normalizes_session_prefix():
    assert (
        query_commands._compact_query_usage_summary(
            "Session: 5.8k tokens (local)"
        )
        == "5.8k tokens (local)"
    )
    assert (
        query_commands._compact_query_usage_summary("Session cost: $0.16")
        == "cost $0.16"
    )


def test_query_answer_text_prefers_separate_text_payload():
    result = {
        "response": "[Thinking: hidden]\n\nFinal answer",
        "text": "Final answer",
    }

    assert query_commands._query_answer_text(result) == "Final answer"


def test_render_human_query_result_uses_compact_footer():
    console = Console(record=True, width=100)
    result = {
        "response": "Hello from Lobster.\n\n- One\n- Two",
        "last_agent": "research_agent",
    }

    query_commands._render_human_query_result(
        console,
        result,
        _Client(),
        reasoning=False,
        verbose=False,
    )
    out = console.export_text()

    assert "Lobster" in out
    assert "Research Agent" in out
    assert "Hello from Lobster." in out
    assert "session=session-123" in out
    assert "5.8k tokens (local)" in out
    assert "continue with --session-id latest" in out
    assert "Lobster Response" not in out


def test_render_human_query_result_uses_text_payload_for_reasoning_mode():
    console = Console(record=True, width=100)
    result = {
        "response": "[Thinking: inspect tools]\n\nFinal answer only",
        "reasoning": "[Thinking: inspect tools]",
        "text": "Final answer only",
        "last_agent": "research_agent",
    }

    query_commands._render_human_query_result(
        console,
        result,
        _Client(),
        reasoning=True,
        verbose=False,
    )
    out = console.export_text()

    assert "Final answer only" in out
    assert out.count("Final answer only") == 1
    assert "[Thinking: inspect tools]" in out


def test_render_human_query_result_skips_reasoning_if_callback_already_showed_it():
    console = Console(record=True, width=100)
    result = {
        "response": "[Thinking: inspect tools]\n\nFinal answer only",
        "reasoning": "[Thinking: inspect tools]",
        "text": "Final answer only",
        "last_agent": "research_agent",
    }

    callback = SimpleNamespace(
        events=[],
        has_displayed_reasoning=lambda: True,
        has_visible_output=lambda: True,
    )

    query_commands._render_human_query_result(
        console,
        result,
        _Client(callbacks=[callback]),
        reasoning=True,
        verbose=False,
    )
    out = console.export_text()

    assert "Final answer only" in out
    assert "[Thinking: inspect tools]" not in out


def test_query_execution_summary_includes_agents_and_tools():
    callback = SimpleNamespace(
        events=[
            SimpleNamespace(
                type=SimpleNamespace(name="HANDOFF"),
                metadata={"to": "research_agent"},
                agent_name="system",
            ),
            SimpleNamespace(
                type=SimpleNamespace(name="TOOL_START"),
                metadata={"tool_name": "search_pubmed"},
                agent_name="research_agent",
            ),
            SimpleNamespace(
                type=SimpleNamespace(name="TOOL_START"),
                metadata={"tool_name": "read_file"},
                agent_name="research_agent",
            ),
        ]
    )

    summary = query_commands._query_execution_summary([callback])

    assert "agents: Research Agent" in summary
    assert "tools: search_pubmed -> read_file" in summary


def test_query_impl_prints_working_indicator_in_reasoning_mode(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("LOBSTER_TEST=1\n")

    recorded_console = Console(record=True, width=120)
    monkeypatch.setattr(query_commands, "console", recorded_console)
    monkeypatch.setattr(query_commands, "_maybe_print_timings", lambda *args, **kwargs: None)
    monkeypatch.setattr(query_commands, "should_show_progress", lambda client: False)

    class _FakeClient:
        def __init__(self):
            self.session_id = "session-test"
            self.token_tracker = None
            self.callbacks = []

        def query(self, question):
            assert question == "debug me"
            return {
                "success": True,
                "response": "[Thinking: plan]\n\nFinal answer",
                "reasoning": "[Thinking: plan]",
                "text": "Final answer",
                "last_agent": "research_agent",
            }

    monkeypatch.setattr(query_commands, "init_client", lambda *args, **kwargs: _FakeClient())

    query_commands.query_impl(
        "debug me",
        workspace=None,
        session_id=None,
        reasoning=True,
        verbose=False,
        debug=False,
        output=None,
        profile_timings=None,
        provider=None,
        model=None,
        stream=False,
        json_output=False,
    )

    out = recorded_console.export_text()
    assert "Working" in out
    assert "Final answer" in out
    assert out.count("Final answer") == 1


def test_query_impl_reports_trace_mode_when_stream_requested(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("LOBSTER_TEST=1\n")

    recorded_console = Console(record=True, width=120)
    monkeypatch.setattr(query_commands, "console", recorded_console)
    monkeypatch.setattr(query_commands, "_maybe_print_timings", lambda *args, **kwargs: None)
    monkeypatch.setattr(query_commands, "should_show_progress", lambda client: False)

    class _FakeClient:
        def __init__(self):
            self.session_id = "session-test"
            self.token_tracker = None
            self.callbacks = []

        def query(self, question):
            return {
                "success": True,
                "response": "Final answer",
                "text": "Final answer",
                "last_agent": "research_agent",
            }

    monkeypatch.setattr(query_commands, "init_client", lambda *args, **kwargs: _FakeClient())

    query_commands.query_impl(
        "debug me",
        workspace=None,
        session_id=None,
        reasoning=True,
        verbose=False,
        debug=False,
        output=None,
        profile_timings=None,
        provider=None,
        model=None,
        stream=True,
        json_output=False,
    )

    out = recorded_console.export_text()
    assert "Trace mode active" in out
    assert "Final answer" in out


def test_display_streaming_response_delegates_to_shared_helper(monkeypatch):
    expected = {"success": True, "response": "streamed"}

    def _fake_impl(client, user_input, console):
        assert client == "client"
        assert user_input == "hello"
        assert console == "console"
        return expected

    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.slash_commands._display_streaming_response",
        _fake_impl,
    )

    result = query_commands._display_streaming_response(
        "client", "hello", "console"
    )

    assert result == expected


def test_query_impl_routes_slash_commands_to_local_command_path(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        query_commands,
        "validate_startup_or_raise_startup_diagnostic",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("startup diagnostics should not run for slash commands")
        ),
    )
    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.slash_commands.command_cmd_impl",
        lambda **kwargs: captured.update(kwargs),
    )

    query_commands.query_impl(
        "/help",
        workspace="workspace",
        session_id="latest",
        reasoning=False,
        verbose=False,
        debug=False,
        output=None,
        profile_timings=None,
        provider=None,
        model=None,
        stream=False,
        json_output=True,
    )

    assert captured == {
        "cmd": "/help",
        "workspace": "workspace",
        "session_id": "latest",
        "json_output": True,
        "help_profile": "query",
    }


def test_resolve_session_continuation_prefers_session_directory_for_latest(tmp_path):
    workspace_path = tmp_path / ".lobster_workspace"
    session_dir = workspace_path / ".lobster" / "sessions" / "session_20260305_120000"
    session_dir.mkdir(parents=True)

    session_file, session_id, found_existing = session_infra.resolve_session_continuation(
        workspace_path,
        "latest",
    )

    assert session_file is None
    assert session_id == "session_20260305_120000"
    assert found_existing is True


def test_resolve_session_continuation_explicit_missing_returns_new_session_id(tmp_path):
    workspace_path = tmp_path / ".lobster_workspace"
    workspace_path.mkdir(parents=True)

    session_file, session_id, found_existing = session_infra.resolve_session_continuation(
        workspace_path,
        "project_alpha",
    )

    assert session_file is None
    assert session_id == "project_alpha"
    assert found_existing is False


def test_resolve_session_continuation_explicit_legacy_json_is_loaded(tmp_path):
    workspace_path = tmp_path / ".lobster_workspace"
    workspace_path.mkdir(parents=True)
    legacy_session = workspace_path / "session_project_beta.json"
    legacy_session.write_text("{}")

    session_file, session_id, found_existing = session_infra.resolve_session_continuation(
        workspace_path,
        "project_beta",
    )

    assert session_file == legacy_session
    assert session_id == "project_beta"
    assert found_existing is True


def test_resolve_session_continuation_latest_falls_back_to_legacy_json(tmp_path):
    workspace_path = tmp_path / ".lobster_workspace"
    workspace_path.mkdir(parents=True)
    legacy_session = workspace_path / "session_project_gamma.json"
    legacy_session.write_text("{}")

    session_file, session_id, found_existing = session_infra.resolve_session_continuation(
        workspace_path,
        "latest",
    )

    assert session_file == legacy_session
    assert session_id == "project_gamma"
    assert found_existing is True
