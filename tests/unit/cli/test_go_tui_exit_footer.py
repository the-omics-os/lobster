from lobster.cli_internal import go_tui_launcher


class _Tracker:
    def get_minimal_summary(self):
        return "Tokens: 321 | Cost: $0.0042"


class _VerboseOnlyTracker:
    def get_verbose_summary(self):
        return "Tokens in: 100 | Tokens out: 200"


class _BrokenTracker:
    def get_minimal_summary(self):
        raise RuntimeError("tracker unavailable")


class _Client:
    def __init__(
        self,
        *,
        session_id="session-123",
        provider_override="openrouter",
        model_override="claude-sonnet",
        tracker=None,
    ):
        self.session_id = session_id
        self.provider_override = provider_override
        self.model_override = model_override
        self.token_tracker = tracker


def test_build_go_tui_exit_footer_includes_parity_and_smart_lines():
    client = _Client(tracker=_Tracker())

    lines = go_tui_launcher._build_go_tui_exit_footer_lines(client)

    assert "Session: session-123 (use --session-id latest to continue)" in lines
    assert f"Feedback: {go_tui_launcher._FEEDBACK_URL}" in lines
    assert f"Issues: {go_tui_launcher._ISSUES_URL}" in lines
    assert f"Contact: {go_tui_launcher._SUPPORT_EMAIL}" in lines
    assert "Runtime: provider=openrouter model=claude-sonnet" in lines
    assert "Tokens: 321 | Cost: $0.0042" in lines
    assert "Next steps:" in lines
    assert "  Resume: lobster chat --ui go --session-id latest" in lines
    assert "  New chat: lobster chat --ui go" in lines
    assert "  Help: lobster --help" in lines


def test_safe_minimal_token_summary_falls_back_to_verbose_summary():
    client = _Client(tracker=_VerboseOnlyTracker())

    summary = go_tui_launcher._safe_minimal_token_summary(client)

    assert summary == "Tokens in: 100 | Tokens out: 200"


def test_emit_go_tui_exit_footer_tolerates_tracker_failure(capsys):
    client = _Client(
        session_id="",
        provider_override="",
        model_override="",
        tracker=_BrokenTracker(),
    )

    go_tui_launcher._emit_go_tui_exit_footer(client)
    out = capsys.readouterr().out

    assert f"Feedback: {go_tui_launcher._FEEDBACK_URL}" in out
    assert f"Issues: {go_tui_launcher._ISSUES_URL}" in out
    assert f"Contact: {go_tui_launcher._SUPPORT_EMAIL}" in out
    assert "Runtime:" not in out
    assert "Session:" not in out
