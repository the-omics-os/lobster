"""Tests for venv-safe init: TTY guard and _uv_tool_env_handoff os._exit paths.

These tests cover the critical bug where `uv tool install` replaces the venv
on disk while the current process is still running, causing Rich to crash with
ModuleNotFoundError on lazy-loaded _unicode_data modules.
"""

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest
import typer

from lobster.cli_internal.commands.heavy.init_commands import (
    _uv_tool_env_handoff,
    init_impl,
)


# ── Shared init_impl kwargs ──────────────────────────────────────────────────

_INIT_KWARGS = dict(
    global_config=False,
    force=False,
    non_interactive=False,
    anthropic_key=None,
    bedrock_access_key=None,
    bedrock_secret_key=None,
    use_ollama=False,
    ollama_model=None,
    gemini_key=None,
    openai_key=None,
    profile=None,
    ncbi_key=None,
    cloud_key=None,
    cloud_endpoint=None,
    skip_ssl_test=False,
    ssl_verify=None,
    ssl_cert_path=None,
    agents=None,
    preset=None,
    auto_agents=False,
    agents_description=None,
    skip_docling=False,
    install_docling=False,
    install_vector_search=False,
    skip_extras=True,
    ui_mode="auto",
)


# ── TTY guard tests ──────────────────────────────────────────────────────────


class TestTTYGuard:
    """TTY guard must block all interactive paths when stdin is not a terminal."""

    def test_non_tty_stdin_exits_with_error(self, monkeypatch, tmp_path, capsys):
        """When stdin is not a TTY, init_impl should exit(1) before any UI."""
        monkeypatch.chdir(tmp_path)
        # Simulate non-TTY stdin (e.g., curl | bash with exhausted pipe)
        fake_stdin = MagicMock()
        fake_stdin.isatty.return_value = False
        monkeypatch.setattr("sys.stdin", fake_stdin)

        with pytest.raises(typer.Exit) as exc_info:
            init_impl(**_INIT_KWARGS)

        assert exc_info.value.exit_code == 1
        captured = capsys.readouterr()
        assert "stdin is not a terminal" in captured.out

    def test_non_tty_stdin_does_not_attempt_go_tui(self, monkeypatch, tmp_path):
        """Go TUI should never be attempted when stdin is not a TTY."""
        monkeypatch.chdir(tmp_path)
        fake_stdin = MagicMock()
        fake_stdin.isatty.return_value = False
        monkeypatch.setattr("sys.stdin", fake_stdin)

        # If Go TUI is attempted, this will raise and fail the test
        monkeypatch.setattr(
            "lobster.ui.bridge.binary_finder.find_tui_binary",
            lambda: (_ for _ in ()).throw(
                AssertionError("Go TUI should not be attempted")
            ),
        )

        with pytest.raises(typer.Exit) as exc_info:
            init_impl(**_INIT_KWARGS)

        assert exc_info.value.exit_code == 1

    def test_non_interactive_flag_bypasses_tty_guard(self, monkeypatch, tmp_path, capsys):
        """--non-interactive should skip the TTY check entirely."""
        monkeypatch.chdir(tmp_path)
        fake_stdin = MagicMock()
        fake_stdin.isatty.return_value = False
        monkeypatch.setattr("sys.stdin", fake_stdin)

        # non_interactive=True should skip the guard and reach the
        # non-interactive code path (which needs env vars or keys).
        # It will fail for other reasons, but NOT with the TTY error.
        with pytest.raises((typer.Exit, SystemExit, Exception)):
            init_impl(**{**_INIT_KWARGS, "non_interactive": True})

        # The TTY guard message must NOT appear — it was bypassed.
        captured = capsys.readouterr()
        assert "stdin is not a terminal" not in captured.out


# ── _uv_tool_env_handoff tests ───────────────────────────────────────────────


class TestUvToolEnvHandoff:
    """_uv_tool_env_handoff must use os._exit after subprocess, never Rich."""

    def _setup_handoff_mocks(self, monkeypatch):
        """Set up mocks so _uv_tool_env_handoff reaches the subprocess call."""
        # Pretend we're in a uv tool env
        monkeypatch.setattr(
            "lobster.core.uv_tool_env.detect_uv_tool_env",
            lambda: MagicMock(installed_packages=[]),
        )
        # Build a command that differs from the no-op baseline
        monkeypatch.setattr(
            "lobster.cli_internal.commands.heavy.init_commands._build_uv_tool_init_command",
            lambda provider, agents, include_vector_search=False: (
                ["uv", "tool", "install", "lobster-ai", "--with", "lobster-research"]
                if agents
                else ["uv", "tool", "install", "lobster-ai"]
            ),
        )

    def test_success_calls_os_exit_zero(self, monkeypatch):
        """On successful install, os._exit(0) is called — not sys.exit."""
        self._setup_handoff_mocks(monkeypatch)
        monkeypatch.setattr(
            "lobster.cli_internal.commands.heavy.init_commands.Confirm.ask",
            lambda *a, **kw: True,
        )

        exit_codes = []

        def _fake_os_exit(code):
            exit_codes.append(code)
            raise SystemExit(code)  # Stop execution

        monkeypatch.setattr("os._exit", _fake_os_exit)
        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0),
        )

        with pytest.raises(SystemExit):
            _uv_tool_env_handoff(
                provider_name="anthropic",
                selected_agents=["lobster-research"],
                skip_extras=False,
            )

        assert exit_codes == [0]

    def test_failure_calls_os_exit_nonzero(self, monkeypatch):
        """On failed install, os._exit(returncode) is called."""
        self._setup_handoff_mocks(monkeypatch)
        monkeypatch.setattr(
            "lobster.cli_internal.commands.heavy.init_commands.Confirm.ask",
            lambda *a, **kw: True,
        )

        exit_codes = []

        def _fake_os_exit(code):
            exit_codes.append(code)
            raise SystemExit(code)

        monkeypatch.setattr("os._exit", _fake_os_exit)
        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stderr="error"),
        )

        with pytest.raises(SystemExit):
            _uv_tool_env_handoff(
                provider_name="anthropic",
                selected_agents=["lobster-research"],
                skip_extras=False,
            )

        assert exit_codes == [1]

    def test_user_declines_returns_normally(self, monkeypatch):
        """When user says no, function returns without os._exit or sys.exit."""
        self._setup_handoff_mocks(monkeypatch)
        monkeypatch.setattr(
            "lobster.cli_internal.commands.heavy.init_commands.Confirm.ask",
            lambda *a, **kw: False,
        )

        # Should return normally — no exception
        _uv_tool_env_handoff(
            provider_name="anthropic",
            selected_agents=["lobster-research"],
            skip_extras=False,
        )

    def test_not_in_uv_tool_env_returns_immediately(self, monkeypatch):
        """When not in a uv tool env, function is a no-op."""
        monkeypatch.setattr(
            "lobster.core.uv_tool_env.detect_uv_tool_env",
            lambda: None,
        )

        # Should return immediately — no prompts, no subprocess
        _uv_tool_env_handoff(
            provider_name="anthropic",
            selected_agents=["lobster-research"],
            skip_extras=False,
        )

    def test_no_rich_calls_after_subprocess(self, monkeypatch, capsys):
        """After subprocess.run, only plain print is used — never console.print."""
        self._setup_handoff_mocks(monkeypatch)
        monkeypatch.setattr(
            "lobster.cli_internal.commands.heavy.init_commands.Confirm.ask",
            lambda *a, **kw: True,
        )

        # Track console.print calls after subprocess
        import lobster.cli_internal.commands.heavy.init_commands as mod

        original_console = mod.console
        post_subprocess_rich_calls = []

        class RichSpy:
            """Proxy that records calls but delegates to the real console."""

            def __getattr__(self, name):
                return getattr(original_console, name)

            def print(self, *args, **kwargs):
                # Record the call — we'll check if any happened after subprocess
                post_subprocess_rich_calls.append(args)
                original_console.print(*args, **kwargs)

        spy = RichSpy()
        monkeypatch.setattr(mod, "console", spy)

        subprocess_ran = []

        def _fake_subprocess_run(cmd, **kw):
            subprocess_ran.append(True)
            # Clear the spy's log — we only care about calls AFTER this
            post_subprocess_rich_calls.clear()
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr("subprocess.run", _fake_subprocess_run)
        monkeypatch.setattr("os._exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))

        with pytest.raises(SystemExit):
            _uv_tool_env_handoff(
                provider_name="anthropic",
                selected_agents=["lobster-research"],
                skip_extras=False,
            )

        assert subprocess_ran, "subprocess.run should have been called"
        assert post_subprocess_rich_calls == [], (
            f"Rich console.print was called after subprocess: {post_subprocess_rich_calls}"
        )
