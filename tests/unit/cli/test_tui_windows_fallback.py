"""On Windows both TUI finders report no binary, so every caller falls back.

The Go TUI is launched with ``pass_fds``, which Windows ``Popen`` rejects.
POSIX discovery must stay as it was.
"""

import os
import stat

import pytest

from lobster.cli_internal.go_tui_launcher import find_tui_binary_fast
from lobster.ui.bridge.binary_finder import find_tui_binary

WINDOWS_LINE = "The Go TUI does not run on Windows; using the classic interface.\n"
FINDERS = [find_tui_binary, find_tui_binary_fast]


@pytest.fixture
def tui_binary(tmp_path, monkeypatch):
    binary = tmp_path / "lobster-tui"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("LOBSTER_TUI_BINARY", str(binary))
    return binary


def _on_windows(monkeypatch, call):
    with monkeypatch.context() as patched:
        patched.setattr(os, "name", "nt")
        return call()


@pytest.mark.parametrize("finder", FINDERS)
def test_windows_finder_reports_no_binary_with_one_line(
    finder, tui_binary, monkeypatch, capsys
):
    assert _on_windows(monkeypatch, finder) is None
    assert capsys.readouterr() == ("", WINDOWS_LINE)


@pytest.mark.parametrize("finder", FINDERS)
def test_posix_finder_is_unchanged(finder, tui_binary, capsys):
    assert finder() == str(tui_binary.resolve())
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("ui_mode", ["go", "auto"])
def test_windows_chat_falls_back_without_an_install_hint(ui_mode, monkeypatch, capsys):
    import lobster.cli as cli
    import lobster.cli_internal.go_tui_launcher as launcher

    monkeypatch.setattr(launcher, "find_tui_binary_fast", lambda: None)
    monkeypatch.setattr(cli, "_ensure_go_chat_tty", lambda ui_mode: None)
    monkeypatch.setattr(cli, "_should_try_go_chat_ui", lambda *args: True)

    def launch():
        return cli._maybe_launch_go_chat_ui(
            ui_mode=ui_mode,
            workspace=None,
            session_id=None,
            reasoning=False,
            verbose=False,
            debug=False,
            profile_timings=None,
            provider=None,
            model=None,
            no_intro=False,
            stream=True,
        )

    assert _on_windows(monkeypatch, launch) is False
    assert capsys.readouterr() == ("", "")
