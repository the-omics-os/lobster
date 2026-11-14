"""mute/restore must hand the consoles back exactly as found, not pinned to a stream."""

import io

from lobster.ui.console_manager import get_console_manager


def test_restore_unpins_a_console_that_followed_sys_stdout(monkeypatch):
    manager = get_console_manager()
    # Other tests may have pinned the shared singleton; start from "follows sys.stdout".
    monkeypatch.setattr(manager.console, "_file", None)
    monkeypatch.setattr(manager.error_console, "_file", None)
    assert manager.mute_terminal_output() is True
    manager.restore_terminal_output()

    # A console with no explicit file follows whatever sys.stdout is at print time;
    # pinning it to one stream breaks every later test (or caller) that swaps stdout.
    assert manager.console._file is None
    assert manager.error_console._file is None


def test_restore_returns_an_explicit_file_unchanged(monkeypatch):
    manager = get_console_manager()
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(manager.console, "_file", out)
    monkeypatch.setattr(manager.error_console, "_file", err)
    manager.mute_terminal_output()
    manager.restore_terminal_output()
    assert (manager.console.file, manager.error_console.file) == (out, err)
