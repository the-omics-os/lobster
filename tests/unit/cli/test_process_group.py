"""Go TUI child process-group handling on POSIX and Windows."""

from __future__ import annotations

import os
import signal
import subprocess  # nosec B404

import pytest

from lobster.cli_internal import process_group as pg


class FakeProc:
    pid = 4321

    def __init__(self):
        self.calls = []

    def send_signal(self, sig):
        self.calls.append(("send_signal", sig))

    def kill(self):
        self.calls.append(("kill",))


@pytest.fixture
def posix(monkeypatch):
    monkeypatch.setattr(pg.sys, "platform", "linux")
    sent = []
    monkeypatch.setattr(pg.os, "getpgid", lambda pid: pid + 1)
    monkeypatch.setattr(pg.os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))
    return sent


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(pg.sys, "platform", "win32")
    # Windows-only constants are absent on POSIX hosts.
    monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)
    monkeypatch.setattr(signal, "CTRL_BREAK_EVENT", 1, raising=False)

    def no_killpg(*_):
        raise AssertionError("os.killpg on Windows")

    monkeypatch.setattr(pg.os, "killpg", no_killpg)


def test_posix_starts_new_session(posix):
    assert pg.new_group_popen_kwargs() == {"preexec_fn": os.setsid}


def test_posix_signals_whole_group(posix):
    proc = FakeProc()
    pg.signal_group(proc)
    pg.signal_group(proc, force=True)
    assert posix == [(4322, signal.SIGTERM), (4322, signal.SIGKILL)]
    assert proc.calls == []


def test_windows_starts_new_process_group(windows):
    assert pg.new_group_popen_kwargs() == {"creationflags": 0x200}


def test_windows_breaks_then_kills(windows):
    proc = FakeProc()
    pg.signal_group(proc)
    pg.signal_group(proc, force=True)
    assert proc.calls == [("send_signal", 1), ("kill",)]
