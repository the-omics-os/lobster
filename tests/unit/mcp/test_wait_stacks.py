"""Thread stacks on stderr when a ``lobster_wait`` cannot reply in time.

A wait that gets no reply leaves the host with nothing to go on. The server
writes every thread's stack to stderr then, and on SIGUSR1, so the host's MCP
log shows where it stopped. A native call that holds the GIL stops every
Python thread, so faulthandler's own watchdog thread writes the stacks then.
"""

from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from lobster.cli_internal import mcp_watch

ROOT = Path(__file__).parents[3]


def hold_the_gil(seconds):
    """A native call that keeps the GIL: no other Python thread runs meanwhile."""
    ctypes.PyDLL(None).sleep(seconds)


@pytest.fixture
def stream(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_watch, "BACKSTOP_S", 0.3)
    # appended like a host's log: the native dump and Python's writes interleave
    with open(tmp_path / "stderr.txt", "a+") as handle:
        yield handle
    mcp_watch.faulthandler.cancel_dump_traceback_later()


def written(handle):
    """The file's text once nothing more is being written to it. Read through
    a new handle: the watch may still be writing through the old one."""
    path, size = Path(handle.name), -1
    for _ in range(30):
        time.sleep(0.1)
        if path.stat().st_size == size:
            break
        size = path.stat().st_size
    return path.read_text()


def test_a_wait_that_replies_in_time_writes_nothing(stream):
    watch = mcp_watch.WaitWatch(stream)
    with watch.watching(0.3):
        time.sleep(0.05)
    time.sleep(0.8)
    assert written(stream) == ""


def test_a_late_wait_says_why_before_the_stacks(stream):
    watch = mcp_watch.WaitWatch(stream)
    with watch.watching(0.2):
        time.sleep(0.9)  # blocked, but Python still runs
    text = written(stream)
    assert text.startswith(mcp_watch.LATE_LINE + "\n"), text
    assert "test_a_late_wait_says_why_before_the_stacks" in text
    assert "Timeout (" not in text  # the backstop stood down: one dump
    assert text.count(mcp_watch.LATE_LINE) == 1


def test_stacks_are_written_while_a_native_call_holds_the_gil(stream):
    watch = mcp_watch.WaitWatch(stream)
    with watch.watching(0.1):
        hold_the_gil(1)
    text = written(stream)
    assert "Timeout (" in text, text  # faulthandler's watchdog wrote it
    assert "hold_the_gil" in text


def test_a_wait_that_ends_does_not_disarm_another_that_is_open(stream):
    watch = mcp_watch.WaitWatch(stream)
    first = watch.watching(0.1)
    first.__enter__()  # open, and never ends in this test
    with watch.watching(30):  # opens and ends while the first is still open
        pass
    hold_the_gil(1)  # no Python thread runs: only the native watchdog can write
    text = written(stream)
    assert "Timeout (" in text, text


def test_without_sigusr1_the_server_still_starts(monkeypatch):
    monkeypatch.delattr(signal, "SIGUSR1", raising=False)
    registered = []
    monkeypatch.setattr(
        mcp_watch.faulthandler, "register", lambda *a, **k: registered.append(a)
    )
    mcp_watch.stacks_on_signal()
    assert registered == []


SERVER = textwrap.dedent("""
    import ctypes
    from lobster.cli_internal import mcp_stdio, mcp_watch

    mcp_watch.GRACE_S = 0.5
    mcp_watch.BACKSTOP_S = 0.5

    class Stuck:
        calls = 0

        def status(self, run_id):
            Stuck.calls += 1
            if Stuck.calls == 2:  # inside the wait: a native call keeps the GIL
                ctypes.PyDLL(None).sleep(4)
            return {"status": "running", "run_id": run_id}

        def stop(self, timeout):
            return True

    raise SystemExit(mcp_stdio.run_mcp_stdio(runner=Stuck()))
    """)


def test_the_server_writes_stacks_to_stderr_and_only_json_rpc_to_stdout(tmp_path):
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "PYTHONPATH": str(ROOT)}
    with open(tmp_path / "stderr.txt", "w+") as err:
        proc = subprocess.Popen(
            [sys.executable, "-c", SERVER],
            cwd=tmp_path,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
        )
        requests = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "lobster_wait",
                    "arguments": {"run_id": "run_x", "timeout_s": 1},
                },
            },
        ]
        proc.stdin.write(json.dumps(requests[0]) + "\n")
        proc.stdin.flush()
        assert json.loads(proc.stdout.readline())["id"] == 1
        if hasattr(signal, "SIGUSR1"):
            proc.send_signal(signal.SIGUSR1)
            time.sleep(0.5)
        proc.stdin.write(json.dumps(requests[1]) + "\n")
        proc.stdin.flush()
        reply = json.loads(proc.stdout.readline())  # it still replies, late
        out, _ = proc.communicate("", timeout=30)
        text = written(err)
    assert reply["id"] == 2 and "result" in reply
    assert json.loads(reply["result"]["content"][0]["text"])["status"] == "running"
    assert out.strip() == ""  # every line on stdout was a JSON-RPC reply
    assert "Timeout (" in text and "in status" in text, text
    if hasattr(signal, "SIGUSR1"):
        # the signal's dump comes first
        assert text.index("most recent call first") < text.index("Timeout (")
