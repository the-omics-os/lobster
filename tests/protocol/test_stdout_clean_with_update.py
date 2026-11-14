"""Machine output stays clean when a newer lobster-ai is on PyPI.

The update notice must never reach stdout ahead of protocol or JSON output.
The version cache is seeded with a newer release in a private home, and a
dead proxy keeps PyPI unreachable, so nothing here touches the network.
stderr is a terminal (a pty), as for a person: the notice is only ever shown
there, so machine modes are checked to skip it, and a control shows it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess  # nosec B404 # runs the lobster CLI under test with a fixed argv
import sys
import threading
import time
from pathlib import Path

import pytest

pty = pytest.importorskip("pty")  # POSIX only

from lobster.cli_internal.command_registry import SCHEMA_VERSION
from tests.protocol.stdio_driver import replay_env

QUIT = json.dumps({"type": "quit", "version": 1, "payload": {}}) + "\n"


def _env_with_newer_release(workspace: Path):
    env = replay_env("query", workspace)
    cache = Path(env["HOME"]) / ".lobster" / ".version_check_cache.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps({"latest_version": "99.0.0", "checked_at": time.time()}),
        encoding="utf-8",
    )
    return env


def _drain(fd, chunks):
    while True:
        try:
            data = os.read(fd, 65536)
        except OSError:  # EIO once the child's side is closed (Linux)
            return
        if not data:
            return
        chunks.append(data)


def _lobster(args, env, stdin=""):
    """Run lobster with stdout captured and stderr on a pty."""
    master, slave = pty.openpty()
    chunks = []
    reader = threading.Thread(target=_drain, args=(master, chunks), daemon=True)
    reader.start()
    try:
        proc = subprocess.run(  # nosec B603
            [sys.executable, "-m", "lobster", *args],
            input=stdin,
            stdout=subprocess.PIPE,
            stderr=slave,
            text=True,
            env=env,
            timeout=180,
        )
    finally:
        os.close(slave)
    reader.join(timeout=10)
    os.close(master)
    text = b"".join(chunks).decode("utf-8", "replace")
    proc.stderr = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)  # terminal styling
    return proc


def test_agent_stdio_first_stdout_line_is_the_handshake(tmp_path):
    workspace = tmp_path / "ws"
    env = _env_with_newer_release(workspace)
    proc = _lobster(
        ["agent", "--stdio", "--workspace", str(workspace)], env, stdin=QUIT
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    first = proc.stdout.splitlines()[0]
    assert "99.0.0" not in proc.stderr
    assert json.loads(first)["type"] == "handshake"


def test_commands_json_parses(tmp_path):
    proc = _lobster(["commands", "--json"], _env_with_newer_release(tmp_path / "ws"))
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads(proc.stdout)["schema_version"] == SCHEMA_VERSION
    assert "99.0.0" not in proc.stderr


def test_skills_list_json_parses(tmp_path):
    proc = _lobster(
        ["skills", "list", "--json"], _env_with_newer_release(tmp_path / "ws")
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    json.loads(proc.stdout)
    assert "99.0.0" not in proc.stderr


def test_control_human_output_shows_the_notice_on_stderr(tmp_path):
    # Without this, the tests above could pass without reading the cache.
    proc = _lobster(["commands"], _env_with_newer_release(tmp_path / "ws"))
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "99.0.0" in proc.stderr
    assert "99.0.0" not in proc.stdout
