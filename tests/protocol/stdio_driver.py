"""Drive ``lobster agent --stdio`` as a host would, on a replay script."""

from __future__ import annotations

import json
import os
import queue
import subprocess  # nosec B404 # runs the lobster CLI under test with a fixed argv
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = ROOT / "lobster" / "schemas"
FIXTURES = ROOT / "tests" / "fixtures" / "protocol-v1"
REPLAY = FIXTURES / "replay"
STDIO = FIXTURES / "stdio"
SCENARIOS = ("query", "decision", "cancel", "failure")

_VOLATILE_KEYS = {
    "duration",
    "duration_ms",
    "elapsed",
    "timestamp",
    "id",
    "tool_call_id",
    "run_id",
    "resumed_by",
    "decision_id",
    "seq",
    "session_id",
    "version",
    "engine_version",
}
_TERMINAL_DONE = {"", "cancelled", "error"}


def quiet_env(workspace: Path) -> Dict[str, str]:
    """Environment that keeps the CLI's update notice off stdout.

    The notice prints ahead of the handshake whenever PyPI has a newer
    release; a private home (no cached check) and a dead proxy (no live check)
    keep the stream clean.
    """
    env = dict(os.environ)
    env.update(
        {
            "HOME": str(workspace.parent / "home"),
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "https_proxy": "http://127.0.0.1:9",
        }
    )
    return env


def replay_env(scenario: str, workspace: Path) -> Dict[str, str]:
    env = quiet_env(workspace)
    env.update(
        {
            "LOBSTER_TEST_REPLAY_FIXTURE": str(REPLAY / f"{scenario}.json"),
            # Startup requires a configured provider; the replay model answers.
            "LOBSTER_LLM_PROVIDER": "ollama-cloud",
            "OLLAMA_API_KEY": "replay-dummy",
        }
    )
    return env


def run_scenario(
    scenario: str, workspace: Path, timeout: float = 300.0
) -> List[Dict[str, Any]]:
    """Run one scenario; returns every stdout message in order."""
    spec = json.loads((REPLAY / f"{scenario}.json").read_text(encoding="utf-8"))
    answers = list(spec.get("answers", []))
    env = replay_env(scenario, workspace)
    proc = subprocess.Popen(  # nosec B603
        [
            sys.executable,
            "-m",
            "lobster",
            "agent",
            "--stdio",
            "--workspace",
            str(workspace),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
        cwd=str(workspace.parent),
    )
    lines: "queue.Queue[str | None]" = queue.Queue()

    def pump():
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()

    def write(message: Dict[str, Any]) -> None:
        proc.stdin.write(json.dumps({"version": 1, **message}) + "\n")
        proc.stdin.flush()

    messages: List[Dict[str, Any]] = []
    versions: Dict[str, int] = {}
    deadline = time.monotonic() + timeout
    cancelled = False
    try:
        while True:
            line = lines.get(timeout=max(0.1, deadline - time.monotonic()))
            if line is None:
                break
            message = json.loads(line)
            messages.append(message)
            mtype = message["type"]
            payload = message.get("payload") or {}
            if mtype == "ready":
                write({"type": "input", "payload": {"content": spec["prompt"]}})
            elif mtype == "run_state" and payload.get("state") == "running":
                if spec.get("cancel_after_s") and not cancelled:
                    cancelled = True
                    write({"type": "cancel", "payload": {}})
            elif mtype == "run_state" and payload.get("state") == "needs_decision":
                versions[payload["decision"]["decision_id"]] = payload["decision"][
                    "request_version"
                ]
            elif mtype == "component_render":
                answer = answers.pop(0) if answers else None
                write(
                    {
                        "type": "decision_answer",
                        "payload": {
                            "decision_id": message["id"],
                            "request_version": versions[message["id"]],
                            "answer": answer,
                            "source": "human",
                        },
                    }
                )
            elif mtype == "done" and payload.get("summary") in _TERMINAL_DONE:
                write({"type": "quit", "payload": {}})
            elif mtype == "alert" and payload.get("code") == "internal":
                break
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
    messages.append({"type": "__exit__", "payload": {"code": proc.returncode}})
    return messages


def normalize(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_scrub(m) for m in messages]


def _scrub(value: Any, key: str = "") -> Any:
    import re

    if isinstance(value, dict):
        return {k: _scrub(v, k) for k, v in value.items() if k not in _VOLATILE_KEYS}
    if isinstance(value, list):
        return [_scrub(v, key) for v in value]
    if key in ("text", "summary") and isinstance(value, str):
        return re.sub(r"\d+(\.\d+)?", "#", value)
    return value


def split_channels(messages: List[Dict[str, Any]]):
    """tool_execution comes from callbacks on the tool thread; order it apart."""
    tools = [m for m in messages if m["type"] == "tool_execution"]
    stream = [m for m in messages if m["type"] != "tool_execution"]
    return stream, tools
