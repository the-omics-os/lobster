"""Gate G0-1: ``lobster agent --stdio`` emits schema-valid protocol v1.

Each scenario runs the real CLI on a replay script, validates every stdout line
against ``lobster/schemas/protocol-v1.schema.json`` and compares the
normalized stream with the golden file in ``tests/fixtures/protocol-v1/stdio``.

Record new golden files only on purpose:
``LOBSTER_UPDATE_SEAM_BASELINE=1 uv run pytest tests/protocol -q``
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404
import sys
import threading

import pytest

from scripts.validate_ndjson import load_validator, validate_lines
from tests.protocol.stdio_driver import (
    REPLAY,
    SCENARIOS,
    SCHEMAS,
    STDIO,
    normalize,
    quiet_env,
    replay_env,
    run_scenario,
    split_channels,
)

EXPECTED_END = {
    "query": "completed",
    "decision": "completed",
    "cancel": "cancelled",
    "failure": "failed",
}


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_agent_stdio_scenario(scenario, tmp_path):
    messages = run_scenario(scenario, tmp_path / "workspace")
    assert messages[-1] == {"type": "__exit__", "payload": {"code": 0}}
    protocol = messages[:-1]

    validator = load_validator(SCHEMAS / "protocol-v1.schema.json")
    errors = validate_lines(validator, scenario, (json.dumps(m) for m in protocol))
    assert not errors, errors
    assert protocol[0]["type"] == "handshake"
    seqs = [m["seq"] for m in protocol]
    assert seqs == list(range(1, len(seqs) + 1))

    states = [m["payload"]["state"] for m in protocol if m["type"] == "run_state"]
    assert states[-1] == EXPECTED_END[scenario]
    if scenario == "decision":
        render = next(m for m in protocol if m["type"] == "component_render")
        assert render["id"].startswith("dec_")
        assert states == [
            "running",
            "needs_decision",
            "completed",
            "running",
            "completed",
        ]
        completed = [m for m in protocol if m["type"] == "run_state"][-1]
        assert completed["payload"]["result"]["schema_version"] == "lobster.result/1"

    # The golden file keeps one raw, schema-valid recording for consumers;
    # comparison ignores values that change on every run.
    golden = STDIO / f"{scenario}.ndjson"
    if os.environ.get("LOBSTER_UPDATE_SEAM_BASELINE") == "1":
        STDIO.mkdir(parents=True, exist_ok=True)
        golden.write_text(
            "".join(json.dumps(m, sort_keys=True) + "\n" for m in protocol)
        )
        pytest.skip(f"golden written: {golden.name}")
    recorded = [json.loads(line) for line in golden.read_text().splitlines() if line]
    for observed_part, expected_part, channel in zip(
        split_channels(normalize(protocol)),
        split_channels(normalize(recorded)),
        ("stream", "tools"),
    ):
        assert observed_part == expected_part, f"{channel} differs from {golden.name}"


def test_agent_without_stdio_points_to_agents(tmp_path):
    result = subprocess.run(  # nosec B603
        [sys.executable, "-m", "lobster", "agent"],
        capture_output=True,
        text=True,
        env=quiet_env(tmp_path / "ws"),
    )
    assert result.returncode == 2
    assert "lobster agents" in result.stderr
    assert result.stdout == ""


def test_agents_with_stdio_points_to_agent(tmp_path):
    result = subprocess.run(  # nosec B603
        [sys.executable, "-m", "lobster", "agents", "--stdio"],
        capture_output=True,
        text=True,
        env=quiet_env(tmp_path / "ws"),
    )
    assert result.returncode == 2
    assert "lobster agent --stdio" in result.stderr


# --- how `input`, `quit` and stdin EOF end a run (wire delta #2) ----------------


def _drive(scenario, workspace, groups, *, hold_open=False, timeout=300.0):
    """Write *groups* of messages to the real CLI; return (messages, exit code).

    Groups after the first wait for ``run_state running``. With *hold_open* the
    host keeps stdin open after the last group, as a host that never closes it.
    """
    proc = subprocess.Popen(  # nosec B603
        [sys.executable, "-m", "lobster", "agent", "--stdio"]
        + ["--workspace", str(workspace)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=replay_env(scenario, workspace),
        cwd=str(workspace.parent),
    )
    watchdog = threading.Timer(timeout, proc.kill)
    watchdog.start()
    pending = [[{"version": 1, **m} for m in group] for group in groups]
    messages = []
    try:
        for line in proc.stdout:
            message = json.loads(line)
            messages.append(message)
            ready = message["type"] == "ready" or (
                message["type"] == "run_state"
                and message["payload"]["state"] == "running"
            )
            if ready and pending:
                for m in pending.pop(0):
                    proc.stdin.write(json.dumps(m) + "\n")
                proc.stdin.flush()
                if not pending and not hold_open:
                    proc.stdin.close()
        code = proc.wait(timeout=30)
    finally:
        watchdog.cancel()
        proc.kill()
    return messages, code


def _states(messages):
    return [m["payload"] for m in messages if m["type"] == "run_state"]


def _done_summaries(messages):
    return [m["payload"].get("summary", "") for m in messages if m["type"] == "done"]


def _input(scenario):
    spec = json.loads((REPLAY / f"{scenario}.json").read_text(encoding="utf-8"))
    return {"type": "input", "payload": {"content": spec["prompt"]}}


QUIT = {"type": "quit", "payload": {}}


def test_input_quit_eof_lets_the_turn_finish(tmp_path):
    # The one-shot pipe: printf '<input>\n<quit>\n' | lobster agent --stdio
    messages, code = _drive("query", tmp_path / "ws", [[_input("query"), QUIT]])
    assert code == 0
    assert [s["state"] for s in _states(messages)] == ["running", "completed"]
    assert any(m["type"] == "text" for m in messages)
    assert _done_summaries(messages) == [""]


def test_input_then_eof_without_quit_cancels_the_turn(tmp_path):
    # The empty second group closes stdin once the run is in flight.
    messages, code = _drive("cancel", tmp_path / "ws", [[_input("cancel")], []])
    assert code == 0
    assert [s["state"] for s in _states(messages)] == ["running", "cancelled"]
    assert _states(messages)[-1]["reason"] == "cancel"
    assert _done_summaries(messages) == ["cancelled"]


@pytest.mark.parametrize("hold_open", [False, True])
def test_quit_cancels_a_run_that_needs_a_decision(tmp_path, hold_open):
    # Nobody can answer once quit is read: cancel at needs_decision, exit 0.
    # hold_open: the host keeps stdin open, so only quit (not EOF) can end it.
    messages, code = _drive(
        "decision", tmp_path / "ws", [[_input("decision"), QUIT]], hold_open=hold_open
    )
    assert code == 0
    states = _states(messages)
    assert [s["state"] for s in states] == ["running", "needs_decision", "cancelled"]
    assert states[-1]["reason"] == "decision_abandoned"
    assert _done_summaries(messages) == ["interrupt", "cancelled"]


def test_cancel_then_quit_is_the_fast_way_out(tmp_path):
    messages, code = _drive(
        "cancel",
        tmp_path / "ws",
        [[_input("cancel")], [{"type": "cancel", "payload": {}}, QUIT]],
    )
    assert code == 0
    assert [s["state"] for s in _states(messages)] == ["running", "cancelled"]
    assert _states(messages)[-1]["reason"] == "cancel"
