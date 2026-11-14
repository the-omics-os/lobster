"""``lobster sessions list --json`` and ``lobster query --json``."""

import json
import os
import subprocess
import sys
from pathlib import Path

from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from lobster.cli import app

ROOT = Path(__file__).resolve().parents[3]
SCHEMAS = ROOT / "lobster" / "schemas"
REPLAY = ROOT / "tests" / "fixtures" / "protocol-v1" / "replay" / "query.json"


def _validate(payload, schema_name):
    schema = json.loads((SCHEMAS / schema_name).read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(payload)


def _session_file(workspace, session_id, conversation, mtime):
    path = workspace / f"session_{session_id}.json"
    path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "metadata": {"created_at": "2026-11-01T10:00:00"},
                "conversation": conversation,
            }
        )
    )
    os.utime(path, (mtime, mtime))


def test_sessions_list_json(tmp_path):
    long_question = "q" * 300
    _session_file(
        tmp_path,
        "old",
        [{"role": "user", "content": "first"}, {"role": "assistant", "content": "a"}],
        1_700_000_000,
    )
    _session_file(
        tmp_path,
        "new",
        [
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": long_question},
        ],
        1_800_000_000,
    )
    corrupt = tmp_path / "session_broken.json"
    corrupt.write_text("{not json")
    os.utime(corrupt, (1_750_000_000, 1_750_000_000))

    result = CliRunner().invoke(
        app, ["sessions", "list", "--json", "--workspace", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    _validate(payload, "lobster.sessions.v1.schema.json")

    sessions = payload["sessions"]
    assert [s["session_id"] for s in sessions] == ["new", "broken", "old"]
    new, broken, old = sessions
    assert new["path"] == "session_new.json"
    assert new["messages"] == 3
    assert new["last_user_message"] == "q" * 120
    assert new["created"] == "2026-11-01T10:00:00"
    assert old["last_user_message"] == "first"
    assert "error" in broken and broken["messages"] is None


def test_sessions_list_table_and_empty_workspace(tmp_path):
    runner = CliRunner()
    empty = runner.invoke(app, ["sessions", "list", "--workspace", str(tmp_path)])
    assert empty.exit_code == 0
    assert "No sessions" in empty.stdout

    _session_file(tmp_path, "abc", [{"role": "user", "content": "hi"}], 1_800_000_000)
    table = runner.invoke(app, ["sessions", "list", "--workspace", str(tmp_path)])
    assert table.exit_code == 0
    assert "abc" in table.stdout and "hi" in table.stdout


def test_query_json_is_lobster_result(tmp_path):
    workspace = tmp_path / "ws"
    env = dict(os.environ)
    env.update(
        {
            "LOBSTER_TEST_REPLAY_FIXTURE": str(REPLAY),
            "LOBSTER_LLM_PROVIDER": "ollama-cloud",
            "OLLAMA_API_KEY": "replay-dummy",
        }
    )
    proc = subprocess.run(  # nosec B603
        [
            sys.executable,
            "-m",
            "lobster",
            "query",
            "What can you do with single-cell data?",
            "--json",
            "--workspace",
            str(workspace),
        ],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env=env,
        cwd=str(tmp_path),
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    payload = json.loads(proc.stdout)
    _validate(payload, "lobster.result.v1.schema.json")
    assert payload["success"] is True  # legacy keys kept
    assert payload["run_id"].startswith("run_")
    assert "single-cell" in payload["response"]
    for key in ("duration", "token_usage", "last_agent", "has_data", "modalities"):
        assert key in payload
    # The cost of this query, beside the last-call figure the field always had.
    usage = payload["token_usage"]
    assert "latest_cost_usd" in usage and "session_total_usd" in usage
    assert isinstance(usage["run_cost_usd"], (int, float))
    assert isinstance(usage["run_total_tokens"], int)
    assert isinstance(usage["run_cost_complete"], bool)
    assert usage["run_unpriced_invocations"] >= 0
    assert usage["run_missing_usage_invocations"] >= 0


def _query_json(tmp_path, workspace, extra_env):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(home), **extra_env}
    return subprocess.run(  # nosec B603
        [sys.executable, "-m", "lobster", "query", "hi", "--json"]
        + ["--workspace", str(workspace)],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env=env,
        cwd=str(tmp_path),
        timeout=180,
    )


def test_query_json_startup_errors_are_lobster_results(tmp_path):
    not_a_dir = tmp_path / "afile"
    not_a_dir.write_text("")
    cases = {
        "invalid_provider": (
            tmp_path / "ws",
            {"LOBSTER_LLM_PROVIDER": "not-a-provider"},
        ),
        "startup_failed": (
            not_a_dir / "ws",
            {"LOBSTER_LLM_PROVIDER": "ollama-cloud", "OLLAMA_API_KEY": "x"},
        ),
    }
    for code, (workspace, extra_env) in cases.items():
        proc = _query_json(tmp_path, workspace, extra_env)
        assert proc.returncode == 1, (code, proc.stderr[-2000:])
        assert "Traceback" not in proc.stdout + proc.stderr, code
        payload = json.loads(proc.stdout)
        _validate(payload, "lobster.result.v1.schema.json")
        assert (payload["success"], payload["error_code"]) == (False, code)
        assert payload["error"] and payload["response"] == ""
        assert payload["session_id"] is None
        assert payload["run_id"].startswith("run_")
