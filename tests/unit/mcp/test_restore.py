"""``lobster_continue`` by ``output_dir``: the session of an earlier server
process is read back from the folder.

The session is scripted; the folder reads (result files, session file) and the
engine's own ``resolve_session_continuation`` are the real ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lobster.cli_internal import mcp_record
from lobster.cli_internal.mcp_stdio import McpServer, ThreadRunner
from tests.unit.mcp.test_continue import ERROR, finish, go_on
from tests.unit.mcp.test_publish import H5AD, hidden, scan  # noqa: F401  (fixture)
from tests.unit.mcp.test_run import (  # noqa: F401  (fixtures)
    COMPLETE,
    call,
    data,
    engine,
    wait_terminal,
)

SESSION_FILE = "session_session_test.json"


@pytest.fixture
def earlier(engine, scan, tmp_path, data):
    """A folder an earlier server process left behind: (output_dir, run_id,
    opened sessions, script)."""
    opened, script = engine
    script["events"] = [COMPLETE]
    script["files"] = {"data/pbmc.h5ad": H5AD, SESSION_FILE: b"{}"}
    out = tmp_path / "out"
    first = ThreadRunner()
    run, done = finish(McpServer(first), data, out)
    assert done["status"] == "completed"
    first.stop(5)  # that process ended: its session no longer holds the folder
    return out, run["run_id"], opened, script


def test_the_second_process_restores_the_session_through_the_engine(earlier):
    out, first_id, opened, script = earlier
    script["files"] = {}  # the folder already holds the session
    second = McpServer(ThreadRunner())  # a new process: nothing kept in memory
    reply = go_on(second, "now cluster the cells", output_dir=str(out))
    assert (reply["continues"], reply["turn"]) == (first_id, 2)
    done = wait_terminal(second, reply["run_id"])

    assert done["status"] == "completed"
    assert Path(done["result_path"]) == out / "result_2.json"
    assert Path(done["notebook_path"]) == out / "lobster_run_2.ipynb"
    restored = opened[1]
    assert restored["workspace"] == hidden(out.parent)
    assert restored["session_id"] == "session_test"  # the engine's "latest"
    assert restored["client"].loaded == [hidden(out.parent) / SESSION_FILE]
    assert restored["client"].prompts == [
        mcp_record.compose_continue_prompt("now cluster the cells")
    ]
    record = json.loads((out / "result_2.json").read_text())
    assert record["continued_from"] == first_id and record["request"]["turn"] == 2
    assert (out / "result.json").exists()


def test_a_restored_session_is_kept_for_the_next_turn(earlier):
    out, _first_id, opened, script = earlier
    second = McpServer(ThreadRunner())
    turn_two = wait_terminal(second, go_on(second, output_dir=str(out))["run_id"])
    by_id = go_on(second, run_id=turn_two["run_id"])
    assert (by_id["continues"], by_id["turn"]) == (turn_two["run_id"], 3)
    done = wait_terminal(second, by_id["run_id"])
    assert Path(done["result_path"]) == out / "result_3.json"
    assert len(opened) == 2  # restored once, then kept


def test_the_turn_after_a_restart_follows_the_newest_result_file(earlier):
    out, _first_id, opened, script = earlier
    runner = ThreadRunner()
    server = McpServer(runner)
    wait_terminal(server, go_on(server, output_dir=str(out))["run_id"])  # restored: 2
    runner.stop(5)  # that process ended too
    third = McpServer(ThreadRunner())  # another restart
    done = wait_terminal(third, go_on(third, output_dir=str(out))["run_id"])
    assert Path(done["result_path"]) == out / "result_3.json"
    assert done["turn"] == 3


def test_a_failed_restored_turn_leaves_the_files_and_the_session_alone(earlier):
    out, _first_id, opened, script = earlier
    before = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    second = McpServer(ThreadRunner())
    opened_events = [ERROR]
    script["events"] = opened_events  # what the restored session plays
    failed = wait_terminal(second, go_on(second, output_dir=str(out))["run_id"])
    assert failed["status"] == "failed"
    assert not (out / "result_2.json").exists()
    assert hidden(out.parent).is_dir()  # earlier work is never removed
    after = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    assert {k: v for k, v in after.items() if k in before} == before


@pytest.mark.parametrize(
    "damage, reason",
    [
        ("no session file", "no saved session"),
        ("no result", "no result.json"),
        ("bad result", "cannot be read"),
        ("result without a request", "cannot be read"),
    ],
)
def test_a_folder_that_cannot_be_restored_is_refused_with_the_reason(
    earlier, damage, reason
):
    out, _first_id, opened, script = earlier
    if damage == "no session file":
        (hidden(out.parent) / SESSION_FILE).unlink()
    elif damage == "no result":
        (out / "result.json").unlink()
    elif damage == "bad result":
        (out / "result.json").write_text("{not json")
    else:
        (out / "result.json").write_text("{}")
    server = McpServer(ThreadRunner())
    refused = go_on(server, output_dir=str(out))
    assert refused["error_text"].startswith("no_session_found:")
    assert reason in refused["error_text"] and "\n" not in refused["error_text"]
    assert len(opened) == 1  # no session was opened for it


def test_an_analysis_this_server_does_not_run_is_not_restored(earlier):
    out, _first_id, opened, script = earlier
    record = json.loads((out / "result.json").read_text())
    record["request"]["analysis"] = "x\u202ey/../../ev il"  # it names a notebook
    (out / "result.json").write_text(json.dumps(record))
    refused = go_on(McpServer(ThreadRunner()), output_dir=str(out))
    assert refused["error_text"].startswith("no_session_found:")
    assert "analysis" in refused["error_text"] and "\n" not in refused["error_text"]
    assert len(opened) == 1
    assert not list(out.glob("result_*.json"))


def test_a_restore_closes_the_session_kept_for_another_folder(
    earlier, engine, scan, tmp_path, data
):
    out, _first_id, opened, script = earlier
    runner = ThreadRunner()
    server = McpServer(runner)
    other, _ = finish(server, data, tmp_path / "other")  # idle: the other folder
    assert runner.slot.peek().output_dir == tmp_path / "other"
    done = wait_terminal(server, go_on(server, output_dir=str(out))["run_id"])
    assert done["status"] == "completed"
    assert runner.slot.peek().output_dir == out
    assert go_on(server, run_id=other["run_id"])["error_text"].startswith(
        "session_closed:"
    )
