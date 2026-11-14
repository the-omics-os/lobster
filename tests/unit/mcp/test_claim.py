"""When a run claims its folder, and when the claim is given back."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from lobster.cli_internal import mcp_lock, mcp_record, mcp_turns
from lobster.cli_internal.mcp_stdio import McpServer, ThreadRunner
from lobster.cli_internal.startup_diagnostics import (
    StartupDiagnostic,
    StartupDiagnosticError,
)
from tests.unit.mcp.test_continue import finish, go_on
from tests.unit.mcp.test_publish import H5AD, scan  # noqa: F401  (fixture)
from tests.unit.mcp.test_run import (  # noqa: F401  (fixtures)
    COMPLETE,
    Gate,
    WaitForCancel,
    call,
    data,
    engine,
    run_args,
    wait_terminal,
)


def lock_of(out):
    return out / mcp_record.WORKSPACE_DIR / mcp_lock.LOCK_NAME


@pytest.fixture
def foreign():
    child = subprocess.Popen(["sleep", "60"])
    yield child.pid
    child.kill()
    child.wait()


def test_the_folder_is_claimed_before_the_run_call_answers(tmp_path, data):
    gate = Gate()  # the work blocks, so the worker has done nothing yet
    server = McpServer(ThreadRunner(gate))
    out = tmp_path / "out"
    run = call(server, "lobster_run", run_args(data, out))
    assert run["status"] == "running" and not gate.release.is_set()
    assert lock_of(out).read_text().strip() == str(os.getpid())
    gate.release.set()
    wait_terminal(server, run["run_id"])
    assert not lock_of(out).exists()  # no session kept: the claim is given back


def test_a_live_foreign_claim_refuses_a_run_and_a_continue(tmp_path, data, foreign):
    out = tmp_path / "out"
    (out / mcp_record.WORKSPACE_DIR).mkdir(parents=True)
    lock_of(out).write_text(f"{foreign}\n")
    (out / "result.json").write_text("{}")
    gate = Gate()
    server = McpServer(ThreadRunner(gate))
    run = call(server, "lobster_run", run_args(data, out))["error_text"]
    cont = go_on(server, output_dir=str(out))["error_text"]
    for text in (run, cont):
        assert text.startswith("output_dir_in_use:") and "\n" not in text
        assert f"another Lobster process (pid {foreign})" in text
    assert gate.calls == []
    assert lock_of(out).read_text().strip() == str(foreign)


def test_a_dead_claim_does_not_block_a_continue(engine, scan, tmp_path, data):
    opened, script = engine
    script["events"] = [COMPLETE]
    script["files"] = {"data/pbmc.h5ad": H5AD, "session_session_test.json": b"{}"}
    out = tmp_path / "out"
    first = ThreadRunner()
    finish(McpServer(first), data, out)
    first.stop(5)
    child = subprocess.Popen(["true"])
    child.wait()
    lock_of(out).write_text(f"{child.pid}\n")  # a process that crashed holding it
    script["files"] = {}
    server = McpServer(ThreadRunner())
    done = wait_terminal(server, go_on(server, output_dir=str(out))["run_id"])
    assert done["status"] == "completed" and done["turn"] == 2
    assert lock_of(out).read_text().strip() == str(os.getpid())  # idle session


def test_the_idle_session_holds_its_folder_until_another_run_closes_it(
    engine, scan, tmp_path, data
):
    _opened, script = engine
    script["events"] = [COMPLETE]
    script["files"] = {"data/pbmc.h5ad": H5AD, "session_session_test.json": b"{}"}
    server = McpServer(ThreadRunner())
    first, _ = finish(server, data, tmp_path / "a")
    assert lock_of(tmp_path / "a").exists()
    finish(server, data, tmp_path / "b")  # a new run closes the idle session
    assert not lock_of(tmp_path / "a").exists() and lock_of(tmp_path / "b").exists()
    server.runner.stop(5)
    assert not lock_of(tmp_path / "b").exists()  # so does the process ending


def test_a_first_run_that_was_cancelled_gives_the_folder_back_for_a_new_run(
    engine, tmp_path, data
):
    opened, script = engine
    wait = WaitForCancel()
    script["events"] = [wait]
    server = McpServer(ThreadRunner())
    out = tmp_path / "out"
    run = call(server, "lobster_run", run_args(data, out))
    assert wait.entered.wait(30) and lock_of(out).exists()
    call(server, "lobster_cancel", {"run_id": run["run_id"]})
    assert wait_terminal(server, run["run_id"])["status"] == "cancelled"
    assert not lock_of(out).exists()
    # There is no result to continue from; a new run may use the folder.
    refused = go_on(server, output_dir=str(out))["error_text"]
    assert refused.startswith("no_session_found:")
    assert not lock_of(out).exists()  # a refused continue takes nothing
    script["events"] = [COMPLETE]
    again = call(server, "lobster_run", run_args(data, out))
    assert again["status"] == "running"
    assert wait_terminal(server, again["run_id"])["status"] == "completed"


def leftover(out, name="left.txt"):
    """What a first run that never finished leaves in the session folder."""
    workspace = out / mcp_record.WORKSPACE_DIR
    (workspace / "data").mkdir(parents=True)
    mcp_turns.mark(workspace)
    (workspace / "data" / name).write_text("old")
    (workspace / ".lobster").mkdir()
    return workspace


def test_a_folder_holding_only_an_unfinished_runs_session_is_moved_aside_for_reuse(
    engine, tmp_path, data
):
    _opened, script = engine
    script["events"] = [COMPLETE]
    out = tmp_path / "out"
    out.mkdir()
    (out / "notes.txt").write_text("mine")
    workspace = leftover(out)
    server = McpServer(ThreadRunner())
    run = call(server, "lobster_run", run_args(data, out))
    assert run["status"] == "running"
    assert wait_terminal(server, run["run_id"])["status"] == "completed"
    assert not (workspace / "data" / "left.txt").exists()
    moved = Path(run["previous_workspace_moved_to"])
    assert (moved / "data" / "left.txt").read_text() == "old"
    assert (out / "notes.txt").read_text() == "mine"


@pytest.mark.parametrize(
    "extra",
    [
        lambda out: (out / "result.json").write_text("{}"),
        lambda out: (out / "result_2.json").write_text("{}"),
        lambda out: (out / ".lobster").mkdir(),
    ],
    ids=["result", "later turn result", "older layout"],
)
def test_a_folder_that_holds_a_result_is_not_cleared(tmp_path, data, extra):
    out = tmp_path / "out"
    out.mkdir()
    workspace = leftover(out)
    extra(out)
    gate = Gate()
    refused = call(McpServer(ThreadRunner(gate)), "lobster_run", run_args(data, out))
    assert refused["error_text"].startswith("output_dir_in_use:")
    assert (workspace / "data" / "left.txt").exists() and gate.calls == []


def test_a_session_folder_that_is_a_link_is_not_cleared(tmp_path, data):
    out, elsewhere = tmp_path / "out", tmp_path / "elsewhere"
    out.mkdir()
    (elsewhere / "data").mkdir(parents=True)
    (elsewhere / "data" / "keep.txt").write_text("x")
    (out / mcp_record.WORKSPACE_DIR).symlink_to(elsewhere)
    refused = call(McpServer(ThreadRunner(Gate())), "lobster_run", run_args(data, out))
    assert refused["error_text"].startswith("output_dir_in_use:")
    assert (elsewhere / "data" / "keep.txt").exists()


def test_a_session_folder_another_process_holds_is_not_cleared(tmp_path, data, foreign):
    out = tmp_path / "out"
    out.mkdir()
    workspace = leftover(out)
    lock_of(out).write_text(f"{foreign}\n")
    refused = call(McpServer(ThreadRunner(Gate())), "lobster_run", run_args(data, out))
    assert f"another Lobster process (pid {foreign})" in refused["error_text"]
    assert (workspace / "data" / "left.txt").exists()
    assert lock_of(out).read_text().strip() == str(foreign)


def test_two_runs_cannot_both_clear_one_folder(tmp_path, data):
    out = tmp_path / "out"
    out.mkdir()
    leftover(out)
    gate = Gate()
    server = McpServer(ThreadRunner(gate))
    assert call(server, "lobster_run", run_args(data, out))["status"] == "running"
    other = McpServer(ThreadRunner(Gate()))
    second = call(other, "lobster_run", run_args(data, out))
    assert second["error_text"].startswith("output_dir_in_use:")
    gate.release.set()


def test_a_run_that_never_started_leaves_no_claim_and_no_session_folder(
    engine, tmp_path, data
):
    _opened, script = engine
    script["raise"] = StartupDiagnosticError(
        StartupDiagnostic(code="no_provider", title="No LLM provider is configured")
    )
    server = McpServer(ThreadRunner())
    out = tmp_path / "out"
    run = call(server, "lobster_run", run_args(data, out))
    assert wait_terminal(server, run["run_id"])["status"] == "failed"
    assert not (out / mcp_record.WORKSPACE_DIR).exists()


def test_a_folder_that_cannot_be_made_is_a_tool_error(tmp_path, data):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    server = McpServer(ThreadRunner(Gate()))
    reply = call(server, "lobster_run", run_args(data, blocker / "out"))
    assert "error_text" in reply and "\n" not in reply["error_text"]
