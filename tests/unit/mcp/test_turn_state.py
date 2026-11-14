"""The server's own record of how each turn ended, and what reads it: a folder
is moved aside for a new run only when no turn of its session completed."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lobster.cli_internal import mcp_record, mcp_turns
from lobster.cli_internal.mcp_stdio import McpServer, ThreadRunner
from tests.unit.mcp.test_continue import ERROR, finish, go_on, started  # noqa: F401
from tests.unit.mcp.test_publish import H5AD, hidden, scan  # noqa: F401  (fixture)
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


def workspace_of(out: Path) -> Path:
    return out / mcp_record.WORKSPACE_DIR


# a finished run whose result file is gone -------------------------------------


def test_a_finished_run_whose_result_file_was_moved_is_not_cleared(started, data):
    server, _opened, _script, out, _first_id = started
    server.runner.stop(5)  # the process that ran it ended
    (out / "result.json").rename(out.parent / "moved.json")
    kept = workspace_of(out) / "data" / "pbmc.h5ad"
    assert kept.exists()
    gate = Gate()
    refused = call(McpServer(ThreadRunner(gate)), "lobster_run", run_args(data, out))
    assert refused["error_text"].startswith("output_dir_in_use:")
    assert "\n" not in refused["error_text"] and gate.calls == []
    assert kept.exists()  # the session was not erased


def test_a_finished_run_is_not_cleared_even_if_every_result_file_is_gone(started, data):
    server, _opened, _script, out, first_id = started
    done = wait_terminal(server, go_on(server, run_id=first_id)["run_id"])
    assert done["status"] == "completed"
    server.runner.stop(5)
    for name in ("result.json", "result_2.json"):
        (out / name).unlink()
    refused = call(McpServer(ThreadRunner(Gate())), "lobster_run", run_args(data, out))
    assert refused["error_text"].startswith("output_dir_in_use:")
    assert (workspace_of(out) / "data" / "pbmc.h5ad").exists()


def test_the_message_says_why_the_folder_is_refused(started, data):
    server, _opened, _script, out, _first_id = started
    server.runner.stop(5)
    (out / "result.json").unlink()
    refused = call(McpServer(ThreadRunner(Gate())), "lobster_run", run_args(data, out))
    assert "result file is not there" in refused["error_text"]


# every ending is on the record ------------------------------------------------


def turns_of(out: Path) -> list:
    return [
        (t["turn"], t["outcome"]) for t in mcp_turns.load(workspace_of(out))["turns"]
    ]


def test_each_ending_of_a_turn_is_recorded_with_the_modalities_then(started):
    server, opened, _script, out, first_id = started
    assert turns_of(out) == [(1, "completed")]
    (entry,) = mcp_turns.load(workspace_of(out))["turns"]
    assert entry["modalities"] == ["pbmc"]
    opened[0]["client"].events = [ERROR]
    wait_terminal(server, go_on(server, run_id=first_id)["run_id"])
    wait = WaitForCancel()
    opened[0]["client"].events = [wait]
    run_id = go_on(server, run_id=first_id)["run_id"]
    assert wait.entered.wait(30)
    call(server, "lobster_cancel", {"run_id": run_id})
    wait_terminal(server, run_id)
    opened[0]["client"].events = [COMPLETE]
    wait_terminal(server, go_on(server, run_id=first_id)["run_id"])
    assert turns_of(out) == [
        (1, "completed"),
        (2, "failed"),
        (2, "cancelled"),
        (2, "completed"),
    ]


@pytest.mark.parametrize("where", ["publish", "result"])
def test_a_turn_is_completed_only_once_its_files_are_out(
    engine, tmp_path, data, monkeypatch, where
):
    _opened, script = engine
    script["events"] = [COMPLETE]

    def refuse(*args, **kwargs):
        raise OSError("disk full")

    if where == "publish":
        monkeypatch.setattr(mcp_record, "publish", refuse)
    else:
        real = Path.write_text
        monkeypatch.setattr(
            Path,
            "write_text",
            lambda self, *a, **k: (
                refuse() if self.name == "result.json" else real(self, *a, **k)
            ),
        )
    server = McpServer(ThreadRunner())
    out = tmp_path / "out"
    run = call(server, "lobster_run", run_args(data, out))
    assert wait_terminal(server, run["run_id"])["status"] == "failed"
    assert turns_of(out) == [(1, "failed")]
    monkeypatch.undo()
    script["events"] = [COMPLETE]
    again = call(McpServer(ThreadRunner()), "lobster_run", run_args(data, out))
    assert again["status"] == "running"  # nothing finished: the folder is reused


def test_a_run_that_raises_is_recorded_as_failed(engine, tmp_path, data):
    _opened, script = engine
    script["events"] = [RuntimeError("model call crashed")]
    server = McpServer(ThreadRunner())
    out = tmp_path / "out"
    run = call(server, "lobster_run", run_args(data, out))
    assert wait_terminal(server, run["run_id"])["status"] == "failed"
    assert turns_of(out) == [(1, "failed")]


def test_a_first_run_that_ended_without_a_turn_finishing_is_cleared_for_reuse(
    engine, tmp_path, data
):
    _opened, script = engine
    script["loads_input"] = False
    script["events"] = [COMPLETE]
    server = McpServer(ThreadRunner())
    out = tmp_path / "out"
    run = call(server, "lobster_run", run_args(data, out))
    assert wait_terminal(server, run["run_id"])["error"]["code"] == "input_not_loaded"
    assert turns_of(out) == [(1, "failed")]
    script["loads_input"] = True
    again = call(server, "lobster_run", run_args(data, out))
    assert wait_terminal(server, again["run_id"])["status"] == "completed"


# the reuse rule ---------------------------------------------------------------


def session(out: Path, *turns) -> Path:
    workspace = workspace_of(out)
    (workspace / "data").mkdir(parents=True)
    (workspace / "data" / "old.txt").write_text("old")
    for number, outcome in turns:
        assert mcp_turns.record(workspace, number, outcome, ["raw"])
    return workspace


def test_a_folder_is_reusable_only_without_a_completed_turn_and_a_result(tmp_path):
    cases = {
        "no record (not made by this server: the CLI's, an older build's)": (
            (),
            False,
        ),
        "a first run that failed": (((1, "failed"),), True),
        "a first run that was cancelled": (((1, "cancelled"),), True),
        "a turn that completed": (((1, "completed"),), False),
        "a completed turn among others": (((1, "failed"), (2, "completed")), False),
    }
    for n, (label, (turns, reusable)) in enumerate(cases.items()):
        out = tmp_path / f"out{n}"
        session(out, *turns)
        assert mcp_record.unfinished_leftover(out) is reusable, label


@pytest.mark.parametrize(
    "name", ["result.json", "result_2.json", "result_final.json", "result.json.bak"]
)
def test_a_result_file_of_any_name_keeps_the_folder(tmp_path, name):
    out = tmp_path / "out"
    session(out, (1, "failed"))
    assert mcp_record.unfinished_leftover(out)
    (out / name).write_text("{}")
    # The glob result*.json does not match a name that ends in .bak.
    assert mcp_record.unfinished_leftover(out) is (name == "result.json.bak")


def test_a_record_that_cannot_be_read_keeps_the_folder(tmp_path):
    out = tmp_path / "out"
    workspace = session(out)
    for text in ("{not json", "[]", json.dumps({"v": 1, "turns": [{"turn": 1}]})):
        (workspace / mcp_turns.STATE_NAME).write_text(text)
        assert mcp_record.unfinished_leftover(out) is False, text


def test_a_record_that_is_a_link_is_not_followed(tmp_path):
    out = tmp_path / "out"
    workspace = session(out)
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text(json.dumps({"v": 1, "turns": []}))
    (workspace / mcp_turns.STATE_NAME).symlink_to(elsewhere)
    assert mcp_record.unfinished_leftover(out) is False
    assert mcp_turns.record(workspace, 1, "failed", [])  # replaced, not written through
    assert json.loads(elsewhere.read_text()) == {"v": 1, "turns": []}
    assert not (workspace / mcp_turns.STATE_NAME).is_symlink()


def test_the_record_is_not_written_into_a_link_or_a_missing_folder(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    link = tmp_path / "link"
    link.symlink_to(elsewhere)
    assert mcp_turns.record(link, 1, "failed", []) is False
    assert mcp_turns.record(tmp_path / "gone", 1, "failed", []) is False
    assert list(elsewhere.iterdir()) == []


def test_clearing_never_follows_a_link_or_leaves_the_session_folder(
    engine, tmp_path, data
):
    _opened, script = engine
    script["events"] = [COMPLETE]
    out, outside = tmp_path / "out", tmp_path / "outside"
    (outside / "sub").mkdir(parents=True)
    (outside / "sub" / "keep.txt").write_text("mine")
    (outside / "keep.txt").write_text("mine")
    workspace = session(out, (1, "failed"))
    (workspace / "to_dir").symlink_to(outside)
    (workspace / "to_file").symlink_to(outside / "keep.txt")
    (workspace / "data" / "inner").symlink_to(outside / "sub")
    (out / "mine.txt").write_text("user file")
    server = McpServer(ThreadRunner())
    run = call(server, "lobster_run", run_args(data, out))
    assert wait_terminal(server, run["run_id"])["status"] == "completed"
    assert sorted(p.name for p in outside.rglob("*")) == ["keep.txt", "keep.txt", "sub"]
    assert (out / "mine.txt").read_text() == "user file"
    assert not (workspace / "data" / "old.txt").exists()


def test_a_session_folder_that_is_a_link_is_refused_and_left_alone(tmp_path, data):
    out, elsewhere = tmp_path / "out", tmp_path / "elsewhere"
    out.mkdir()
    elsewhere.mkdir()
    (elsewhere / "keep.txt").write_text("x")
    mcp_turns.record(elsewhere, 1, "failed", [])
    workspace_of(out).symlink_to(elsewhere)
    assert mcp_record.unfinished_leftover(out) is False
    refused = call(McpServer(ThreadRunner(Gate())), "lobster_run", run_args(data, out))
    assert refused["error_text"].startswith("output_dir_in_use:")
    assert sorted(p.name for p in elsewhere.iterdir()) == [
        "keep.txt",
        mcp_turns.STATE_NAME,
    ]


# the record itself ------------------------------------------------------------


def test_the_record_keeps_the_newest_turns_and_the_names_it_is_given(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    for n in range(mcp_turns.MAX_TURNS_KEPT + 5):
        assert mcp_turns.record(workspace, n + 1, "completed", ["a"] * 3)
    state = mcp_turns.load(workspace)
    assert len(state["turns"]) == mcp_turns.MAX_TURNS_KEPT
    assert state["turns"][-1]["turn"] == mcp_turns.MAX_TURNS_KEPT + 5
    assert mcp_turns.last_modalities(state) == ["a"] * 3
    mcp_turns.record(workspace, 1, "failed", [str(i) for i in range(5000)])
    assert len(mcp_turns.last_modalities(mcp_turns.load(workspace))) == (
        mcp_turns.MAX_NAMES_KEPT
    )
    assert [p.name for p in workspace.iterdir()] == [mcp_turns.STATE_NAME]


def test_a_write_that_fails_says_so_and_leaves_no_partial_file(
    tmp_path, monkeypatch, capsys
):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def refuse(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", refuse)
    assert mcp_turns.record(workspace, 1, "failed", []) is False
    assert list(workspace.iterdir()) == []
    assert "not written" in capsys.readouterr().err
