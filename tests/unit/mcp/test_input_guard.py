"""A run analyses the file it was given, and no other."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from lobster.cli_internal import mcp_record, mcp_stdio
from lobster.cli_internal.mcp_stdio import McpServer, RunRequest, ThreadRunner
from lobster.core.runs import new_run_id
from tests.unit.mcp.test_publish import H5AD, hidden, scan  # noqa: F401  (fixture)
from tests.unit.mcp.test_run import (  # noqa: F401  (fixtures)
    COMPLETE,
    Act,
    Gate,
    call,
    data,
    engine,
    run_args,
    wait_terminal,
)


def load(path, name="m"):
    return Act("load_dataset", name=name, source=str(path), adapter="x")


@pytest.fixture
def other(tmp_path):
    path = tmp_path / "other.h5ad"
    path.write_bytes(b"another file")
    return path


def run_first(tmp_path, data, **extra):
    request = RunRequest(
        new_run_id(), str(data), str(tmp_path / "out"), "scrna_qc", **extra
    )
    return mcp_stdio.run_engine(request, threading.Event())


# before the run ------------------------------------------------------------


def test_a_path_that_is_not_a_file_or_folder_is_refused(tmp_path, data):
    if not hasattr(os, "mkfifo"):
        pytest.skip("no named pipes")
    pipe = tmp_path / "pipe.h5ad"
    os.mkfifo(pipe)
    server = McpServer(ThreadRunner(Gate()))
    refused = call(server, "lobster_run", run_args(pipe, tmp_path / "out"))
    assert "regular file or a folder" in refused["error_text"]
    folder = tmp_path / "matrix"
    folder.mkdir()
    assert "run_id" in call(server, "lobster_run", run_args(folder, tmp_path / "o2"))


# after the turn ------------------------------------------------------------


def test_another_file_loaded_in_place_of_the_input_fails_the_run(
    engine, tmp_path, data, other
):
    _opened, script = engine
    script["events"] = [load(data, "notes"), load(other, "pbmc"), COMPLETE]
    status = run_first(tmp_path, data)
    assert status["status"] == "failed"
    assert status["error"]["code"] == "other_file_loaded"
    # The summary is cut to one short line; the message holds both paths whole.
    assert status["summary"].startswith("other_file_loaded: ")
    message = status["error"]["message"]
    assert str(data) in message and str(other) in message
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == [".lobster_workspace"]


def test_only_another_file_loaded_fails_the_run(engine, tmp_path, data, other):
    _opened, script = engine
    script["events"] = [load(other), COMPLETE]
    assert run_first(tmp_path, data)["error"]["code"] == "other_file_loaded"


@pytest.mark.parametrize(
    "steps",
    [
        [],  # nothing was loaded
        [Act("load_dataset", name="m", source="AnnData object", adapter="x")],
    ],
    ids=["no load", "an object made in memory"],
)
def test_a_first_run_that_loaded_no_data_from_its_input_fails(
    engine, tmp_path, data, steps
):
    _opened, script = engine
    script["loads_input"] = False
    script["events"] = [*steps, {**COMPLETE, "response": "All is well."}]
    status = run_first(tmp_path, data)
    assert status["status"] == "failed"
    assert status["error"]["code"] == "input_not_loaded"
    assert f"The file at {data} was not loaded" in status["summary"]
    assert "nothing was analysed" in status["summary"]
    assert status["notes"] == "All is well."
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        ".lobster_workspace"
    ]


def test_a_run_that_loaded_its_input_among_other_work_is_not_failed(
    engine, tmp_path, data
):
    _opened, script = engine
    script["events"] = [
        Act("load_dataset", name="x", source="AnnData object"),
        COMPLETE,
    ]
    status = run_first(tmp_path, data)  # the scripted client loads the input first
    assert status["status"] == "completed" and "notes" not in status


def test_a_session_with_no_tracker_is_not_judged(engine, tmp_path, data):
    _opened, script = engine
    script["activities"] = None
    script["events"] = [COMPLETE]
    assert run_first(tmp_path, data)["status"] == "completed"


def test_the_input_and_the_sessions_own_files_are_not_foreign(engine, tmp_path, data):
    _opened, script = engine
    inside = tmp_path / "out" / ".lobster_workspace" / "data" / "geo.h5ad"
    relative = "data/saved.h5ad"  # the engine resolves it against the workspace
    link = tmp_path / "link.h5ad"
    link.symlink_to(data)
    script["events"] = [
        load(link),
        load(data),
        load(inside),
        load(relative),
        COMPLETE,
    ]
    assert run_first(tmp_path, data)["status"] == "completed"


def test_a_copy_of_the_input_is_the_input(engine, tmp_path, data):
    """The model may copy the file into the output folder and read the copy."""
    _opened, script = engine
    data.write_bytes(b"counts" * 100)
    copy = tmp_path / "out" / "copy.h5ad"
    copy.parent.mkdir()
    copy.write_bytes(data.read_bytes())
    script["events"] = [load(copy), COMPLETE]
    assert run_first(tmp_path, data)["status"] == "completed"


def test_a_file_of_the_same_size_but_other_content_is_another_file(
    engine, tmp_path, data
):
    _opened, script = engine
    data.write_bytes(b"a" * 600)
    other = tmp_path / "other.h5ad"
    other.write_bytes(b"b" * 600)
    script["events"] = [load(other), COMPLETE]
    assert run_first(tmp_path, data)["error"]["code"] == "other_file_loaded"


def test_a_file_inside_an_input_folder_is_part_of_the_input(engine, tmp_path):
    _opened, script = engine
    folder = tmp_path / "matrix"
    folder.mkdir()
    script["events"] = [
        load(folder / "matrix.mtx"),
        load(folder / "genes.tsv"),
        COMPLETE,
    ]
    assert run_first(tmp_path, folder)["status"] == "completed"


# a later turn --------------------------------------------------------------


@pytest.fixture
def idle(engine, scan, tmp_path, data):
    opened, script = engine
    script["events"] = [COMPLETE]
    script["files"] = {"data/pbmc.h5ad": H5AD}
    server = McpServer(ThreadRunner())
    run = call(server, "lobster_run", run_args(data, tmp_path / "out"))
    assert wait_terminal(server, run["run_id"])["status"] == "completed"
    return server, opened, run["run_id"]


def test_a_later_turn_may_load_another_file_only_if_the_instruction_names_it(
    idle, other
):
    server, opened, first_id = idle
    client = opened[0]["client"]
    client.events = [load(other), COMPLETE]
    reply = call(
        server,
        "lobster_continue",
        {"run_id": first_id, "instruction": "also look at the cells"},
    )
    failed = wait_terminal(server, reply["run_id"])
    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "other_file_loaded"
    assert failed["summary"].startswith("other_file_loaded: ")
    assert str(other) in failed["error"]["message"]
    # The failed turn is on the record for the next one.
    notes = [a for a in client.data_manager.provenance.activities]
    assert notes[-1]["parameters"] == {
        "outcome": "failed",
        "steps_completed": "load_dataset",
    }

    client.events = [load(other), COMPLETE]
    reply = call(
        server,
        "lobster_continue",
        {"run_id": first_id, "instruction": f"compare with {other}"},
    )
    assert wait_terminal(server, reply["run_id"])["status"] == "completed"


def test_the_recipe_says_to_use_only_the_input_file():
    from lobster.cli_internal import mcp_contract

    recipe = mcp_contract.ANALYSIS_PROMPTS["scrna_qc"]
    assert "Analyse only the file at the input path" in recipe
    assert "stop and say so" in recipe
