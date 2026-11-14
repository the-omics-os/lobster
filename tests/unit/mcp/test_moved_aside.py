"""The server never deletes a session folder that already exists.

A folder it made whose run never finished can have gained files since (the user
ran ``lobster chat`` there, or the record was copied into someone else's
workspace). A new run therefore moves the old session folder aside, whole, to
``.lobster_workspace.unfinished-<time>`` and starts a fresh one; when the move
is not possible the run is refused and nothing is touched.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from lobster.cli_internal import mcp_lock, mcp_record, mcp_turns
from lobster.cli_internal.mcp_stdio import McpServer, ThreadRunner
from tests.unit.mcp.test_cli_workspace import cli_folder, cli_made, tree  # noqa: F401
from tests.unit.mcp.test_run import (  # noqa: F401  (fixtures)
    COMPLETE,
    ROOT,
    Gate,
    call,
    data,
    engine,
    run_args,
    wait_terminal,
)

STAMP = "20261113T191500Z"
REJECTED = {
    "type": "error",
    "error": "Error code: 401 - invalid x-api-key",
    "error_type": "AuthenticationError",
}


@pytest.fixture
def stamp(monkeypatch):
    monkeypatch.setattr(mcp_lock, "_stamp", lambda: STAMP)
    return STAMP


def moved_name(suffix=""):
    return f"{mcp_record.WORKSPACE_DIR}.unfinished-{STAMP}{suffix}"


def test_data_added_to_an_unfinished_server_folder_is_moved_not_deleted(
    engine, tmp_path, data, stamp
):
    _opened, script = engine
    out = tmp_path / "out"
    server = McpServer(ThreadRunner())
    script["events"] = [REJECTED]
    first = call(server, "lobster_run", run_args(data, out))
    assert wait_terminal(server, first["run_id"])["status"] == "failed"
    workspace = out / mcp_record.WORKSPACE_DIR
    (workspace / "data").mkdir(exist_ok=True)
    (workspace / "data" / "cli_added.h5ad").write_bytes(b"made by lobster chat")
    before = tree(workspace)

    script["events"] = [COMPLETE]
    again = call(server, "lobster_run", run_args(data, out))
    moved = out / moved_name()
    assert again["previous_workspace_moved_to"] == str(moved)
    done = wait_terminal(server, again["run_id"])
    assert done["status"] == "completed"
    assert tree(moved) == before  # every file, byte for byte
    assert done["previous_workspace_moved_to"] == str(moved)
    record = json.loads(Path(done["result_path"]).read_text())
    assert record["previous_workspace_moved_to"] == str(moved)
    assert mcp_turns.is_marked(workspace)  # the fresh one is the server's


def test_a_forged_record_in_a_cli_workspace_moves_it_whole(
    cli_folder, data, stamp  # noqa: F811
):
    workspace = cli_folder / mcp_record.WORKSPACE_DIR
    (workspace / mcp_turns.STATE_NAME).write_text('{"v": 1, "turns": []}')
    before = tree(workspace)
    gate = Gate()
    reply = call(
        McpServer(ThreadRunner(gate)), "lobster_run", run_args(data, cli_folder)
    )
    assert reply["status"] == "running"
    gate.release.set()
    assert tree(cli_folder / moved_name()) == before
    assert (cli_folder / "my_analysis.R").read_text() == "library(Seurat)\n"


def test_a_taken_name_gets_a_number(tmp_path, stamp):
    out = tmp_path / "out"
    mcp_lock.claim_new(out).release()
    (out / moved_name()).mkdir()
    (out / moved_name("-2")).mkdir()
    claim = mcp_lock.claim_new(out, lambda: True)
    try:
        assert claim.moved_to == out / moved_name("-3")
        assert (claim.moved_to / mcp_turns.STATE_NAME).exists()
    finally:
        claim.release()


def test_a_move_that_fails_refuses_and_touches_nothing(
    tmp_path, data, stamp, monkeypatch
):
    out = tmp_path / "out"
    mcp_lock.claim_new(out).release()
    workspace = out / mcp_record.WORKSPACE_DIR
    (workspace / "data").mkdir()
    (workspace / "data" / "left.h5ad").write_bytes(b"x")
    before = tree(out)

    def refuse(source, target):
        raise OSError("cross-device link")

    monkeypatch.setattr(mcp_lock.os, "rename", refuse)
    gate = Gate()
    refused = call(McpServer(ThreadRunner(gate)), "lobster_run", run_args(data, out))
    assert refused["error_text"].startswith("output_dir_in_use:")
    assert gate.calls == [] and tree(out) == before


def test_a_linked_session_folder_with_a_record_is_not_moved(tmp_path, data, stamp):
    out, elsewhere = tmp_path / "out", tmp_path / "elsewhere"
    out.mkdir()
    mcp_lock.claim_new(elsewhere).release()
    target = elsewhere / mcp_record.WORKSPACE_DIR
    (out / mcp_record.WORKSPACE_DIR).symlink_to(target)
    before = tree(tmp_path)
    refused = call(McpServer(ThreadRunner(Gate())), "lobster_run", run_args(data, out))
    assert refused["error_text"].startswith("output_dir_in_use:")
    assert tree(tmp_path) == before


def test_rmtree_is_only_used_for_a_folder_made_in_the_same_call():
    """Every ``rmtree`` of the server is ``FolderClaim.discard``'s."""
    found = []
    for path in sorted((ROOT / "lobster" / "cli_internal").glob("mcp_*.py")):
        module = ast.parse(path.read_text())
        for node in ast.walk(module):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for call_node in ast.walk(node):
                func = getattr(call_node, "func", None)
                name = getattr(func, "attr", None) or getattr(func, "id", None)
                if isinstance(call_node, ast.Call) and name == "rmtree":
                    found.append((path.name, node.name))
    assert sorted(set(found)) == [("mcp_lock.py", "discard")]
