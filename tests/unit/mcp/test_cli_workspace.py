"""A ``.lobster_workspace`` the server did not create is never emptied.

``lobster chat`` and ``lobster query`` keep their data and sessions in
``<folder>/.lobster_workspace``, the same name the server uses for a run's
session. The server moves a session folder aside for a new run only when it
carries the server's own state file, written when the server made the folder;
any other one is refused and left exactly as it was.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from lobster.cli_internal import mcp_lock, mcp_record, mcp_turns
from lobster.cli_internal.mcp_stdio import McpServer, ThreadRunner
from tests.unit.mcp.test_run import (  # noqa: F401  (fixtures)
    COMPLETE,
    REPLAY,
    ROOT,
    Gate,
    call,
    data,
    engine,
    run_args,
    wait_terminal,
)


def tree(root: Path) -> dict:
    """Every entry under *root*: a file's sha256, a link's target, a folder."""
    found = {}
    for path in sorted(root.rglob("*")):
        key = str(path.relative_to(root))
        if path.is_symlink():
            found[key] = "link:" + os.readlink(path)
        elif path.is_file():
            found[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            found[key] = "dir"
    return found


@pytest.fixture(scope="module")
def cli_made(tmp_path_factory):
    """A folder where ``lobster query`` ran (real CLI, replay model), with a data
    file the user saved in its workspace and a script of theirs beside it."""
    base = tmp_path_factory.mktemp("cli")
    project, home = base / "project", base / "home"
    project.mkdir()
    home.mkdir()
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(
            [str(ROOT), str(ROOT / "packages" / "lobster-transcriptomics")]
        ),
        "LOBSTER_TEST_REPLAY_FIXTURE": str(REPLAY / "query.json"),
        "LOBSTER_LLM_PROVIDER": "ollama-cloud",
        "OLLAMA_API_KEY": "replay-dummy",
    }
    done = subprocess.run(  # nosec B603
        [sys.executable, "-m", "lobster", "query", "What can you do?"],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    workspace = project / mcp_record.WORKSPACE_DIR
    assert done.returncode == 0, done.stderr[-2000:]
    assert list(workspace.glob("session_*.json")), sorted(tree(workspace))
    (workspace / "data" / "my_geo_download.h5ad").write_bytes(b"user data")
    (project / "my_analysis.R").write_text("library(Seurat)\n")
    return project


@pytest.fixture
def cli_folder(cli_made, tmp_path):
    folder = tmp_path / "project"
    shutil.copytree(cli_made, folder, symlinks=True)
    return folder


def refused_text(out: Path, data: Path) -> str:
    gate = Gate()
    refused = call(McpServer(ThreadRunner(gate)), "lobster_run", run_args(data, out))
    assert gate.calls == []
    return refused["error_text"]


def test_a_cli_workspace_is_refused_and_left_byte_for_byte(cli_folder, data):
    before = tree(cli_folder)
    text = refused_text(cli_folder, data)
    assert text.startswith("output_dir_in_use:") and "\n" not in text
    assert "did not create" in text and "choose a new folder" in text
    assert tree(cli_folder) == before


def test_a_cli_workspace_is_refused_while_the_session_is_open(tmp_path, data):
    """The session writer itself, as a live ``lobster chat`` holds it."""
    from lobster.core.runtime.data_manager import DataManagerV2

    out = tmp_path / "project"
    dm = DataManagerV2(workspace_path=out / mcp_record.WORKSPACE_DIR)
    (out / mcp_record.WORKSPACE_DIR / "data" / "kept.h5ad").write_bytes(b"x")
    before = tree(out)
    assert refused_text(out, data).startswith("output_dir_in_use:")
    assert tree(out) == before
    assert dm.workspace_path.is_dir()


def test_a_linked_workspace_without_the_record_is_refused(cli_folder, tmp_path, data):
    out = tmp_path / "out"
    out.mkdir()
    (out / mcp_record.WORKSPACE_DIR).symlink_to(cli_folder / mcp_record.WORKSPACE_DIR)
    before = tree(cli_folder)
    assert refused_text(out, data).startswith("output_dir_in_use:")
    assert tree(cli_folder) == before


def test_the_lock_alone_never_moves_a_folder_it_did_not_make(tmp_path):
    """Even a caller that says "clear it" gets ``InUse`` without the record."""
    out = tmp_path / "out"
    workspace = out / mcp_record.WORKSPACE_DIR
    (workspace / "data").mkdir(parents=True)
    (workspace / "data" / "keep.h5ad").write_bytes(b"x")
    with pytest.raises(mcp_lock.InUse):
        mcp_lock.claim_new(out, lambda: True)
    assert tree(out) == {
        ".lobster_workspace": "dir",
        ".lobster_workspace/data": "dir",
        ".lobster_workspace/data/keep.h5ad": hashlib.sha256(b"x").hexdigest(),
    }


def test_a_folder_the_server_makes_carries_its_record_from_the_start(tmp_path):
    out = tmp_path / "out"
    claim = mcp_lock.claim_new(out)
    try:
        assert mcp_turns.load(out / mcp_record.WORKSPACE_DIR) == {"v": 1, "turns": []}
    finally:
        claim.release()


def test_the_fresh_folder_after_a_move_is_marked_as_the_servers(tmp_path):
    out = tmp_path / "out"
    mcp_lock.claim_new(out).release()
    workspace = out / mcp_record.WORKSPACE_DIR
    assert mcp_turns.record(workspace, 1, "failed", ["raw"])
    (workspace / "data").mkdir()
    (workspace / "data" / "left.h5ad").write_bytes(b"x")
    claim = mcp_lock.claim_new(out, lambda: mcp_record.unfinished_leftover(out))
    try:
        assert sorted(p.name for p in workspace.iterdir()) == [
            mcp_lock.LOCK_NAME,
            mcp_turns.STATE_NAME,
        ]
        assert mcp_turns.load(workspace) == {"v": 1, "turns": []}
    finally:
        claim.release()


def test_a_server_folder_without_a_completed_turn_is_reused(engine, tmp_path, data):
    _opened, script = engine
    script["events"] = [COMPLETE]
    out = tmp_path / "out"
    mcp_lock.claim_new(out).release()  # a first run that never got further
    server = McpServer(ThreadRunner())
    run = call(server, "lobster_run", run_args(data, out))
    assert wait_terminal(server, run["run_id"])["status"] == "completed"
