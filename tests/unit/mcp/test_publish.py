"""``output_dir`` holds the deliverables; the session lives in a hidden folder.

The session workspace is ``<output_dir>/.lobster_workspace``. After a run the
server publishes ``result.json``, the notebook and the saved ``.h5ad`` files at
the top level and reports only those paths.
"""

from __future__ import annotations

import errno
import json
import os
import threading
from pathlib import Path

import pytest

from lobster.cli_internal import mcp_lock, mcp_record, mcp_stdio
from lobster.cli_internal.mcp_stdio import McpServer, RunRequest, ThreadRunner
from lobster.cli_internal.startup_diagnostics import (
    StartupDiagnostic,
    StartupDiagnosticError,
)
from lobster.core.runs import new_run_id
from tests.unit.mcp.test_run import (  # noqa: F401  (fixtures)
    COMPLETE,
    FakeDataManager,
    Gate,
    call,
    data,
    engine,
    run_args,
    run_engine,
    wait_terminal,
)

H5AD = b"not really hdf5"


@pytest.fixture
def scan(monkeypatch):
    """FakeDataManager lists the files under its workspace, as the real one does."""

    def files_written_since(self, since=None, **kwargs):
        entries = []
        for category in ("data", "exports", "plots"):
            for path in sorted((self.workspace / category).glob("*")):
                entries.append(
                    {
                        "path": str(path),
                        "relative": f"{category}/{path.name}",
                        "category": category,
                        "size_bytes": path.stat().st_size,
                        "modified": "2026-11-13T10:00:00+00:00",
                    }
                )
        return entries

    monkeypatch.setattr(
        FakeDataManager, "files_written_since", files_written_since, raising=False
    )


@pytest.fixture
def session_files(engine, scan):
    _opened, script = engine
    script["events"] = [COMPLETE]
    script["files"] = {
        "data/pbmc.h5ad": H5AD,
        "exports/qc.csv": b"a,b",
        "plots/p.json": b"{}",
    }
    return script


def hidden(tmp_path):
    return tmp_path / "out" / ".lobster_workspace"


def test_the_session_workspace_is_hidden_inside_output_dir(engine, tmp_path, data):
    opened, script = engine
    script["events"] = [COMPLETE]
    run_engine(tmp_path, data)
    assert opened[0]["workspace"] == hidden(tmp_path)
    assert list((tmp_path / "cwd").iterdir()) == []


def test_output_dir_holds_only_the_deliverables(session_files, tmp_path, data):
    status = run_engine(tmp_path, data)
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == [
        ".lobster_workspace",
        "lobster_run.ipynb",
        "pbmc.h5ad",
        "result.json",
    ]
    # Everything else stays in the hidden session folder.
    assert (hidden(tmp_path) / "exports" / "qc.csv").exists()
    assert (hidden(tmp_path) / "plots" / "p.json").exists()
    assert (hidden(tmp_path) / "notebooks" / "lobster_run.ipynb").exists()

    assert Path(status["result_path"]) == out / "result.json"
    assert Path(status["notebook_path"]) == out / "lobster_run.ipynb"
    assert status["files"] == [
        {
            "path": str(out / "result.json"),
            "category": "result",
            "size_bytes": (out / "result.json").stat().st_size,
        },
        {
            "path": str(out / "lobster_run.ipynb"),
            "category": "notebook",
            "size_bytes": 2,
        },
        {"path": str(out / "pbmc.h5ad"), "category": "data", "size_bytes": len(H5AD)},
    ]
    assert status["modality_details"][0]["file"] == str(out / "pbmc.h5ad")
    assert not hidden(tmp_path).as_posix() in json.dumps(status["files"])


def test_a_published_data_file_and_the_session_copy_are_independent(
    session_files, tmp_path, data
):
    run_engine(tmp_path, data)
    published = tmp_path / "out" / "pbmc.h5ad"
    original = hidden(tmp_path) / "data" / "pbmc.h5ad"
    assert os.stat(published).st_ino != os.stat(original).st_ino
    assert os.stat(published).st_nlink == 1 and published.read_bytes() == H5AD
    with open(original, "wb") as handle:  # the session rewrites its file in place
        handle.write(b"rewritten by the session")
    assert published.read_bytes() == H5AD
    with open(published, "wb") as handle:  # the user rewrites theirs in place
        handle.write(b"rewritten by the user")
    assert original.read_bytes() == b"rewritten by the session"


@pytest.mark.parametrize(
    "category, name", [("data", "a.h5ad"), ("notebook", "n.ipynb")]
)
def test_a_published_file_takes_the_mode_of_a_new_file(tmp_path, category, name):
    out = tmp_path / "out"
    out.mkdir()
    source = tmp_path / "session-copy"
    source.write_bytes(b"x")
    source.chmod(0o600)
    old = os.umask(0o022)
    try:
        [entry] = mcp_record.publish(out, [(category, source, name)])
    finally:
        os.umask(old)
    assert (out / name).stat().st_mode & 0o777 == 0o644
    assert entry["size_bytes"] == 1


def test_a_copy_that_fails_leaves_no_partial_file(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    source = tmp_path / "session-copy"
    source.write_bytes(b"x" * 100)

    def full_disk(src, dst, **kwargs):
        with open(dst, "wb") as handle:
            handle.write(b"x" * 10)
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(mcp_record.shutil, "copyfile", full_disk)
    assert mcp_record.publish(out, [("data", source, "a.h5ad")]) == []
    assert list(out.iterdir()) == []


def test_a_file_already_in_output_dir_is_never_overwritten_nor_hidden_behind_it(
    session_files, tmp_path, data
):
    out = tmp_path / "out"
    out.mkdir()
    (out / "pbmc.h5ad").write_bytes(b"the user's own file")
    (out / "lobster_run.ipynb").write_bytes(b"the user's own notebook")
    status = run_engine(tmp_path, data)
    assert (out / "pbmc.h5ad").read_bytes() == b"the user's own file"
    assert (out / "lobster_run.ipynb").read_bytes() == b"the user's own notebook"
    # What the session wrote is published under a name that is free.
    assert (out / "pbmc_2.h5ad").read_bytes() == H5AD
    assert Path(status["notebook_path"]) == out / "lobster_run_2.ipynb"
    assert [Path(f["path"]).name for f in status["files"]] == [
        "result.json",
        "lobster_run_2.ipynb",
        "pbmc_2.h5ad",
    ]
    codes = [w["code"] for w in status["warnings"]]
    assert "no_data_file_saved" not in codes and "notebook_not_exported" not in codes
    assert status["modality_details"][0]["file"] == str(out / "pbmc_2.h5ad")


def test_result_json_lists_published_paths_and_keeps_the_unpublished_ones(
    session_files, tmp_path, data
):
    status = run_engine(tmp_path, data)
    out = tmp_path / "out"
    on_disk = json.loads((out / "result.json").read_text())
    listed = {f["relative"]: f for f in on_disk["files_written"]}
    assert set(listed) == {
        "pbmc.h5ad",
        "lobster_run.ipynb",
        ".lobster_workspace/exports/qc.csv",
        ".lobster_workspace/plots/p.json",
    }
    assert listed["pbmc.h5ad"]["path"] == str(out / "pbmc.h5ad")
    assert listed["pbmc.h5ad"]["category"] == "data"
    assert listed["pbmc.h5ad"]["size_bytes"] == len(H5AD)
    assert listed[".lobster_workspace/plots/p.json"]["path"] == str(
        hidden(tmp_path) / "plots" / "p.json"
    )
    assert "files_written" not in status["result"]


# a folder that holds a run is not reused ---------------------------------------


@pytest.mark.parametrize(
    "make, continuable",
    [
        (lambda out: (out / "result.json").write_text("{}"), True),
        # No result to continue from: no lobster_continue hint.
        (lambda out: (out / ".lobster").mkdir(), False),  # the older layout
    ],
)
def test_output_dir_in_use_is_refused_before_anything_starts(
    tmp_path, data, make, continuable
):
    out = tmp_path / "out"
    out.mkdir()
    make(out)
    gate = Gate()
    server = McpServer(ThreadRunner(gate))
    refused = call(server, "lobster_run", run_args(data, out))
    text = refused["error_text"]
    assert text.startswith(f"output_dir_in_use: {out} ") and "\n" not in text
    assert "choose a new folder" in text
    assert (mcp_record.CONTINUE_HINT in text) is continuable
    assert ("no result to continue from" in text) is not continuable
    assert gate.calls == []


@pytest.mark.parametrize("prepare", ["missing", "empty", "unrelated file"])
def test_a_new_empty_or_unrelated_folder_is_fine(tmp_path, data, prepare):
    out = tmp_path / "out"
    if prepare != "missing":
        out.mkdir()
    if prepare == "unrelated file":
        (out / "notes.txt").write_text("mine")
    gate = Gate()
    gate.release.set()
    server = McpServer(ThreadRunner(gate))
    started = call(server, "lobster_run", run_args(data, out))
    assert started["status"] == "running"
    wait_terminal(server, started["run_id"])
    if prepare == "unrelated file":
        assert (out / "notes.txt").read_text() == "mine"


def test_a_run_that_completed_holds_its_folder(engine, scan, tmp_path, data):
    _opened, script = engine
    script["events"] = [COMPLETE]
    server = McpServer(ThreadRunner())
    out = tmp_path / "out"
    first = call(server, "lobster_run", run_args(data, out))
    assert wait_terminal(server, first["run_id"])["status"] == "completed"
    again = call(server, "lobster_run", run_args(data, out))
    assert again["error_text"].startswith("output_dir_in_use: ")
    elsewhere = call(server, "lobster_run", run_args(data, tmp_path / "other"))
    assert elsewhere["status"] == "running"
    wait_terminal(server, elsewhere["run_id"])


def test_a_run_that_never_started_does_not_hold_its_folder(engine, tmp_path, data):
    _opened, script = engine
    script["raise"] = StartupDiagnosticError(
        StartupDiagnostic(code="no_provider", title="No LLM provider is configured")
    )
    # The folder is the claim's, made in the same lobster_run call.
    claim = mcp_lock.claim_new(tmp_path / "out")
    request = RunRequest(
        new_run_id(),
        str(data),
        str(tmp_path / "out"),
        "What is in this file?",
        claim=claim,
    )
    status = mcp_stdio.run_engine(request, threading.Event())
    assert status["status"] == "failed"
    assert not hidden(tmp_path).exists()  # the folder can be used again
    assert not mcp_record.output_dir_in_use(tmp_path / "out")


def test_a_cancel_while_the_session_starts_leaves_the_folder_free(
    engine, monkeypatch, tmp_path, data
):
    import threading

    _opened, script = engine
    script["events"] = [COMPLETE]
    cancel = threading.Event()
    from lobster.cli_internal.commands.heavy import session_infra

    opening = session_infra.init_client_or_raise_startup_diagnostic

    def cancelling(**kwargs):
        client = opening(**kwargs)
        cancel.set()
        return client

    monkeypatch.setattr(
        session_infra, "init_client_or_raise_startup_diagnostic", cancelling
    )
    claim = mcp_lock.claim_new(tmp_path / "out")
    request = mcp_stdio.RunRequest(
        new_run_id(),
        str(data),
        str(tmp_path / "out"),
        "What is in this file?",
        claim=claim,
    )
    assert mcp_stdio.run_engine(request, cancel)["status"] == "cancelled"
    assert not hidden(tmp_path).exists()


# the real data manager keeps working with the hidden workspace ----------------


def test_a_real_workspace_inside_output_dir_lists_and_publishes_its_files(tmp_path):
    import anndata as ad
    import numpy as np

    from lobster.core.runtime.data_manager import DataManagerV2

    out = tmp_path / "out"
    dm = DataManagerV2(workspace_path=out / ".lobster_workspace")
    dm.store_modality("pbmc", ad.AnnData(np.ones((4, 3))))
    since = mcp_record.Turn.begin(dm).started_at
    dm.save_modality("pbmc", "pbmc.h5ad")
    listed = dm.files_written_since(since, categories=("data",))
    # relative to the hidden workspace: that is why the server publishes its own list
    assert [f["relative"] for f in listed] == ["data/pbmc.h5ad"]

    published = mcp_record.publish(out, mcp_record.deliverables(listed, None))
    assert [(p["category"], Path(p["path"])) for p in published] == [
        ("data", out / "pbmc.h5ad")
    ]
    assert (out / "pbmc.h5ad").stat().st_size == listed[0]["size_bytes"]


def test_every_reported_path_is_under_output_dir_as_the_host_gave_it(
    engine, tmp_path, data
):
    # A symlinked folder (/tmp on macOS) must not turn result_path into a
    # different spelling than the notebook and file paths.
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    opened, script = engine
    script["events"] = [COMPLETE]
    request = RunRequest(
        new_run_id(), str(data), str(tmp_path / "link" / "out"), "What is in this file?"
    )
    status = mcp_stdio.run_engine(request, threading.Event())
    prefix = str(tmp_path / "link" / "out")
    paths = [status["result_path"], status["notebook_path"]] + [
        f["path"] for f in status["files"]
    ]
    assert paths and all(p.startswith(prefix) for p in paths if p), paths
