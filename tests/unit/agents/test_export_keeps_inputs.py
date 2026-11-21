"""An export never overwrites a file the session loaded or imported.

Export tools take a model-chosen path anywhere in the workspace, and data_expert's
write_file writes model text to one: either could replace the counts file the
session was started from. A target that already exists is now refused, with the
export refusal (which now names this cause too), when it is a file the session loaded or imported in this process (by real
path), or when its size and SHA-256 match a file a load recorded (which a restored
session keeps). Overwriting the session's own earlier export is unchanged. A modality
save is not checked here (a known limit of this release).
"""

import gc
import os
import weakref
from importlib.util import find_spec

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.core.runtime.data_manager import DataManagerV2
from lobster.testing.provenance_replay import factory_tools

HAS_TX = find_spec("lobster.agents.transcriptomics") is not None
HAS_ML = find_spec("lobster.agents.machine_learning") is not None
needs_tx = pytest.mark.skipif(not HAS_TX, reason="transcriptomics package absent")
needs_ml = pytest.mark.skipif(not HAS_ML, reason="ML package absent")

DE_MODULE = "lobster.agents.transcriptomics.de_analysis_expert"
ANNOTATION_MODULE = "lobster.agents.transcriptomics.annotation_expert"
ML_MODULE = "lobster.agents.machine_learning.machine_learning_expert"
TX_MODULE = "lobster.agents.transcriptomics.transcriptomics_expert"
RESULTS_KEY = "de_results_condition_sample_b_vs_sample_a"
REFUSAL = "Error: " + (
    "this export path is inside the managed figure folder (plots/<figure>/...), "
    "or it is a file this session loaded or imported (or a copy of one). "
    "Exports cannot write there; choose a new file name under exports/ instead."
)


def _counts(path):
    pd.DataFrame(
        np.arange(12).reshape(3, 4) + 1.0,
        index=["s1", "s2", "s3"],
        columns=["g1", "g2", "g3", "g4"],
    ).to_csv(path)
    return path.read_bytes()


def _results_modality():
    adata = ad.AnnData(
        X=np.ones((6, 2)),
        obs=pd.DataFrame(
            {
                "condition": ["sample_a"] * 3 + ["sample_b"] * 3,
                "cell_type_manual": ["type_a"] * 3 + ["type_b"] * 3,
            },
            index=[f"sample_{i}" for i in range(6)],
        ),
        var=pd.DataFrame(index=["gene_a", "gene_b"]),
    )
    adata.uns["is_pseudobulk"] = True
    adata.uns[RESULTS_KEY] = {
        "results_df": pd.DataFrame(
            {"log2FoldChange": [1.0, -1.0], "padj": [0.01, 0.02]},
            index=pd.Index(["gene_a", "gene_b"], name="gene"),
        ),
        "parameters": {},
    }
    return adata


def _manager(ws):
    dm = DataManagerV2(workspace_path=ws, session_dir=ws / "session")
    dm.store_modality("de_results", _results_modality())
    dm.exports_dir.mkdir(parents=True, exist_ok=True)
    return dm


@pytest.fixture
def session(tmp_path, monkeypatch):
    """A session that loaded ``ws/inputs/counts.csv`` (and nothing else)."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    ws = tmp_path / "ws"
    dm = _manager(ws)
    source = ws / "inputs" / "counts.csv"
    source.parent.mkdir()
    original = _counts(source)
    dm.load_modality("counts", source, adapter="transcriptomics_bulk")
    return dm, source, original


def _files(dm):
    """Every file in the workspace with its bytes, the session record left out."""
    ws = dm.workspace_path
    return {
        str(p.relative_to(ws)): p.read_bytes()
        for p in sorted(ws.rglob("*"))
        if p.is_file() and "session" not in p.relative_to(ws).parts
    }


def _de_export(dm, tool_name, output_path):
    from lobster.agents.transcriptomics.de_analysis_expert import de_analysis_expert

    args = {
        "modality_name": "de_results",
        "output_format": "csv",
        "output_path": output_path,
    }
    if tool_name == "extract_and_export_de_results":
        args["shrink_lfc"] = False
    tools = factory_tools(de_analysis_expert, DE_MODULE, dm)
    return tools[tool_name].invoke(args)


DE_TOOLS = ["export_de_results", "extract_and_export_de_results"]


@needs_tx
@pytest.mark.parametrize("tool_name", DE_TOOLS)
def test_a_de_export_onto_the_loaded_file_is_refused(session, tool_name):
    dm, source, original = session
    before = _files(dm)

    reply = _de_export(dm, tool_name, str(source))

    assert reply == REFUSAL, reply
    assert source.read_bytes() == original
    assert _files(dm) == before


@needs_tx
@pytest.mark.parametrize("spelling", ["dotdot", "symlink"])
def test_another_spelling_of_the_loaded_file_is_refused(session, spelling):
    dm, source, original = session
    if spelling == "symlink":
        os.symlink(source, dm.exports_dir / "link.csv")
        target = "link.csv"
    else:
        target = "../inputs/counts.csv"  # relative to exports/
    before = _files(dm)

    reply = _de_export(dm, "export_de_results", target)

    assert reply == REFUSAL, reply
    assert source.read_bytes() == original
    assert _files(dm) == before


@needs_tx
def test_the_sessions_own_earlier_export_is_overwritten_as_before(session):
    dm, source, original = session

    first = _de_export(dm, "export_de_results", "mine.csv")
    (dm.exports_dir / "mine.csv").write_text("edited by hand\n")
    second = _de_export(dm, "export_de_results", "mine.csv")

    assert REFUSAL not in first and REFUSAL not in second, (first, second)
    assert "gene_a" in (dm.exports_dir / "mine.csv").read_text()
    assert source.read_bytes() == original


@needs_tx
def test_a_file_of_the_same_size_but_other_bytes_is_overwritten(session):
    dm, source, original = session
    other = dm.exports_dir / "same_size.csv"
    other.write_bytes(bytes(reversed(original)))

    reply = _de_export(dm, "export_de_results", str(other))

    assert REFUSAL not in reply, reply
    assert "gene_a" in other.read_text()


@needs_tx
def test_a_restored_session_refuses_by_the_recorded_size_and_sha256(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    ws = tmp_path / "ws"
    first = _manager(ws)
    source = ws / "counts.csv"
    original = _counts(source)
    first.load_modality("counts", source, adapter="transcriptomics_bulk")
    gone = weakref.ref(first)
    del first
    gc.collect()
    assert gone() is None  # the earlier process's in-memory record is gone

    restored = _manager(ws)  # a new process restores the session record
    reply = _de_export(restored, "export_de_results", str(source))

    assert reply == REFUSAL, reply
    assert source.read_bytes() == original


@needs_tx
def test_an_input_under_a_listed_export_root_is_refused(tmp_path, monkeypatch, session):
    import lobster.core.export_scope as scope

    dm, _, _ = session
    outside = (tmp_path / "outside").resolve()
    outside.mkdir()
    monkeypatch.setattr(scope, "_EXPORT_ROOTS", (outside,))
    source = outside / "counts.csv"
    original = _counts(source)
    dm.load_modality("outside_counts", source, adapter="transcriptomics_bulk")

    reply = _de_export(dm, "export_de_results", str(source))

    assert reply == REFUSAL, reply
    assert source.read_bytes() == original


@needs_tx
def test_an_annotation_export_onto_the_loaded_file_is_refused(session):
    from lobster.agents.transcriptomics.annotation_expert import annotation_expert

    dm, source, original = session
    before = _files(dm)
    tools = factory_tools(annotation_expert, ANNOTATION_MODULE, dm)

    reply = tools["export_annotation_mapping"].invoke(
        {"modality_name": "de_results", "format": "csv", "output_filename": str(source)}
    )

    assert reply == REFUSAL, reply
    assert source.read_bytes() == original
    assert _files(dm) == before


@needs_ml
def test_an_ml_export_folder_onto_the_loaded_file_is_refused(tmp_path, monkeypatch):
    from lobster.agents.machine_learning.machine_learning_expert import (
        machine_learning_expert,
    )

    monkeypatch.chdir(tmp_path)
    dm = _manager(tmp_path / "ws")
    source = dm.workspace_path / "inputs" / "de_results_ml_data.csv"
    source.parent.mkdir()
    original = _counts(source)
    dm.load_modality("counts", source, adapter="transcriptomics_bulk")
    before = _files(dm)
    tools = factory_tools(machine_learning_expert, ML_MODULE, dm)

    reply = tools["export_for_ml_framework"].invoke(
        {
            "modality_name": "de_results",
            "format": "csv",
            "include_labels": False,
            "output_dir": "../inputs",
        }
    )

    assert reply == REFUSAL, reply
    assert source.read_bytes() == original
    assert _files(dm) == before  # no other file of the export was written


def test_write_file_onto_a_loaded_project_file_is_refused(tmp_path, monkeypatch):
    """data_expert's file tools work at the project folder that holds the workspace."""
    from lobster.tools.filesystem_tools import create_filesystem_tools

    monkeypatch.chdir(tmp_path)
    project = tmp_path / "project"
    dm = _manager(project / ".lobster_workspace")
    source = project / "counts.csv"
    original = _counts(source)
    dm.load_modality("counts", source, adapter="transcriptomics_bulk")
    tools = {t.name: t for t in create_filesystem_tools(workspace_path=project)}

    reply = tools["write_file"].invoke({"path": "counts.csv", "content": "x\n"})
    other = tools["write_file"].invoke({"path": "notes.txt", "content": "x\n"})

    assert reply == REFUSAL, reply
    assert source.read_bytes() == original
    assert other.startswith("Wrote"), other


@needs_tx
def test_import_bulk_counts_records_its_file_as_an_input(tmp_path, monkeypatch):
    from lobster.agents.transcriptomics.transcriptomics_expert import (
        transcriptomics_expert,
    )

    monkeypatch.chdir(tmp_path)
    dm = _manager(tmp_path / "ws")
    source = dm.workspace_path / "bulk.csv"
    original = _counts(source)
    tools = factory_tools(transcriptomics_expert, TX_MODULE, dm)
    imported = tools["import_bulk_counts"].invoke(
        {"file_path": str(source), "modality_name": "bulk"}
    )
    assert "Successfully imported" in imported, imported

    reply = _de_export(dm, "export_de_results", str(source))

    assert reply == REFUSAL, reply
    assert source.read_bytes() == original


@needs_tx
def test_merge_sample_metadata_records_its_file_as_an_input(tmp_path, monkeypatch):
    from lobster.agents.transcriptomics.transcriptomics_expert import (
        transcriptomics_expert,
    )

    monkeypatch.chdir(tmp_path)
    dm = _manager(tmp_path / "ws")
    meta = dm.workspace_path / "samples.csv"
    pd.DataFrame(
        {"sample": [f"sample_{i}" for i in range(6)], "batch": list("aabbcc")}
    ).to_csv(meta, index=False)
    original = meta.read_bytes()
    tools = factory_tools(transcriptomics_expert, TX_MODULE, dm)
    merged = tools["merge_sample_metadata"].invoke(
        {
            "modality_name": "de_results",
            "metadata_file": str(meta),
            "sample_id_column": "sample",
        }
    )
    assert "Error" not in merged, merged

    reply = _de_export(dm, "export_de_results", str(meta))

    assert reply == REFUSAL, reply
    assert meta.read_bytes() == original


def test_a_modality_save_over_its_loaded_file_is_not_checked_here(
    tmp_path, monkeypatch
):
    """A known limit of this release: a save writes data/<name>.h5ad where told."""
    monkeypatch.chdir(tmp_path)
    dm = _manager(tmp_path / "ws")
    source = dm.data_dir / "counts.h5ad"
    source.parent.mkdir(parents=True, exist_ok=True)
    ad.AnnData(np.ones((3, 2))).write_h5ad(source)
    dm.load_modality("counts", source, adapter="transcriptomics_bulk")

    dm.save_modality("counts", "counts.h5ad")

    assert source.exists()


@needs_tx
def test_a_session_without_a_record_refuses_by_the_real_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ws = tmp_path / "ws"
    dm = DataManagerV2(workspace_path=ws, enable_provenance=False)
    dm.store_modality("de_results", _results_modality())
    source = ws / "counts.csv"
    original = _counts(source)
    dm.load_modality("counts", source, adapter="transcriptomics_bulk")

    reply = _de_export(dm, "export_de_results", str(source))

    assert reply == REFUSAL, reply
    assert source.read_bytes() == original


def test_every_file_under_an_imported_folder_is_an_input(tmp_path):
    from lobster.core import session_inputs
    from lobster.core.figures.errors import ManagedPathError
    from lobster.core.figures.export_guard import assert_export_allowed

    ws = tmp_path / "ws"
    dm = DataManagerV2(workspace_path=ws)
    for folder in ("quant", "quant2"):
        (ws / folder).mkdir()
        (ws / folder / "quant.sf").write_text("Name\tNumReads\n")
    session_inputs.note_input(dm, ws / "quant")

    with pytest.raises(ManagedPathError):
        assert_export_allowed(ws, ws / "quant" / "quant.sf")
    assert assert_export_allowed(ws, ws / "quant2" / "quant.sf") is None
    assert assert_export_allowed(ws, ws / "quant" / "new.csv") is None


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_a_file_that_cannot_be_checked_is_kept(session):
    from lobster.core.figures.errors import ManagedPathError
    from lobster.core.figures.export_guard import assert_export_allowed

    dm, source, original = session
    locked = dm.exports_dir / "locked.csv"
    locked.write_bytes(bytes(reversed(original)))  # same size: its bytes are read
    locked.chmod(0)
    try:
        with pytest.raises(ManagedPathError):
            assert_export_allowed(dm.workspace_path, locked)
    finally:
        locked.chmod(0o600)
