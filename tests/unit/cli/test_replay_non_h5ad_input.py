"""A session that loaded a csv replays from that csv; the input is not read as h5ad."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tests.unit.cli.test_replay_commands import _clean_env, _verify_real, needs_kernel


def _csv_session(tmp_path, monkeypatch):
    """A session that loaded ``counts.csv`` through its adapter, then one step."""
    from lobster.core.provenance.analysis_ir import AnalysisStep
    from lobster.core.runtime.data_manager import DataManagerV2

    _clean_env(monkeypatch, tmp_path)
    workspace = tmp_path / "workspace"
    table = pd.DataFrame(
        np.random.default_rng(0).poisson(3, (20, 12)).astype(float),
        index=[f"c{i}" for i in range(20)],
        columns=[f"g{i}" for i in range(12)],
    )
    source = tmp_path / "counts.csv"
    table.to_csv(source)
    manager = DataManagerV2(workspace_path=workspace, auto_scan=False)
    manager.load_modality("s1", source, adapter="transcriptomics_single_cell")
    adata = manager.get_modality("s1")
    adata.X = np.log1p(adata.X)
    manager.modalities["s1"] = adata
    step = AnalysisStep(
        operation="numpy.log1p", tool_name="normalize_log1p", description="d",
        library="scanpy", code_template="adata.X = np.log1p(adata.X)",
        imports=["import numpy as np"], parameters={}, parameter_schema={},
        input_entities=["adata"], output_entities=["adata"], exportable=True,
    )  # fmt: skip
    manager.log_tool_usage("normalize_log1p", {}, "d", ir=step)
    manager.save_modality("s1", workspace / "data" / "s1.h5ad")
    notebook = Path(manager.export_notebook("nb", "S", filter_strategy="replay"))
    return workspace, notebook, source


def _tenx_h5(path, n_cells=30, n_genes=40):
    """A CellRanger-v3 style matrix file, written by hand."""
    import h5py
    import scipy.sparse as sp

    counts = sp.csc_matrix(
        np.random.default_rng(1).poisson(1.0, (n_genes, n_cells)).astype("int32")
    )
    with h5py.File(path, "w") as handle:
        matrix = handle.create_group("matrix")
        matrix["barcodes"] = np.array([f"B{i}-1".encode() for i in range(n_cells)])
        matrix["data"], matrix["indices"] = counts.data, counts.indices
        matrix["indptr"] = counts.indptr
        matrix["shape"] = np.array(counts.shape, dtype="int32")
        features = matrix.create_group("features")
        features["id"] = np.array([f"E{i}".encode() for i in range(n_genes)])
        features["name"] = np.array([f"G{i}".encode() for i in range(n_genes)])
        features["feature_type"] = np.array([b"Gene Expression"] * n_genes)
    return path


@needs_kernel
def test_a_session_that_loaded_a_csv_replays_from_that_csv(tmp_path, monkeypatch):
    workspace, notebook, source = _csv_session(tmp_path, monkeypatch)
    result = _verify_real(workspace, notebook, source)
    out = json.loads(result.stdout)
    assert out["verdict"] == "MATCH", out.get("report", {}).get("refusal")
    assert result.exit_code == 0
    assert any("not an h5ad file" in n for n in out["notes"]), out["notes"]


@needs_kernel
def test_a_different_csv_than_the_one_the_session_loaded_is_refused(
    tmp_path, monkeypatch
):
    workspace, notebook, source = _csv_session(tmp_path, monkeypatch)
    other = tmp_path / "other.csv"
    table = pd.read_csv(source, index_col=0)
    (table + 1).to_csv(other)
    result = _verify_real(workspace, notebook, other)
    out = json.loads(result.stdout)
    assert result.exit_code == 4 and out["verdict"] == "REFUSED"
    reason = out["report"]["refusal"]["reason"]
    assert "The input is not a file the session loaded" in reason


@needs_kernel
def test_a_different_h5_than_the_one_the_session_loaded_is_never_a_match(
    tmp_path, monkeypatch
):
    workspace, notebook, _source = _csv_session(tmp_path, monkeypatch)
    result = _verify_real(workspace, notebook, _tenx_h5(tmp_path / "other.h5"))
    out = json.loads(result.stdout)
    assert out["verdict"] in ("REFUSED", "NOTEBOOK_FAILED")
    assert result.exit_code != 0


@needs_kernel
def test_an_unreadable_input_is_never_a_match(tmp_path, monkeypatch):
    workspace, notebook, source = _csv_session(tmp_path, monkeypatch)
    source.write_bytes(b"\x00\x01 not a table")
    result = _verify_real(workspace, notebook, source)
    out = json.loads(result.stdout)
    assert out["verdict"] in ("REFUSED", "NOTEBOOK_FAILED")
    assert result.exit_code != 0
