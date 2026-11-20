"""Simple bulk DE saves a new file and replays to MATCH, on a fresh and a resumed session.

A resumed session loads ``data/<modality>.h5ad``: that file is the session's
recorded input, and the DE save must never write it. The result is its own
modality, ``<modality>_de_<g1>_vs_<g2>``, saved under that name, so
``lobster replay --verify`` pairs the saved file with it. Skipped without the
transcriptomics package, pyDESeq2 or a Jupyter kernel.
"""

import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from tests.unit.cli.test_replay_commands import app, needs_kernel, runner

pytest.importorskip("pydeseq2")
de_module = pytest.importorskip("lobster.agents.transcriptomics.de_analysis_expert")

import anndata as ad  # noqa: E402

from lobster.core.runtime.data_manager import DataManagerV2  # noqa: E402
from lobster.testing.provenance_replay import factory_tools  # noqa: E402

TOOLS = ["run_bulk_de_direct", "run_differential_expression"]
RESULT = "bulk_de_ctrl_vs_treat"


def _bulk():
    rng = np.random.default_rng(5)
    samples = [f"C{i}" for i in range(1, 7)] + [f"T{i}" for i in range(1, 7)]
    mu = rng.gamma(2.0, 50.0, size=300)[None, :] * np.ones((12, 1))
    mu[6:, :30] *= 4
    adata = ad.AnnData(
        X=rng.negative_binomial(5, 5 / (5 + mu)).astype(np.float32),
        obs=pd.DataFrame({"condition": ["ctrl"] * 6 + ["treat"] * 6}, index=samples),
        var=pd.DataFrame(index=[f"GENE{i:04d}" for i in range(300)]),
    )
    adata.layers["counts"] = adata.X.copy()
    return adata


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _session(tmp_path, monkeypatch, resumed, filter_genes=False):
    """Load the fixture (from data/ when resumed), run DE with a save."""
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "in" / "bulk.h5ad"
    source.parent.mkdir()
    _bulk().write_h5ad(source)
    workspace = tmp_path / "ws"
    if resumed:
        (workspace / "data").mkdir(parents=True)
        shutil.copy(source, workspace / "data" / "bulk.h5ad")
    dm = DataManagerV2(workspace_path=workspace)
    loaded_from = dm.data_dir / "bulk.h5ad" if resumed else source
    dm.load_modality("bulk", str(loaded_from), adapter="transcriptomics_bulk")
    if filter_genes:  # filtered in memory after the input was saved
        current = dm.get_modality("bulk")[:, :200].copy()
        dm.store_modality(name="bulk", adata=current, step_summary="keep 200 genes")
    return dm, source, loaded_from


def _run_de(dm, tool):
    args = {"modality_name": "bulk", "group1": "ctrl", "group2": "treat"}
    args["group_key" if tool == "run_bulk_de_direct" else "groupby"] = "condition"
    if tool == "run_differential_expression":
        args["method"] = "deseq2"
    tools = factory_tools(de_module.de_analysis_expert, de_module.__name__, dm)
    return tools[tool].invoke({**args, "save_result": True})


@pytest.mark.parametrize("tool", TOOLS)
def test_a_resumed_session_never_writes_its_input_file(tmp_path, monkeypatch, tool):
    dm, _, loaded_from = _session(tmp_path, monkeypatch, resumed=True)
    before = _sha(loaded_from)
    reply = _run_de(dm, tool)
    assert "Complete" in reply, reply
    assert _sha(loaded_from) == before
    assert sorted(p.name for p in dm.data_dir.glob("*.h5ad")) == sorted(
        ["bulk.h5ad", f"{RESULT}.h5ad"]
    )
    assert f"'{RESULT}'" in reply


@pytest.mark.parametrize("tool", TOOLS)
def test_the_result_file_is_the_current_modality_with_the_results(
    tmp_path, monkeypatch, tool
):
    dm, _, _ = _session(tmp_path, monkeypatch, resumed=True, filter_genes=True)
    _run_de(dm, tool)
    saved = ad.read_h5ad(dm.data_dir / f"{RESULT}.h5ad")
    current = dm.get_modality("bulk")
    assert saved.n_vars == 200
    assert list(saved.var_names) == list(current.var_names)
    key = current.uns["de_current_results_key"]
    assert saved.uns["de_current_results_key"] == key
    pd.testing.assert_frame_equal(
        saved.uns[key]["results_df"], current.uns[key]["results_df"]
    )


@needs_kernel
@pytest.mark.parametrize("resumed", [False, True], ids=["fresh", "resumed"])
@pytest.mark.parametrize("tool", TOOLS)
def test_the_exported_notebook_replays_to_match(tmp_path, monkeypatch, tool, resumed):
    dm, source, loaded_from = _session(tmp_path, monkeypatch, resumed=resumed)
    before = _sha(loaded_from)
    _run_de(dm, tool)
    notebook = dm.export_notebook("de", "bulk DE", filter_strategy="successful")
    result = runner.invoke(
        app,
        ["--verify", str(notebook), "--input", str(source),
         "-w", str(dm.workspace_path), "--json"],
    )  # fmt: skip
    assert result.exit_code == 0, (result.output, result.exception)
    out = json.loads(result.stdout)
    assert out["verdict"] == "MATCH", out
    assert not any("has no hash" in note for note in out["notes"]), out["notes"]
    assert _sha(loaded_from) == before
