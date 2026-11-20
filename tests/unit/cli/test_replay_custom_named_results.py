"""A custom-code cell replays on what a step made under its own name.

Some steps record their result only by a variable (``adata_with_markers``,
``adata_filtered``, ...), not by the modality the session stored it under:
``find_marker_genes`` stores ``<modality>_markers``, ``filter_bulk_genes``
stores ``<modality>_filtered`` and bulk DE stores ``<modality>_de_<g1>_vs_<g2>``.
A custom cell that read one of these (by ``data/<name>.h5ad``, or as its
``modality_name``) stopped the replay with "which this notebook does not
compute", where the notebook does compute it. Each row is a native session
exported and replayed; skipped without the transcriptomics package, pyDESeq2
or a Jupyter kernel.
"""

import asyncio
import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from tests.unit.cli.test_replay_commands import app, needs_kernel, runner

te = pytest.importorskip("lobster.agents.transcriptomics.transcriptomics_expert")
pytest.importorskip("pydeseq2")
de_module = pytest.importorskip("lobster.agents.transcriptomics.de_analysis_expert")

import anndata as ad  # noqa: E402

from lobster.core.runtime.data_manager import DataManagerV2  # noqa: E402
from lobster.services.execution.custom_code_execution_service import (  # noqa: E402
    CustomCodeExecutionService,
)
from lobster.testing.provenance_replay import factory_tools  # noqa: E402
from lobster.tools.custom_code_tool import create_execute_custom_code_tool  # noqa: E402

pytestmark = [pytest.mark.unit, needs_kernel]

STOP = "which this notebook does not compute"


def _custom(dm, code, agent, modality_name=None):
    tool = create_execute_custom_code_tool(
        dm, CustomCodeExecutionService(dm), agent_name=agent
    )
    args = {"python_code": code, "persist": True, "description": "custom step"}
    if modality_name:
        args["modality_name"] = modality_name
    call = {"args": args, "id": "c1", "name": tool.name, "type": "tool_call"}
    reply = json.loads(asyncio.run(tool.ainvoke(call)).content)
    assert reply["success"], reply


def _replay(dm, source):
    notebook = dm.export_notebook("custom", "custom", filter_strategy="replay")
    result = runner.invoke(
        app,
        ["--verify", str(notebook), "--input", str(source),
         "-w", str(dm.workspace_path), "--json"],
    )  # fmt: skip
    return json.loads(result.stdout)


def _ran(out):
    """The notebook ran through: no stop on a name it computes."""
    assert not any(STOP in note for note in out["notes"]), out["notes"]
    assert out["verdict"] != "NOTEBOOK_FAILED", out


def _diffs(out):
    return [d for slot in out["report"]["slots"] for d in slot.get("diffs", [])]


def _cells(tmp_path, monkeypatch):
    """Single cell: QC, filter and normalize, cluster, find_marker_genes."""
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(5)
    genes = [f"MT-{i}" for i in range(6)] + [f"G{i}" for i in range(194)]
    base = rng.gamma(0.6, 2.0, size=(300, 200))
    base[:150, 10:40] *= 4
    base[150:, 40:70] *= 4
    source = tmp_path / "in" / "s.h5ad"
    source.parent.mkdir()
    ad.AnnData(
        X=sparse.csr_matrix(rng.poisson(base).astype(np.float32)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(300)]),
        var=pd.DataFrame(index=genes),
    ).write_h5ad(source)
    dm = DataManagerV2(workspace_path=tmp_path / "ws", auto_scan=False)
    dm.load_modality("s", str(source), adapter="transcriptomics_single_cell")
    tools = factory_tools(te.transcriptomics_expert, te.__name__, dm)
    tools["filter_and_normalize"].invoke(
        {"modality_name": "s", "min_genes_per_cell": 20, "max_genes_per_cell": 200,
         "min_cells_per_gene": 3, "max_mito_percent": 100, "save_result": True}
    )  # fmt: skip
    normalized = dm.list_modalities()[-1]
    tools["cluster_cells"].invoke(
        {"modality_name": normalized, "resolution": 0.5, "n_pcs": 10,
         "n_neighbors": 10, "n_features": 100, "save_result": True}
    )  # fmt: skip
    clustered = dm.list_modalities()[-1]
    key = [c for c in dm.get_modality(clustered).obs.columns if "leiden" in c][0]
    tools["find_marker_genes"].invoke(
        {"modality_name": clustered, "groupby": key, "n_genes": 5,
         "min_fold_change": 0.0, "min_pct": 0.0, "max_out_pct": 1.0,
         "save_result": True}
    )  # fmt: skip
    markers = f"{clustered}_markers"
    assert (dm.data_dir / f"{markers}.h5ad").exists()
    return dm, source, markers


def test_a_read_of_the_saved_markers_replays(tmp_path, monkeypatch):
    dm, source, markers = _cells(tmp_path, monkeypatch)
    _custom(
        dm,
        "import scanpy as sc\n"
        f"m = sc.read_h5ad('data/{markers}.h5ad')\n"
        "names = m.uns['rank_genes_groups']['names']\n"
        "result = {'n_groups': len(names.dtype.names), 'first': str(names[0][0])}\n",
        "transcriptomics_expert",
    )
    out = _replay(dm, source)
    _ran(out)
    # MATCH, or only the PCA variances' float noise every clustered session shows
    assert all("/uns/pca/variance" in d for d in _diffs(out)), _diffs(out)


def _bulk(tmp_path, monkeypatch, resumed=False):
    """The bulk fixture, loaded from ``data/bulk.h5ad`` when ``resumed``."""
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(5)
    samples = [f"C{i}" for i in range(1, 7)] + [f"T{i}" for i in range(1, 7)]
    mu = rng.gamma(2.0, 50.0, size=300)[None, :] * np.ones((12, 1))
    mu[6:, :30] *= 4
    counts = rng.negative_binomial(5, 5 / (5 + mu)).astype(np.float32)
    counts[:, -20:] = 0  # genes the filter removes
    adata = ad.AnnData(
        X=counts,
        obs=pd.DataFrame({"condition": ["ctrl"] * 6 + ["treat"] * 6}, index=samples),
        var=pd.DataFrame(index=[f"GENE{i:04d}" for i in range(300)]),
    )
    adata.layers["counts"] = adata.X.copy()
    source = tmp_path / "in" / "bulk.h5ad"
    source.parent.mkdir()
    adata.write_h5ad(source)
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    loaded = source
    if resumed:
        loaded = dm.data_dir / "bulk.h5ad"
        loaded.write_bytes(source.read_bytes())
    dm.load_modality("bulk", str(loaded), adapter="transcriptomics_bulk")
    return dm, source


def _filter_bulk(dm):
    tools = factory_tools(te.transcriptomics_expert, te.__name__, dm)
    reply = tools["filter_bulk_genes"].invoke(
        {"modality_name": "bulk", "min_counts": 10, "min_samples": 3}
    )
    assert "bulk_filtered" in dm.list_modalities(), reply
    dm.save_modality("bulk_filtered", "bulk_filtered.h5ad")


def test_a_read_of_the_filtered_bulk_replays_to_match(tmp_path, monkeypatch):
    dm, source = _bulk(tmp_path, monkeypatch)
    _filter_bulk(dm)
    _custom(
        dm,
        "import anndata\n"
        "f = anndata.read_h5ad('data/bulk_filtered.h5ad')\n"
        "result = {'n_vars': int(f.n_vars)}\n",
        "transcriptomics_expert",
    )
    out = _replay(dm, source)
    _ran(out)
    assert out["verdict"] == "MATCH", out


def test_a_custom_step_on_the_filtered_bulk_replays_to_match(tmp_path, monkeypatch):
    """rqea's shape C: the cell runs on bulk_filtered and reads the loaded input."""
    dm, source = _bulk(tmp_path, monkeypatch, resumed=True)
    _filter_bulk(dm)
    _custom(
        dm,
        "import anndata\nimport numpy as np\n"
        "raw = anndata.read_h5ad(WORKSPACE / 'data' / 'bulk.h5ad')\n"
        "adata.obs['raw_total'] = np.asarray(raw[:, adata.var_names].X.sum(axis=1)).ravel()\n",
        "transcriptomics_expert",
        modality_name="bulk_filtered",
    )
    out = _replay(dm, source)
    _ran(out)
    assert out["verdict"] == "MATCH", out


def test_a_read_of_the_bulk_de_result_replays_to_match(tmp_path, monkeypatch):
    dm, source = _bulk(tmp_path, monkeypatch)
    tools = factory_tools(de_module.de_analysis_expert, de_module.__name__, dm)
    reply = tools["run_bulk_de_direct"].invoke(
        {"modality_name": "bulk", "group_key": "condition", "group1": "ctrl",
         "group2": "treat", "save_result": True}
    )  # fmt: skip
    assert "bulk_de_ctrl_vs_treat" in dm.list_modalities(), reply
    _custom(
        dm,
        "import scanpy as sc\n"
        "d = sc.read_h5ad('data/bulk_de_ctrl_vs_treat.h5ad')\n"
        "key = d.uns['de_current_results_key']\n"
        "result = {'n_genes': int(len(d.uns[key]['results_df']))}\n",
        "de_analysis_expert",
    )
    out = _replay(dm, source)
    _ran(out)
    assert out["verdict"] == "MATCH", out
