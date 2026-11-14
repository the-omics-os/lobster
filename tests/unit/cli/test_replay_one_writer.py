"""The notebook's save cell writes through the session's writer (``write_session_h5ad``).

Real sessions: the transcriptomics tools run on a loaded file, the last modality is
saved, the notebook is exported and replayed on the loaded file. The default policy
and tolerance 0 are untouched. Skipped without scanpy and the transcriptomics package
(PYTHONPATH=<repo>/packages/lobster-transcriptomics, plus scanpy).
"""

import ast
import inspect
import json
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from tests.unit.cli import test_replay_commands as _base

globals().update(
    {
        k: v
        for k, v in vars(_base).items()
        if not k.startswith("test_") and not k.startswith("__")
    }
)

pytest.importorskip("scanpy")
tools_module = pytest.importorskip("lobster.agents.transcriptomics.shared_tools")

CHAINS = {
    "A": ("assess_data_quality", "filter_and_normalize"),
    "B1": ("assess_data_quality", "filter_and_normalize", "select_variable_features"),
    "B2": (
        "assess_data_quality",
        "filter_and_normalize",
        "select_variable_features",
        "run_pca",
    ),
    "B4": (
        "assess_data_quality",
        "filter_and_normalize",
        "cluster_cells",
    ),
    "B3": (
        "assess_data_quality",
        "filter_and_normalize",
        "select_variable_features",
        "run_pca",
        "compute_neighbors_and_embed",
    ),
}


def _counts_file(tmp_path):
    import anndata as ad

    rng = np.random.default_rng(3)
    x = rng.poisson(1.2, size=(240, 300)).astype("float32")
    adata = ad.AnnData(x)
    adata.obs_names = [f"c{i}" for i in range(240)]
    adata.var_names = [f"MT-{i}" if i < 5 else f"g{i}" for i in range(300)]
    raw = tmp_path / "raw" / "counts.h5ad"
    raw.parent.mkdir()
    adata.write_h5ad(raw)
    return raw


def _tool_session(tmp_path, monkeypatch, chain):
    from lobster.core.runtime.data_manager import DataManagerV2
    from lobster.services.analysis.clustering_service import ClusteringService
    from lobster.services.quality.preprocessing_service import PreprocessingService
    from lobster.services.quality.quality_service import QualityService

    _clean_env(monkeypatch, tmp_path)
    raw = _counts_file(tmp_path)
    workspace = tmp_path / "workspace"
    manager = DataManagerV2(workspace_path=workspace, auto_scan=False)
    manager.load_modality("pbmc", raw, adapter="transcriptomics_single_cell")
    tools = {
        t.name: t
        for t in tools_module.create_shared_tools(
            manager,
            QualityService(),
            PreprocessingService(),
            ClusteringService(manager),
        )
    }
    name = "pbmc"
    for tool in chain:
        before = set(manager.list_modalities())
        if tool == "cluster_cells":  # the agent's tool body, without the LLM
            clustered, stats, ir = ClusteringService().cluster_and_visualize(
                adata=manager.get_modality(name),
                resolution=1.0,
                n_neighbors=10,
                n_pcs=10,
            )
            out = f"{name}_clustered"
            manager.store_modality(name=out, adata=clustered, parent_name=name)
            manager.log_tool_usage("cluster_cells", {"modality_name": name}, "d", ir=ir)
            name = out
            continue
        args = {"modality_name": name}
        if tool == "filter_and_normalize":
            args["save_result"] = False  # only the last modality is saved
        result = tools[tool].invoke(args)
        assert "rror" not in result[:80], result[:300]
        made = [m for m in manager.list_modalities() if m not in before]
        name = made[-1] if made else name  # a tool may work in place
    manager.save_modality(name, workspace / "data" / f"{name}.h5ad")
    notebook = Path(manager.export_notebook("nb", "S", filter_strategy="replay"))
    return workspace, notebook, raw


@needs_kernel
@pytest.mark.parametrize(
    "row",
    [
        "A",
        "B1",
        "B2",
        "B3",
        "B4",
    ],
)
def test_the_notebook_writes_what_the_session_wrote(tmp_path, monkeypatch, row):
    workspace, notebook, raw = _tool_session(tmp_path, monkeypatch, CHAINS[row])
    result = _verify_real(workspace, notebook, raw)
    out = json.loads(result.stdout)
    assert result.exit_code == 0 and out["verdict"] == "MATCH", (
        out["verdict"],
        [s["diffs"][:5] for s in out["report"]["slots"]],
        out["notes"],
    )


@needs_kernel
def test_w4_a_fallback_write_that_differs_stays_a_mismatch_and_says_so(
    tmp_path, monkeypatch
):
    """A write without the session writer that differs stays a MISMATCH and says so.

    The difference is a number nested in ``uns``: the session writer stores it as
    text, a plain anndata write keeps the number.
    """
    import anndata as ad

    plain_counts = _counts_file

    def counts_with_a_nested_number(path):
        raw = plain_counts(path)
        adata = ad.read_h5ad(raw)
        adata.uns["cfg"] = {"n_top": 3}
        adata.write_h5ad(raw)
        return raw

    monkeypatch.setattr(
        sys.modules[__name__], "_counts_file", counts_with_a_nested_number
    )
    workspace, notebook, raw = _tool_session(tmp_path, monkeypatch, CHAINS["A"])
    site = tmp_path / "kernel_site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(
        "import sys\nsys.modules['lobster.core.backends.h5ad_backend'] = None\n"
    )
    monkeypatch.setenv("PYTHONPATH", f"{site}:{os.environ.get('PYTHONPATH', '')}")
    result = _verify_real(workspace, notebook, raw)
    out = json.loads(result.stdout)
    assert result.exit_code == 1 and out["verdict"] == "MISMATCH", out
    assert any("uns/cfg/n_top" in d for s in out["report"]["slots"] for d in s["diffs"])
    assert "The notebook wrote without the session writer." in out["notes"]


@needs_kernel
def test_the_notebook_keeps_the_category_order_of_ten_or_more_clusters(
    tmp_path, monkeypatch
):
    import anndata as ad

    workspace, notebook, raw = _tool_session(tmp_path, monkeypatch, CHAINS["B4"])
    saved = ad.read_h5ad(next((workspace / "data").glob("*clustered.h5ad")))
    categories = list(saved.obs["leiden"].cat.categories)
    assert (
        len(categories) >= 10
    ), categories  # the order differs only from 10 clusters on
    assert categories == sorted(categories, key=int)  # the session: number order

    result = _verify_real(workspace, notebook, raw)
    out = json.loads(result.stdout)
    replayed = ad.read_h5ad(next(workspace.glob("replays/*/out_*clustered.h5ad")))
    # every cell has the same label: only the order of the categories can differ
    assert saved.obs["leiden"].astype(str).tolist() == (
        replayed.obs["leiden"].astype(str).tolist()
    )
    assert list(replayed.obs["leiden"].cat.categories) == categories
    assert result.exit_code == 0 and out["verdict"] == "MATCH", (
        out["verdict"],
        [s["diffs"][:5] for s in out["report"]["slots"]],
    )
