"""A session that selected highly variable genes replays to a match, also when some
gene had no expression left after the cell filters."""

import json
from pathlib import Path

import numpy as np
import pytest

from tests.unit.cli.test_replay_commands import (
    _clean_env,
    _save,
    _verify_real,
    needs_kernel,
)

tools_module = pytest.importorskip("lobster.agents.transcriptomics.shared_tools")


def _counts(with_dead_genes: bool):
    """300 cells x 400 genes; the cells that fail the mito filter hold the only counts
    of the genes that end up with none (they pass the minimum-cells filter first)."""
    import anndata as ad

    rng = np.random.default_rng(5)
    X = rng.poisson(1.5, size=(300, 400)).astype("float32")
    names = [f"MT-{i}" if i < 5 else f"g{i}" for i in range(400)]
    if with_dead_genes:
        X[:, 380:383] = 0
        X[:3, 380:383] = 5  # exactly the three cells below, no others
        X[:3, :5] = 400  # those cells are mostly mitochondrial: filtered out
    else:
        X[:, 380:383] += 1  # every gene keeps counts
    adata = ad.AnnData(X)
    adata.obs_names = [f"c{i}" for i in range(300)]
    adata.var_names = names
    return adata


def _session(tmp_path, monkeypatch, with_dead_genes):
    from lobster.core.runtime.data_manager import DataManagerV2
    from lobster.services.analysis.clustering_service import ClusteringService
    from lobster.services.quality.preprocessing_service import PreprocessingService
    from lobster.services.quality.quality_service import QualityService

    _clean_env(monkeypatch, tmp_path)
    workspace = tmp_path / "workspace"
    source = tmp_path / "counts.h5ad"
    _save(_counts(with_dead_genes), source)
    manager = DataManagerV2(workspace_path=workspace, auto_scan=False)
    manager.load_modality("raw", source, adapter="transcriptomics_single_cell")
    tools = {
        t.name: t
        for t in tools_module.create_shared_tools(
            manager,
            QualityService(),
            PreprocessingService(),
            ClusteringService(manager),
        )
    }
    name = "raw"
    steps = (
        ("filter_and_normalize", {"save_result": False}),
        ("select_variable_features", {"method": "hvg", "n_top_genes": 100}),
        ("run_pca", {"n_comps": 10}),
        ("compute_neighbors_and_embed", {"n_neighbors": 10, "n_pcs": 10}),
    )
    sizes = []
    for tool, args in steps:
        before = set(manager.list_modalities())
        result = tools[tool].invoke({"modality_name": name, **args})
        assert "rror" not in result[:80], result[:300]
        made = [m for m in manager.list_modalities() if m not in before]
        name = made[-1] if made else name
        sizes.append(manager.get_modality(name).n_vars)
    manager.save_modality(name, workspace / "data" / f"{name}.h5ad")
    notebook = Path(manager.export_notebook("nb", "S", filter_strategy="replay"))
    return workspace, notebook, source, sizes


@needs_kernel
@pytest.mark.parametrize("dead", [True, False])
def test_a_session_that_selected_variable_genes_replays_to_a_match(
    tmp_path, monkeypatch, dead
):
    workspace, notebook, source, sizes = _session(tmp_path, monkeypatch, dead)
    if dead:  # the selection step dropped genes the cell filters had emptied
        assert sizes[1] < sizes[0], sizes
    else:
        assert sizes[1] == sizes[0], sizes
    result = _verify_real(workspace, notebook, source)
    out = json.loads(result.stdout)
    assert out["verdict"] == "MATCH", (out["notes"], out["report"].get("refusal"))
    assert result.exit_code == 0
