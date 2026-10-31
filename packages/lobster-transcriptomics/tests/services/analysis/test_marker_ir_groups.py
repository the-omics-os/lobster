"""The exported marker cell compares the same groups as the session.

Found in a manual CLI test: ``find_marker_genes(groups=[...])`` passed
``groups`` to scanpy, but the exported cell did not, so it tested every group
(in default order). With three or more groups the results differ.
"""

import anndata
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from lobster.services.analysis.enhanced_singlecell_service import (
    EnhancedSingleCellService,
)


def _adata():
    rng = np.random.default_rng(0)
    counts = rng.poisson(2.0, size=(90, 30)).astype(np.float32)
    counts[:30, :5] += 6  # group "a" markers
    adata = anndata.AnnData(counts)
    adata.var_names = [f"g{i}" for i in range(30)]
    adata.obs["grp"] = pd.Categorical(np.repeat(["a", "b", "c"], 30))
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata


@pytest.mark.parametrize("groups", [None, ["c", "a"]])
def test_exported_marker_cell_matches_the_session(groups, tmp_path, monkeypatch):
    adata = _adata()
    result, _, ir = EnhancedSingleCellService().find_marker_genes(
        adata.copy(), groupby="grp", groups=groups, n_genes=10
    )
    # Execute the actual complete exported notebook, including its deduplicated
    # imports/helper cells. A bare code block omits the S6 marker-name repair.
    from types import SimpleNamespace

    import nbformat

    from lobster.core.notebooks.exporter import NotebookExporter
    from lobster.core.provenance.provenance import ProvenanceTracker

    adata.write_h5ad(tmp_path / "dataset.h5ad")
    tracker = ProvenanceTracker()
    tracker.create_activity(activity_type="find_marker_genes", agent="test", ir=ir)
    context = SimpleNamespace(
        workspace_path=tmp_path, workspace_dir=tmp_path, modalities={"fixture": adata}
    )
    path = NotebookExporter(tracker, context).export("marker_replay")
    notebook = nbformat.read(path, as_version=4)
    import_cells = [
        c.source
        for c in notebook.cells
        if c.cell_type == "code" and c.source.startswith("# Required imports")
    ]
    assert len(import_cells) == 1
    assert (
        "from lobster.core.utils.h5ad_utils import repair_marker_names"
        in import_cells[0]
    )
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            exec(
                compile(cell.source, f"marker_notebook_cell_{index}", "exec"), namespace
            )  # nosec B102
    replay = namespace["adata"]
    assert replay.shape == adata.shape

    session = result.uns["rank_genes_groups"]
    exported = replay.uns["rank_genes_groups"]
    assert session["names"].dtype.names == exported["names"].dtype.names
    for group in session["names"].dtype.names:
        assert list(session["names"][group]) == list(exported["names"][group])
        assert np.allclose(session["scores"][group], exported["scores"][group])
