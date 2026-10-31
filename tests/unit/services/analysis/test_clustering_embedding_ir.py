"""Clustering on an existing embedding exports the steps it actually ran.

Found in a manual CLI test: with ``use_rep="X_pca"`` the service clusters on
the session's 10-component PCA, but the exported IR re-ran normalization,
feature selection and a 30-component PCA, so the notebook computed something
else.
"""

import anndata
import numpy as np
import pytest
import scanpy as sc

from lobster.services.analysis.clustering_service import (
    ClusteringError,
    ClusteringService,
)

pytest.importorskip("leidenalg")


def _adata():
    rng = np.random.default_rng(0)
    adata = anndata.AnnData(rng.poisson(1.0, size=(120, 60)).astype(np.float32))
    adata.var_names = [f"g{i}" for i in range(60)]
    adata.obs_names = [f"c{i}" for i in range(120)]
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=10)
    return adata


def test_existing_embedding_is_exported_and_replays_the_same_clusters():
    adata = _adata()
    service = ClusteringService()
    result, _, ir = service.cluster_and_visualize(
        adata.copy(), resolution=0.5, use_rep="X_pca"
    )
    code = ir.render()

    assert "sc.pp.pca" not in code
    assert "normalize_total" not in code
    replay = adata.copy()
    exec(
        compile(code, "cluster_cell", "exec"), {"adata": replay, "sc": sc}
    )  # nosec B102

    assert replay.obsm["X_pca"].shape == (120, 10)
    assert result.obsm["X_pca"].shape == (120, 10)
    assert list(replay.obs["leiden"]) == list(result.obs["leiden"])


def test_without_existing_embedding_the_full_pipeline_is_exported():
    adata = _adata()
    del adata.obsm["X_pca"]
    service = ClusteringService()
    result, _, ir = service.cluster_and_visualize(
        adata, resolution=0.5, use_rep=None, demo_mode=True
    )
    assert "sc.pp.pca" in ir.render() or "sc.tl.pca" in ir.render()
    assert result.shape == adata.shape
    assert {"leiden", "leiden_res0_5"} <= set(result.obs)
    assert "X_umap" in result.obsm
    assert "X_pca" not in adata.obsm


def test_explicit_missing_embedding_refuses_without_mutation():
    adata = _adata()
    del adata.obsm["X_pca"]
    original_x = adata.X.copy()
    original_obs = adata.obs.copy(deep=True)
    original_var = adata.var.copy(deep=True)
    original_uns = dict(adata.uns)
    original_varm = {key: value.copy() for key, value in adata.varm.items()}
    original_raw = adata.raw
    with pytest.raises(
        ClusteringError,
        match="Representation 'X_pca' not found; prepare and name it explicitly",
    ):
        ClusteringService().cluster_and_visualize(
            adata, resolution=0.5, use_rep="X_pca", demo_mode=True
        )
    np.testing.assert_array_equal(adata.X, original_x)
    assert adata.obs.equals(original_obs)
    assert adata.var.equals(original_var)
    assert set(adata.uns) == set(original_uns)
    assert all(adata.uns[key] is value for key, value in original_uns.items())
    assert set(adata.varm) == set(original_varm)
    for key, value in original_varm.items():
        np.testing.assert_array_equal(adata.varm[key], value)
    assert not adata.obsm and not adata.obsp and not adata.layers
    assert adata.raw is original_raw
