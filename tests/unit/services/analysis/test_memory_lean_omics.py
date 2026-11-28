"""Bulk, proteomics and metabolomics services reuse scratch buffers.

Each reference below is the pre-change computation written out with fresh
copies; results must match it bit for bit and the input must stay untouched.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def _bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape
    assert a.tobytes() == b.tobytes()


def _snapshot(adata):
    x = adata.X
    if sparse.issparse(x):
        return [x.data.tobytes(), x.indices.tobytes(), x.indptr.tobytes()]
    return [np.asarray(x).tobytes()]


def _samples(n, rng):
    return pd.DataFrame(
        {"group": pd.Categorical(np.where(np.arange(n) % 2 == 0, "A", "B"))},
        index=[f"s{i}" for i in range(n)],
    )


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.float32])
def test_bulk_sample_qc_matches_reference(dtype):
    from lobster.services.analysis.bulk_preprocessing_service import (
        BulkPreprocessingService,
    )

    rng = np.random.default_rng(1)
    adata = ad.AnnData(rng.poisson(6, (30, 800)).astype(dtype), obs=_samples(30, rng))
    before = _snapshot(adata)

    # The service's PCA uses random_state=None (randomized solver here), so
    # both runs draw from the same seeded global state.
    np.random.seed(0)
    result, stats, _ = BulkPreprocessingService().assess_sample_quality(adata)

    X = np.asarray(adata.X, dtype=np.float64)
    library_sizes = X.sum(axis=1, keepdims=True)
    library_sizes = np.where(library_sizes == 0, 1, library_sizes)
    log_cpm = np.log1p(X / library_sizes * 1e6)
    np.random.seed(0)
    coords = PCA(n_components=min(30, 800, 10)).fit_transform(log_cpm)
    _bits(result.obsm["X_pca_samples"], coords)
    _bits(result.uns["sample_correlation_matrix"], np.corrcoef(log_cpm))
    assert _snapshot(adata) == before
    assert result.X is not adata.X


@pytest.mark.parametrize("layout", ["csr-int", "dense-float"])
def test_bulk_cpm_matches_reference(layout):
    from lobster.services.analysis.bulk_preprocessing_service import (
        BulkPreprocessingService,
    )

    rng = np.random.default_rng(2)
    counts = rng.poisson(5, (25, 900))
    x = (
        sparse.csr_matrix(counts.astype(np.int64))
        if layout == "csr-int"
        else counts.astype(float)
    )
    adata = ad.AnnData(x, obs=_samples(25, rng))
    before = _snapshot(adata)

    result, stats, _ = BulkPreprocessingService().normalize_counts(adata, method="cpm")

    dense = counts.astype(np.int64) if layout == "csr-int" else counts.astype(float)
    _bits(result.layers["counts"], dense)
    X = dense.astype(np.float64)
    library_sizes = X.sum(axis=1, keepdims=True)
    library_sizes = np.where(library_sizes == 0, 1, library_sizes)
    _bits(result.X, X / library_sizes * 1e6)
    assert not np.shares_memory(result.X, result.layers["counts"])
    assert _snapshot(adata) == before


@pytest.mark.parametrize("missing", [True, False])
def test_proteomics_clustering_matches_reference(missing):
    proteomics = pytest.importorskip(
        "lobster.services.analysis.proteomics_analysis_service"
    )
    rng = np.random.default_rng(3)
    x = rng.normal(size=(60, 300))
    if missing:
        x[::7, ::11] = np.nan
    adata = ad.AnnData(x, obs=_samples(60, rng))
    before = _snapshot(adata)

    np.random.seed(0)
    result, _, _ = proteomics.ProteomicsAnalysisService().perform_clustering_analysis(
        adata
    )

    scaled = StandardScaler().fit_transform(
        SimpleImputer(strategy="mean").fit_transform(x.copy())
    )
    n_comp = result.obsm["X_pca_clustering"].shape[1]
    np.random.seed(0)
    _bits(
        result.obsm["X_pca_clustering"], PCA(n_components=n_comp).fit_transform(scaled)
    )
    assert _snapshot(adata) == before
    assert result.X is not adata.X


@pytest.mark.parametrize("layout", ["dense", "csr-float32"])
def test_metabolomics_pca_matches_reference(layout):
    metabolomics = pytest.importorskip(
        "lobster.services.analysis.metabolomics_analysis_service"
    )
    rng = np.random.default_rng(4)
    x = rng.random((50, 400)) * 100
    if layout == "csr-float32":
        x[x < 40] = 0
        x = sparse.csr_matrix(x.astype(np.float32))
    adata = ad.AnnData(x, obs=_samples(50, rng))
    before = _snapshot(adata)

    np.random.seed(0)
    result, stats, _ = metabolomics.MetabolomicsAnalysisService().run_pca(adata)

    X = (
        x.toarray().astype(np.float64)
        if sparse.issparse(x)
        else np.array(x, dtype=np.float64)
    )
    scaled = StandardScaler().fit_transform(X)
    np.random.seed(0)
    _bits(
        result.obsm["X_pca"],
        PCA(n_components=stats["n_components"]).fit_transform(scaled),
    )
    assert _snapshot(adata) == before
