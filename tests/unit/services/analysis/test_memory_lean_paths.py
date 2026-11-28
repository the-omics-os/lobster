"""Memory-lean paths must reproduce the plain Scanpy/scikit-learn results exactly.

Every comparison here is bit-for-bit: no tolerances, no PCA sign alignment.
"""

from copy import deepcopy

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy import sparse

import lobster.services.analysis.clustering_service as clustering_module
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.analysis.clustering_service import (
    ClusteringService,
    _feature_scratch,
    _pca_for_clustering,
    _scale_for_pca,
)


def _same(a, b, where="value"):
    """Exact equality for the containers AnnData stores."""
    if sparse.issparse(a) or sparse.issparse(b):
        assert sparse.issparse(a) and sparse.issparse(b), where
        assert a.format == b.format and a.shape == b.shape, where
        for part in ("data", "indices", "indptr"):
            x, y = getattr(a, part), getattr(b, part)
            assert x.dtype == y.dtype and x.tobytes() == y.tobytes(), f"{where}.{part}"
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype and a.shape == b.shape, where
        if a.dtype.hasobject:
            assert list(a.ravel()) == list(b.ravel()), where
        else:
            assert a.tobytes() == b.tobytes(), where
    elif isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, check_exact=True, obj=where)
    elif isinstance(a, dict) or hasattr(a, "keys"):
        assert sorted(map(repr, a.keys())) == sorted(map(repr, b.keys())), where
        for key in a.keys():
            _same(a[key], b[key], f"{where}[{key!r}]")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{where}[{i}]")
    else:
        assert a == b or (a != a and b != b), where  # NaN-safe scalars


def _same_anndata(a, b, skip_uns=("lineage",)):
    _same(a.X, b.X, "X")
    _same(a.obs, b.obs, "obs")
    _same(a.var, b.var, "var")
    for attr in ("obsm", "varm", "obsp", "layers"):
        _same(
            {k: v for k, v in getattr(a, attr).items() if k is not None},
            {k: v for k, v in getattr(b, attr).items() if k is not None},
            attr,
        )
    _same(
        {k: v for k, v in a.uns.items() if k not in skip_uns},
        {k: v for k, v in b.uns.items() if k not in skip_uns},
        "uns",
    )
    assert (a.raw is None) == (b.raw is None), "raw"
    if a.raw is not None:
        _same(a.raw.X, b.raw.X, "raw.X")
        _same(a.raw.var, b.raw.var, "raw.var")


def _counts(n_obs=240, n_vars=600, seed=7):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(0.6, size=(n_obs, n_vars)).astype(np.float32)
    for group in range(4):
        rows = slice(group * n_obs // 4, (group + 1) * n_obs // 4)
        cols = slice(group * 40, group * 40 + 40)
        counts[rows, cols] += rng.poisson(6, size=(n_obs // 4, 40))
    adata = ad.AnnData(sparse.csr_matrix(counts))
    adata.obs_names = [f"c{i}" for i in range(n_obs)]
    adata.var_names = [f"g{i}" for i in range(n_vars)]
    return adata


def _normalized(adata):
    adata = adata.copy()
    adata.raw = adata.copy()  # counts, as filter_and_normalize leaves them
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


@pytest.fixture
def plain_paths(monkeypatch):
    def use(enabled):
        monkeypatch.setattr(clustering_module, "MEMORY_LEAN_CLUSTERING", enabled)

    return use


def test_scale_matches_scanpy():
    counts = _counts()
    dense = counts.X.toarray()
    dense[:, 3] = 0  # a zero-variance gene exercises the std == 0 guard
    counts.X = sparse.csr_matrix(dense)
    source = _normalized(counts).X
    mask = np.zeros(source.shape[1], dtype=bool)
    mask[::3] = True  # includes gene 3; more than 512 rows would span blocks
    expected, _ = _feature_scratch(ad.AnnData(source), mask)
    sc.pp.scale(expected, max_value=10)

    actual, positions = _feature_scratch(ad.AnnData(source), mask)
    _scale_for_pca(actual, source, positions, max_value=10)

    _same(actual.X, expected.X, "scaled X")
    _same(actual.var, expected.var, "var")


def test_pca_matches_scanpy():
    source = _normalized(_counts()).X
    mask = np.ones(source.shape[1], dtype=bool)
    expected, positions = _feature_scratch(ad.AnnData(source), mask)
    _scale_for_pca(expected, source, positions)
    actual = expected.copy()
    sc.tl.pca(expected, svd_solver="arpack", n_comps=20)

    _pca_for_clustering(actual, n_comps=20, svd_solver="arpack")

    _same(actual.obsm["X_pca"], expected.obsm["X_pca"], "X_pca")
    _same(actual.varm["PCs"], expected.varm["PCs"], "PCs")
    _same(actual.uns["pca"], expected.uns["pca"], "uns.pca")


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="deviance"),
        pytest.param({"feature_selection_method": "hvg"}, id="hvg"),
        pytest.param(
            {"demo_mode": True, "n_features": 1100, "subsample_size": 200},
            id="demo-named-genes",
        ),
    ],
)
def test_clustering_matches_plain_scanpy_calls(plain_paths, kwargs):
    n_vars = 1500 if kwargs.get("demo_mode") else 600
    adata = _normalized(_counts(n_vars=n_vars))
    reference = adata.copy()

    plain_paths(False)
    expected, expected_stats, _ = ClusteringService().cluster_and_visualize(
        adata.copy(), resolution=0.5, **kwargs
    )
    plain_paths(True)
    actual, actual_stats, _ = ClusteringService().cluster_and_visualize(
        adata, resolution=0.5, **kwargs
    )

    _same_anndata(actual, expected)
    _same_anndata(adata, reference)  # the input is untouched
    assert actual_stats["n_clusters"] == expected_stats["n_clusters"]


def _store(dm, name, adata, parent=None):
    dm.store_modality(name=name, adata=adata, parent_name=parent)
    return dm.get_modality(name)


@pytest.fixture
def dm(tmp_path, monkeypatch):
    monkeypatch.setenv("LOBSTER_MODALITY_MEMORY_BUDGET_GB", "0")
    monkeypatch.delenv("LOBSTER_SHARE_MODALITY_BUFFERS", raising=False)
    return DataManagerV2(workspace_path=tmp_path, enable_provenance=False)


def test_registry_shares_identical_buffers(dm):
    parent = _store(dm, "pbmc", _counts(n_obs=1200, n_vars=2000))
    child = parent.copy()
    child.obs["qc"] = 1.0
    data_before = parent.X.data.copy()
    single = dm._resident_bytes()

    child = _store(dm, "pbmc_qc", child, parent="pbmc")

    for part in ("data", "indices", "indptr"):
        if getattr(parent.X, part).nbytes >= 1 << 20:
            assert getattr(child.X, part) is getattr(parent.X, part)
            assert not getattr(parent.X, part).flags.writeable
    assert dm._resident_bytes() < 2 * single
    with pytest.raises(ValueError):
        child.X.data *= 2  # a write through either object would change both
    np.testing.assert_array_equal(parent.X.data, data_before)


def test_registry_keeps_distinct_or_replaced_buffers_apart(dm):
    first = _store(dm, "a", _counts(n_obs=1200, n_vars=2000, seed=1))
    other = _store(dm, "b", _counts(n_obs=1200, n_vars=2000, seed=2))
    assert other.X.data is not first.X.data
    assert first.X.data.flags.writeable and other.X.data.flags.writeable

    # Re-storing under the same name never shares with the entry it replaces.
    replacement = first.copy()
    replacement = _store(dm, "a", replacement)
    assert replacement.X.data is not first.X.data
    assert replacement.X.data.flags.writeable


def test_registry_sharing_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("LOBSTER_MODALITY_MEMORY_BUDGET_GB", "0")
    monkeypatch.setenv("LOBSTER_SHARE_MODALITY_BUFFERS", "0")
    manager = DataManagerV2(workspace_path=tmp_path, enable_provenance=False)
    parent = _store(manager, "pbmc", _counts(n_obs=1200, n_vars=2000))
    child = _store(manager, "pbmc_qc", parent.copy(), parent="pbmc")
    assert child.X.data is not parent.X.data
    assert child.X.data.flags.writeable


def test_marker_x_is_shared_only_when_frozen(dm):
    from lobster.services.analysis.enhanced_singlecell_service import (
        EnhancedSingleCellService,
    )

    adata = _normalized(_counts(n_obs=1200, n_vars=2000))
    adata.raw = adata.copy()
    adata.obs["leiden"] = pd.Categorical(np.repeat(list("0123"), 300))

    loose, _, _ = EnhancedSingleCellService().find_marker_genes(adata, groupby="leiden")
    assert loose.X is not adata.X and loose.X.data is not adata.X.data

    stored = _store(dm, "pbmc_clustered", adata)
    stored = _store(
        dm, "pbmc_clustered_copy", deepcopy(stored), parent="pbmc_clustered"
    )
    frozen, _, _ = EnhancedSingleCellService().find_marker_genes(
        stored, groupby="leiden"
    )
    assert frozen.X is stored.X
    # .raw.X is wrapped around the stored buffers, not copied.
    assert np.shares_memory(frozen.raw.X.data, stored.raw.X.data)
    _same_anndata(frozen, loose, skip_uns=("lineage",))


def test_clustering_shares_its_input_structure_and_raw_is_frozen_once_stored(dm):
    # Large enough that X.data passes the 1 MiB sharing threshold.
    adata = _normalized(_counts(n_obs=1200, n_vars=1000))
    data_before = adata.X.data.copy()
    clustered, _, _ = ClusteringService().cluster_and_visualize(adata, resolution=0.5)

    # Normalization rewrote values in a private copy; the structure is shared
    # (SciPy may rebind it as a view of the same buffer).
    assert np.shares_memory(clustered.X.indices, adata.X.indices)
    assert np.shares_memory(clustered.X.indptr, adata.X.indptr)
    assert not adata.X.indices.flags.writeable
    assert not np.shares_memory(clustered.X.data, adata.X.data)
    np.testing.assert_array_equal(adata.X.data, data_before)

    assert clustered.raw.X is clustered.X  # one matrix, no stored zeros
    clustered = _store(dm, "pbmc_clustered", clustered)
    for part in ("data", "indices", "indptr"):
        assert not getattr(clustered.X, part).flags.writeable
    with pytest.raises(ValueError):
        clustered.raw.X.data[0] = 0  # .X and .raw.X must not diverge silently


def test_noop_eliminate_zeros_guard_is_scoped():
    from lobster.core.runtime.buffer_sharing import skip_noop_eliminate_zeros

    rng = np.random.default_rng(3)
    frozen = sparse.csr_matrix(rng.random((50, 40)) * (rng.random((50, 40)) > 0.5))
    for part in ("data", "indices", "indptr"):
        getattr(frozen, part).flags.writeable = False
    with_zero = frozen.copy()
    with_zero.data[0] = 0
    for part in ("data", "indices", "indptr"):
        getattr(with_zero, part).flags.writeable = False
    other = sparse.csr_matrix(np.eye(5))
    other.data[0] = 0

    with pytest.raises(ValueError):
        frozen.eliminate_zeros()  # SciPy rewrites buffers even when nothing changes
    with skip_noop_eliminate_zeros(frozen, with_zero):
        frozen.eliminate_zeros()  # nothing to remove: skipped
        with pytest.raises(ValueError):
            with_zero.eliminate_zeros()  # a real change still fails loudly
        other.eliminate_zeros()  # unrelated matrices run SciPy as usual
    assert other.nnz == 4
    assert "eliminate_zeros" not in sparse.csr_matrix.__dict__  # restored
    with pytest.raises(ValueError):
        frozen.eliminate_zeros()
