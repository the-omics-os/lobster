"""Regression tests ensuring marker ranking receives log-normalized values.

``find_marker_genes`` passed whatever matrix it was given straight to
``sc.tl.rank_genes_groups``.  Scanpy only warns ("It seems you use
rank_genes_groups on the raw count data"), and that warning went to the child
log where nothing acted on it.

For ``method="wilcoxon"`` (the default), log1p preserves ranks, but depth
normalization can change them on depth-confounded data. The tests therefore
require both normalization and log transformation, not log1p alone.
"""

import numpy as np
import pytest
import scanpy as sc
import scipy.sparse as sp

from lobster.services.analysis.enhanced_singlecell_service import (
    EnhancedSingleCellService,
    _looks_like_raw_counts,
)

N_OBS = 300
N_VARS = 200
N_GROUPS = 3


def _counts_adata(seed: int = 7, depth_span: float = 3.0) -> "sc.AnnData":
    """Raw integer counts with per-cell depth variation and real group signal."""
    rng = np.random.default_rng(seed)
    base_rate = rng.gamma(2.0, 2.0, size=N_VARS)
    X = np.zeros((N_OBS, N_VARS))
    labels = []
    for i in range(N_OBS):
        g = i % N_GROUPS
        labels.append(str(g))
        rate = base_rate.copy()
        rate[g * 20 : (g + 1) * 20] *= 4.0
        depth = 1.0 + (depth_span - 1.0) * rng.random()
        X[i] = rng.poisson(rate * depth)
    adata = sc.AnnData(X=sp.csr_matrix(X.astype(np.float32)))
    adata.var_names = [f"gene{i}" for i in range(N_VARS)]
    adata.obs_names = [f"cell{i}" for i in range(N_OBS)]
    adata.obs["leiden"] = labels
    return adata


def _top_names(adata, k=20):
    names = adata.uns["rank_genes_groups"]["names"]
    return {g: list(names[g][:k]) for g in names.dtype.names}


# --- the raw-count detector ------------------------------------------------


def test_detector_flags_integer_counts():
    adata = _counts_adata()
    assert _looks_like_raw_counts(adata.X) is True


def test_detector_does_not_flag_log_normalized_data():
    adata = _counts_adata()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    assert _looks_like_raw_counts(adata.X) is False


def test_detector_does_not_flag_small_integer_matrix():
    """A log-scale matrix that happens to be integral must not be flagged.

    Guards against normalizing already-transformed data, which would be a new
    bug in the opposite direction.
    """
    small = sp.csr_matrix(np.full((10, 10), 3.0, dtype=np.float32))
    assert _looks_like_raw_counts(small) is False


def test_detector_ignores_negative_matrices():
    """Scaled (z-scored) data has negatives and is never raw counts."""
    scaled = np.linspace(-80, 80, 400).reshape(20, 20).astype(np.float32)
    assert _looks_like_raw_counts(scaled) is False


def test_detector_is_safe_on_none_and_empty():
    assert _looks_like_raw_counts(None) is False
    assert _looks_like_raw_counts(sp.csr_matrix((5, 5), dtype=np.float32)) is False


# --- the fix itself -------------------------------------------------------


def test_find_marker_genes_normalizes_raw_counts_before_ranking():
    """The ranking must match a normalize+log reference, not the raw one."""
    adata = _counts_adata()
    adata.raw = adata.copy()

    result, stats, ir = EnhancedSingleCellService().find_marker_genes(
        adata.copy(), groupby="leiden", n_genes=20
    )
    got = _top_names(result)

    # Reference A: what the service should now be doing.
    ref_fixed = adata.copy()
    sc.pp.normalize_total(ref_fixed, target_sum=1e4)
    sc.pp.log1p(ref_fixed)
    sc.tl.rank_genes_groups(
        ref_fixed, "leiden", method="wilcoxon", n_genes=20, use_raw=False
    )
    expected = _top_names(ref_fixed)

    # Reference B: the pre-fix behaviour, ranking raw counts directly.
    ref_raw = adata.copy()
    sc.tl.rank_genes_groups(
        ref_raw, "leiden", method="wilcoxon", n_genes=20, use_raw=False
    )
    pre_fix = _top_names(ref_raw)

    for group in expected:
        assert (
            got[group] == expected[group]
        ), f"group {group}: ranking does not match the normalize+log reference"

    # And the fix must actually have changed something, or it proves nothing.
    assert (
        got != pre_fix
    ), "post-fix ranking is identical to ranking raw counts — the fix is inert"


def test_log1p_alone_would_not_change_wilcoxon_ranking():
    """Pins the reason the fix normalizes rather than only logarithmizing.

    log1p is monotonic, and wilcoxon ranks per gene across cells, so log1p alone
    leaves both names and scores bit-identical. A 'fix' that only called log1p
    would silently do nothing for the default method.
    """
    adata = _counts_adata()

    raw = adata.copy()
    sc.tl.rank_genes_groups(raw, "leiden", method="wilcoxon", n_genes=20, use_raw=False)

    logged = adata.copy()
    sc.pp.log1p(logged)
    sc.tl.rank_genes_groups(
        logged, "leiden", method="wilcoxon", n_genes=20, use_raw=False
    )

    assert _top_names(raw) == _top_names(logged)
    raw_scores = raw.uns["rank_genes_groups"]["scores"]
    log_scores = logged.uns["rank_genes_groups"]["scores"]
    for group in raw_scores.dtype.names:
        np.testing.assert_allclose(raw_scores[group], log_scores[group])


def test_find_marker_genes_leaves_already_normalized_data_alone():
    """Must not normalize twice — that would corrupt correct input."""
    adata = _counts_adata()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata.copy()

    result, _, _ = EnhancedSingleCellService().find_marker_genes(
        adata.copy(), groupby="leiden", n_genes=20
    )

    ref = adata.copy()
    sc.tl.rank_genes_groups(ref, "leiden", method="wilcoxon", n_genes=20, use_raw=True)
    assert _top_names(result) == _top_names(ref)


def test_find_marker_genes_works_without_raw(caplog):
    """use_raw was hardcoded True, which breaks when .raw is unset."""
    adata = _counts_adata()
    assert adata.raw is None

    result, stats, ir = EnhancedSingleCellService().find_marker_genes(
        adata.copy(), groupby="leiden", n_genes=20
    )
    assert "rank_genes_groups" in result.uns
    names = result.uns["rank_genes_groups"]["names"]
    assert len(names.dtype.names) == N_GROUPS


def test_find_marker_genes_normalizes_raw_when_only_X_is_counts():
    """The .X path (no .raw) must be normalized too."""
    adata = _counts_adata()
    result, _, _ = EnhancedSingleCellService().find_marker_genes(
        adata.copy(), groupby="leiden", n_genes=20
    )

    ref = adata.copy()
    sc.pp.normalize_total(ref, target_sum=1e4)
    sc.pp.log1p(ref)
    sc.tl.rank_genes_groups(ref, "leiden", method="wilcoxon", n_genes=20, use_raw=False)
    assert _top_names(result) == _top_names(ref)


def test_clustering_service_marker_step_is_not_affected():
    """Scope check: clustering_service's own rank_genes_groups was already clean.

    Verified on this branch: cluster_and_visualize() log-normalizes before its
    marker step, .X and .raw.X are both transformed, and scanpy emits no
    raw-count warning. The normalization check belongs in find_marker_genes. Without this test
    a later reader may 'fix' the clustering path too and normalize twice.
    """
    from lobster.services.analysis.clustering_service import ClusteringService

    rng = np.random.default_rng(0)
    X = rng.negative_binomial(20, 0.3, size=(200, 150)).astype(np.float32)
    adata = sc.AnnData(X=sp.csr_matrix(X))
    adata.var_names = [f"gene{i}" for i in range(150)]
    adata.obs_names = [f"cell{i}" for i in range(200)]

    out = ClusteringService().cluster_and_visualize(adata, resolution=0.5)
    result = out[0] if isinstance(out, tuple) else out

    assert "log1p" in result.uns, "clustering should have log-normalized"
    assert not _looks_like_raw_counts(result.X)
