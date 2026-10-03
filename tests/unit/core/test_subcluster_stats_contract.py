"""Regression tests ensuring sub-clustering stats satisfy the response formatter contract.

The service and response formatter must agree on the keys they exchange. These tests
assert that the service provides the fields consumed by the formatter, preventing a
successful computation from failing during response construction.
"""

import numpy as np
import pytest
import scanpy as sc
import scipy.sparse as sp

from lobster.agents.transcriptomics.transcriptomics_expert import (
    SUBCLUSTER_STATS_KEYS,
    format_subcluster_response,
)
from lobster.services.analysis.clustering_service import ClusteringService

# The keys the agent's formatter reads from the service's stats dict.
# Kept here deliberately: this list is the contract, and the test below is what
# fails if either side drifts again.
FORMATTER_REQUIRED_KEYS = {
    "n_cells_subclustered",
    "parent_clusters",
    "resolutions_tested",
    "subclustering_results",
    "primary_subcluster_key",
    "execution_time_seconds",
    "cluster_sizes",
}


@pytest.fixture(scope="module")
def clustered_adata():
    rng = np.random.default_rng(0)
    n_obs, n_vars = 400, 200
    X = rng.negative_binomial(20, 0.3, size=(n_obs, n_vars)).astype(np.float32)
    adata = sc.AnnData(X=sp.csr_matrix(X))
    adata.var_names = [f"gene{i}" for i in range(n_vars)]
    adata.obs_names = [f"cell{i}" for i in range(n_obs)]
    adata.obs["leiden"] = [str(i % 4) for i in range(n_obs)]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=20)
    return adata


@pytest.mark.parametrize(
    "kwargs,expect_multi",
    [({"resolution": 0.5}, False), ({"resolutions": [0.25, 0.5]}, True)],
)
def test_service_stats_satisfy_the_formatter_contract(
    clustered_adata, kwargs, expect_multi
):
    """The guard against recurrence: the service must supply what the agent reads."""
    _, stats, _ = ClusteringService().subcluster_cells(
        clustered_adata.copy(),
        cluster_key="leiden",
        clusters_to_refine=["0", "1"],
        **kwargs,
    )

    missing = FORMATTER_REQUIRED_KEYS - set(stats)
    assert not missing, f"service stats missing formatter keys: {sorted(missing)}"

    assert (len(stats["resolutions_tested"]) > 1) is expect_multi


@pytest.mark.parametrize("kwargs", [{"resolution": 0.5}, {"resolutions": [0.25, 0.5]}])
def test_stats_are_internally_consistent(clustered_adata, kwargs):
    """Every derivation the formatter performs must actually resolve."""
    _, stats, _ = ClusteringService().subcluster_cells(
        clustered_adata.copy(),
        cluster_key="leiden",
        clusters_to_refine=["0", "1"],
        **kwargs,
    )

    primary = stats["primary_subcluster_key"]
    # The primary column must have an entry in cluster_sizes...
    assert primary in stats["cluster_sizes"], (
        f"primary column '{primary}' absent from cluster_sizes "
        f"{sorted(stats['cluster_sizes'])}"
    )
    sizes = stats["cluster_sizes"][primary]
    assert sizes, "primary column has no sub-cluster sizes"
    assert all(isinstance(v, int) for v in sizes.values())

    # ...and a per-resolution record carrying its sub-cluster count.
    key_names = {
        res_data["key_name"]: res_data["n_total_subclusters"]
        for res_data in stats["subclustering_results"].values()
    }
    assert primary in key_names
    assert key_names[primary] > 0
    assert len(stats["subclustering_results"]) == len(stats["resolutions_tested"])
    assert isinstance(stats["execution_time_seconds"], (int, float))


@pytest.mark.parametrize(
    "kwargs,expect_multi",
    [({"resolution": 0.5}, False), ({"resolutions": [0.25, 0.5]}, True)],
)
def test_real_formatter_renders_both_paths_without_keyerror(
    clustered_adata, kwargs, expect_multi
):
    """The actual formatter, on real service output, on both paths.

    Pre-fix this raised KeyError on 'primary_column' (single) and
    'multi_resolution_summary' (multi).
    """
    _, stats, _ = ClusteringService().subcluster_cells(
        clustered_adata.copy(),
        cluster_key="leiden",
        clusters_to_refine=["0", "1"],
        **kwargs,
    )

    response = format_subcluster_response(
        new_name="pbmc_subclustered",
        cluster_key="leiden",
        resolution=0.5,
        stats=stats,
    )

    assert response.startswith("Sub-clustering complete!")
    assert "pbmc_subclustered" in response
    assert "Execution time:" in response
    # The real sub-cluster sizes must appear, not an empty section.
    primary = stats["primary_subcluster_key"]
    first_id = next(iter(stats["cluster_sizes"][primary]))
    assert str(first_id) in response
    assert "cells" in response

    if expect_multi:
        assert "Tested 2 resolutions" in response
        assert "(primary)" in response
        assert "Interpretation" in response
    else:
        assert "Next steps" in response
        assert "sub-clusters at resolution" in response


def test_formatter_declared_contract_matches_the_test_list():
    """SUBCLUSTER_STATS_KEYS is the documented contract; keep it honest."""
    assert set(SUBCLUSTER_STATS_KEYS) == FORMATTER_REQUIRED_KEYS


def test_formatter_degrades_rather_than_raising_on_empty_stats():
    """Defensive: a partial stats dict must not resurrect the KeyError."""
    response = format_subcluster_response(
        new_name="m_sub", cluster_key="leiden", resolution=0.5, stats={}
    )
    assert response.startswith("Sub-clustering complete!")
    assert "m_sub" in response
