"""Regression tests for cluster-key mismatches between clustering and downstream readers.

The clustering service names its columns by resolution (``leiden_res0_5``,
``louvain_res1_0``, ``leiden_subcluster``) while downstream readers may receive a
literal ``"leiden"`` default. Where that column is absent, annotation can fail
immediately after clustering succeeds.

Resolution order under test, in priority order:

1. explicit ``cluster_key``
2. exact ``leiden`` / ``louvain``
3. the *unique* ``leiden_res*`` column
4. raise ONLY when genuinely ambiguous

The implementation accounts for paths that write an unsuffixed alias and paths
that do not. Readers must resolve a real column rather than assume one exists.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.core.utils.cluster_keys import (
    ClusterKeyError,
    list_cluster_key_candidates,
    resolve_cluster_key,
)


def _adata_with_cluster_columns(*names: str, n_obs: int = 30) -> "ad.AnnData":
    rng = np.random.default_rng(0)
    adata = ad.AnnData(X=rng.random((n_obs, 5)))
    for name in names:
        adata.obs[name] = pd.Categorical([str(i % 3) for i in range(n_obs)])
    return adata


# --- 1. explicit cluster_key wins -----------------------------------------


def test_explicit_key_wins_even_when_others_exist():
    adata = _adata_with_cluster_columns("leiden", "leiden_res0_5", "leiden_res1_0")
    assert resolve_cluster_key(adata, "leiden_res1_0") == "leiden_res1_0"


def test_explicit_non_cluster_column_is_accepted():
    """Seurat/other conventions must work: the caller knows best."""
    adata = _adata_with_cluster_columns("leiden")
    adata.obs["RNA_snn_res.1"] = pd.Categorical(["a"] * adata.n_obs)
    assert resolve_cluster_key(adata, "RNA_snn_res.1") == "RNA_snn_res.1"


def test_explicit_missing_key_raises_and_lists_candidates():
    """A wrong explicit key must fail loudly, never be silently replaced."""
    adata = _adata_with_cluster_columns("leiden_res0_5")
    with pytest.raises(ClusterKeyError) as excinfo:
        resolve_cluster_key(adata, "leiden")
    message = str(excinfo.value)
    assert "leiden" in message
    assert "leiden_res0_5" in message, "the error must name the available column"


# --- 2. exact leiden/louvain ----------------------------------------------


def test_exact_leiden_preferred_over_suffixed():
    adata = _adata_with_cluster_columns("leiden", "leiden_res0_5", "leiden_res1_0")
    assert resolve_cluster_key(adata) == "leiden"


def test_exact_louvain_resolves_without_leiden_present():
    """The louvain path must resolve an actual clustering column."""
    adata = _adata_with_cluster_columns("louvain", "louvain_res0_5")
    assert resolve_cluster_key(adata) == "louvain"


def test_both_exact_columns_is_ambiguous_unless_algorithm_given():
    adata = _adata_with_cluster_columns("leiden", "louvain")
    with pytest.raises(ClusterKeyError):
        resolve_cluster_key(adata)
    assert resolve_cluster_key(adata, algorithm="louvain") == "louvain"
    assert resolve_cluster_key(adata, algorithm="leiden") == "leiden"


# --- 3. the unique leiden_res* column -------------------------------------


def test_unique_suffixed_column_resolves():
    """The writer and reader must agree on the cluster column."""
    adata = _adata_with_cluster_columns("leiden_res0_5")
    assert resolve_cluster_key(adata) == "leiden_res0_5"


def test_unique_subcluster_column_resolves():
    adata = _adata_with_cluster_columns("leiden_subcluster")
    assert resolve_cluster_key(adata) == "leiden_subcluster"


def test_algorithm_disambiguates_across_algorithms():
    adata = _adata_with_cluster_columns("leiden_res0_5", "louvain_res0_5")
    assert resolve_cluster_key(adata, algorithm="leiden") == "leiden_res0_5"
    assert resolve_cluster_key(adata, algorithm="louvain") == "louvain_res0_5"


# --- 4. raise only when genuinely ambiguous -------------------------------


def test_multiple_suffixed_columns_is_genuinely_ambiguous():
    adata = _adata_with_cluster_columns("leiden_res0_5", "leiden_res1_0")
    with pytest.raises(ClusterKeyError) as excinfo:
        resolve_cluster_key(adata)
    message = str(excinfo.value)
    assert "leiden_res0_5" in message and "leiden_res1_0" in message


def test_no_cluster_columns_raises_with_actionable_message():
    rng = np.random.default_rng(0)
    adata = ad.AnnData(X=rng.random((10, 3)))
    with pytest.raises(ClusterKeyError) as excinfo:
        resolve_cluster_key(adata)
    assert "clustering" in str(excinfo.value).lower()


def test_cluster_key_error_is_a_value_error():
    """Existing `except ValueError` handlers around annotation must still work."""
    assert issubclass(ClusterKeyError, ValueError)


def test_list_cluster_key_candidates_is_sorted_and_filtered():
    adata = _adata_with_cluster_columns("leiden_res1_0", "leiden", "louvain_res0_5")
    adata.obs["total_counts"] = np.arange(adata.n_obs)
    assert list_cluster_key_candidates(adata) == [
        "leiden",
        "leiden_res1_0",
        "louvain_res0_5",
    ]


# --- end-to-end: annotation after clustering ------------------------------


def test_annotation_resolves_suffixed_key_written_by_clustering():
    """Only a suffixed column exists in this end-to-end scenario.

    Pre-fix this raised "No clustering results found" with a 'leiden' default
    against an AnnData whose only cluster column was 'leiden_res0_5'.
    """
    pytest.importorskip("lobster.services.analysis.enhanced_singlecell_service")
    from lobster.services.analysis.enhanced_singlecell_service import (
        EnhancedSingleCellService,
    )

    rng = np.random.default_rng(0)
    n_obs = 60
    genes = ["CD3D", "CD3E", "MS4A1", "CD79A", "LYZ", "NKG7", "GNLY", "FCGR3A"]
    X = rng.random((n_obs, len(genes))) * 5
    adata = ad.AnnData(X=X)
    adata.var_names = genes
    adata.obs_names = [f"cell{i}" for i in range(n_obs)]
    # The writer's actual output: a resolution-suffixed column and no alias.
    adata.obs["leiden_res0_5"] = pd.Categorical([str(i % 3) for i in range(n_obs)])

    service = EnhancedSingleCellService()
    annotated, stats, ir = service.annotate_cell_types(adata)

    assert stats["cluster_key"] == "leiden_res0_5"
    assert "cell_type" in annotated.obs.columns
    # The IR must record the column actually read, so the exported notebook
    # replays against a column that exists.
    assert "leiden_res0_5" in str(ir.parameters) or "leiden_res0_5" in str(ir.code)


def test_clustering_ir_template_names_the_same_column_the_service_writes():
    """The exported notebook must produce the service's column names.

    The template emitted only `key_added='leiden'` while
    cluster_and_visualize() writes 'leiden_res<res>' plus an alias, so a
    replayed notebook lacked the resolution-suffixed column entirely.
    The notebook should use the same cluster-key resolution as the service.
    """
    from lobster.services.analysis.clustering_service import ClusteringService

    for method in ("deviance", "hvg"):
        ir = ClusteringService()._create_clustering_ir(
            resolution=0.5, feature_selection_method=method, algorithm="leiden"
        )
        code = ir.render()
        assert (
            "leiden_res0_5" in code
        ), f"{method} template does not name the suffixed column: {code}"
        # And the unsuffixed alias, so either name works downstream.
        assert "adata.obs['leiden']" in code


def test_clustering_ir_template_respects_louvain():
    from lobster.services.analysis.clustering_service import ClusteringService

    ir = ClusteringService()._create_clustering_ir(
        resolution=1.0, feature_selection_method="deviance", algorithm="louvain"
    )
    code = ir.render()
    assert "louvain_res1_0" in code
    assert "sc.tl.louvain(" in code


def test_annotation_still_raises_when_no_clustering_was_run():
    """The fix must not paper over a genuinely missing clustering step."""
    pytest.importorskip("lobster.services.analysis.enhanced_singlecell_service")
    from lobster.services.analysis.enhanced_singlecell_service import (
        EnhancedSingleCellService,
        SingleCellError,
    )

    rng = np.random.default_rng(0)
    adata = ad.AnnData(X=rng.random((20, 4)))
    adata.var_names = ["CD3D", "MS4A1", "LYZ", "NKG7"]

    with pytest.raises(SingleCellError):
        EnhancedSingleCellService().annotate_cell_types(adata)
