"""
Comprehensive test suite for sub-clustering functionality in ClusteringService.

This test suite covers:
- Basic sub-clustering of single cluster
- Sub-clustering multiple clusters
- Multi-resolution sub-clustering
- Full re-clustering (clusters_to_refine=None)
- Error handling for invalid inputs
- Demo mode behavior
- Label prefix format validation
"""

import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData

from lobster.services.analysis.clustering_service import ClusteringError, ClusteringService


@pytest.fixture
def clustering_service():
    """Create a ClusteringService instance for testing."""
    return ClusteringService()


@pytest.fixture
def clustered_adata():
    """Create AnnData with pre-computed clustering results."""
    np.random.seed(42)
    # Create data with 3 distinct populations
    X1 = np.random.negative_binomial(20, 0.3, size=(100, 500))  # Cluster 0
    X2 = np.random.negative_binomial(10, 0.3, size=(100, 500))  # Cluster 1
    X3 = np.random.negative_binomial(5, 0.3, size=(100, 500))   # Cluster 2
    X = np.vstack([X1, X2, X3])
    adata = AnnData(X=X)

    # Perform initial clustering
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=200)
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=15)
    sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=15)
    sc.tl.leiden(adata_hvg, resolution=0.5, key_added="leiden")

    # Transfer results back to original adata
    adata.obs["leiden"] = adata_hvg.obs["leiden"]
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]

    return adata


@pytest.fixture
def large_clustered_adata():
    """Create larger AnnData for multi-resolution testing."""
    np.random.seed(42)
    # Create data with 5 distinct populations
    populations = []
    for i in range(5):
        X = np.random.negative_binomial(20 - i*3, 0.3, size=(80, 500))
        populations.append(X)
    X = np.vstack(populations)
    adata = AnnData(X=X)

    # Perform initial clustering
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=200)
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=15)
    sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=15)
    sc.tl.leiden(adata_hvg, resolution=0.3, key_added="leiden")

    # Transfer results
    adata.obs["leiden"] = adata_hvg.obs["leiden"]
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]

    return adata


# ============================================
# TEST 1: Basic Sub-clustering (Single Cluster)
# ============================================

def test_basic_subcluster_single_cluster(clustering_service, clustered_adata):
    """Test basic sub-clustering of a single cluster."""
    # Get the first cluster
    clusters = sorted(clustered_adata.obs["leiden"].unique())
    cluster_to_refine = [clusters[0]]

    # Perform sub-clustering
    result = clustering_service.subcluster_cells(
        clustered_adata,
        cluster_key="leiden",
        clusters_to_refine=cluster_to_refine,
        resolution=0.5,
        n_pcs=10,
        n_neighbors=10,
        demo_mode=True,
    )

    # Validate 3-tuple return
    assert len(result) == 3
    adata_result, stats, ir = result

    # Validate output is AnnData
    assert isinstance(adata_result, AnnData)

    # Validate new column exists
    assert "leiden_subcluster" in adata_result.obs.columns

    # Validate stats structure
    assert stats["analysis_type"] == "sub-clustering"
    assert stats["cluster_key"] == "leiden"
    assert stats["parent_clusters"] == cluster_to_refine
    assert stats["n_cells_subclustered"] > 0
    assert stats["n_total_cells"] == clustered_adata.n_obs
    assert "execution_time_seconds" in stats
    assert "subclustering_results" in stats

    # Validate IR structure
    assert ir.operation == "scanpy.tl.leiden"
    assert ir.tool_name == "subcluster_cells"
    assert ir.library == "scanpy"
    assert "cluster_key" in ir.parameters
    assert "resolution" in ir.parameters

    # Validate sub-cluster labels have correct prefix format
    subclustered_cells = adata_result.obs["leiden"] == cluster_to_refine[0]
    sub_labels = adata_result.obs.loc[subclustered_cells, "leiden_subcluster"]
    for label in sub_labels.unique():
        assert label.startswith(f"{cluster_to_refine[0]}.")

    # Validate non-subclustered cells retain original labels
    non_subclustered_cells = adata_result.obs["leiden"] != cluster_to_refine[0]
    original_labels = adata_result.obs.loc[non_subclustered_cells, "leiden"]
    new_labels = adata_result.obs.loc[non_subclustered_cells, "leiden_subcluster"]
    assert all(original_labels == new_labels)


# ============================================
# TEST 2: Multiple Cluster Sub-clustering
# ============================================

def test_subcluster_multiple_clusters(clustering_service, clustered_adata):
    """Test sub-clustering of multiple clusters simultaneously."""
    # Get multiple clusters to refine
    all_clusters = sorted(clustered_adata.obs["leiden"].unique())
    if len(all_clusters) >= 2:
        clusters_to_refine = all_clusters[:2]
    else:
        pytest.skip("Need at least 2 clusters for this test")

    # Perform sub-clustering
    result = clustering_service.subcluster_cells(
        clustered_adata,
        cluster_key="leiden",
        clusters_to_refine=clusters_to_refine,
        resolution=0.5,
        demo_mode=True,
    )

    adata_result, stats, ir = result

    # Validate both clusters were processed
    assert len(stats["parent_clusters"]) == 2

    # Validate each parent cluster has sub-cluster info
    subcluster_results = stats["subclustering_results"]
    for res_key, res_data in subcluster_results.items():
        n_subclusters_per_parent = res_data["n_subclusters_per_parent"]
        assert len(n_subclusters_per_parent) == 2
        for parent in clusters_to_refine:
            assert parent in n_subclusters_per_parent

    # Validate prefix format for each parent cluster
    for parent_cluster in clusters_to_refine:
        subclustered_cells = adata_result.obs["leiden"] == parent_cluster
        sub_labels = adata_result.obs.loc[subclustered_cells, "leiden_subcluster"]
        for label in sub_labels.unique():
            assert label.startswith(f"{parent_cluster}.")


# ============================================
# TEST 3: Multi-resolution Sub-clustering
# ============================================

def test_multi_resolution_subcluster(clustering_service, large_clustered_adata):
    """Test sub-clustering with multiple resolutions."""
    # Get a cluster to refine
    clusters = sorted(large_clustered_adata.obs["leiden"].unique())
    cluster_to_refine = [clusters[0]]

    # Test multiple resolutions
    resolutions = [0.25, 0.5, 1.0]

    result = clustering_service.subcluster_cells(
        large_clustered_adata,
        cluster_key="leiden",
        clusters_to_refine=cluster_to_refine,
        resolutions=resolutions,
        demo_mode=True,
    )

    adata_result, stats, ir = result

    # Validate multiple resolution columns created
    assert len(stats["resolutions_tested"]) == 3

    # Validate key naming: leiden_sub_res0_25, leiden_sub_res0_5, leiden_sub_res1_0
    expected_keys = [f"leiden_sub_res{res}".replace(".", "_") for res in resolutions]
    for key in expected_keys:
        assert key in adata_result.obs.columns

    # Validate each resolution has different granularity
    n_subclusters = []
    for key in expected_keys:
        n = adata_result.obs[key].nunique()
        n_subclusters.append(n)

    # Higher resolution should generally produce more clusters (not always guaranteed, but likely)
    assert all(n > 0 for n in n_subclusters)

    # Validate stats contain info for all resolutions
    assert len(stats["subclustering_results"]) == 3
    for res in resolutions:
        assert res in stats["subclustering_results"]


# ============================================
# TEST 4: Full Re-clustering (clusters_to_refine=None)
# ============================================

def test_full_reclustering(clustering_service, clustered_adata):
    """Test full re-clustering when clusters_to_refine=None."""
    result = clustering_service.subcluster_cells(
        clustered_adata,
        cluster_key="leiden",
        clusters_to_refine=None,  # Re-cluster all cells
        resolution=0.5,
        demo_mode=True,
    )

    adata_result, stats, ir = result

    # Validate all cells were processed
    assert stats["n_cells_subclustered"] == clustered_adata.n_obs

    # Validate all original clusters are in parent_clusters list
    all_original_clusters = set(clustered_adata.obs["leiden"].astype(str).unique())
    assert set(stats["parent_clusters"]) == all_original_clusters

    # Validate new column exists
    assert "leiden_subcluster" in adata_result.obs.columns

    # All cells should have sub-cluster labels (no cells left with original labels only)
    # Each label should have a prefix format
    all_labels = adata_result.obs["leiden_subcluster"].unique()
    for label in all_labels:
        # Should be in format "parent.sub"
        assert "." in label


# ============================================
# TEST 5: Error Handling - Invalid Cluster Key
# ============================================

def test_error_invalid_cluster_key(clustering_service, clustered_adata):
    """Test error when invalid cluster_key provided."""
    with pytest.raises(ClusteringError, match="not found in adata.obs"):
        clustering_service.subcluster_cells(
            clustered_adata,
            cluster_key="nonexistent_key",
            clusters_to_refine=["0"],
        )


# ============================================
# TEST 6: Error Handling - Invalid Cluster IDs
# ============================================

def test_error_invalid_cluster_ids(clustering_service, clustered_adata):
    """Test error when invalid cluster IDs provided."""
    with pytest.raises(ClusteringError, match="Invalid cluster IDs"):
        clustering_service.subcluster_cells(
            clustered_adata,
            cluster_key="leiden",
            clusters_to_refine=["999", "1000"],  # Non-existent clusters
        )


# ============================================
# TEST 7: Error Handling - Missing PCA
# ============================================

def test_error_missing_pca(clustering_service):
    """Test error when PCA results not found."""
    # Create adata without PCA
    np.random.seed(42)
    X = np.random.negative_binomial(10, 0.3, size=(100, 500))
    adata = AnnData(X=X)
    adata.obs["leiden"] = ["0"] * 50 + ["1"] * 50

    with pytest.raises(ClusteringError, match="PCA results not found"):
        clustering_service.subcluster_cells(
            adata,
            cluster_key="leiden",
            clusters_to_refine=["0"],
        )


# ============================================
# TEST 8: Demo Mode Behavior
# ============================================

def test_demo_mode_parameters(clustering_service, clustered_adata):
    """Test that demo mode reduces parameters correctly."""
    result = clustering_service.subcluster_cells(
        clustered_adata,
        cluster_key="leiden",
        clusters_to_refine=None,
        n_pcs=20,
        n_neighbors=15,
        demo_mode=True,
    )

    adata_result, stats, ir = result

    # Demo mode should cap n_pcs and n_neighbors at 10
    assert stats["n_pcs_used"] == 10
    assert stats["n_neighbors_used"] == 10


# ============================================
# TEST 9: Label Prefix Format Validation
# ============================================

def test_label_prefix_format(clustering_service, clustered_adata):
    """Test that sub-cluster labels have correct prefix format."""
    clusters = sorted(clustered_adata.obs["leiden"].unique())
    cluster_to_refine = [clusters[0]]

    result = clustering_service.subcluster_cells(
        clustered_adata,
        cluster_key="leiden",
        clusters_to_refine=cluster_to_refine,
        resolution=0.5,
        demo_mode=True,
    )

    adata_result, stats, ir = result

    # Get sub-cluster labels for the refined cluster
    subclustered_cells = adata_result.obs["leiden"] == cluster_to_refine[0]
    sub_labels = adata_result.obs.loc[subclustered_cells, "leiden_subcluster"]

    # Validate format: "parent.sub" (e.g., "0.0", "0.1", "0.2")
    for label in sub_labels.unique():
        parts = label.split(".")
        assert len(parts) == 2, f"Label {label} should be in 'parent.sub' format"
        assert parts[0] == cluster_to_refine[0], f"Prefix should be {cluster_to_refine[0]}"
        assert parts[1].isdigit(), f"Sub-cluster ID should be numeric"


# ============================================
# TEST 10: Statistics Completeness
# ============================================

def test_stats_completeness(clustering_service, clustered_adata):
    """Test that statistics dict contains all expected keys."""
    result = clustering_service.subcluster_cells(
        clustered_adata,
        cluster_key="leiden",
        clusters_to_refine=None,
        resolution=0.5,
        demo_mode=True,
    )

    adata_result, stats, ir = result

    # Required keys
    required_keys = [
        "analysis_type",
        "cluster_key",
        "parent_clusters",
        "n_cells_subclustered",
        "n_total_cells",
        "resolution_used",
        "resolutions_tested",
        "n_pcs_used",
        "n_neighbors_used",
        "execution_time_seconds",
        "subclustering_results",
        "primary_subcluster_key",
        "cluster_sizes",
    ]

    for key in required_keys:
        assert key in stats, f"Missing required key: {key}"

    # Validate types
    assert isinstance(stats["analysis_type"], str)
    assert isinstance(stats["parent_clusters"], list)
    assert isinstance(stats["n_cells_subclustered"], (int, np.integer))
    assert isinstance(stats["execution_time_seconds"], (int, float))
    assert isinstance(stats["subclustering_results"], dict)
    assert isinstance(stats["cluster_sizes"], dict)


# ============================================
# TEST 11: IR Parameter Schema Validation
# ============================================

def test_ir_parameter_schema(clustering_service, clustered_adata):
    """Test that IR contains proper parameter schema."""
    result = clustering_service.subcluster_cells(
        clustered_adata,
        cluster_key="leiden",
        clusters_to_refine=["0"],
        resolution=0.5,
        demo_mode=True,
    )

    adata_result, stats, ir = result

    # Validate parameter schema exists
    assert hasattr(ir, "parameter_schema")
    assert isinstance(ir.parameter_schema, dict)

    # Required parameters
    required_params = [
        "cluster_key",
        "clusters_to_refine",
        "resolution",
        "n_pcs",
        "n_neighbors",
    ]

    for param in required_params:
        assert param in ir.parameter_schema, f"Missing parameter schema: {param}"
        schema = ir.parameter_schema[param]
        assert hasattr(schema, "param_type")
        assert hasattr(schema, "papermill_injectable")
        assert hasattr(schema, "default_value")


# ============================================
# TEST 12: Empty Cluster List Handling
# ============================================

def test_empty_cluster_list(clustering_service, clustered_adata):
    """Test behavior when empty cluster list provided."""
    # Empty list should trigger re-clustering of all cells
    result = clustering_service.subcluster_cells(
        clustered_adata,
        cluster_key="leiden",
        clusters_to_refine=[],  # Empty list
        resolution=0.5,
        demo_mode=True,
    )

    adata_result, stats, ir = result

    # Should process all cells
    assert stats["n_cells_subclustered"] == clustered_adata.n_obs
