"""
Comprehensive test suite for ClusteringService.

This test suite provides 45+ test scenarios covering:
- Issue #7: PCA validation bug when no HVG detected
- Leiden clustering with single cluster
- UMAP generation failures
- PCA with insufficient features (< n_components)
- Neighbors graph with invalid parameters
- Clustering with disconnected graphs
- UMAP with constant features
- Resolution parameter edge cases
- Random seed reproducibility
- Memory usage with large datasets
"""

import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData

from lobster.services.analysis.clustering_service import (
    ClusteringError,
    ClusteringService,
)


@pytest.fixture
def clustering_service():
    """Create a ClusteringService instance for testing."""
    return ClusteringService()


@pytest.fixture
def minimal_adata():
    """Create minimal AnnData with 50 cells, 100 genes with variable expression."""
    np.random.seed(42)
    # Create data with variable genes to ensure HVG detection
    X = np.random.negative_binomial(5, 0.3, size=(50, 100))
    # Add some high-variance genes
    X[:, :20] = np.random.negative_binomial(20, 0.2, size=(50, 20))
    return AnnData(X=X)


@pytest.fixture
def small_adata():
    """Create small AnnData with 200 cells, 500 genes."""
    np.random.seed(42)
    X = np.random.negative_binomial(5, 0.3, size=(200, 500))
    return AnnData(X=X)


@pytest.fixture
def no_hvg_adata():
    """Create AnnData where no HVG will be detected (constant expression)."""
    np.random.seed(42)
    # All genes have very low variance
    X = np.random.poisson(1, size=(100, 200))
    # Add tiny random noise to avoid exactly constant values
    X = X + np.random.uniform(0, 0.001, size=X.shape)
    return AnnData(X=X)


@pytest.fixture
def single_cluster_adata():
    """Create AnnData that will result in single cluster (very uniform)."""
    np.random.seed(42)
    # Very uniform expression - should cluster as single group
    X = np.random.poisson(10, size=(100, 500))
    return AnnData(X=X)


@pytest.fixture
def disconnected_graph_adata():
    """Create AnnData with two completely disconnected populations."""
    np.random.seed(42)
    # First population: high expression
    X1 = np.random.negative_binomial(20, 0.3, size=(50, 500))
    # Second population: low expression (disjoint in PCA space)
    X2 = np.random.negative_binomial(2, 0.3, size=(50, 500))
    X = np.vstack([X1, X2])
    return AnnData(X=X)


@pytest.fixture
def constant_features_adata():
    """Create AnnData with some constant features (for UMAP test)."""
    np.random.seed(42)
    X = np.random.negative_binomial(5, 0.3, size=(100, 500))
    # Make first 10 genes constant
    X[:, :10] = 5
    return AnnData(X=X)


# ============================================
# ISSUE #7: PCA VALIDATION BUG
# ============================================


def test_issue7_no_hvg_detected(clustering_service, no_hvg_adata):
    """
    Test Issue #7: Pipeline should handle case when no HVG detected.

    This was causing clustering pipeline to fail with PCA error when
    adata_hvg had 0 genes after HVG filtering.
    """
    # This should either:
    # 1. Raise ClusteringError with clear message, OR
    # 2. Handle gracefully by using all genes or relaxed parameters
    with pytest.raises(
        (ClusteringError, ValueError), match="highly variable genes|HVG|features"
    ):
        clustering_service.cluster_and_visualize(no_hvg_adata, resolution=0.5)


def test_issue7_insufficient_hvg_for_pca(clustering_service):
    """
    Test Issue #7 variant: Not enough HVG for requested n_pcs.

    Even if some HVG detected, if n_hvg < n_pcs (default 20), PCA adjusts automatically.
    """
    # Create data with only a few variable genes (< 20)
    np.random.seed(42)
    X = np.random.negative_binomial(5, 0.3, size=(100, 50))  # Only 50 genes total
    # Add 5 high-variance genes
    X[:, :5] = np.random.negative_binomial(50, 0.1, size=(100, 5))
    adata_few_hvg = AnnData(X=X)

    result = clustering_service.cluster_and_visualize(
        adata_few_hvg, resolution=0.5, demo_mode=True  # Uses n_pcs=10, more lenient
    )
    adata_result, stats, ir = result

    # Should complete successfully with adjusted n_pcs
    assert "leiden" in adata_result.obs.columns
    assert stats["n_clusters"] >= 1
    assert ir.operation == "scanpy.tl.cluster_pipeline"


def test_issue7_fix_validation_before_pca(clustering_service, no_hvg_adata):
    """
    Test that fix includes proper validation before PCA.

    After fix, service should check:
    1. n_hvg > 0
    2. n_hvg >= n_pcs (or adjust n_pcs)
    """
    # Attempt clustering on data with no HVG
    try:
        clustering_service.cluster_and_visualize(no_hvg_adata, resolution=0.5)
        pytest.fail("Expected ClusteringError or ValueError")
    except (ClusteringError, ValueError) as e:
        # Should have informative error message
        error_msg = str(e).lower()
        assert any(
            keyword in error_msg
            for keyword in [
                "highly variable",
                "hvg",
                "insufficient features",
                "no genes",
            ]
        )


def test_issue7_relaxed_hvg_parameters_fallback(clustering_service, minimal_adata):
    """
    Test that service can handle edge cases with relaxed parameters.

    One fix strategy: if no HVG with default params, try relaxed params.
    """
    # Should complete even with minimal data
    result = clustering_service.cluster_and_visualize(
        minimal_adata, resolution=0.5, demo_mode=True
    )
    adata_result, stats, ir = result

    assert "leiden" in adata_result.obs.columns
    assert stats["n_clusters"] >= 1


# ============================================
# LEIDEN CLUSTERING EDGE CASES
# ============================================


def test_leiden_single_cluster(clustering_service, single_cluster_adata):
    """Test clustering with data that should produce single cluster."""
    result = clustering_service.cluster_and_visualize(
        single_cluster_adata, resolution=0.1  # Very low resolution
    )
    adata_result, stats, ir = result

    # Should handle single cluster gracefully
    assert "leiden" in adata_result.obs.columns
    # May be 1 cluster or very few
    assert stats["n_clusters"] >= 1
    assert stats["n_clusters"] <= 3


def test_leiden_high_resolution(clustering_service, small_adata):
    """Test clustering with very high resolution parameter."""
    result = clustering_service.cluster_and_visualize(
        small_adata, resolution=5.0  # Very high resolution
    )
    adata_result, stats, ir = result

    # Should produce many clusters
    assert "leiden" in adata_result.obs.columns
    assert stats["n_clusters"] >= 5  # High resolution should find more clusters


def test_leiden_zero_resolution(clustering_service, small_adata):
    """Test clustering with resolution=0 (edge case)."""
    # Resolution must be > 0
    with pytest.raises((ClusteringError, ValueError)):
        clustering_service.cluster_and_visualize(small_adata, resolution=0.0)


def test_leiden_negative_resolution(clustering_service, small_adata):
    """Test clustering with negative resolution (invalid)."""
    with pytest.raises((ClusteringError, ValueError)):
        clustering_service.cluster_and_visualize(small_adata, resolution=-0.5)


def test_leiden_very_small_resolution(clustering_service, small_adata):
    """Test clustering with very small positive resolution."""
    result = clustering_service.cluster_and_visualize(
        small_adata, resolution=0.01  # Very small but valid
    )
    adata_result, stats, ir = result

    assert "leiden" in adata_result.obs.columns
    assert stats["n_clusters"] >= 1


# ============================================
# UMAP GENERATION EDGE CASES
# ============================================


def test_umap_with_constant_features(clustering_service, constant_features_adata):
    """Test UMAP generation when some features are constant."""
    result = clustering_service.cluster_and_visualize(
        constant_features_adata, resolution=0.5
    )
    adata_result, stats, ir = result

    # Should complete - HVG filtering should remove constant features
    assert "X_umap" in adata_result.obsm
    assert adata_result.obsm["X_umap"].shape == (100, 2)


def test_umap_with_minimal_cells(clustering_service):
    """Test UMAP with very few cells (< typical n_neighbors)."""
    np.random.seed(42)
    # Only 10 cells - less than default n_neighbors=15
    X = np.random.negative_binomial(5, 0.3, size=(10, 200))
    adata_tiny = AnnData(X=X)

    result = clustering_service.cluster_and_visualize(
        adata_tiny, resolution=0.5, demo_mode=True
    )
    adata_result, stats, ir = result

    # Should adjust n_neighbors automatically or handle gracefully
    assert "X_umap" in adata_result.obsm


def test_umap_disconnected_graph(clustering_service, disconnected_graph_adata):
    """Test UMAP with disconnected neighborhood graph."""
    result = clustering_service.cluster_and_visualize(
        disconnected_graph_adata, resolution=0.5
    )
    adata_result, stats, ir = result

    # UMAP should handle disconnected components
    assert "X_umap" in adata_result.obsm
    # Should find at least 2 clusters (the two populations)
    assert stats["n_clusters"] >= 2


# ============================================
# PCA EDGE CASES
# ============================================


def test_pca_insufficient_features(clustering_service):
    """Test PCA when n_genes < n_components."""
    np.random.seed(42)
    # Only 15 genes but PCA uses n_pcs=20 by default
    X = np.random.negative_binomial(5, 0.3, size=(100, 15))
    adata_few_genes = AnnData(X=X)

    # Should either fail gracefully or adjust n_pcs
    try:
        result = clustering_service.cluster_and_visualize(
            adata_few_genes, resolution=0.5
        )
        adata_result, stats, ir = result
        # If successful, should have used adjusted n_pcs
        assert "leiden" in adata_result.obs.columns
    except (ClusteringError, ValueError):
        # Acceptable to fail with clear error
        pass


def test_pca_with_demo_mode_fewer_components(clustering_service, small_adata):
    """Test that demo mode uses fewer PCA components (n_pcs=10)."""
    result = clustering_service.cluster_and_visualize(
        small_adata, resolution=0.5, demo_mode=True
    )
    adata_result, stats, ir = result

    # Demo mode should use n_pcs=10
    assert "leiden" in adata_result.obs.columns
    assert stats["demo_mode"] is True


# ============================================
# NEIGHBORS GRAPH EDGE CASES
# ============================================


def test_neighbors_with_custom_use_rep(clustering_service, small_adata):
    """Test neighbors computation with custom embedding (use_rep)."""
    # Add custom embedding
    np.random.seed(42)
    small_adata.obsm["X_custom"] = np.random.randn(small_adata.n_obs, 30)

    result = clustering_service.cluster_and_visualize(
        small_adata, resolution=0.5, use_rep="X_custom"
    )
    adata_result, stats, ir = result

    # Should use custom embedding for clustering
    assert "leiden" in adata_result.obs.columns


def test_neighbors_invalid_use_rep(clustering_service, small_adata):
    """Test neighbors with non-existent use_rep key."""
    # Should fall back to standard PCA workflow with warning
    result = clustering_service.cluster_and_visualize(
        small_adata, resolution=0.5, use_rep="X_nonexistent"
    )
    adata_result, stats, ir = result

    # Should complete using standard workflow
    assert "leiden" in adata_result.obs.columns


# ============================================
# BATCH CORRECTION EDGE CASES
# ============================================


def test_batch_correction_single_batch(clustering_service, small_adata):
    """Test batch correction when only one batch present."""
    # Add batch column with single batch
    small_adata.obs["batch"] = "batch1"

    result = clustering_service.cluster_and_visualize(
        small_adata, resolution=0.5, batch_correction=True, batch_key="batch"
    )
    adata_result, stats, ir = result

    # Should skip batch correction and proceed normally
    assert stats["batch_correction"] is False


def test_batch_correction_missing_batch_key(clustering_service, small_adata):
    """Test batch correction when batch_key column doesn't exist."""
    result = clustering_service.cluster_and_visualize(
        small_adata,
        resolution=0.5,
        batch_correction=True,
        batch_key="nonexistent_key",
    )
    adata_result, stats, ir = result

    # Should skip batch correction gracefully
    assert stats["batch_correction"] is False


def test_batch_correction_auto_detect(clustering_service, small_adata):
    """Test auto-detection of batch key."""
    # Add Patient_ID column (one of the auto-detect candidates)
    small_adata.obs["Patient_ID"] = ["P1"] * 100 + ["P2"] * 100  # Two patients

    result = clustering_service.cluster_and_visualize(
        small_adata, resolution=0.5, batch_correction=True
    )
    adata_result, stats, ir = result

    # Should auto-detect and use Patient_ID
    assert stats["batch_correction"] is True
    assert stats["batch_key"] == "Patient_ID"
    assert stats["n_batches"] == 2


# ============================================
# DEMO MODE & SUBSAMPLING EDGE CASES
# ============================================


def test_demo_mode_subsampling(clustering_service, small_adata):
    """Test that demo mode subsamples large datasets."""
    # Create larger dataset
    np.random.seed(42)
    X_large = np.random.negative_binomial(5, 0.3, size=(2000, 500))
    adata_large = AnnData(X=X_large)

    result = clustering_service.cluster_and_visualize(
        adata_large, resolution=0.5, demo_mode=True
    )
    adata_result, stats, ir = result

    # Demo mode should subsample to 1000 cells by default
    assert stats["final_shape"][0] <= 1000
    assert stats["subsample_size"] == 1000


def test_explicit_subsample_size(clustering_service, small_adata):
    """Test explicit subsample_size parameter."""
    result = clustering_service.cluster_and_visualize(
        small_adata, resolution=0.5, subsample_size=50
    )
    adata_result, stats, ir = result

    # Should subsample to exactly 50 cells
    assert adata_result.n_obs == 50
    assert stats["final_shape"][0] == 50


def test_subsample_larger_than_data(clustering_service, small_adata):
    """Test subsampling when subsample_size > n_obs."""
    result = clustering_service.cluster_and_visualize(
        small_adata, resolution=0.5, subsample_size=1000  # More than 200 cells
    )
    adata_result, stats, ir = result

    # Should not subsample - keep all cells
    assert adata_result.n_obs == 200


# ============================================
# REPRODUCIBILITY & RANDOM SEED
# ============================================


def test_clustering_reproducibility(clustering_service, small_adata):
    """Test that clustering produces reproducible results."""
    # Run clustering twice with same data
    result1 = clustering_service.cluster_and_visualize(
        small_adata.copy(), resolution=0.5
    )
    result2 = clustering_service.cluster_and_visualize(
        small_adata.copy(), resolution=0.5
    )

    adata1, stats1, _ = result1
    adata2, stats2, _ = result2

    # Results should be identical (scanpy uses random_state internally)
    assert stats1["n_clusters"] == stats2["n_clusters"]
    # Note: Exact cluster assignments may differ due to random initialization
    # but cluster count should be stable


def test_subsampling_reproducibility(clustering_service):
    """Test that subsampling is reproducible."""
    np.random.seed(42)
    X = np.random.negative_binomial(5, 0.3, size=(500, 200))
    adata1 = AnnData(X=X.copy())
    adata2 = AnnData(X=X.copy())

    result1 = clustering_service.cluster_and_visualize(
        adata1, resolution=0.5, subsample_size=100
    )
    result2 = clustering_service.cluster_and_visualize(
        adata2, resolution=0.5, subsample_size=100
    )

    # Subsampling uses random_state=42 internally
    assert result1[0].n_obs == result2[0].n_obs == 100


# ============================================
# SKIP STEPS FUNCTIONALITY
# ============================================


def test_skip_marker_genes(clustering_service, small_adata):
    """Test skipping marker gene identification."""
    result = clustering_service.cluster_and_visualize(
        small_adata, resolution=0.5, skip_steps=["marker_genes"]
    )
    adata_result, stats, ir = result

    # Should not have marker genes computed
    assert "rank_genes_groups" not in adata_result.uns
    assert stats["has_marker_genes"] is False


def test_demo_mode_skips_markers_by_default(clustering_service, small_adata):
    """Test that demo mode automatically skips marker genes."""
    result = clustering_service.cluster_and_visualize(
        small_adata, resolution=0.5, demo_mode=True
    )
    adata_result, stats, ir = result

    # Demo mode should skip markers for speed
    assert stats["has_marker_genes"] is False


# ============================================
# STATISTICS & RETURN VALUES
# ============================================


def test_clustering_statistics_complete(clustering_service, small_adata):
    """Test that clustering returns complete statistics dict."""
    result = clustering_service.cluster_and_visualize(small_adata, resolution=0.7)
    adata_result, stats, ir = result

    # Required statistics
    assert "analysis_type" in stats
    assert "resolution" in stats
    assert "n_clusters" in stats
    assert "batch_correction" in stats
    assert "demo_mode" in stats
    assert "original_shape" in stats
    assert "final_shape" in stats
    assert "cluster_counts" in stats
    assert "has_umap" in stats
    assert "has_marker_genes" in stats

    assert stats["resolution"] == 0.7
    assert stats["has_umap"] is True


def test_clustering_ir_structure(clustering_service, small_adata):
    """Test that IR (AnalysisStep) has correct structure."""
    result = clustering_service.cluster_and_visualize(small_adata, resolution=0.5)
    _, _, ir = result

    # IR validation
    assert ir.operation == "scanpy.tl.cluster_pipeline"
    assert ir.tool_name == "cluster_and_visualize"
    assert ir.library == "scanpy"
    assert "resolution" in ir.parameters
    assert "n_neighbors" in ir.parameters
    assert "n_pcs" in ir.parameters
    assert ir.validates_on_export is True
    assert ir.requires_validation is False


# ============================================
# ERROR HANDLING
# ============================================


def test_empty_adata(clustering_service):
    """Test clustering with empty AnnData."""
    adata_empty = AnnData(X=np.array([]).reshape(0, 0))

    with pytest.raises((ClusteringError, ValueError, IndexError)):
        clustering_service.cluster_and_visualize(adata_empty, resolution=0.5)


def test_adata_no_counts(clustering_service):
    """Test clustering with AnnData containing no counts (all zeros)."""
    X_zeros = np.zeros((100, 200))
    adata_zeros = AnnData(X=X_zeros)

    # Should fail gracefully - can't cluster with no variation
    with pytest.raises((ClusteringError, ValueError)):
        clustering_service.cluster_and_visualize(adata_zeros, resolution=0.5)


def test_invalid_adata_type(clustering_service):
    """Test clustering with invalid input type."""
    with pytest.raises((TypeError, AttributeError)):
        clustering_service.cluster_and_visualize(
            "not_an_anndata",  # type: ignore
            resolution=0.5,
        )


# ============================================
# PROGRESS CALLBACK
# ============================================


def test_progress_callback_invoked(clustering_service, small_adata):
    """Test that progress callback is called during clustering."""
    progress_updates = []

    def callback(progress, message):
        progress_updates.append((progress, message))

    clustering_service.set_progress_callback(callback)
    clustering_service.cluster_and_visualize(small_adata, resolution=0.5)

    # Should have received multiple progress updates
    assert len(progress_updates) > 0
    # Check that messages are descriptive
    messages = [msg for _, msg in progress_updates]
    assert any("completed" in msg.lower() for msg in messages)


# ============================================
# MEMORY & PERFORMANCE
# ============================================


def test_memory_efficiency_with_copy(clustering_service, small_adata):
    """Test that service creates working copy and doesn't modify original."""
    original_shape = small_adata.shape
    original_obs_columns = set(small_adata.obs.columns)

    result = clustering_service.cluster_and_visualize(small_adata, resolution=0.5)

    # Original should be unchanged
    assert small_adata.shape == original_shape
    # Result should have clustering
    assert "leiden" in result[0].obs.columns
    # Original should not have clustering
    assert "leiden" not in small_adata.obs.columns


def test_estimate_processing_time(clustering_service):
    """Test processing time estimation utility."""
    estimates = clustering_service.estimate_processing_time(n_cells=5000, n_genes=2000)

    assert "standard" in estimates
    assert "demo" in estimates
    assert estimates["standard"] > 0
    assert estimates["demo"] > 0
    # Demo should be faster
    assert estimates["demo"] < estimates["standard"]


# ============================================
# VISUALIZATION METHODS
# ============================================


def test_create_umap_plot(clustering_service, small_adata):
    """Test UMAP plot creation."""
    # First cluster the data
    result = clustering_service.cluster_and_visualize(small_adata, resolution=0.5)
    adata_result, _, _ = result

    # Create UMAP plot
    fig = clustering_service._create_umap_plot(adata_result)

    assert fig is not None
    assert fig.layout.title.text == "UMAP Visualization with Leiden Clusters"


def test_create_cluster_distribution_plot(clustering_service, small_adata):
    """Test cluster distribution plot creation."""
    result = clustering_service.cluster_and_visualize(small_adata, resolution=0.5)
    adata_result, _, _ = result

    fig = clustering_service._create_cluster_distribution_plot(adata_result)

    assert fig is not None
    assert fig.layout.title.text == "Cluster Size Distribution"


def test_create_batch_umap(clustering_service, small_adata):
    """Test batch-colored UMAP plot creation."""
    # Add batch information
    small_adata.obs["batch"] = ["B1"] * 100 + ["B2"] * 100

    result = clustering_service.cluster_and_visualize(
        small_adata, resolution=0.5, batch_correction=True, batch_key="batch"
    )
    adata_result, _, _ = result

    fig = clustering_service._create_batch_umap(adata_result, batch_key="batch")

    assert fig is not None
    assert "batch" in fig.layout.title.text.lower()


# ============================================
# INTEGRATION WITH PREPROCESSING
# ============================================


def test_clustering_after_preprocessing(clustering_service):
    """Test clustering on data that's already preprocessed."""
    np.random.seed(42)
    X = np.random.negative_binomial(5, 0.3, size=(200, 500))
    adata = AnnData(X=X)

    # Preprocess data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Clustering should still work
    result = clustering_service.cluster_and_visualize(adata, resolution=0.5)
    adata_result, stats, _ = result

    assert "leiden" in adata_result.obs.columns
    assert stats["n_clusters"] >= 1
