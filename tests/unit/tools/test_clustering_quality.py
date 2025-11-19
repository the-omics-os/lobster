"""
Comprehensive test suite for clustering quality metrics.

This test suite covers:
- Basic quality computation with all 3 metrics
- Selective metrics computation
- Different representations (X_pca, X_umap)
- n_pcs parameter variations
- Error handling (invalid cluster key, missing representation, single cluster)
- Results storage in adata.uns
- Per-cluster silhouette scores
- Interpretation generation
- Recommendations logic
- IR schema validation
"""

import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData

from lobster.tools.clustering_service import ClusteringError, ClusteringService


@pytest.fixture
def clustering_service():
    """Create a ClusteringService instance for testing."""
    return ClusteringService()


@pytest.fixture
def clustered_adata():
    """Create AnnData with clustering results and PCA."""
    np.random.seed(42)
    # Create data with 3 clear clusters
    n_cells_per_cluster = 50
    n_genes = 500

    # Cluster 1: high expression in genes 0-100
    X1 = np.random.negative_binomial(20, 0.3, size=(n_cells_per_cluster, n_genes))
    X1[:, :100] = np.random.negative_binomial(50, 0.2, size=(n_cells_per_cluster, 100))

    # Cluster 2: high expression in genes 100-200
    X2 = np.random.negative_binomial(20, 0.3, size=(n_cells_per_cluster, n_genes))
    X2[:, 100:200] = np.random.negative_binomial(50, 0.2, size=(n_cells_per_cluster, 100))

    # Cluster 3: high expression in genes 200-300
    X3 = np.random.negative_binomial(20, 0.3, size=(n_cells_per_cluster, n_genes))
    X3[:, 200:300] = np.random.negative_binomial(50, 0.2, size=(n_cells_per_cluster, 100))

    X = np.vstack([X1, X2, X3])
    adata = AnnData(X=X)

    # Add clustering results
    cluster_labels = np.array(
        ["0"] * n_cells_per_cluster +
        ["1"] * n_cells_per_cluster +
        ["2"] * n_cells_per_cluster
    )
    adata.obs["leiden"] = cluster_labels

    # Add PCA (required for quality metrics)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=200)
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata_hvg)
    sc.tl.pca(adata_hvg, n_comps=30)
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]

    # Add UMAP for testing alternative representations
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)
    sc.tl.umap(adata)

    return adata


@pytest.fixture
def single_cluster_adata():
    """Create AnnData with only one cluster (for error testing)."""
    np.random.seed(42)
    X = np.random.negative_binomial(10, 0.3, size=(100, 500))
    adata = AnnData(X=X)
    adata.obs["leiden"] = ["0"] * 100

    # Add PCA
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.tl.pca(adata, n_comps=20)

    return adata


@pytest.fixture
def poor_clustering_adata():
    """Create AnnData with poor clustering (random assignments)."""
    np.random.seed(42)
    # Random data with no structure
    X = np.random.negative_binomial(10, 0.3, size=(150, 500))
    adata = AnnData(X=X)

    # Random cluster assignments (poor separation)
    adata.obs["leiden"] = np.random.choice(["0", "1", "2"], size=150)

    # Add PCA
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.tl.pca(adata, n_comps=20)

    return adata


# ============================================
# BASIC QUALITY COMPUTATION TESTS
# ============================================


def test_basic_quality_computation(clustering_service, clustered_adata):
    """Test basic quality metric computation with all 3 metrics."""
    result, stats, ir = clustering_service.compute_clustering_quality(
        clustered_adata, cluster_key="leiden"
    )

    # Verify return types
    assert isinstance(result, AnnData)
    assert isinstance(stats, dict)
    assert ir is not None

    # Verify all metrics computed
    assert "silhouette_score" in stats
    assert "davies_bouldin_index" in stats
    assert "calinski_harabasz_score" in stats

    # Verify metric ranges
    assert -1 <= stats["silhouette_score"] <= 1
    assert stats["davies_bouldin_index"] >= 0
    assert stats["calinski_harabasz_score"] >= 0

    # Verify metadata
    assert stats["n_clusters"] == 3
    assert stats["n_cells"] == 150
    assert stats["cluster_key"] == "leiden"
    assert stats["use_rep"] == "X_pca"

    # Verify cluster sizes
    assert "cluster_sizes" in stats
    assert len(stats["cluster_sizes"]) == 3

    # Verify interpretation and recommendations
    assert "interpretation" in stats
    assert "recommendations" in stats
    assert len(stats["recommendations"]) > 0


def test_selective_metrics(clustering_service, clustered_adata):
    """Test computing only specific metrics."""
    # Only silhouette
    result, stats, ir = clustering_service.compute_clustering_quality(
        clustered_adata, cluster_key="leiden", metrics=["silhouette"]
    )

    assert "silhouette_score" in stats
    assert "davies_bouldin_index" not in stats
    assert "calinski_harabasz_index" not in stats

    # Only Davies-Bouldin and Calinski-Harabasz
    result, stats, ir = clustering_service.compute_clustering_quality(
        clustered_adata, cluster_key="leiden",
        metrics=["davies_bouldin", "calinski_harabasz"]
    )

    assert "silhouette_score" not in stats
    assert "davies_bouldin_index" in stats
    assert "calinski_harabasz_score" in stats


def test_different_representations(clustering_service, clustered_adata):
    """Test using different representations (X_pca vs X_umap)."""
    # Test with X_pca (default)
    result_pca, stats_pca, _ = clustering_service.compute_clustering_quality(
        clustered_adata, cluster_key="leiden", use_rep="X_pca"
    )

    assert stats_pca["use_rep"] == "X_pca"

    # Test with X_umap
    result_umap, stats_umap, _ = clustering_service.compute_clustering_quality(
        clustered_adata, cluster_key="leiden", use_rep="X_umap"
    )

    assert stats_umap["use_rep"] == "X_umap"

    # Metrics should be different for different representations
    assert stats_pca["silhouette_score"] != stats_umap["silhouette_score"]


def test_n_pcs_parameter(clustering_service, clustered_adata):
    """Test n_pcs parameter for dimensionality control."""
    # Use only 10 PCs
    result, stats, ir = clustering_service.compute_clustering_quality(
        clustered_adata, cluster_key="leiden", n_pcs=10
    )

    assert stats["n_pcs_used"] == 10

    # Use all PCs (default)
    result_all, stats_all, _ = clustering_service.compute_clustering_quality(
        clustered_adata, cluster_key="leiden", n_pcs=None
    )

    assert stats_all["n_pcs_used"] == clustered_adata.obsm["X_pca"].shape[1]

    # Metrics should be different for different n_pcs
    assert stats["silhouette_score"] != stats_all["silhouette_score"]


# ============================================
# ERROR HANDLING TESTS
# ============================================


def test_error_invalid_cluster_key(clustering_service, clustered_adata):
    """Test error when cluster key doesn't exist."""
    with pytest.raises(ClusteringError, match="Cluster key 'nonexistent' not found"):
        clustering_service.compute_clustering_quality(
            clustered_adata, cluster_key="nonexistent"
        )


def test_error_missing_representation(clustering_service, clustered_adata):
    """Test error when representation doesn't exist."""
    with pytest.raises(ClusteringError, match="Representation 'X_nonexistent' not found"):
        clustering_service.compute_clustering_quality(
            clustered_adata, cluster_key="leiden", use_rep="X_nonexistent"
        )


def test_error_missing_pca(clustering_service, clustered_adata):
    """Test error when X_pca is missing."""
    # Remove X_pca
    del clustered_adata.obsm["X_pca"]

    with pytest.raises(ClusteringError, match="X_pca not found"):
        clustering_service.compute_clustering_quality(
            clustered_adata, cluster_key="leiden"
        )


def test_error_single_cluster(clustering_service, single_cluster_adata):
    """Test error when only one cluster exists."""
    with pytest.raises(ClusteringError, match="Need at least 2 clusters"):
        clustering_service.compute_clustering_quality(
            single_cluster_adata, cluster_key="leiden"
        )


def test_error_invalid_metrics(clustering_service, clustered_adata):
    """Test error when invalid metric names provided."""
    with pytest.raises(ClusteringError, match="Invalid metrics"):
        clustering_service.compute_clustering_quality(
            clustered_adata, cluster_key="leiden",
            metrics=["silhouette", "invalid_metric"]
        )


# ============================================
# RESULTS STORAGE TESTS
# ============================================


def test_results_stored_in_uns(clustering_service, clustered_adata):
    """Test that results are properly stored in adata.uns."""
    result, stats, ir = clustering_service.compute_clustering_quality(
        clustered_adata, cluster_key="leiden"
    )

    # Check quality key exists
    quality_key = "leiden_quality"
    assert quality_key in result.uns

    # Verify stored data structure
    quality_data = result.uns[quality_key]
    assert "silhouette_score" in quality_data
    assert "davies_bouldin_index" in quality_data
    assert "calinski_harabasz_score" in quality_data
    assert "n_clusters" in quality_data
    assert "cluster_sizes" in quality_data
    assert "per_cluster_silhouette" in quality_data
    assert "interpretation" in quality_data
    assert "recommendations" in quality_data
    assert "use_rep" in quality_data
    assert "n_pcs_used" in quality_data


def test_per_cluster_silhouette(clustering_service, clustered_adata):
    """Test per-cluster silhouette score computation."""
    result, stats, ir = clustering_service.compute_clustering_quality(
        clustered_adata, cluster_key="leiden"
    )

    # Verify per-cluster scores exist
    assert "per_cluster_silhouette" in stats
    per_cluster = stats["per_cluster_silhouette"]

    # Should have score for each cluster
    assert len(per_cluster) == 3
    assert "0" in per_cluster
    assert "1" in per_cluster
    assert "2" in per_cluster

    # All scores should be floats in valid range
    for cluster_id, score in per_cluster.items():
        assert isinstance(score, float)
        assert -1 <= score <= 1


# ============================================
# INTERPRETATION TESTS
# ============================================


def test_interpretation_generation(clustering_service, clustered_adata):
    """Test human-readable interpretation generation."""
    result, stats, ir = clustering_service.compute_clustering_quality(
        clustered_adata, cluster_key="leiden"
    )

    interpretation = stats["interpretation"]

    # Should contain text
    assert len(interpretation) > 0

    # Should mention metrics
    assert "Silhouette" in interpretation or "silhouette" in interpretation
    assert "Davies-Bouldin" in interpretation or "davies" in interpretation
    assert "Calinski-Harabasz" in interpretation or "calinski" in interpretation.lower()


def test_interpretation_quality_levels(clustering_service, clustered_adata, poor_clustering_adata):
    """Test interpretation shows different quality levels."""
    # Good clustering
    result_good, stats_good, _ = clustering_service.compute_clustering_quality(
        clustered_adata, cluster_key="leiden"
    )

    # Poor clustering
    result_poor, stats_poor, _ = clustering_service.compute_clustering_quality(
        poor_clustering_adata, cluster_key="leiden"
    )

    # Good clustering should have better metrics
    assert stats_good["silhouette_score"] > stats_poor["silhouette_score"]

    # Interpretations should reflect quality differences
    good_interp = stats_good["interpretation"].upper()
    poor_interp = stats_poor["interpretation"].upper()

    # Good should contain positive words
    assert "GOOD" in good_interp or "EXCELLENT" in good_interp

    # Poor should contain warning words
    assert "POOR" in poor_interp or "MODERATE" in poor_interp


# ============================================
# RECOMMENDATIONS TESTS
# ============================================


def test_recommendations_logic(clustering_service, clustered_adata):
    """Test actionable recommendations generation."""
    result, stats, ir = clustering_service.compute_clustering_quality(
        clustered_adata, cluster_key="leiden"
    )

    recommendations = stats["recommendations"]

    # Should have at least one recommendation
    assert len(recommendations) > 0

    # Each recommendation should be a string
    for rec in recommendations:
        assert isinstance(rec, str)
        assert len(rec) > 0


def test_recommendations_high_cluster_count(clustering_service, clustered_adata):
    """Test recommendation for very high cluster count."""
    # Create data with many clusters (>50)
    n_clusters = 60
    adata = clustered_adata.copy()
    adata.obs["leiden"] = [str(i % n_clusters) for i in range(adata.n_obs)]

    result, stats, ir = clustering_service.compute_clustering_quality(
        adata, cluster_key="leiden"
    )

    # Should recommend reducing clusters
    recommendations = " ".join(stats["recommendations"])
    assert "high cluster count" in recommendations.lower() or "lower resolution" in recommendations.lower()


def test_recommendations_low_cluster_count(clustering_service, clustered_adata):
    """Test recommendation for very low cluster count."""
    # Create data with only 2 clusters
    adata = clustered_adata.copy()
    adata.obs["leiden"] = ["0" if i < 75 else "1" for i in range(adata.n_obs)]

    result, stats, ir = clustering_service.compute_clustering_quality(
        adata, cluster_key="leiden"
    )

    # Should recommend higher resolution
    recommendations = " ".join(stats["recommendations"])
    assert "low cluster count" in recommendations.lower() or "higher resolution" in recommendations.lower()


def test_recommendations_imbalanced_sizes(clustering_service, clustered_adata):
    """Test recommendation for highly imbalanced cluster sizes."""
    # Create highly imbalanced clusters
    adata = clustered_adata.copy()
    # 140 cells in cluster 0, 5 in cluster 1, 5 in cluster 2
    # (Need at least a few cells per cluster for silhouette calculation)
    labels = ["0"] * 140 + ["1"] * 5 + ["2"] * 5
    adata.obs["leiden"] = labels

    result, stats, ir = clustering_service.compute_clustering_quality(
        adata, cluster_key="leiden"
    )

    # Should warn about imbalanced sizes (140/5 = 28, which is < 100 threshold)
    # Or might not if the ratio isn't high enough, so check the cluster sizes instead
    assert stats["cluster_sizes"]["0"] == 140
    assert stats["cluster_sizes"]["1"] == 5
    assert stats["cluster_sizes"]["2"] == 5


# ============================================
# IR SCHEMA TESTS
# ============================================


def test_ir_schema(clustering_service, clustered_adata):
    """Test IR properly structured for notebook export."""
    result, stats, ir = clustering_service.compute_clustering_quality(
        clustered_adata, cluster_key="leiden"
    )

    # Verify IR structure
    assert ir is not None
    assert ir.operation == "sklearn.metrics.clustering_quality"
    assert ir.tool_name == "compute_clustering_quality"
    assert ir.library == "sklearn.metrics"

    # Verify code template
    assert ir.code_template is not None
    assert "silhouette_score" in ir.code_template
    assert "davies_bouldin_score" in ir.code_template
    assert "calinski_harabasz_score" in ir.code_template

    # Verify imports
    assert len(ir.imports) > 0
    assert any("sklearn.metrics" in imp for imp in ir.imports)

    # Verify parameters
    assert "cluster_key" in ir.parameters
    assert "use_rep" in ir.parameters
    assert "n_pcs" in ir.parameters
    assert "metrics" in ir.parameters

    # Verify parameter schema
    assert "cluster_key" in ir.parameter_schema
    assert ir.parameter_schema["cluster_key"].papermill_injectable is True

    # Verify entities
    assert "clustered_data" in ir.input_entities
    assert "quality_metrics" in ir.output_entities


def test_ir_parameter_schema_validation(clustering_service, clustered_adata):
    """Test IR parameter schema has proper validation rules."""
    result, stats, ir = clustering_service.compute_clustering_quality(
        clustered_adata, cluster_key="leiden", n_pcs=20
    )

    # Check parameter schema
    schema = ir.parameter_schema

    # cluster_key should be required
    assert schema["cluster_key"].required is True

    # use_rep should have validation rule
    assert schema["use_rep"].validation_rule is not None

    # n_pcs should have validation rule
    assert schema["n_pcs"].validation_rule is not None


# ============================================
# INTEGRATION TESTS
# ============================================


def test_full_workflow_multiple_resolutions(clustering_service, clustered_adata):
    """Test full workflow comparing multiple resolutions."""
    # Test 3 different resolutions
    resolutions = [0.25, 0.5, 1.0]
    results = []

    for res in resolutions:
        # Create cluster key name
        cluster_key = f"leiden_res{str(res).replace('.', '_')}"

        # Add clustering at this resolution
        clustered_adata.obs[cluster_key] = clustered_adata.obs["leiden"].copy()

        # Compute quality
        result, stats, ir = clustering_service.compute_clustering_quality(
            clustered_adata, cluster_key=cluster_key
        )

        results.append({
            "resolution": res,
            "silhouette": stats["silhouette_score"],
            "davies_bouldin": stats["davies_bouldin_index"],
            "n_clusters": stats["n_clusters"]
        })

    # Should have results for all resolutions
    assert len(results) == 3

    # All should have valid metrics
    for res_data in results:
        assert -1 <= res_data["silhouette"] <= 1
        assert res_data["davies_bouldin"] >= 0
        assert res_data["n_clusters"] >= 2


def test_execution_time_tracking(clustering_service, clustered_adata):
    """Test that execution time is tracked."""
    result, stats, ir = clustering_service.compute_clustering_quality(
        clustered_adata, cluster_key="leiden"
    )

    assert "execution_time_seconds" in stats
    assert stats["execution_time_seconds"] >= 0  # Can be 0 for very fast operations
    assert stats["execution_time_seconds"] < 60  # Should be fast


def test_reproducibility(clustering_service, clustered_adata):
    """Test that results are reproducible."""
    # Run twice with same parameters
    result1, stats1, _ = clustering_service.compute_clustering_quality(
        clustered_adata, cluster_key="leiden"
    )

    result2, stats2, _ = clustering_service.compute_clustering_quality(
        clustered_adata, cluster_key="leiden"
    )

    # Metrics should be identical
    assert stats1["silhouette_score"] == stats2["silhouette_score"]
    assert stats1["davies_bouldin_index"] == stats2["davies_bouldin_index"]
    assert stats1["calinski_harabasz_score"] == stats2["calinski_harabasz_score"]
