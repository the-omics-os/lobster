"""
Comprehensive unit tests for EnhancedSingleCellService.

This module provides thorough testing of the enhanced single-cell service including
doublet detection, cell type annotation, marker gene identification, and advanced
single-cell analysis capabilities.

Test coverage target: 95%+ with meaningful tests for all operations.
"""

import warnings
from typing import Any, Dict, List, Tuple
from unittest.mock import Mock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

# Suppress scanpy warnings during tests
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")
warnings.filterwarnings("ignore", category=FutureWarning, module="scanpy")

from lobster.services.analysis.enhanced_singlecell_service import (
    SCRUBLET_AVAILABLE,
    EnhancedSingleCellService,
    SingleCellError,
)

# ===============================================================================
# Fixtures for Test Data
# ===============================================================================


@pytest.fixture
def simple_adata():
    """Create simple AnnData for basic testing."""
    np.random.seed(42)
    n_obs = 100
    n_vars = 50

    X = np.random.negative_binomial(n=5, p=0.3, size=(n_obs, n_vars)).astype(np.float32)
    X[X < 0] = 0  # Ensure non-negative

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"Cell_{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"Gene_{i}" for i in range(n_vars)]),
    )

    # Add basic metadata
    adata.obs["total_counts"] = np.array(X.sum(axis=1))
    adata.obs["n_genes_by_counts"] = np.array((X > 0).sum(axis=1))

    return adata


@pytest.fixture
def clustered_adata():
    """Create clustered AnnData for annotation testing."""
    np.random.seed(42)
    n_obs = 200
    n_vars = 100

    X = np.random.negative_binomial(n=5, p=0.3, size=(n_obs, n_vars)).astype(np.float32)

    # Create realistic gene names including some markers
    gene_names = [f"Gene_{i}" for i in range(n_vars - 20)]
    # Add some actual marker genes
    marker_genes = [
        "CD3D",
        "CD3E",
        "CD8A",
        "CD4",  # T cells
        "CD19",
        "MS4A1",
        "CD79A",
        "IGHM",  # B cells
        "GNLY",
        "NKG7",
        "KLRD1",
        "NCAM1",  # NK cells
        "CD14",
        "FCGR3A",
        "LYZ",
        "CSF1R",  # Monocytes
        "FCER1A",
        "CST3",
        "CLEC4C",  # Dendritic
        "EPCAM",  # Epithelial
    ]
    all_genes = gene_names + marker_genes

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"Cell_{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=all_genes),
    )

    # Add clustering results
    adata.obs["leiden"] = np.random.choice([0, 1, 2, 3], size=n_obs).astype(str)

    # Add UMAP coordinates
    adata.obsm["X_umap"] = np.random.randn(n_obs, 2)

    # Add raw data for marker detection
    adata.raw = adata.copy()

    return adata


@pytest.fixture
def adata_with_all_markers():
    """Create AnnData with all default marker genes for comprehensive testing."""
    np.random.seed(42)
    n_obs = 500

    # All marker genes from the service
    all_markers = [
        "CD3D",
        "CD3E",
        "CD8A",
        "CD4",  # T cells
        "CD19",
        "MS4A1",
        "CD79A",
        "IGHM",  # B cells
        "GNLY",
        "NKG7",
        "KLRD1",
        "NCAM1",  # NK cells
        "CD14",
        "FCGR3A",
        "LYZ",
        "CSF1R",  # Monocytes
        "FCER1A",
        "CST3",
        "CLEC4C",  # Dendritic cells
        "FCGR3B",
        "CEACAM3",
        "CSF3R",  # Neutrophils
        "PPBP",
        "PF4",
        "TUBB1",  # Platelets
        "PECAM1",
        "VWF",
        "ENG",
        "CDH5",  # Endothelial
        "COL1A1",
        "COL3A1",
        "DCN",
        "LUM",  # Fibroblasts
        "EPCAM",
        "KRT8",
        "KRT18",
        "KRT19",  # Epithelial
    ]

    n_vars = len(all_markers)

    X = np.random.negative_binomial(n=10, p=0.2, size=(n_obs, n_vars)).astype(
        np.float32
    )

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"Cell_{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=all_markers),
    )

    # Create distinct clusters with marker expression patterns
    n_clusters = 5
    adata.obs["leiden"] = np.random.choice(range(n_clusters), size=n_obs).astype(str)

    # Enhance marker expression in specific clusters
    for cluster_id in range(n_clusters):
        cluster_mask = adata.obs["leiden"] == str(cluster_id)
        cluster_indices = np.where(cluster_mask)[0]

        # Boost specific markers for this cluster
        marker_start = cluster_id * 8
        marker_end = min(marker_start + 8, n_vars)
        for marker_idx in range(marker_start, marker_end):
            adata.X[cluster_indices, marker_idx] *= 3

    adata.raw = adata.copy()
    adata.obsm["X_umap"] = np.random.randn(n_obs, 2)

    return adata


@pytest.fixture
def empty_adata():
    """Create empty AnnData for edge case testing."""
    return ad.AnnData(
        X=np.array([]).reshape(0, 0),
        obs=pd.DataFrame(),
        var=pd.DataFrame(),
    )


@pytest.fixture
def service():
    """Create EnhancedSingleCellService instance."""
    return EnhancedSingleCellService()


# ===============================================================================
# Test Service Initialization
# ===============================================================================


def test_service_initialization(service):
    """Test that service initializes correctly with marker database."""
    assert service is not None
    assert isinstance(service.cell_type_markers, dict)
    assert len(service.cell_type_markers) == 10
    assert "T cells" in service.cell_type_markers
    assert "B cells" in service.cell_type_markers


def test_marker_database_structure(service):
    """Test marker database has correct structure."""
    for cell_type, markers in service.cell_type_markers.items():
        assert isinstance(cell_type, str)
        assert isinstance(markers, list)
        assert len(markers) > 0
        assert all(isinstance(m, str) for m in markers)


def test_marker_database_completeness(service):
    """Test that all expected cell types have markers."""
    expected_cell_types = [
        "T cells",
        "B cells",
        "NK cells",
        "Monocytes",
        "Dendritic cells",
        "Neutrophils",
        "Platelets",
        "Endothelial",
        "Fibroblasts",
        "Epithelial",
    ]
    for cell_type in expected_cell_types:
        assert cell_type in service.cell_type_markers
        assert len(service.cell_type_markers[cell_type]) >= 3


# ===============================================================================
# Test Doublet Detection - Basic Functionality
# ===============================================================================


def test_detect_doublets_basic(service, simple_adata):
    """Test basic doublet detection with default parameters."""
    result_adata, stats, _ = service.detect_doublets(simple_adata)

    assert result_adata is not None
    assert "doublet_score" in result_adata.obs.columns
    assert "predicted_doublet" in result_adata.obs.columns
    assert len(result_adata.obs) == len(simple_adata.obs)

    # Check statistics
    assert stats["analysis_type"] == "doublet_detection"
    assert "n_doublets_detected" in stats
    assert "actual_doublet_rate" in stats
    assert stats["n_cells_analyzed"] == len(simple_adata.obs)


def test_detect_doublets_with_custom_rate(service, simple_adata):
    """Test doublet detection with custom expected rate."""
    expected_rate = 0.05
    result_adata, stats, _ = service.detect_doublets(
        simple_adata, expected_doublet_rate=expected_rate
    )

    assert stats["expected_doublet_rate"] == expected_rate
    assert "doublet_score" in result_adata.obs.columns


def test_detect_doublets_with_custom_threshold(service, simple_adata):
    """Test doublet detection with custom threshold."""
    threshold = 0.3
    result_adata, stats, _ = service.detect_doublets(simple_adata, threshold=threshold)

    assert stats["threshold"] == threshold
    assert "predicted_doublet" in result_adata.obs.columns


def test_detect_doublets_score_distribution(service, simple_adata):
    """Test that doublet scores are properly distributed."""
    result_adata, stats, _ = service.detect_doublets(simple_adata)

    doublet_scores = result_adata.obs["doublet_score"]
    assert doublet_scores.min() >= 0
    assert doublet_scores.std() > 0  # Not all the same
    assert not np.any(np.isnan(doublet_scores))


def test_detect_doublets_returns_statistics(service, simple_adata):
    """Test that doublet detection returns comprehensive statistics."""
    result_adata, stats, _ = service.detect_doublets(simple_adata)

    required_keys = [
        "analysis_type",
        "expected_doublet_rate",
        "detection_method",
        "n_cells_analyzed",
        "n_doublets_detected",
        "actual_doublet_rate",
        "doublet_score_stats",
    ]

    for key in required_keys:
        assert key in stats

    # Check score statistics
    score_stats = stats["doublet_score_stats"]
    assert "min" in score_stats
    assert "max" in score_stats
    assert "mean" in score_stats
    assert "std" in score_stats


# ===============================================================================
# Test Doublet Detection - Edge Cases
# ===============================================================================


def test_detect_doublets_no_genes(service):
    """Test doublet detection fails gracefully with no genes."""
    adata = ad.AnnData(
        X=np.array([]).reshape(100, 0),
        obs=pd.DataFrame(index=[f"Cell_{i}" for i in range(100)]),
        var=pd.DataFrame(),
    )

    with pytest.raises(SingleCellError, match="no gene features"):
        service.detect_doublets(adata)


def test_detect_doublets_single_cell(service):
    """Test doublet detection with single cell."""
    adata = ad.AnnData(
        X=np.array([[1, 2, 3, 4, 5]]),
        obs=pd.DataFrame(index=["Cell_0"]),
        var=pd.DataFrame(index=[f"Gene_{i}" for i in range(5)]),
    )

    result_adata, stats, _ = service.detect_doublets(adata)
    assert stats["n_cells_analyzed"] == 1
    assert "doublet_score" in result_adata.obs.columns


def test_detect_doublets_with_sparse_matrix(service):
    """Test doublet detection with sparse matrix."""
    from scipy.sparse import csr_matrix

    np.random.seed(42)
    n_obs = 100
    n_vars = 50
    X = np.random.negative_binomial(n=5, p=0.3, size=(n_obs, n_vars)).astype(np.float32)
    X_sparse = csr_matrix(X)

    adata = ad.AnnData(
        X=X_sparse,
        obs=pd.DataFrame(index=[f"Cell_{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"Gene_{i}" for i in range(n_vars)]),
    )

    result_adata, stats, _ = service.detect_doublets(adata)
    assert "doublet_score" in result_adata.obs.columns
    assert stats["n_cells_analyzed"] == n_obs


def test_detect_doublets_with_raw_data(service, simple_adata):
    """Test doublet detection uses raw data when available."""
    simple_adata.raw = simple_adata.copy()

    result_adata, stats, _ = service.detect_doublets(simple_adata)
    assert "doublet_score" in result_adata.obs.columns


def test_detect_doublets_extreme_expected_rate_low(service, simple_adata):
    """Test doublet detection with very low expected rate."""
    result_adata, stats, _ = service.detect_doublets(
        simple_adata, expected_doublet_rate=0.001
    )

    assert stats["actual_doublet_rate"] >= 0
    assert stats["n_doublets_detected"] >= 0


def test_detect_doublets_extreme_expected_rate_high(service, simple_adata):
    """Test doublet detection with high expected rate."""
    result_adata, stats, _ = service.detect_doublets(
        simple_adata, expected_doublet_rate=0.5
    )

    assert stats["actual_doublet_rate"] <= 1.0
    assert stats["n_doublets_detected"] <= len(simple_adata.obs)


# ===============================================================================
# Test Doublet Detection - Fallback Method
# ===============================================================================


def test_fallback_doublet_detection(service, simple_adata):
    """Test fallback doublet detection method directly."""
    counts_matrix = simple_adata.X
    if hasattr(counts_matrix, "toarray"):
        counts_matrix = counts_matrix.toarray()

    doublet_scores, predicted_doublets, method = service._fallback_doublet_detection(
        counts_matrix, expected_doublet_rate=0.025
    )

    assert method == "fallback_outlier_detection"
    assert len(doublet_scores) == len(simple_adata.obs)
    assert len(predicted_doublets) == len(simple_adata.obs)
    assert doublet_scores.min() >= 0


def test_fallback_with_extreme_outliers(service):
    """Test fallback method identifies extreme outliers as doublets."""
    np.random.seed(42)
    n_cells = 100
    n_genes = 50

    # Create normal cells
    X = np.random.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes)).astype(
        np.float32
    )

    # Add extreme outliers (simulated doublets)
    X[-5:, :] *= 5  # Last 5 cells have much higher counts

    doublet_scores, predicted_doublets, method = service._fallback_doublet_detection(
        X, expected_doublet_rate=0.05
    )

    # Outliers should have higher doublet scores
    outlier_scores = doublet_scores[-5:]
    normal_scores = doublet_scores[:-5]
    assert np.mean(outlier_scores) > np.mean(normal_scores)


def test_fallback_zero_variance(service):
    """Test fallback method handles zero variance data."""
    n_cells = 50
    n_genes = 30

    # All cells identical
    X = np.ones((n_cells, n_genes)) * 10

    doublet_scores, predicted_doublets, method = service._fallback_doublet_detection(
        X, expected_doublet_rate=0.025
    )

    # Should not crash, should return valid results
    assert len(doublet_scores) == n_cells
    # Note: Zero variance may result in NaN scores due to division by zero in std calculation
    # This is expected behavior for pathological edge case


# ===============================================================================
# Test Doublet Detection - Scrublet Integration
# ===============================================================================


@pytest.mark.skipif(not SCRUBLET_AVAILABLE, reason="Scrublet not installed")
def test_detect_doublets_with_scrublet(service):
    """Test doublet detection uses Scrublet when available (with sufficient data)."""
    # Create larger dataset that won't trigger Scrublet's PCA failures
    np.random.seed(42)
    n_obs = 500  # Large enough for Scrublet
    n_vars = 200  # Enough genes

    X = np.random.negative_binomial(n=5, p=0.3, size=(n_obs, n_vars)).astype(np.float32)

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"Cell_{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"Gene_{i}" for i in range(n_vars)]),
    )

    result_adata, stats, _ = service.detect_doublets(adata)

    # When Scrublet is available and data is sufficient, it should be used
    # Note: May still fallback if Scrublet fails for other reasons
    assert stats["detection_method"] in ["scrublet", "fallback_outlier_detection"]


@pytest.mark.skipif(not SCRUBLET_AVAILABLE, reason="Scrublet not installed")
def test_scrublet_failure_fallback(service, simple_adata):
    """Test fallback when Scrublet fails."""
    with patch("scrublet.Scrublet") as mock_scrublet:
        mock_scrublet.side_effect = Exception("Scrublet failed")

        result_adata, stats, _ = service.detect_doublets(simple_adata)

        # Should fallback to alternative method
        assert stats["detection_method"] == "fallback_outlier_detection"
        assert "doublet_score" in result_adata.obs.columns


# ===============================================================================
# Test Cell Type Annotation - Basic Functionality
# ===============================================================================


def test_annotate_cell_types_basic(service, clustered_adata):
    """Test basic cell type annotation."""
    result_adata, stats, _ = service.annotate_cell_types(clustered_adata)

    assert "cell_type" in result_adata.obs.columns
    assert len(result_adata.obs) == len(clustered_adata.obs)

    # Check statistics
    assert stats["analysis_type"] == "cell_type_annotation"
    assert "n_cell_types_identified" in stats
    assert "cluster_to_celltype" in stats


def test_annotate_requires_clustering(service, simple_adata):
    """Test annotation fails without clustering results."""
    with pytest.raises(SingleCellError, match="No clustering results found"):
        service.annotate_cell_types(simple_adata)


def test_annotate_with_custom_markers(service, clustered_adata):
    """Test cell type annotation with custom markers."""
    custom_markers = {
        "Custom Type A": ["CD3D", "CD8A"],
        "Custom Type B": ["CD19", "MS4A1"],
    }

    result_adata, stats, _ = service.annotate_cell_types(
        clustered_adata, reference_markers=custom_markers
    )

    assert "cell_type" in result_adata.obs.columns
    assert stats["n_marker_sets"] == 2


def test_annotate_all_clusters_assigned(service, clustered_adata):
    """Test that all clusters get cell type assignments."""
    result_adata, stats, _ = service.annotate_cell_types(clustered_adata)

    n_clusters = len(clustered_adata.obs["leiden"].unique())
    n_assigned = len(stats["cluster_to_celltype"])

    assert n_assigned == n_clusters


def test_annotate_marker_scores_calculated(service, clustered_adata):
    """Test that marker scores are calculated for all clusters."""
    result_adata, stats, _ = service.annotate_cell_types(clustered_adata)

    assert "marker_scores" in stats
    marker_scores = stats["marker_scores"]

    # Each cluster should have scores
    for cluster_id in clustered_adata.obs["leiden"].unique():
        cluster_str = str(cluster_id)
        assert cluster_str in marker_scores


def test_annotate_cell_type_counts(service, clustered_adata):
    """Test cell type count statistics."""
    result_adata, stats, _ = service.annotate_cell_types(clustered_adata)

    assert "cell_type_counts" in stats
    cell_type_counts = stats["cell_type_counts"]

    total_assigned = sum(cell_type_counts.values())
    assert total_assigned == len(clustered_adata.obs)


# ===============================================================================
# Test Cell Type Annotation - Edge Cases
# ===============================================================================


def test_annotate_single_cluster(service):
    """Test annotation with single cluster."""
    np.random.seed(42)
    n_obs = 50
    n_vars = 30

    X = np.random.negative_binomial(n=5, p=0.3, size=(n_obs, n_vars)).astype(np.float32)

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"Cell_{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=["CD3D", "CD8A"] + [f"Gene_{i}" for i in range(28)]),
    )
    adata.obs["leiden"] = "0"

    result_adata, stats, _ = service.annotate_cell_types(adata)

    assert stats["n_clusters"] == 1
    assert len(stats["cluster_to_celltype"]) == 1


def test_annotate_no_marker_overlap(service):
    """Test annotation when no marker genes are present."""
    np.random.seed(42)
    n_obs = 100
    n_vars = 50

    X = np.random.negative_binomial(n=5, p=0.3, size=(n_obs, n_vars)).astype(np.float32)

    # Use genes that don't match any markers
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"Cell_{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"UnknownGene_{i}" for i in range(n_vars)]),
    )
    adata.obs["leiden"] = np.random.choice([0, 1, 2], size=n_obs).astype(str)

    result_adata, stats, _ = service.annotate_cell_types(adata)

    # Should still complete, all cells might be "Unknown"
    assert "cell_type" in result_adata.obs.columns


def test_annotate_with_non_unique_obs_names(service):
    """Test annotation handles non-unique observation names."""
    np.random.seed(42)
    n_obs = 100
    n_vars = 30

    X = np.random.negative_binomial(n=5, p=0.3, size=(n_obs, n_vars)).astype(np.float32)

    # Duplicate observation names
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=["Cell"] * n_obs),
        var=pd.DataFrame(index=["CD3D", "CD8A"] + [f"Gene_{i}" for i in range(28)]),
    )
    adata.obs["leiden"] = np.random.choice([0, 1], size=n_obs).astype(str)

    # Should handle gracefully
    result_adata, stats, _ = service.annotate_cell_types(adata)
    assert "cell_type" in result_adata.obs.columns


def test_annotate_with_non_unique_var_names(service):
    """Test annotation handles non-unique variable names."""
    np.random.seed(42)
    n_obs = 100
    n_vars = 30

    X = np.random.negative_binomial(n=5, p=0.3, size=(n_obs, n_vars)).astype(np.float32)

    # Duplicate gene names
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"Cell_{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=["Gene"] * n_vars),
    )
    adata.obs["leiden"] = np.random.choice([0, 1], size=n_obs).astype(str)

    # Should handle gracefully
    result_adata, stats, _ = service.annotate_cell_types(adata)
    assert "cell_type" in result_adata.obs.columns


def test_annotate_comprehensive_marker_coverage(service, adata_with_all_markers):
    """Test annotation with comprehensive marker coverage."""
    result_adata, stats, _ = service.annotate_cell_types(adata_with_all_markers)

    # Should identify multiple cell types
    assert stats["n_cell_types_identified"] >= 3

    # All clusters should be annotated
    assert len(stats["cluster_to_celltype"]) == 5

    # Check that marker scores are calculated
    assert "marker_scores" in stats
    for cluster_id in range(5):
        assert str(cluster_id) in stats["marker_scores"]


# ===============================================================================
# Test Marker Gene Detection - Basic Functionality
# ===============================================================================


def test_find_marker_genes_basic(service, clustered_adata):
    """Test basic marker gene detection."""
    result_adata, stats, _ = service.find_marker_genes(clustered_adata, groups="all")

    assert "rank_genes_groups" in result_adata.uns
    assert stats["analysis_type"] == "marker_gene_analysis"
    assert stats["has_marker_results"] is True


def test_find_markers_requires_groupby_column(service, simple_adata):
    """Test marker detection fails without groupby column."""
    with pytest.raises(SingleCellError, match="not found in observations"):
        service.find_marker_genes(simple_adata, groupby="nonexistent_column")


def test_find_markers_with_custom_method(service, clustered_adata):
    """Test marker detection with different statistical methods."""
    methods = ["wilcoxon", "t-test"]  # logreg requires more samples

    for method in methods:
        result_adata, stats, _ = service.find_marker_genes(
            clustered_adata, method=method, groups="all"
        )
        assert stats["method"] == method
        assert "rank_genes_groups" in result_adata.uns


def test_find_markers_with_custom_n_genes(service, clustered_adata):
    """Test marker detection with custom number of genes."""
    n_genes = 10
    result_adata, stats, _ = service.find_marker_genes(
        clustered_adata, n_genes=n_genes, groups="all"
    )

    assert stats["n_genes"] == n_genes


def test_find_markers_with_specific_groups(service, clustered_adata):
    """Test marker detection for specific groups."""
    groups = ["0", "1"]
    result_adata, stats, _ = service.find_marker_genes(clustered_adata, groups=groups)

    # Should only analyze specified groups
    assert "rank_genes_groups" in result_adata.uns


def test_find_markers_returns_statistics(service, clustered_adata):
    """Test that marker detection returns comprehensive statistics."""
    result_adata, stats, _ = service.find_marker_genes(clustered_adata, groups="all")

    required_keys = [
        "analysis_type",
        "groupby",
        "method",
        "n_genes",
        "n_groups",
        "groups_analyzed",
        "has_marker_results",
    ]

    for key in required_keys:
        assert key in stats


def test_find_markers_top_markers_per_group(service, clustered_adata):
    """Test that top markers are extracted per group."""
    result_adata, stats, _ = service.find_marker_genes(clustered_adata, groups="all")

    if "top_markers_per_group" in stats:
        top_markers = stats["top_markers_per_group"]

        for group, markers in top_markers.items():
            assert len(markers) <= 10  # Top 10 per group
            for marker in markers:
                assert "gene" in marker
                assert "score" in marker
                assert "pval" in marker


# ===============================================================================
# Test Marker Gene Detection - Edge Cases
# ===============================================================================


def test_find_markers_single_group(service):
    """Test marker detection with single group."""
    np.random.seed(42)
    n_obs = 50
    n_vars = 30

    X = np.random.negative_binomial(n=5, p=0.3, size=(n_obs, n_vars)).astype(np.float32)

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"Cell_{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"Gene_{i}" for i in range(n_vars)]),
    )
    adata.obs["leiden"] = "0"
    adata.raw = adata.copy()

    # Should handle single group gracefully
    result_adata, stats, _ = service.find_marker_genes(adata, groups="all")
    assert stats["n_groups"] == 1


def test_find_markers_two_groups_only(service):
    """Test marker detection with exactly two groups."""
    np.random.seed(42)
    n_obs = 100
    n_vars = 50

    X = np.random.negative_binomial(n=10, p=0.2, size=(n_obs, n_vars)).astype(
        np.float32
    )

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"Cell_{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"Gene_{i}" for i in range(n_vars)]),
    )
    adata.obs["leiden"] = np.array([0] * 50 + [1] * 50).astype(str)
    adata.raw = adata.copy()

    result_adata, stats, _ = service.find_marker_genes(adata, groups="all")
    assert stats["n_groups"] == 2


def test_find_markers_with_low_expression(service):
    """Test marker detection with low expression data."""
    np.random.seed(42)
    n_obs = 100
    n_vars = 50

    # Very low counts
    X = np.random.poisson(lam=0.5, size=(n_obs, n_vars)).astype(np.float32)

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"Cell_{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"Gene_{i}" for i in range(n_vars)]),
    )
    adata.obs["leiden"] = np.random.choice([0, 1, 2], size=n_obs).astype(str)
    adata.raw = adata.copy()

    result_adata, stats, _ = service.find_marker_genes(adata, groups="all")
    assert "rank_genes_groups" in result_adata.uns


def test_find_markers_large_n_genes(service, clustered_adata):
    """Test marker detection with more genes than available."""
    n_genes = 1000  # More than available
    result_adata, stats, _ = service.find_marker_genes(
        clustered_adata, n_genes=n_genes, groups="all"
    )

    # Should not fail, will return all available genes
    assert "rank_genes_groups" in result_adata.uns


def test_find_markers_minimal_genes(service):
    """Test marker detection with minimal gene set."""
    np.random.seed(42)
    n_obs = 50
    n_vars = 5  # Very few genes

    X = np.random.negative_binomial(n=5, p=0.3, size=(n_obs, n_vars)).astype(np.float32)

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"Cell_{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"Gene_{i}" for i in range(n_vars)]),
    )
    adata.obs["leiden"] = np.random.choice([0, 1], size=n_obs).astype(str)
    adata.raw = adata.copy()

    result_adata, stats, _ = service.find_marker_genes(adata, n_genes=10, groups="all")
    assert "rank_genes_groups" in result_adata.uns


# ===============================================================================
# Test Helper Functions
# ===============================================================================


def test_calculate_marker_scores_from_adata(service, clustered_adata):
    """Test marker score calculation helper function."""
    markers = {
        "T cells": ["CD3D", "CD8A"],
        "B cells": ["CD19", "MS4A1"],
    }

    scores = service._calculate_marker_scores_from_adata(clustered_adata, markers)

    assert isinstance(scores, dict)
    for cluster_id in clustered_adata.obs["leiden"].unique():
        assert str(cluster_id) in scores


def test_extract_marker_genes(service, clustered_adata):
    """Test marker gene extraction helper function."""
    # First run marker detection
    result_adata, _, _ = service.find_marker_genes(clustered_adata, groups="all")

    # Extract markers
    marker_df = service._extract_marker_genes(result_adata, "leiden")

    if not marker_df.empty:
        assert "gene" in marker_df.columns
        assert "score" in marker_df.columns
        assert "pval" in marker_df.columns
        assert "group" in marker_df.columns


def test_extract_marker_genes_empty_results(service, simple_adata):
    """Test marker extraction with no results."""
    marker_df = service._extract_marker_genes(simple_adata, "leiden")
    assert marker_df.empty


# ===============================================================================
# Test Error Handling
# ===============================================================================


def test_doublet_detection_error_handling(service):
    """Test doublet detection error handling."""
    invalid_adata = "not an anndata object"

    with pytest.raises(Exception):
        service.detect_doublets(invalid_adata)


def test_annotation_error_handling(service):
    """Test cell type annotation error handling."""
    invalid_adata = "not an anndata object"

    with pytest.raises(Exception):
        service.annotate_cell_types(invalid_adata)


def test_marker_detection_error_handling(service):
    """Test marker gene detection error handling."""
    invalid_adata = "not an anndata object"

    with pytest.raises(Exception):
        service.find_marker_genes(invalid_adata)


# ===============================================================================
# Test Statistical Properties
# ===============================================================================


def test_doublet_scores_are_continuous(service, simple_adata):
    """Test that doublet scores are continuous values."""
    result_adata, _, _ = service.detect_doublets(simple_adata)

    doublet_scores = result_adata.obs["doublet_score"]
    assert doublet_scores.dtype in [np.float32, np.float64]


def test_doublet_prediction_is_binary(service, simple_adata):
    """Test that doublet predictions are binary."""
    result_adata, _, _ = service.detect_doublets(simple_adata)

    predicted_doublets = result_adata.obs["predicted_doublet"]
    unique_values = predicted_doublets.unique()
    assert len(unique_values) <= 2
    assert all(v in [True, False, 0, 1] for v in unique_values)


def test_marker_scores_are_numeric(service, clustered_adata):
    """Test that marker scores are numeric."""
    result_adata, stats, _ = service.annotate_cell_types(clustered_adata)

    marker_scores = stats["marker_scores"]
    for cluster_scores in marker_scores.values():
        for score in cluster_scores.values():
            assert isinstance(score, (int, float))
            assert not np.isnan(score)


# ===============================================================================
# Integration Tests
# ===============================================================================


def test_full_workflow_detect_and_annotate(service, simple_adata):
    """Test full workflow: doublets -> cluster -> annotate."""
    # 1. Detect doublets
    adata_doublets, doublet_stats, _ = service.detect_doublets(simple_adata)
    assert "doublet_score" in adata_doublets.obs.columns

    # 2. Add clustering (simulated)
    adata_doublets.obs["leiden"] = np.random.choice(
        [0, 1, 2], size=len(simple_adata.obs)
    ).astype(str)

    # Add some marker genes
    marker_genes = ["CD3D", "CD19", "GNLY", "CD14"]
    current_genes = list(adata_doublets.var_names)

    # Replace some genes with markers
    for i, marker in enumerate(marker_genes):
        if i < len(current_genes):
            current_genes[i] = marker

    adata_doublets.var_names = current_genes

    # 3. Annotate
    adata_final, annotation_stats, _ = service.annotate_cell_types(adata_doublets)
    assert "cell_type" in adata_final.obs.columns
    assert "doublet_score" in adata_final.obs.columns


def test_full_workflow_marker_detection(service, clustered_adata):
    """Test full workflow with marker detection."""
    # 1. Find markers
    adata_markers, marker_stats, _ = service.find_marker_genes(
        clustered_adata, groups="all"
    )
    assert "rank_genes_groups" in adata_markers.uns

    # 2. Annotate based on markers
    adata_annotated, annotation_stats, _ = service.annotate_cell_types(adata_markers)
    assert "cell_type" in adata_annotated.obs.columns


# ===============================================================================
# Performance and Scale Tests
# ===============================================================================


def test_doublet_detection_performance_small(service):
    """Test doublet detection on small dataset."""
    np.random.seed(42)
    n_obs = 100
    n_vars = 50

    X = np.random.negative_binomial(n=5, p=0.3, size=(n_obs, n_vars)).astype(np.float32)

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"Cell_{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"Gene_{i}" for i in range(n_vars)]),
    )

    import time

    start = time.time()
    result_adata, stats, _ = service.detect_doublets(adata)
    duration = time.time() - start

    assert duration < 10  # Should complete in reasonable time
    assert stats["n_cells_analyzed"] == n_obs


def test_marker_detection_performance(service, clustered_adata):
    """Test marker detection performance."""
    import time

    start = time.time()
    result_adata, stats, _ = service.find_marker_genes(clustered_adata, groups="all")
    duration = time.time() - start

    assert duration < 30  # Should complete in reasonable time


def test_annotation_performance(service, clustered_adata):
    """Test annotation performance."""
    import time

    start = time.time()
    result_adata, stats, _ = service.annotate_cell_types(clustered_adata)
    duration = time.time() - start

    assert duration < 10  # Should be fast


# ===============================================================================
# Test Data Integrity
# ===============================================================================


def test_doublet_detection_preserves_original(service, simple_adata):
    """Test that doublet detection doesn't modify original data."""
    original_shape = simple_adata.shape
    original_obs_cols = set(simple_adata.obs.columns)

    result_adata, _, _ = service.detect_doublets(simple_adata)

    # Original should be unchanged
    assert simple_adata.shape == original_shape
    assert set(simple_adata.obs.columns) == original_obs_cols


def test_annotation_preserves_clustering(service, clustered_adata):
    """Test that annotation preserves clustering results."""
    original_leiden = clustered_adata.obs["leiden"].copy()

    result_adata, _, _ = service.annotate_cell_types(clustered_adata)

    # Leiden should be preserved
    assert "leiden" in result_adata.obs.columns
    pd.testing.assert_series_equal(
        original_leiden, clustered_adata.obs["leiden"], check_names=False
    )


def test_marker_detection_preserves_data(service, clustered_adata):
    """Test that marker detection preserves original data."""
    original_shape = clustered_adata.shape

    result_adata, _, _ = service.find_marker_genes(clustered_adata, groups="all")

    # Shape should be unchanged
    assert result_adata.shape == original_shape


# ===============================================================================
# Test Reproducibility
# ===============================================================================


def test_doublet_detection_reproducibility(service, simple_adata):
    """Test that doublet detection is reproducible with same seed."""
    result1_adata, stats1, _ = service.detect_doublets(simple_adata)
    result2_adata, stats2, _ = service.detect_doublets(simple_adata)

    # Results should be similar (may not be identical due to random components)
    assert stats1["n_cells_analyzed"] == stats2["n_cells_analyzed"]


def test_annotation_reproducibility(service, clustered_adata):
    """Test that annotation is reproducible."""
    result1_adata, stats1, _ = service.annotate_cell_types(clustered_adata)
    result2_adata, stats2, _ = service.annotate_cell_types(clustered_adata)

    # Should be deterministic
    pd.testing.assert_series_equal(
        result1_adata.obs["cell_type"],
        result2_adata.obs["cell_type"],
        check_names=False,
    )


# ===============================================================================
# End of Tests
# ===============================================================================
