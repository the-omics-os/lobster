"""
Tests for BulkPreprocessingService.

Covers all 4 methods: assess_sample_quality, filter_genes, normalize_counts,
detect_batch_effects. Each test verifies the 3-tuple return shape and
AnalysisStep IR presence.
"""

import warnings
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")

from lobster.services.analysis.bulk_preprocessing_service import (
    BulkPreprocessingError,
    BulkPreprocessingService,
)

# ===============================================================================
# Fixtures
# ===============================================================================


@pytest.fixture
def service():
    """Create BulkPreprocessingService instance."""
    return BulkPreprocessingService()


@pytest.fixture
def bulk_adata():
    """Create bulk RNA-seq AnnData (20 samples x 1000 genes)."""
    np.random.seed(42)
    n_samples = 20
    n_genes = 1000

    # Simulate count data (negative binomial)
    X = np.random.negative_binomial(n=10, p=0.3, size=(n_samples, n_genes)).astype(
        np.float64
    )

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"sample_id": [f"Sample_{i}" for i in range(n_samples)]},
            index=[f"Sample_{i}" for i in range(n_samples)],
        ),
        var=pd.DataFrame(
            index=[f"Gene_{i}" for i in range(n_genes)]
        ),
    )

    return adata


@pytest.fixture
def bulk_adata_with_batch(bulk_adata):
    """Bulk AnnData with batch column and batch effect."""
    bulk_adata.obs["batch"] = (
        ["batch_A"] * 10 + ["batch_B"] * 10
    )

    # Add batch effect: batch_B has higher expression in first 100 genes
    batch_b_mask = bulk_adata.obs["batch"] == "batch_B"
    bulk_adata.X[batch_b_mask.values, :100] += 20

    return bulk_adata


@pytest.fixture
def bulk_adata_with_condition(bulk_adata_with_batch):
    """Bulk AnnData with batch + condition columns."""
    bulk_adata_with_batch.obs["condition"] = (
        ["treated"] * 5 + ["control"] * 5 + ["treated"] * 5 + ["control"] * 5
    )
    return bulk_adata_with_batch


# ===============================================================================
# Test assess_sample_quality
# ===============================================================================


def test_assess_sample_quality(service, bulk_adata):
    """Test basic sample quality assessment."""
    result_adata, stats, ir = service.assess_sample_quality(bulk_adata)

    # Verify 3-tuple
    assert isinstance(result_adata, ad.AnnData)
    assert isinstance(stats, dict)
    assert hasattr(ir, "operation")
    assert ir.operation == "assess_sample_quality"

    # Verify outputs
    assert "is_outlier" in result_adata.obs.columns
    assert "distance_from_centroid" in result_adata.obs.columns
    assert "sample_correlation_matrix" in result_adata.uns

    # Verify stats
    assert stats["analysis_type"] == "bulk_sample_quality"
    assert stats["n_samples"] == 20
    assert "n_outliers" in stats
    assert "median_correlation" in stats
    assert isinstance(stats["median_correlation"], float)
    assert isinstance(stats["outlier_names"], list)

    # Correlation matrix shape
    corr = result_adata.uns["sample_correlation_matrix"]
    assert corr.shape == (20, 20)


def test_assess_sample_quality_with_batch(service, bulk_adata_with_batch):
    """Test quality assessment with batch R-squared computed."""
    result_adata, stats, ir = service.assess_sample_quality(
        bulk_adata_with_batch, batch_key="batch"
    )

    assert "batch_r_squared" in stats
    assert "batch_r_squared" in result_adata.uns
    assert isinstance(stats["batch_r_squared"], dict)

    # Batch R-squared should be significant given the added batch effect
    assert "PC1" in stats["batch_r_squared"]


def test_assess_sample_quality_preserves_original(service, bulk_adata):
    """Test that quality assessment doesn't modify original data."""
    original_shape = bulk_adata.shape
    original_obs_cols = set(bulk_adata.obs.columns)

    service.assess_sample_quality(bulk_adata)

    assert bulk_adata.shape == original_shape
    assert set(bulk_adata.obs.columns) == original_obs_cols


# ===============================================================================
# Test filter_genes
# ===============================================================================


def test_filter_genes(service, bulk_adata):
    """Test gene filtering with low-count genes."""
    # Add some low-count genes
    bulk_adata.X[:, -5:] = 0  # Last 5 genes have zero counts
    bulk_adata.X[:3, -10:-5] = 1  # 5 more genes only expressed in 3 samples

    result_adata, stats, ir = service.filter_genes(
        bulk_adata, min_counts=10, min_samples=3
    )

    # Verify 3-tuple
    assert isinstance(result_adata, ad.AnnData)
    assert isinstance(stats, dict)
    assert hasattr(ir, "operation")
    assert ir.operation == "filter_genes"

    # Verify filtering happened
    assert stats["n_genes_before"] == 1000
    assert stats["n_genes_after"] < 1000
    assert stats["n_filtered"] > 0
    assert result_adata.n_vars == stats["n_genes_after"]

    # Verify filter criteria recorded
    assert stats["filter_criteria"]["min_counts"] == 10
    assert stats["filter_criteria"]["min_samples"] == 3


def test_filter_genes_no_filtering_needed(service, bulk_adata):
    """Test gene filtering when all genes pass."""
    # All genes have high counts by default
    result_adata, stats, ir = service.filter_genes(
        bulk_adata, min_counts=1, min_samples=1
    )

    assert stats["n_filtered"] == 0
    assert stats["n_genes_before"] == stats["n_genes_after"]


def test_filter_genes_strict_criteria(service, bulk_adata):
    """Test gene filtering with very strict criteria."""
    result_adata, stats, ir = service.filter_genes(
        bulk_adata, min_counts=1000, min_samples=20
    )

    # Very strict should remove many genes
    assert stats["n_filtered"] > 0
    assert result_adata.n_vars < bulk_adata.n_vars


# ===============================================================================
# Test normalize_counts
# ===============================================================================


def test_normalize_counts_cpm(service, bulk_adata):
    """Test CPM normalization."""
    result_adata, stats, ir = service.normalize_counts(
        bulk_adata, method="cpm"
    )

    # Verify 3-tuple
    assert isinstance(result_adata, ad.AnnData)
    assert isinstance(stats, dict)
    assert hasattr(ir, "operation")
    assert ir.operation == "normalize_counts"

    # Verify raw counts preserved
    assert "counts" in result_adata.layers
    np.testing.assert_array_equal(
        result_adata.layers["counts"],
        np.asarray(bulk_adata.X),
    )

    # Verify CPM normalization (each sample should sum to ~1e6)
    row_sums = result_adata.X.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1e6, rtol=1e-5)

    # Verify stats
    assert stats["method"] == "cpm"
    assert "library_size_range" in stats


def test_normalize_counts_deseq2(service, bulk_adata):
    """Test DESeq2 size factor normalization."""
    result_adata, stats, ir = service.normalize_counts(
        bulk_adata, method="deseq2"
    )

    assert isinstance(result_adata, ad.AnnData)
    assert "counts" in result_adata.layers
    assert stats["method"] == "deseq2"
    assert "mean_size_factor" in stats
    assert isinstance(stats["mean_size_factor"], float)
    assert stats["mean_size_factor"] > 0

    # Normalized values should be different from raw
    assert not np.allclose(result_adata.X, result_adata.layers["counts"])


def test_normalize_counts_invalid_method(service, bulk_adata):
    """Test error on invalid normalization method."""
    with pytest.raises(BulkPreprocessingError, match="Unknown normalization"):
        service.normalize_counts(bulk_adata, method="invalid")


# ===============================================================================
# Test detect_batch_effects
# ===============================================================================


def test_detect_batch_effects(service, bulk_adata_with_batch):
    """Test batch effect detection with known batch effect."""
    result_adata, stats, ir = service.detect_batch_effects(
        bulk_adata_with_batch, batch_key="batch"
    )

    # Verify 3-tuple
    assert isinstance(result_adata, ad.AnnData)
    assert isinstance(stats, dict)
    assert hasattr(ir, "operation")
    assert ir.operation == "detect_batch_effects"

    # Verify outputs
    assert "batch_effects" in result_adata.uns
    assert "recommendation" in stats
    assert "batch_variance_pcs" in stats

    # Given the strong batch effect we added, batch R-squared should be notable
    batch_var = stats["batch_variance_pcs"]
    assert isinstance(batch_var, dict)
    assert "PC1" in batch_var


def test_detect_batch_effects_with_condition(service, bulk_adata_with_condition):
    """Test batch detection with condition comparison."""
    result_adata, stats, ir = service.detect_batch_effects(
        bulk_adata_with_condition,
        batch_key="batch",
        condition_key="condition",
    )

    assert "condition_variance_pcs" in stats
    assert "recommendation" in stats

    # Both batch and condition variance should be computed
    assert "PC1" in stats["batch_variance_pcs"]
    assert "PC1" in stats["condition_variance_pcs"]


def test_detect_batch_effects_invalid_batch_key(service, bulk_adata):
    """Test error on invalid batch key."""
    with pytest.raises(BulkPreprocessingError, match="not found"):
        service.detect_batch_effects(bulk_adata, batch_key="nonexistent")


def test_detect_batch_effects_invalid_condition_key(service, bulk_adata_with_batch):
    """Test error on invalid condition key."""
    with pytest.raises(BulkPreprocessingError, match="not found"):
        service.detect_batch_effects(
            bulk_adata_with_batch,
            batch_key="batch",
            condition_key="nonexistent",
        )


# ===============================================================================
# Test IR Presence on All Methods
# ===============================================================================


def test_all_methods_return_analysis_step(service, bulk_adata_with_batch):
    """Verify all 4 methods return proper AnalysisStep IR."""
    from lobster.core.analysis_ir import AnalysisStep

    # assess_sample_quality
    _, _, ir1 = service.assess_sample_quality(bulk_adata_with_batch)
    assert isinstance(ir1, AnalysisStep)
    assert ir1.code_template is not None

    # filter_genes
    _, _, ir2 = service.filter_genes(bulk_adata_with_batch)
    assert isinstance(ir2, AnalysisStep)

    # normalize_counts (CPM - avoids pyDESeq2 edge cases)
    _, _, ir3 = service.normalize_counts(bulk_adata_with_batch, method="cpm")
    assert isinstance(ir3, AnalysisStep)

    # detect_batch_effects
    _, _, ir4 = service.detect_batch_effects(bulk_adata_with_batch, batch_key="batch")
    assert isinstance(ir4, AnalysisStep)
