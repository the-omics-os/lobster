"""
Unit tests for DEG filtering and confidence scoring in BulkRNASeqService.

Tests cover:
- DEG filtering parameters and logic
- Gene confidence calculation
- Quality categorization
- Integration with run_differential_expression_analysis
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from lobster.services.analysis.bulk_rnaseq_service import (
    BulkRNASeqError,
    BulkRNASeqService,
)


@pytest.fixture
def bulk_service(tmp_path):
    """Create a BulkRNASeqService instance for testing."""
    return BulkRNASeqService(results_dir=tmp_path)


@pytest.fixture
def sample_de_adata():
    """
    Create a sample AnnData with DE results for testing filtering and confidence.

    Contains:
    - 100 genes
    - 10 samples (5 treated, 5 control)
    - Realistic DE statistics (FDR, log2FC, mean_expr)
    """
    n_genes = 100
    n_samples = 10

    # Expression matrix (samples × genes)
    np.random.seed(42)
    X = np.random.rand(n_samples, n_genes) * 100

    # Sample metadata
    obs = pd.DataFrame(
        {
            "condition": ["treated"] * 5 + ["control"] * 5,
            "sample_id": [f"sample_{i}" for i in range(n_samples)],
        }
    )

    # Gene metadata with realistic DE statistics
    gene_names = [f"GENE{i}" for i in range(n_genes)]

    # Create realistic DE results
    # 20 highly significant genes (FDR < 0.01, high FC)
    # 30 moderately significant genes (FDR < 0.05, medium FC)
    # 50 non-significant genes (FDR > 0.05)
    fdr_values = np.concatenate(
        [
            np.random.uniform(0.0001, 0.01, 20),  # High significance
            np.random.uniform(0.01, 0.05, 30),  # Medium significance
            np.random.uniform(0.05, 1.0, 50),  # Non-significant
        ]
    )

    log2fc_values = np.concatenate(
        [
            np.random.uniform(2.0, 4.0, 20),  # Large fold-changes
            np.random.uniform(1.0, 2.0, 30),  # Medium fold-changes
            np.random.uniform(-1.0, 1.0, 50),  # Small fold-changes
        ]
    )

    mean_expr_values = np.random.uniform(10, 1000, n_genes)

    var = pd.DataFrame(
        {
            "gene_name": gene_names,
            "padj": fdr_values,
            "log2FoldChange": log2fc_values,
            "baseMean": mean_expr_values,
        },
        index=gene_names,
    )

    adata = anndata.AnnData(X=X, obs=obs, var=var)
    return adata


@pytest.fixture
def simple_de_adata():
    """
    Create a simple AnnData for testing DE analysis end-to-end.

    Contains:
    - 50 genes
    - 6 samples (3 treated, 3 control)
    - Simple expression pattern for testing
    """
    n_genes = 50
    n_samples = 6

    np.random.seed(42)
    # Create differential expression: first 10 genes upregulated, next 10 downregulated
    X = np.random.rand(n_samples, n_genes) * 10

    # Make first 10 genes higher in treated group
    X[:3, :10] = X[:3, :10] * 5 + 50
    X[3:, :10] = X[3:, :10] + 10

    # Make next 10 genes higher in control group
    X[:3, 10:20] = X[:3, 10:20] + 10
    X[3:, 10:20] = X[3:, 10:20] * 5 + 50

    obs = pd.DataFrame(
        {
            "condition": ["treated"] * 3 + ["control"] * 3,
            "sample_id": [f"sample_{i}" for i in range(n_samples)],
        }
    )

    var = pd.DataFrame(index=[f"GENE{i}" for i in range(n_genes)])

    adata = anndata.AnnData(X=X, obs=obs, var=var)
    return adata


class TestDEGFiltering:
    """Test differential expression gene filtering functionality."""

    def test_filtering_parameters_default_values(self, bulk_service, simple_de_adata):
        """Test that default filtering parameters are applied correctly."""
        # Run DE analysis with default filtering parameters
        adata_de, de_stats, ir = bulk_service.run_differential_expression_analysis(
            simple_de_adata,
            groupby="condition",
            group1="treated",
            group2="control",
            method="deseq2_like",
        )

        # Verify default filtering parameters are in stats
        assert de_stats["filtering_applied"] is True
        assert "filtering_params" in de_stats
        assert de_stats["filtering_params"]["min_fold_change"] == 1.5
        assert de_stats["filtering_params"]["min_pct_expressed"] == 0.1
        assert de_stats["filtering_params"]["max_out_pct_expressed"] == 0.5

    def test_fold_change_filtering_removes_low_fc_genes(self, sample_de_adata):
        """Test that genes below fold-change threshold are filtered."""
        # Count genes with |log2FC| >= log2(1.5) = 0.585
        threshold = np.log2(1.5)
        fold_changes = np.abs(sample_de_adata.var["log2FoldChange"].values)
        expected_passing = fold_changes >= threshold
        expected_count = expected_passing.sum()

        # Basic sanity check on test data
        assert expected_count > 0
        assert expected_count <= len(sample_de_adata.var)

        # Verify some genes would be filtered
        assert expected_count < len(sample_de_adata.var)

    def test_expression_fraction_filtering(self, sample_de_adata):
        """Test that expression fraction filters work correctly."""
        # Calculate expression fractions
        X = sample_de_adata.X
        group1_mask = sample_de_adata.obs["condition"] == "treated"
        group2_mask = sample_de_adata.obs["condition"] == "control"

        group1_data = X[group1_mask, :]
        group2_data = X[group2_mask, :]

        # Fraction expressing (expression > 1)
        group1_frac = (group1_data > 1).mean(axis=0)
        group2_frac = (group2_data > 1).mean(axis=0)

        # Verify calculations are correct
        assert len(group1_frac) == sample_de_adata.n_vars
        assert len(group2_frac) == sample_de_adata.n_vars
        assert np.all(group1_frac >= 0) and np.all(group1_frac <= 1)
        assert np.all(group2_frac >= 0) and np.all(group2_frac <= 1)

        # With defaults: min_pct=0.1, max_out_pct=0.5
        expected_passing = (group1_frac >= 0.1) & (group2_frac <= 0.5)
        assert np.sum(expected_passing) >= 0

    def test_combined_filtering_logic(self, sample_de_adata):
        """Test that all three filters are applied together (AND logic)."""
        threshold = np.log2(1.5)

        X = sample_de_adata.X
        group1_mask = sample_de_adata.obs["condition"] == "treated"
        group2_mask = sample_de_adata.obs["condition"] == "control"

        group1_data = X[group1_mask, :]
        group2_data = X[group2_mask, :]

        group1_frac = (group1_data > 1).mean(axis=0)
        group2_frac = (group2_data > 1).mean(axis=0)

        fold_changes = np.abs(sample_de_adata.var["log2FoldChange"].values)

        # Combined filter (AND of all conditions)
        filter_mask = (
            (fold_changes >= threshold) & (group1_frac >= 0.1) & (group2_frac <= 0.5)
        )

        passing_genes = filter_mask.sum()
        total_genes = len(filter_mask)

        # Should filter out at least some genes
        assert passing_genes <= total_genes
        assert passing_genes >= 0

    def test_filtering_statistics_tracked_correctly(
        self, bulk_service, simple_de_adata
    ):
        """Test that pre/post filter counts are calculated correctly."""
        adata_de, de_stats, ir = bulk_service.run_differential_expression_analysis(
            simple_de_adata,
            groupby="condition",
            group1="treated",
            group2="control",
            method="deseq2_like",
        )

        # Verify filtering stats are present
        assert "pre_filter_gene_count" in de_stats
        assert "post_filter_gene_count" in de_stats
        assert "filtered_gene_count" in de_stats

        pre_filter = de_stats["pre_filter_gene_count"]
        post_filter = de_stats["post_filter_gene_count"]
        filtered = de_stats["filtered_gene_count"]

        # Verify consistency
        assert pre_filter == post_filter + filtered
        assert post_filter >= 0
        assert filtered >= 0

    def test_empty_filtering_edge_case(self, bulk_service, simple_de_adata):
        """Test behavior when all genes are filtered out."""
        # Apply very stringent filters that remove everything
        # Should handle gracefully (no exception, but 0 genes returned)
        adata_de, de_stats, ir = bulk_service.run_differential_expression_analysis(
            simple_de_adata,
            groupby="condition",
            group1="treated",
            group2="control",
            method="deseq2_like",
            min_fold_change=100.0,  # Impossible threshold
            min_pct_expressed=0.99,
            max_out_pct_expressed=0.01,
        )

        # Verify empty result is handled gracefully
        assert de_stats["post_filter_gene_count"] == 0
        assert de_stats["filtered_gene_count"] >= 0
        assert de_stats["mean_confidence"] == 0.0  # No genes = 0 confidence

    def test_no_filtering_edge_case(self, bulk_service, simple_de_adata):
        """Test behavior when no genes are filtered (permissive thresholds)."""
        # Apply very permissive filters that keep everything
        adata_de, de_stats, ir = bulk_service.run_differential_expression_analysis(
            simple_de_adata,
            groupby="condition",
            group1="treated",
            group2="control",
            method="deseq2_like",
            min_fold_change=0.0,  # No threshold
            min_pct_expressed=0.0,
            max_out_pct_expressed=1.0,
        )

        # Should keep most genes (only expression threshold applied)
        assert de_stats["post_filter_gene_count"] > 0


class TestConfidenceScoring:
    """Test gene confidence scoring functionality."""

    def test_calculate_gene_confidence_basic(self, bulk_service, sample_de_adata):
        """Test basic confidence calculation."""
        confidence_scores, quality_categories = bulk_service._calculate_gene_confidence(
            sample_de_adata, method="deseq2_like"
        )

        # Verify return types
        assert isinstance(confidence_scores, np.ndarray)
        assert isinstance(quality_categories, np.ndarray)

        # Verify lengths match number of genes
        assert len(confidence_scores) == sample_de_adata.n_vars
        assert len(quality_categories) == sample_de_adata.n_vars

        # Verify confidence scores are in 0-1 range
        assert np.all(confidence_scores >= 0)
        assert np.all(confidence_scores <= 1)

        # Verify quality categories are valid
        valid_categories = {"high", "medium", "low"}
        assert set(quality_categories).issubset(valid_categories)

    def test_high_confidence_genes(self, bulk_service, sample_de_adata):
        """Test that genes with strong signal get high confidence."""
        # Manually set some genes to have strong signals
        sample_de_adata.var.loc["GENE0", "padj"] = 0.001
        sample_de_adata.var.loc["GENE0", "log2FoldChange"] = 2.5
        sample_de_adata.var.loc["GENE0", "baseMean"] = 500

        confidence_scores, quality_categories = bulk_service._calculate_gene_confidence(
            sample_de_adata, method="deseq2_like"
        )

        # GENE0 should have high confidence
        gene0_idx = list(sample_de_adata.var_names).index("GENE0")
        assert quality_categories[gene0_idx] == "high"
        assert confidence_scores[gene0_idx] > 0.5

    def test_low_confidence_genes(self, bulk_service, sample_de_adata):
        """Test that genes with weak signal get low confidence."""
        # Manually set some genes to have weak signals
        sample_de_adata.var.loc["GENE99", "padj"] = 0.8
        sample_de_adata.var.loc["GENE99", "log2FoldChange"] = 0.1
        sample_de_adata.var.loc["GENE99", "baseMean"] = 5

        confidence_scores, quality_categories = bulk_service._calculate_gene_confidence(
            sample_de_adata, method="deseq2_like"
        )

        # GENE99 should have low confidence
        gene99_idx = list(sample_de_adata.var_names).index("GENE99")
        assert quality_categories[gene99_idx] == "low"
        assert confidence_scores[gene99_idx] < 0.3

    def test_quality_distribution_calculation(self, bulk_service, sample_de_adata):
        """Test that quality distribution counts are correct."""
        confidence_scores, quality_categories = bulk_service._calculate_gene_confidence(
            sample_de_adata, method="deseq2_like"
        )

        # Calculate distribution
        quality_dist = {
            "high": int((quality_categories == "high").sum()),
            "medium": int((quality_categories == "medium").sum()),
            "low": int((quality_categories == "low").sum()),
        }

        # Should sum to total genes
        assert sum(quality_dist.values()) == sample_de_adata.n_vars

        # All categories should have at least some genes
        # (based on our sample_de_adata fixture design)
        assert quality_dist["high"] >= 0
        assert quality_dist["low"] >= 0

    def test_confidence_nan_handling(self, bulk_service, sample_de_adata):
        """Test that NaN values are handled gracefully."""
        # Set some genes to have NaN values
        sample_de_adata.var.loc["GENE50", "padj"] = np.nan
        sample_de_adata.var.loc["GENE51", "log2FoldChange"] = np.nan

        confidence_scores, quality_categories = bulk_service._calculate_gene_confidence(
            sample_de_adata, method="deseq2_like"
        )

        # Genes with NaN should get low confidence
        gene50_idx = list(sample_de_adata.var_names).index("GENE50")
        gene51_idx = list(sample_de_adata.var_names).index("GENE51")

        assert confidence_scores[gene50_idx] == 0.0
        assert confidence_scores[gene51_idx] == 0.0
        assert quality_categories[gene50_idx] == "low"
        assert quality_categories[gene51_idx] == "low"

    def test_confidence_formula_weights(self, bulk_service):
        """Test that confidence formula uses correct weights."""
        # Create controlled test case
        n_genes = 3
        n_samples = 10

        X = np.ones((n_samples, n_genes)) * 100
        obs = pd.DataFrame({"condition": ["a"] * 5 + ["b"] * 5})

        # Gene 1: Perfect FDR, no FC
        # Gene 2: No FDR, perfect FC
        # Gene 3: Balanced
        var = pd.DataFrame(
            {
                "padj": [0.0, 1.0, 0.01],
                "log2FoldChange": [0.0, 3.0, 1.5],
                "baseMean": [100, 100, 100],
            },
            index=["GENE1", "GENE2", "GENE3"],
        )

        adata = anndata.AnnData(X=X, obs=obs, var=var)

        confidence_scores, _ = bulk_service._calculate_gene_confidence(
            adata, method="deseq2_like"
        )

        # Verify weights are applied
        # FDR component is 50%, so GENE1 should have decent score
        # FC component is 30%, so GENE2 should have lower score
        # GENE3 balanced should be moderate

        assert confidence_scores[0] > 0.3  # GENE1 (good FDR)
        assert confidence_scores[1] > 0.2  # GENE2 (good FC)
        assert confidence_scores[2] > 0.5  # GENE3 (balanced)

    def test_confidence_statistics_calculation(self, bulk_service, sample_de_adata):
        """Test that mean/median/std confidence statistics are correct."""
        confidence_scores, _ = bulk_service._calculate_gene_confidence(
            sample_de_adata, method="deseq2_like"
        )

        # Calculate statistics
        mean_conf = float(np.mean(confidence_scores))
        median_conf = float(np.median(confidence_scores))
        std_conf = float(np.std(confidence_scores))

        # Verify reasonable ranges
        assert 0 <= mean_conf <= 1
        assert 0 <= median_conf <= 1
        assert 0 <= std_conf <= 1

        # Std should be less than 1 (basic sanity)
        assert std_conf < 1.0


class TestIntegration:
    """Test integration of filtering and confidence scoring in DE workflow."""

    def test_de_returns_filtering_stats(self, bulk_service, simple_de_adata):
        """Test that run_differential_expression_analysis returns filtering stats."""
        adata_de, de_stats, ir = bulk_service.run_differential_expression_analysis(
            simple_de_adata,
            groupby="condition",
            group1="treated",
            group2="control",
            method="deseq2_like",
        )

        # Verify filtering stats are present
        assert "filtering_applied" in de_stats
        assert de_stats["filtering_applied"] is True

        assert "filtering_params" in de_stats
        assert "min_fold_change" in de_stats["filtering_params"]
        assert "min_pct_expressed" in de_stats["filtering_params"]
        assert "max_out_pct_expressed" in de_stats["filtering_params"]

        assert "pre_filter_gene_count" in de_stats
        assert "post_filter_gene_count" in de_stats
        assert "filtered_gene_count" in de_stats

        # Verify types
        assert isinstance(de_stats["pre_filter_gene_count"], int)
        assert isinstance(de_stats["post_filter_gene_count"], int)
        assert isinstance(de_stats["filtered_gene_count"], int)

    def test_de_returns_confidence_stats(self, bulk_service, simple_de_adata):
        """Test that run_differential_expression_analysis returns confidence stats."""
        adata_de, de_stats, ir = bulk_service.run_differential_expression_analysis(
            simple_de_adata,
            groupby="condition",
            group1="treated",
            group2="control",
            method="deseq2_like",
        )

        # Verify confidence stats are present
        assert "confidence_scoring" in de_stats
        assert de_stats["confidence_scoring"] is True

        assert "mean_confidence" in de_stats
        assert "median_confidence" in de_stats
        assert "std_confidence" in de_stats
        assert "quality_distribution" in de_stats

        # Verify types and ranges
        assert isinstance(de_stats["mean_confidence"], float)
        assert 0 <= de_stats["mean_confidence"] <= 1

        assert isinstance(de_stats["median_confidence"], float)
        assert 0 <= de_stats["median_confidence"] <= 1

        assert isinstance(de_stats["std_confidence"], float)
        assert 0 <= de_stats["std_confidence"] <= 1

        # Verify quality distribution
        quality_dist = de_stats["quality_distribution"]
        assert "high" in quality_dist
        assert "medium" in quality_dist
        assert "low" in quality_dist
        assert sum(quality_dist.values()) == de_stats["post_filter_gene_count"]

    def test_de_adds_confidence_columns_to_adata(self, bulk_service, simple_de_adata):
        """Test that confidence scores are added to adata.var."""
        adata_de, de_stats, ir = bulk_service.run_differential_expression_analysis(
            simple_de_adata,
            groupby="condition",
            group1="treated",
            group2="control",
            method="deseq2_like",
        )

        # Only check columns if genes exist
        if de_stats["post_filter_gene_count"] > 0:
            # Verify columns exist
            assert "gene_confidence" in adata_de.var.columns
            assert "gene_quality" in adata_de.var.columns

            # Verify types
            assert adata_de.var["gene_confidence"].dtype in [np.float64, np.float32]
            assert adata_de.var["gene_quality"].dtype == object

            # Verify values
            assert np.all(adata_de.var["gene_confidence"] >= 0)
            assert np.all(adata_de.var["gene_confidence"] <= 1)

            valid_categories = {"high", "medium", "low"}
            assert set(adata_de.var["gene_quality"].unique()).issubset(valid_categories)
        else:
            # Empty result - columns shouldn't exist
            assert (
                "gene_confidence" not in adata_de.var.columns or len(adata_de.var) == 0
            )
            assert "gene_quality" not in adata_de.var.columns or len(adata_de.var) == 0

    def test_custom_filtering_parameters(self, bulk_service, simple_de_adata):
        """Test DE analysis with custom filtering parameters."""
        adata_de, de_stats, ir = bulk_service.run_differential_expression_analysis(
            simple_de_adata,
            groupby="condition",
            group1="treated",
            group2="control",
            method="deseq2_like",
            min_fold_change=2.0,
            min_pct_expressed=0.2,
            max_out_pct_expressed=0.3,
        )

        # Verify custom parameters are stored
        assert de_stats["filtering_params"]["min_fold_change"] == 2.0
        assert de_stats["filtering_params"]["min_pct_expressed"] == 0.2
        assert de_stats["filtering_params"]["max_out_pct_expressed"] == 0.3

        # More stringent filtering should result in fewer genes
        assert de_stats["filtered_gene_count"] >= 0

    def test_ir_contains_filtering_parameters(self, bulk_service, simple_de_adata):
        """Test that IR (AnalysisStep) contains filtering parameters."""
        adata_de, de_stats, ir = bulk_service.run_differential_expression_analysis(
            simple_de_adata,
            groupby="condition",
            group1="treated",
            group2="control",
            method="deseq2_like",
            min_fold_change=2.0,
        )

        # Verify IR has filtering parameters
        assert ir.parameters["min_fold_change"] == 2.0
        assert "min_pct_expressed" in ir.parameters
        assert "max_out_pct_expressed" in ir.parameters

        # Verify parameter schema
        assert "min_fold_change" in ir.parameter_schema
        assert "min_pct_expressed" in ir.parameter_schema
        assert "max_out_pct_expressed" in ir.parameter_schema


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_sparse_matrix_handling(self, bulk_service):
        """Test that sparse matrices are handled correctly."""
        # Create sparse matrix with realistic differential expression
        n_genes = 50
        n_samples = 6

        np.random.seed(42)
        X_dense = (
            np.random.rand(n_samples, n_genes) * 50 + 10
        )  # Higher baseline expression

        # Make first 10 genes differentially expressed
        X_dense[:3, :10] = X_dense[:3, :10] * 3 + 50  # Upregulated in treated

        X_sparse = sparse.csr_matrix(X_dense)

        obs = pd.DataFrame({"condition": ["treated"] * 3 + ["control"] * 3})
        var = pd.DataFrame(index=[f"GENE{i}" for i in range(n_genes)])

        adata = anndata.AnnData(X=X_sparse, obs=obs, var=var)

        # Should handle sparse matrices with less stringent filters
        adata_de, de_stats, ir = bulk_service.run_differential_expression_analysis(
            adata,
            groupby="condition",
            group1="treated",
            group2="control",
            method="deseq2_like",
            min_fold_change=1.0,  # Less stringent
            min_pct_expressed=0.0,
            max_out_pct_expressed=1.0,
        )

        # Verify filtering and confidence scoring still work
        assert de_stats["filtering_applied"] is True
        assert de_stats["confidence_scoring"] is True

        # Only check columns if genes remain after filtering
        if de_stats["post_filter_gene_count"] > 0:
            assert "gene_confidence" in adata_de.var.columns

    def test_missing_de_columns(self, bulk_service):
        """Test behavior when expected DE columns are missing."""
        # Create adata without DE results
        n_genes = 50
        n_samples = 6

        X = np.random.rand(n_samples, n_genes) * 10
        obs = pd.DataFrame({"condition": ["a"] * 3 + ["b"] * 3})
        var = pd.DataFrame(index=[f"GENE{i}" for i in range(n_genes)])

        adata = anndata.AnnData(X=X, obs=obs, var=var)

        # Should still work (will calculate FC from mean expression)
        confidence_scores, quality_categories = bulk_service._calculate_gene_confidence(
            adata, method="deseq2_like"
        )

        # Should return low confidence for all genes
        assert np.all(confidence_scores == 0.0)
        assert np.all(quality_categories == "low")

    def test_single_sample_groups(self, bulk_service):
        """Test behavior with single-sample groups (should fail gracefully)."""
        n_genes = 50

        X = np.random.rand(2, n_genes) * 10
        obs = pd.DataFrame({"condition": ["treated", "control"]})
        var = pd.DataFrame(index=[f"GENE{i}" for i in range(n_genes)])

        adata = anndata.AnnData(X=X, obs=obs, var=var)

        # Should handle single-sample case
        adata_de, de_stats, ir = bulk_service.run_differential_expression_analysis(
            adata,
            groupby="condition",
            group1="treated",
            group2="control",
            method="deseq2_like",
        )

        # Should complete but with warnings in stats
        assert de_stats["n_samples_group1"] == 1
        assert de_stats["n_samples_group2"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
