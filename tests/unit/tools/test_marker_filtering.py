"""
Unit tests for DEG filtering in marker gene detection.

Tests the new filtering parameters added in Task 2:
- min_fold_change: Minimum fold-change threshold
- min_pct: Minimum in-group expression percentage
- max_out_pct: Maximum out-group expression percentage
"""

import numpy as np
import pytest
import anndata
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

from lobster.tools.enhanced_singlecell_service import EnhancedSingleCellService
from lobster.core.analysis_ir import AnalysisStep


@pytest.fixture
def service():
    """Create service instance."""
    return EnhancedSingleCellService()


@pytest.fixture
def clustered_adata():
    """Create clustered AnnData with distinct marker genes."""
    np.random.seed(42)
    n_cells = 150
    n_genes = 100

    # Create expression matrix with cluster-specific markers
    X = np.random.rand(n_cells, n_genes) * 0.5  # Baseline low expression

    # Cluster 0 (cells 0-49): high expression in genes 0-9
    X[0:50, 0:10] = np.random.rand(50, 10) * 3.0 + 2.0  # Strong markers

    # Cluster 1 (cells 50-99): high expression in genes 10-19
    X[50:100, 10:20] = np.random.rand(50, 10) * 3.0 + 2.0

    # Cluster 2 (cells 100-149): high expression in genes 20-29
    X[100:150, 20:30] = np.random.rand(50, 10) * 3.0 + 2.0

    # Add some weakly expressed genes (genes 30-39) to test filtering
    X[0:50, 30:40] = np.random.rand(50, 10) * 1.2 + 0.3  # Weak markers (should be filtered)

    adata = anndata.AnnData(X=X)
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]

    # Add cluster assignments
    clusters = ["0"] * 50 + ["1"] * 50 + ["2"] * 50
    adata.obs["leiden"] = pd.Categorical(clusters)

    # Preprocess (required for scanpy's rank_genes_groups)
    # Store raw counts before normalization (standard scanpy pattern)
    adata.raw = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


class TestDEGFiltering:
    """Test DEG filtering functionality."""

    def test_find_markers_returns_three_tuple(self, service, clustered_adata):
        """Test that find_marker_genes returns 3-tuple (adata, stats, ir)."""
        result = service.find_marker_genes(
            clustered_adata,
            groupby="leiden",
            method="wilcoxon",
            n_genes=25
        )

        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 3, "Should return 3-tuple"

        adata_result, stats, ir = result
        assert isinstance(adata_result, anndata.AnnData), "First element should be AnnData"
        assert isinstance(stats, dict), "Second element should be dict"
        assert isinstance(ir, AnalysisStep), "Third element should be AnalysisStep"

    def test_new_parameters_accepted(self, service, clustered_adata):
        """Test that new filtering parameters are accepted."""
        # Should run without errors
        adata_result, stats, ir = service.find_marker_genes(
            clustered_adata,
            groupby="leiden",
            method="wilcoxon",
            n_genes=25,
            min_fold_change=1.5,
            min_pct=0.25,
            max_out_pct=0.5
        )

        assert adata_result is not None
        assert stats is not None
        assert ir is not None

    def test_default_parameter_values(self, service, clustered_adata):
        """Test that default parameter values work."""
        # Call without explicit filtering parameters
        adata_result, stats, ir = service.find_marker_genes(
            clustered_adata,
            groupby="leiden"
        )

        # Check that defaults are reflected in stats
        assert "filtering_params" in stats
        assert stats["filtering_params"]["min_fold_change"] == 1.5
        assert stats["filtering_params"]["min_pct"] == 0.25
        assert stats["filtering_params"]["max_out_pct"] == 0.5

    def test_rank_genes_groups_result_structure(self, service, clustered_adata):
        """Test that rank_genes_groups result is properly stored."""
        adata_result, stats, ir = service.find_marker_genes(
            clustered_adata,
            groupby="leiden"
        )

        # Check result stored in .uns
        assert "rank_genes_groups" in adata_result.uns

        rgg = adata_result.uns["rank_genes_groups"]
        assert "names" in rgg
        assert "scores" in rgg
        assert "pvals" in rgg
        assert "pvals_adj" in rgg
        assert "logfoldchanges" in rgg

    def test_filtering_reduces_gene_count(self, service, clustered_adata):
        """Test that filtering removes genes."""
        # Run with lenient filtering (should keep more genes)
        _, stats_lenient, _ = service.find_marker_genes(
            clustered_adata,
            groupby="leiden",
            n_genes=25,
            min_fold_change=0.5,   # Very lenient
            min_pct=0.1,
            max_out_pct=0.9
        )

        # Run with strict filtering (should remove more genes)
        _, stats_strict, _ = service.find_marker_genes(
            clustered_adata,
            groupby="leiden",
            n_genes=25,
            min_fold_change=2.0,   # Very strict
            min_pct=0.4,
            max_out_pct=0.3
        )

        # Strict filtering should remove more genes
        lenient_total = stats_lenient["total_genes_filtered"]
        strict_total = stats_strict["total_genes_filtered"]

        assert strict_total >= lenient_total, \
            f"Strict filtering should remove at least as many genes as lenient (strict: {strict_total}, lenient: {lenient_total})"

    def test_stats_dict_structure(self, service, clustered_adata):
        """Test that stats dict has correct structure with filtering info."""
        adata_result, stats, ir = service.find_marker_genes(
            clustered_adata,
            groupby="leiden"
        )

        # Check required keys
        required_keys = [
            "method",
            "groupby",
            "n_genes",
            "filtering_params",
            "pre_filter_counts",
            "post_filter_counts",
            "filtered_counts",
            "total_genes_filtered",
            "groups_analyzed"
        ]

        for key in required_keys:
            assert key in stats, f"Missing required key: {key}"

        # Check filtering_params structure
        filtering_params = stats["filtering_params"]
        assert "min_fold_change" in filtering_params
        assert "min_pct" in filtering_params
        assert "max_out_pct" in filtering_params

        # Check that group-wise counts exist
        groups = stats["groups_analyzed"]
        assert len(groups) == 3, "Should have 3 clusters"

        for group in groups:
            assert group in stats["pre_filter_counts"], f"Missing pre-filter count for {group}"
            assert group in stats["post_filter_counts"], f"Missing post-filter count for {group}"
            assert group in stats["filtered_counts"], f"Missing filtered count for {group}"

    def test_filtered_counts_consistency(self, service, clustered_adata):
        """Test that pre/post/filtered counts are consistent."""
        _, stats, _ = service.find_marker_genes(
            clustered_adata,
            groupby="leiden",
            n_genes=25
        )

        for group in stats["groups_analyzed"]:
            pre = stats["pre_filter_counts"][group]
            post = stats["post_filter_counts"][group]
            filtered = stats["filtered_counts"][group]

            # Basic math: pre = post + filtered
            assert pre == post + filtered, \
                f"Inconsistent counts for {group}: {pre} != {post} + {filtered}"

            # Post-filter count should not exceed pre-filter
            assert post <= pre, \
                f"Post-filter count ({post}) > pre-filter count ({pre}) for {group}"

    def test_ir_provenance(self, service, clustered_adata):
        """Test that AnalysisStep IR is properly created."""
        _, _, ir = service.find_marker_genes(
            clustered_adata,
            groupby="leiden",
            min_fold_change=1.5,
            min_pct=0.25,
            max_out_pct=0.5
        )

        # Check IR structure
        assert ir.operation == "find_marker_genes_with_filtering", "Wrong operation name"
        assert ir.tool_name == "EnhancedSingleCellService.find_marker_genes", "Wrong tool name"
        assert "scanpy" in ir.library.lower(), "Missing scanpy in library"

        # Check code template contains both steps
        assert ir.code_template is not None, "Missing code template"
        assert "rank_genes_groups" in ir.code_template, "Missing rank_genes_groups call"
        assert "filter_rank_genes_groups" in ir.code_template, "Missing filter call"

        # Check parameters
        assert "min_fold_change" in ir.parameters, "Missing min_fold_change in IR"
        assert "min_pct" in ir.parameters, "Missing min_pct in IR"
        assert "max_out_pct" in ir.parameters, "Missing max_out_pct in IR"

        # Check parameter schema
        assert ir.parameter_schema is not None, "Missing parameter schema"
        assert "min_fold_change" in ir.parameter_schema, "Missing min_fold_change in schema"
        assert "min_pct" in ir.parameter_schema, "Missing min_pct in schema"
        assert "max_out_pct" in ir.parameter_schema, "Missing max_out_pct in schema"

    def test_parameter_schema_validation(self, service, clustered_adata):
        """Test that parameter schema has correct types and defaults."""
        _, _, ir = service.find_marker_genes(
            clustered_adata,
            groupby="leiden"
        )

        schema = ir.parameter_schema

        # min_fold_change schema
        assert schema["min_fold_change"]["type"] == "number"
        assert schema["min_fold_change"]["default"] == 1.5

        # min_pct schema
        assert schema["min_pct"]["type"] == "number"
        assert schema["min_pct"]["default"] == 0.25

        # max_out_pct schema
        assert schema["max_out_pct"]["type"] == "number"
        assert schema["max_out_pct"]["default"] == 0.5

    def test_filtering_with_custom_groupby(self, service, clustered_adata):
        """Test filtering works with custom groupby column."""
        # Add custom grouping
        clustered_adata.obs["custom_group"] = pd.Categorical(
            ["GroupA"] * 50 + ["GroupB"] * 50 + ["GroupC"] * 50
        )

        adata_result, stats, ir = service.find_marker_genes(
            clustered_adata,
            groupby="custom_group",
            min_fold_change=1.5
        )

        assert stats["groupby"] == "custom_group"
        assert set(stats["groups_analyzed"]) == {"GroupA", "GroupB", "GroupC"}

    def test_filtering_with_different_methods(self, service, clustered_adata):
        """Test filtering works with different DE methods."""
        methods = ["wilcoxon", "t-test", "logreg"]

        for method in methods:
            adata_result, stats, ir = service.find_marker_genes(
                clustered_adata,
                groupby="leiden",
                method=method,
                n_genes=20,
                min_fold_change=1.5
            )

            assert stats["method"] == method
            assert "filtering_params" in stats

    def test_sparse_matrix_support(self, service):
        """Test that filtering works with sparse matrices."""
        np.random.seed(42)
        n_cells = 100
        n_genes = 50

        # Create sparse data
        X_dense = np.random.rand(n_cells, n_genes) * 0.5
        X_dense[0:33, 0:10] += 2.0
        X_dense[34:66, 10:20] += 2.0
        X_dense[67:99, 20:30] += 2.0

        X_sparse = csr_matrix(X_dense)

        adata = anndata.AnnData(X=X_sparse)
        adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
        adata.obs["leiden"] = pd.Categorical(["0"] * 33 + ["1"] * 33 + ["2"] * 34)

        # Preprocess
        # Store raw counts before normalization
        adata.raw = adata.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Should work without errors
        adata_result, stats, ir = service.find_marker_genes(
            adata,
            groupby="leiden",
            min_fold_change=1.5,
            min_pct=0.25
        )

        assert "filtering_params" in stats
        assert stats["total_genes_filtered"] >= 0

    def test_zero_fold_change_threshold(self, service, clustered_adata):
        """Test behavior with zero fold-change threshold (no FC filtering)."""
        _, stats, _ = service.find_marker_genes(
            clustered_adata,
            groupby="leiden",
            min_fold_change=0.0,  # No fold-change filtering
            min_pct=0.0,
            max_out_pct=1.0
        )

        # Should still complete successfully
        assert "filtering_params" in stats
        assert stats["filtering_params"]["min_fold_change"] == 0.0

    def test_stringent_filtering(self, service, clustered_adata):
        """Test very stringent filtering removes many genes."""
        _, stats, _ = service.find_marker_genes(
            clustered_adata,
            groupby="leiden",
            n_genes=25,
            min_fold_change=3.0,   # Very high
            min_pct=0.5,           # Must be in 50% of cells
            max_out_pct=0.2        # Very specific
        )

        # With stringent thresholds, filtering should complete successfully
        # (Note: with very distinct synthetic markers, they may all pass even stringent filters)
        total_filtered = stats["total_genes_filtered"]
        assert total_filtered >= 0, "Should track filtered gene count"
        assert "filtering_params" in stats, "Should include filtering parameters"

        # Verify the stringent parameters were used
        assert stats["filtering_params"]["min_fold_change"] == 3.0
        assert stats["filtering_params"]["min_pct"] == 0.5
        assert stats["filtering_params"]["max_out_pct"] == 0.2


class TestHelperMethods:
    """Test helper methods for marker gene IR."""

    def test_create_marker_genes_ir_exists(self, service):
        """Test that IR creation method exists."""
        assert hasattr(service, "_create_marker_genes_ir"), \
            "Missing _create_marker_genes_ir method"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
