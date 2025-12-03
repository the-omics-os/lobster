"""
Comprehensive test suite for BulkVisualizationService.

This test suite provides comprehensive test coverage for:
- Volcano plot creation and validation
- MA plot creation and validation
- Expression heatmap creation and validation
- IR (AnalysisStep) structure and completeness
- Edge cases and error handling
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from anndata import AnnData
from jinja2 import Template

from lobster.core.analysis_ir import AnalysisStep
from lobster.services.visualization.bulk_visualization_service import (
    BulkVisualizationError,
    BulkVisualizationService,
)

# ============================================
# FIXTURES
# ============================================


@pytest.fixture
def bulk_viz_service():
    """Create BulkVisualizationService instance."""
    return BulkVisualizationService()


@pytest.fixture
def sample_de_adata():
    """Create sample DE results AnnData for testing."""
    np.random.seed(42)
    n_genes = 100

    # Create DE results with realistic patterns
    log2fc = np.random.uniform(-3, 3, n_genes)
    padj = np.random.beta(0.5, 2, n_genes)  # Skewed toward significant
    base_mean = np.random.lognormal(3, 2, n_genes)

    var = pd.DataFrame(
        {"log2FoldChange": log2fc, "padj": padj, "baseMean": base_mean},
        index=[f"gene_{i}" for i in range(n_genes)],
    )

    adata = AnnData(
        X=np.random.randn(10, n_genes),  # Dummy expression
        var=var,
    )
    return adata


@pytest.fixture
def sample_expression_adata():
    """Create sample expression AnnData for heatmap testing."""
    np.random.seed(42)
    n_samples = 12
    n_genes = 50

    X = np.random.randn(n_samples, n_genes)

    obs = pd.DataFrame(
        {
            "condition": ["control"] * 6 + ["treated"] * 6,
            "batch": ["batch1", "batch2"] * 6,
        },
        index=[f"sample_{i}" for i in range(n_samples)],
    )

    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

    adata = AnnData(X=X, obs=obs, var=var)
    return adata


@pytest.fixture
def all_significant_de_adata():
    """Create DE AnnData where all genes are significant."""
    np.random.seed(42)
    n_genes = 50

    log2fc = np.random.uniform(-4, 4, n_genes)
    padj = np.random.uniform(0, 0.01, n_genes)  # All below 0.05
    base_mean = np.random.lognormal(3, 1, n_genes)

    var = pd.DataFrame(
        {"log2FoldChange": log2fc, "padj": padj, "baseMean": base_mean},
        index=[f"gene_{i}" for i in range(n_genes)],
    )

    adata = AnnData(X=np.random.randn(10, n_genes), var=var)
    return adata


@pytest.fixture
def no_significant_de_adata():
    """Create DE AnnData where no genes are significant."""
    np.random.seed(42)
    n_genes = 50

    log2fc = np.random.uniform(-0.5, 0.5, n_genes)  # Small fold changes
    padj = np.random.uniform(0.5, 1.0, n_genes)  # All above 0.05
    base_mean = np.random.lognormal(3, 1, n_genes)

    var = pd.DataFrame(
        {"log2FoldChange": log2fc, "padj": padj, "baseMean": base_mean},
        index=[f"gene_{i}" for i in range(n_genes)],
    )

    adata = AnnData(X=np.random.randn(10, n_genes), var=var)
    return adata


# ============================================
# TEST CLASS 1: TestVolcanoPlot
# ============================================


class TestVolcanoPlot:
    """Test suite for volcano plot creation."""

    def test_create_volcano_plot_basic(self, bulk_viz_service, sample_de_adata):
        """Test basic volcano plot creation."""
        fig, stats, ir = bulk_viz_service.create_volcano_plot(
            sample_de_adata, fdr_threshold=0.05, fc_threshold=1.0, top_n_genes=10
        )

        # Verify types
        assert isinstance(fig, go.Figure)
        assert isinstance(stats, dict)
        assert isinstance(ir, AnalysisStep)

    def test_volcano_plot_returns_three_tuple(self, bulk_viz_service, sample_de_adata):
        """Verify volcano plot returns (fig, stats, ir) tuple."""
        result = bulk_viz_service.create_volcano_plot(
            sample_de_adata, fdr_threshold=0.05, fc_threshold=1.0, top_n_genes=10
        )

        # Unpack
        fig, stats, ir = result

        # Type checks
        assert isinstance(fig, go.Figure)
        assert isinstance(stats, dict)
        assert isinstance(ir, AnalysisStep)

    def test_volcano_plot_validates_columns(self, bulk_viz_service):
        """Test error handling when required columns are missing."""
        # Create AnnData without DE results
        adata = AnnData(
            X=np.random.randn(10, 50),
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(50)]),
        )

        with pytest.raises(BulkVisualizationError, match="Missing required columns"):
            bulk_viz_service.create_volcano_plot(adata, 0.05, 1.0)

    def test_volcano_plot_with_custom_thresholds(
        self, bulk_viz_service, sample_de_adata
    ):
        """Test volcano plot with custom FDR and FC thresholds."""
        fig, stats, ir = bulk_viz_service.create_volcano_plot(
            sample_de_adata, fdr_threshold=0.01, fc_threshold=2.0, top_n_genes=5
        )

        # Verify thresholds in stats
        assert stats["fdr_threshold"] == 0.01
        assert stats["fc_threshold"] == 2.0

        # Verify IR parameters
        assert ir.parameters["fdr_threshold"] == 0.01
        assert ir.parameters["fc_threshold"] == 2.0

    def test_volcano_plot_with_top_genes(self, bulk_viz_service, sample_de_adata):
        """Test gene labeling with different top_n values."""
        # With top_n = 0, no genes should be labeled in stats
        fig_no_labels, stats_no_labels, _ = bulk_viz_service.create_volcano_plot(
            sample_de_adata, fdr_threshold=0.05, fc_threshold=1.0, top_n_genes=0
        )

        # Should have 0 labeled genes in stats
        assert stats_no_labels["top_n_genes_labeled"] == 0

        # With top_n = 10
        fig_labels, stats_labels, _ = bulk_viz_service.create_volcano_plot(
            sample_de_adata, fdr_threshold=0.05, fc_threshold=1.0, top_n_genes=10
        )

        # Should have annotations (if significant genes exist)
        n_significant = stats_labels["n_genes_up"] + stats_labels["n_genes_down"]
        expected_labels = min(10, n_significant)
        assert stats_labels["top_n_genes_labeled"] == expected_labels

        # Check that more genes are labeled with top_n = 10 vs top_n = 0
        assert (
            stats_labels["top_n_genes_labeled"]
            >= stats_no_labels["top_n_genes_labeled"]
        )

    def test_volcano_plot_handles_no_significant_genes(
        self, bulk_viz_service, no_significant_de_adata
    ):
        """Test volcano plot when all genes are non-significant."""
        fig, stats, ir = bulk_viz_service.create_volcano_plot(
            no_significant_de_adata,
            fdr_threshold=0.05,
            fc_threshold=1.0,
            top_n_genes=10,
        )

        # Should have zero significant genes
        assert stats["n_genes_up"] == 0
        assert stats["n_genes_down"] == 0
        assert stats["n_genes_not_significant"] == stats["n_genes_total"]

        # Figure should still be created
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1  # At least non-significant trace

    def test_volcano_plot_handles_all_significant(
        self, bulk_viz_service, all_significant_de_adata
    ):
        """Test volcano plot when all genes are significant."""
        fig, stats, ir = bulk_viz_service.create_volcano_plot(
            all_significant_de_adata,
            fdr_threshold=0.05,
            fc_threshold=1.0,
            top_n_genes=10,
        )

        # Most genes should be significant (depending on fc_threshold)
        n_significant = stats["n_genes_up"] + stats["n_genes_down"]
        assert n_significant > 0

        # Figure should have multiple traces
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # At least up and down (or not_sig)

    def test_volcano_plot_statistics(self, bulk_viz_service, sample_de_adata):
        """Verify stats dict includes all required metrics."""
        _, stats, _ = bulk_viz_service.create_volcano_plot(
            sample_de_adata, fdr_threshold=0.05, fc_threshold=1.0, top_n_genes=10
        )

        required_keys = [
            "plot_type",
            "n_genes_total",
            "n_genes_up",
            "n_genes_down",
            "n_genes_not_significant",
            "fdr_threshold",
            "fc_threshold",
            "top_n_genes_labeled",
        ]

        for key in required_keys:
            assert key in stats, f"Missing key: {key}"

        # Verify counts add up
        total = (
            stats["n_genes_up"]
            + stats["n_genes_down"]
            + stats["n_genes_not_significant"]
        )
        assert total == stats["n_genes_total"]

        # Verify plot type
        assert stats["plot_type"] == "volcano_plot"

    def test_volcano_plot_figure_structure(self, bulk_viz_service, sample_de_adata):
        """Verify Plotly figure has correct traces and layout."""
        fig, stats, ir = bulk_viz_service.create_volcano_plot(
            sample_de_adata, fdr_threshold=0.05, fc_threshold=1.0
        )

        # Should have at least 1 trace (not_significant always exists)
        assert len(fig.data) >= 1

        # Check axis labels
        assert "log2 Fold Change" in fig.layout.xaxis.title.text
        assert "-log10(FDR)" in fig.layout.yaxis.title.text

        # Check threshold lines exist
        shapes = fig.layout.shapes
        assert len(shapes) >= 3  # At least 1 horizontal, 2 vertical

    def test_volcano_plot_handles_nan_values(self, bulk_viz_service):
        """Test volcano plot handles NaN/inf values gracefully."""
        np.random.seed(42)
        n_genes = 50

        log2fc = np.random.uniform(-3, 3, n_genes)
        padj = np.random.beta(0.5, 2, n_genes)
        base_mean = np.random.lognormal(3, 1, n_genes)

        # Introduce NaN values
        padj[0:5] = np.nan
        log2fc[10:15] = np.inf

        var = pd.DataFrame(
            {"log2FoldChange": log2fc, "padj": padj, "baseMean": base_mean},
            index=[f"gene_{i}" for i in range(n_genes)],
        )

        adata = AnnData(X=np.random.randn(10, n_genes), var=var)

        # Should handle gracefully
        fig, stats, ir = bulk_viz_service.create_volcano_plot(
            adata, fdr_threshold=0.05, fc_threshold=1.0
        )

        assert isinstance(fig, go.Figure)
        assert stats["n_genes_total"] == n_genes


# ============================================
# TEST CLASS 2: TestMAPlot
# ============================================


class TestMAPlot:
    """Test suite for MA plot creation."""

    def test_create_ma_plot_basic(self, bulk_viz_service, sample_de_adata):
        """Test basic MA plot creation."""
        fig, stats, ir = bulk_viz_service.create_ma_plot(
            sample_de_adata, fdr_threshold=0.05
        )

        # Verify types
        assert isinstance(fig, go.Figure)
        assert isinstance(stats, dict)
        assert isinstance(ir, AnalysisStep)

    def test_ma_plot_returns_three_tuple(self, bulk_viz_service, sample_de_adata):
        """Verify MA plot returns (fig, stats, ir) tuple."""
        result = bulk_viz_service.create_ma_plot(sample_de_adata, fdr_threshold=0.05)

        # Unpack
        fig, stats, ir = result

        # Type checks
        assert isinstance(fig, go.Figure)
        assert isinstance(stats, dict)
        assert isinstance(ir, AnalysisStep)

    def test_ma_plot_validates_columns(self, bulk_viz_service):
        """Test error handling when required columns are missing."""
        # Create AnnData without DE results
        adata = AnnData(
            X=np.random.randn(10, 50),
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(50)]),
        )

        with pytest.raises(BulkVisualizationError, match="Missing required columns"):
            bulk_viz_service.create_ma_plot(adata, 0.05)

    def test_ma_plot_with_custom_threshold(self, bulk_viz_service, sample_de_adata):
        """Test MA plot with custom FDR threshold."""
        fig, stats, ir = bulk_viz_service.create_ma_plot(
            sample_de_adata, fdr_threshold=0.01
        )

        # Verify threshold in stats
        assert stats["fdr_threshold"] == 0.01

        # Verify IR parameters
        assert ir.parameters["fdr_threshold"] == 0.01

    def test_ma_plot_statistics(self, bulk_viz_service, sample_de_adata):
        """Verify stats dict includes all required metrics."""
        _, stats, _ = bulk_viz_service.create_ma_plot(
            sample_de_adata, fdr_threshold=0.05
        )

        required_keys = [
            "plot_type",
            "n_genes_total",
            "n_genes_significant",
            "fdr_threshold",
            "mean_base_mean",
            "median_base_mean",
        ]

        for key in required_keys:
            assert key in stats, f"Missing key: {key}"

        # Verify plot type
        assert stats["plot_type"] == "ma_plot"

        # Verify mean/median are reasonable
        assert stats["mean_base_mean"] > 0
        assert stats["median_base_mean"] > 0

    def test_ma_plot_figure_structure(self, bulk_viz_service, sample_de_adata):
        """Verify Plotly figure has correct traces and layout."""
        fig, stats, ir = bulk_viz_service.create_ma_plot(
            sample_de_adata, fdr_threshold=0.05
        )

        # Should have at least 1 trace (not_significant always exists)
        assert len(fig.data) >= 1

        # Check axis labels
        assert "log10(Mean Expression)" in fig.layout.xaxis.title.text
        assert "log2 Fold Change" in fig.layout.yaxis.title.text

        # Check horizontal line at y=0 exists
        shapes = fig.layout.shapes
        assert len(shapes) >= 1  # At least horizontal line

    def test_ma_plot_handles_no_significant(
        self, bulk_viz_service, no_significant_de_adata
    ):
        """Test MA plot when no genes are significant."""
        fig, stats, ir = bulk_viz_service.create_ma_plot(
            no_significant_de_adata, fdr_threshold=0.05
        )

        # Should have zero significant genes
        assert stats["n_genes_significant"] == 0

        # Figure should still be created
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1  # At least non-significant trace

    def test_ma_plot_handles_nan_values(self, bulk_viz_service):
        """Test MA plot handles NaN/inf values gracefully."""
        np.random.seed(42)
        n_genes = 50

        log2fc = np.random.uniform(-3, 3, n_genes)
        padj = np.random.beta(0.5, 2, n_genes)
        base_mean = np.random.lognormal(3, 1, n_genes)

        # Introduce NaN values
        padj[0:5] = np.nan
        base_mean[10:15] = np.nan

        var = pd.DataFrame(
            {"log2FoldChange": log2fc, "padj": padj, "baseMean": base_mean},
            index=[f"gene_{i}" for i in range(n_genes)],
        )

        adata = AnnData(X=np.random.randn(10, n_genes), var=var)

        # Should handle gracefully
        fig, stats, ir = bulk_viz_service.create_ma_plot(adata, fdr_threshold=0.05)

        assert isinstance(fig, go.Figure)
        assert stats["n_genes_total"] == n_genes


# ============================================
# TEST CLASS 3: TestExpressionHeatmap
# ============================================


class TestExpressionHeatmap:
    """Test suite for expression heatmap creation."""

    def test_create_heatmap_basic(self, bulk_viz_service, sample_expression_adata):
        """Test basic heatmap with gene list."""
        gene_list = ["gene_0", "gene_1", "gene_2", "gene_3", "gene_4"]

        fig, stats, ir = bulk_viz_service.create_expression_heatmap(
            sample_expression_adata,
            gene_list=gene_list,
            cluster_samples=True,
            cluster_genes=True,
            z_score=True,
        )

        # Verify types
        assert isinstance(fig, go.Figure)
        assert isinstance(stats, dict)
        assert isinstance(ir, AnalysisStep)

    def test_heatmap_returns_three_tuple(
        self, bulk_viz_service, sample_expression_adata
    ):
        """Verify heatmap returns (fig, stats, ir) tuple."""
        gene_list = ["gene_0", "gene_1", "gene_2"]

        result = bulk_viz_service.create_expression_heatmap(
            sample_expression_adata, gene_list=gene_list
        )

        # Unpack
        fig, stats, ir = result

        # Type checks
        assert isinstance(fig, go.Figure)
        assert isinstance(stats, dict)
        assert isinstance(ir, AnalysisStep)

    def test_heatmap_with_clustering(self, bulk_viz_service, sample_expression_adata):
        """Test heatmap with hierarchical clustering enabled."""
        gene_list = ["gene_0", "gene_1", "gene_2", "gene_3", "gene_4"]

        fig, stats, ir = bulk_viz_service.create_expression_heatmap(
            sample_expression_adata,
            gene_list=gene_list,
            cluster_samples=True,
            cluster_genes=True,
        )

        # Verify clustering flags in stats
        assert stats["clustered_samples"] is True
        assert stats["clustered_genes"] is True

    def test_heatmap_without_clustering(
        self, bulk_viz_service, sample_expression_adata
    ):
        """Test heatmap without clustering."""
        gene_list = ["gene_0", "gene_1", "gene_2"]

        fig, stats, ir = bulk_viz_service.create_expression_heatmap(
            sample_expression_adata,
            gene_list=gene_list,
            cluster_samples=False,
            cluster_genes=False,
        )

        # Verify clustering flags in stats
        assert stats["clustered_samples"] is False
        assert stats["clustered_genes"] is False

    def test_heatmap_with_zscore(self, bulk_viz_service, sample_expression_adata):
        """Test heatmap with z-score normalization."""
        gene_list = ["gene_0", "gene_1", "gene_2"]

        fig, stats, ir = bulk_viz_service.create_expression_heatmap(
            sample_expression_adata, gene_list=gene_list, z_score=True
        )

        # Verify z-score flag in stats
        assert stats["z_score_normalized"] is True

        # Verify colorbar title indicates z-score
        assert "z-score" in fig.data[0].colorbar.title.text.lower()

    def test_heatmap_without_zscore(self, bulk_viz_service, sample_expression_adata):
        """Test heatmap with raw expression values."""
        gene_list = ["gene_0", "gene_1", "gene_2"]

        fig, stats, ir = bulk_viz_service.create_expression_heatmap(
            sample_expression_adata, gene_list=gene_list, z_score=False
        )

        # Verify z-score flag in stats
        assert stats["z_score_normalized"] is False

        # Verify colorbar title does not indicate z-score
        assert "z-score" not in fig.data[0].colorbar.title.text.lower()

    def test_heatmap_auto_gene_selection(
        self, bulk_viz_service, sample_expression_adata
    ):
        """Test heatmap with automatic gene selection (top 50 variable genes)."""
        # Don't provide gene_list
        fig, stats, ir = bulk_viz_service.create_expression_heatmap(
            sample_expression_adata, gene_list=None
        )

        # Should select up to 50 genes (or all if < 50)
        assert stats["n_genes"] <= 50
        assert stats["n_genes"] > 0

    def test_heatmap_filters_missing_genes(
        self, bulk_viz_service, sample_expression_adata
    ):
        """Test heatmap filters genes not in adata.var_names."""
        gene_list = ["gene_0", "gene_1", "nonexistent_gene", "gene_2"]

        # Should filter out 'nonexistent_gene'
        fig, stats, ir = bulk_viz_service.create_expression_heatmap(
            sample_expression_adata, gene_list=gene_list
        )

        # Should only include 3 genes
        assert stats["n_genes"] == 3

    def test_heatmap_handles_insufficient_genes(
        self, bulk_viz_service, sample_expression_adata
    ):
        """Test heatmap error when no valid genes found."""
        gene_list = ["nonexistent_gene1", "nonexistent_gene2"]

        # Should raise error
        with pytest.raises(BulkVisualizationError, match="No valid genes found"):
            bulk_viz_service.create_expression_heatmap(
                sample_expression_adata, gene_list=gene_list
            )

    def test_heatmap_statistics(self, bulk_viz_service, sample_expression_adata):
        """Verify stats dict includes all required metrics."""
        gene_list = ["gene_0", "gene_1", "gene_2"]

        _, stats, _ = bulk_viz_service.create_expression_heatmap(
            sample_expression_adata, gene_list=gene_list
        )

        required_keys = [
            "plot_type",
            "n_samples",
            "n_genes",
            "clustered_samples",
            "clustered_genes",
            "z_score_normalized",
        ]

        for key in required_keys:
            assert key in stats, f"Missing key: {key}"

        # Verify plot type
        assert stats["plot_type"] == "expression_heatmap"

        # Verify counts
        assert stats["n_samples"] == 12
        assert stats["n_genes"] == 3

    def test_heatmap_figure_structure(self, bulk_viz_service, sample_expression_adata):
        """Verify Plotly heatmap structure."""
        gene_list = ["gene_0", "gene_1", "gene_2"]

        fig, stats, ir = bulk_viz_service.create_expression_heatmap(
            sample_expression_adata, gene_list=gene_list
        )

        # Should have 1 heatmap trace
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)

        # Check axis labels
        assert "Samples" in fig.layout.xaxis.title.text
        assert "Genes" in fig.layout.yaxis.title.text


# ============================================
# TEST CLASS 4: TestVisualizationIR
# ============================================


class TestVisualizationIR:
    """Test suite for IR (AnalysisStep) structure and completeness."""

    def test_volcano_ir_structure(self, bulk_viz_service, sample_de_adata):
        """Validate volcano plot IR completeness."""
        _, _, ir = bulk_viz_service.create_volcano_plot(
            sample_de_adata, fdr_threshold=0.05, fc_threshold=1.0
        )

        # Required fields
        assert ir.operation == "visualization.volcano_plot"
        assert ir.tool_name == "create_volcano_plot"
        assert ir.library == "plotly"
        assert ir.description is not None
        assert ir.code_template is not None
        assert len(ir.imports) > 0
        assert ir.parameter_schema is not None

        # Parameter schema
        schema = ir.parameter_schema
        assert "fdr_threshold" in schema
        assert "fc_threshold" in schema
        assert "top_n_genes" in schema

        # Check types
        assert schema["fdr_threshold"]["type"] == "float"
        assert schema["fc_threshold"]["type"] == "float"
        assert schema["top_n_genes"]["type"] == "int"

    def test_volcano_ir_parameter_schema(self, bulk_viz_service, sample_de_adata):
        """Validate volcano plot parameter schema completeness."""
        _, _, ir = bulk_viz_service.create_volcano_plot(
            sample_de_adata, fdr_threshold=0.05, fc_threshold=1.0
        )

        schema = ir.parameter_schema

        # Check all parameters have required fields
        for param_name, param_def in schema.items():
            assert "type" in param_def
            assert "description" in param_def
            # default is optional but should exist for most
            if param_name != "top_n_genes":
                assert "default" in param_def

    def test_volcano_ir_code_template_renders(self, bulk_viz_service, sample_de_adata):
        """Test that Jinja2 code template renders without errors."""
        _, _, ir = bulk_viz_service.create_volcano_plot(
            sample_de_adata, fdr_threshold=0.05, fc_threshold=1.5
        )

        # Render template with parameters
        template = Template(ir.code_template)
        rendered = template.render(**ir.parameters)

        # Should not have any unreplaced Jinja2 variables
        assert "{{" not in rendered
        assert "{%" not in rendered

        # Should be valid Python (compile check)
        compile(rendered, "<string>", "exec")

    def test_ma_ir_structure(self, bulk_viz_service, sample_de_adata):
        """Validate MA plot IR completeness."""
        _, _, ir = bulk_viz_service.create_ma_plot(sample_de_adata, fdr_threshold=0.05)

        # Required fields
        assert ir.operation == "visualization.ma_plot"
        assert ir.tool_name == "create_ma_plot"
        assert ir.library == "plotly"
        assert ir.description is not None
        assert ir.code_template is not None
        assert len(ir.imports) > 0
        assert ir.parameter_schema is not None

        # Parameter schema
        schema = ir.parameter_schema
        assert "fdr_threshold" in schema
        assert schema["fdr_threshold"]["type"] == "float"

    def test_ma_ir_code_template_renders(self, bulk_viz_service, sample_de_adata):
        """Test that Jinja2 code template renders without errors for MA plot."""
        _, _, ir = bulk_viz_service.create_ma_plot(sample_de_adata, fdr_threshold=0.01)

        # Render template with parameters
        template = Template(ir.code_template)
        rendered = template.render(**ir.parameters)

        # Should not have any unreplaced Jinja2 variables
        assert "{{" not in rendered
        assert "{%" not in rendered

        # Should be valid Python (compile check)
        compile(rendered, "<string>", "exec")

    def test_heatmap_ir_structure(self, bulk_viz_service, sample_expression_adata):
        """Validate heatmap IR completeness."""
        gene_list = ["gene_0", "gene_1", "gene_2"]

        _, _, ir = bulk_viz_service.create_expression_heatmap(
            sample_expression_adata, gene_list=gene_list
        )

        # Required fields
        assert ir.operation == "visualization.expression_heatmap"
        assert ir.tool_name == "create_expression_heatmap"
        assert ir.library == "plotly"
        assert ir.description is not None
        assert ir.code_template is not None
        assert len(ir.imports) > 0
        assert ir.parameter_schema is not None

        # Parameter schema
        schema = ir.parameter_schema
        assert "cluster_samples" in schema
        assert "cluster_genes" in schema
        assert "z_score" in schema
        assert "n_genes" in schema

    def test_heatmap_ir_code_template_renders(
        self, bulk_viz_service, sample_expression_adata
    ):
        """Test that Jinja2 code template renders without errors for heatmap."""
        gene_list = ["gene_0", "gene_1", "gene_2"]

        _, _, ir = bulk_viz_service.create_expression_heatmap(
            sample_expression_adata,
            gene_list=gene_list,
            cluster_samples=True,
            z_score=True,
        )

        # Render template with parameters
        template = Template(ir.code_template)
        rendered = template.render(**ir.parameters)

        # Should not have any unreplaced Jinja2 variables
        assert "{{" not in rendered
        assert "{%" not in rendered

        # Should be valid Python (compile check)
        compile(rendered, "<string>", "exec")

    def test_all_ir_imports(
        self, bulk_viz_service, sample_de_adata, sample_expression_adata
    ):
        """Validate imports lists for all methods."""
        # Volcano plot
        _, _, ir_volcano = bulk_viz_service.create_volcano_plot(
            sample_de_adata, fdr_threshold=0.05, fc_threshold=1.0
        )
        assert "import plotly.graph_objects as go" in ir_volcano.imports
        assert "import numpy as np" in ir_volcano.imports

        # MA plot
        _, _, ir_ma = bulk_viz_service.create_ma_plot(
            sample_de_adata, fdr_threshold=0.05
        )
        assert "import plotly.graph_objects as go" in ir_ma.imports
        assert "import numpy as np" in ir_ma.imports

        # Heatmap
        gene_list = ["gene_0", "gene_1", "gene_2"]
        _, _, ir_heatmap = bulk_viz_service.create_expression_heatmap(
            sample_expression_adata, gene_list=gene_list
        )
        assert "import plotly.graph_objects as go" in ir_heatmap.imports
        assert "import numpy as np" in ir_heatmap.imports
        assert (
            "from scipy.cluster.hierarchy import linkage, dendrogram"
            in ir_heatmap.imports
        )


# ============================================
# TEST CLASS 5: TestVisualizationEdgeCases
# ============================================


class TestVisualizationEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_volcano_with_zero_genes(self, bulk_viz_service):
        """Test volcano plot with empty AnnData."""
        # Create empty AnnData with proper columns
        var = pd.DataFrame({"log2FoldChange": [], "padj": [], "baseMean": []}, index=[])
        adata = AnnData(X=np.array([]).reshape(0, 0), var=var)

        # Should handle gracefully
        fig, stats, ir = bulk_viz_service.create_volcano_plot(
            adata, fdr_threshold=0.05, fc_threshold=1.0
        )

        assert stats["n_genes_total"] == 0
        assert stats["n_genes_up"] == 0
        assert stats["n_genes_down"] == 0

    def test_ma_with_zero_mean_expression(self, bulk_viz_service):
        """Test MA plot with zero baseMean handling."""
        np.random.seed(42)
        n_genes = 50

        log2fc = np.random.uniform(-3, 3, n_genes)
        padj = np.random.beta(0.5, 2, n_genes)
        base_mean = np.zeros(n_genes)  # All zeros

        var = pd.DataFrame(
            {"log2FoldChange": log2fc, "padj": padj, "baseMean": base_mean},
            index=[f"gene_{i}" for i in range(n_genes)],
        )

        adata = AnnData(X=np.random.randn(10, n_genes), var=var)

        # Should handle gracefully (adds 1 to avoid log(0))
        fig, stats, ir = bulk_viz_service.create_ma_plot(adata, fdr_threshold=0.05)

        assert isinstance(fig, go.Figure)
        assert stats["mean_base_mean"] == 0.0

    def test_heatmap_with_single_sample(self, bulk_viz_service):
        """Test heatmap with single sample edge case."""
        np.random.seed(42)
        n_genes = 20

        X = np.random.randn(1, n_genes)  # Single sample

        obs = pd.DataFrame({"condition": ["control"]}, index=["sample_0"])
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

        adata = AnnData(X=X, obs=obs, var=var)

        gene_list = ["gene_0", "gene_1", "gene_2"]

        # Should handle gracefully (clustering will be skipped)
        fig, stats, ir = bulk_viz_service.create_expression_heatmap(
            adata, gene_list=gene_list, cluster_samples=True
        )

        assert stats["n_samples"] == 1
        assert isinstance(fig, go.Figure)

    def test_heatmap_with_single_gene(self, bulk_viz_service):
        """Test heatmap with single gene edge case."""
        np.random.seed(42)
        n_samples = 10

        X = np.random.randn(n_samples, 1)  # Single gene

        obs = pd.DataFrame(
            {"condition": ["control"] * 5 + ["treated"] * 5},
            index=[f"sample_{i}" for i in range(n_samples)],
        )
        var = pd.DataFrame(index=["gene_0"])

        adata = AnnData(X=X, obs=obs, var=var)

        gene_list = ["gene_0"]

        # Should handle gracefully (gene clustering will be skipped)
        fig, stats, ir = bulk_viz_service.create_expression_heatmap(
            adata, gene_list=gene_list, cluster_genes=True
        )

        assert stats["n_genes"] == 1
        assert isinstance(fig, go.Figure)

    def test_heatmap_with_constant_expression(self, bulk_viz_service):
        """Test heatmap with constant expression (z-score should handle)."""
        # Create AnnData with constant gene expression
        X = np.ones((10, 5))  # All values are 1.0
        obs = pd.DataFrame(
            {"condition": ["control"] * 5 + ["treated"] * 5},
            index=[f"sample_{i}" for i in range(10)],
        )
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(5)])
        adata = AnnData(X=X, obs=obs, var=var)

        gene_list = ["gene_0", "gene_1", "gene_2"]

        # Should handle gracefully (z-score will handle division by zero)
        fig, stats, ir = bulk_viz_service.create_expression_heatmap(
            adata,
            gene_list=gene_list,
            cluster_samples=False,
            cluster_genes=False,
            z_score=True,
        )

        assert isinstance(fig, go.Figure)
        assert stats["n_genes"] == 3
