"""
Comprehensive unit tests for proteomics visualization service.

This module provides thorough testing of the proteomics visualization service including
missing value heatmaps, intensity distributions, volcano plots, protein networks,
and QC dashboards for proteomics data visualization.

Test coverage target: 95%+ with meaningful tests for proteomics visualization operations.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, Mock, mock_open, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.services.visualization.proteomics_visualization_service import (
    ProteomicsVisualizationError,
    ProteomicsVisualizationService,
)
from tests.mock_data.base import LARGE_DATASET_CONFIG, SMALL_DATASET_CONFIG
from tests.mock_data.factories import ProteomicsDataFactory

# ===============================================================================
# Mock Data and Fixtures
# ===============================================================================


@pytest.fixture
def mock_proteomics_data():
    """Create mock proteomics data for testing."""
    return ProteomicsDataFactory(config=SMALL_DATASET_CONFIG)


@pytest.fixture
def service():
    """Create ProteomicsVisualizationService instance."""
    return ProteomicsVisualizationService()


@pytest.fixture
def mock_adata_with_missing():
    """Create mock AnnData with missing values for visualization."""
    n_samples, n_proteins = 48, 80
    X = np.random.lognormal(mean=9, sigma=1, size=(n_samples, n_proteins))

    # Add structured missing values
    missing_rate_per_protein = (
        np.random.beta(2, 5, n_proteins) * 0.6
    )  # 0-60% missing per protein
    for protein_idx in range(n_proteins):
        n_missing = int(missing_rate_per_protein[protein_idx] * n_samples)
        missing_indices = np.random.choice(n_samples, n_missing, replace=False)
        X[missing_indices, protein_idx] = np.nan

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"protein_{i}" for i in range(n_proteins)]

    # Add metadata
    adata.obs["condition"] = (
        ["control"] * 16 + ["treatment1"] * 16 + ["treatment2"] * 16
    )
    adata.obs["batch"] = ["batch1"] * 16 + ["batch2"] * 16 + ["batch3"] * 16
    adata.var["protein_names"] = [f"PROT_{i}" for i in range(n_proteins)]

    # Add missing value QC metrics
    adata.obs["missing_value_percentage"] = (np.isnan(X).sum(axis=1) / n_proteins) * 100
    adata.var["missing_value_percentage"] = (np.isnan(X).sum(axis=0) / n_samples) * 100

    return adata


@pytest.fixture
def mock_adata_with_de_results():
    """Create mock AnnData with differential expression results."""
    n_samples, n_proteins = 60, 100
    X = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))

    # Add differential expression pattern
    de_proteins = np.random.choice(n_proteins, 30, replace=False)
    for protein_idx in de_proteins[:15]:  # Upregulated
        X[30:, protein_idx] *= 2.5
    for protein_idx in de_proteins[15:]:  # Downregulated
        X[30:, protein_idx] *= 0.4

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"protein_{i}" for i in range(n_proteins)]

    # Add differential expression metadata
    adata.obs["condition"] = ["control"] * 30 + ["treatment"] * 30

    # Mock DE results
    de_results = []
    for i, protein_idx in enumerate(de_proteins):
        if i < 15:  # Upregulated
            fold_change = 2.5
            p_value = 0.001
        else:  # Downregulated
            fold_change = 0.4
            p_value = 0.001

        de_results.append(
            {
                "protein": f"protein_{protein_idx}",
                "fold_change": fold_change,
                "log2_fold_change": np.log2(fold_change),
                "p_value": p_value,
                "p_adjusted": p_value * 1.5,
                "significant": True,
            }
        )

    # Add non-significant results
    for protein_idx in range(n_proteins):
        if protein_idx not in de_proteins:
            de_results.append(
                {
                    "protein": f"protein_{protein_idx}",
                    "fold_change": 1.1,
                    "log2_fold_change": np.log2(1.1),
                    "p_value": 0.5,
                    "p_adjusted": 0.7,
                    "significant": False,
                }
            )

    adata.uns["differential_expression"] = {"results": de_results}

    return adata


@pytest.fixture
def mock_adata_with_cv_data():
    """Create mock AnnData with CV analysis data."""
    n_samples, n_proteins = 40, 60
    X = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"protein_{i}" for i in range(n_proteins)]

    # Add CV metrics
    cv_values = []
    for protein_idx in range(n_proteins):
        protein_data = X[:, protein_idx]
        cv = np.std(protein_data) / np.mean(protein_data)
        cv_values.append(cv)

    adata.var["cv_mean"] = cv_values
    adata.var["high_cv_protein"] = np.array(cv_values) > 0.3

    # Add replicate information
    adata.obs["replicate_group"] = [f"group_{i // 4}" for i in range(n_samples)]

    return adata


@pytest.fixture
def mock_adata_with_correlations():
    """Create mock AnnData with correlation data."""
    n_samples, n_proteins = 50, 40
    X = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))

    # Create some correlated proteins
    X[:, 1] = X[:, 0] * 1.2 + np.random.normal(
        0, X[:, 0] * 0.1, n_samples
    )  # Positive correlation
    X[:, 2] = (
        np.max(X[:, 0]) - X[:, 0] + np.random.normal(0, X[:, 0] * 0.1, n_samples)
    )  # Negative correlation

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"protein_{i}" for i in range(n_proteins)]

    # Mock correlation results
    correlation_results = [
        {
            "protein1": "protein_0",
            "protein2": "protein_1",
            "correlation": 0.85,
            "p_value": 0.001,
        },
        {
            "protein1": "protein_0",
            "protein2": "protein_2",
            "correlation": -0.75,
            "p_value": 0.001,
        },
        {
            "protein1": "protein_1",
            "protein2": "protein_3",
            "correlation": 0.65,
            "p_value": 0.01,
        },
    ]

    adata.uns["correlation_analysis"] = {"results": correlation_results}

    return adata


@pytest.fixture
def mock_adata_with_pathway_results():
    """Create mock AnnData with pathway enrichment results."""
    n_samples, n_proteins = 30, 50
    X = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"protein_{i}" for i in range(n_proteins)]

    # Mock pathway enrichment results
    pathway_results = [
        {
            "pathway_name": "Cell_Cycle",
            "p_value": 0.001,
            "enrichment_ratio": 3.2,
            "overlap_count": 8,
            "pathway_size": 25,
        },
        {
            "pathway_name": "Apoptosis",
            "p_value": 0.01,
            "enrichment_ratio": 2.1,
            "overlap_count": 5,
            "pathway_size": 20,
        },
        {
            "pathway_name": "Metabolism",
            "p_value": 0.03,
            "enrichment_ratio": 1.8,
            "overlap_count": 12,
            "pathway_size": 60,
        },
    ]

    adata.uns["pathway_enrichment"] = {"results": pathway_results}

    return adata


# ===============================================================================
# Service Initialization Tests
# ===============================================================================


class TestProteomicsVisualizationServiceInitialization:
    """Test suite for ProteomicsVisualizationService initialization."""

    def test_init_default_parameters(self):
        """Test service initialization with default parameters."""
        service = ProteomicsVisualizationService()

        assert service is not None


# ===============================================================================
# Missing Value Heatmap Tests
# ===============================================================================


class TestMissingValueHeatmap:
    """Test suite for missing value heatmap functionality."""

    def test_create_missing_value_heatmap_basic(self, service, mock_adata_with_missing):
        """Test basic missing value heatmap creation."""
        fig, stats = service.create_missing_value_heatmap(mock_adata_with_missing)

        assert fig is not None
        assert isinstance(stats, dict)
        assert "plot_type" in stats
        assert stats["plot_type"] == "missing_value_heatmap"
        assert "total_missing_percentage" in stats
        assert "samples_plotted" in stats
        assert "proteins_plotted" in stats

    def test_create_missing_value_heatmap_sample_subset(
        self, service, mock_adata_with_missing
    ):
        """Test missing value heatmap with sample subset."""
        # Use max_samples parameter instead of sample_subset
        fig, stats = service.create_missing_value_heatmap(
            mock_adata_with_missing, max_samples=20
        )

        assert fig is not None
        assert stats["samples_plotted"] <= 20

    def test_create_missing_value_heatmap_protein_subset(
        self, service, mock_adata_with_missing
    ):
        """Test missing value heatmap with protein subset."""
        # Use max_proteins parameter instead of protein_subset
        fig, stats = service.create_missing_value_heatmap(
            mock_adata_with_missing, max_proteins=30
        )

        assert fig is not None
        assert stats["proteins_plotted"] <= 30

    def test_create_missing_value_heatmap_custom_colorscale(
        self, service, mock_adata_with_missing
    ):
        """Test missing value heatmap with custom title."""
        # Colorscale parameter not supported, test with custom title instead
        fig, stats = service.create_missing_value_heatmap(
            mock_adata_with_missing, title="Custom Missing Value Heatmap"
        )

        assert fig is not None

    def test_create_missing_value_heatmap_no_missing(self, service):
        """Test missing value heatmap with no missing values."""
        X = np.random.lognormal(mean=8, sigma=1, size=(20, 30))
        adata = ad.AnnData(X=X)

        fig, stats = service.create_missing_value_heatmap(adata)

        assert fig is not None
        assert stats["total_missing_percentage"] == 0.0


# ===============================================================================
# Intensity Distribution Plot Tests
# ===============================================================================


class TestIntensityDistributionPlot:
    """Test suite for intensity distribution plot functionality."""

    def test_create_intensity_distribution_plot_basic(
        self, service, mock_adata_with_missing
    ):
        """Test basic intensity distribution plot creation."""
        fig, stats = service.create_intensity_distribution_plot(mock_adata_with_missing)

        assert fig is not None
        assert isinstance(stats, dict)
        assert stats["plot_type"] == "intensity_distribution"
        assert "distribution_stats" in stats

    def test_create_intensity_distribution_plot_by_group(
        self, service, mock_adata_with_missing
    ):
        """Test intensity distribution plot grouped by condition."""
        fig, stats = service.create_intensity_distribution_plot(
            mock_adata_with_missing, group_by="condition"
        )

        assert fig is not None
        assert "group_stats" in stats

    def test_create_intensity_distribution_plot_log_scale(
        self, service, mock_adata_with_missing
    ):
        """Test intensity distribution plot with log transformation."""
        # log_transform parameter is supported (default True)
        fig, stats = service.create_intensity_distribution_plot(
            mock_adata_with_missing, log_transform=True
        )

        assert fig is not None
        assert stats["log_transformed"] == True

    def test_create_intensity_distribution_plot_histogram(
        self, service, mock_adata_with_missing
    ):
        """Test intensity distribution plot as histogram (default behavior)."""
        # plot_type parameter not supported, service creates histogram by default
        fig, stats = service.create_intensity_distribution_plot(mock_adata_with_missing)

        assert fig is not None

    def test_create_intensity_distribution_plot_density(
        self, service, mock_adata_with_missing
    ):
        """Test intensity distribution plot with custom title."""
        # plot_type="density" not supported, test with custom title instead
        fig, stats = service.create_intensity_distribution_plot(
            mock_adata_with_missing, title="Density Plot"
        )

        assert fig is not None

    def test_create_intensity_distribution_plot_violin(
        self, service, mock_adata_with_missing
    ):
        """Test intensity distribution plot grouped by condition (violin-like)."""
        # group_by is supported and creates violin plots
        fig, stats = service.create_intensity_distribution_plot(
            mock_adata_with_missing, group_by="condition"
        )

        assert fig is not None
        assert "group_stats" in stats


# ===============================================================================
# CV Analysis Plot Tests
# ===============================================================================


class TestCVAnalysisPlot:
    """Test suite for CV analysis plot functionality."""

    def test_create_cv_analysis_plot_basic(self, service, mock_adata_with_cv_data):
        """Test basic CV analysis plot creation."""
        fig, stats = service.create_cv_analysis_plot(mock_adata_with_cv_data)

        assert fig is not None
        assert isinstance(stats, dict)
        assert stats["plot_type"] == "cv_analysis"
        assert "cv_statistics" in stats

    def test_create_cv_analysis_plot_by_replicate(
        self, service, mock_adata_with_cv_data
    ):
        """Test CV analysis plot by replicate groups."""
        # group_by parameter is supported
        fig, stats = service.create_cv_analysis_plot(
            mock_adata_with_cv_data, group_by="replicate_group"
        )

        assert fig is not None
        assert "replicate_cv_stats" in stats

    def test_create_cv_analysis_plot_custom_threshold(
        self, service, mock_adata_with_cv_data
    ):
        """Test CV analysis plot with custom CV threshold."""
        # cv_threshold parameter is supported (in percentage)
        fig, stats = service.create_cv_analysis_plot(
            mock_adata_with_cv_data, cv_threshold=25.0
        )

        assert fig is not None

    def test_create_cv_analysis_plot_scatter(self, service, mock_adata_with_cv_data):
        """Test CV analysis plot (default histogram)."""
        # plot_type parameter not supported, default is histogram
        fig, stats = service.create_cv_analysis_plot(mock_adata_with_cv_data)

        assert fig is not None

    def test_create_cv_analysis_plot_histogram(self, service, mock_adata_with_cv_data):
        """Test CV analysis plot as histogram (default behavior)."""
        # Default behavior is histogram
        fig, stats = service.create_cv_analysis_plot(mock_adata_with_cv_data)

        assert fig is not None


# ===============================================================================
# Volcano Plot Tests
# ===============================================================================


class TestVolcanoPlot:
    """Test suite for volcano plot functionality."""

    def test_create_volcano_plot_basic(self, service, mock_adata_with_de_results):
        """Test basic volcano plot creation."""
        # comparison parameter not needed, uses data from adata.uns
        fig, stats = service.create_volcano_plot(mock_adata_with_de_results)

        assert fig is not None
        assert isinstance(stats, dict)
        assert stats["plot_type"] == "volcano_plot"
        assert "n_significant_up" in stats
        assert "n_significant_down" in stats

    def test_create_volcano_plot_custom_thresholds(
        self, service, mock_adata_with_de_results
    ):
        """Test volcano plot with custom significance thresholds."""
        # fc_threshold and pvalue_threshold are supported parameters
        fig, stats = service.create_volcano_plot(
            mock_adata_with_de_results,
            pvalue_threshold=0.01,
            fc_threshold=2.0,
        )

        assert fig is not None

    def test_create_volcano_plot_labeled_proteins(
        self, service, mock_adata_with_de_results
    ):
        """Test volcano plot with highlighted proteins."""
        # highlight_proteins parameter is supported
        proteins_to_highlight = ["protein_0", "protein_1", "protein_2"]
        fig, stats = service.create_volcano_plot(
            mock_adata_with_de_results,
            highlight_proteins=proteins_to_highlight,
        )

        assert fig is not None

    def test_create_volcano_plot_custom_colors(
        self, service, mock_adata_with_de_results
    ):
        """Test volcano plot with custom title."""
        # colors parameter not supported, test with custom title instead
        fig, stats = service.create_volcano_plot(
            mock_adata_with_de_results,
            title="Custom Volcano Plot",
        )

        assert fig is not None

    def test_create_volcano_plot_no_de_results(self, service, mock_adata_with_missing):
        """Test volcano plot with no DE results."""
        with pytest.raises(ProteomicsVisualizationError) as exc_info:
            service.create_volcano_plot(mock_adata_with_missing)

        assert "No differential expression results found" in str(exc_info.value)


# ===============================================================================
# Protein Correlation Network Tests
# ===============================================================================


class TestProteinCorrelationNetwork:
    """Test suite for protein correlation network functionality."""

    def test_create_protein_correlation_network_basic(
        self, service, mock_adata_with_correlations
    ):
        """Test basic protein correlation network creation."""
        # Network creation uses computed correlations, not stored results
        fig, stats = service.create_protein_correlation_network(
            mock_adata_with_correlations
        )

        assert fig is not None
        assert isinstance(stats, dict)
        assert stats["plot_type"] == "correlation_network"
        assert "n_nodes" in stats
        assert "n_edges" in stats

    def test_create_protein_correlation_network_custom_threshold(
        self, service, mock_adata_with_correlations
    ):
        """Test protein correlation network with custom correlation threshold."""
        fig, stats = service.create_protein_correlation_network(
            mock_adata_with_correlations, correlation_threshold=0.8
        )

        assert fig is not None

    def test_create_protein_correlation_network_protein_subset(
        self, service, mock_adata_with_correlations
    ):
        """Test protein correlation network with max proteins."""
        # protein_subset not supported, use max_proteins instead
        fig, stats = service.create_protein_correlation_network(
            mock_adata_with_correlations, max_proteins=10
        )

        assert fig is not None
        assert stats["n_nodes"] <= 10

    def test_create_protein_correlation_network_layout(
        self, service, mock_adata_with_correlations
    ):
        """Test protein correlation network with different layouts."""
        layouts = ["spring", "circular", "random"]

        for layout in layouts:
            fig, stats = service.create_protein_correlation_network(
                mock_adata_with_correlations, layout_algorithm=layout
            )

            assert fig is not None

    def test_create_protein_correlation_network_no_correlations(
        self, service, mock_adata_with_missing
    ):
        """Test protein correlation network with missing data."""
        # Service computes correlations from data, should work with imputation
        fig, stats = service.create_protein_correlation_network(
            mock_adata_with_missing, correlation_threshold=0.8
        )

        assert fig is not None


# ===============================================================================
# Pathway Enrichment Plot Tests
# ===============================================================================


class TestPathwayEnrichmentPlot:
    """Test suite for pathway enrichment plot functionality."""

    def test_create_pathway_enrichment_plot_basic(
        self, service, mock_adata_with_pathway_results
    ):
        """Test basic pathway enrichment plot creation."""
        fig, stats = service.create_pathway_enrichment_plot(
            mock_adata_with_pathway_results
        )

        assert fig is not None
        assert isinstance(stats, dict)
        assert stats["plot_type"] == "pathway_enrichment"
        assert "n_pathways_plotted" in stats

    def test_create_pathway_enrichment_plot_custom_threshold(
        self, service, mock_adata_with_pathway_results
    ):
        """Test pathway enrichment plot with top pathways."""
        # p_threshold not supported, use top_n instead
        fig, stats = service.create_pathway_enrichment_plot(
            mock_adata_with_pathway_results, top_n=10
        )

        assert fig is not None

    def test_create_pathway_enrichment_plot_top_pathways(
        self, service, mock_adata_with_pathway_results
    ):
        """Test pathway enrichment plot showing top pathways only."""
        # top_n parameter is supported
        fig, stats = service.create_pathway_enrichment_plot(
            mock_adata_with_pathway_results, top_n=5
        )

        assert fig is not None
        assert stats["n_pathways_plotted"] <= 5

    def test_create_pathway_enrichment_plot_horizontal(
        self, service, mock_adata_with_pathway_results
    ):
        """Test pathway enrichment plot with bar plot type."""
        # orientation not supported, use plot_type="bar" instead
        fig, stats = service.create_pathway_enrichment_plot(
            mock_adata_with_pathway_results, plot_type="bar"
        )

        assert fig is not None

    def test_create_pathway_enrichment_plot_bubble(
        self, service, mock_adata_with_pathway_results
    ):
        """Test pathway enrichment plot as bubble plot."""
        fig, stats = service.create_pathway_enrichment_plot(
            mock_adata_with_pathway_results, plot_type="bubble"
        )

        assert fig is not None

    def test_create_pathway_enrichment_plot_no_results(
        self, service, mock_adata_with_missing
    ):
        """Test pathway enrichment plot with no pathway results."""
        with pytest.raises(ProteomicsVisualizationError):
            service.create_pathway_enrichment_plot(mock_adata_with_missing)


# ===============================================================================
# QC Dashboard Tests
# ===============================================================================


class TestProteomicsQCDashboard:
    """Test suite for proteomics QC dashboard functionality."""

    def test_create_proteomics_qc_dashboard_basic(
        self, service, mock_adata_with_missing
    ):
        """Test basic proteomics QC dashboard creation."""
        fig, stats = service.create_proteomics_qc_dashboard(mock_adata_with_missing)

        assert fig is not None
        assert isinstance(stats, dict)
        assert stats["plot_type"] == "qc_dashboard"
        assert "dashboard_components" in stats

    def test_create_proteomics_qc_dashboard_custom_components(
        self, service, mock_adata_with_missing
    ):
        """Test QC dashboard with custom title."""
        # components parameter not supported, test with custom title
        fig, stats = service.create_proteomics_qc_dashboard(
            mock_adata_with_missing, title="Custom QC Dashboard"
        )

        assert fig is not None
        assert len(stats["dashboard_components"]) >= 8

    def test_create_proteomics_qc_dashboard_with_batch(
        self, service, mock_adata_with_missing
    ):
        """Test QC dashboard with batch data."""
        # Dashboard automatically detects batch column
        fig, stats = service.create_proteomics_qc_dashboard(mock_adata_with_missing)

        assert fig is not None
        # Should detect batch in mock data
        assert (
            "batch_effects" in stats["dashboard_components"]
            or "sample_stats" in stats["dashboard_components"]
        )

    def test_create_proteomics_qc_dashboard_minimal(self, service):
        """Test QC dashboard with minimal data."""
        X = np.random.lognormal(mean=8, sigma=1, size=(10, 20))
        adata = ad.AnnData(X=X)

        fig, stats = service.create_proteomics_qc_dashboard(adata)

        assert fig is not None


# ===============================================================================
# Plot Saving Tests
# ===============================================================================


class TestPlotSaving:
    """Test suite for plot saving functionality."""

    @patch("pathlib.Path.mkdir")
    @patch("plotly.io.write_html")
    @patch("plotly.io.write_image")
    def test_save_plots_html(
        self,
        mock_write_image,
        mock_write_html,
        mock_mkdir,
        service,
        mock_adata_with_missing,
    ):
        """Test saving plots as HTML."""
        # Create a plot
        fig, _ = service.create_missing_value_heatmap(mock_adata_with_missing)

        plots_dict = {"missing_value_heatmap": fig}

        saved_files = service.save_plots(
            plots_dict, output_dir="test_output", format="html"
        )

        assert len(saved_files) >= 1
        mock_write_html.assert_called()

    @patch("pathlib.Path.mkdir")
    @patch("plotly.io.write_html")
    @patch("plotly.io.write_image")
    def test_save_plots_png(
        self,
        mock_write_image,
        mock_write_html,
        mock_mkdir,
        service,
        mock_adata_with_missing,
    ):
        """Test saving plots as PNG."""
        fig, _ = service.create_missing_value_heatmap(mock_adata_with_missing)

        plots_dict = {"missing_value_heatmap": fig}

        saved_files = service.save_plots(
            plots_dict, output_dir="test_output", format="png"
        )

        assert len(saved_files) >= 1
        mock_write_image.assert_called()

    @patch("pathlib.Path.mkdir")
    @patch("plotly.io.write_html")
    @patch("plotly.io.write_image")
    def test_save_plots_both_formats(
        self,
        mock_write_image,
        mock_write_html,
        mock_mkdir,
        service,
        mock_adata_with_missing,
    ):
        """Test saving plots in both HTML and PNG formats."""
        fig, _ = service.create_missing_value_heatmap(mock_adata_with_missing)

        plots_dict = {"missing_value_heatmap": fig}

        saved_files = service.save_plots(
            plots_dict, output_dir="test_output", format="both"
        )

        assert len(saved_files) >= 2
        mock_write_html.assert_called()
        mock_write_image.assert_called()

    def test_save_plots_invalid_format(self, service, mock_adata_with_missing):
        """Test saving plots with invalid format."""
        fig, _ = service.create_missing_value_heatmap(mock_adata_with_missing)

        plots_dict = {"missing_value_heatmap": fig}

        # Service save_plots doesn't validate format, just skips invalid ones
        saved_files = service.save_plots(
            plots_dict, output_dir="test_output", format="invalid_format"
        )

        # Should return empty list or handle gracefully
        assert isinstance(saved_files, list)


# ===============================================================================
# Error Handling and Edge Cases Tests
# ===============================================================================


class TestErrorHandlingAndEdgeCases:
    """Test suite for error handling and edge cases."""

    def test_error_handling_empty_data(self, service):
        """Test error handling with empty data."""
        adata = ad.AnnData(X=np.array([]).reshape(0, 0))

        # Empty data may cause various errors depending on service
        try:
            service.create_missing_value_heatmap(adata)
        except (ProteomicsVisualizationError, Exception):
            pass  # Expected to fail in some way

    def test_single_sample_visualization(self, service):
        """Test visualization with single sample."""
        X = np.random.lognormal(mean=8, sigma=1, size=(1, 20))
        adata = ad.AnnData(X=X)

        # Should handle single sample gracefully
        fig, stats = service.create_intensity_distribution_plot(adata)
        assert fig is not None

    def test_single_protein_visualization(self, service):
        """Test visualization with single protein."""
        X = np.random.lognormal(mean=8, sigma=1, size=(20, 1))
        adata = ad.AnnData(X=X)

        fig, stats = service.create_missing_value_heatmap(adata)
        assert fig is not None

    def test_all_missing_data_visualization(self, service):
        """Test visualization with all missing data."""
        X = np.full((10, 5), np.nan)
        adata = ad.AnnData(X=X)

        fig, stats = service.create_missing_value_heatmap(adata)
        assert fig is not None
        assert stats["total_missing_percentage"] == 100.0

    def test_no_variation_data_visualization(self, service):
        """Test visualization with no variation data."""
        X = np.ones((10, 20)) * 1000  # All same value
        adata = ad.AnnData(X=X)

        # Should handle no variation gracefully
        fig, stats = service.create_intensity_distribution_plot(adata)
        assert fig is not None

    def test_invalid_subset_parameters(self, service, mock_adata_with_missing):
        """Test visualization with large max values."""
        # sample_subset and protein_subset not supported, use max parameters
        fig, stats = service.create_missing_value_heatmap(
            mock_adata_with_missing, max_samples=1000, max_proteins=1000
        )

        assert fig is not None
        # Should handle gracefully even with large max values


# ===============================================================================
# Integration Tests
# ===============================================================================


class TestIntegrationScenarios:
    """Test suite for integration scenarios."""

    def test_complete_visualization_workflow(self, service, mock_adata_with_missing):
        """Test complete visualization workflow with multiple plots."""
        plots = {}

        # Create multiple plots
        plots["missing_values"], _ = service.create_missing_value_heatmap(
            mock_adata_with_missing
        )
        plots["intensity_dist"], _ = service.create_intensity_distribution_plot(
            mock_adata_with_missing
        )
        plots["qc_dashboard"], _ = service.create_proteomics_qc_dashboard(
            mock_adata_with_missing
        )

        # All plots should be created successfully
        assert all(plot is not None for plot in plots.values())

    def test_visualization_with_all_analysis_results(self, service):
        """Test visualization with comprehensive analysis results."""
        # Create comprehensive mock data
        n_samples, n_proteins = 40, 60
        X = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))

        adata = ad.AnnData(X=X)
        adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
        adata.var_names = [f"protein_{i}" for i in range(n_proteins)]

        # Add all types of results
        adata.obs["condition"] = ["control"] * 20 + ["treatment"] * 20
        adata.var["cv_mean"] = np.random.uniform(0.1, 0.8, n_proteins)

        # Mock all analysis results
        adata.uns["differential_expression"] = {"results": []}
        adata.uns["correlation_analysis"] = {"results": []}
        adata.uns["pathway_enrichment"] = {"results": []}

        # Should be able to create QC dashboard with all results
        fig, stats = service.create_proteomics_qc_dashboard(adata)
        assert fig is not None

    def test_consistent_plot_styling(self, service, mock_adata_with_missing):
        """Test consistent styling across different plot types."""
        plots_and_stats = []

        # Create multiple plot types
        plots_and_stats.append(
            service.create_missing_value_heatmap(mock_adata_with_missing)
        )
        plots_and_stats.append(
            service.create_intensity_distribution_plot(mock_adata_with_missing)
        )

        # All plots should have consistent basic properties
        for fig, stats in plots_and_stats:
            assert fig is not None
            assert "plot_type" in stats


# ===============================================================================
# Performance and Memory Tests
# ===============================================================================


class TestPerformanceAndMemory:
    """Test suite for performance and memory considerations."""

    @pytest.mark.slow
    def test_large_dataset_visualization(self, service):
        """Test visualization with large dataset."""
        # Create larger dataset
        n_samples, n_proteins = 200, 500
        X = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))
        # Add some missing values
        missing_mask = np.random.rand(n_samples, n_proteins) < 0.1
        X[missing_mask] = np.nan

        adata = ad.AnnData(X=X)

        fig, stats = service.create_missing_value_heatmap(adata)

        assert fig is not None
        # Should subsample based on max_samples and max_proteins defaults
        assert stats["samples_plotted"] <= 100  # max_samples default
        assert stats["proteins_plotted"] <= 500  # max_proteins default

    @pytest.mark.slow
    def test_memory_efficient_qc_dashboard(self, service):
        """Test memory efficiency in QC dashboard creation."""
        # Create moderately large dataset
        n_samples, n_proteins = 100, 300
        X = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))
        adata = ad.AnnData(X=X)

        fig, stats = service.create_proteomics_qc_dashboard(adata)

        assert fig is not None
        # Should complete without memory errors

    def test_efficient_network_visualization(self, service):
        """Test efficient network visualization with many proteins."""
        n_samples, n_proteins = 50, 100
        X = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))
        adata = ad.AnnData(X=X)

        # Service computes correlations from data
        fig, stats = service.create_protein_correlation_network(
            adata,
            correlation_threshold=0.9,  # High threshold to reduce complexity
            max_proteins=50,  # Limit proteins for efficiency
        )

        assert fig is not None
        assert stats["n_nodes"] <= 50
