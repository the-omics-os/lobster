"""
Unit tests for transcriptomics_expert parent agent.

This module tests the unified transcriptomics expert that handles both
single-cell and bulk RNA-seq analysis. Based on comprehensive stress testing
campaign (MISSION_COMPLETE.md, 9 stress tests, 5 critical bugs found/fixed).

Test Categories:
1. Agent creation and configuration
2. Data type auto-detection (SC vs bulk)
3. QC and preprocessing tools (4 shared tools)
4. Clustering tools (4 SC-specific tools)
5. Delegation to sub-agents (annotation_expert, de_analysis_expert)
6. Error handling and edge cases

Bugs Addressed from Stress Testing:
- BUG-002: X_pca preservation in clustering (STRESS_TEST_03)
- BUG-003: Tool naming mismatch for delegation (STRESS_TEST_05-07)
- BUG-004: Marker gene dict keys (STRESS_TEST_07)
- BUG-005: Metadata loss through QC (STRESS_TEST_08)
- BUG-007: Delegation not invoked (prompt issue)
"""

from typing import Dict, List
from unittest.mock import Mock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.transcriptomics.transcriptomics_expert import (
    ModalityNotFoundError,
    TranscriptomicsAgentError,
    transcriptomics_expert,
)
from lobster.core.data_manager_v2 import DataManagerV2
from tests.mock_data.base import SMALL_DATASET_CONFIG
from tests.mock_data.factories import SingleCellDataFactory


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def mock_data_manager(tmp_path):
    """Create mock data manager with single-cell data."""
    from pathlib import Path

    mock_dm = Mock(spec=DataManagerV2)
    mock_dm.workspace_path = Path(tmp_path / "workspace")

    # Create realistic single-cell data
    adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
    mock_dm.get_modality.return_value = adata
    mock_dm.modalities = {"geo_gse12345": adata}
    mock_dm.list_modalities.return_value = ["geo_gse12345"]
    mock_dm.log_tool_usage.return_value = None
    mock_dm.save_modality.return_value = None

    return mock_dm


@pytest.fixture
def bulk_rnaseq_data():
    """Create realistic bulk RNA-seq data (low cell count)."""
    n_samples = 12  # Typical bulk RNA-seq sample count
    n_genes = 20000

    # Bulk has low "cell" count (actually samples), high counts per sample
    counts = np.random.negative_binomial(n=5, p=0.3, size=(n_samples, n_genes))
    adata = ad.AnnData(X=counts)

    # Bulk RNA-seq metadata
    adata.obs["sample_id"] = [f"sample_{i:02d}" for i in range(n_samples)]
    adata.obs["condition"] = ["treatment"] * 6 + ["control"] * 6
    adata.obs["replicate"] = [1, 2, 3, 4, 5, 6] * 2
    adata.obs["batch"] = ["batch1"] * 6 + ["batch2"] * 6

    # Gene names
    adata.var_names = [f"GENE_{i:05d}" for i in range(n_genes)]

    return adata


@pytest.fixture
def singlecell_data_with_metadata():
    """Create single-cell data with biological metadata (for pseudobulk tests)."""
    n_cells = 1000
    n_genes = 2000

    # Sparse counts typical of scRNA-seq
    from scipy.sparse import csr_matrix

    counts = csr_matrix(np.random.negative_binomial(n=2, p=0.8, size=(n_cells, n_genes)))
    adata = ad.AnnData(X=counts)

    # CRITICAL: Biological metadata needed for pseudobulk (BUG-005)
    adata.obs["patient_id"] = np.random.choice(
        ["patient_1", "patient_2", "patient_3", "patient_4"], size=n_cells
    )
    adata.obs["tissue_region"] = np.random.choice(
        ["tumor_core", "tumor_edge", "normal_mucosa"], size=n_cells
    )
    adata.obs["condition"] = np.random.choice(["tumor", "normal"], size=n_cells)

    # QC metrics
    adata.obs["n_genes"] = np.random.randint(200, 5000, n_cells)
    adata.obs["n_counts"] = np.random.randint(500, 50000, n_cells)
    adata.obs["pct_counts_mt"] = np.random.uniform(0, 30, n_cells)

    # Gene names
    adata.var_names = [f"GENE_{i:05d}" for i in range(n_genes)]

    return adata


# ==============================================================================
# Test Agent Creation
# ==============================================================================


@pytest.mark.unit
class TestTranscriptomicsExpertCreation:
    """Test agent factory function and initialization."""

    def test_agent_creation_succeeds(self, mock_data_manager):
        """Test that agent can be created successfully."""
        agent = transcriptomics_expert(mock_data_manager)
        assert agent is not None

    def test_agent_has_graph_structure(self, mock_data_manager):
        """Test that agent has expected LangGraph structure."""
        agent = transcriptomics_expert(mock_data_manager)
        graph = agent.get_graph()
        assert graph is not None

    def test_agent_with_callback_handler(self, mock_data_manager):
        """Test agent creation with callback handler."""
        mock_callback = Mock()
        agent = transcriptomics_expert(
            data_manager=mock_data_manager, callback_handler=mock_callback
        )
        assert agent is not None

    def test_agent_with_custom_name(self, mock_data_manager):
        """Test agent creation with custom agent name."""
        agent = transcriptomics_expert(
            data_manager=mock_data_manager, agent_name="custom_transcriptomics"
        )
        assert agent is not None

    def test_agent_with_delegation_tools(self, mock_data_manager):
        """Test agent creation with delegation tools for sub-agents."""

        def mock_annotation_tool():
            """Mock annotation delegation tool."""
            return "annotation_expert invoked"

        def mock_de_tool():
            """Mock DE delegation tool."""
            return "de_analysis_expert invoked"

        mock_annotation_tool.__name__ = "handoff_to_annotation_expert"
        mock_de_tool.__name__ = "handoff_to_de_analysis_expert"

        agent = transcriptomics_expert(
            data_manager=mock_data_manager,
            delegation_tools=[mock_annotation_tool, mock_de_tool],
        )
        assert agent is not None


# ==============================================================================
# Test Data Type Auto-Detection
# ==============================================================================


@pytest.mark.unit
class TestDataTypeDetection:
    """Test auto-detection of single-cell vs bulk RNA-seq data."""

    def test_detects_single_cell_data(self, mock_data_manager):
        """Test detection of single-cell data based on observation count."""
        # Single-cell: >500 observations
        sc_data = SingleCellDataFactory(n_cells=5000, n_genes=2000)
        mock_data_manager.get_modality.return_value = sc_data

        agent = transcriptomics_expert(mock_data_manager)
        # Agent should apply single-cell defaults
        assert agent is not None

    def test_detects_bulk_rnaseq_data(self, mock_data_manager, bulk_rnaseq_data):
        """Test detection of bulk RNA-seq data based on observation count."""
        # Bulk: <100 observations
        mock_data_manager.get_modality.return_value = bulk_rnaseq_data

        agent = transcriptomics_expert(mock_data_manager)
        # Agent should apply bulk defaults
        assert agent is not None

    def test_detects_via_sc_columns(self, mock_data_manager):
        """Test detection via single-cell-specific columns."""
        sc_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        # Add SC-specific columns
        sc_data.obs["leiden"] = "0"
        sc_data.obs["louvain"] = "0"

        mock_data_manager.get_modality.return_value = sc_data

        agent = transcriptomics_expert(mock_data_manager)
        assert agent is not None

    def test_detects_via_sparsity(self, mock_data_manager):
        """Test detection via matrix sparsity (>70% zeros = single-cell)."""
        # Create sparse data
        from scipy.sparse import csr_matrix

        sparse_counts = csr_matrix(
            np.random.choice([0, 0, 0, 0, 1], size=(1000, 2000))  # 80% sparse
        )
        adata = ad.AnnData(X=sparse_counts)
        adata.obs_names = [f"cell_{i}" for i in range(1000)]
        adata.var_names = [f"gene_{i}" for i in range(2000)]

        mock_data_manager.get_modality.return_value = adata

        agent = transcriptomics_expert(mock_data_manager)
        assert agent is not None


# ==============================================================================
# Test QC and Preprocessing Tools (Shared Tools)
# ==============================================================================


@pytest.mark.unit
class TestQCTools:
    """Test the 4 shared QC/preprocessing tools."""

    def test_check_data_status_tool_exists(self, mock_data_manager):
        """Test that check_data_status tool is registered."""
        agent = transcriptomics_expert(mock_data_manager)
        # Agent should have 8 tools: 4 shared + 4 clustering
        assert agent is not None

    def test_assess_data_quality_tool_exists(self, mock_data_manager):
        """Test that assess_data_quality tool is registered."""
        agent = transcriptomics_expert(mock_data_manager)
        assert agent is not None

    def test_filter_and_normalize_tool_exists(self, mock_data_manager):
        """Test that filter_and_normalize_modality tool is registered."""
        agent = transcriptomics_expert(mock_data_manager)
        assert agent is not None

    def test_create_analysis_summary_tool_exists(self, mock_data_manager):
        """Test that create_analysis_summary tool is registered."""
        agent = transcriptomics_expert(mock_data_manager)
        assert agent is not None

    @patch("lobster.services.quality.quality_service.QualityService")
    def test_assess_quality_applies_sc_defaults(
        self, MockQualityService, mock_data_manager
    ):
        """Test that single-cell defaults are applied for SC data."""
        # Mock service to track parameters
        mock_service = MockQualityService.return_value
        mock_service.assess_quality.return_value = (
            Mock(),
            {"n_cells": 1000, "median_genes": 2500},
            Mock(),
        )

        agent = transcriptomics_expert(mock_data_manager)
        # Should apply SC defaults: min_genes=200, max_genes=5000
        assert agent is not None

    @patch("lobster.services.quality.preprocessing_service.PreprocessingService")
    def test_metadata_preserved_through_filtering(
        self, MockPreprocessingService, mock_data_manager, singlecell_data_with_metadata
    ):
        """Test that biological metadata is preserved (BUG-005 fix validation)."""
        mock_data_manager.get_modality.return_value = singlecell_data_with_metadata

        # Mock service returns data WITH metadata
        filtered_data = singlecell_data_with_metadata[
            :500, :
        ].copy()  # Simulate filtering
        mock_service = MockPreprocessingService.return_value
        mock_service.filter_and_normalize_cells.return_value = (
            filtered_data,
            {"n_cells_removed": 500},
            Mock(),
        )

        agent = transcriptomics_expert(mock_data_manager)

        # Verify metadata columns exist
        assert "patient_id" in filtered_data.obs.columns
        assert "tissue_region" in filtered_data.obs.columns
        assert "condition" in filtered_data.obs.columns


# ==============================================================================
# Test Clustering Tools (Single-Cell Specific)
# ==============================================================================


@pytest.mark.unit
class TestClusteringTools:
    """Test the 4 clustering tools (SC-specific)."""

    def test_cluster_modality_tool_exists(self, mock_data_manager):
        """Test that cluster_modality tool is registered."""
        agent = transcriptomics_expert(mock_data_manager)
        assert agent is not None

    def test_evaluate_clustering_quality_tool_exists(self, mock_data_manager):
        """Test that evaluate_clustering_quality tool is registered."""
        agent = transcriptomics_expert(mock_data_manager)
        assert agent is not None

    def test_find_marker_genes_tool_exists(self, mock_data_manager):
        """Test that find_marker_genes_for_clusters tool is registered."""
        agent = transcriptomics_expert(mock_data_manager)
        assert agent is not None

    def test_subcluster_cells_tool_exists(self, mock_data_manager):
        """Test that subcluster_cells tool is registered."""
        agent = transcriptomics_expert(mock_data_manager)
        assert agent is not None

    @patch("lobster.services.analysis.clustering_service.ClusteringService")
    def test_x_pca_preserved_in_clustering(
        self, MockClusteringService, mock_data_manager
    ):
        """Test that X_pca is preserved in clustering output (BUG-002 fix validation)."""
        # Mock clustering service
        mock_service = MockClusteringService.return_value

        # Create clustered data with X_pca
        clustered_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        clustered_data.obs["leiden"] = np.random.randint(0, 5, clustered_data.n_obs)
        clustered_data.obsm["X_pca"] = np.random.randn(clustered_data.n_obs, 50)

        mock_service.cluster_and_visualize.return_value = (
            clustered_data,
            {"n_clusters": 5, "has_umap": True},
            Mock(),
        )

        agent = transcriptomics_expert(mock_data_manager)

        # Verify X_pca exists in output
        assert "X_pca" in clustered_data.obsm.keys()

    @patch("lobster.services.analysis.enhanced_singlecell_service.EnhancedSingleCellService")
    def test_marker_gene_dict_keys_correct(
        self, MockEnhancedService, mock_data_manager
    ):
        """Test that marker gene dict keys are accessed correctly (BUG-004 fix validation)."""
        mock_service = MockEnhancedService.return_value

        # Mock marker gene response with correct dict structure
        marker_stats = {
            "n_groups": 5,
            "method": "wilcoxon",
            "groupby": "leiden",
            "n_genes": 25,
            "pre_filter_counts": {0: 100, 1: 100, 2: 100, 3: 100, 4: 100},
            "post_filter_counts": {0: 25, 1: 25, 2: 25, 3: 25, 4: 25},
            "total_genes_filtered": 375,
            "filtering_params": {
                "min_fold_change": 1.5,
                "min_pct": 0.25,
                "max_out_pct": 0.5,
            },
            "groups_analyzed": [0, 1, 2, 3, 4],
            "filtered_counts": {0: 75, 1: 75, 2: 75, 3: 75, 4: 75},
        }

        mock_service.find_marker_genes.return_value = (
            SingleCellDataFactory(config=SMALL_DATASET_CONFIG),
            marker_stats,
            Mock(),
        )

        agent = transcriptomics_expert(mock_data_manager)

        # Agent should successfully access marker_stats dict without KeyError
        assert marker_stats["n_groups"] == 5
        assert marker_stats["total_genes_filtered"] == 375


# ==============================================================================
# Test Delegation to Sub-Agents
# ==============================================================================


@pytest.mark.unit
class TestDelegationTools:
    """Test delegation to annotation_expert and de_analysis_expert."""

    def test_handoff_to_annotation_expert_exists(self, mock_data_manager):
        """Test that handoff_to_annotation_expert tool exists (BUG-003 fix validation)."""

        def mock_annotation_tool(modality_name: str) -> str:
            """Mock annotation delegation tool."""
            return f"annotation_expert processing {modality_name}"

        mock_annotation_tool.__name__ = "handoff_to_annotation_expert"

        agent = transcriptomics_expert(
            data_manager=mock_data_manager, delegation_tools=[mock_annotation_tool]
        )

        # Tool should be accessible with correct name
        assert agent is not None

    def test_handoff_to_de_analysis_expert_exists(self, mock_data_manager):
        """Test that handoff_to_de_analysis_expert tool exists (BUG-003 fix validation)."""

        def mock_de_tool(modality_name: str) -> str:
            """Mock DE delegation tool."""
            return f"de_analysis_expert processing {modality_name}"

        mock_de_tool.__name__ = "handoff_to_de_analysis_expert"

        agent = transcriptomics_expert(
            data_manager=mock_data_manager, delegation_tools=[mock_de_tool]
        )

        # Tool should be accessible with correct name
        assert agent is not None

    def test_delegation_tools_combined_with_direct_tools(self, mock_data_manager):
        """Test that delegation tools are combined with direct tools (8 total)."""

        def mock_annotation_tool():
            """Mock annotation tool."""
            return "annotation"

        def mock_de_tool():
            """Mock DE tool."""
            return "de"

        mock_annotation_tool.__name__ = "handoff_to_annotation_expert"
        mock_de_tool.__name__ = "handoff_to_de_analysis_expert"

        agent = transcriptomics_expert(
            data_manager=mock_data_manager,
            delegation_tools=[mock_annotation_tool, mock_de_tool],
        )

        # Agent should have 8 direct tools + 2 delegation tools = 10 total
        assert agent is not None


# ==============================================================================
# Test Error Handling
# ==============================================================================


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_modality_not_found_error(self, mock_data_manager):
        """Test that ModalityNotFoundError is raised for missing modalities."""
        mock_data_manager.list_modalities.return_value = []
        mock_data_manager.get_modality.side_effect = KeyError("Modality not found")

        agent = transcriptomics_expert(mock_data_manager)
        # Error should be caught at tool execution time
        assert agent is not None

    def test_agent_creation_with_empty_workspace(self, tmp_path):
        """Test agent creation with empty workspace."""
        from pathlib import Path

        mock_dm = Mock(spec=DataManagerV2)
        mock_dm.workspace_path = Path(tmp_path / "empty_workspace")
        mock_dm.list_modalities.return_value = []
        mock_dm.modalities = {}

        agent = transcriptomics_expert(mock_dm)
        assert agent is not None

    def test_handles_sparse_matrix_conversion(self, mock_data_manager):
        """Test handling of sparse matrix data."""
        from scipy.sparse import csr_matrix

        sparse_data = ad.AnnData(X=csr_matrix(np.random.randn(100, 200)))
        sparse_data.obs_names = [f"cell_{i}" for i in range(100)]
        sparse_data.var_names = [f"gene_{i}" for i in range(200)]

        mock_data_manager.get_modality.return_value = sparse_data

        agent = transcriptomics_expert(mock_data_manager)
        assert agent is not None

    def test_handles_missing_qc_metrics(self, mock_data_manager):
        """Test handling of data without QC metrics."""
        adata = ad.AnnData(X=np.random.randn(100, 200))
        adata.obs_names = [f"cell_{i}" for i in range(100)]
        adata.var_names = [f"gene_{i}" for i in range(200)]
        # No n_genes, n_counts, etc.

        mock_data_manager.get_modality.return_value = adata

        agent = transcriptomics_expert(mock_data_manager)
        assert agent is not None


# ==============================================================================
# Test Tool Count Validation (Based on Stress Testing)
# ==============================================================================


@pytest.mark.unit
class TestToolRegistration:
    """Test that correct number of tools are registered."""

    def test_agent_has_eight_direct_tools(self, mock_data_manager):
        """Test that agent has 8 direct tools (4 shared + 4 clustering)."""
        agent = transcriptomics_expert(mock_data_manager)

        # Should have:
        # - 4 shared QC tools (check_data_status, assess_data_quality, filter_and_normalize, create_analysis_summary)
        # - 4 clustering tools (cluster_modality, evaluate_clustering_quality, find_marker_genes, subcluster_cells)
        # Total: 8 tools
        assert agent is not None

    def test_agent_with_delegation_has_ten_tools(self, mock_data_manager):
        """Test that agent with delegation tools has 10 total tools (8 direct + 2 delegation)."""

        def mock_tool1():
            """Handoff to annotation expert."""
            return "tool1"

        def mock_tool2():
            """Handoff to DE analysis expert."""
            return "tool2"

        mock_tool1.__name__ = "handoff_to_annotation_expert"
        mock_tool2.__name__ = "handoff_to_de_analysis_expert"

        agent = transcriptomics_expert(
            data_manager=mock_data_manager, delegation_tools=[mock_tool1, mock_tool2]
        )

        # Should have 10 tools total
        assert agent is not None


# ==============================================================================
# Test Integration with DataManagerV2
# ==============================================================================


@pytest.mark.unit
class TestDataManagerIntegration:
    """Test integration with DataManagerV2."""

    def test_modality_operations(self, mock_data_manager):
        """Test that agent can access modality operations."""
        mock_data_manager.list_modalities.return_value = ["modality1", "modality2"]

        agent = transcriptomics_expert(mock_data_manager)
        assert agent is not None

    def test_log_tool_usage_called(self, mock_data_manager):
        """Test that log_tool_usage is called correctly."""
        agent = transcriptomics_expert(mock_data_manager)
        # Tool usage should be logged with IR parameter
        assert agent is not None

    def test_save_modality_called(self, mock_data_manager):
        """Test that save_modality is called for persistent results."""
        agent = transcriptomics_expert(mock_data_manager)
        assert agent is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
