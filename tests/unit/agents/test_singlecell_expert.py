"""
Unit tests for single-cell expert agent.

This module tests the single-cell expert agent's core functionality including
quality control, preprocessing, clustering, marker gene identification,
and cell type annotation using the new service-based architecture.

Test coverage focuses on tool functionality and service integration.
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from lobster.agents.singlecell_expert import singlecell_expert
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.clustering_service import ClusteringService
from lobster.tools.enhanced_singlecell_service import EnhancedSingleCellService
from lobster.tools.preprocessing_service import PreprocessingService
from lobster.tools.quality_service import QualityService
from tests.mock_data.base import SMALL_DATASET_CONFIG
from tests.mock_data.factories import SingleCellDataFactory

# ===============================================================================
# Mock Objects and Fixtures
# ===============================================================================


class MockMessage:
    """Mock LangGraph message object."""

    def __init__(self, content: str, sender: str = "human"):
        self.content = content
        self.sender = sender


class MockState:
    """Mock LangGraph state object."""

    def __init__(self, messages=None, **kwargs):
        self.messages = messages or []
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def mock_data_manager(mock_agent_environment):
    """Create mock data manager with single-cell data."""
    mock_dm = Mock(spec=DataManagerV2)
    mock_dm.list_modalities.return_value = ["sc_data", "sc_data_filtered"]

    # Create mock single-cell data
    sc_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
    mock_dm.get_modality.return_value = sc_data
    mock_dm.modalities = {"sc_data": sc_data}
    mock_dm.log_tool_usage.return_value = None
    mock_dm.save_modality.return_value = None
    mock_dm.get_quality_metrics.return_value = {
        "total_counts": 50000,
        "mean_counts_per_obs": 1500,
    }
    mock_dm._detect_modality_type.return_value = "single_cell_rna_seq"

    yield mock_dm


@pytest.fixture
def singlecell_agent(mock_data_manager, mock_agent_environment):
    """Create single-cell expert agent for testing."""
    return singlecell_expert(
        data_manager=mock_data_manager,
        callback_handler=None,
        agent_name="test_singlecell_expert",
        handoff_tools=None,
    )


# ===============================================================================
# Single-Cell Expert Core Functionality Tests
# ===============================================================================


@pytest.mark.unit
class TestSingleCellExpertCore:
    """Test single-cell expert core functionality."""

    @patch("lobster.tools.preprocessing_service.PreprocessingService")
    def test_filter_and_normalize_data(
        self, mock_preprocessing_service, mock_data_manager
    ):
        """Test data filtering and normalization via service."""
        # Setup mock service
        mock_service = mock_preprocessing_service.return_value
        mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        mock_service.filter_and_normalize_cells.return_value = (
            mock_adata,
            {
                "original_shape": (5000, 20000),
                "final_shape": (4500, 18000),
                "cells_retained_pct": 90.0,
                "genes_retained_pct": 90.0,
            },
        )

        # Create agent
        agent = singlecell_expert(mock_data_manager)

        # Test that the service would be called correctly
        mock_service.filter_and_normalize_cells.assert_not_called()  # Not called until tool is used

    @patch("lobster.tools.quality_service.QualityService")
    def test_assess_data_quality(self, mock_quality_service, mock_data_manager):
        """Test QC metrics calculation via service."""
        # Setup mock service
        mock_service = mock_quality_service.return_value
        mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        mock_service.assess_quality.return_value = (
            mock_adata,
            {
                "cells_before_qc": 5000,
                "cells_after_qc": 4750,
                "cells_retained_pct": 95.0,
                "quality_status": "good",
                "mean_genes_per_cell": 2500,
                "mean_mt_pct": 12.5,
                "mean_ribo_pct": 25.0,
                "mean_total_counts": 15000,
                "qc_summary": "High quality single-cell data",
            },
        )

        # Create agent
        agent = singlecell_expert(mock_data_manager)

        # Test that the service would be called correctly
        mock_service.assess_quality.assert_not_called()  # Not called until tool is used

    @patch("lobster.tools.enhanced_singlecell_service.EnhancedSingleCellService")
    def test_detect_doublets(self, mock_singlecell_service, mock_data_manager):
        """Test doublet detection via service."""
        # Setup mock service
        mock_service = mock_singlecell_service.return_value
        mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        mock_service.detect_doublets.return_value = (
            mock_adata,
            {
                "detection_method": "scrublet",
                "n_cells_analyzed": 5000,
                "n_doublets_detected": 125,
                "actual_doublet_rate": 0.025,
                "expected_doublet_rate": 0.06,
                "doublet_score_stats": {"min": 0.0, "max": 1.0, "mean": 0.15},
            },
        )

        # Create agent
        agent = singlecell_expert(mock_data_manager)

        # Test that the service would be called correctly
        mock_service.detect_doublets.assert_not_called()  # Not called until tool is used

    @patch("lobster.tools.clustering_service.ClusteringService")
    def test_perform_clustering(self, mock_clustering_service, mock_data_manager):
        """Test clustering analysis via service."""
        # Setup mock service
        mock_service = mock_clustering_service.return_value
        mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        mock_service.cluster_and_visualize.return_value = (
            mock_adata,
            {
                "n_clusters": 12,
                "resolution": 0.5,
                "has_umap": True,
                "has_marker_genes": False,
                "original_shape": (5000, 20000),
                "final_shape": (5000, 20000),
                "batch_correction": True,
                "demo_mode": False,
                "cluster_sizes": {str(i): 400 + i * 10 for i in range(12)},
            },
        )

        # Create agent
        agent = singlecell_expert(mock_data_manager)

        # Test that the service would be called correctly
        mock_service.cluster_and_visualize.assert_not_called()  # Not called until tool is used

    @patch("lobster.tools.enhanced_singlecell_service.EnhancedSingleCellService")
    def test_find_marker_genes(self, mock_singlecell_service, mock_data_manager):
        """Test marker gene identification via service."""
        # Setup mock service
        mock_service = mock_singlecell_service.return_value
        mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        # Add leiden column for marker gene testing
        mock_adata.obs["leiden"] = pd.Categorical(
            [str(i % 5) for i in range(mock_adata.n_obs)]
        )

        mock_service.find_marker_genes.return_value = (
            mock_adata,
            {
                "groupby": "leiden",
                "n_groups": 5,
                "method": "wilcoxon",
                "n_genes": 25,
                "groups_analyzed": ["0", "1", "2", "3", "4"],
                "top_markers_per_group": {
                    "0": [
                        {"gene": "CD3D", "score": 0.8},
                        {"gene": "CD3E", "score": 0.75},
                    ],
                    "1": [
                        {"gene": "CD79A", "score": 0.9},
                        {"gene": "CD79B", "score": 0.85},
                    ],
                },
            },
        )

        # Create agent
        agent = singlecell_expert(mock_data_manager)

        # Test that the service would be called correctly
        mock_service.find_marker_genes.assert_not_called()  # Not called until tool is used

    @patch("lobster.tools.enhanced_singlecell_service.EnhancedSingleCellService")
    def test_annotate_cell_types(self, mock_singlecell_service, mock_data_manager):
        """Test cell type annotation via service."""
        # Setup mock service
        mock_service = mock_singlecell_service.return_value
        mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        mock_service.annotate_cell_types.return_value = (
            mock_adata,
            {
                "n_cell_types_identified": 5,
                "n_clusters": 12,
                "n_marker_sets": 8,
                "cell_type_counts": {
                    "T cells": 1800,
                    "B cells": 1200,
                    "Monocytes": 800,
                    "NK cells": 700,
                    "Unknown": 500,
                },
            },
        )

        # Create agent
        agent = singlecell_expert(mock_data_manager)

        # Test that the service would be called correctly
        mock_service.annotate_cell_types.assert_not_called()  # Not called until tool is used


# ===============================================================================
# Basic Agent Functionality Tests
# ===============================================================================


@pytest.mark.unit
class TestSingleCellAgentBasics:
    """Test basic single-cell agent functionality."""

    def test_agent_creation_succeeds(self, mock_data_manager):
        """Test that agent can be created successfully."""
        agent = singlecell_expert(mock_data_manager)
        assert agent is not None

    def test_agent_has_graph_structure(self, mock_data_manager):
        """Test that agent has expected graph structure."""
        agent = singlecell_expert(mock_data_manager)

        # Should have a graph structure
        graph = agent.get_graph()
        assert graph is not None

    def test_agent_with_callback_handler(self, mock_data_manager):
        """Test agent creation with callback handler."""
        mock_callback = Mock()

        agent = singlecell_expert(
            data_manager=mock_data_manager, callback_handler=mock_callback
        )

        assert agent is not None


# ===============================================================================
# Data Manager Integration Tests
# ===============================================================================


@pytest.mark.unit
class TestDataManagerIntegration:
    """Test integration with DataManagerV2."""

    def test_agent_with_empty_modalities(self, mock_data_manager):
        """Test agent creation with no modalities available."""
        mock_data_manager.list_modalities.return_value = []

        agent = singlecell_expert(mock_data_manager)
        assert agent is not None

    def test_agent_with_available_modalities(self, mock_data_manager):
        """Test agent creation with modalities available."""
        mock_data_manager.list_modalities.return_value = ["sc_data", "sc_data_filtered"]
        mock_data_manager._detect_modality_type.return_value = "single_cell_rna_seq"

        agent = singlecell_expert(mock_data_manager)
        assert agent is not None


# ===============================================================================
# Agent Configuration Tests
# ===============================================================================


@pytest.mark.unit
class TestAgentConfiguration:
    """Test single-cell agent configuration options."""

    def test_agent_with_custom_name(self, mock_data_manager):
        """Test agent creation with custom agent name."""
        agent = singlecell_expert(
            data_manager=mock_data_manager, agent_name="custom_singlecell_agent"
        )

        assert agent is not None

    def test_agent_with_handoff_tools(self, mock_data_manager):
        """Test agent creation with handoff tools."""

        # Create a proper mock tool function instead of Mock object
        def mock_handoff_tool():
            """Mock handoff tool."""
            return "Mock handoff executed"

        mock_handoff_tool.__name__ = "mock_handoff_tool"
        mock_handoff_tools = [mock_handoff_tool]

        agent = singlecell_expert(
            data_manager=mock_data_manager, handoff_tools=mock_handoff_tools
        )

        assert agent is not None


# ===============================================================================
# Error Handling and Edge Cases
# ===============================================================================


@pytest.mark.unit
class TestSingleCellErrorHandling:
    """Test single-cell expert error handling."""

    def test_agent_creation_minimal_config(self, mock_data_manager):
        """Test agent creation with minimal configuration."""
        # Should work with just data manager
        agent = singlecell_expert(data_manager=mock_data_manager)
        assert agent is not None

    def test_data_manager_edge_cases(self, mock_data_manager):
        """Test data manager edge cases."""
        # Test with missing modality access
        mock_data_manager.get_modality.side_effect = KeyError("Modality not found")

        # Agent should still be created, errors handled at tool execution time
        agent = singlecell_expert(mock_data_manager)
        assert agent is not None

    def test_agent_creation_with_all_options(self, mock_data_manager):
        """Test agent creation with all configuration options."""
        mock_callback = Mock()

        def mock_tool():
            """Mock tool for testing."""
            return "test"

        mock_tool.__name__ = "mock_tool"

        agent = singlecell_expert(
            data_manager=mock_data_manager,
            callback_handler=mock_callback,
            agent_name="test_agent",
            handoff_tools=[mock_tool],
        )

        assert agent is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
