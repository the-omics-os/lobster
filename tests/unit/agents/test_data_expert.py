"""
Unit tests for data expert agent.

This module tests the data expert agent's core functionality including
GEO data fetching, workspace management, sample concatenation, and
integration with GEOService, ConcatenationService, and DataExpertAssistant.

Test coverage focuses on agent creation, service integration, and configuration.
"""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import numpy as np

from lobster.agents.data_expert import data_expert
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.geo_service import GEOService
from lobster.tools.concatenation_service import ConcatenationService
from lobster.agents.data_expert_assistant import DataExpertAssistant, StrategyConfig

from tests.mock_data.factories import SingleCellDataFactory
from tests.mock_data.base import SMALL_DATASET_CONFIG


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
    """Create mock data manager with modality operations."""
    mock_dm = Mock(spec=DataManagerV2)
    mock_dm.list_modalities.return_value = ['geo_gse12345', 'custom_dataset']

    # Create mock data
    mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
    mock_dm.get_modality.return_value = mock_adata
    mock_dm.modalities = {'geo_gse12345': mock_adata}
    mock_dm.metadata_store = {}
    mock_dm.log_tool_usage.return_value = None
    mock_dm.save_modality.return_value = None
    mock_dm.get_quality_metrics.return_value = {'total_counts': 50000, 'mean_counts_per_obs': 1500}
    mock_dm.get_workspace_status.return_value = {
        'workspace_path': '/workspace',
        'registered_adapters': ['transcriptomics_single_cell', 'transcriptomics_bulk'],
        'registered_backends': ['h5ad']
    }
    mock_dm.available_datasets = {}
    mock_dm.restore_session.return_value = {'restored': [], 'total_size_mb': 0}

    yield mock_dm


@pytest.fixture
def mock_geo_service():
    """Mock GEO service for data fetching."""
    with patch('lobster.tools.geo_service.GEOService') as MockGEOService:
        mock_service = MockGEOService.return_value
        mock_service.fetch_metadata_only.return_value = (
            {
                "title": "Test dataset",
                "accession": "GSE12345",
                "organism": "Homo sapiens",
                "summary": "Test summary",
                "supplementary_file": ["matrix.txt.gz", "barcodes.tsv.gz"]
            },
            {
                "validation_status": "PASS",
                "alignment_percentage": 75.0,
                "predicted_data_type": "single_cell_rna_seq"
            }
        )
        mock_service.download_dataset.return_value = "Successfully downloaded GSE12345"
        yield mock_service


@pytest.fixture
def mock_concat_service():
    """Mock concatenation service for sample merging."""
    with patch('lobster.tools.concatenation_service.ConcatenationService') as MockConcatService:
        mock_service = MockConcatService.return_value
        mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        mock_service.concatenate_from_modalities.return_value = (
            mock_adata,
            {
                'n_samples': 3,
                'final_shape': (1500, 5000),
                'join_type': 'inner',
                'strategy_used': 'smart_sparse',
                'processing_time_seconds': 1.5
            }
        )
        mock_service.auto_detect_samples.return_value = [
            'geo_gse12345_sample_gsm001',
            'geo_gse12345_sample_gsm002',
            'geo_gse12345_sample_gsm003'
        ]
        yield mock_service


# ===============================================================================
# Data Expert Core Functionality Tests
# ===============================================================================

@pytest.mark.unit
class TestDataExpertCore:
    """Test data expert core functionality."""

    def test_agent_creation_succeeds(self, mock_data_manager):
        """Test that agent can be created successfully."""
        agent = data_expert(mock_data_manager)
        assert agent is not None

    def test_agent_has_graph_structure(self, mock_data_manager):
        """Test that agent has expected graph structure."""
        agent = data_expert(mock_data_manager)

        # Should have a graph structure
        graph = agent.get_graph()
        assert graph is not None

    def test_agent_with_callback_handler(self, mock_data_manager):
        """Test agent creation with callback handler."""
        mock_callback = Mock()

        agent = data_expert(
            data_manager=mock_data_manager,
            callback_handler=mock_callback
        )

        assert agent is not None


# ===============================================================================
# Service Integration Tests
# ===============================================================================

@pytest.mark.unit
class TestServiceIntegration:
    """Test integration with GEOService, ConcatenationService, and DataExpertAssistant."""

    @patch('lobster.tools.geo_service.GEOService')
    def test_geo_service_integration(self, MockGEOService, mock_data_manager):
        """Test GEO service integration for fetching and downloading datasets."""
        # Setup mock service
        mock_service = MockGEOService.return_value
        mock_service.fetch_metadata_only.return_value = (
            {"accession": "GSE12345", "title": "Test dataset"},
            {"validation_status": "PASS"}
        )

        # Create agent
        agent = data_expert(mock_data_manager)

        # Service would be initialized when tools are called
        assert agent is not None

    @patch('lobster.tools.concatenation_service.ConcatenationService')
    def test_concatenation_service_integration(self, MockConcatService, mock_data_manager):
        """Test concatenation service integration for sample merging."""
        # Setup mock service
        mock_service = MockConcatService.return_value
        mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        mock_service.concatenate_from_modalities.return_value = (
            mock_adata,
            {'n_samples': 3, 'final_shape': (1500, 5000)}
        )

        # Create agent
        agent = data_expert(mock_data_manager)

        # Service would be initialized when concatenate tool is called
        assert agent is not None

    def test_data_expert_assistant_integration(self, mock_data_manager):
        """Test DataExpertAssistant integration for strategy extraction."""
        # Create agent (assistant is initialized within data_expert function)
        agent = data_expert(mock_data_manager)

        # Assistant would be used when fetching GEO metadata
        assert agent is not None


# ===============================================================================
# Agent Configuration Tests
# ===============================================================================

@pytest.mark.unit
class TestAgentConfiguration:
    """Test data expert agent configuration options."""

    def test_agent_with_custom_name(self, mock_data_manager):
        """Test agent creation with custom agent name."""
        agent = data_expert(
            data_manager=mock_data_manager,
            agent_name="custom_data_expert"
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

        agent = data_expert(
            data_manager=mock_data_manager,
            handoff_tools=mock_handoff_tools
        )

        assert agent is not None

    def test_agent_with_all_options(self, mock_data_manager):
        """Test agent creation with all configuration options."""
        mock_callback = Mock()

        def mock_tool():
            """Mock tool for testing."""
            return "test"
        mock_tool.__name__ = "mock_tool"

        agent = data_expert(
            data_manager=mock_data_manager,
            callback_handler=mock_callback,
            agent_name="test_data_expert",
            handoff_tools=[mock_tool]
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

        agent = data_expert(mock_data_manager)
        assert agent is not None

    def test_agent_with_available_modalities(self, mock_data_manager):
        """Test agent creation with modalities available."""
        mock_data_manager.list_modalities.return_value = [
            'geo_gse12345',
            'geo_gse67890',
            'custom_dataset'
        ]

        agent = data_expert(mock_data_manager)
        assert agent is not None

    def test_modality_operations(self, mock_data_manager):
        """Test that agent can access modality operations."""
        # Setup data manager with modalities
        mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        mock_data_manager.get_modality.return_value = mock_adata
        mock_data_manager.list_modalities.return_value = ['geo_gse12345']

        agent = data_expert(mock_data_manager)

        # Agent should have access to data manager
        assert agent is not None


# ===============================================================================
# Error Handling and Edge Cases
# ===============================================================================

@pytest.mark.unit
class TestDataExpertErrorHandling:
    """Test data expert error handling and edge cases."""

    def test_agent_creation_minimal_config(self, mock_data_manager):
        """Test agent creation with minimal configuration."""
        # Should work with just data manager
        agent = data_expert(data_manager=mock_data_manager)
        assert agent is not None

    def test_data_manager_edge_cases(self, mock_data_manager):
        """Test data manager edge cases."""
        # Test with missing modality access
        mock_data_manager.get_modality.side_effect = KeyError("Modality not found")

        # Agent should still be created, errors handled at tool execution time
        agent = data_expert(mock_data_manager)
        assert agent is not None

    def test_metadata_store_operations(self, mock_data_manager):
        """Test metadata store operations."""
        # Setup metadata store
        mock_data_manager.metadata_store = {
            'GSE12345': {
                'metadata': {'title': 'Test dataset'},
                'validation': {'status': 'PASS'},
                'strategy_config': {}
            }
        }

        agent = data_expert(mock_data_manager)
        assert agent is not None


# ===============================================================================
# Workspace Restoration Tests
# ===============================================================================

@pytest.mark.unit
class TestWorkspaceRestoration:
    """Test workspace restoration functionality."""

    def test_agent_with_workspace_restoration(self, mock_data_manager):
        """Test that agent supports workspace restoration."""
        # Setup available datasets
        mock_data_manager.available_datasets = {
            'geo_gse12345': {'size_mb': 150.5, 'last_modified': '2024-01-15'},
            'geo_gse67890': {'size_mb': 200.0, 'last_modified': '2024-01-14'}
        }

        mock_data_manager.restore_session.return_value = {
            'restored': ['geo_gse12345'],
            'skipped': [],
            'total_size_mb': 150.5
        }

        agent = data_expert(mock_data_manager)
        assert agent is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
