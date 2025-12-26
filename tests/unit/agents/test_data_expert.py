"""
Unit tests for data expert agent.

This module tests the data expert agent's core functionality including
GEO data fetching, workspace management, sample concatenation, and
integration with GEOService, ConcatenationService, and DataExpertAssistant.

Test coverage focuses on agent creation, service integration, and configuration.
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from lobster.agents.data_expert import data_expert
from lobster.agents.data_expert_assistant import DataExpertAssistant, StrategyConfig
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.data_access.geo_service import GEOService
from lobster.services.data_management.concatenation_service import ConcatenationService
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
def mock_data_manager(mock_provider_config, tmp_path):
    """Create mock data manager with modality operations.

    Note: This fixture now requires mock_provider_config to ensure LLM
    creation works properly in the refactored provider system.
    """
    mock_dm = Mock(spec=DataManagerV2)
    mock_dm.list_modalities.return_value = ["geo_gse12345", "custom_dataset"]
    mock_dm.workspace_path = str(tmp_path / "workspace")

    # Create mock data
    mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
    mock_dm.get_modality.return_value = mock_adata
    mock_dm.modalities = {"geo_gse12345": mock_adata}
    mock_dm.metadata_store = {}
    mock_dm.log_tool_usage.return_value = None
    mock_dm.save_modality.return_value = None
    mock_dm.get_quality_metrics.return_value = {
        "total_counts": 50000,
        "mean_counts_per_obs": 1500,
    }
    mock_dm.get_workspace_status.return_value = {
        "workspace_path": str(tmp_path / "workspace"),
        "registered_adapters": ["transcriptomics_single_cell", "transcriptomics_bulk"],
        "registered_backends": ["h5ad"],
    }
    mock_dm.available_datasets = {}
    mock_dm.restore_session.return_value = {"restored": [], "total_size_mb": 0}

    yield mock_dm


@pytest.fixture
def mock_geo_service():
    """Mock GEO service for data fetching."""
    with patch("lobster.services.data_access.geo_service.GEOService") as MockGEOService:
        mock_service = MockGEOService.return_value
        mock_service.fetch_metadata_only.return_value = (
            {
                "title": "Test dataset",
                "accession": "GSE12345",
                "organism": "Homo sapiens",
                "summary": "Test summary",
                "supplementary_file": ["matrix.txt.gz", "barcodes.tsv.gz"],
            },
            {
                "validation_status": "PASS",
                "alignment_percentage": 75.0,
                "predicted_data_type": "single_cell_rna_seq",
            },
        )
        mock_service.download_dataset.return_value = "Successfully downloaded GSE12345"
        yield mock_service


@pytest.fixture
def mock_concat_service():
    """Mock concatenation service for sample merging."""
    with patch(
        "lobster.services.data_management.concatenation_service.ConcatenationService"
    ) as MockConcatService:
        mock_service = MockConcatService.return_value
        mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        mock_service.concatenate_from_modalities.return_value = (
            mock_adata,
            {
                "n_samples": 3,
                "final_shape": (1500, 5000),
                "join_type": "inner",
                "strategy_used": "smart_sparse",
                "processing_time_seconds": 1.5,
            },
        )
        mock_service.auto_detect_samples.return_value = [
            "geo_gse12345_sample_gsm001",
            "geo_gse12345_sample_gsm002",
            "geo_gse12345_sample_gsm003",
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
            data_manager=mock_data_manager, callback_handler=mock_callback
        )

        assert agent is not None


# ===============================================================================
# Service Integration Tests
# ===============================================================================


@pytest.mark.unit
class TestServiceIntegration:
    """Test integration with GEOService, ConcatenationService, and DataExpertAssistant."""

    @patch("lobster.services.data_access.geo_service.GEOService")
    def test_geo_service_integration(self, MockGEOService, mock_data_manager):
        """Test GEO service integration for fetching and downloading datasets."""
        # Setup mock service
        mock_service = MockGEOService.return_value
        mock_service.fetch_metadata_only.return_value = (
            {"accession": "GSE12345", "title": "Test dataset"},
            {"validation_status": "PASS"},
        )

        # Create agent
        agent = data_expert(mock_data_manager)

        # Service would be initialized when tools are called
        assert agent is not None

    @patch("lobster.services.data_management.concatenation_service.ConcatenationService")
    def test_concatenation_service_integration(
        self, MockConcatService, mock_data_manager
    ):
        """Test concatenation service integration for sample merging."""
        # Setup mock service
        mock_service = MockConcatService.return_value
        mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        mock_service.concatenate_from_modalities.return_value = (
            mock_adata,
            {"n_samples": 3, "final_shape": (1500, 5000)},
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
            data_manager=mock_data_manager, agent_name="custom_data_expert"
        )

        assert agent is not None

    def test_agent_with_delegation_tools(self, mock_data_manager):
        """Test agent creation with delegation tools."""

        # Create a proper mock tool function instead of Mock object
        def mock_delegation_tool():
            """Mock delegation tool."""
            return "Mock delegation executed"

        mock_delegation_tool.__name__ = "mock_delegation_tool"
        mock_delegation_tools = [mock_delegation_tool]

        agent = data_expert(
            data_manager=mock_data_manager, delegation_tools=mock_delegation_tools
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
            delegation_tools=[mock_tool],
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
            "geo_gse12345",
            "geo_gse67890",
            "custom_dataset",
        ]

        agent = data_expert(mock_data_manager)
        assert agent is not None

    def test_modality_operations(self, mock_data_manager):
        """Test that agent can access modality operations."""
        # Setup data manager with modalities
        mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        mock_data_manager.get_modality.return_value = mock_adata
        mock_data_manager.list_modalities.return_value = ["geo_gse12345"]

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
            "GSE12345": {
                "metadata": {"title": "Test dataset"},
                "validation": {"status": "PASS"},
                "strategy_config": {},
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
            "geo_gse12345": {"size_mb": 150.5, "last_modified": "2024-01-15"},
            "geo_gse67890": {"size_mb": 200.0, "last_modified": "2024-01-14"},
        }

        mock_data_manager.restore_session.return_value = {
            "restored": ["geo_gse12345"],
            "skipped": [],
            "total_size_mb": 150.5,
        }

        agent = data_expert(mock_data_manager)
        assert agent is not None


# ===============================================================================
# DataExpertAssistant Tests (File Formatting and Strategy Extraction)
# ===============================================================================


@pytest.mark.unit
class TestDataExpertAssistantFileFormatting:
    """Test DataExpertAssistant file formatting and strategy extraction."""

    @pytest.fixture
    def assistant(self):
        """Create DataExpertAssistant instance."""
        return DataExpertAssistant()

    def test_format_supplementary_files_with_raw_tar_only(self, assistant):
        """Test file formatting with only GSE*_RAW.tar (common 10X pattern)."""
        metadata = {
            "supplementary_file": [
                "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE248nnn/GSE248556/suppl/GSE248556_RAW.tar"
            ],
            "samples": {},
        }

        result = assistant._format_supplementary_files_for_llm(metadata)

        # Should format with bullet points
        assert "Series-level files (1 total):" in result
        assert "  - ftp://ftp.ncbi.nlm.nih.gov" in result
        assert "GSE248556_RAW.tar" in result
        # Should NOT show Python list representation
        assert "['" not in result
        assert "']" not in result

    def test_format_supplementary_files_with_sample_level_files(self, assistant):
        """Test file formatting with sample-level files."""
        metadata = {
            "supplementary_file": [],  # No series-level files
            "samples": {
                "GSM123": {
                    "supplementary_file": [
                        "GSM123_sample1_atac_peaks.bed",
                        "GSM123_sample1_atac_fragments.tsv.gz",
                    ]
                },
                "GSM124": {
                    "supplementary_file": [
                        "GSM124_sample2_atac_peaks.bed",
                        "GSM124_sample2_atac_fragments.tsv.gz",
                    ]
                },
                "GSM125": {
                    "supplementary_file": [
                        "GSM125_sample3_atac_peaks.bed",
                        "GSM125_sample3_atac_fragments.tsv.gz",
                    ]
                },
            },
        }

        result = assistant._format_supplementary_files_for_llm(metadata)

        # Should include sample-level files
        assert "Sample-level files" in result
        assert "GSM123_sample1" in result or "GSM124_sample2" in result
        assert "across 3 samples" in result

    def test_format_supplementary_files_with_mixed_files(self, assistant):
        """Test file formatting with both series-level and sample-level files."""
        metadata = {
            "supplementary_file": [
                "GSE12345_summary_matrix.txt.gz",
                "GSE12345_cell_annotations.csv",
            ],
            "samples": {
                "GSM001": {"supplementary_file": ["GSM001_raw.mtx"]},
                "GSM002": {"supplementary_file": ["GSM002_raw.mtx"]},
            },
        }

        result = assistant._format_supplementary_files_for_llm(metadata)

        # Should show both sections
        assert "Series-level files (2 total):" in result
        assert "Sample-level files" in result
        assert "GSE12345_summary_matrix.txt.gz" in result
        assert "GSM001_raw.mtx" in result or "GSM002_raw.mtx" in result

    def test_format_supplementary_files_with_no_files(self, assistant):
        """Test file formatting with no files available."""
        metadata = {"supplementary_file": [], "samples": {}}

        result = assistant._format_supplementary_files_for_llm(metadata)

        assert result == "No supplementary files listed"

    def test_format_supplementary_files_with_many_series_files(self, assistant):
        """Test file formatting with >15 series-level files (truncation)."""
        # Create 20 dummy files
        many_files = [f"GSE12345_file_{i}.txt" for i in range(20)]

        metadata = {"supplementary_file": many_files, "samples": {}}

        result = assistant._format_supplementary_files_for_llm(metadata)

        # Should show first 15 + truncation message
        assert "Series-level files (20 total):" in result
        assert "... and 5 more series-level files" in result
        assert "GSE12345_file_0.txt" in result
        assert "GSE12345_file_14.txt" in result
        # Should NOT show file 15+ (truncated)
        assert "GSE12345_file_15.txt" not in result

    @patch.object(DataExpertAssistant, "llm")
    def test_extract_strategy_config_with_raw_tar(self, mock_llm, assistant):
        """Test strategy extraction with RAW.tar file."""
        # Mock LLM response simulating successful RAW.tar recognition
        mock_response = Mock()
        mock_response.content = [
            Mock(
                text='{"summary_file_name": "", "summary_file_type": "", '
                '"processed_matrix_name": "", "processed_matrix_filetype": "", '
                '"raw_UMI_like_matrix_name": "GSE248556", "raw_UMI_like_matrix_filetype": "", '
                '"cell_annotation_name": "", "cell_annotation_filetype": "", '
                '"raw_data_available": true}'
            )
        ]
        mock_llm.invoke.return_value = mock_response

        metadata = {
            "title": "Single-cell transcriptome study",
            "summary": "10X scRNA-seq analysis",
            "overall_design": "Droplet-based scRNA-seq (10X Genomics)",
            "supplementary_file": [
                "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE248nnn/GSE248556/suppl/GSE248556_RAW.tar"
            ],
            "samples": {},
        }

        result = assistant.extract_strategy_config(metadata, "GSE248556")

        # Should successfully extract strategy
        assert result is not None
        assert result.raw_data_available is True
        # With RAW.tar guidance, should at least identify the GSE ID
        assert (
            result.raw_UMI_like_matrix_name == "GSE248556"
            or result.raw_UMI_like_matrix_name == ""
        )

        # Verify LLM was called with properly formatted files
        call_args = mock_llm.invoke.call_args
        assert call_args is not None
        user_prompt = call_args[0][0][1]["content"]  # Extract user prompt

        # Should have formatted file display (not Python list)
        assert "Series-level files" in user_prompt
        assert "GSE248556_RAW.tar" in user_prompt
        assert "['" not in user_prompt  # No Python list representation

    @patch.object(DataExpertAssistant, "llm")
    def test_extract_strategy_config_with_sample_level_files(self, mock_llm, assistant):
        """Test strategy extraction with only sample-level files."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = [
            Mock(
                text='{"summary_file_name": "", "summary_file_type": "", '
                '"processed_matrix_name": "", "processed_matrix_filetype": "", '
                '"raw_UMI_like_matrix_name": "sample_raw", "raw_UMI_like_matrix_filetype": "mtx", '
                '"cell_annotation_name": "barcodes", "cell_annotation_filetype": "tsv", '
                '"raw_data_available": true}'
            )
        ]
        mock_llm.invoke.return_value = mock_response

        metadata = {
            "title": "Multiome study",
            "summary": "Joint GEX + ATAC profiling",
            "overall_design": "10X Multiome",
            "supplementary_file": [],  # No series-level files
            "samples": {
                "GSM001": {
                    "supplementary_file": [
                        "GSM001_gex_matrix.mtx",
                        "GSM001_atac_fragments.tsv.gz",
                    ]
                },
                "GSM002": {
                    "supplementary_file": [
                        "GSM002_gex_matrix.mtx",
                        "GSM002_atac_fragments.tsv.gz",
                    ]
                },
            },
        }

        result = assistant.extract_strategy_config(metadata, "GSE12345")

        # Should successfully extract strategy
        assert result is not None

        # Verify LLM received sample-level files
        call_args = mock_llm.invoke.call_args
        user_prompt = call_args[0][0][1]["content"]

        # Should show sample-level files
        assert "Sample-level files" in user_prompt
        assert "GSM001" in user_prompt or "GSM002" in user_prompt

    def test_extract_strategy_config_system_prompt_has_raw_tar_guidance(
        self, assistant
    ):
        """Verify system prompt includes RAW.tar handling guidance."""
        # We can't easily test LLM invocation without mocking, but we can verify
        # the code structure includes RAW.tar guidance in the source

        # Check that the source code contains RAW.tar guidance
        import inspect

        source = inspect.getsource(assistant.extract_strategy_config)

        # Verify RAW.tar guidance is present in system prompt
        assert "RAW.tar" in source or "RAW.tar File Handling" in source
        assert "bundled archives" in source or "bundled" in source
        assert "raw_data_available" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
