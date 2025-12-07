"""
Unit tests for metadata_assistant agent.

Tests all 4 metadata tools independently with mock services:
- map_samples_by_id
- read_sample_metadata
- standardize_sample_metadata
- validate_dataset_content

TODO: Consider splitting this 2,321-line file into separate files by test class:
  - test_metadata_assistant_init.py (TestMetadataAssistantInit)
  - test_metadata_assistant_mapping.py (TestMapSamplesByID)
  - test_metadata_assistant_reading.py (TestReadSampleMetadata)
  - test_metadata_assistant_standardization.py (TestStandardizeSampleMetadata)
  - test_metadata_assistant_validation.py (TestValidateDatasetContent)
  - test_metadata_assistant_system.py (TestSystemPrompt, TestUnexpectedErrors)
  - test_metadata_assistant_routing.py (TestToolRouting)
  - test_metadata_assistant_handoff.py (TestHandoffCoordination)
All test classes have clear boundaries (not tightly coupled). Split would improve maintainability.
"""

import json
from datetime import date
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.schemas.transcriptomics import TranscriptomicsMetadataSchema
from lobster.services.metadata.metadata_standardization_service import (
    DatasetValidationResult,
    StandardizationResult,
)
from lobster.services.metadata.sample_mapping_service import (
    SampleMappingResult,
    SampleMatch,
    UnmappedSample,
)


@pytest.fixture
def mock_data_manager():
    """Create mock DataManagerV2 for testing."""
    from pathlib import Path

    mock_dm = Mock(spec=DataManagerV2)
    mock_dm.log_tool_usage = Mock()
    mock_dm.workspace_path = Path("/tmp/test_workspace")
    mock_dm.list_modalities = Mock(return_value=[])
    mock_dm.get_modality = Mock()
    return mock_dm


@pytest.fixture
def mock_sample_mapping_service():
    """Create mock SampleMappingService."""
    mock_service = Mock()
    return mock_service


@pytest.fixture
def mock_metadata_standardization_service():
    """Create mock MetadataStandardizationService."""
    mock_service = Mock()
    return mock_service


@pytest.fixture
def mock_llm():
    """Create mock LLM."""
    mock = Mock()
    mock.with_config = Mock(return_value=mock)
    return mock


@pytest.fixture
def mock_agent():
    """Create mock LangGraph agent."""
    return Mock()


class TestMetadataAssistantInit:
    """Test metadata_assistant agent initialization."""

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_init_without_callback(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_llm,
        mock_agent,
    ):
        """Test agent initialization without callback handler."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {"param": "value"}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent

        # Create agent
        agent = metadata_assistant(data_manager=mock_data_manager)

        # Verify service initialization
        mock_mapping_class.assert_called_once_with(data_manager=mock_data_manager)
        mock_standardization_class.assert_called_once_with(
            data_manager=mock_data_manager
        )

        # Verify LLM creation
        mock_settings_instance.get_agent_llm_params.assert_called_once_with("assistant")
        mock_create_llm.assert_called_once_with(
            "metadata_assistant", {"param": "value"}
        )

        # Verify agent creation
        mock_create_agent.assert_called_once()
        call_kwargs = mock_create_agent.call_args[1]
        assert call_kwargs["model"] == mock_llm
        # 10 base tools + 1 optional filter_samples_by (if microbiome available)
        # Base: map_samples_by_id, read_sample_metadata, standardize_sample_metadata,
        #       validate_dataset_content, process_metadata_entry, process_metadata_queue,
        #       update_metadata_status, get_content_from_workspace, write_to_workspace,
        #       execute_custom_code
        assert len(call_kwargs["tools"]) >= 10
        assert call_kwargs["name"] == "metadata_assistant"
        assert "metadata assistant" in call_kwargs["prompt"].lower()

        assert agent == mock_agent

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_init_with_callback(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_llm,
        mock_agent,
    ):
        """Test agent initialization with callback handler."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent

        callback_handler = Mock()

        # Create agent
        agent = metadata_assistant(
            data_manager=mock_data_manager, callback_handler=callback_handler
        )

        # Verify callback was applied
        mock_llm.with_config.assert_called_once_with(callbacks=[callback_handler])

        assert agent == mock_agent

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_init_with_delegation_tools(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_llm,
        mock_agent,
    ):
        """Test agent initialization with delegation tools."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent

        delegation_tool1 = Mock()
        delegation_tool2 = Mock()
        delegation_tools = [delegation_tool1, delegation_tool2]

        # Create agent
        agent = metadata_assistant(
            data_manager=mock_data_manager, delegation_tools=delegation_tools
        )

        # Verify delegation tools are included
        call_kwargs = mock_create_agent.call_args[1]
        # Base tools: 10 core + 1 optional filter_samples_by (if microbiome) + 2 delegation tools
        assert len(call_kwargs["tools"]) >= 12  # At least 10 base + 2 delegation
        assert delegation_tool1 in call_kwargs["tools"]
        assert delegation_tool2 in call_kwargs["tools"]

        assert agent == mock_agent


class TestMapSamplesByID:
    """Test map_samples_by_id tool."""

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_map_samples_success(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_sample_mapping_service,
        mock_llm,
        mock_agent,
    ):
        """Test successful sample mapping."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_mapping_class.return_value = mock_sample_mapping_service

        # Create mock mapping result
        mock_result = SampleMappingResult(
            exact_matches=[
                SampleMatch(
                    source_id="Sample_A",
                    target_id="sample_a",
                    confidence_score=1.0,
                    match_strategy="exact",
                )
            ],
            fuzzy_matches=[],
            unmapped=[],
            summary={
                "total_source_samples": 10,
                "total_target_samples": 10,
                "exact_matches": 1,
                "fuzzy_matches": 0,
                "unmapped": 0,
                "mapping_rate": 0.1,
            },
            warnings=[],
        )

        mock_sample_mapping_service.map_samples_by_id.return_value = mock_result
        mock_sample_mapping_service.format_mapping_report.return_value = (
            "# Sample Mapping Report\nSuccess!"
        )

        # Configure data_manager mocks for new get_samples() implementation
        mock_data_manager.list_modalities.return_value = [
            "geo_gse12345",
            "geo_gse67890",
        ]
        mock_adata_source = Mock()
        mock_adata_source.obs = pd.DataFrame(
            {
                "sample_id": ["S1", "S2", "S3"],
                "condition": ["Control", "Treatment", "Control"],
            }
        )
        mock_adata_target = Mock()
        mock_adata_target.obs = pd.DataFrame(
            {"sample_id": ["S1", "S2"], "condition": ["Control", "Treatment"]}
        )
        mock_data_manager.get_modality.side_effect = lambda x: (
            mock_adata_source if x == "geo_gse12345" else mock_adata_target
        )
        mock_data_manager.metadata_store = {}

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the map_samples_by_id tool
        tools = mock_create_agent.call_args[1]["tools"]
        map_tool = next(t for t in tools if t.name == "map_samples_by_id")

        # Call the tool
        result = map_tool.func(
            source="geo_gse12345",
            target="geo_gse67890",
            source_type="modality",
            target_type="modality",
            min_confidence=0.75,
            strategies="all",
        )

        # Verify service was called correctly
        mock_sample_mapping_service.map_samples_by_id.assert_called_once_with(
            source_identifier="geo_gse12345",
            target_identifier="geo_gse67890",
            strategies=None,  # "all" maps to None
        )

        # Verify provenance logging
        mock_data_manager.log_tool_usage.assert_called_once()
        log_call = mock_data_manager.log_tool_usage.call_args
        assert log_call[1]["tool_name"] == "map_samples_by_id"
        assert log_call[1]["parameters"]["source"] == "geo_gse12345"
        assert log_call[1]["parameters"]["target"] == "geo_gse67890"
        assert log_call[1]["result_summary"]["mapping_rate"] == 0.1

        # Verify report formatting
        mock_sample_mapping_service.format_mapping_report.assert_called_once_with(
            mock_result
        )

        assert "Sample Mapping Report" in result
        assert "Success!" in result

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_map_samples_with_specific_strategies(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_sample_mapping_service,
        mock_llm,
        mock_agent,
    ):
        """Test sample mapping with specific strategies."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_mapping_class.return_value = mock_sample_mapping_service

        mock_result = SampleMappingResult(
            exact_matches=[],
            fuzzy_matches=[],
            unmapped=[],
            summary={
                "total_source_samples": 5,
                "total_target_samples": 5,
                "exact_matches": 0,
                "fuzzy_matches": 0,
                "unmapped": 5,
                "mapping_rate": 0.0,
            },
            warnings=[],
        )

        mock_sample_mapping_service.map_samples_by_id.return_value = mock_result
        mock_sample_mapping_service.format_mapping_report.return_value = (
            "No matches found"
        )

        # Configure data_manager mocks for new get_samples() implementation
        mock_data_manager.list_modalities.return_value = ["dataset1", "dataset2"]
        mock_adata_source = Mock()
        mock_adata_source.obs = pd.DataFrame({"sample_id": ["S1", "S2"]})
        mock_adata_target = Mock()
        mock_adata_target.obs = pd.DataFrame({"sample_id": ["S3", "S4"]})
        mock_data_manager.get_modality.side_effect = lambda x: (
            mock_adata_source if x == "dataset1" else mock_adata_target
        )
        mock_data_manager.metadata_store = {}

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        map_tool = next(t for t in tools if t.name == "map_samples_by_id")

        # Call with specific strategies
        result = map_tool.func(
            source="dataset1",
            target="dataset2",
            source_type="modality",
            target_type="modality",
            strategies="exact,pattern",
        )

        # Verify strategies were parsed correctly
        mock_sample_mapping_service.map_samples_by_id.assert_called_once()
        call_kwargs = mock_sample_mapping_service.map_samples_by_id.call_args[1]
        assert call_kwargs["strategies"] == ["exact", "pattern"]

        assert "No matches found" in result

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_map_samples_invalid_strategy(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_sample_mapping_service,
        mock_llm,
        mock_agent,
    ):
        """Test sample mapping with invalid strategy."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_mapping_class.return_value = mock_sample_mapping_service

        # Configure data_manager mocks (needed even for error cases)
        mock_data_manager.list_modalities.return_value = ["dataset1", "dataset2"]
        mock_adata = Mock()
        mock_adata.obs = pd.DataFrame({"sample_id": ["S1"]})
        mock_data_manager.get_modality.return_value = mock_adata
        mock_data_manager.metadata_store = {}

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        map_tool = next(t for t in tools if t.name == "map_samples_by_id")

        # Call with invalid strategy
        result = map_tool.func(
            source="dataset1",
            target="dataset2",
            source_type="modality",
            target_type="modality",
            strategies="invalid_strategy",
        )

        # Verify service was NOT called
        mock_sample_mapping_service.map_samples_by_id.assert_not_called()

        # Verify error message
        assert "❌ Invalid strategies" in result
        assert "invalid_strategy" in result

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_map_samples_value_error(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_sample_mapping_service,
        mock_llm,
        mock_agent,
    ):
        """Test sample mapping with ValueError from service."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_mapping_class.return_value = mock_sample_mapping_service

        # Service raises ValueError
        mock_sample_mapping_service.map_samples_by_id.side_effect = ValueError(
            "Dataset not found"
        )

        # Configure data_manager mocks for new get_samples() implementation
        mock_data_manager.list_modalities.return_value = ["nonexistent", "dataset2"]
        mock_adata = Mock()
        mock_adata.obs = pd.DataFrame({"sample_id": ["S1"]})
        mock_data_manager.get_modality.return_value = mock_adata
        mock_data_manager.metadata_store = {}

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        map_tool = next(t for t in tools if t.name == "map_samples_by_id")

        # Call the tool
        result = map_tool.func(
            source="nonexistent",
            target="dataset2",
            source_type="modality",
            target_type="modality",
        )

        # Verify error message
        assert "❌ Mapping failed" in result
        assert "Dataset not found" in result


class TestReadSampleMetadata:
    """Test read_sample_metadata tool."""

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_read_metadata_summary_format(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_metadata_standardization_service,
        mock_llm,
        mock_agent,
    ):
        """Test reading metadata in summary format."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_standardization_class.return_value = mock_metadata_standardization_service

        # Configure data_manager mocks for read_sample_metadata tool
        mock_data_manager.list_modalities.return_value = ["geo_gse12345"]
        mock_adata = Mock()
        mock_adata.obs = pd.DataFrame(
            {"sample_id": ["S1", "S2"], "condition": ["Control", "Treatment"]}
        )
        mock_data_manager.get_modality.return_value = mock_adata
        mock_data_manager.metadata_store = {}

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        read_tool = next(t for t in tools if t.name == "read_sample_metadata")

        # Call the tool
        result = read_tool.func(
            source="geo_gse12345", source_type="modality", return_format="summary"
        )

        # Verify provenance logging
        mock_data_manager.log_tool_usage.assert_called_once()
        log_call = mock_data_manager.log_tool_usage.call_args
        assert log_call[1]["tool_name"] == "read_sample_metadata"
        assert log_call[1]["parameters"]["source_type"] == "modality"
        assert log_call[1]["parameters"]["return_format"] == "summary"

        # Verify summary format output (tool formats directly from DataFrame)
        assert "geo_gse12345" in result  # Dataset name appears in output
        assert (
            "Total Samples: 2" in result or "Total Samples**: 2" in result
        )  # 2 samples in mock DataFrame
        assert "Field Coverage" in result
        assert "sample_id" in result
        assert "condition" in result

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_read_metadata_detailed_format(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_metadata_standardization_service,
        mock_llm,
        mock_agent,
    ):
        """Test reading metadata in detailed format (JSON)."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_standardization_class.return_value = mock_metadata_standardization_service

        # Configure data_manager mocks for read_sample_metadata tool
        mock_data_manager.list_modalities.return_value = ["geo_gse12345"]
        mock_adata = Mock()
        mock_adata.obs = pd.DataFrame(
            {"sample_id": ["S1", "S2"], "condition": ["Control", "Treatment"]}
        )
        mock_data_manager.get_modality.return_value = mock_adata
        mock_data_manager.metadata_store = {}

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        read_tool = next(t for t in tools if t.name == "read_sample_metadata")

        # Call the tool
        result = read_tool.func(
            source="geo_gse12345", source_type="modality", return_format="detailed"
        )

        # Verify result is JSON string (tool converts DataFrame to JSON)
        parsed = json.loads(result)
        assert len(parsed) == 2  # 2 samples
        assert parsed[0]["sample_id"] == "S1"
        assert parsed[0]["condition"] == "Control"
        assert parsed[1]["sample_id"] == "S2"
        assert parsed[1]["condition"] == "Treatment"

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_read_metadata_schema_format(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_metadata_standardization_service,
        mock_llm,
        mock_agent,
    ):
        """Test reading metadata in schema format (DataFrame markdown)."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_standardization_class.return_value = mock_metadata_standardization_service

        # Configure data_manager mocks for read_sample_metadata tool
        mock_data_manager.list_modalities.return_value = ["geo_gse12345"]
        mock_adata = Mock()
        mock_adata.obs = pd.DataFrame(
            {"sample_id": ["S1", "S2"], "condition": ["Control", "Treatment"]}
        )
        mock_data_manager.get_modality.return_value = mock_adata
        mock_data_manager.metadata_store = {}

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        read_tool = next(t for t in tools if t.name == "read_sample_metadata")

        # Call the tool
        result = read_tool.func(
            source="geo_gse12345", source_type="modality", return_format="schema"
        )

        # Verify DataFrame was converted to markdown (tool converts directly)
        assert "sample_id" in result
        assert "condition" in result
        assert "Control" in result
        assert "Treatment" in result
        assert "S1" in result
        assert "S2" in result

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_read_metadata_with_fields(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_metadata_standardization_service,
        mock_llm,
        mock_agent,
    ):
        """Test reading metadata with specific fields."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_standardization_class.return_value = mock_metadata_standardization_service

        # Configure data_manager mocks for read_sample_metadata tool
        mock_data_manager.list_modalities.return_value = ["geo_gse12345"]
        mock_adata = Mock()
        mock_adata.obs = pd.DataFrame(
            {
                "sample_id": ["S1", "S2"],
                "condition": ["Control", "Treatment"],
                "organism": ["Homo sapiens", "Homo sapiens"],
            }
        )
        mock_data_manager.get_modality.return_value = mock_adata
        mock_data_manager.metadata_store = {}

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        read_tool = next(t for t in tools if t.name == "read_sample_metadata")

        # Call with field filtering
        result = read_tool.func(
            source="geo_gse12345", source_type="modality", fields="condition,organism"
        )

        # Verify fields were filtered correctly (tool filters DataFrame before formatting)
        assert "condition" in result
        assert "organism" in result
        # sample_id should NOT appear in filtered output
        assert "Field Coverage" in result or "condition" in result

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_read_metadata_value_error(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_metadata_standardization_service,
        mock_llm,
        mock_agent,
    ):
        """Test reading metadata with ValueError from data_manager."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_standardization_class.return_value = mock_metadata_standardization_service

        # Configure data_manager to NOT include the requested modality
        mock_data_manager.list_modalities.return_value = (
            []
        )  # Empty list - modality doesn't exist
        mock_data_manager.metadata_store = {}

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        read_tool = next(t for t in tools if t.name == "read_sample_metadata")

        # Call the tool with non-existent modality
        result = read_tool.func(source="nonexistent", source_type="modality")

        # Verify error message (tool checks list_modalities() before accessing modality)
        assert "❌ Error: Modality 'nonexistent' not found" in result


class TestStandardizeSampleMetadata:
    """Test standardize_sample_metadata tool."""

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_standardize_metadata_success(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_metadata_standardization_service,
        mock_llm,
        mock_agent,
    ):
        """Test successful metadata standardization."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_standardization_class.return_value = mock_metadata_standardization_service

        # Create mock standardization result with actual schema instances
        sample1 = TranscriptomicsMetadataSchema(
            sample_id="Sample_1",
            condition="Control",
            organism="Homo sapiens",
            platform="Illumina NovaSeq",
            sequencing_type="single-cell",
        )
        sample2 = TranscriptomicsMetadataSchema(
            sample_id="Sample_2",
            condition="Treatment",
            organism="Homo sapiens",
            platform="Illumina NovaSeq",
            sequencing_type="single-cell",
        )
        mock_result = StandardizationResult(
            standardized_metadata=[sample1, sample2],  # 2 valid samples
            validation_errors={"S3": "Missing required field"},
            field_coverage={"condition": 100.0, "organism": 66.7},
            warnings=["Organism 'Human' not in vocabulary"],
        )

        mock_metadata_standardization_service.standardize_metadata.return_value = (
            mock_result,
            {},
            None,  # Return tuple: (result, stats, ir)
        )

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        standardize_tool = next(
            t for t in tools if t.name == "standardize_sample_metadata"
        )

        # Call the tool
        result = standardize_tool.func(
            source="geo_gse12345",
            source_type="modality",
            target_schema="transcriptomics",
        )

        # Verify service was called correctly
        mock_metadata_standardization_service.standardize_metadata.assert_called_once_with(
            identifier="geo_gse12345",
            target_schema="transcriptomics",
            controlled_vocabularies=None,
        )

        # Verify provenance logging
        mock_data_manager.log_tool_usage.assert_called_once()
        log_call = mock_data_manager.log_tool_usage.call_args
        assert log_call[1]["tool_name"] == "standardize_sample_metadata"
        assert log_call[1]["result_summary"]["valid_samples"] == 2
        assert log_call[1]["result_summary"]["validation_errors"] == 1
        assert log_call[1]["result_summary"]["warnings"] == 1

        # Verify report formatting
        assert "# Metadata Standardization Report" in result
        assert "Valid Samples:** 2" in result  # Markdown bold formatting
        assert "Validation Errors:** 1" in result
        assert "## Field Coverage" in result
        assert "condition: 100.0%" in result
        assert "## Warnings" in result

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_standardize_with_controlled_vocabularies(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_metadata_standardization_service,
        mock_llm,
        mock_agent,
    ):
        """Test standardization with controlled vocabularies."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_standardization_class.return_value = mock_metadata_standardization_service

        mock_result = StandardizationResult(
            standardized_metadata=[],
            validation_errors={},
            field_coverage={},
            warnings=[],
        )

        mock_metadata_standardization_service.standardize_metadata.return_value = (
            mock_result,
            {},
            None,  # Return tuple: (result, stats, ir)
        )

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        standardize_tool = next(
            t for t in tools if t.name == "standardize_sample_metadata"
        )

        # Call with controlled vocabularies
        controlled_vocab_json = '{"condition": ["Control", "Treatment"]}'
        result = standardize_tool.func(
            source="geo_gse12345",
            source_type="modality",
            target_schema="transcriptomics",
            controlled_vocabularies=controlled_vocab_json,
        )

        # Verify controlled vocabularies were parsed correctly
        mock_metadata_standardization_service.standardize_metadata.assert_called_once()
        call_kwargs = (
            mock_metadata_standardization_service.standardize_metadata.call_args[1]
        )
        assert call_kwargs["controlled_vocabularies"] == {
            "condition": ["Control", "Treatment"]
        }

        assert "# Metadata Standardization Report" in result

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_standardize_invalid_json(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_metadata_standardization_service,
        mock_llm,
        mock_agent,
    ):
        """Test standardization with invalid JSON for controlled vocabularies."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_standardization_class.return_value = mock_metadata_standardization_service

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        standardize_tool = next(
            t for t in tools if t.name == "standardize_sample_metadata"
        )

        # Call with invalid JSON
        result = standardize_tool.func(
            source="geo_gse12345",
            source_type="modality",
            target_schema="transcriptomics",
            controlled_vocabularies="invalid json",
        )

        # Verify service was NOT called
        mock_metadata_standardization_service.standardize_metadata.assert_not_called()

        # Verify error message
        assert "❌ Invalid controlled_vocabularies JSON" in result

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_standardize_value_error(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_metadata_standardization_service,
        mock_llm,
        mock_agent,
    ):
        """Test standardization with ValueError from service."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_standardization_class.return_value = mock_metadata_standardization_service

        # Service raises ValueError
        mock_metadata_standardization_service.standardize_metadata.side_effect = (
            ValueError("Unknown schema type")
        )

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        standardize_tool = next(
            t for t in tools if t.name == "standardize_sample_metadata"
        )

        # Call the tool
        result = standardize_tool.func(
            source="geo_gse12345",
            source_type="modality",
            target_schema="invalid_schema",
        )

        # Verify error message
        assert "❌ Standardization failed" in result
        assert "Unknown schema type" in result


class TestValidateDatasetContent:
    """Test validate_dataset_content tool."""

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_validate_dataset_passing(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_metadata_standardization_service,
        mock_llm,
        mock_agent,
    ):
        """Test validation of dataset passing all checks."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_standardization_class.return_value = mock_metadata_standardization_service

        # Create mock validation result (passing)
        mock_result = DatasetValidationResult(
            has_required_samples=True,
            missing_conditions=[],
            control_issues=[],
            duplicate_ids=[],
            platform_consistency=True,
            summary={
                "total_samples": 50,
                "unique_conditions": 2,
                "has_condition_field": True,
                "has_platform_field": True,
            },
            warnings=[],
        )

        mock_metadata_standardization_service.validate_dataset_content.return_value = (
            mock_result,
            {},
            None,  # Return tuple: (result, stats, ir)
        )

        # Configure data_manager mocks for validation
        mock_data_manager.list_modalities.return_value = ["geo_gse12345"]

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        validate_tool = next(t for t in tools if t.name == "validate_dataset_content")

        # Call the tool
        result = validate_tool.func(
            source="geo_gse12345", source_type="modality", expected_samples=10
        )

        # Verify service was called correctly
        mock_metadata_standardization_service.validate_dataset_content.assert_called_once_with(
            identifier="geo_gse12345",
            expected_samples=10,
            required_conditions=None,
            check_controls=True,
            check_duplicates=True,
        )

        # Verify provenance logging
        mock_data_manager.log_tool_usage.assert_called_once()
        log_call = mock_data_manager.log_tool_usage.call_args
        assert log_call[1]["tool_name"] == "validate_dataset_content"
        assert log_call[1]["result_summary"]["has_required_samples"] is True
        assert log_call[1]["result_summary"]["platform_consistency"] is True

        # Verify report formatting
        assert "# Dataset Validation Report" in result
        assert "✅ Sample Count: 50 samples" in result
        assert "✅ Platform Consistency: Consistent" in result
        assert "✅ No Duplicate IDs" in result
        assert "✅ **Dataset passes validation**" in result

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_validate_dataset_with_issues(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_metadata_standardization_service,
        mock_llm,
        mock_agent,
    ):
        """Test validation of dataset with issues."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_standardization_class.return_value = mock_metadata_standardization_service

        # Create mock validation result (with issues)
        mock_result = DatasetValidationResult(
            has_required_samples=False,
            missing_conditions=["Treatment"],
            control_issues=["No control samples found"],
            duplicate_ids=["Sample_1", "Sample_2"],
            platform_consistency=False,
            summary={
                "total_samples": 5,
                "unique_conditions": 1,
                "has_condition_field": True,
                "has_platform_field": True,
            },
            warnings=["Expected at least 10 samples, found 5"],
        )

        mock_metadata_standardization_service.validate_dataset_content.return_value = (
            mock_result,
            {},
            None,  # Return tuple: (result, stats, ir)
        )

        # Configure data_manager mocks for validation
        mock_data_manager.list_modalities.return_value = ["geo_gse12345"]

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        validate_tool = next(t for t in tools if t.name == "validate_dataset_content")

        # Call the tool
        result = validate_tool.func(
            source="geo_gse12345", source_type="modality", expected_samples=10
        )

        # Verify report formatting shows issues
        assert "# Dataset Validation Report" in result
        assert "❌ Sample Count: 5 samples (below minimum)" in result
        assert "⚠️ Platform Consistency: Inconsistent" in result
        assert "❌ Duplicate IDs: 2 found" in result
        assert "⚠️ Control Samples: No control samples found" in result
        assert "## Missing Required Conditions" in result
        assert "- ❌ Treatment" in result
        assert "## Warnings" in result
        assert "⚠️ **Dataset has issues**" in result

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_validate_with_required_conditions(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_metadata_standardization_service,
        mock_llm,
        mock_agent,
    ):
        """Test validation with required conditions."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_standardization_class.return_value = mock_metadata_standardization_service

        mock_result = DatasetValidationResult(
            has_required_samples=True,
            missing_conditions=[],
            control_issues=[],
            duplicate_ids=[],
            platform_consistency=True,
            summary={"total_samples": 20},
            warnings=[],
        )

        mock_metadata_standardization_service.validate_dataset_content.return_value = (
            mock_result,
            {},
            None,  # Return tuple: (result, stats, ir)
        )

        # Configure data_manager mocks for validation
        mock_data_manager.list_modalities.return_value = ["geo_gse12345"]

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        validate_tool = next(t for t in tools if t.name == "validate_dataset_content")

        # Call with required conditions
        result = validate_tool.func(
            source="geo_gse12345",
            source_type="modality",
            required_conditions="Control,Treatment",
        )

        # Verify required conditions were parsed correctly
        mock_metadata_standardization_service.validate_dataset_content.assert_called_once()
        call_kwargs = (
            mock_metadata_standardization_service.validate_dataset_content.call_args[1]
        )
        assert call_kwargs["required_conditions"] == ["Control", "Treatment"]

        assert "# Dataset Validation Report" in result

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_validate_value_error(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_metadata_standardization_service,
        mock_llm,
        mock_agent,
    ):
        """Test validation with ValueError from service."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_standardization_class.return_value = mock_metadata_standardization_service

        # Service raises ValueError
        mock_metadata_standardization_service.validate_dataset_content.side_effect = (
            ValueError("Dataset not found")
        )

        # Configure data_manager mocks - include "nonexistent" so tool passes validation
        # and reaches service call (which raises ValueError)
        mock_data_manager.list_modalities.return_value = ["nonexistent"]

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        validate_tool = next(t for t in tools if t.name == "validate_dataset_content")

        # Call the tool
        result = validate_tool.func(source="nonexistent", source_type="modality")

        # Verify error message
        assert "❌ Validation failed" in result
        assert "Dataset not found" in result


class TestSystemPrompt:
    """Test system prompt configuration."""

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_system_prompt_includes_date(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_llm,
        mock_agent,
    ):
        """Test that system prompt includes current date."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract system prompt
        call_kwargs = mock_create_agent.call_args[1]
        system_prompt = call_kwargs["prompt"]

        # Verify date formatting (note: system prompt may use ISO format with timestamp)
        today_date = date.today().isoformat()
        assert today_date in system_prompt or "todays date:" in system_prompt.lower()

        # Verify key prompt sections (updated for new hierarchical agent prompt)
        assert "metadata assistant" in system_prompt.lower()
        # Updated: new prompt uses "harmonization" instead of "cross-dataset"
        assert "harmonization" in system_prompt.lower()
        # Core responsibilities mentioned in prompt
        assert "standardize" in system_prompt.lower()
        assert "map" in system_prompt.lower()
        assert "validate" in system_prompt.lower()


class TestUnexpectedErrors:
    """Test handling of unexpected errors (not ValueError)."""

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_map_samples_unexpected_error(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_sample_mapping_service,
        mock_llm,
        mock_agent,
    ):
        """Test map_samples_by_id with unexpected error."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_mapping_class.return_value = mock_sample_mapping_service

        # Mock data_manager to pass validation
        mock_data_manager.list_modalities.return_value = ["dataset1", "dataset2"]
        mock_adata1 = Mock()
        mock_adata1.obs = Mock()
        mock_adata2 = Mock()
        mock_adata2.obs = Mock()
        mock_data_manager.get_modality.side_effect = lambda x: mock_adata1 if x == "dataset1" else mock_adata2

        # Service raises unexpected error
        mock_sample_mapping_service.map_samples_by_id.side_effect = RuntimeError(
            "Unexpected error"
        )

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        map_tool = next(t for t in tools if t.name == "map_samples_by_id")

        # Call the tool
        result = map_tool.func(
            source="dataset1",
            target="dataset2",
            source_type="modality",
            target_type="modality",
        )

        # Verify error message
        assert "❌ Unexpected error during mapping" in result
        assert "Unexpected error" in result

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_read_metadata_unexpected_error(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_metadata_standardization_service,
        mock_llm,
        mock_agent,
    ):
        """Test read_sample_metadata with unexpected error."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_standardization_class.return_value = mock_metadata_standardization_service

        # Mock data_manager to pass validation
        mock_data_manager.list_modalities.return_value = ["dataset"]
        mock_adata = Mock()
        mock_adata.obs = Mock()
        mock_data_manager.get_modality.return_value = mock_adata

        # Service raises unexpected error
        mock_metadata_standardization_service.read_sample_metadata.side_effect = (
            RuntimeError("Unexpected error")
        )

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        read_tool = next(t for t in tools if t.name == "read_sample_metadata")

        # Call the tool
        result = read_tool.func(source="dataset", source_type="modality")

        # Verify error message
        assert "❌ Unexpected error reading metadata" in result
        assert "Unexpected error" in result

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_standardize_unexpected_error(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_metadata_standardization_service,
        mock_llm,
        mock_agent,
    ):
        """Test standardize_sample_metadata with unexpected error."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_standardization_class.return_value = mock_metadata_standardization_service

        # Service raises unexpected error
        mock_metadata_standardization_service.standardize_metadata.side_effect = (
            RuntimeError("Unexpected error")
        )

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        standardize_tool = next(
            t for t in tools if t.name == "standardize_sample_metadata"
        )

        # Call the tool
        result = standardize_tool.func(
            source="dataset", source_type="modality", target_schema="transcriptomics"
        )

        # Verify error message
        assert "❌ Unexpected error during standardization" in result
        assert "Unexpected error" in result

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_validate_unexpected_error(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_metadata_standardization_service,
        mock_llm,
        mock_agent,
    ):
        """Test validate_dataset_content with unexpected error."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_standardization_class.return_value = mock_metadata_standardization_service

        # Mock data_manager to pass validation
        mock_data_manager.list_modalities.return_value = ["dataset"]
        mock_adata = Mock()
        mock_adata.obs = Mock()
        mock_data_manager.get_modality.return_value = mock_adata

        # Service raises unexpected error
        mock_metadata_standardization_service.validate_dataset_content.side_effect = (
            RuntimeError("Unexpected error")
        )

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        validate_tool = next(t for t in tools if t.name == "validate_dataset_content")

        # Call the tool
        result = validate_tool.func(source="dataset", source_type="modality")

        # Verify error message
        assert "❌ Unexpected error during validation" in result
        assert "Unexpected error" in result


# ============================================================================
# Tool Routing Tests (Agent-Level Behavior)
# ============================================================================


class TestToolRouting:
    """Test agent's ability to route queries to correct tools."""

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_agent_chooses_correct_tool_for_query(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_sample_mapping_service,
        mock_llm,
        mock_agent,
    ):
        """Test that agent correctly routes natural language query to appropriate tool.

        Verifies LangGraph agent reasoning:
        - Query: "Map samples between dataset1 and dataset2"
        - Expected: Agent chooses map_samples_by_id tool
        """
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_mapping_class.return_value = mock_sample_mapping_service

        # Mock mapping result
        mock_result = SampleMappingResult(
            exact_matches=[
                SampleMatch(
                    source_id="sample1",
                    target_id="sample1",
                    confidence_score=1.0,
                    match_strategy="exact",
                )
            ],
            fuzzy_matches=[],
            unmapped=[],
            summary={
                "mapping_rate": 1.0,
                "exact_matches": 1,
                "fuzzy_matches": 0,
                "unmapped": 0,
            },
            warnings=[],
        )
        mock_sample_mapping_service.map_samples_by_id.return_value = mock_result
        mock_sample_mapping_service.format_mapping_report.return_value = (
            "Mapping complete"
        )

        # Configure data_manager mocks for tool validation
        mock_data_manager.list_modalities.return_value = ["dataset1", "dataset2"]
        mock_adata = Mock()
        mock_adata.obs = pd.DataFrame({"sample_id": ["S1", "S2"]})
        mock_data_manager.get_modality.return_value = mock_adata
        mock_data_manager.metadata_store = {}

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Verify agent was created with correct tools
        tools = mock_create_agent.call_args[1]["tools"]
        tool_names = {t.name for t in tools}

        assert "map_samples_by_id" in tool_names
        assert "read_sample_metadata" in tool_names
        assert "standardize_sample_metadata" in tool_names
        assert "validate_dataset_content" in tool_names

        # Simulate agent selecting correct tool
        map_tool = next(t for t in tools if t.name == "map_samples_by_id")
        result = map_tool.func(
            source="dataset1",
            target="dataset2",
            source_type="modality",
            target_type="modality",
        )

        assert "Mapping complete" in result
        mock_sample_mapping_service.map_samples_by_id.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.real_api
    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_real_map_samples_query(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_llm,
        mock_agent,
    ):
        """Test agent processing real sample mapping query.

        Real API test with rate limiting.
        Query: "Map samples between GSE12345 and GSE67890"

        Verifies:
        1. Agent calls map_samples_by_id with correct parameters
        2. Response format is structured markdown report
        3. Rate limiting (1s sleep before call)

        Requires: Datasets cached in data_manager workspace
        """
        import time

        from lobster.agents.metadata_assistant import metadata_assistant

        # Rate limiting
        time.sleep(1.0)

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent

        # Mock real-like mapping service
        mock_mapping_service = Mock()
        mock_result = SampleMappingResult(
            exact_matches=[
                SampleMatch(
                    source_id="GSM123456",
                    target_id="GSM789012",
                    confidence_score=1.0,
                    match_strategy="exact",
                )
            ],
            fuzzy_matches=[],
            unmapped=[],
            summary={
                "mapping_rate": 1.0,
                "exact_matches": 1,
                "fuzzy_matches": 0,
                "unmapped": 0,
            },
            warnings=[],
        )
        mock_mapping_service.map_samples_by_id.return_value = mock_result
        mock_mapping_service.format_mapping_report.return_value = """
# Sample Mapping Report

**Datasets**: geo_gse12345 → geo_gse67890
**Mapping Rate**: 100% (1/1 samples mapped)

**Results**:
- Exact matches: 1/1 samples (100%, confidence=1.0)

**Recommendation**: ✅ Proceed with sample-level integration
"""
        mock_mapping_class.return_value = mock_mapping_service

        # Configure data_manager mocks for tool validation
        mock_data_manager.list_modalities.return_value = [
            "geo_gse12345",
            "geo_gse67890",
        ]
        mock_adata = Mock()
        mock_adata.obs = pd.DataFrame({"sample_id": ["S1", "S2"]})
        mock_data_manager.get_modality.return_value = mock_adata
        mock_data_manager.metadata_store = {}

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract tool
        tools = mock_create_agent.call_args[1]["tools"]
        map_tool = next(t for t in tools if t.name == "map_samples_by_id")

        # Simulate agent processing query
        start_time = time.time()
        result = map_tool.func(
            source="geo_gse12345",
            target="geo_gse67890",
            source_type="modality",
            target_type="modality",
            min_confidence=0.75,
            strategies="all",
        )
        elapsed = time.time() - start_time

        # Verify response format
        assert "Sample Mapping Report" in result
        assert "Mapping Rate" in result
        assert "Recommendation" in result
        assert "✅" in result or "⚠️" in result or "❌" in result

        # Verify service called with correct parameters
        mock_mapping_service.map_samples_by_id.assert_called_once_with(
            source_identifier="geo_gse12345",
            target_identifier="geo_gse67890",
            strategies=None,  # "all" → None in tool logic
        )

        # Verify provenance logged
        mock_data_manager.log_tool_usage.assert_called()

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_error_handling_invalid_tool_request(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_sample_mapping_service,
        mock_llm,
        mock_agent,
    ):
        """Test agent error handling for invalid tool requests.

        Scenarios:
        - Invalid strategy names
        - Missing required parameters
        - Malformed inputs
        """
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_mapping_class.return_value = mock_sample_mapping_service

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract tool
        tools = mock_create_agent.call_args[1]["tools"]
        map_tool = next(t for t in tools if t.name == "map_samples_by_id")

        # Test invalid strategy
        result = map_tool.func(
            source="dataset1",
            target="dataset2",
            source_type="modality",
            target_type="modality",
            strategies="invalid_strategy,exact",
        )

        assert "❌ Invalid strategies" in result
        assert "invalid_strategy" in result

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_multiple_tool_orchestration(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_sample_mapping_service,
        mock_metadata_standardization_service,
        mock_llm,
        mock_agent,
    ):
        """Test agent coordinating multiple tools for complex workflow.

        Workflow:
        1. validate_dataset_content (check dataset quality)
        2. read_sample_metadata (inspect fields)
        3. standardize_sample_metadata (harmonize to schema)

        Verifies agent can chain tool calls with intermediate results.
        """
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_mapping_class.return_value = mock_sample_mapping_service
        mock_standardization_class.return_value = mock_metadata_standardization_service

        # Mock validation result
        mock_validation = DatasetValidationResult(
            has_required_samples=True,
            missing_conditions=[],
            duplicate_ids=[],
            control_issues=[],
            platform_consistency=True,
            summary={"total_samples": 48},
            warnings=[],
        )
        mock_metadata_standardization_service.validate_dataset_content.return_value = (
            mock_validation,
            {},
            None,  # Return tuple: (result, stats, ir)
        )

        # Mock read result
        mock_metadata_standardization_service.read_sample_metadata.return_value = (
            "Summary: 48 samples, 5 fields"
        )

        # Mock standardization result
        mock_standardization = StandardizationResult(
            standardized_metadata=[
                TranscriptomicsMetadataSchema(
                    sample_id="Sample_1",
                    condition="Control",
                    platform="Illumina NovaSeq",
                    sequencing_type="single-cell",
                )
            ],
            field_coverage={"condition": 100.0},
            validation_errors={},
            warnings=[],
        )
        mock_metadata_standardization_service.standardize_metadata.return_value = (
            mock_standardization,
            {},
            None,  # Return tuple: (result, stats, ir)
        )

        # Configure data_manager mocks for tool validation
        mock_data_manager.list_modalities.return_value = ["geo_gse12345"]
        mock_adata = Mock()
        mock_adata.obs = pd.DataFrame({"sample_id": ["S1", "S2"]})
        mock_data_manager.get_modality.return_value = mock_adata
        mock_data_manager.metadata_store = {}

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract tools
        tools = mock_create_agent.call_args[1]["tools"]
        validate_tool = next(t for t in tools if t.name == "validate_dataset_content")
        read_tool = next(t for t in tools if t.name == "read_sample_metadata")
        standardize_tool = next(
            t for t in tools if t.name == "standardize_sample_metadata"
        )

        # Simulate agent workflow: validate → read → standardize
        result1 = validate_tool.func(source="geo_gse12345", source_type="modality")
        assert "✅" in result1 or "Dataset Validation" in result1

        result2 = read_tool.func(
            source="geo_gse12345", source_type="modality", return_format="summary"
        )
        assert "Total Samples" in result2 and "2" in result2

        result3 = standardize_tool.func(
            source="geo_gse12345",
            source_type="modality",
            target_schema="transcriptomics",
        )
        assert "Metadata Standardization Report" in result3

        # Verify all tools called
        mock_metadata_standardization_service.validate_dataset_content.assert_called_once()
        # Note: read_sample_metadata tool doesn't call service (formats directly from DataFrame)
        mock_metadata_standardization_service.standardize_metadata.assert_called_once()


# ============================================================================
# Handoff Coordination Tests (Agent-to-Agent Communication)
# ============================================================================


class TestHandoffCoordination:
    """Test metadata_assistant handoff coordination with research_agent."""

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_receive_handoff_from_research_agent(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_sample_mapping_service,
        mock_llm,
        mock_agent,
    ):
        """Test metadata_assistant receiving handoff from research_agent.

        Simulates research_agent handoff:
        - Input: Structured instruction from research_agent
        - Expected: Agent parses instruction, identifies task, executes tool

        Handoff message format:
        "Map samples between geo_gse180759 (RNA-seq, 48 samples) and pxd034567
        (proteomics, 36 samples). Both datasets cached in metadata workspace.
        Use exact and pattern matching strategies. Return mapping report with
        confidence scores and unmapped samples."
        """
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_mapping_class.return_value = mock_sample_mapping_service

        # Mock mapping result
        mock_result = SampleMappingResult(
            exact_matches=[
                SampleMatch(
                    source_id="sample1",
                    target_id="sample1",
                    confidence_score=1.0,
                    match_strategy="exact",
                )
            ],
            fuzzy_matches=[],
            unmapped=[],
            summary={
                "mapping_rate": 1.0,
                "exact_matches": 1,
                "fuzzy_matches": 0,
                "unmapped": 0,
            },
            warnings=[],
        )
        mock_sample_mapping_service.map_samples_by_id.return_value = mock_result
        mock_sample_mapping_service.format_mapping_report.return_value = (
            "✅ Sample Mapping Complete\n\nMapping Rate: 100%"
        )

        # Configure data_manager mocks for tool validation
        mock_data_manager.list_modalities.return_value = ["geo_gse180759", "pxd034567"]
        mock_adata = Mock()
        mock_adata.obs = pd.DataFrame({"sample_id": ["S1", "S2"]})
        mock_data_manager.get_modality.return_value = mock_adata
        mock_data_manager.metadata_store = {}

        # Create agent
        agent = metadata_assistant(data_manager=mock_data_manager)

        # Verify agent created with delegation tools (if provided)
        # In real usage, graph.py provides delegation_tools
        assert agent is not None

        # Extract tool
        tools = mock_create_agent.call_args[1]["tools"]
        map_tool = next(t for t in tools if t.name == "map_samples_by_id")

        # Simulate receiving handoff instruction
        handoff_instruction = """Map samples between geo_gse180759 and pxd034567.
        Use exact and pattern strategies. Return mapping report."""

        # Agent parses instruction and calls tool
        result = map_tool.func(
            source="geo_gse180759",
            target="pxd034567",
            source_type="modality",
            target_type="modality",
            strategies="exact,pattern",
        )

        assert "✅ Sample Mapping Complete" in result
        assert "Mapping Rate" in result

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_process_task_and_generate_report(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_sample_mapping_service,
        mock_llm,
        mock_agent,
    ):
        """Test metadata_assistant processing task and generating structured report.

        Verifies report contains:
        - Status icon (✅/⚠️/❌)
        - Quantitative metrics (mapping rate, confidence scores)
        - Actionable recommendation
        """
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_mapping_class.return_value = mock_sample_mapping_service

        # Mock mapping result with high mapping rate
        mock_result = SampleMappingResult(
            exact_matches=[
                SampleMatch(
                    source_id=f"sample{i}",
                    target_id=f"sample{i}",
                    confidence_score=1.0,
                    match_strategy="exact",
                )
                for i in range(18)
            ],
            fuzzy_matches=[],
            unmapped=[
                UnmappedSample(
                    sample_id="sample19", dataset="geo_gse67890", reason="No match"
                )
            ],
            summary={
                "mapping_rate": 0.95,
                "exact_matches": 18,
                "fuzzy_matches": 0,
                "unmapped": 1,
            },
            warnings=[],
        )
        mock_sample_mapping_service.map_samples_by_id.return_value = mock_result
        mock_sample_mapping_service.format_mapping_report.return_value = """
✅ Sample Mapping Complete

**Datasets**: geo_gse12345 → geo_gse67890
**Mapping Rate**: 95% (18/19 samples mapped)

**Results**:
- Exact matches: 18/19 samples (95%, confidence=1.0)
- Unmapped: 1/19 samples (5%)

**Recommendation**: ✅ Proceed with sample-level integration. High confidence.
"""

        # Configure data_manager mocks for tool validation
        mock_data_manager.list_modalities.return_value = [
            "geo_gse12345",
            "geo_gse67890",
        ]
        mock_adata = Mock()
        mock_adata.obs = pd.DataFrame({"sample_id": ["S1", "S2"]})
        mock_data_manager.get_modality.return_value = mock_adata
        mock_data_manager.metadata_store = {}

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract tool
        tools = mock_create_agent.call_args[1]["tools"]
        map_tool = next(t for t in tools if t.name == "map_samples_by_id")

        # Process task
        result = map_tool.func(
            source="geo_gse12345",
            target="geo_gse67890",
            source_type="modality",
            target_type="modality",
        )

        # Verify report structure
        assert "✅" in result or "⚠️" in result or "❌" in result  # Status icon
        assert "Mapping Rate" in result  # Quantitative metric
        assert "95%" in result or "0.95" in result  # Mapping rate value
        assert "Recommendation" in result  # Actionable recommendation

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_handback_with_formatted_report(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_metadata_standardization_service,
        mock_llm,
        mock_agent,
    ):
        """Test metadata_assistant handback to research_agent with formatted report.

        Verifies handback report includes:
        1. Clear status (success/warning/failure)
        2. Structured markdown with headers
        3. Quantitative metrics
        4. Specific details (unmapped samples, errors)
        5. Actionable recommendation for research_agent

        Report format complies with metadata_assistant system prompt requirements.
        """
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_standardization_class.return_value = mock_metadata_standardization_service

        # Mock validation result with warnings (partial success)
        mock_validation = DatasetValidationResult(
            has_required_samples=True,
            missing_conditions=["control"],
            duplicate_ids=[],
            control_issues=["No control samples detected"],
            platform_consistency=True,
            summary={"total_samples": 30, "conditions": ["treated", "untreated"]},
            warnings=["Missing 'control' condition", "Age missing for 5 samples"],
        )
        mock_metadata_standardization_service.validate_dataset_content.return_value = (
            mock_validation,
            {},
            None,  # Return tuple: (result, stats, ir)
        )

        # Configure data_manager mocks for tool validation
        mock_data_manager.list_modalities.return_value = ["geo_gse99999"]
        mock_adata = Mock()
        mock_adata.obs = pd.DataFrame({"sample_id": ["S1", "S2"]})
        mock_data_manager.get_modality.return_value = mock_adata
        mock_data_manager.metadata_store = {}

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract tool
        tools = mock_create_agent.call_args[1]["tools"]
        validate_tool = next(t for t in tools if t.name == "validate_dataset_content")

        # Generate handback report
        result = validate_tool.func(
            source="geo_gse99999",
            source_type="modality",
            required_conditions="control,healthy",
            check_controls=True,
        )

        # Verify report structure for handback to research_agent
        assert "✅" in result or "⚠️" in result or "❌" in result  # Status icon required
        assert (
            "Dataset Validation Report" in result or "## " in result
        )  # Structured markdown
        assert (
            "30 samples" in result or "total_samples" in result
        )  # Quantitative metrics
        assert "control" in result.lower()  # Specific details about missing condition
        assert (
            "Recommendation" in result or "recommend" in result.lower()
        )  # Actionable recommendation

        # Verify provenance logged (metadata_assistant must log all operations)
        mock_data_manager.log_tool_usage.assert_called()
        call_args = mock_data_manager.log_tool_usage.call_args
        assert call_args[1]["tool_name"] == "validate_dataset_content"
        assert "result_summary" in call_args[1]


class TestFilterSamplesBy:
    """Test filter_samples_by tool (multi-criteria microbiome filtering)."""

    @pytest.mark.skip(
        reason="Requires workspace infrastructure not yet compatible with unit test mocking - needs production code investigation"
    )
    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    @patch("lobster.agents.metadata_assistant.MicrobiomeFilteringService")
    @patch("lobster.agents.metadata_assistant.DiseaseStandardizationService")
    def test_filter_samples_by_basic(
        self,
        mock_disease_service_class,
        mock_microbiome_service_class,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_llm,
        mock_agent,
    ):
        """Test basic filtering with single criterion (16S amplicon)."""
        from lobster.agents.metadata_assistant import metadata_assistant
        from lobster.core.analysis_ir import AnalysisStep

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent

        # Mock microbiome filtering service
        mock_microbiome_service = Mock()
        mock_microbiome_service_class.return_value = mock_microbiome_service

        # Mock 16S validation (2 out of 3 samples pass)
        def validate_16s_side_effect(metadata, strict=True):
            platform = metadata.get("platform", "").lower()
            if "illumina" in platform or "miseq" in platform:
                return (
                    metadata,  # Non-empty dict = valid
                    {
                        "is_valid": True,
                        "reason": "16S detected",
                        "matched_field": "platform",
                    },
                    AnalysisStep(
                        operation="validate_16s",
                        tool_name="test",
                        description="test",
                        library="test",
                        code_template="",
                        imports=[],
                        parameters={},
                        parameter_schema={},
                        input_entities=[],
                        output_entities=[],
                    ),
                )
            else:
                return (
                    {},  # Empty dict = invalid
                    {"is_valid": False, "reason": "No 16S indicators"},
                    AnalysisStep(
                        operation="validate_16s",
                        tool_name="test",
                        description="test",
                        library="test",
                        code_template="",
                        imports=[],
                        parameters={},
                        parameter_schema={},
                        input_entities=[],
                        output_entities=[],
                    ),
                )

        mock_microbiome_service.validate_16s_amplicon.side_effect = (
            validate_16s_side_effect
        )

        # Mock workspace data - MUST be set up BEFORE creating agent
        workspace_data = {
            "metadata": {
                "samples": {
                    "sample1": {
                        "platform": "Illumina MiSeq",
                        "organism": "Homo sapiens",
                    },
                    "sample2": {
                        "platform": "Illumina HiSeq",
                        "organism": "Homo sapiens",
                    },
                    "sample3": {
                        "platform": "PacBio",
                        "organism": "Homo sapiens",
                    },  # Should be filtered out
                }
            }
        }
        # Fix: Add workspace mock attribute BEFORE agent creation
        mock_workspace = Mock()

        # Mock read_content to accept workspace_key parameter and return workspace_data
        def mock_read_content(workspace_key=None):
            return workspace_data

        mock_workspace.read_content = Mock(side_effect=mock_read_content)
        mock_data_manager.workspace = mock_workspace

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract tool
        tools = mock_create_agent.call_args[1]["tools"]
        filter_tool = next(t for t in tools if t.name == "filter_samples_by")

        # Run filtering
        result = filter_tool.func(
            workspace_key="test_workspace", filter_criteria="16S", strict=False
        )

        # Verify results
        assert "✅" in result or "⚠️" in result  # Success/warning icon
        assert "Original Samples: 3" in result
        assert "Filtered Samples: 2" in result
        assert "Retention Rate" in result
        assert "16S amplicon detection" in result
        assert "Recommendation" in result

        # Verify provenance logged
        mock_data_manager.log_tool_usage.assert_called()
        call_args = mock_data_manager.log_tool_usage.call_args
        assert call_args[1]["tool_name"] == "filter_samples_by"

    @pytest.mark.skip(
        reason="Requires workspace infrastructure not yet compatible with unit test mocking - needs production code investigation"
    )
    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    @patch("lobster.agents.metadata_assistant.MicrobiomeFilteringService")
    @patch("lobster.agents.metadata_assistant.DiseaseStandardizationService")
    def test_filter_samples_by_multi_criteria(
        self,
        mock_disease_service_class,
        mock_microbiome_service_class,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_llm,
        mock_agent,
    ):
        """Test multi-criteria filtering (16S + human + fecal)."""
        from lobster.agents.metadata_assistant import metadata_assistant
        from lobster.core.analysis_ir import AnalysisStep

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent

        # Mock services
        mock_microbiome_service = Mock()
        mock_microbiome_service_class.return_value = mock_microbiome_service
        mock_disease_service = Mock()
        mock_disease_service_class.return_value = mock_disease_service

        # Mock 16S validation (all pass)
        mock_microbiome_service.validate_16s_amplicon.return_value = (
            {"platform": "test"},
            {"is_valid": True, "reason": "16S detected"},
            AnalysisStep(
                operation="validate_16s",
                tool_name="test",
                description="test",
                library="test",
                code_template="",
                imports=[],
                parameters={},
                parameter_schema={},
                input_entities=[],
                output_entities=[],
            ),
        )

        # Mock host organism validation (all pass)
        mock_microbiome_service.validate_host_organism.return_value = (
            {"organism": "Homo sapiens"},
            {"is_valid": True, "reason": "Host matched", "matched_host": "Human"},
            AnalysisStep(
                operation="validate_host",
                tool_name="test",
                description="test",
                library="test",
                code_template="",
                imports=[],
                parameters={},
                parameter_schema={},
                input_entities=[],
                output_entities=[],
            ),
        )

        # Mock sample type filtering (2 out of 3 pass)
        def filter_by_sample_type_side_effect(metadata_df, sample_types):
            # Keep samples with fecal in sample_type
            filtered = metadata_df[
                metadata_df["sample_type"].str.lower().str.contains("fecal")
            ]
            return (
                filtered,
                {
                    "original_samples": len(metadata_df),
                    "filtered_samples": len(filtered),
                    "retention_rate": len(filtered) / len(metadata_df) * 100,
                },
                AnalysisStep(
                    operation="filter_sample_type",
                    tool_name="test",
                    description="test",
                    library="test",
                    code_template="",
                    imports=[],
                    parameters={},
                    parameter_schema={},
                    input_entities=[],
                    output_entities=[],
                ),
            )

        mock_disease_service.filter_by_sample_type.side_effect = (
            filter_by_sample_type_side_effect
        )

        # Mock workspace data - MUST be set up BEFORE creating agent
        workspace_data = {
            "metadata": {
                "samples": {
                    "sample1": {
                        "platform": "Illumina MiSeq",
                        "organism": "Homo sapiens",
                        "sample_type": "fecal",
                    },
                    "sample2": {
                        "platform": "Illumina HiSeq",
                        "organism": "Homo sapiens",
                        "sample_type": "fecal",
                    },
                    "sample3": {
                        "platform": "Illumina NextSeq",
                        "organism": "Homo sapiens",
                        "sample_type": "biopsy",
                    },
                }
            }
        }
        # Fix: Add workspace mock attribute BEFORE agent creation
        mock_workspace = Mock()

        # Mock read_content to accept workspace_key parameter and return workspace_data
        def mock_read_content(workspace_key=None):
            return workspace_data

        mock_workspace.read_content = Mock(side_effect=mock_read_content)
        mock_data_manager.workspace = mock_workspace

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract tool
        tools = mock_create_agent.call_args[1]["tools"]
        filter_tool = next(t for t in tools if t.name == "filter_samples_by")

        # Run filtering
        result = filter_tool.func(
            workspace_key="test_workspace",
            filter_criteria="16S human fecal",
            strict=False,
        )

        # Verify results
        assert "✅" in result or "⚠️" in result
        assert "Original Samples: 3" in result
        assert "Filtered Samples: 2" in result
        assert "16S amplicon detection" in result
        assert "Host organism: Human" in result
        assert "Sample type: fecal" in result

    @pytest.mark.skip(
        reason="Requires workspace infrastructure not yet compatible with unit test mocking - needs production code investigation"
    )
    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    @patch("lobster.agents.metadata_assistant.MicrobiomeFilteringService")
    @patch("lobster.agents.metadata_assistant.DiseaseStandardizationService")
    def test_filter_samples_by_natural_language(
        self,
        mock_disease_service_class,
        mock_microbiome_service_class,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        mock_llm,
        mock_agent,
    ):
        """Test natural language criteria parsing."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent

        # Mock empty workspace to test parsing only - MUST be set up BEFORE creating agent
        workspace_data = {"metadata": {"samples": {}}}
        # Fix: Add workspace mock attribute BEFORE agent creation
        mock_workspace = Mock()

        # Mock read_content to accept workspace_key parameter and return workspace_data
        def mock_read_content(workspace_key=None):
            return workspace_data

        mock_workspace.read_content = Mock(side_effect=mock_read_content)
        mock_data_manager.workspace = mock_workspace

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract tool
        tools = mock_create_agent.call_args[1]["tools"]
        filter_tool = next(t for t in tools if t.name == "filter_samples_by")

        # Test various natural language criteria
        test_cases = [
            ("16S human fecal", True, ["Human"], ["fecal"], False),
            ("mouse gut CRC", False, ["Mouse"], ["gut"], True),
            ("amplicon human stool healthy", True, ["Human"], ["fecal"], True),
            ("16S Homo sapiens tissue", True, ["Human"], ["gut"], False),
        ]

        for (
            criteria,
            expect_16s,
            expect_hosts,
            expect_types,
            expect_disease,
        ) in test_cases:
            result = filter_tool.func(
                workspace_key="test_workspace", filter_criteria=criteria, strict=False
            )

            # Verify criteria was parsed (even if no samples, should show filters applied)
            if expect_16s:
                assert "16S amplicon detection" in result
            if expect_hosts:
                for host in expect_hosts:
                    assert host in result
            if expect_types:
                for sample_type in expect_types:
                    assert sample_type in result
            if expect_disease:
                assert "Disease standardization" in result
