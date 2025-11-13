"""
Unit tests for metadata_assistant agent.

Tests all 4 metadata tools independently with mock services:
- map_samples_by_id
- read_sample_metadata
- standardize_sample_metadata
- validate_dataset_content
"""

import json
from datetime import date
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.schemas.transcriptomics import TranscriptomicsMetadataSchema
from lobster.tools.metadata_standardization_service import (
    DatasetValidationResult,
    StandardizationResult,
)
from lobster.tools.sample_mapping_service import (
    SampleMappingResult,
    SampleMatch,
    UnmappedSample,
)


@pytest.fixture
def mock_data_manager():
    """Create mock DataManagerV2 for testing."""
    mock_dm = Mock(spec=DataManagerV2)
    mock_dm.log_tool_usage = Mock()
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
        assert len(call_kwargs["tools"]) == 4  # 4 base tools
        assert call_kwargs["name"] == "metadata_assistant"
        assert "metadata librarian" in call_kwargs["prompt"].lower()

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
    def test_init_with_handoff_tools(
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
        """Test agent initialization with handoff tools."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent

        handoff_tool1 = Mock()
        handoff_tool2 = Mock()
        handoff_tools = [handoff_tool1, handoff_tool2]

        # Create agent
        agent = metadata_assistant(
            data_manager=mock_data_manager, handoff_tools=handoff_tools
        )

        # Verify handoff tools are included
        call_kwargs = mock_create_agent.call_args[1]
        assert len(call_kwargs["tools"]) == 6  # 4 base tools + 2 handoff tools
        assert handoff_tool1 in call_kwargs["tools"]
        assert handoff_tool2 in call_kwargs["tools"]

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

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the map_samples_by_id tool
        tools = mock_create_agent.call_args[1]["tools"]
        map_tool = next(t for t in tools if t.name == "map_samples_by_id")

        # Call the tool
        result = map_tool.func(
            source_identifier="geo_gse12345",
            target_identifier="geo_gse67890",
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

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        map_tool = next(t for t in tools if t.name == "map_samples_by_id")

        # Call with specific strategies
        result = map_tool.func(
            source_identifier="dataset1",
            target_identifier="dataset2",
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

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        map_tool = next(t for t in tools if t.name == "map_samples_by_id")

        # Call with invalid strategy
        result = map_tool.func(
            source_identifier="dataset1",
            target_identifier="dataset2",
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

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        map_tool = next(t for t in tools if t.name == "map_samples_by_id")

        # Call the tool
        result = map_tool.func(
            source_identifier="nonexistent", target_identifier="dataset2"
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

        # Service returns summary string
        mock_metadata_standardization_service.read_sample_metadata.return_value = (
            "Dataset: geo_gse12345\nTotal Samples: 10"
        )

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        read_tool = next(t for t in tools if t.name == "read_sample_metadata")

        # Call the tool
        result = read_tool.func(identifier="geo_gse12345", return_format="summary")

        # Verify service was called correctly
        mock_metadata_standardization_service.read_sample_metadata.assert_called_once_with(
            identifier="geo_gse12345", fields=None, return_format="summary"
        )

        # Verify provenance logging
        mock_data_manager.log_tool_usage.assert_called_once()
        log_call = mock_data_manager.log_tool_usage.call_args
        assert log_call[1]["tool_name"] == "read_sample_metadata"
        assert log_call[1]["parameters"]["return_format"] == "summary"

        assert "Dataset: geo_gse12345" in result
        assert "Total Samples: 10" in result

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

        # Service returns dict
        mock_metadata_standardization_service.read_sample_metadata.return_value = {
            "identifier": "geo_gse12345",
            "total_samples": 10,
            "fields": ["condition", "organism"],
        }

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        read_tool = next(t for t in tools if t.name == "read_sample_metadata")

        # Call the tool
        result = read_tool.func(identifier="geo_gse12345", return_format="detailed")

        # Verify result is JSON string
        parsed = json.loads(result)
        assert parsed["identifier"] == "geo_gse12345"
        assert parsed["total_samples"] == 10

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

        # Service returns DataFrame
        mock_df = pd.DataFrame(
            {"sample_id": ["S1", "S2"], "condition": ["Control", "Treatment"]}
        )
        mock_metadata_standardization_service.read_sample_metadata.return_value = (
            mock_df
        )

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        read_tool = next(t for t in tools if t.name == "read_sample_metadata")

        # Call the tool
        result = read_tool.func(identifier="geo_gse12345", return_format="schema")

        # Verify DataFrame was converted to markdown
        assert "sample_id" in result
        assert "condition" in result
        assert "Control" in result

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

        mock_metadata_standardization_service.read_sample_metadata.return_value = (
            "Summary with filtered fields"
        )

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        read_tool = next(t for t in tools if t.name == "read_sample_metadata")

        # Call with field filtering
        result = read_tool.func(identifier="geo_gse12345", fields="condition,organism")

        # Verify fields were parsed correctly
        mock_metadata_standardization_service.read_sample_metadata.assert_called_once()
        call_kwargs = (
            mock_metadata_standardization_service.read_sample_metadata.call_args[1]
        )
        assert call_kwargs["fields"] == ["condition", "organism"]

        assert "Summary with filtered fields" in result

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
        """Test reading metadata with ValueError from service."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_create_llm.return_value = mock_llm
        mock_create_agent.return_value = mock_agent
        mock_standardization_class.return_value = mock_metadata_standardization_service

        # Service raises ValueError
        mock_metadata_standardization_service.read_sample_metadata.side_effect = (
            ValueError("Dataset not found")
        )

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        read_tool = next(t for t in tools if t.name == "read_sample_metadata")

        # Call the tool
        result = read_tool.func(identifier="nonexistent")

        # Verify error message
        assert "❌ Failed to read metadata" in result
        assert "Dataset not found" in result


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
            mock_result
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
            identifier="geo_gse12345", target_schema="transcriptomics"
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
            mock_result
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
            identifier="geo_gse12345",
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
            identifier="geo_gse12345",
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
            identifier="geo_gse12345", target_schema="invalid_schema"
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
            mock_result
        )

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        validate_tool = next(t for t in tools if t.name == "validate_dataset_content")

        # Call the tool
        result = validate_tool.func(identifier="geo_gse12345", expected_samples=10)

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
            mock_result
        )

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        validate_tool = next(t for t in tools if t.name == "validate_dataset_content")

        # Call the tool
        result = validate_tool.func(identifier="geo_gse12345", expected_samples=10)

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
            mock_result
        )

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        validate_tool = next(t for t in tools if t.name == "validate_dataset_content")

        # Call with required conditions
        result = validate_tool.func(
            identifier="geo_gse12345", required_conditions="Control,Treatment"
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

        # Create agent
        metadata_assistant(data_manager=mock_data_manager)

        # Extract the tool
        tools = mock_create_agent.call_args[1]["tools"]
        validate_tool = next(t for t in tools if t.name == "validate_dataset_content")

        # Call the tool
        result = validate_tool.func(identifier="nonexistent")

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

        # Verify date formatting
        today = date.today().isoformat()
        assert today in system_prompt

        # Verify key prompt sections
        assert "metadata librarian" in system_prompt.lower()
        assert "cross-dataset" in system_prompt.lower()
        assert "map_samples_by_id" in system_prompt.lower()
        assert "read_sample_metadata" in system_prompt.lower()
        assert "standardize_sample_metadata" in system_prompt.lower()
        assert "validate_dataset_content" in system_prompt.lower()


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
            source_identifier="dataset1", target_identifier="dataset2"
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
        result = read_tool.func(identifier="dataset")

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
            identifier="dataset", target_schema="transcriptomics"
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
        result = validate_tool.func(identifier="dataset")

        # Verify error message
        assert "❌ Unexpected error during validation" in result
        assert "Unexpected error" in result
