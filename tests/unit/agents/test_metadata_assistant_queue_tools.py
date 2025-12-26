"""
Unit tests for metadata_assistant publication queue processing tools.

Tests the 3 new queue tools added in Phase 4c:
- process_metadata_entry
- process_metadata_queue
- update_metadata_status

These tools enable metadata_assistant to process publication queue entries,
apply filtering criteria, and aggregate results for CSV export.
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.publication_queue import PublicationQueue
from lobster.core.schemas.publication_queue import (
    HandoffStatus,
    PublicationQueueEntry,
    PublicationStatus,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_workspace(tmp_path):
    """Create temporary workspace with structure."""
    workspace = tmp_path / ".lobster_workspace"
    workspace.mkdir()
    (workspace / "metadata").mkdir()
    return workspace


@pytest.fixture
def real_queue_file(tmp_path):
    """Create temporary queue file."""
    return tmp_path / "pub_queue.jsonl"


@pytest.fixture
def real_queue(real_queue_file):
    """Create real PublicationQueue instance."""
    return PublicationQueue(real_queue_file)


@pytest.fixture
def mock_data_manager(mock_provider_config, temp_workspace, real_queue):
    """Create mock DataManagerV2 with real queue.

    Note: This fixture now requires mock_provider_config to ensure LLM
    creation works properly in the refactored provider system.
    """
    mock_dm = Mock(spec=DataManagerV2)
    mock_dm.publication_queue = real_queue
    mock_dm.workspace_path = temp_workspace
    mock_dm.metadata_store = {}
    mock_dm.log_tool_usage = Mock()
    return mock_dm


@pytest.fixture
def sample_workspace_metadata():
    """Sample SRA metadata structure from workspace (matches SRASampleSchema)."""
    return {
        "identifier": "sra_PRJNA123_samples",
        "content_type": "sra_samples",
        "data": {
            "samples": [
                {
                    "run_accession": "SRR001",
                    "experiment_accession": "SRX001",
                    "sample_accession": "SRS001",
                    "study_accession": "SRP001",
                    "bioproject": "PRJNA123",
                    "biosample": "SAMN001",
                    "organism_name": "Homo sapiens",
                    "organism_taxid": "9606",
                    "library_strategy": "AMPLICON",
                    "library_source": "METAGENOMIC",
                    "library_selection": "PCR",
                    "library_layout": "PAIRED",
                    "instrument": "Illumina MiSeq",
                    "instrument_model": "Illumina MiSeq",
                    "public_url": "https://test.com/SRR001",
                    "env_medium": "Stool",
                },
                {
                    "run_accession": "SRR002",
                    "experiment_accession": "SRX002",
                    "sample_accession": "SRS002",
                    "study_accession": "SRP002",
                    "bioproject": "PRJNA123",
                    "biosample": "SAMN002",
                    "organism_name": "Mus musculus",
                    "organism_taxid": "10090",
                    "library_strategy": "RNA-Seq",
                    "library_source": "TRANSCRIPTOMIC",
                    "library_selection": "cDNA",
                    "library_layout": "PAIRED",
                    "instrument": "Illumina NovaSeq",
                    "instrument_model": "NovaSeq 6000",
                    "ncbi_url": "https://test.com/SRR002",
                },
                {
                    "run_accession": "SRR003",
                    "experiment_accession": "SRX003",
                    "sample_accession": "SRS003",
                    "study_accession": "SRP003",
                    "bioproject": "PRJNA123",
                    "biosample": "SAMN003",
                    "organism_name": "Homo sapiens",
                    "organism_taxid": "9606",
                    "library_strategy": "AMPLICON",
                    "library_source": "METAGENOMIC",
                    "library_selection": "PCR",
                    "library_layout": "PAIRED",
                    "instrument": "Illumina MiSeq",
                    "instrument_model": "Illumina MiSeq",
                    "aws_url": "https://test.com/SRR003",
                    "env_medium": "Stool",
                },
            ],
            "sample_count": 3,
        },
    }


@pytest.fixture
def queue_entry_with_metadata_keys(real_queue):
    """Create queue entry with workspace_metadata_keys."""
    entry = PublicationQueueEntry(
        entry_id="test_entry_001",
        pmid="12345678",
        doi="10.1234/journal.2024",
        title="Test Publication",
        status=PublicationStatus.HANDOFF_READY,
        workspace_metadata_keys=["sra_PRJNA123_samples"],
        dataset_ids=["PRJNA123", "GSE123456"],
        extracted_identifiers={"bioproject": ["PRJNA123"], "geo": ["GSE123456"]},
    )
    real_queue.add_entry(entry)
    return entry


@pytest.fixture
def mock_workspace_service(sample_workspace_metadata):
    """Create mock WorkspaceContentService."""
    mock_ws = Mock()
    mock_ws.read_content = Mock(return_value=sample_workspace_metadata)
    mock_ws.write_content = Mock(return_value="/workspace/metadata/output.json")
    return mock_ws


@pytest.fixture
def patched_workspace_service(sample_workspace_metadata):
    """Patch WorkspaceContentService for tests."""
    with patch(
        "lobster.agents.metadata_assistant.WorkspaceContentService"
    ) as mock_ws_class:
        mock_ws_instance = Mock()
        mock_ws_instance.read_content = Mock(return_value=sample_workspace_metadata)
        mock_ws_instance.write_content = Mock(
            return_value="/workspace/metadata/output.json"
        )
        mock_ws_class.return_value = mock_ws_instance
        yield mock_ws_class


@pytest.fixture
def mock_microbiome_filtering():
    """Mock MicrobiomeFilteringService for predictable filtering."""
    mock_service = Mock()
    # Mock validate_16s_amplicon: returns (filtered_dict, stats, ir)
    # Returns dict if valid, empty dict if invalid
    mock_service.validate_16s_amplicon.return_value = (
        {"sample_id": "S1", "library_strategy": "AMPLICON"},
        {"is_valid": True, "reason": "16S amplicon detected"},
        Mock(),  # AnalysisStep IR
    )
    mock_service.validate_host_organism.return_value = (
        {"sample_id": "S1", "organism": "Homo sapiens"},
        {"is_valid": True, "matched_host": "Human", "confidence_score": 95.0},
        Mock(),
    )
    return mock_service


# =============================================================================
# Test Classes
# =============================================================================


class TestProcessMetadataEntry:
    """Test process_metadata_entry tool."""

    @patch(
        "lobster.services.data_access.workspace_content_service.WorkspaceContentService"
    )
    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_process_entry_success(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_ws_class,
        mock_data_manager,
        queue_entry_with_metadata_keys,
        sample_workspace_metadata,
    ):
        """Process single entry with filter criteria."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance

        mock_llm = Mock()
        mock_llm.with_config = Mock(return_value=mock_llm)
        mock_create_llm.return_value = mock_llm

        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent

        # Mock WorkspaceContentService
        mock_ws_instance = Mock()
        mock_ws_instance.read_content = Mock(return_value=sample_workspace_metadata)
        mock_ws_class.return_value = mock_ws_instance

        # Create agent
        agent = metadata_assistant(data_manager=mock_data_manager)

        # Extract process_metadata_entry tool
        tools = mock_create_agent.call_args[1]["tools"]
        process_entry_tool = next(
            (t for t in tools if t.name == "process_metadata_entry"), None
        )
        assert process_entry_tool is not None, "process_metadata_entry tool not found"

        # Call tool
        result = process_entry_tool.func(
            entry_id="test_entry_001", filter_criteria="16S human fecal"
        )

        # Verify success message
        assert "Entry Processed:" in result
        assert "test_entry_001" in result
        assert "Samples Extracted:" in result or "Samples Extracted**:" in result
        assert "Samples Valid:" in result or "Samples Valid**:" in result

        # Verify queue entry updated
        updated_entry = mock_data_manager.publication_queue.get_entry("test_entry_001")
        assert updated_entry.handoff_status == HandoffStatus.METADATA_COMPLETE
        assert updated_entry.harmonization_metadata is not None
        assert "samples" in updated_entry.harmonization_metadata
        assert "filter_criteria" in updated_entry.harmonization_metadata
        assert "stats" in updated_entry.harmonization_metadata
        # Verify new stats structure
        stats = updated_entry.harmonization_metadata["stats"]
        assert "samples_extracted" in stats
        assert "samples_valid" in stats
        assert "validation_errors" in stats
        assert "validation_warnings" in stats

    @patch(
        "lobster.services.data_access.workspace_content_service.WorkspaceContentService"
    )
    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_process_entry_no_workspace_keys(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_ws_class,
        mock_data_manager,
        real_queue,
    ):
        """Error when entry has no workspace_metadata_keys."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance

        mock_llm = Mock()
        mock_llm.with_config = Mock(return_value=mock_llm)
        mock_create_llm.return_value = mock_llm

        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent

        # Create entry WITHOUT workspace_metadata_keys
        entry = PublicationQueueEntry(
            entry_id="test_entry_no_keys",
            pmid="12345678",
            status=PublicationStatus.COMPLETED,
            workspace_metadata_keys=[],  # Empty
        )
        real_queue.add_entry(entry)

        # Create agent
        agent = metadata_assistant(data_manager=mock_data_manager)

        # Extract tool
        tools = mock_create_agent.call_args[1]["tools"]
        process_entry_tool = next(
            t for t in tools if t.name == "process_metadata_entry"
        )

        # Call tool - should return error
        result = process_entry_tool.func(entry_id="test_entry_no_keys")

        assert "❌ Error" in result
        assert "no workspace_metadata_keys" in result


class TestProcessMetadataQueue:
    """Test process_metadata_queue tool."""

    @patch(
        "lobster.services.data_access.workspace_content_service.WorkspaceContentService"
    )
    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_process_queue_multiple_entries(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_ws_class,
        mock_data_manager,
        real_queue,
        sample_workspace_metadata,
    ):
        """Batch process multiple HANDOFF_READY entries."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance

        mock_llm = Mock()
        mock_llm.with_config = Mock(return_value=mock_llm)
        mock_create_llm.return_value = mock_llm

        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent

        # Mock WorkspaceContentService
        mock_ws_instance = Mock()
        mock_ws_instance.read_content = Mock(return_value=sample_workspace_metadata)
        mock_ws_instance.write_content = Mock(
            return_value="/workspace/metadata/output.json"
        )
        mock_ws_class.return_value = mock_ws_instance

        # Create 3 entries with HANDOFF_READY status
        for i in range(3):
            entry = PublicationQueueEntry(
                entry_id=f"test_entry_{i:03d}",
                pmid=f"1234567{i}",
                status=PublicationStatus.HANDOFF_READY,
                workspace_metadata_keys=[f"sra_PRJNA{i}_samples"],
                dataset_ids=[f"PRJNA{i}"],
            )
            real_queue.add_entry(entry)

        # Create agent
        agent = metadata_assistant(data_manager=mock_data_manager)

        # Extract tool
        tools = mock_create_agent.call_args[1]["tools"]
        process_queue_tool = next(
            t for t in tools if t.name == "process_metadata_queue"
        )

        # Call tool
        result = process_queue_tool.func(
            status_filter="handoff_ready", filter_criteria="16S human fecal"
        )

        # Verify success message
        assert "Queue Processing Complete" in result
        assert "Entries Processed**:" in result or "Entries Processed:" in result

        # Verify all entries updated to METADATA_COMPLETE
        for i in range(3):
            entry = real_queue.get_entry(f"test_entry_{i:03d}")
            assert entry.handoff_status == HandoffStatus.METADATA_COMPLETE

    @patch(
        "lobster.services.data_access.workspace_content_service.WorkspaceContentService"
    )
    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_process_queue_respects_max_entries(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_ws_class,
        mock_data_manager,
        real_queue,
        sample_workspace_metadata,
    ):
        """max_entries parameter limits processing."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance

        mock_llm = Mock()
        mock_llm.with_config = Mock(return_value=mock_llm)
        mock_create_llm.return_value = mock_llm

        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent

        # Mock WorkspaceContentService
        mock_ws_instance = Mock()
        mock_ws_instance.read_content = Mock(return_value=sample_workspace_metadata)
        mock_ws_instance.write_content = Mock(
            return_value="/workspace/metadata/output.json"
        )
        mock_ws_class.return_value = mock_ws_instance

        # Create 10 HANDOFF_READY entries
        for i in range(10):
            entry = PublicationQueueEntry(
                entry_id=f"test_entry_{i:03d}",
                pmid=f"1234567{i}",
                status=PublicationStatus.HANDOFF_READY,
                workspace_metadata_keys=[f"sra_PRJNA{i}_samples"],
            )
            real_queue.add_entry(entry)

        # Create agent
        agent = metadata_assistant(data_manager=mock_data_manager)

        # Extract tool
        tools = mock_create_agent.call_args[1]["tools"]
        process_queue_tool = next(
            t for t in tools if t.name == "process_metadata_queue"
        )

        # Call with max_entries=5
        result = process_queue_tool.func(status_filter="handoff_ready", max_entries=5)

        # Verify only 5 processed
        assert "Entries Processed**:" in result or "Entries Processed:" in result
        assert "5" in result  # Should have 5 entries

        # Verify first 5 entries updated, last 5 unchanged
        for i in range(5):
            entry = real_queue.get_entry(f"test_entry_{i:03d}")
            assert entry.handoff_status == HandoffStatus.METADATA_COMPLETE

        for i in range(5, 10):
            entry = real_queue.get_entry(f"test_entry_{i:03d}")
            assert entry.handoff_status == HandoffStatus.NOT_READY  # Unchanged

    @patch(
        "lobster.services.data_access.workspace_content_service.WorkspaceContentService"
    )
    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_process_queue_stores_in_metadata_store(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_ws_class,
        mock_data_manager,
        queue_entry_with_metadata_keys,
        sample_workspace_metadata,
    ):
        """Aggregated samples stored for write_to_workspace access."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance

        mock_llm = Mock()
        mock_llm.with_config = Mock(return_value=mock_llm)
        mock_create_llm.return_value = mock_llm

        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent

        # Mock WorkspaceContentService
        mock_ws_instance = Mock()
        mock_ws_instance.read_content = Mock(return_value=sample_workspace_metadata)
        mock_ws_instance.write_content = Mock(
            return_value="/workspace/metadata/output.json"
        )
        mock_ws_class.return_value = mock_ws_instance

        # Create agent
        agent = metadata_assistant(data_manager=mock_data_manager)

        # Extract tool
        tools = mock_create_agent.call_args[1]["tools"]
        process_queue_tool = next(
            t for t in tools if t.name == "process_metadata_queue"
        )

        # Call tool (no filter - test metadata_store storage)
        output_key = "aggregated_filtered_samples"
        result = process_queue_tool.func(
            status_filter="handoff_ready",
            filter_criteria=None,  # No filter for this test
            output_key=output_key,
        )

        # Verify metadata_store updated
        assert output_key in mock_data_manager.metadata_store
        stored_data = mock_data_manager.metadata_store[output_key]
        assert "samples" in stored_data
        assert "filter_criteria" in stored_data
        assert "stats" in stored_data


class TestUpdateMetadataStatus:
    """Test update_metadata_status tool."""

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_update_handoff_status(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        queue_entry_with_metadata_keys,
    ):
        """Manually update handoff_status."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance

        mock_llm = Mock()
        mock_llm.with_config = Mock(return_value=mock_llm)
        mock_create_llm.return_value = mock_llm

        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent

        # Set entry to in_progress
        mock_data_manager.publication_queue.update_status(
            "test_entry_001",
            PublicationStatus.HANDOFF_READY,
            handoff_status=HandoffStatus.METADATA_IN_PROGRESS,
        )

        # Create agent
        agent = metadata_assistant(data_manager=mock_data_manager)

        # Extract tool
        tools = mock_create_agent.call_args[1]["tools"]
        update_status_tool = next(
            t for t in tools if t.name == "update_metadata_status"
        )

        # Call tool to mark complete
        result = update_status_tool.func(
            entry_id="test_entry_001", handoff_status="metadata_complete"
        )

        # Verify success
        assert "✓ Updated" in result
        assert "metadata_complete" in result

        # Verify entry updated
        entry = mock_data_manager.publication_queue.get_entry("test_entry_001")
        assert entry.handoff_status == HandoffStatus.METADATA_COMPLETE

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_update_with_error_message(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
        queue_entry_with_metadata_keys,
    ):
        """Add error message to entry."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance

        mock_llm = Mock()
        mock_llm.with_config = Mock(return_value=mock_llm)
        mock_create_llm.return_value = mock_llm

        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent

        # Create agent
        agent = metadata_assistant(data_manager=mock_data_manager)

        # Extract tool
        tools = mock_create_agent.call_args[1]["tools"]
        update_status_tool = next(
            t for t in tools if t.name == "update_metadata_status"
        )

        # Call tool with error message
        error_msg = "Filtering failed: no samples matched criteria"
        result = update_status_tool.func(
            entry_id="test_entry_001", error_message=error_msg
        )

        # Verify entry has error logged
        entry = mock_data_manager.publication_queue.get_entry("test_entry_001")
        assert len(entry.error_log) > 0
        assert error_msg in entry.error_log[-1]


class TestSharedWorkspaceToolsInAgent:
    """Test that shared workspace tools are available in metadata_assistant."""

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_write_to_workspace_available(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
    ):
        """Verify write_to_workspace tool is available."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance

        mock_llm = Mock()
        mock_llm.with_config = Mock(return_value=mock_llm)
        mock_create_llm.return_value = mock_llm

        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent

        # Create agent
        agent = metadata_assistant(data_manager=mock_data_manager)

        # Verify write_to_workspace in tools
        tools = mock_create_agent.call_args[1]["tools"]
        tool_names = [t.name for t in tools]

        assert "write_to_workspace" in tool_names
        assert "get_content_from_workspace" in tool_names

    @patch("lobster.agents.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.get_settings")
    @patch("lobster.agents.metadata_assistant.MetadataStandardizationService")
    @patch("lobster.agents.metadata_assistant.SampleMappingService")
    def test_tool_count_updated(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        mock_data_manager,
    ):
        """Verify tool count includes new queue tools + shared tools."""
        from lobster.agents.metadata_assistant import metadata_assistant

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance

        mock_llm = Mock()
        mock_llm.with_config = Mock(return_value=mock_llm)
        mock_create_llm.return_value = mock_llm

        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent

        # Create agent
        agent = metadata_assistant(data_manager=mock_data_manager)

        # Extract tools
        tools = mock_create_agent.call_args[1]["tools"]
        tool_names = [t.name for t in tools]

        # Expected tools:
        # Original: map_samples_by_id, read_sample_metadata, standardize_sample_metadata,
        #           validate_dataset_content, filter_samples_by
        # New queue: process_metadata_entry, process_metadata_queue, update_metadata_status
        # Shared: get_content_from_workspace, write_to_workspace
        # Total: 10 tools

        expected_tools = [
            "map_samples_by_id",
            "read_sample_metadata",
            "standardize_sample_metadata",
            "validate_dataset_content",
            "filter_samples_by",
            "process_metadata_entry",
            "process_metadata_queue",
            "update_metadata_status",
            "get_content_from_workspace",
            "write_to_workspace",
        ]

        for expected in expected_tools:
            assert expected in tool_names, f"Missing tool: {expected}"
