"""
Integration tests for metadata_assistant with publication queue.

Tests complete workflows:
- metadata_assistant processing queue entries
- Updating harmonization_metadata in queue
- CSV export from aggregated samples
- Shared workspace tool usage

These tests use real DataManagerV2 and PublicationQueue instances
with mocked external services (workspace content, filtering).
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
def integration_workspace(tmp_path):
    """Create real workspace with full structure."""
    workspace = tmp_path / ".lobster_workspace"
    workspace.mkdir()
    (workspace / "metadata").mkdir()
    (workspace / "literature").mkdir()
    (workspace / "data").mkdir()
    return workspace


@pytest.fixture
def integration_data_manager(integration_workspace):
    """Create real DataManagerV2 instance."""
    return DataManagerV2(workspace_path=integration_workspace)


@pytest.fixture
def sample_sra_metadata_files(integration_workspace):
    """Create sample SRA metadata files in workspace."""
    metadata_dir = integration_workspace / "metadata"

    # Create sample SRA metadata file (matches SRASampleSchema)
    sra_data = {
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
            ],
            "sample_count": 2,
        },
    }

    sra_file = metadata_dir / "sra_PRJNA123_samples.json"
    sra_file.write_text(json.dumps(sra_data, indent=2))

    return ["sra_PRJNA123_samples"]


@pytest.fixture
def queue_entry_with_metadata(integration_data_manager, sample_sra_metadata_files):
    """Create queue entry with workspace_metadata_keys pointing to real files."""
    entry = PublicationQueueEntry(
        entry_id="test_entry_001",
        pmid="12345678",
        doi="10.1234/journal.2024",
        title="Test Microbiome Study",
        status=PublicationStatus.HANDOFF_READY,
        workspace_metadata_keys=sample_sra_metadata_files,
        dataset_ids=["PRJNA123", "GSE123456"],
        extracted_identifiers={"bioproject": ["PRJNA123"], "geo": ["GSE123456"]},
    )

    integration_data_manager.publication_queue.add_entry(entry)
    return entry


# =============================================================================
# Integration Tests
# =============================================================================


class TestMetadataAssistantQueueIntegration:
    """Integration tests for metadata_assistant updating publication queue."""

    @pytest.mark.integration
    @patch("lobster.agents.metadata_assistant.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.metadata_assistant.get_settings")
    @patch(
        "lobster.agents.metadata_assistant.metadata_assistant.MetadataStandardizationService"
    )
    @patch("lobster.agents.metadata_assistant.metadata_assistant.SampleMappingService")
    @patch(
        "lobster.agents.metadata_assistant.metadata_assistant.MicrobiomeFilteringService"
    )
    def test_process_entry_updates_harmonization_metadata(
        self,
        mock_microbiome_class,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        integration_data_manager,
        queue_entry_with_metadata,
    ):
        """Test processing entry updates harmonization_metadata field."""
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

        # Mock filtering to return human samples only
        mock_filtering = Mock()
        mock_filtering.validate_16s_amplicon.return_value = (
            {"run_accession": "SRR001"},  # Human sample passes
            {"is_valid": True},
            Mock(),
        )
        mock_filtering.validate_host_organism.return_value = (
            {"run_accession": "SRR001"},
            {"is_valid": True, "matched_host": "Human"},
            Mock(),
        )
        mock_microbiome_class.return_value = mock_filtering

        # Create agent
        agent = metadata_assistant(data_manager=integration_data_manager)

        # Extract process_metadata_entry tool
        tools = mock_create_agent.call_args[1]["tools"]
        process_entry_tool = next(
            t for t in tools if t.name == "process_metadata_entry"
        )

        # Call tool
        result = process_entry_tool.func(
            entry_id="test_entry_001", filter_criteria="16S human fecal"
        )

        # Verify result message
        assert "Entry Processed" in result
        assert "test_entry_001" in result

        # Verify queue entry updated with harmonization_metadata
        updated_entry = integration_data_manager.publication_queue.get_entry(
            "test_entry_001"
        )
        assert updated_entry.handoff_status == HandoffStatus.METADATA_COMPLETE
        assert updated_entry.harmonization_metadata is not None

        harmonization = updated_entry.harmonization_metadata
        assert "samples" in harmonization
        assert "filter_criteria" in harmonization
        assert harmonization["filter_criteria"] == "16S human fecal"
        assert "stats" in harmonization

    @pytest.mark.integration
    @patch("lobster.agents.metadata_assistant.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.metadata_assistant.get_settings")
    @patch(
        "lobster.agents.metadata_assistant.metadata_assistant.MetadataStandardizationService"
    )
    @patch("lobster.agents.metadata_assistant.metadata_assistant.SampleMappingService")
    def test_process_queue_updates_all_entries(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        integration_data_manager,
        sample_sra_metadata_files,
    ):
        """Batch processing updates all matching entries."""
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

        # Create 3 entries with HANDOFF_READY status
        for i in range(3):
            entry = PublicationQueueEntry(
                entry_id=f"test_entry_{i:03d}",
                pmid=f"1234567{i}",
                doi=f"10.1234/journal.{i}",
                title=f"Test Publication {i}",
                status=PublicationStatus.HANDOFF_READY,
                workspace_metadata_keys=sample_sra_metadata_files,
                dataset_ids=[f"PRJNA{i}"],
            )
            integration_data_manager.publication_queue.add_entry(entry)

        # Create agent
        agent = metadata_assistant(data_manager=integration_data_manager)

        # Extract process_metadata_queue tool
        tools = mock_create_agent.call_args[1]["tools"]
        process_queue_tool = next(
            t for t in tools if t.name == "process_metadata_queue"
        )

        # Call tool
        result = process_queue_tool.func(
            status_filter="handoff_ready",
            filter_criteria=None,  # No filtering
            output_key="aggregated_samples",
        )

        # Verify result
        assert "Queue Processing Complete" in result
        assert "Entries Processed**:" in result or "Entries Processed:" in result
        assert "3" in result  # Should have 3 entries

        # Verify all entries updated to METADATA_COMPLETE
        for i in range(3):
            entry = integration_data_manager.publication_queue.get_entry(
                f"test_entry_{i:03d}"
            )
            assert entry.handoff_status == HandoffStatus.METADATA_COMPLETE

        # Verify aggregated output in metadata_store
        assert "aggregated_samples" in integration_data_manager.metadata_store

    @pytest.mark.integration
    @patch("lobster.agents.metadata_assistant.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.metadata_assistant.get_settings")
    @patch(
        "lobster.agents.metadata_assistant.metadata_assistant.MetadataStandardizationService"
    )
    @patch("lobster.agents.metadata_assistant.metadata_assistant.SampleMappingService")
    @patch("lobster.tools.workspace_tool.WorkspaceContentService")
    def test_write_to_workspace_creates_csv(
        self,
        mock_ws_service_class,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        integration_data_manager,
    ):
        """Test CSV export from aggregated samples."""
        from lobster.agents.metadata_assistant import metadata_assistant
        from lobster.tools.workspace_tool import create_write_to_workspace_tool

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_llm = Mock()
        mock_llm.with_config = Mock(return_value=mock_llm)
        mock_create_llm.return_value = mock_llm
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent

        # Create aggregated samples in metadata_store
        integration_data_manager.metadata_store["aggregated_samples"] = {
            "samples": [
                {
                    "publication_entry_id": "pub_001",
                    "run_accession": "SRR001",
                    "organism": "Homo sapiens",
                    "sample_type": "fecal",
                },
                {
                    "publication_entry_id": "pub_002",
                    "run_accession": "SRR002",
                    "organism": "Homo sapiens",
                    "sample_type": "fecal",
                },
            ],
            "filter_criteria": "16S human fecal",
        }

        # Mock workspace service to return CSV path
        mock_ws_instance = Mock()
        mock_ws_instance.write_content = Mock(
            return_value=str(
                integration_data_manager.workspace_path
                / "metadata"
                / "aggregated_samples.csv"
            )
        )
        mock_ws_service_class.return_value = mock_ws_instance

        # Create write_to_workspace tool
        write_tool = create_write_to_workspace_tool(integration_data_manager)

        # Call tool with CSV format
        result = write_tool.func(
            identifier="aggregated_samples",
            workspace="metadata",
            output_format="csv",
        )

        # Verify CSV message
        assert "Content Cached Successfully" in result
        assert "CSV" in result or "csv" in result
        assert "aggregated_samples.csv" in result

        # Verify CSV file was created (integration test uses real WorkspaceContentService)
        csv_file = (
            integration_data_manager.workspace_path
            / "metadata"
            / "aggregated_samples.csv"
        )
        assert csv_file.exists(), "CSV file should be created"


class TestSharedWorkspaceToolsIntegration:
    """Test shared workspace tools across agents."""

    @pytest.mark.integration
    @patch("lobster.agents.research.research_agent.create_react_agent")
    @patch("lobster.agents.research.research_agent.create_llm")
    @patch("lobster.agents.research.research_agent.get_settings")
    def test_research_agent_has_write_to_workspace(
        self,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        integration_data_manager,
    ):
        """Verify research_agent can use write_to_workspace."""
        from lobster.agents.research import research_agent

        # Setup mocks
        mock_settings_instance = Mock()
        mock_settings_instance.get_agent_llm_params.return_value = {}
        mock_settings.return_value = mock_settings_instance
        mock_llm = Mock()
        mock_create_llm.return_value = mock_llm
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent

        # Create agent
        agent = research_agent(data_manager=integration_data_manager)

        # Extract tools
        tools = mock_create_agent.call_args[1]["tools"]
        tool_names = [t.name for t in tools]

        # Verify write_to_workspace available
        assert "write_to_workspace" in tool_names
        assert "get_content_from_workspace" in tool_names

    @pytest.mark.integration
    @patch("lobster.agents.metadata_assistant.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.metadata_assistant.get_settings")
    @patch(
        "lobster.agents.metadata_assistant.metadata_assistant.MetadataStandardizationService"
    )
    @patch("lobster.agents.metadata_assistant.metadata_assistant.SampleMappingService")
    def test_metadata_assistant_has_write_to_workspace(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        integration_data_manager,
    ):
        """Verify metadata_assistant can use write_to_workspace."""
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
        agent = metadata_assistant(data_manager=integration_data_manager)

        # Extract tools
        tools = mock_create_agent.call_args[1]["tools"]
        tool_names = [t.name for t in tools]

        # Verify workspace tools available
        assert "write_to_workspace" in tool_names
        assert "get_content_from_workspace" in tool_names
        assert "process_metadata_entry" in tool_names
        assert "process_metadata_queue" in tool_names
        assert "update_metadata_status" in tool_names


class TestQueueStatusTransitions:
    """Test queue status transitions during metadata processing."""

    @pytest.mark.integration
    @patch("lobster.agents.metadata_assistant.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.metadata_assistant.get_settings")
    @patch(
        "lobster.agents.metadata_assistant.metadata_assistant.MetadataStandardizationService"
    )
    @patch("lobster.agents.metadata_assistant.metadata_assistant.SampleMappingService")
    def test_handoff_ready_to_metadata_complete(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        integration_data_manager,
        queue_entry_with_metadata,
    ):
        """Test status transition: HANDOFF_READY → METADATA_IN_PROGRESS → METADATA_COMPLETE."""
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

        # Verify initial status
        entry = integration_data_manager.publication_queue.get_entry("test_entry_001")
        assert entry.status == PublicationStatus.HANDOFF_READY
        assert entry.handoff_status == HandoffStatus.NOT_READY

        # Create agent
        agent = metadata_assistant(data_manager=integration_data_manager)

        # Extract process_metadata_entry tool
        tools = mock_create_agent.call_args[1]["tools"]
        process_entry_tool = next(
            t for t in tools if t.name == "process_metadata_entry"
        )

        # Call tool
        result = process_entry_tool.func(entry_id="test_entry_001")

        # Verify final status
        entry = integration_data_manager.publication_queue.get_entry("test_entry_001")
        assert entry.handoff_status == HandoffStatus.METADATA_COMPLETE
        assert entry.harmonization_metadata is not None


class TestWorkspaceToQueueRoundtrip:
    """Test complete roundtrip: workspace → processing → queue update."""

    @pytest.mark.integration
    @patch("lobster.agents.metadata_assistant.metadata_assistant.create_react_agent")
    @patch("lobster.agents.metadata_assistant.metadata_assistant.create_llm")
    @patch("lobster.agents.metadata_assistant.metadata_assistant.get_settings")
    @patch(
        "lobster.agents.metadata_assistant.metadata_assistant.MetadataStandardizationService"
    )
    @patch("lobster.agents.metadata_assistant.metadata_assistant.SampleMappingService")
    def test_complete_workflow(
        self,
        mock_mapping_class,
        mock_standardization_class,
        mock_settings,
        mock_create_llm,
        mock_create_agent,
        integration_data_manager,
        queue_entry_with_metadata,
    ):
        """Complete workflow: read workspace → filter → update queue → store aggregated."""
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
        agent = metadata_assistant(data_manager=integration_data_manager)

        # Extract process_metadata_queue tool
        tools = mock_create_agent.call_args[1]["tools"]
        process_queue_tool = next(
            t for t in tools if t.name == "process_metadata_queue"
        )

        # Call tool
        result = process_queue_tool.func(
            status_filter="handoff_ready",
            output_key="test_aggregated",
        )

        # Verify processing succeeded
        assert "Queue Processing Complete" in result

        # Verify aggregated output stored
        assert "test_aggregated" in integration_data_manager.metadata_store
        aggregated = integration_data_manager.metadata_store["test_aggregated"]
        assert "samples" in aggregated
        assert len(aggregated["samples"]) > 0

        # Verify each sample has publication context
        for sample in aggregated["samples"]:
            assert "publication_entry_id" in sample
            assert "publication_title" in sample
            assert "publication_doi" in sample
            assert "publication_pmid" in sample
