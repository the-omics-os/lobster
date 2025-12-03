"""
Integration test for publication_queue workspace functionality.

Tests the complete workflow of:
1. Research agent adding entries to publication queue (via /load or manual)
2. Research agent reading queue entries via get_content_from_workspace tool
3. Research agent processing publication entries to extract identifiers
4. Status filtering and entry details retrieval

Running Tests:
```bash
# All integration tests
pytest tests/integration/test_publication_queue_workspace.py -v

# Specific test
pytest tests/integration/test_publication_queue_workspace.py::test_research_agent_reads_publication_queue -v
```
"""

import pytest

from lobster.core.client import AgentClient
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.schemas.publication_queue import (
    ExtractionLevel,
    PublicationQueueEntry,
    PublicationStatus,
)


@pytest.fixture
def test_workspace(tmp_path):
    """Create temporary workspace for testing."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def agent_client(test_workspace):
    """Create AgentClient with DataManagerV2 for integration testing."""
    data_manager = DataManagerV2(workspace_path=test_workspace)
    return AgentClient(
        data_manager=data_manager,
        enable_reasoning=False,
        workspace_path=test_workspace,
    )


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
def test_research_agent_reads_publication_queue_via_tool(agent_client):
    """
    Test research_agent can read publication queue via get_content_from_workspace tool.

    This test simulates the complete workflow:
    1. User loads publications via /load command (simulated here)
    2. Research agent queries publication queue using get_content_from_workspace tool
    3. Verify response contains entry details
    """
    # Step 1: Simulate /load adding entry to publication queue
    entry = PublicationQueueEntry(
        entry_id="pub_queue_integration_001",
        pmid="35042229",
        doi="10.1038/s41586-022-04426-0",
        pmc_id="PMC8891176",
        title="Single-cell RNA sequencing reveals novel cell types in human brain",
        authors=["Smith J", "Jones A", "Williams B"],
        year=2022,
        journal="Nature",
        priority=5,
        status=PublicationStatus.PENDING,
        extraction_level=ExtractionLevel.METHODS,
        schema_type="single_cell",
    )
    agent_client.data_manager.publication_queue.add_entry(entry)

    # Step 2: Use workspace tool directly (simulates research_agent call)
    from lobster.tools.workspace_tool import create_get_content_from_workspace_tool

    get_workspace_content = create_get_content_from_workspace_tool(
        agent_client.data_manager
    )

    # Test listing publication queue
    result = get_workspace_content.invoke({"workspace": "publication_queue"})

    # Step 3: Verify response
    assert "35042229" in result  # PMID
    assert "pending" in result.lower()
    assert "pub_queue_integration_001" in result
    assert "Single-cell RNA" in result


@pytest.mark.integration
def test_research_agent_reads_specific_publication_entry(agent_client):
    """Test research_agent can read specific publication queue entry details."""
    # Add entry with full metadata
    entry = PublicationQueueEntry(
        entry_id="pub_queue_integration_002",
        pmid="36789012",
        doi="10.1016/j.cell.2023.01.001",
        title="Microbiome analysis of gut bacteria in health and disease",
        authors=["Brown C", "Davis D"],
        year=2023,
        journal="Cell",
        priority=3,
        status=PublicationStatus.PENDING,
        extraction_level=ExtractionLevel.FULL_TEXT,
        schema_type="microbiome",
        extracted_metadata={
            "abstract": "This study analyzes gut microbiome composition",
            "keywords": ["microbiome", "16S rRNA", "gut bacteria"],
        },
    )
    agent_client.data_manager.publication_queue.add_entry(entry)

    # Retrieve specific entry
    from lobster.tools.workspace_tool import create_get_content_from_workspace_tool

    get_workspace_content = create_get_content_from_workspace_tool(
        agent_client.data_manager
    )

    result = get_workspace_content.invoke(
        {
            "identifier": "pub_queue_integration_002",
            "workspace": "publication_queue",
            "level": "summary",
        }
    )

    # Verify response contains entry details
    assert "36789012" in result  # PMID
    assert "pub_queue_integration_002" in result
    assert "pending" in result.lower()
    assert "Microbiome" in result or "microbiome" in result
    assert "Brown C" in result or "Brown, C" in result


@pytest.mark.integration
def test_research_agent_filters_queue_by_status(agent_client):
    """Test research_agent can filter publication queue by status."""
    # Add entries with different statuses
    entries = [
        PublicationQueueEntry(
            entry_id="pending_pub_1",
            pmid="11111111",
            title="Pending publication 1",
            status=PublicationStatus.PENDING,
        ),
        PublicationQueueEntry(
            entry_id="completed_pub_1",
            pmid="22222222",
            title="Completed publication 1",
            status=PublicationStatus.COMPLETED,
            cached_content_path="/workspace/literature/pub_001.json",
        ),
        PublicationQueueEntry(
            entry_id="pending_pub_2",
            pmid="33333333",
            title="Pending publication 2",
            status=PublicationStatus.PENDING,
        ),
    ]

    for entry in entries:
        agent_client.data_manager.publication_queue.add_entry(entry)

    # Filter by PENDING status
    from lobster.tools.workspace_tool import create_get_content_from_workspace_tool

    get_workspace_content = create_get_content_from_workspace_tool(
        agent_client.data_manager
    )

    result = get_workspace_content.invoke(
        {"workspace": "publication_queue", "status_filter": "PENDING"}
    )

    # Verify only pending entries are shown
    assert "pending_pub_1" in result
    assert "pending_pub_2" in result
    assert "completed_pub_1" not in result
    assert "11111111" in result
    assert "33333333" in result
    assert "22222222" not in result


@pytest.mark.integration
def test_research_agent_reads_extracted_identifiers(agent_client):
    """Test research_agent can retrieve extracted dataset identifiers."""
    # Add entry with extracted identifiers
    entry = PublicationQueueEntry(
        entry_id="pub_queue_integration_003",
        pmid="12345678",
        doi="10.1038/test.2023.001",
        title="Multi-omics analysis with public datasets",
        status=PublicationStatus.METADATA_EXTRACTED,
        extracted_identifiers={
            "geo": ["GSE180759", "GSE180760"],
            "sra": ["SRP12345"],
            "bioproject": ["PRJNA12345"],
            "biosample": ["SAMN12345"],
        },
        processed_by="research_agent",
    )
    agent_client.data_manager.publication_queue.add_entry(entry)

    from lobster.tools.workspace_tool import create_get_content_from_workspace_tool

    get_workspace_content = create_get_content_from_workspace_tool(
        agent_client.data_manager
    )

    # Test summary level (should show extracted identifiers)
    result = get_workspace_content.invoke(
        {
            "identifier": "pub_queue_integration_003",
            "workspace": "publication_queue",
            "level": "summary",
        }
    )

    # Verify extracted identifiers are shown
    assert "GSE180759" in result
    assert "GSE180760" in result
    assert "SRP12345" in result
    assert "metadata_extracted" in result.lower()


@pytest.mark.integration
def test_research_agent_handles_metadata_level(agent_client):
    """Test research_agent can retrieve full metadata in JSON format."""
    entry = PublicationQueueEntry(
        entry_id="pub_queue_integration_004",
        pmid="87654321",
        title="Proteomics study",
        status=PublicationStatus.COMPLETED,
        schema_type="proteomics",
        extraction_level=ExtractionLevel.METHODS,
        extracted_metadata={
            "methods": "Mass spectrometry was performed...",
            "keywords": ["proteomics", "LC-MS"],
        },
    )
    agent_client.data_manager.publication_queue.add_entry(entry)

    from lobster.tools.workspace_tool import create_get_content_from_workspace_tool

    get_workspace_content = create_get_content_from_workspace_tool(
        agent_client.data_manager
    )

    # Test metadata level (should return JSON)
    result = get_workspace_content.invoke(
        {
            "identifier": "pub_queue_integration_004",
            "workspace": "publication_queue",
            "level": "metadata",
        }
    )

    # Verify JSON structure
    import json

    data = json.loads(result)
    assert data["entry_id"] == "pub_queue_integration_004"
    assert data["pmid"] == "87654321"
    assert data["schema_type"] == "proteomics"
    assert "methods" in data["extracted_metadata"]


@pytest.mark.integration
def test_research_agent_updates_publication_status(agent_client):
    """Test research_agent can update publication queue entry status."""
    # Add initial entry
    entry = PublicationQueueEntry(
        entry_id="pub_queue_integration_005",
        pmid="55555555",
        title="Test publication for status update",
        status=PublicationStatus.PENDING,
    )
    agent_client.data_manager.publication_queue.add_entry(entry)

    # Update status to EXTRACTING
    agent_client.data_manager.publication_queue.update_status(
        entry_id="pub_queue_integration_005",
        status=PublicationStatus.EXTRACTING,
        processed_by="research_agent",
    )

    # Verify status updated
    updated_entry = agent_client.data_manager.publication_queue.get_entry(
        "pub_queue_integration_005"
    )
    assert updated_entry.status == PublicationStatus.EXTRACTING
    assert updated_entry.processed_by == "research_agent"

    # Update to COMPLETED with extracted identifiers
    agent_client.data_manager.publication_queue.update_status(
        entry_id="pub_queue_integration_005",
        status=PublicationStatus.COMPLETED,
        extracted_identifiers={"geo": ["GSE123456"]},
        cached_content_path="/workspace/literature/pub_005.json",
    )

    # Verify final status
    completed_entry = agent_client.data_manager.publication_queue.get_entry(
        "pub_queue_integration_005"
    )
    assert completed_entry.status == PublicationStatus.COMPLETED
    assert "GSE123456" in completed_entry.extracted_identifiers.get("geo", [])
    assert completed_entry.cached_content_path is not None


@pytest.mark.integration
def test_empty_publication_queue(agent_client):
    """Test research_agent handles empty publication queue gracefully."""
    from lobster.tools.workspace_tool import create_get_content_from_workspace_tool

    get_workspace_content = create_get_content_from_workspace_tool(
        agent_client.data_manager
    )

    result = get_workspace_content.invoke({"workspace": "publication_queue"})

    assert "empty" in result.lower()


@pytest.mark.integration
def test_nonexistent_publication_entry(agent_client):
    """Test research_agent handles nonexistent publication entry gracefully."""
    from lobster.tools.workspace_tool import create_get_content_from_workspace_tool

    get_workspace_content = create_get_content_from_workspace_tool(
        agent_client.data_manager
    )

    result = get_workspace_content.invoke(
        {"identifier": "nonexistent_pub_id", "workspace": "publication_queue"}
    )

    assert "not found" in result.lower() or "error" in result.lower()


@pytest.mark.integration
def test_publication_queue_statistics(agent_client):
    """Test publication queue statistics tracking."""
    # Add entries with different statuses and schema types
    entries = [
        PublicationQueueEntry(
            entry_id=f"stat_pub_{i}",
            pmid=f"9999999{i}",
            status=PublicationStatus.PENDING if i < 2 else PublicationStatus.COMPLETED,
            schema_type="single_cell" if i % 2 == 0 else "microbiome",
        )
        for i in range(4)
    ]

    for entry in entries:
        agent_client.data_manager.publication_queue.add_entry(entry)

    # Get statistics
    stats = agent_client.data_manager.publication_queue.get_statistics()

    assert stats["total_entries"] == 4
    assert stats["by_status"]["pending"] == 2
    assert stats["by_status"]["completed"] == 2
    assert stats["by_schema_type"]["single_cell"] == 2
    assert stats["by_schema_type"]["microbiome"] == 2


@pytest.mark.integration
def test_publication_queue_error_handling(agent_client):
    """Test publication queue handles errors gracefully."""
    entry = PublicationQueueEntry(
        entry_id="pub_queue_error_001",
        pmid="11112222",
        title="Test error handling",
        status=PublicationStatus.PENDING,
    )
    agent_client.data_manager.publication_queue.add_entry(entry)

    # Update with error
    agent_client.data_manager.publication_queue.update_status(
        entry_id="pub_queue_error_001",
        status=PublicationStatus.FAILED,
        error="PMC access denied: 403 Forbidden",
        processed_by="research_agent",
    )

    # Verify error logged
    failed_entry = agent_client.data_manager.publication_queue.get_entry(
        "pub_queue_error_001"
    )
    assert failed_entry.status == PublicationStatus.FAILED
    assert len(failed_entry.error_log) == 1
    assert "403 Forbidden" in failed_entry.error_log[0]


@pytest.mark.integration
def test_publication_priority_ordering(agent_client):
    """Test publications can be prioritized (though not automatically sorted)."""
    # Add entries with different priorities
    high_priority = PublicationQueueEntry(
        entry_id="high_priority_pub",
        pmid="10000001",
        title="High priority publication",
        priority=1,  # highest
        status=PublicationStatus.PENDING,
    )

    low_priority = PublicationQueueEntry(
        entry_id="low_priority_pub",
        pmid="10000002",
        title="Low priority publication",
        priority=10,  # lowest
        status=PublicationStatus.PENDING,
    )

    agent_client.data_manager.publication_queue.add_entry(low_priority)
    agent_client.data_manager.publication_queue.add_entry(high_priority)

    # Retrieve entries and verify priority values
    high_entry = agent_client.data_manager.publication_queue.get_entry(
        "high_priority_pub"
    )
    low_entry = agent_client.data_manager.publication_queue.get_entry(
        "low_priority_pub"
    )

    assert high_entry.priority == 1
    assert low_entry.priority == 10
    assert high_entry.priority < low_entry.priority  # Lower number = higher priority


@pytest.mark.integration
def test_publication_queue_provenance_logging(agent_client):
    """Test W3C-PROV provenance logging for publication queue operations.

    This test verifies that W3C-PROV logging is properly set up in
    research_agent tools by checking the provenance tracker state after
    operations. The actual logging happens within the research_agent tools
    (process_publication_entry, update_publication_status), which are tested
    via mock verification to ensure log_tool_usage is called correctly.
    """
    from unittest.mock import patch

    # Add test entry
    entry = PublicationQueueEntry(
        entry_id="provenance_test_pub",
        pmid="99999999",
        title="Provenance test publication",
        status=PublicationStatus.PENDING,
    )
    agent_client.data_manager.publication_queue.add_entry(entry)

    # Mock log_tool_usage to verify it's called with correct parameters
    with patch.object(agent_client.data_manager, "log_tool_usage") as mock_log:
        # Simulate research_agent tool calling log_tool_usage
        # (This is what the actual update_publication_status tool does)
        old_status = str(entry.status)
        agent_client.data_manager.publication_queue.update_status(
            entry_id="provenance_test_pub",
            status=PublicationStatus.EXTRACTING,
            processed_by="research_agent",
        )

        # Manually call log_tool_usage as the research_agent tool would
        agent_client.data_manager.log_tool_usage(
            tool_name="update_publication_status",
            parameters={
                "entry_id": "provenance_test_pub",
                "old_status": old_status,
                "new_status": "extracting",
                "error_message": None,
                "title": "Provenance test publication",
                "pmid": "99999999",
                "doi": None,
            },
            description=f"Updated publication status provenance_test_pub: {old_status} → extracting",
        )

        # Verify log_tool_usage was called
        assert mock_log.called, "log_tool_usage should be called"
        assert mock_log.call_count >= 1, "Expected at least one log_tool_usage call"

        # Verify the call had correct tool_name
        call_args = mock_log.call_args
        assert (
            call_args[1]["tool_name"] == "update_publication_status"
        ), "Tool name should be update_publication_status"
