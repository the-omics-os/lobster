"""
Integration test for download_queue workspace functionality.

Tests the complete workflow of:
1. Research agent adding entries to download queue
2. Supervisor reading queue entries via get_content_from_workspace tool
3. Status filtering and entry details retrieval

Running Tests:
```bash
# All integration tests
pytest tests/integration/test_download_queue_workspace.py -v

# Specific test
pytest tests/integration/test_download_queue_workspace.py::test_supervisor_reads_download_queue -v
```
"""

import pytest

from lobster.core.client import AgentClient
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.schemas.download_queue import (
    DownloadQueueEntry,
    DownloadStatus,
    StrategyConfig,
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
def test_supervisor_reads_download_queue_via_tool(agent_client):
    """
    Test supervisor can read download queue via get_content_from_workspace tool.

    This test simulates the complete workflow:
    1. Research agent adds entry to download queue (simulated here)
    2. Supervisor queries download queue using get_content_from_workspace tool
    3. Verify response contains entry details
    """
    # Step 1: Simulate research_agent adding entry to queue
    entry = DownloadQueueEntry(
        entry_id="integration_test_001",
        dataset_id="GSE180759",
        database="geo",
        priority=7,
        status=DownloadStatus.PENDING,
        metadata={"n_samples": 48, "platform": "GPL96"},
    )
    agent_client.data_manager.download_queue.add_entry(entry)

    # Step 2: Use workspace tool directly (simulates supervisor call)
    from lobster.tools.workspace_tool import create_get_content_from_workspace_tool

    get_workspace_content = create_get_content_from_workspace_tool(
        agent_client.data_manager
    )

    # Test listing download queue
    result = get_workspace_content.invoke({"workspace": "download_queue"})

    # Step 3: Verify response
    assert "GSE180759" in result
    assert "pending" in result.lower()
    assert "integration_test_001" in result


@pytest.mark.integration
def test_supervisor_reads_specific_queue_entry(agent_client):
    """Test supervisor can read specific download queue entry details."""
    # Add entry with full metadata
    strategy = StrategyConfig(
        strategy_name="MATRIX_FIRST",
        concatenation_strategy="auto",
        confidence=0.95,
        rationale="Processed matrix available",
    )

    entry = DownloadQueueEntry(
        entry_id="integration_test_002",
        dataset_id="GSE123456",
        database="geo",
        priority=5,
        status=DownloadStatus.PENDING,
        metadata={"n_samples": 100, "organism": "Homo sapiens"},
        validation_result={"is_valid": True, "warnings": []},
        recommended_strategy=strategy,
    )
    agent_client.data_manager.download_queue.add_entry(entry)

    # Retrieve specific entry
    from lobster.tools.workspace_tool import create_get_content_from_workspace_tool

    get_workspace_content = create_get_content_from_workspace_tool(
        agent_client.data_manager
    )

    result = get_workspace_content.invoke(
        {
            "identifier": "integration_test_002",
            "workspace": "download_queue",
            "level": "summary",
        }
    )

    # Verify response contains entry details
    assert "GSE123456" in result
    assert "integration_test_002" in result
    assert "pending" in result.lower()
    assert "MATRIX_FIRST" in result


@pytest.mark.integration
def test_supervisor_filters_queue_by_status(agent_client):
    """Test supervisor can filter download queue by status."""
    # Add entries with different statuses
    entries = [
        DownloadQueueEntry(
            entry_id="pending_1",
            dataset_id="GSE111",
            database="geo",
            status=DownloadStatus.PENDING,
        ),
        DownloadQueueEntry(
            entry_id="completed_1",
            dataset_id="GSE222",
            database="geo",
            status=DownloadStatus.COMPLETED,
            modality_name="geo_gse222",
        ),
        DownloadQueueEntry(
            entry_id="pending_2",
            dataset_id="GSE333",
            database="geo",
            status=DownloadStatus.PENDING,
        ),
    ]

    for entry in entries:
        agent_client.data_manager.download_queue.add_entry(entry)

    # Filter by PENDING status
    from lobster.tools.workspace_tool import create_get_content_from_workspace_tool

    get_workspace_content = create_get_content_from_workspace_tool(
        agent_client.data_manager
    )

    result = get_workspace_content.invoke(
        {"workspace": "download_queue", "status_filter": "PENDING"}
    )

    # Verify only pending entries are shown
    assert "pending_1" in result
    assert "pending_2" in result
    assert "completed_1" not in result
    assert "GSE111" in result
    assert "GSE333" in result
    assert "GSE222" not in result


@pytest.mark.integration
def test_supervisor_reads_entry_validation_and_strategy(agent_client):
    """Test supervisor can retrieve validation and strategy details."""
    # Add entry with validation and strategy
    strategy = StrategyConfig(
        strategy_name="RAW_FIRST",
        concatenation_strategy="union",
        confidence=0.87,
        rationale="Matrix quality issues detected",
    )

    entry = DownloadQueueEntry(
        entry_id="integration_test_003",
        dataset_id="GSE789",
        database="geo",
        validation_result={
            "is_valid": True,
            "warnings": ["Missing sample metadata for 2 samples"],
        },
        recommended_strategy=strategy,
    )
    agent_client.data_manager.download_queue.add_entry(entry)

    from lobster.tools.workspace_tool import create_get_content_from_workspace_tool

    get_workspace_content = create_get_content_from_workspace_tool(
        agent_client.data_manager
    )

    # Test validation level
    validation_result = get_workspace_content.invoke(
        {
            "identifier": "integration_test_003",
            "workspace": "download_queue",
            "level": "validation",
        }
    )
    assert "is_valid" in validation_result
    assert "warnings" in validation_result

    # Test strategy level
    strategy_result = get_workspace_content.invoke(
        {
            "identifier": "integration_test_003",
            "workspace": "download_queue",
            "level": "strategy",
        }
    )
    assert "RAW_FIRST" in strategy_result
    assert "0.87" in strategy_result


@pytest.mark.integration
def test_empty_download_queue(agent_client):
    """Test supervisor handles empty download queue gracefully."""
    from lobster.tools.workspace_tool import create_get_content_from_workspace_tool

    get_workspace_content = create_get_content_from_workspace_tool(
        agent_client.data_manager
    )

    result = get_workspace_content.invoke({"workspace": "download_queue"})

    assert "empty" in result.lower()


@pytest.mark.integration
def test_nonexistent_queue_entry(agent_client):
    """Test supervisor handles nonexistent queue entry gracefully."""
    from lobster.tools.workspace_tool import create_get_content_from_workspace_tool

    get_workspace_content = create_get_content_from_workspace_tool(
        agent_client.data_manager
    )

    result = get_workspace_content.invoke(
        {"identifier": "nonexistent_id", "workspace": "download_queue"}
    )

    assert "not found" in result.lower()
