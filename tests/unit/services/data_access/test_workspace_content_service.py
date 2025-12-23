"""
Unit tests for WorkspaceContentService with download_queue support.

Tests the workspace content caching system including the new download_queue
category introduced in Phase 0 of the download queue implementation.

Coverage:
- ContentType enum including DOWNLOAD_QUEUE
- WorkspaceContentService initialization
- Download queue entry reading and listing
- Status filtering for download queue entries
- Error handling for missing entries

Running Tests:
```bash
# All tests
pytest tests/unit/tools/test_workspace_content_service.py -v

# Specific test class
pytest tests/unit/tools/test_workspace_content_service.py::TestDownloadQueueWorkspace -v
```
"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.download_queue import DownloadQueue
from lobster.core.schemas.download_queue import DownloadQueueEntry, DownloadStatus
from lobster.services.data_access.workspace_content_service import (
    ContentType,
    WorkspaceContentService,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def download_queue(tmp_path):
    """Create a DownloadQueue instance for testing."""
    queue_file = tmp_path / "download_queue.jsonl"
    return DownloadQueue(queue_file)


@pytest.fixture
def data_manager(temp_workspace, download_queue):
    """Create a mock DataManagerV2 with download_queue."""
    dm = Mock(spec=DataManagerV2)
    dm.workspace_path = temp_workspace
    dm.download_queue = download_queue
    return dm


@pytest.fixture
def workspace_service(data_manager):
    """Create WorkspaceContentService instance."""
    return WorkspaceContentService(data_manager=data_manager)


# ============================================================================
# Test ContentType Enum
# ============================================================================


class TestContentTypeEnum:
    """Test ContentType enum includes DOWNLOAD_QUEUE."""

    def test_content_type_includes_download_queue(self):
        """Test DOWNLOAD_QUEUE is in ContentType enum."""
        assert hasattr(ContentType, "DOWNLOAD_QUEUE")
        assert ContentType.DOWNLOAD_QUEUE == "download_queue"

    def test_content_type_has_all_expected_values(self):
        """Test ContentType enum has all expected values."""
        expected_values = {
            "publication",
            "dataset",
            "metadata",
            "download_queue",
            "publication_queue",
        }
        actual_values = {ct.value for ct in ContentType}
        assert actual_values == expected_values


# ============================================================================
# Test WorkspaceContentService Initialization
# ============================================================================


class TestWorkspaceServiceInitialization:
    """Test WorkspaceContentService initialization."""

    def test_initialization_creates_all_directories(
        self, workspace_service, temp_workspace
    ):
        """Test all workspace directories are created.

        Note: Queue directories (download_queue, publication_queue) are managed
        by DataManagerV2 at .lobster/queues/ as JSONL files, not by this service.
        """
        expected_dirs = ["literature", "data", "metadata"]
        for dir_name in expected_dirs:
            dir_path = temp_workspace / dir_name
            assert dir_path.exists(), f"{dir_name} directory not created"
            assert dir_path.is_dir(), f"{dir_name} is not a directory"


# ============================================================================
# Test Download Queue Entry Reading
# ============================================================================


class TestDownloadQueueWorkspace:
    """Test download_queue workspace functionality."""

    def test_list_download_queue_entries_empty(self, workspace_service):
        """Test listing empty download queue."""
        entries = workspace_service.list_download_queue_entries()
        assert entries == []

    def test_list_download_queue_entries_with_data(
        self, workspace_service, data_manager
    ):
        """Test listing download queue with entries."""
        # Add test entry to queue
        entry = DownloadQueueEntry(
            entry_id="test_001",
            dataset_id="GSE12345",
            database="geo",
            priority=5,
            metadata={"test": "data"},
        )
        data_manager.download_queue.add_entry(entry)

        # List entries
        entries = workspace_service.list_download_queue_entries()
        assert len(entries) == 1
        assert entries[0]["entry_id"] == "test_001"
        assert entries[0]["dataset_id"] == "GSE12345"

    def test_list_download_queue_entries_with_status_filter_pending(
        self, workspace_service, data_manager
    ):
        """Test listing download queue with PENDING status filter."""
        # Add entries with different statuses
        entry1 = DownloadQueueEntry(
            entry_id="e1",
            dataset_id="GSE1",
            database="geo",
            status=DownloadStatus.PENDING,
        )
        entry2 = DownloadQueueEntry(
            entry_id="e2",
            dataset_id="GSE2",
            database="geo",
            status=DownloadStatus.COMPLETED,
        )

        data_manager.download_queue.add_entry(entry1)
        data_manager.download_queue.add_entry(entry2)

        # Filter by PENDING
        pending = workspace_service.list_download_queue_entries(status_filter="PENDING")
        assert len(pending) == 1
        assert pending[0]["entry_id"] == "e1"

    def test_list_download_queue_entries_with_status_filter_completed(
        self, workspace_service, data_manager
    ):
        """Test listing download queue with COMPLETED status filter."""
        # Add entries with different statuses
        entry1 = DownloadQueueEntry(
            entry_id="e1",
            dataset_id="GSE1",
            database="geo",
            status=DownloadStatus.PENDING,
        )
        entry2 = DownloadQueueEntry(
            entry_id="e2",
            dataset_id="GSE2",
            database="geo",
            status=DownloadStatus.COMPLETED,
        )

        data_manager.download_queue.add_entry(entry1)
        data_manager.download_queue.add_entry(entry2)

        # Filter by COMPLETED
        completed = workspace_service.list_download_queue_entries(
            status_filter="COMPLETED"
        )
        assert len(completed) == 1
        assert completed[0]["entry_id"] == "e2"

    def test_list_download_queue_entries_with_invalid_status_filter(
        self, workspace_service
    ):
        """Test invalid status filter returns empty list."""
        result = workspace_service.list_download_queue_entries(
            status_filter="INVALID_STATUS"
        )
        assert result == []

    def test_read_download_queue_entry_success(self, workspace_service, data_manager):
        """Test reading specific download queue entry."""
        entry = DownloadQueueEntry(
            entry_id="test_read",
            dataset_id="GSE99999",
            database="geo",
            priority=8,
            metadata={"key": "value"},
        )
        data_manager.download_queue.add_entry(entry)

        # Read entry
        result = workspace_service.read_download_queue_entry("test_read")
        assert result["entry_id"] == "test_read"
        assert result["dataset_id"] == "GSE99999"
        assert result["priority"] == 8

    def test_read_download_queue_entry_not_found(self, workspace_service):
        """Test reading non-existent entry raises error."""
        with pytest.raises(
            FileNotFoundError, match="Download queue entry 'nonexistent' not found"
        ):
            workspace_service.read_download_queue_entry("nonexistent")

    def test_read_download_queue_entry_without_download_queue(self, temp_workspace):
        """Test reading entry when download_queue not available."""
        # Create data manager without download_queue
        dm = Mock(spec=DataManagerV2)
        dm.workspace_path = temp_workspace
        # Explicitly set download_queue to None
        dm.download_queue = None

        service = WorkspaceContentService(data_manager=dm)

        with pytest.raises(AttributeError, match="DataManager download_queue"):
            service.read_download_queue_entry("any_id")

    def test_list_download_queue_entries_without_download_queue(self, temp_workspace):
        """Test listing entries when download_queue not available."""
        # Create data manager without download_queue
        dm = Mock(spec=DataManagerV2)
        dm.workspace_path = temp_workspace
        dm.download_queue = None

        service = WorkspaceContentService(data_manager=dm)

        result = service.list_download_queue_entries()
        assert result == []


# ============================================================================
# Test Download Queue Entry Details
# ============================================================================


class TestDownloadQueueEntryDetails:
    """Test download queue entry with full metadata."""

    def test_entry_with_validation_result(self, workspace_service, data_manager):
        """Test entry with validation result."""
        entry = DownloadQueueEntry(
            entry_id="test_validation",
            dataset_id="GSE123",
            database="geo",
            validation_result={"is_valid": True, "warnings": []},
        )
        data_manager.download_queue.add_entry(entry)

        result = workspace_service.read_download_queue_entry("test_validation")
        assert result["validation_result"]["is_valid"] is True

    def test_entry_with_recommended_strategy(self, workspace_service, data_manager):
        """Test entry with recommended strategy."""
        from lobster.core.schemas.download_queue import StrategyConfig

        strategy = StrategyConfig(
            strategy_name="MATRIX_FIRST",
            concatenation_strategy="auto",
            confidence=0.95,
            rationale="Matrix available",
        )

        entry = DownloadQueueEntry(
            entry_id="test_strategy",
            dataset_id="GSE456",
            database="geo",
            recommended_strategy=strategy,
        )
        data_manager.download_queue.add_entry(entry)

        result = workspace_service.read_download_queue_entry("test_strategy")
        assert result["recommended_strategy"]["strategy_name"] == "MATRIX_FIRST"
        assert result["recommended_strategy"]["confidence"] == 0.95

    def test_entry_with_urls(self, workspace_service, data_manager):
        """Test entry with download URLs."""
        entry = DownloadQueueEntry(
            entry_id="test_urls",
            dataset_id="GSE789",
            database="geo",
            matrix_url="https://example.com/matrix.h5",
            h5_url="https://example.com/data.h5",
            raw_urls=[
                "https://example.com/raw1.fastq",
                "https://example.com/raw2.fastq",
            ],
        )
        data_manager.download_queue.add_entry(entry)

        result = workspace_service.read_download_queue_entry("test_urls")
        assert result["matrix_url"] == "https://example.com/matrix.h5"
        assert len(result["raw_urls"]) == 2

    def test_entry_with_modality_name(self, workspace_service, data_manager):
        """Test entry with modality_name after completion."""
        entry = DownloadQueueEntry(
            entry_id="test_modality",
            dataset_id="GSE999",
            database="geo",
            status=DownloadStatus.COMPLETED,
            modality_name="geo_gse999",
        )
        data_manager.download_queue.add_entry(entry)

        result = workspace_service.read_download_queue_entry("test_modality")
        assert result["modality_name"] == "geo_gse999"
        assert result["status"] == "completed"


# ============================================================================
# Test Multiple Entries and Filtering
# ============================================================================


class TestMultipleEntriesAndFiltering:
    """Test operations with multiple queue entries."""

    def test_list_multiple_entries_all_statuses(self, workspace_service, data_manager):
        """Test listing multiple entries with various statuses."""
        entries_to_add = [
            DownloadQueueEntry(
                entry_id="e1",
                dataset_id="GSE1",
                database="geo",
                status=DownloadStatus.PENDING,
            ),
            DownloadQueueEntry(
                entry_id="e2",
                dataset_id="GSE2",
                database="geo",
                status=DownloadStatus.IN_PROGRESS,
            ),
            DownloadQueueEntry(
                entry_id="e3",
                dataset_id="GSE3",
                database="geo",
                status=DownloadStatus.COMPLETED,
            ),
            DownloadQueueEntry(
                entry_id="e4",
                dataset_id="GSE4",
                database="geo",
                status=DownloadStatus.FAILED,
            ),
        ]

        for entry in entries_to_add:
            data_manager.download_queue.add_entry(entry)

        # List all
        all_entries = workspace_service.list_download_queue_entries()
        assert len(all_entries) == 4

        # Filter by each status
        pending = workspace_service.list_download_queue_entries(status_filter="PENDING")
        assert len(pending) == 1

        in_progress = workspace_service.list_download_queue_entries(
            status_filter="IN_PROGRESS"
        )
        assert len(in_progress) == 1

        completed = workspace_service.list_download_queue_entries(
            status_filter="COMPLETED"
        )
        assert len(completed) == 1

        failed = workspace_service.list_download_queue_entries(status_filter="FAILED")
        assert len(failed) == 1

    def test_entry_priority_ordering(self, workspace_service, data_manager):
        """Test entries with different priorities."""
        entries_to_add = [
            DownloadQueueEntry(
                entry_id="low", dataset_id="GSE1", database="geo", priority=8
            ),
            DownloadQueueEntry(
                entry_id="high", dataset_id="GSE2", database="geo", priority=1
            ),
            DownloadQueueEntry(
                entry_id="medium", dataset_id="GSE3", database="geo", priority=5
            ),
        ]

        for entry in entries_to_add:
            data_manager.download_queue.add_entry(entry)

        all_entries = workspace_service.list_download_queue_entries()
        assert len(all_entries) == 3

        # Check priorities are preserved
        priorities = {e["entry_id"]: e["priority"] for e in all_entries}
        assert priorities["low"] == 8
        assert priorities["high"] == 1
        assert priorities["medium"] == 5


# ============================================================================
# NoneType Handling Tests (Bug Fix)
# ============================================================================


class TestNoneTypeHandling:
    """
    Test suite for None handling in queue listing methods.

    These tests verify the bug fix for NoneType errors when queue.list_entries()
    returns None instead of an empty list.

    Bug Report: workspace_tool.py:969 - object of type 'NoneType' has no len()
    """

    def test_list_download_queue_entries_handles_none_from_queue(self, temp_workspace):
        """Test that list_download_queue_entries returns [] if queue.list_entries() returns None."""
        # Create mock data manager with download_queue that returns None
        dm = Mock(spec=DataManagerV2)
        dm.workspace_path = temp_workspace
        dm.download_queue = Mock()
        dm.download_queue.list_entries.return_value = None

        service = WorkspaceContentService(dm)
        result = service.list_download_queue_entries()

        # Should not crash, should return empty list
        assert result == []
        assert isinstance(result, list)

    def test_list_publication_queue_entries_handles_none_from_queue(self, temp_workspace):
        """Test that list_publication_queue_entries returns [] if queue.list_entries() returns None."""
        # Create mock data manager with publication_queue that returns None
        dm = Mock(spec=DataManagerV2)
        dm.workspace_path = temp_workspace
        dm.publication_queue = Mock()
        dm.publication_queue.list_entries.return_value = None

        service = WorkspaceContentService(dm)
        result = service.list_publication_queue_entries()

        # Should not crash, should return empty list
        assert result == []
        assert isinstance(result, list)

    def test_list_download_queue_entries_handles_exception(self, temp_workspace):
        """Test graceful degradation when queue.list_entries() raises exception."""
        # Create mock data manager with download_queue that raises exception
        dm = Mock(spec=DataManagerV2)
        dm.workspace_path = temp_workspace
        dm.download_queue = Mock()
        dm.download_queue.list_entries.side_effect = RuntimeError("Queue corrupted")

        service = WorkspaceContentService(dm)
        result = service.list_download_queue_entries()

        # Should not crash, should return empty list
        assert result == []
        assert isinstance(result, list)

    def test_list_publication_queue_entries_handles_exception(self, temp_workspace):
        """Test graceful degradation when queue.list_entries() raises exception."""
        # Create mock data manager with publication_queue that raises exception
        dm = Mock(spec=DataManagerV2)
        dm.workspace_path = temp_workspace
        dm.publication_queue = Mock()
        dm.publication_queue.list_entries.side_effect = RuntimeError("Queue corrupted")

        service = WorkspaceContentService(dm)
        result = service.list_publication_queue_entries()

        # Should not crash, should return empty list
        assert result == []
        assert isinstance(result, list)

    def test_list_download_queue_entries_with_none_and_status_filter(
        self, temp_workspace
    ):
        """Test None handling with status filter applied."""
        dm = Mock(spec=DataManagerV2)
        dm.workspace_path = temp_workspace
        dm.download_queue = Mock()
        dm.download_queue.list_entries.return_value = None

        service = WorkspaceContentService(dm)
        result = service.list_download_queue_entries(status_filter="PENDING")

        assert result == []
        assert isinstance(result, list)

    def test_list_publication_queue_entries_with_none_and_status_filter(
        self, temp_workspace
    ):
        """Test None handling with status filter applied."""
        dm = Mock(spec=DataManagerV2)
        dm.workspace_path = temp_workspace
        dm.publication_queue = Mock()
        dm.publication_queue.list_entries.return_value = None

        service = WorkspaceContentService(dm)
        result = service.list_publication_queue_entries(status_filter="pending")

        assert result == []
        assert isinstance(result, list)
