"""
Unit tests for download queue infrastructure.

Tests cover:
- Pydantic schema validation (valid/invalid entries)
- CRUD operations (add, get, update, list, remove, clear)
- Persistence (save/load from JSON Lines)
- Status filtering (list by PENDING/COMPLETED/FAILED)
- Error handling (entry not found, invalid status, file corruption)
- Backup functionality (verify backup files created)
- Thread safety (concurrent operations)
- Statistics and reporting
"""

import json
import threading
import time
from datetime import datetime

import pytest

from lobster.core.download_queue import (
    DownloadQueue,
    DownloadQueueError,
    EntryNotFoundError,
)
from lobster.core.schemas.download_queue import (
    DownloadQueueEntry,
    DownloadStatus,
    StrategyConfig,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_queue_file(tmp_path):
    """Create temporary queue file path."""
    return tmp_path / "test_queue.jsonl"


@pytest.fixture
def download_queue(temp_queue_file):
    """Create DownloadQueue instance."""
    return DownloadQueue(temp_queue_file)


@pytest.fixture
def sample_strategy():
    """Create sample strategy configuration."""
    return StrategyConfig(
        strategy_name="MATRIX_FIRST",
        concatenation_strategy="auto",
        confidence=0.95,
        rationale="Processed matrix available with complete metadata",
    )


@pytest.fixture
def sample_entry(sample_strategy):
    """Create sample download queue entry."""
    return DownloadQueueEntry(
        entry_id="test_entry_001",
        dataset_id="GSE180759",
        database="geo",
        priority=5,
        status=DownloadStatus.PENDING,
        metadata={"title": "Test dataset", "organism": "Homo sapiens"},
        validation_result={"is_valid": True, "warnings": []},
        recommended_strategy=sample_strategy,
        matrix_url="https://example.com/matrix.h5",
        raw_urls=["https://example.com/raw1.fastq", "https://example.com/raw2.fastq"],
        supplementary_urls=["https://example.com/suppl.txt"],
        h5_url="https://example.com/data.h5",
    )


# =============================================================================
# Schema Validation Tests
# =============================================================================


class TestSchemaValidation:
    """Test Pydantic schema validation."""

    def test_strategy_config_valid(self):
        """Test valid StrategyConfig creation."""
        strategy = StrategyConfig(
            strategy_name="MATRIX_FIRST",
            concatenation_strategy="auto",
            confidence=0.95,
            rationale="Test rationale",
        )

        assert strategy.strategy_name == "MATRIX_FIRST"
        assert strategy.concatenation_strategy == "auto"
        assert strategy.confidence == 0.95
        assert strategy.rationale == "Test rationale"

    def test_strategy_config_invalid_concatenation(self):
        """Test StrategyConfig with invalid concatenation strategy."""
        with pytest.raises(ValueError, match="must be one of"):
            StrategyConfig(
                strategy_name="MATRIX_FIRST",
                concatenation_strategy="invalid",
                confidence=0.95,
            )

    def test_strategy_config_empty_name(self):
        """Test StrategyConfig with empty strategy name."""
        with pytest.raises(ValueError, match="cannot be empty"):
            StrategyConfig(
                strategy_name="",
                concatenation_strategy="auto",
                confidence=0.95,
            )

    def test_strategy_config_confidence_bounds(self):
        """Test StrategyConfig confidence validation."""
        # Valid confidence
        strategy = StrategyConfig(
            strategy_name="MATRIX_FIRST",
            concatenation_strategy="auto",
            confidence=0.5,
        )
        assert strategy.confidence == 0.5

        # Invalid confidence (too low)
        with pytest.raises(ValueError):
            StrategyConfig(
                strategy_name="MATRIX_FIRST",
                concatenation_strategy="auto",
                confidence=-0.1,
            )

        # Invalid confidence (too high)
        with pytest.raises(ValueError):
            StrategyConfig(
                strategy_name="MATRIX_FIRST",
                concatenation_strategy="auto",
                confidence=1.5,
            )

    def test_download_queue_entry_valid(self, sample_strategy):
        """Test valid DownloadQueueEntry creation."""
        entry = DownloadQueueEntry(
            entry_id="test_001",
            dataset_id="GSE12345",
            database="geo",
            priority=5,
            metadata={"key": "value"},
            recommended_strategy=sample_strategy,
        )

        assert entry.entry_id == "test_001"
        assert entry.dataset_id == "GSE12345"
        assert entry.database == "geo"
        assert entry.priority == 5
        assert entry.status == DownloadStatus.PENDING
        assert isinstance(entry.created_at, datetime)
        assert isinstance(entry.updated_at, datetime)

    def test_download_queue_entry_empty_fields(self):
        """Test DownloadQueueEntry with empty required fields."""
        with pytest.raises(ValueError, match="cannot be empty"):
            DownloadQueueEntry(
                entry_id="",
                dataset_id="GSE12345",
                database="geo",
            )

        with pytest.raises(ValueError, match="cannot be empty"):
            DownloadQueueEntry(
                entry_id="test_001",
                dataset_id="",
                database="geo",
            )

        with pytest.raises(ValueError, match="cannot be empty"):
            DownloadQueueEntry(
                entry_id="test_001",
                dataset_id="GSE12345",
                database="",
            )

    def test_download_queue_entry_invalid_database(self):
        """Test DownloadQueueEntry with invalid database."""
        with pytest.raises(ValueError, match="must be one of"):
            DownloadQueueEntry(
                entry_id="test_001",
                dataset_id="GSE12345",
                database="invalid_db",
            )

    def test_download_queue_entry_priority_bounds(self):
        """Test DownloadQueueEntry priority validation."""
        # Valid priority
        entry = DownloadQueueEntry(
            entry_id="test_001",
            dataset_id="GSE12345",
            database="geo",
            priority=1,
        )
        assert entry.priority == 1

        entry = DownloadQueueEntry(
            entry_id="test_002",
            dataset_id="GSE12345",
            database="geo",
            priority=10,
        )
        assert entry.priority == 10

        # Invalid priority (too low)
        with pytest.raises(ValueError):
            DownloadQueueEntry(
                entry_id="test_003",
                dataset_id="GSE12345",
                database="geo",
                priority=0,
            )

        # Invalid priority (too high)
        with pytest.raises(ValueError):
            DownloadQueueEntry(
                entry_id="test_004",
                dataset_id="GSE12345",
                database="geo",
                priority=11,
            )

    def test_download_queue_entry_update_status(self, sample_entry):
        """Test update_status method."""
        original_updated_at = sample_entry.updated_at

        # Wait a bit to ensure timestamp changes
        time.sleep(0.01)

        # Update status
        sample_entry.update_status(
            status=DownloadStatus.COMPLETED,
            modality_name="test_modality",
            downloaded_by="test_agent",
        )

        assert sample_entry.status == DownloadStatus.COMPLETED
        assert sample_entry.modality_name == "test_modality"
        assert sample_entry.downloaded_by == "test_agent"
        assert sample_entry.updated_at > original_updated_at

    def test_download_queue_entry_update_with_error(self, sample_entry):
        """Test update_status with error logging."""
        sample_entry.update_status(
            status=DownloadStatus.FAILED,
            error="Test error message",
        )

        assert sample_entry.status == DownloadStatus.FAILED
        assert len(sample_entry.error_log) == 1
        assert "Test error message" in sample_entry.error_log[0]

    def test_download_queue_entry_serialization(self, sample_entry):
        """Test to_dict and from_dict methods."""
        # Serialize
        data = sample_entry.to_dict()
        assert isinstance(data, dict)
        assert data["entry_id"] == sample_entry.entry_id
        assert data["dataset_id"] == sample_entry.dataset_id

        # Deserialize
        restored_entry = DownloadQueueEntry.from_dict(data)
        assert restored_entry.entry_id == sample_entry.entry_id
        assert restored_entry.dataset_id == sample_entry.dataset_id
        assert restored_entry.database == sample_entry.database
        assert restored_entry.priority == sample_entry.priority
        assert restored_entry.recommended_strategy.strategy_name == "MATRIX_FIRST"


# =============================================================================
# CRUD Operations Tests
# =============================================================================


class TestCRUDOperations:
    """Test create, read, update, delete operations."""

    def test_add_entry_success(self, download_queue, sample_entry):
        """Test successfully adding entry to queue."""
        entry_id = download_queue.add_entry(sample_entry)

        assert entry_id == sample_entry.entry_id
        assert download_queue.queue_file.exists()

        # Verify entry was written
        retrieved = download_queue.get_entry(entry_id)
        assert retrieved.entry_id == sample_entry.entry_id
        assert retrieved.dataset_id == sample_entry.dataset_id

    def test_add_entry_duplicate(self, download_queue, sample_entry):
        """Test adding duplicate entry raises error."""
        download_queue.add_entry(sample_entry)

        with pytest.raises(DownloadQueueError, match="already exists"):
            download_queue.add_entry(sample_entry)

    def test_get_entry_success(self, download_queue, sample_entry):
        """Test retrieving entry by ID."""
        download_queue.add_entry(sample_entry)

        retrieved = download_queue.get_entry(sample_entry.entry_id)
        assert retrieved.entry_id == sample_entry.entry_id
        assert retrieved.dataset_id == sample_entry.dataset_id
        assert retrieved.database == sample_entry.database

    def test_get_entry_not_found(self, download_queue):
        """Test retrieving non-existent entry raises error."""
        with pytest.raises(EntryNotFoundError, match="not found"):
            download_queue.get_entry("non_existent_id")

    def test_update_status_success(self, download_queue, sample_entry):
        """Test updating entry status."""
        download_queue.add_entry(sample_entry)

        updated = download_queue.update_status(
            entry_id=sample_entry.entry_id,
            status=DownloadStatus.IN_PROGRESS,
            downloaded_by="test_agent",
        )

        assert updated.status == DownloadStatus.IN_PROGRESS
        assert updated.downloaded_by == "test_agent"

        # Verify persistence
        retrieved = download_queue.get_entry(sample_entry.entry_id)
        assert retrieved.status == DownloadStatus.IN_PROGRESS

    def test_update_status_not_found(self, download_queue):
        """Test updating non-existent entry raises error."""
        with pytest.raises(EntryNotFoundError, match="not found"):
            download_queue.update_status(
                entry_id="non_existent_id",
                status=DownloadStatus.COMPLETED,
            )

    def test_update_status_with_error(self, download_queue, sample_entry):
        """Test updating status with error message."""
        download_queue.add_entry(sample_entry)

        updated = download_queue.update_status(
            entry_id=sample_entry.entry_id,
            status=DownloadStatus.FAILED,
            error="Download failed: connection timeout",
        )

        assert updated.status == DownloadStatus.FAILED
        assert len(updated.error_log) == 1
        assert "connection timeout" in updated.error_log[0]

    def test_list_entries_all(self, download_queue, sample_strategy):
        """Test listing all entries."""
        # Add multiple entries
        for i in range(3):
            entry = DownloadQueueEntry(
                entry_id=f"test_entry_{i}",
                dataset_id=f"GSE{i}",
                database="geo",
                priority=i + 1,
            )
            download_queue.add_entry(entry)

        entries = download_queue.list_entries()
        assert len(entries) == 3
        assert all(isinstance(e, DownloadQueueEntry) for e in entries)

    def test_list_entries_by_status(self, download_queue, sample_strategy):
        """Test listing entries filtered by status."""
        # Add entries with different statuses
        entry1 = DownloadQueueEntry(
            entry_id="entry_1",
            dataset_id="GSE001",
            database="geo",
            status=DownloadStatus.PENDING,
        )
        entry2 = DownloadQueueEntry(
            entry_id="entry_2",
            dataset_id="GSE002",
            database="geo",
            status=DownloadStatus.COMPLETED,
        )
        entry3 = DownloadQueueEntry(
            entry_id="entry_3",
            dataset_id="GSE003",
            database="geo",
            status=DownloadStatus.PENDING,
        )

        download_queue.add_entry(entry1)
        download_queue.add_entry(entry2)
        download_queue.add_entry(entry3)

        # List pending only
        pending = download_queue.list_entries(status=DownloadStatus.PENDING)
        assert len(pending) == 2
        assert all(e.status == DownloadStatus.PENDING for e in pending)

        # List completed only
        completed = download_queue.list_entries(status=DownloadStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].status == DownloadStatus.COMPLETED

    def test_remove_entry_success(self, download_queue, sample_entry):
        """Test removing entry from queue."""
        download_queue.add_entry(sample_entry)

        # Verify entry exists
        assert download_queue.get_entry(sample_entry.entry_id)

        # Remove entry
        download_queue.remove_entry(sample_entry.entry_id)

        # Verify entry is gone
        with pytest.raises(EntryNotFoundError):
            download_queue.get_entry(sample_entry.entry_id)

    def test_remove_entry_not_found(self, download_queue):
        """Test removing non-existent entry raises error."""
        with pytest.raises(EntryNotFoundError, match="not found"):
            download_queue.remove_entry("non_existent_id")

    def test_clear_queue_success(self, download_queue, sample_strategy):
        """Test clearing all entries from queue."""
        # Add multiple entries
        for i in range(5):
            entry = DownloadQueueEntry(
                entry_id=f"entry_{i}",
                dataset_id=f"GSE{i}",
                database="geo",
            )
            download_queue.add_entry(entry)

        # Verify entries exist
        assert len(download_queue.list_entries()) == 5

        # Clear queue
        cleared_count = download_queue.clear_queue()
        assert cleared_count == 5

        # Verify queue is empty
        assert len(download_queue.list_entries()) == 0

    def test_clear_queue_empty(self, download_queue):
        """Test clearing empty queue."""
        cleared_count = download_queue.clear_queue()
        assert cleared_count == 0


# =============================================================================
# Persistence Tests
# =============================================================================


class TestPersistence:
    """Test persistence and JSON Lines format."""

    def test_persistence_across_instances(self, temp_queue_file, sample_entry):
        """Test that entries persist across DownloadQueue instances."""
        # Create first instance and add entry
        queue1 = DownloadQueue(temp_queue_file)
        queue1.add_entry(sample_entry)

        # Create second instance and verify entry exists
        queue2 = DownloadQueue(temp_queue_file)
        retrieved = queue2.get_entry(sample_entry.entry_id)

        assert retrieved.entry_id == sample_entry.entry_id
        assert retrieved.dataset_id == sample_entry.dataset_id

    def test_json_lines_format(self, download_queue, sample_entry):
        """Test that queue file uses JSON Lines format."""
        download_queue.add_entry(sample_entry)

        # Read raw file content
        with open(download_queue.queue_file, "r") as f:
            lines = f.readlines()

        assert len(lines) == 1
        assert lines[0].strip()  # Not empty

        # Verify it's valid JSON
        data = json.loads(lines[0])
        assert data["entry_id"] == sample_entry.entry_id

    def test_multiple_entries_json_lines(self, download_queue, sample_strategy):
        """Test multiple entries in JSON Lines format."""
        # Add multiple entries
        for i in range(3):
            entry = DownloadQueueEntry(
                entry_id=f"entry_{i}",
                dataset_id=f"GSE{i}",
                database="geo",
            )
            download_queue.add_entry(entry)

        # Read raw file
        with open(download_queue.queue_file, "r") as f:
            lines = f.readlines()

        assert len(lines) == 3

        # Verify each line is valid JSON
        for line in lines:
            data = json.loads(line.strip())
            assert "entry_id" in data
            assert "dataset_id" in data

    def test_corrupted_line_handling(self, download_queue, temp_queue_file):
        """Test handling of corrupted JSON lines."""
        # Write valid entry
        entry = DownloadQueueEntry(
            entry_id="valid_entry",
            dataset_id="GSE001",
            database="geo",
        )
        download_queue.add_entry(entry)

        # Manually add corrupted line
        with open(temp_queue_file, "a") as f:
            f.write("this is not valid json\n")

        # Add another valid entry
        entry2 = DownloadQueueEntry(
            entry_id="valid_entry_2",
            dataset_id="GSE002",
            database="geo",
        )
        download_queue.add_entry(entry2)

        # Should load valid entries, skip corrupted
        entries = download_queue.list_entries()
        assert len(entries) == 2
        assert entries[0].entry_id == "valid_entry"
        assert entries[1].entry_id == "valid_entry_2"


# =============================================================================
# Backup Tests
# =============================================================================


class TestBackup:
    """Test backup functionality when explicitly enabled."""

    def _create_queue(self, temp_queue_file, monkeypatch):
        monkeypatch.setenv("LOBSTER_ENABLE_QUEUE_BACKUPS", "1")
        return DownloadQueue(temp_queue_file)

    def test_backup_created_on_modification(
        self, temp_queue_file, sample_entry, monkeypatch
    ):
        download_queue = self._create_queue(temp_queue_file, monkeypatch)
        download_queue.add_entry(sample_entry)

        download_queue.update_status(
            entry_id=sample_entry.entry_id,
            status=DownloadStatus.COMPLETED,
        )

        backup_files = list(download_queue.backup_dir.glob("queue_backup_*.jsonl"))
        assert len(backup_files) >= 1

    def test_backup_preserves_content(self, temp_queue_file, sample_entry, monkeypatch):
        download_queue = self._create_queue(temp_queue_file, monkeypatch)
        download_queue.add_entry(sample_entry)

        download_queue.update_status(
            entry_id=sample_entry.entry_id,
            status=DownloadStatus.IN_PROGRESS,
        )

        backup_files = sorted(download_queue.backup_dir.glob("queue_backup_*.jsonl"))
        assert backup_files

        latest_backup = backup_files[-1]
        with open(latest_backup, "r") as f:
            lines = f.readlines()

        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["entry_id"] == sample_entry.entry_id

    def test_multiple_backups(self, temp_queue_file, sample_entry, monkeypatch):
        download_queue = self._create_queue(temp_queue_file, monkeypatch)
        download_queue.add_entry(sample_entry)

        for status in [
            DownloadStatus.IN_PROGRESS,
            DownloadStatus.COMPLETED,
        ]:
            time.sleep(0.01)
            download_queue.update_status(
                entry_id=sample_entry.entry_id,
                status=status,
            )

        backup_files = list(download_queue.backup_dir.glob("queue_backup_*.jsonl"))
        assert len(backup_files) >= 2


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Test thread-safe operations."""

    def test_concurrent_add_operations(self, download_queue):
        """Test concurrent add operations are thread-safe."""
        num_threads = 10
        threads = []
        errors = []

        def add_entry(thread_id):
            try:
                entry = DownloadQueueEntry(
                    entry_id=f"thread_{thread_id}_entry",
                    dataset_id=f"GSE{thread_id}",
                    database="geo",
                )
                download_queue.add_entry(entry)
            except Exception as e:
                errors.append(e)

        # Create and start threads
        for i in range(num_threads):
            thread = threading.Thread(target=add_entry, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify no errors
        assert len(errors) == 0

        # Verify all entries added
        entries = download_queue.list_entries()
        assert len(entries) == num_threads

    def test_concurrent_read_operations(self, download_queue, sample_entry):
        """Test concurrent read operations are thread-safe."""
        download_queue.add_entry(sample_entry)

        num_threads = 20
        threads = []
        results = []
        errors = []

        def read_entry():
            try:
                entry = download_queue.get_entry(sample_entry.entry_id)
                results.append(entry)
            except Exception as e:
                errors.append(e)

        # Create and start threads
        for _ in range(num_threads):
            thread = threading.Thread(target=read_entry)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify no errors
        assert len(errors) == 0
        assert len(results) == num_threads

        # Verify all reads got correct data
        assert all(r.entry_id == sample_entry.entry_id for r in results)

    def test_concurrent_update_operations(self, download_queue, sample_entry):
        """Test concurrent update operations are thread-safe."""
        download_queue.add_entry(sample_entry)

        num_threads = 10
        threads = []
        errors = []

        def update_entry(thread_id):
            try:
                download_queue.update_status(
                    entry_id=sample_entry.entry_id,
                    status=DownloadStatus.IN_PROGRESS,
                    error=f"Update from thread {thread_id}",
                )
            except Exception as e:
                errors.append(e)

        # Create and start threads
        for i in range(num_threads):
            thread = threading.Thread(target=update_entry, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify no errors
        assert len(errors) == 0

        # Verify final state is consistent
        entry = download_queue.get_entry(sample_entry.entry_id)
        assert entry.status == DownloadStatus.IN_PROGRESS
        assert len(entry.error_log) == num_threads

    def test_cross_instance_concurrent_updates(self, temp_queue_file):
        """Ensure multiple queue instances coordinate safely."""
        queue_a = DownloadQueue(temp_queue_file)
        queue_b = DownloadQueue(temp_queue_file)

        total_entries = 30
        for i in range(total_entries):
            entry = DownloadQueueEntry(
                entry_id=f"cross_dl_{i}",
                dataset_id=f"GSE{i:05d}",
                database="geo",
            )
            queue_a.add_entry(entry)

        errors = []

        def worker(queue, ids, status):
            for idx in ids:
                try:
                    queue.update_status(
                        entry_id=f"cross_dl_{idx}",
                        status=status,
                    )
                except Exception as exc:  # pragma: no cover - diagnostic
                    errors.append(exc)

        even_ids = list(range(0, total_entries, 2))
        odd_ids = list(range(1, total_entries, 2))

        thread_even = threading.Thread(
            target=worker,
            args=(queue_a, even_ids, DownloadStatus.COMPLETED),
        )
        thread_odd = threading.Thread(
            target=worker,
            args=(queue_b, odd_ids, DownloadStatus.FAILED),
        )

        thread_even.start()
        thread_odd.start()
        thread_even.join()
        thread_odd.join()

        assert not errors

        entries = queue_a.list_entries()
        assert len(entries) == total_entries

        for entry in entries:
            idx = int(entry.entry_id.split("_")[-1])
            expected = (
                DownloadStatus.COMPLETED if idx % 2 == 0 else DownloadStatus.FAILED
            )
            assert entry.status == expected


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Test statistics and reporting functionality."""

    def test_get_statistics_empty_queue(self, download_queue):
        """Test statistics on empty queue."""
        stats = download_queue.get_statistics()

        assert stats["total_entries"] == 0
        assert stats["by_status"]["pending"] == 0
        assert stats["by_status"]["completed"] == 0

    def test_get_statistics_with_entries(self, download_queue):
        """Test statistics with multiple entries."""
        # Add entries with different statuses
        entries_data = [
            ("entry_1", "GSE001", "geo", DownloadStatus.PENDING, 1),
            ("entry_2", "GSE002", "geo", DownloadStatus.PENDING, 2),
            ("entry_3", "SRA001", "sra", DownloadStatus.COMPLETED, 3),
            ("entry_4", "PXD001", "pride", DownloadStatus.FAILED, 1),
        ]

        for entry_id, dataset_id, database, status, priority in entries_data:
            entry = DownloadQueueEntry(
                entry_id=entry_id,
                dataset_id=dataset_id,
                database=database,
                status=status,
                priority=priority,
            )
            download_queue.add_entry(entry)

        stats = download_queue.get_statistics()

        assert stats["total_entries"] == 4
        assert stats["by_status"]["pending"] == 2
        assert stats["by_status"]["completed"] == 1
        assert stats["by_status"]["failed"] == 1
        assert stats["by_status"]["in_progress"] == 0

        assert stats["by_database"]["geo"] == 2
        assert stats["by_database"]["sra"] == 1
        assert stats["by_database"]["pride"] == 1

        assert stats["by_priority"]["1"] == 2
        assert stats["by_priority"]["2"] == 1
        assert stats["by_priority"]["3"] == 1


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_queue_operations(self, download_queue):
        """Test operations on empty queue."""
        # List entries on empty queue
        entries = download_queue.list_entries()
        assert entries == []

        # Clear empty queue
        cleared = download_queue.clear_queue()
        assert cleared == 0

        # Get statistics on empty queue
        stats = download_queue.get_statistics()
        assert stats["total_entries"] == 0

    def test_special_characters_in_fields(self, download_queue):
        """Test handling of special characters."""
        entry = DownloadQueueEntry(
            entry_id="entry_with_special_chars",
            dataset_id="GSE!@#$%",
            database="geo",
            metadata={"title": "Dataset with 'quotes' and \"double quotes\""},
        )

        download_queue.add_entry(entry)
        retrieved = download_queue.get_entry(entry.entry_id)

        assert retrieved.dataset_id == "GSE!@#$%"
        assert "quotes" in retrieved.metadata["title"]

    def test_unicode_in_fields(self, download_queue):
        """Test handling of Unicode characters."""
        entry = DownloadQueueEntry(
            entry_id="unicode_entry",
            dataset_id="GSE12345",
            database="geo",
            metadata={
                "title": "研究データ (Japanese)",
                "organism": "Homo sapiens 🧬",
            },
        )

        download_queue.add_entry(entry)
        retrieved = download_queue.get_entry(entry.entry_id)

        assert "研究データ" in retrieved.metadata["title"]
        assert "🧬" in retrieved.metadata["organism"]

    def test_large_error_log(self, download_queue, sample_entry):
        """Test handling of large error logs."""
        download_queue.add_entry(sample_entry)

        # Add many errors
        for i in range(100):
            download_queue.update_status(
                entry_id=sample_entry.entry_id,
                status=DownloadStatus.FAILED,
                error=f"Error message {i}",
            )

        retrieved = download_queue.get_entry(sample_entry.entry_id)
        assert len(retrieved.error_log) == 100

    def test_datetime_serialization(self, download_queue, sample_entry):
        """Test proper datetime serialization."""
        download_queue.add_entry(sample_entry)

        # Read raw JSON
        with open(download_queue.queue_file, "r") as f:
            line = f.readline()

        data = json.loads(line)

        # Verify datetime fields are serialized as ISO format strings
        assert isinstance(data["created_at"], str)
        assert isinstance(data["updated_at"], str)

        # Verify they can be parsed back
        created_at = datetime.fromisoformat(data["created_at"])
        assert isinstance(created_at, datetime)
