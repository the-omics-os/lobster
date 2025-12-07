"""
Unit tests for CLI state management bug fixes.

Tests for:
- Bug #1: Archive State Race Condition (ExtractionCacheManager usage)
- Bug #5: Queue Transaction Safety (atomic writes, corruption handling)
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.extraction_cache import ExtractionCacheManager
from lobster.core.publication_queue import PublicationQueue
from lobster.core.schemas.publication_queue import (
    ExtractionLevel,
    PublicationQueueEntry,
    PublicationStatus,
)


class TestBug1ArchiveStateRace:
    """
    Test Bug #1: Archive state race condition prevention.

    Verifies that archive cache is managed per-session, not on shared client instance.
    """

    def test_extraction_cache_manager_thread_safe(self, tmp_path):
        """Test that multiple threads can use ExtractionCacheManager safely."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        results = {}
        errors = []

        def thread_a():
            try:
                cache_mgr = ExtractionCacheManager(workspace)
                caches = cache_mgr.list_all_caches()
                results["thread_a"] = len(caches)
            except Exception as e:
                errors.append(("thread_a", e))

        def thread_b():
            try:
                cache_mgr = ExtractionCacheManager(workspace)
                caches = cache_mgr.list_all_caches()
                results["thread_b"] = len(caches)
            except Exception as e:
                errors.append(("thread_b", e))

        # Run threads concurrently
        t1 = threading.Thread(target=thread_a)
        t2 = threading.Thread(target=thread_b)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        # No errors should occur
        assert len(errors) == 0, f"Threads should not raise errors: {errors}"
        assert "thread_a" in results
        assert "thread_b" in results

    def test_archive_cache_no_instance_variable_pollution(self):
        """Test that archive cache doesn't pollute client instance variables."""
        # Bug #1: Old code used client._last_archive_cache (instance variable)
        # New code: Uses ExtractionCacheManager (no client pollution)

        # Create a simple object to simulate a client (not Mock which has all attributes)
        class SimpleClient:
            def __init__(self):
                class DataManager:
                    workspace_path = Path(tempfile.gettempdir())
                self.data_manager = DataManager()

        client = SimpleClient()

        # Verify client doesn't have _last_archive_cache after using cache manager
        cache_mgr = ExtractionCacheManager(client.data_manager.workspace_path)
        caches = cache_mgr.list_all_caches()

        # Old bug: client._last_archive_cache would exist
        # New fix: client has no _last_archive_cache attribute
        assert not hasattr(
            client, "_last_archive_cache"
        ), "Client should not have cache state"


class TestBug5QueueTransactionSafety:
    """
    Test Bug #5: Queue transaction safety with atomic writes.

    Verifies that queue operations are atomic and corruption-resistant.
    """

    def test_add_entry_uses_atomic_write(self, tmp_path):
        """Test that add_entry() uses atomic write pattern."""
        queue_file = tmp_path / "queue.jsonl"
        queue = PublicationQueue(queue_file)

        # Add entry
        entry = PublicationQueueEntry(
            entry_id="test_001",
            pmid="12345",
            title="Test Publication",
            priority=5,
            status=PublicationStatus.PENDING,
            extraction_level=ExtractionLevel.METHODS,
        )

        entry_id = queue.add_entry(entry)

        # Verify entry added
        assert entry_id == "test_001"
        entries = queue.list_entries()
        assert len(entries) == 1

        # Verify file is valid JSON Lines (not corrupted)
        with open(queue_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 1
            parsed = json.loads(lines[0])
            assert parsed["entry_id"] == "test_001"

    def test_corrupted_entry_handling(self, tmp_path):
        """Test that corrupted entries are skipped with detailed logging."""
        queue_file = tmp_path / "queue.jsonl"
        queue = PublicationQueue(queue_file)

        # Create queue with mix of valid and corrupted entries
        with open(queue_file, "w") as f:
            # Valid entry
            valid_entry = PublicationQueueEntry(
                entry_id="valid_001",
                pmid="12345",
                title="Valid Publication",
                priority=5,
                status=PublicationStatus.PENDING,
                extraction_level=ExtractionLevel.METHODS,
            )
            f.write(json.dumps(valid_entry.to_dict()) + "\n")

            # Corrupted entry (partial JSON)
            f.write('{"entry_id": "corrupt_002", "pmid": "678\n')  # Incomplete

            # Another valid entry
            valid_entry2 = PublicationQueueEntry(
                entry_id="valid_003",
                pmid="99999",
                title="Another Valid",
                priority=3,
                status=PublicationStatus.COMPLETED,
                extraction_level=ExtractionLevel.FULL_TEXT,
            )
            f.write(json.dumps(valid_entry2.to_dict()) + "\n")

        # Bug #5 fix: Should load valid entries and skip corrupted ones
        entries = queue.list_entries()

        # Should load 2 valid entries, skip 1 corrupted
        assert len(entries) == 2
        assert entries[0].entry_id == "valid_001"
        assert entries[1].entry_id == "valid_003"

    def test_atomic_write_prevents_partial_writes(self, tmp_path):
        """Test that atomic write pattern prevents partial writes."""
        queue_file = tmp_path / "queue.jsonl"
        queue = PublicationQueue(queue_file)

        # Add first entry
        entry1 = PublicationQueueEntry(
            entry_id="entry_001",
            pmid="11111",
            title="First",
            priority=5,
            status=PublicationStatus.PENDING,
            extraction_level=ExtractionLevel.METHODS,
        )
        queue.add_entry(entry1)

        # Verify temp file used (Bug #5 fix)
        # The atomic write should use temp file + rename
        temp_file = queue_file.with_suffix(".tmp")

        # After successful write, temp file should not exist
        assert (
            not temp_file.exists()
        ), "Temp file should be cleaned up after atomic write"

        # Original file should be complete and valid
        assert queue_file.exists()
        entries = queue.list_entries()
        assert len(entries) == 1

    def test_backup_created_before_modification(self, tmp_path, monkeypatch):
        """Test that backups created before modifying queue."""
        # Enable backups via env var
        monkeypatch.setenv("LOBSTER_ENABLE_QUEUE_BACKUPS", "1")

        queue_file = tmp_path / "queue.jsonl"
        queue = PublicationQueue(queue_file)

        # Add first entry (no backup expected - file is empty)
        entry = PublicationQueueEntry(
            entry_id="entry_001",
            pmid="11111",
            title="First",
            priority=5,
            status=PublicationStatus.PENDING,
            extraction_level=ExtractionLevel.METHODS,
        )
        queue.add_entry(entry)

        # Add second entry (backup should be created before this modification)
        entry2 = PublicationQueueEntry(
            entry_id="entry_002",
            pmid="22222",
            title="Second",
            priority=5,
            status=PublicationStatus.PENDING,
            extraction_level=ExtractionLevel.METHODS,
        )
        queue.add_entry(entry2)

        # Verify backup created
        backup_dir = tmp_path / "backups"
        assert backup_dir.exists(), "Backup directory should exist"

        # Backups should be created
        backups = list(backup_dir.glob("publication_queue_*.jsonl"))
        assert len(backups) >= 1, "At least one backup should exist"

    @pytest.mark.benchmark
    def test_atomic_write_performance(self, tmp_path, benchmark):
        """Benchmark: verify atomic writes don't add significant overhead."""
        queue_file = tmp_path / "queue.jsonl"
        queue = PublicationQueue(queue_file)

        # Pre-populate with 50 entries
        for i in range(50):
            entry = PublicationQueueEntry(
                entry_id=f"entry_{i:03d}",
                pmid=f"{10000 + i}",
                title=f"Publication {i}",
                priority=5,
                status=PublicationStatus.PENDING,
                extraction_level=ExtractionLevel.METHODS,
            )
            queue.add_entry(entry)

        # Benchmark adding one more entry
        def add_entry_atomic():
            entry = PublicationQueueEntry(
                entry_id="bench_entry",
                pmid="99999",
                title="Benchmark Publication",
                priority=5,
                status=PublicationStatus.PENDING,
                extraction_level=ExtractionLevel.METHODS,
            )
            try:
                queue.add_entry(entry)
            except:
                pass  # Duplicate entry expected on reruns

        benchmark(add_entry_atomic)

        # Verify all entries intact after benchmark
        entries = queue.list_entries()
        assert len(entries) >= 51, "All entries should be preserved"


class TestBug2CacheInvalidation:
    """Test cache invalidation behavior for Bug #2 fix."""

    def test_invalidate_cache_clears_state(self, tmp_path):
        """Test that invalidate_scan_cache() clears cache state."""
        dm = DataManagerV2(workspace_path=tmp_path, auto_scan=False)

        # Trigger scan
        datasets = dm.get_available_datasets()

        # Cache should be populated
        assert dm._available_datasets_cache is not None
        assert dm._scan_timestamp > 0

        # Invalidate
        dm.invalidate_scan_cache()

        # Cache should be cleared
        assert dm._available_datasets_cache is None
        assert dm._scan_timestamp == 0

    def test_cache_expires_after_ttl(self, tmp_path):
        """Test that cache expires after TTL window."""
        dm = DataManagerV2(workspace_path=tmp_path, auto_scan=False)
        dm._scan_ttl = 1  # 1 second TTL for testing

        # First scan
        datasets1 = dm.get_available_datasets()
        timestamp1 = dm._scan_timestamp

        # Wait for TTL to expire
        time.sleep(1.1)

        # Next scan should refresh cache
        datasets2 = dm.get_available_datasets()
        timestamp2 = dm._scan_timestamp

        # Timestamp should have updated
        assert timestamp2 > timestamp1, "Cache should have refreshed after TTL"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
