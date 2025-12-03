"""
Unit tests for queue storage utilities.

Tests cover:
- atomic_write_json: Basic functionality, concurrent writes, crash recovery
- atomic_write_jsonl: Basic functionality (existing coverage)
- InterProcessFileLock: Lock acquisition and release
- queue_file_lock: Combined thread + process lock
"""

import json
import multiprocessing
import os
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pytest

from lobster.core.queue_storage import (
    InterProcessFileLock,
    atomic_write_json,
    atomic_write_jsonl,
    queue_file_lock,
)

# =============================================================================
# Helper functions for multiprocessing tests (must be at module level)
# =============================================================================


def _mp_increment_counter(args):
    """Worker function for multi-process counter increment test."""
    json_file, lock_file, num_increments = args
    json_file = Path(json_file)
    lock_file = Path(lock_file)

    # Each process needs its own thread lock (not shared across processes)
    thread_lock = threading.Lock()

    errors = []
    for i in range(num_increments):
        try:
            with queue_file_lock(thread_lock, lock_file):
                with open(json_file, "r") as f:
                    data = json.load(f)
                data["count"] += 1
                atomic_write_json(json_file, data)
        except Exception as e:
            errors.append(str(e))

    return errors


def _mp_write_session_data(args):
    """Worker function simulating DataManagerV2 session writes."""
    json_file, lock_file, process_id, num_writes = args
    json_file = Path(json_file)
    lock_file = Path(lock_file)

    thread_lock = threading.Lock()

    errors = []
    for i in range(num_writes):
        try:
            session_data = {
                "session_id": f"session_{process_id}",
                "last_modified": time.time(),
                "process_id": process_id,
                "iteration": i,
                "active_modalities": {f"mod_{j}": {"size": j * 100} for j in range(5)},
                "workspace_stats": {
                    "total_datasets": 10 + i,
                    "total_loaded": 3 + (i % 5),
                },
            }

            with queue_file_lock(thread_lock, lock_file):
                atomic_write_json(json_file, session_data)
        except Exception as e:
            errors.append(f"Process {process_id}, iter {i}: {e}")

    return errors


def _mp_cache_metadata_operations(args):
    """Worker function simulating ExtractionCacheManager operations."""
    json_file, lock_file, process_id, num_operations = args
    json_file = Path(json_file)
    lock_file = Path(lock_file)

    thread_lock = threading.Lock()

    errors = []
    for i in range(num_operations):
        try:
            cache_id = f"cache_{process_id}_{i}"

            with queue_file_lock(thread_lock, lock_file):
                # Read existing metadata
                all_metadata = {}
                if json_file.exists():
                    with open(json_file, "r") as f:
                        all_metadata = json.load(f)

                # Add new cache entry
                all_metadata[cache_id] = {
                    "cache_id": cache_id,
                    "process_id": process_id,
                    "iteration": i,
                    "extracted_at": time.time(),
                    "nested_info": {"archives": list(range(10))},
                }

                atomic_write_json(json_file, all_metadata)
        except Exception as e:
            errors.append(f"Process {process_id}, iter {i}: {e}")

    return errors


def _mp_verify_json_valid(json_file):
    """Verify JSON file is valid and not corrupted."""
    json_file = Path(json_file)
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
        return True, data
    except json.JSONDecodeError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def _mp_mutual_exclusion_worker(args):
    """Worker function for mutual exclusion test."""
    data_file, lock_file, iterations = args
    data_file = Path(data_file)
    lock_file = Path(lock_file)
    pid = os.getpid()
    thread_lock = threading.Lock()

    violations = []
    for _ in range(iterations):
        with queue_file_lock(thread_lock, lock_file):
            # Write our PID
            atomic_write_json(data_file, {"holder": pid})
            # Small delay to increase chance of race
            time.sleep(0.001)
            # Read back - should still be our PID
            with open(data_file, "r") as f:
                data = json.load(f)
            if data["holder"] != pid:
                violations.append(f"PID {pid} wrote but read back {data['holder']}")

    return violations


def _mp_stress_test_worker(args):
    """Worker function for stress test."""
    json_file, lock_file, process_id, num_ops = args
    json_file = Path(json_file)
    lock_file = Path(lock_file)
    thread_lock = threading.Lock()

    errors = []
    for i in range(num_ops):
        try:
            with queue_file_lock(thread_lock, lock_file):
                with open(json_file, "r") as f:
                    data = json.load(f)

                # Append our operation
                data["operations"].append(
                    {
                        "process": process_id,
                        "op": i,
                        "timestamp": time.time(),
                    }
                )

                atomic_write_json(json_file, data)
        except Exception as e:
            errors.append(str(e))

    return errors


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for tests."""
    return tmp_path


@pytest.fixture
def temp_json_file(temp_dir):
    """Create temporary JSON file path."""
    return temp_dir / "test_data.json"


@pytest.fixture
def temp_jsonl_file(temp_dir):
    """Create temporary JSONL file path."""
    return temp_dir / "test_data.jsonl"


@pytest.fixture
def temp_lock_file(temp_dir):
    """Create temporary lock file path."""
    return temp_dir / "test.lock"


# =============================================================================
# atomic_write_json Tests
# =============================================================================


class TestAtomicWriteJson:
    """Test atomic_write_json function."""

    def test_basic_write(self, temp_json_file):
        """Test basic JSON writing."""
        data = {"key": "value", "number": 42, "nested": {"a": 1}}

        atomic_write_json(temp_json_file, data)

        assert temp_json_file.exists()
        with open(temp_json_file, "r") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_overwrite_existing(self, temp_json_file):
        """Test overwriting existing file."""
        # Write initial data
        initial_data = {"version": 1}
        atomic_write_json(temp_json_file, initial_data)

        # Overwrite with new data
        new_data = {"version": 2, "new_field": "hello"}
        atomic_write_json(temp_json_file, new_data)

        with open(temp_json_file, "r") as f:
            loaded = json.load(f)
        assert loaded == new_data

    def test_creates_parent_directories(self, temp_dir):
        """Test that parent directories are created if they don't exist."""
        nested_path = temp_dir / "a" / "b" / "c" / "data.json"

        atomic_write_json(nested_path, {"test": True})

        assert nested_path.exists()
        with open(nested_path, "r") as f:
            loaded = json.load(f)
        assert loaded == {"test": True}

    def test_custom_indent(self, temp_json_file):
        """Test custom indent parameter."""
        data = {"key": "value"}

        atomic_write_json(temp_json_file, data, indent=4)

        with open(temp_json_file, "r") as f:
            content = f.read()
        # With indent=4, should have proper formatting
        assert "    " in content or "\n" in content

    def test_datetime_serialization(self, temp_json_file):
        """Test that datetime objects are serialized (via default=str)."""
        from datetime import datetime

        data = {"timestamp": datetime(2024, 1, 15, 10, 30, 0)}

        atomic_write_json(temp_json_file, data)

        with open(temp_json_file, "r") as f:
            loaded = json.load(f)
        assert "2024-01-15" in loaded["timestamp"]

    def test_temp_file_cleanup_on_success(self, temp_dir, temp_json_file):
        """Test that temp files are cleaned up after successful write."""
        atomic_write_json(temp_json_file, {"test": True})

        # Check no leftover temp files
        temp_files = list(temp_dir.glob("*.tmp"))
        assert len(temp_files) == 0

    def test_concurrent_writes_no_corruption(self, temp_json_file):
        """Test that concurrent writes don't corrupt the file."""
        num_threads = 10
        num_writes_per_thread = 20
        results = []
        errors = []

        def writer(thread_id):
            for i in range(num_writes_per_thread):
                try:
                    data = {
                        "thread": thread_id,
                        "iteration": i,
                        "data": list(range(100)),
                    }
                    atomic_write_json(temp_json_file, data)
                    results.append((thread_id, i))
                except Exception as e:
                    errors.append((thread_id, i, str(e)))

        threads = [
            threading.Thread(target=writer, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # File should be valid JSON
        with open(temp_json_file, "r") as f:
            final_data = json.load(f)
        assert "thread" in final_data
        assert "iteration" in final_data

    def test_empty_dict(self, temp_json_file):
        """Test writing empty dict."""
        atomic_write_json(temp_json_file, {})

        with open(temp_json_file, "r") as f:
            loaded = json.load(f)
        assert loaded == {}

    def test_large_data(self, temp_json_file):
        """Test writing large data structure."""
        large_data = {f"key_{i}": f"value_{i}" * 100 for i in range(1000)}

        atomic_write_json(temp_json_file, large_data)

        with open(temp_json_file, "r") as f:
            loaded = json.load(f)
        assert len(loaded) == 1000


# =============================================================================
# InterProcessFileLock Tests
# =============================================================================


class TestInterProcessFileLock:
    """Test InterProcessFileLock class."""

    def test_acquire_and_release(self, temp_lock_file):
        """Test basic lock acquire and release."""
        lock = InterProcessFileLock(temp_lock_file)

        lock.acquire()
        assert lock._handle is not None

        lock.release()
        assert lock._handle is None

    def test_context_manager(self, temp_lock_file):
        """Test lock as context manager."""
        with InterProcessFileLock(temp_lock_file) as lock:
            assert lock._handle is not None
        # After context, handle should be None (released)

    def test_creates_lock_file_directory(self, temp_dir):
        """Test that lock creates parent directory if needed."""
        lock_path = temp_dir / "subdir" / "test.lock"
        lock = InterProcessFileLock(lock_path)

        lock.acquire()
        assert lock_path.parent.exists()
        lock.release()


# =============================================================================
# queue_file_lock Tests
# =============================================================================


class TestQueueFileLock:
    """Test queue_file_lock context manager."""

    def test_basic_usage(self, temp_lock_file):
        """Test basic lock usage."""
        thread_lock = threading.Lock()

        with queue_file_lock(thread_lock, temp_lock_file):
            # Should be able to write safely here
            pass

    def test_mutual_exclusion(self, temp_lock_file):
        """Test that lock provides mutual exclusion."""
        thread_lock = threading.Lock()
        shared_counter = [0]
        num_threads = 5
        increments_per_thread = 100

        def increment():
            for _ in range(increments_per_thread):
                with queue_file_lock(thread_lock, temp_lock_file):
                    current = shared_counter[0]
                    time.sleep(0.0001)  # Small delay to increase contention
                    shared_counter[0] = current + 1

        threads = [threading.Thread(target=increment) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # With proper locking, counter should be exactly num_threads * increments_per_thread
        expected = num_threads * increments_per_thread
        assert shared_counter[0] == expected


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for atomic writes with locking."""

    def test_locked_atomic_write(self, temp_dir):
        """Test atomic write with file lock protection."""
        json_file = temp_dir / "protected.json"
        lock_file = temp_dir / "protected.lock"
        thread_lock = threading.Lock()

        def safe_write(data):
            with queue_file_lock(thread_lock, lock_file):
                atomic_write_json(json_file, data)

        # Concurrent safe writes
        num_threads = 5
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=safe_write, args=({"thread": i},))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # File should be valid JSON
        with open(json_file, "r") as f:
            data = json.load(f)
        assert "thread" in data

    def test_read_modify_write_pattern(self, temp_dir):
        """Test common read-modify-write pattern with locking."""
        json_file = temp_dir / "counter.json"
        lock_file = temp_dir / "counter.lock"
        thread_lock = threading.Lock()

        # Initialize file
        atomic_write_json(json_file, {"count": 0})

        def increment():
            for _ in range(10):
                with queue_file_lock(thread_lock, lock_file):
                    with open(json_file, "r") as f:
                        data = json.load(f)
                    data["count"] += 1
                    atomic_write_json(json_file, data)

        num_threads = 5
        threads = [threading.Thread(target=increment) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with open(json_file, "r") as f:
            final_data = json.load(f)

        assert final_data["count"] == num_threads * 10


# =============================================================================
# Multi-Process Tests (Simulating Multiple Lobster Sessions)
# =============================================================================


class TestMultiProcessConcurrency:
    """
    Test multi-process concurrent access to shared files.

    These tests simulate the real-world scenario of multiple Lobster CLI
    instances (separate processes) operating on the same workspace concurrently.
    """

    def test_multi_process_counter_increment(self, temp_dir):
        """
        Test that multiple processes can safely increment a shared counter.

        This is the classic read-modify-write pattern that would fail
        without proper inter-process locking.
        """
        json_file = temp_dir / "mp_counter.json"
        lock_file = temp_dir / "mp_counter.lock"

        # Initialize counter
        atomic_write_json(json_file, {"count": 0})

        num_processes = 4
        increments_per_process = 25
        expected_total = num_processes * increments_per_process

        # Spawn worker processes
        args_list = [
            (str(json_file), str(lock_file), increments_per_process)
            for _ in range(num_processes)
        ]

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(_mp_increment_counter, args_list))

        # Check for errors
        all_errors = [e for errors in results for e in errors]
        assert len(all_errors) == 0, f"Errors occurred: {all_errors}"

        # Verify final count
        with open(json_file, "r") as f:
            final_data = json.load(f)

        assert final_data["count"] == expected_total, (
            f"Expected {expected_total}, got {final_data['count']}. "
            "This indicates a race condition in inter-process locking."
        )

    def test_multi_process_session_file_writes(self, temp_dir):
        """
        Test concurrent session file writes from multiple processes.

        Simulates multiple DataManagerV2 instances updating .session.json
        simultaneously (the actual use case we're protecting against).
        """
        json_file = temp_dir / ".session.json"
        lock_file = temp_dir / ".session.lock"

        num_processes = 4
        writes_per_process = 20

        args_list = [
            (str(json_file), str(lock_file), pid, writes_per_process)
            for pid in range(num_processes)
        ]

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(_mp_write_session_data, args_list))

        # Check for errors
        all_errors = [e for errors in results for e in errors]
        assert len(all_errors) == 0, f"Errors occurred: {all_errors}"

        # Verify file is valid JSON (not corrupted)
        is_valid, result = _mp_verify_json_valid(json_file)
        assert is_valid, f"Session file corrupted: {result}"

        # Verify structure is intact
        assert "session_id" in result
        assert "active_modalities" in result
        assert "workspace_stats" in result

    def test_multi_process_cache_metadata_operations(self, temp_dir):
        """
        Test concurrent cache metadata read-modify-write operations.

        Simulates multiple ExtractionCacheManager instances adding cache
        entries simultaneously (the actual use case we're protecting against).
        """
        json_file = temp_dir / "cache_metadata.json"
        lock_file = temp_dir / "cache_metadata.lock"

        num_processes = 4
        operations_per_process = 15

        args_list = [
            (str(json_file), str(lock_file), pid, operations_per_process)
            for pid in range(num_processes)
        ]

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(_mp_cache_metadata_operations, args_list))

        # Check for errors
        all_errors = [e for errors in results for e in errors]
        assert len(all_errors) == 0, f"Errors occurred: {all_errors}"

        # Verify file is valid JSON
        is_valid, result = _mp_verify_json_valid(json_file)
        assert is_valid, f"Cache metadata corrupted: {result}"

        # Verify all cache entries are present (no lost writes)
        expected_entries = num_processes * operations_per_process
        assert len(result) == expected_entries, (
            f"Expected {expected_entries} cache entries, got {len(result)}. "
            "This indicates lost writes due to race conditions."
        )

        # Verify each entry has correct structure
        for cache_id, metadata in result.items():
            assert "cache_id" in metadata
            assert "process_id" in metadata
            assert "extracted_at" in metadata
            assert "nested_info" in metadata

    def test_multi_process_file_lock_mutual_exclusion(self, temp_dir):
        """
        Test that InterProcessFileLock provides true mutual exclusion.

        Each process writes its PID to a file while holding the lock.
        After writing, it reads back immediately. If we ever read a
        different PID, mutual exclusion is broken.
        """
        data_file = temp_dir / "exclusion_test.json"
        lock_file = temp_dir / "exclusion_test.lock"

        num_processes = 4
        iterations_per_process = 50

        args_list = [
            (str(data_file), str(lock_file), iterations_per_process)
            for _ in range(num_processes)
        ]

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(_mp_mutual_exclusion_worker, args_list))

        all_violations = [v for violations in results for v in violations]
        assert len(all_violations) == 0, (
            f"Mutual exclusion violated! {len(all_violations)} violations: "
            f"{all_violations[:5]}..."  # Show first 5
        )

    def test_multi_process_stress_test(self, temp_dir):
        """
        Stress test with high contention: many processes, many operations.

        This test creates maximum contention to expose any race conditions.
        """
        json_file = temp_dir / "stress_test.json"
        lock_file = temp_dir / "stress_test.lock"

        # Initialize with a list to track all operations
        atomic_write_json(json_file, {"operations": []})

        num_processes = 6
        ops_per_process = 30
        expected_total_ops = num_processes * ops_per_process

        args_list = [
            (str(json_file), str(lock_file), pid, ops_per_process)
            for pid in range(num_processes)
        ]

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(_mp_stress_test_worker, args_list))

        all_errors = [e for errors in results for e in errors]
        assert len(all_errors) == 0, f"Errors: {all_errors}"

        # Verify all operations recorded (no lost writes)
        with open(json_file, "r") as f:
            final_data = json.load(f)

        actual_ops = len(final_data["operations"])
        assert actual_ops == expected_total_ops, (
            f"Lost writes! Expected {expected_total_ops} operations, "
            f"got {actual_ops}. Missing {expected_total_ops - actual_ops} operations."
        )

        # Verify operation distribution across processes
        ops_per_pid = {}
        for op in final_data["operations"]:
            pid = op["process"]
            ops_per_pid[pid] = ops_per_pid.get(pid, 0) + 1

        for pid in range(num_processes):
            assert ops_per_pid.get(pid, 0) == ops_per_process, (
                f"Process {pid}: expected {ops_per_process} ops, "
                f"got {ops_per_pid.get(pid, 0)}"
            )
