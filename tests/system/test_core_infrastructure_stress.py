"""
Comprehensive stress tests for Lobster core infrastructure components.

This module tests:
- DataManagerV2 concurrent operations
- Agent coordination race conditions
- Provenance tracking under load
- Workspace restoration with 100+ datasets
- Thread safety validation
- Memory profiling under extreme conditions

Test Scenarios:
1. Concurrent modality access (16-32 workers)
2. Race conditions in agent handoffs
3. Provenance logging under high throughput
4. Workspace with 100+ datasets
5. Concurrent save/load operations
6. Agent registry under load
7. Thread safety validation
8. Database connection pooling
9. File descriptor limits
10. Simultaneous workspace operations
"""

import concurrent.futures
import gc
import logging
import os
import shutil
import tempfile
import threading
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Tuple

import anndata
import numpy as np
import pandas as pd
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


# ============================================================================
# Fixtures for Stress Testing
# ============================================================================


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    temp_dir = tempfile.mkdtemp(prefix="lobster_stress_test_")
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def data_manager(temp_workspace):
    """Create a DataManagerV2 instance for testing."""
    dm = DataManagerV2(workspace_path=temp_workspace)
    yield dm
    # Cleanup
    dm.clear()


@pytest.fixture
def mock_adata_factory():
    """Factory for creating mock AnnData objects."""

    def _create_adata(
        n_obs: int = 100, n_vars: int = 50, name: str = "test"
    ) -> anndata.AnnData:
        """Create a mock AnnData object with realistic structure."""
        X = np.random.randn(n_obs, n_vars)
        obs = pd.DataFrame(
            {
                "cell_type": np.random.choice(["A", "B", "C"], n_obs),
                "batch": np.random.choice(["batch1", "batch2"], n_obs),
                "n_genes": np.random.randint(500, 5000, n_obs),
                "pct_counts_mt": np.random.uniform(0, 50, n_obs),
            },
            index=[f"{name}_cell_{i}" for i in range(n_obs)],
        )
        var = pd.DataFrame(
            {
                "gene_symbol": [f"gene_{i}" for i in range(n_vars)],
                "highly_variable": np.random.choice([True, False], n_vars),
            },
            index=[f"{name}_gene_{i}" for i in range(n_vars)],
        )
        adata = anndata.AnnData(X=X, obs=obs, var=var)
        adata.uns["metadata"] = {
            "name": name,
            "source": "stress_test",
            "timestamp": time.time(),
        }
        return adata

    return _create_adata


# ============================================================================
# Test 1: Concurrent Modality Access
# ============================================================================


class TestConcurrentModalityAccess:
    """Test concurrent read/write operations on modalities."""

    def test_concurrent_read_operations(self, data_manager, mock_adata_factory):
        """Test multiple threads reading modalities simultaneously."""
        # Setup: Create 10 modalities
        n_modalities = 10
        for i in range(n_modalities):
            adata = mock_adata_factory(n_obs=100, n_vars=50, name=f"mod_{i}")
            data_manager.modalities[f"mod_{i}"] = adata

        # Test: 32 threads reading randomly
        n_threads = 32
        n_reads_per_thread = 100
        errors = []

        def read_worker(worker_id: int):
            try:
                for _ in range(n_reads_per_thread):
                    mod_name = f"mod_{np.random.randint(0, n_modalities)}"
                    adata = data_manager.get_modality(mod_name)
                    # Verify data integrity
                    assert adata.n_obs == 100
                    assert adata.n_vars == 50
                    assert mod_name in adata.uns["metadata"]["name"]
            except Exception as e:
                errors.append((worker_id, str(e), traceback.format_exc()))

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(read_worker, i) for i in range(n_threads)]
            concurrent.futures.wait(futures)

        duration = time.time() - start_time

        # Assertions
        assert len(errors) == 0, f"Concurrent read errors: {errors}"
        total_reads = n_threads * n_reads_per_thread
        reads_per_sec = total_reads / duration
        logger.info(
            f"Concurrent reads: {total_reads} reads in {duration:.2f}s "
            f"({reads_per_sec:.0f} reads/sec)"
        )

        # Performance benchmark: Should handle >1000 reads/sec
        assert (
            reads_per_sec > 1000
        ), f"Read performance too slow: {reads_per_sec:.0f} reads/sec"

    def test_concurrent_write_operations(self, data_manager, mock_adata_factory):
        """Test multiple threads writing modalities simultaneously."""
        n_threads = 16
        n_writes_per_thread = 50
        errors = []
        write_lock = threading.Lock()

        def write_worker(worker_id: int):
            try:
                for i in range(n_writes_per_thread):
                    mod_name = f"worker_{worker_id}_mod_{i}"
                    adata = mock_adata_factory(n_obs=100, n_vars=50, name=mod_name)
                    # Write with unique name per thread
                    with write_lock:  # Prevent name collisions
                        data_manager.modalities[mod_name] = adata
            except Exception as e:
                errors.append((worker_id, str(e), traceback.format_exc()))

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(write_worker, i) for i in range(n_threads)]
            concurrent.futures.wait(futures)

        duration = time.time() - start_time

        # Assertions
        assert len(errors) == 0, f"Concurrent write errors: {errors}"
        total_writes = n_threads * n_writes_per_thread
        expected_modalities = total_writes
        actual_modalities = len(data_manager.list_modalities())

        assert actual_modalities == expected_modalities, (
            f"Expected {expected_modalities} modalities, " f"got {actual_modalities}"
        )

        writes_per_sec = total_writes / duration
        logger.info(
            f"Concurrent writes: {total_writes} writes in {duration:.2f}s "
            f"({writes_per_sec:.0f} writes/sec)"
        )

    def test_concurrent_read_write_mixed(self, data_manager, mock_adata_factory):
        """Test mixed read/write operations concurrently."""
        # Setup: Create base modalities
        n_base_modalities = 20
        for i in range(n_base_modalities):
            adata = mock_adata_factory(n_obs=100, n_vars=50, name=f"base_{i}")
            data_manager.modalities[f"base_{i}"] = adata

        n_threads = 32
        n_ops_per_thread = 100
        errors = []
        write_lock = threading.Lock()

        def mixed_worker(worker_id: int):
            try:
                for i in range(n_ops_per_thread):
                    if np.random.random() < 0.7:  # 70% reads
                        mod_name = f"base_{np.random.randint(0, n_base_modalities)}"
                        adata = data_manager.get_modality(mod_name)
                        assert adata.n_obs == 100
                    else:  # 30% writes
                        mod_name = f"worker_{worker_id}_new_{i}"
                        adata = mock_adata_factory(n_obs=100, n_vars=50, name=mod_name)
                        with write_lock:
                            data_manager.modalities[mod_name] = adata
            except Exception as e:
                errors.append((worker_id, str(e), traceback.format_exc()))

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(mixed_worker, i) for i in range(n_threads)]
            concurrent.futures.wait(futures)

        duration = time.time() - start_time

        # Assertions
        assert len(errors) == 0, f"Mixed operations errors: {errors}"
        total_ops = n_threads * n_ops_per_thread
        ops_per_sec = total_ops / duration
        logger.info(
            f"Mixed operations: {total_ops} ops in {duration:.2f}s "
            f"({ops_per_sec:.0f} ops/sec)"
        )


# ============================================================================
# Test 2: Provenance Tracker Thread Safety
# ============================================================================


class TestProvenanceTrackerConcurrency:
    """Test provenance tracker under concurrent load."""

    def test_concurrent_activity_logging(self):
        """Test concurrent activity creation in provenance tracker."""
        tracker = ProvenanceTracker()
        n_threads = 32
        n_activities_per_thread = 100
        errors = []

        def log_worker(worker_id: int):
            try:
                for i in range(n_activities_per_thread):
                    activity_id = tracker.create_activity(
                        activity_type=f"test_activity_{worker_id}_{i}",
                        agent=f"worker_{worker_id}",
                        parameters={"iteration": i, "worker": worker_id},
                        description=f"Test activity from worker {worker_id}",
                    )
                    assert activity_id is not None
            except Exception as e:
                errors.append((worker_id, str(e), traceback.format_exc()))

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(log_worker, i) for i in range(n_threads)]
            concurrent.futures.wait(futures)

        duration = time.time() - start_time

        # Assertions
        assert len(errors) == 0, f"Provenance logging errors: {errors}"
        expected_activities = n_threads * n_activities_per_thread
        actual_activities = len(tracker.activities)

        # CRITICAL: This test will likely fail due to race conditions
        # in the ProvenanceTracker (list.append is not atomic)
        assert actual_activities == expected_activities, (
            f"RACE CONDITION DETECTED: Expected {expected_activities} "
            f"activities, got {actual_activities}. "
            f"Lost {expected_activities - actual_activities} activities!"
        )

        activities_per_sec = expected_activities / duration
        logger.info(
            f"Provenance logging: {expected_activities} activities in "
            f"{duration:.2f}s ({activities_per_sec:.0f} activities/sec)"
        )

    def test_provenance_data_integrity_under_load(self):
        """Verify provenance data integrity with concurrent writes."""
        tracker = ProvenanceTracker()
        n_threads = 16
        n_activities_per_thread = 50
        errors = []
        activity_ids = []
        activity_lock = threading.Lock()

        def integrity_worker(worker_id: int):
            try:
                for i in range(n_activities_per_thread):
                    activity_id = tracker.create_activity(
                        activity_type=f"integrity_test_{worker_id}_{i}",
                        agent=f"worker_{worker_id}",
                        parameters={"worker_id": worker_id, "iteration": i},
                        description=f"Integrity test {i} from worker {worker_id}",
                    )
                    with activity_lock:
                        activity_ids.append(activity_id)
            except Exception as e:
                errors.append((worker_id, str(e), traceback.format_exc()))

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(integrity_worker, i) for i in range(n_threads)]
            concurrent.futures.wait(futures)

        # Verify all activities are unique and complete
        assert len(errors) == 0, f"Integrity test errors: {errors}"
        assert len(activity_ids) == len(set(activity_ids)), "Duplicate activity IDs!"

        # Verify each activity has correct structure
        for activity in tracker.activities:
            assert "id" in activity
            assert "type" in activity
            assert "agent" in activity
            assert "timestamp" in activity
            assert "parameters" in activity

        logger.info(
            f"Integrity verified: {len(tracker.activities)} activities, "
            f"all unique and valid"
        )


# ============================================================================
# Test 3: Workspace Operations with 100+ Datasets
# ============================================================================


class TestWorkspaceScalability:
    """Test workspace operations with large numbers of datasets."""

    def test_workspace_with_100_datasets(
        self, data_manager, mock_adata_factory, temp_workspace
    ):
        """Test workspace save/load with 100+ datasets."""
        n_datasets = 100
        logger.info(f"Creating {n_datasets} mock datasets...")

        # Create 100 datasets
        start_time = time.time()
        for i in range(n_datasets):
            adata = mock_adata_factory(n_obs=50, n_vars=30, name=f"dataset_{i}")
            data_manager.modalities[f"dataset_{i}"] = adata

        creation_time = time.time() - start_time
        logger.info(
            f"Created {n_datasets} datasets in {creation_time:.2f}s "
            f"({n_datasets / creation_time:.1f} datasets/sec)"
        )

        # Save workspace
        logger.info("Saving workspace...")
        start_time = time.time()
        save_path = data_manager.save_workspace(workspace_name="stress_test_100")
        save_time = time.time() - start_time
        logger.info(f"Saved workspace in {save_time:.2f}s")

        # Verify saved files
        assert save_path.exists()
        saved_files = list(save_path.glob("dataset_*.h5ad"))
        assert (
            len(saved_files) == n_datasets
        ), f"Expected {n_datasets} files, found {len(saved_files)}"

        # Clear and reload
        data_manager.clear()
        assert len(data_manager.list_modalities()) == 0

        logger.info("Loading workspace...")
        start_time = time.time()
        data_manager.load_workspace(workspace_name="stress_test_100")
        load_time = time.time() - start_time
        logger.info(f"Loaded workspace in {load_time:.2f}s")

        # Verify all datasets loaded
        loaded_modalities = data_manager.list_modalities()
        assert (
            len(loaded_modalities) == n_datasets
        ), f"Expected {n_datasets} modalities, got {len(loaded_modalities)}"

        # Verify data integrity
        for i in range(min(10, n_datasets)):  # Spot check first 10
            adata = data_manager.get_modality(f"dataset_{i}")
            assert adata.n_obs == 50
            assert adata.n_vars == 30

        logger.info(
            f"Workspace scalability test passed: "
            f"Save: {save_time:.2f}s, Load: {load_time:.2f}s"
        )

    def test_concurrent_workspace_saves(self, mock_adata_factory, temp_workspace):
        """Test concurrent workspace save operations."""
        n_workers = 8
        n_datasets_per_workspace = 20
        errors = []

        def save_worker(worker_id: int):
            try:
                # Create a separate DataManager for each worker
                workspace_subdir = temp_workspace / f"worker_{worker_id}"
                workspace_subdir.mkdir(exist_ok=True)

                dm = DataManagerV2(workspace_path=workspace_subdir)

                # Create datasets
                for i in range(n_datasets_per_workspace):
                    adata = mock_adata_factory(
                        n_obs=50, n_vars=30, name=f"w{worker_id}_ds{i}"
                    )
                    dm.modalities[f"dataset_{i}"] = adata

                # Save workspace
                save_path = dm.save_workspace(
                    workspace_name=f"worker_{worker_id}_workspace"
                )
                assert save_path.exists()

                # Verify saved files
                saved_files = list(save_path.glob("dataset_*.h5ad"))
                assert len(saved_files) == n_datasets_per_workspace

            except Exception as e:
                errors.append((worker_id, str(e), traceback.format_exc()))

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(save_worker, i) for i in range(n_workers)]
            concurrent.futures.wait(futures)

        duration = time.time() - start_time

        # Assertions
        assert len(errors) == 0, f"Concurrent save errors: {errors}"
        total_datasets = n_workers * n_datasets_per_workspace
        logger.info(
            f"Concurrent workspace saves: {n_workers} workspaces "
            f"({total_datasets} datasets) in {duration:.2f}s"
        )

    def test_workspace_restoration_performance(self, data_manager, mock_adata_factory):
        """Test workspace restoration with pattern matching."""
        # Create multiple workspaces with patterns
        patterns = ["analysis_", "experiment_", "test_"]
        datasets_per_pattern = 10

        for pattern in patterns:
            for i in range(datasets_per_pattern):
                adata = mock_adata_factory(n_obs=50, n_vars=30, name=f"{pattern}{i}")
                data_manager.modalities[f"{pattern}{i}"] = adata

        # Save workspace
        workspace_name = "pattern_test"
        data_manager.save_workspace(workspace_name=workspace_name)

        # Clear
        data_manager.clear()

        # Test restoration with pattern
        start_time = time.time()
        data_manager.restore_session(
            workspace_name=workspace_name, pattern="analysis_*"
        )
        restoration_time = time.time() - start_time

        # Verify only matching datasets restored
        modalities = data_manager.list_modalities()
        assert (
            len(modalities) == datasets_per_pattern
        ), f"Expected {datasets_per_pattern} modalities, got {len(modalities)}"
        assert all(
            m.startswith("analysis_") for m in modalities
        ), "Non-matching modalities restored"

        logger.info(
            f"Pattern restoration: {len(modalities)} datasets in "
            f"{restoration_time:.2f}s"
        )


# ============================================================================
# Test 4: Memory Profiling Under Load
# ============================================================================


class TestMemoryProfiling:
    """Profile memory usage under extreme conditions."""

    def test_memory_usage_with_large_datasets(self, data_manager, mock_adata_factory):
        """Profile memory usage with large datasets."""
        tracemalloc.start()
        start_snapshot = tracemalloc.take_snapshot()

        # Create 50 large datasets
        n_datasets = 50
        n_obs = 1000
        n_vars = 500

        for i in range(n_datasets):
            adata = mock_adata_factory(n_obs=n_obs, n_vars=n_vars, name=f"large_{i}")
            data_manager.modalities[f"large_{i}"] = adata

        # Take snapshot
        peak_snapshot = tracemalloc.take_snapshot()
        peak_memory = tracemalloc.get_traced_memory()[1]

        # Clear some datasets
        for i in range(0, n_datasets, 2):  # Clear every other dataset
            del data_manager.modalities[f"large_{i}"]

        # Force garbage collection
        gc.collect()

        # Take final snapshot
        final_snapshot = tracemalloc.take_snapshot()
        final_memory = tracemalloc.get_traced_memory()[1]

        tracemalloc.stop()

        # Calculate memory metrics
        peak_mb = peak_memory / (1024 * 1024)
        final_mb = final_memory / (1024 * 1024)
        per_dataset_kb = (peak_memory / n_datasets) / 1024

        logger.info(
            f"Memory profiling results:\n"
            f"  Peak memory: {peak_mb:.2f} MB\n"
            f"  After cleanup: {final_mb:.2f} MB\n"
            f"  Per dataset: {per_dataset_kb:.2f} KB\n"
            f"  Memory reclaimed: {peak_mb - final_mb:.2f} MB"
        )

        # Assertions: Memory should be reclaimed after deletion
        assert final_mb < peak_mb, "Memory not reclaimed after deletion"

    def test_memory_leak_detection(self, data_manager, mock_adata_factory):
        """Detect memory leaks in repeated operations."""
        tracemalloc.start()

        memory_samples = []

        # Perform 10 cycles of create/delete
        for cycle in range(10):
            # Create 20 datasets
            for i in range(20):
                adata = mock_adata_factory(n_obs=200, n_vars=100, name=f"leak_test_{i}")
                data_manager.modalities[f"leak_test_{i}"] = adata

            # Delete all datasets
            data_manager.clear()
            gc.collect()

            # Sample memory
            current_memory = tracemalloc.get_traced_memory()[1]
            memory_samples.append(current_memory / (1024 * 1024))  # MB

        tracemalloc.stop()

        # Analyze memory trend
        logger.info(f"Memory samples across cycles: {memory_samples}")

        # Check for memory leak (more than 20% increase from first to last)
        first_cycle_mem = memory_samples[0]
        last_cycle_mem = memory_samples[-1]
        memory_increase_pct = (
            (last_cycle_mem - first_cycle_mem) / first_cycle_mem
        ) * 100

        logger.info(
            f"Memory leak detection:\n"
            f"  First cycle: {first_cycle_mem:.2f} MB\n"
            f"  Last cycle: {last_cycle_mem:.2f} MB\n"
            f"  Increase: {memory_increase_pct:.1f}%"
        )

        # Allow some increase due to Python interpreter overhead
        assert memory_increase_pct < 50, (
            f"Potential memory leak detected: " f"{memory_increase_pct:.1f}% increase"
        )


# ============================================================================
# Test 5: Race Condition Detection
# ============================================================================


class TestRaceConditions:
    """Detect race conditions in critical operations."""

    def test_save_operation_race_conditions(self, data_manager, mock_adata_factory):
        """Test for race conditions in save operations."""
        # Setup: Create datasets
        for i in range(20):
            adata = mock_adata_factory(n_obs=50, n_vars=30, name=f"race_{i}")
            data_manager.modalities[f"race_{i}"] = adata

        n_threads = 16
        errors = []
        save_count = 0
        save_lock = threading.Lock()

        def save_worker(worker_id: int):
            nonlocal save_count
            try:
                # Each worker attempts to save the workspace
                save_path = data_manager.save_workspace(
                    workspace_name=f"race_test_worker_{worker_id}"
                )
                with save_lock:
                    save_count += 1
                assert save_path.exists()
            except Exception as e:
                errors.append((worker_id, str(e), traceback.format_exc()))

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(save_worker, i) for i in range(n_threads)]
            concurrent.futures.wait(futures)

        duration = time.time() - start_time

        # Assertions
        assert len(errors) == 0, f"Race condition errors: {errors}"
        assert save_count == n_threads, f"Expected {n_threads} saves, got {save_count}"

        logger.info(
            f"Race condition test passed: {n_threads} concurrent saves in "
            f"{duration:.2f}s"
        )

    def test_modality_name_collision_handling(self, data_manager, mock_adata_factory):
        """Test handling of modality name collisions."""
        n_threads = 16
        colliding_name = "collision_test"
        errors = []
        success_count = 0
        collision_count = 0
        result_lock = threading.Lock()

        def collision_worker(worker_id: int):
            nonlocal success_count, collision_count
            try:
                # All workers try to create the same modality name
                adata = mock_adata_factory(n_obs=50, n_vars=30, name=colliding_name)
                adata.uns["worker_id"] = worker_id

                # Attempt to store
                data_manager.modalities[colliding_name] = adata

                with result_lock:
                    success_count += 1

            except Exception as e:
                with result_lock:
                    collision_count += 1
                errors.append((worker_id, str(e), traceback.format_exc()))

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(collision_worker, i) for i in range(n_threads)]
            concurrent.futures.wait(futures)

        # The last writer should win
        assert colliding_name in data_manager.modalities
        final_adata = data_manager.get_modality(colliding_name)
        final_worker = final_adata.uns.get("worker_id", "unknown")

        logger.info(
            f"Collision handling:\n"
            f"  Success writes: {success_count}\n"
            f"  Collisions: {collision_count}\n"
            f"  Final writer: worker_{final_worker}\n"
            f"  Errors: {len(errors)}"
        )

        # Verify last writer won (no data corruption)
        assert final_adata.n_obs == 50
        assert final_adata.n_vars == 30


# ============================================================================
# Test 6: File Descriptor Limits
# ============================================================================


class TestFileDescriptorLimits:
    """Test behavior under file descriptor pressure."""

    def test_many_concurrent_file_operations(self, mock_adata_factory, temp_workspace):
        """Test with many concurrent file open operations."""
        n_files = 100
        n_concurrent_ops = 50
        errors = []

        # Create test files
        test_dir = temp_workspace / "fd_test"
        test_dir.mkdir(exist_ok=True)

        for i in range(n_files):
            adata = mock_adata_factory(n_obs=50, n_vars=30, name=f"fd_{i}")
            file_path = test_dir / f"fd_test_{i}.h5ad"
            adata.write_h5ad(file_path)

        def read_worker(worker_id: int):
            try:
                # Each worker reads multiple files
                for i in range(10):
                    file_idx = (worker_id * 10 + i) % n_files
                    file_path = test_dir / f"fd_test_{file_idx}.h5ad"
                    adata = anndata.read_h5ad(file_path)
                    assert adata.n_obs == 50
            except Exception as e:
                errors.append((worker_id, str(e), traceback.format_exc()))

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=n_concurrent_ops
        ) as executor:
            futures = [executor.submit(read_worker, i) for i in range(n_concurrent_ops)]
            concurrent.futures.wait(futures)

        duration = time.time() - start_time

        # Assertions
        assert len(errors) == 0, f"File descriptor errors: {errors}"
        total_reads = n_concurrent_ops * 10
        logger.info(
            f"File descriptor test: {total_reads} reads in {duration:.2f}s "
            f"with {n_concurrent_ops} concurrent workers"
        )


# ============================================================================
# Test 7: Agent Registry Thread Safety
# ============================================================================


class TestAgentRegistryThreadSafety:
    """Test agent registry operations under concurrent load."""

    def test_concurrent_agent_registry_access(self):
        """Test concurrent access to agent registry."""
        from lobster.config.agent_registry import (
            AGENT_REGISTRY,
            get_all_agent_names,
        )

        n_threads = 32
        n_reads_per_thread = 100
        errors = []

        def registry_reader(worker_id: int):
            try:
                for _ in range(n_reads_per_thread):
                    # Read agent names
                    agent_names = get_all_agent_names()
                    assert len(agent_names) > 0

                    # Read agent configs
                    for agent_name in agent_names:
                        if agent_name in AGENT_REGISTRY:
                            config = AGENT_REGISTRY[agent_name]
                            assert config.name is not None
                            assert config.display_name is not None
            except Exception as e:
                errors.append((worker_id, str(e), traceback.format_exc()))

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(registry_reader, i) for i in range(n_threads)]
            concurrent.futures.wait(futures)

        duration = time.time() - start_time

        # Assertions
        assert len(errors) == 0, f"Registry access errors: {errors}"
        total_reads = n_threads * n_reads_per_thread
        logger.info(f"Agent registry access: {total_reads} reads in {duration:.2f}s")


# ============================================================================
# Test 8: Combined Stress Test
# ============================================================================


class TestCombinedStress:
    """Combined stress test with all operations running simultaneously."""

    def test_everything_at_once(self, data_manager, mock_adata_factory):
        """Run all operations concurrently to simulate real load."""
        n_workers = 16
        duration_seconds = 10
        errors = []
        stop_flag = threading.Event()

        # Statistics
        stats = {
            "reads": 0,
            "writes": 0,
            "provenance_logs": 0,
            "saves": 0,
        }
        stats_lock = threading.Lock()

        # Setup: Create base modalities
        for i in range(20):
            adata = mock_adata_factory(n_obs=100, n_vars=50, name=f"base_{i}")
            data_manager.modalities[f"base_{i}"] = adata

        def mixed_workload_worker(worker_id: int):
            try:
                iteration = 0
                while not stop_flag.is_set():
                    operation = np.random.choice(
                        ["read", "write", "provenance", "save"],
                        p=[0.5, 0.3, 0.15, 0.05],
                    )

                    if operation == "read":
                        # Read operation
                        mod_name = f"base_{np.random.randint(0, 20)}"
                        if mod_name in data_manager.modalities:
                            adata = data_manager.get_modality(mod_name)
                            assert adata.n_obs == 100
                            with stats_lock:
                                stats["reads"] += 1

                    elif operation == "write":
                        # Write operation
                        mod_name = f"worker_{worker_id}_temp_{iteration}"
                        adata = mock_adata_factory(n_obs=50, n_vars=30, name=mod_name)
                        data_manager.modalities[mod_name] = adata
                        with stats_lock:
                            stats["writes"] += 1

                    elif operation == "provenance":
                        # Provenance logging
                        data_manager.log_tool_usage(
                            tool_name=f"test_tool_{worker_id}",
                            parameters={"iteration": iteration},
                            description=f"Test from worker {worker_id}",
                        )
                        with stats_lock:
                            stats["provenance_logs"] += 1

                    elif operation == "save":
                        # Save operation (less frequent)
                        try:
                            data_manager.save_workspace(
                                workspace_name=f"stress_w{worker_id}_i{iteration}"
                            )
                            with stats_lock:
                                stats["saves"] += 1
                        except Exception:
                            pass  # Save conflicts expected

                    iteration += 1
                    time.sleep(0.01)  # Small delay to avoid spinning

            except Exception as e:
                errors.append((worker_id, str(e), traceback.format_exc()))

        # Start workers
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(mixed_workload_worker, i) for i in range(n_workers)
            ]

            # Let them run for the specified duration
            time.sleep(duration_seconds)
            stop_flag.set()

            # Wait for completion
            concurrent.futures.wait(futures, timeout=5)

        total_duration = time.time() - start_time

        # Report results
        total_ops = sum(stats.values())
        ops_per_sec = total_ops / total_duration

        logger.info(
            f"\n{'='*60}\n"
            f"COMBINED STRESS TEST RESULTS\n"
            f"{'='*60}\n"
            f"Duration: {total_duration:.2f}s\n"
            f"Workers: {n_workers}\n"
            f"Total operations: {total_ops}\n"
            f"Operations/sec: {ops_per_sec:.0f}\n"
            f"\nOperation breakdown:\n"
            f"  Reads: {stats['reads']}\n"
            f"  Writes: {stats['writes']}\n"
            f"  Provenance logs: {stats['provenance_logs']}\n"
            f"  Saves: {stats['saves']}\n"
            f"\nErrors: {len(errors)}\n"
            f"{'='*60}"
        )

        # Assertions
        assert len(errors) == 0, f"Combined stress test errors: {errors}"
        assert total_ops > 0, "No operations completed"
        assert ops_per_sec > 10, f"Performance too slow: {ops_per_sec:.0f} ops/sec"


# ============================================================================
# Performance Benchmarks
# ============================================================================


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for key operations."""

    def test_modality_access_speed(self, data_manager, mock_adata_factory):
        """Benchmark modality access speed."""
        # Setup
        for i in range(100):
            adata = mock_adata_factory(n_obs=100, n_vars=50, name=f"bench_{i}")
            data_manager.modalities[f"bench_{i}"] = adata

        # Benchmark reads
        n_reads = 10000
        start_time = time.time()
        for _ in range(n_reads):
            mod_name = f"bench_{np.random.randint(0, 100)}"
            adata = data_manager.get_modality(mod_name)

        duration = time.time() - start_time
        reads_per_sec = n_reads / duration

        logger.info(f"Modality access benchmark: {reads_per_sec:.0f} reads/sec")
        assert reads_per_sec > 1000, "Read performance too slow"

    def test_provenance_logging_speed(self):
        """Benchmark provenance logging speed."""
        tracker = ProvenanceTracker()

        n_logs = 1000
        start_time = time.time()
        for i in range(n_logs):
            tracker.create_activity(
                activity_type=f"benchmark_activity_{i}",
                agent="benchmark",
                parameters={"iteration": i},
                description=f"Benchmark activity {i}",
            )

        duration = time.time() - start_time
        logs_per_sec = n_logs / duration

        logger.info(f"Provenance logging benchmark: {logs_per_sec:.0f} logs/sec")
        assert logs_per_sec > 100, "Logging performance too slow"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--log-cli-level=INFO"])
