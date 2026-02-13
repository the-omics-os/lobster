"""
Agent 18: Large-Scale Stress Testing for Lobster Platform

Focused performance and scalability testing with large datasets:
- 50K-100K cell single-cell datasets
- Memory profiling and leak detection
- Concurrent operations scaling
- File I/O performance
- Provenance scaling

Focus: Raw performance metrics, not full pipeline workflows.
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

import anndata
import numpy as np
import pandas as pd
import psutil
import pytest
from scipy import sparse

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


# ============================================================================
# Test Configuration
# ============================================================================

# Mark all tests as slow
pytestmark = pytest.mark.slow


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_workspace():
    """Create temporary workspace."""
    temp_dir = tempfile.mkdtemp(prefix="agent18_stress_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def data_manager(temp_workspace):
    """Create DataManagerV2 instance."""
    return DataManagerV2(workspace_path=str(temp_workspace))


# ============================================================================
# Helper Functions
# ============================================================================


def create_synthetic_sc_data(n_obs: int, n_vars: int = 2000, density: float = 0.1):
    """Create synthetic single-cell dataset."""
    logger.info(f"Creating synthetic dataset: {n_obs} × {n_vars} (density={density})")
    start = time.time()

    # Create sparse matrix (realistic scRNA-seq sparsity)
    X = sparse.random(n_obs, n_vars, density=density, format="csr", dtype=np.float32)
    X.data = np.random.negative_binomial(5, 0.3, size=len(X.data)).astype(np.float32)

    # Observations
    obs = pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in range(n_obs)],
            "n_genes": np.random.randint(500, 5000, n_obs),
            "n_counts": np.random.randint(1000, 50000, n_obs),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )

    # Variables
    var = pd.DataFrame(
        {
            "gene": [f"GENE{i}" for i in range(n_vars)],
        },
        index=[f"gene_{i}" for i in range(n_vars)],
    )

    adata = anndata.AnnData(X=X, obs=obs, var=var)

    duration = time.time() - start
    logger.info(f"Created in {duration:.2f}s")

    return adata, duration


def get_memory_info():
    """Get current process memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        "rss_mb": mem_info.rss / (1024 * 1024),  # Resident Set Size
        "vms_mb": mem_info.vms / (1024 * 1024),  # Virtual Memory Size
    }


# ============================================================================
# Test 1: 50K Cell Creation and Basic Operations
# ============================================================================


class TestLargeDatasetCreation:
    """Test large dataset creation and basic operations."""

    def test_50k_cell_creation_and_storage(self, data_manager):
        """Test creating and storing 50K cell dataset."""
        logger.info("\n" + "=" * 70)
        logger.info("TEST: 50K Cell Creation and Storage")
        logger.info("=" * 70)

        mem_before = get_memory_info()
        tracemalloc.start()

        # Create dataset
        adata, creation_time = create_synthetic_sc_data(n_obs=50000, n_vars=2000)

        # Store in data manager
        store_start = time.time()
        data_manager.modalities["test_50k"] = adata
        store_time = time.time() - store_start

        # Basic operations
        ops_start = time.time()
        retrieved = data_manager.get_modality("test_50k")
        assert retrieved.n_obs == 50000
        assert retrieved.n_vars == 2000
        ops_time = time.time() - ops_start

        mem_after = get_memory_info()
        peak_traced = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        tracemalloc.stop()

        # Results
        logger.info("\nRESULTS:")
        logger.info(f"  Creation time: {creation_time:.2f}s")
        logger.info(f"  Store time: {store_time:.4f}s")
        logger.info(f"  Retrieval time: {ops_time:.4f}s")
        logger.info(f"  Memory before: {mem_before['rss_mb']:.2f} MB")
        logger.info(f"  Memory after: {mem_after['rss_mb']:.2f} MB")
        logger.info(
            f"  Memory used: {mem_after['rss_mb'] - mem_before['rss_mb']:.2f} MB"
        )
        logger.info(f"  Peak traced: {peak_traced:.2f} MB")
        logger.info(f"  Cells/second: {50000 / creation_time:.0f}")
        logger.info(
            f"  Memory per cell: {(mem_after['rss_mb'] - mem_before['rss_mb']) / 50000 * 1000:.2f} KB"
        )
        logger.info("=" * 70)

        # Performance assertions
        assert creation_time < 60, f"Creation too slow: {creation_time:.0f}s"
        assert (
            mem_after["rss_mb"] - mem_before["rss_mb"]
        ) < 2000, "Memory usage too high"

    def test_100k_cell_memory_profile(self, data_manager):
        """Test 100K cell dataset memory usage."""
        logger.info("\n" + "=" * 70)
        logger.info("TEST: 100K Cell Memory Profiling")
        logger.info("=" * 70)

        mem_before = get_memory_info()
        tracemalloc.start()
        gc.collect()

        # Create 100K dataset
        adata, creation_time = create_synthetic_sc_data(n_obs=100000, n_vars=2000)

        mem_after_create = get_memory_info()

        # Store it
        data_manager.modalities["test_100k"] = adata
        mem_after_store = get_memory_info()

        # Delete and check reclamation
        del adata
        del data_manager.modalities["test_100k"]
        gc.collect()

        mem_after_delete = get_memory_info()

        peak_traced = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        tracemalloc.stop()

        # Calculate metrics
        mem_created = mem_after_create["rss_mb"] - mem_before["rss_mb"]
        mem_stored = mem_after_store["rss_mb"] - mem_before["rss_mb"]
        mem_final = mem_after_delete["rss_mb"] - mem_before["rss_mb"]
        reclaimed = mem_stored - mem_final
        reclaimed_pct = (reclaimed / mem_stored * 100) if mem_stored > 0 else 0

        logger.info("\nRESULTS:")
        logger.info(f"  Creation time: {creation_time:.2f}s")
        logger.info(f"  Memory after create: +{mem_created:.2f} MB")
        logger.info(f"  Memory after store: +{mem_stored:.2f} MB")
        logger.info(f"  Memory after delete: +{mem_final:.2f} MB")
        logger.info(f"  Reclaimed: {reclaimed:.2f} MB ({reclaimed_pct:.1f}%)")
        logger.info(f"  Peak traced: {peak_traced:.2f} MB")
        logger.info(f"  Cells/second: {100000 / creation_time:.0f}")
        logger.info("=" * 70)

        # Check for memory leak
        if reclaimed_pct < 50:
            logger.warning(
                f"⚠️  POTENTIAL MEMORY LEAK: Only {reclaimed_pct:.1f}% reclaimed"
            )

        assert creation_time < 120, f"Creation too slow: {creation_time:.0f}s"


# ============================================================================
# Test 2: Memory Leak Detection
# ============================================================================


class TestMemoryLeaks:
    """Detect memory leaks through repeated cycles."""

    def test_repeated_create_delete_cycles(self, data_manager):
        """Test for memory leaks over 20 create/delete cycles."""
        logger.info("\n" + "=" * 70)
        logger.info("TEST: Memory Leak Detection (20 Cycles)")
        logger.info("=" * 70)

        tracemalloc.start()
        gc.collect()

        memory_samples = []
        time_samples = []

        # Run 20 cycles of create/delete
        for cycle in range(20):
            cycle_start = time.time()

            # Create 5 datasets of 10K cells each
            for i in range(5):
                adata, _ = create_synthetic_sc_data(
                    n_obs=10000, n_vars=1000, density=0.1
                )
                data_manager.modalities[f"cycle{cycle}_ds{i}"] = adata

            # Delete all
            for i in range(5):
                del data_manager.modalities[f"cycle{cycle}_ds{i}"]

            gc.collect()

            # Sample memory
            current_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
            memory_samples.append(current_mem)

            cycle_time = time.time() - cycle_start
            time_samples.append(cycle_time)

            if cycle % 5 == 0:
                logger.info(f"Cycle {cycle}: {current_mem:.2f} MB, {cycle_time:.2f}s")

        tracemalloc.stop()

        # Analyze trends
        first_5_mem = np.mean(memory_samples[:5])
        last_5_mem = np.mean(memory_samples[-5:])
        mem_increase_pct = ((last_5_mem - first_5_mem) / first_5_mem) * 100

        first_5_time = np.mean(time_samples[:5])
        last_5_time = np.mean(time_samples[-5:])
        time_increase_pct = ((last_5_time - first_5_time) / first_5_time) * 100

        logger.info("\nRESULTS:")
        logger.info(f"  First 5 cycles mem avg: {first_5_mem:.2f} MB")
        logger.info(f"  Last 5 cycles mem avg: {last_5_mem:.2f} MB")
        logger.info(f"  Memory increase: {mem_increase_pct:.1f}%")
        logger.info(f"  First 5 cycles time avg: {first_5_time:.2f}s")
        logger.info(f"  Last 5 cycles time avg: {last_5_time:.2f}s")
        logger.info(f"  Time increase: {time_increase_pct:.1f}%")
        logger.info("=" * 70)

        # Alerts
        if mem_increase_pct > 30:
            logger.error(f"⚠️  MEMORY LEAK DETECTED: {mem_increase_pct:.1f}% increase")
        if time_increase_pct > 20:
            logger.warning(
                f"⚠️  PERFORMANCE DEGRADATION: {time_increase_pct:.1f}% slower"
            )

        # Allow some increase but not excessive
        assert mem_increase_pct < 50, f"Excessive memory leak: {mem_increase_pct:.1f}%"


# ============================================================================
# Test 3: Concurrent Operations
# ============================================================================


class TestConcurrentScaling:
    """Test concurrent operations at scale."""

    def test_32_workers_concurrent_access(self, data_manager):
        """Test 32 workers doing concurrent read/write operations."""
        logger.info("\n" + "=" * 70)
        logger.info("TEST: 32 Workers Concurrent Operations")
        logger.info("=" * 70)

        # Create base datasets
        logger.info("Creating 20 base datasets...")
        for i in range(20):
            adata, _ = create_synthetic_sc_data(n_obs=1000, n_vars=500, density=0.1)
            data_manager.modalities[f"base_{i}"] = adata

        n_workers = 32
        n_ops_per_worker = 100
        errors = []

        stats = {"reads": 0, "writes": 0}
        stats_lock = threading.Lock()

        def worker(worker_id: int):
            try:
                for i in range(n_ops_per_worker):
                    if np.random.random() < 0.7:  # 70% reads
                        dataset_id = np.random.randint(0, 20)
                        adata = data_manager.get_modality(f"base_{dataset_id}")
                        assert adata.n_obs == 1000
                        with stats_lock:
                            stats["reads"] += 1
                    else:  # 30% writes
                        adata, _ = create_synthetic_sc_data(
                            n_obs=500, n_vars=300, density=0.1
                        )
                        name = f"worker{worker_id}_temp{i}"
                        data_manager.modalities[name] = adata
                        with stats_lock:
                            stats["writes"] += 1
            except Exception as e:
                errors.append((worker_id, str(e)))

        # Run workers
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(worker, i) for i in range(n_workers)]
            concurrent.futures.wait(futures)

        duration = time.time() - start_time
        total_ops = sum(stats.values())
        ops_per_sec = total_ops / duration

        logger.info("\nRESULTS:")
        logger.info(f"  Duration: {duration:.2f}s")
        logger.info(f"  Total operations: {total_ops}")
        logger.info(f"  Operations/second: {ops_per_sec:.0f}")
        logger.info(f"  Reads: {stats['reads']}")
        logger.info(f"  Writes: {stats['writes']}")
        logger.info(f"  Errors: {len(errors)}")
        logger.info(f"  Final modality count: {len(data_manager.list_modalities())}")
        logger.info("=" * 70)

        assert len(errors) == 0, f"Errors occurred: {errors[:5]}"
        assert ops_per_sec > 50, f"Performance too slow: {ops_per_sec:.0f} ops/sec"


# ============================================================================
# Test 4: Provenance Scaling
# ============================================================================


class TestProvenanceScaling:
    """Test provenance tracking at scale."""

    def test_10k_provenance_activities(self):
        """Test creating and retrieving 10K provenance activities."""
        logger.info("\n" + "=" * 70)
        logger.info("TEST: 10K Provenance Activities")
        logger.info("=" * 70)

        tracker = ProvenanceTracker()
        n_activities = 10000

        # Create activities
        create_start = time.time()
        for i in range(n_activities):
            tracker.create_activity(
                activity_type=f"test_activity_{i}",
                agent="stress_test",
                parameters={"iteration": i, "value": np.random.random()},
                description=f"Stress test activity {i}",
            )
        create_time = time.time() - create_start

        # Retrieve all
        retrieve_start = time.time()
        activities = tracker.get_all_activities()
        retrieve_time = time.time() - retrieve_start

        # Export
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = Path(temp_dir) / "provenance.json"
            export_start = time.time()
            tracker.export_to_json(str(export_path))
            export_time = time.time() - export_start
            file_size_mb = export_path.stat().st_size / (1024 * 1024)

        logger.info("\nRESULTS:")
        logger.info(
            f"  Creation time: {create_time:.2f}s ({n_activities / create_time:.0f} activities/sec)"
        )
        logger.info(f"  Retrieval time: {retrieve_time:.4f}s")
        logger.info(f"  Export time: {export_time:.2f}s")
        logger.info(f"  Export file size: {file_size_mb:.2f} MB")
        logger.info(f"  Size per activity: {file_size_mb / n_activities * 1000:.2f} KB")
        logger.info("=" * 70)

        assert len(activities) == n_activities
        assert n_activities / create_time > 1000, "Creation too slow"
        assert retrieve_time < 1.0, f"Retrieval too slow: {retrieve_time:.2f}s"
        assert export_time < 10.0, f"Export too slow: {export_time:.2f}s"


# ============================================================================
# Test 5: File I/O Performance
# ============================================================================


class TestFileIOPerformance:
    """Test file I/O performance with large datasets."""

    def test_50k_cell_save_load_performance(self, temp_workspace):
        """Test H5AD save/load performance with 50K cells."""
        logger.info("\n" + "=" * 70)
        logger.info("TEST: File I/O Performance (50K cells)")
        logger.info("=" * 70)

        # Create dataset
        adata, creation_time = create_synthetic_sc_data(n_obs=50000, n_vars=2000)

        # Save
        save_path = temp_workspace / "test_50k.h5ad"
        save_start = time.time()
        adata.write_h5ad(save_path)
        save_time = time.time() - save_start

        file_size_mb = save_path.stat().st_size / (1024 * 1024)

        # Load
        load_start = time.time()
        adata_loaded = anndata.read_h5ad(save_path)
        load_time = time.time() - load_start

        # Calculate metrics
        save_speed = file_size_mb / save_time
        load_speed = file_size_mb / load_time

        logger.info("\nRESULTS:")
        logger.info(f"  File size: {file_size_mb:.2f} MB")
        logger.info(f"  Save time: {save_time:.2f}s ({save_speed:.2f} MB/s)")
        logger.info(f"  Load time: {load_time:.2f}s ({load_speed:.2f} MB/s)")
        logger.info(
            f"  Data verified: {adata_loaded.n_obs == 50000 and adata_loaded.n_vars == 2000}"
        )
        logger.info("=" * 70)

        assert adata_loaded.n_obs == 50000
        assert adata_loaded.n_vars == 2000
        assert save_time < 60, f"Save too slow: {save_time:.0f}s"
        assert load_time < 30, f"Load too slow: {load_time:.0f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--log-cli-level=INFO"])
