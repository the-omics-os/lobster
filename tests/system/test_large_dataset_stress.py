"""
Comprehensive large dataset stress testing for Lobster platform.

This module extends Agent 4's stress tests with:
- 50K-100K cell single-cell analysis
- 10K sample proteomics workflows
- Deep memory profiling and leak detection
- Comprehensive performance benchmarking
- Scaling limit identification

Test Categories:
1. Large-Scale Single-Cell Analysis (50K-100K cells)
2. Large-Scale Proteomics (10K samples)
3. Memory Profiling & Leak Detection
4. File I/O Performance Analysis
5. Concurrent Operations at Scale (16-32 workers)
6. End-to-End Pipeline Performance
7. Provenance Scaling (10K+ activities)
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
from scipy import sparse

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.provenance import ProvenanceTracker
from lobster.tools.clustering_service import ClusteringService
from lobster.tools.enhanced_singlecell_service import EnhancedSingleCellService
from lobster.tools.preprocessing_service import PreprocessingService
from lobster.tools.quality_service import QualityService

logger = logging.getLogger(__name__)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing."""
    temp_dir = tempfile.mkdtemp(prefix="lobster_large_stress_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def data_manager(temp_workspace):
    """Create DataManagerV2 instance."""
    return DataManagerV2(workspace_path=str(temp_workspace))


@pytest.fixture
def services():
    """Create service instances (stateless services)."""
    return {
        "preprocessing": PreprocessingService(),
        "quality": QualityService(),
        "clustering": ClusteringService(),
        "enhanced_sc": EnhancedSingleCellService(),
    }


def create_large_singlecell_adata(
    n_obs: int,
    n_vars: int = 2000,
    use_sparse: bool = True,
    name: str = "large_sc",
) -> anndata.AnnData:
    """Create large single-cell dataset with realistic structure."""
    logger.info(f"Creating large single-cell dataset: {n_obs} cells × {n_vars} genes")

    # Create sparse or dense matrix
    if use_sparse:
        # Realistic sparsity for scRNA-seq (90-95% zeros)
        density = 0.1
        X = sparse.random(n_obs, n_vars, density=density, format="csr")
        X.data = np.random.negative_binomial(5, 0.3, size=len(X.data))
    else:
        X = np.random.negative_binomial(5, 0.3, (n_obs, n_vars))

    # Realistic observations
    obs = pd.DataFrame(
        {
            "cell_type": np.random.choice(
                ["T_cell", "B_cell", "NK_cell", "Monocyte", "DC"], n_obs
            ),
            "batch": np.random.choice([f"batch_{i}" for i in range(10)], n_obs),
            "sample": np.random.choice([f"sample_{i}" for i in range(20)], n_obs),
            "n_genes": np.random.randint(500, 5000, n_obs),
            "n_counts": np.random.randint(1000, 50000, n_obs),
            "pct_counts_mt": np.random.uniform(0, 30, n_obs),
            "pct_counts_ribo": np.random.uniform(0, 50, n_obs),
        },
        index=[f"{name}_cell_{i}" for i in range(n_obs)],
    )

    # Realistic variables
    var = pd.DataFrame(
        {
            "gene_symbol": [f"GENE{i}" for i in range(n_vars)],
            "highly_variable": np.random.choice([True, False], n_vars, p=[0.2, 0.8]),
            "mean_counts": np.random.exponential(10, n_vars),
            "dispersions": np.random.exponential(1, n_vars),
        },
        index=[f"{name}_gene_{i}" for i in range(n_vars)],
    )

    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.uns["metadata"] = {
        "name": name,
        "source": "stress_test_large",
        "n_obs": n_obs,
        "n_vars": n_vars,
        "sparse": use_sparse,
        "created": time.time(),
    }

    return adata


def create_large_proteomics_adata(
    n_obs: int,
    n_vars: int = 5000,
    missing_rate: float = 0.3,
    name: str = "large_prot",
) -> anndata.AnnData:
    """Create large proteomics dataset with realistic missing values."""
    logger.info(
        f"Creating large proteomics dataset: {n_obs} samples × {n_vars} proteins"
    )

    # Intensity values (log2-transformed)
    X = np.random.normal(20, 3, (n_obs, n_vars))

    # Add realistic missing values (MNAR pattern)
    missing_mask = np.random.random((n_obs, n_vars)) < missing_rate
    X[missing_mask] = np.nan

    # Observations
    obs = pd.DataFrame(
        {
            "sample_id": [f"{name}_sample_{i}" for i in range(n_obs)],
            "condition": np.random.choice(["control", "treatment"], n_obs),
            "batch": np.random.choice([f"batch_{i}" for i in range(5)], n_obs),
            "replicate": np.random.randint(1, 4, n_obs),
            "missing_pct": np.sum(missing_mask, axis=1) / n_vars * 100,
        },
        index=[f"{name}_obs_{i}" for i in range(n_obs)],
    )

    # Variables
    var = pd.DataFrame(
        {
            "protein_id": [f"PROT{i:05d}" for i in range(n_vars)],
            "gene_name": [f"GENE{i}" for i in range(n_vars)],
            "missing_pct": np.sum(missing_mask, axis=0) / n_obs * 100,
            "mean_intensity": np.nanmean(X, axis=0),
            "cv": np.nanstd(X, axis=0) / np.nanmean(X, axis=0),
        },
        index=[f"{name}_var_{i}" for i in range(n_vars)],
    )

    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.uns["metadata"] = {
        "name": name,
        "source": "stress_test_large_proteomics",
        "n_obs": n_obs,
        "n_vars": n_vars,
        "missing_rate": missing_rate,
        "created": time.time(),
    }

    return adata


# ============================================================================
# Test 1: 50K Cell Single-Cell Analysis
# ============================================================================


@pytest.mark.slow
class TestLargeScaleSingleCell:
    """Test large-scale single-cell analysis workflows."""

    def test_50k_cell_full_pipeline(self, data_manager, services):
        """Test full pipeline with 50K cells."""
        logger.info("\n" + "=" * 70)
        logger.info("STRESS TEST: 50K Cell Full Pipeline")
        logger.info("=" * 70)

        # Track memory
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]

        # Create dataset
        creation_start = time.time()
        adata = create_large_singlecell_adata(
            n_obs=50000, n_vars=2000, use_sparse=True, name="stress_50k"
        )
        creation_time = time.time() - creation_start

        data_manager.modalities["stress_50k"] = adata
        logger.info(f"Dataset creation: {creation_time:.2f}s")

        # Step 1: Quality control
        qc_start = time.time()
        adata_qc = services["quality"].assess_quality(adata.copy())
        qc_time = time.time() - qc_start
        data_manager.modalities["stress_50k_qc"] = adata_qc
        logger.info(f"Quality control: {qc_time:.2f}s")

        # Step 2: Preprocessing
        preproc_start = time.time()
        adata_proc, _ = services["preprocessing"].preprocess(
            adata_qc.copy(),
            filter_cells=True,
            min_genes=200,
            normalize=True,
            log_transform=True,
            scale=False,  # Skip scaling to save memory
        )
        preproc_time = time.time() - preproc_start
        data_manager.modalities["stress_50k_proc"] = adata_proc
        logger.info(f"Preprocessing: {preproc_time:.2f}s")

        # Step 3: Find highly variable genes
        hvg_start = time.time()
        adata_hvg, hvg_stats = services["preprocessing"].identify_highly_variable_genes(
            adata_proc.copy(), n_top_genes=2000
        )
        hvg_time = time.time() - hvg_start
        data_manager.modalities["stress_50k_hvg"] = adata_hvg
        logger.info(f"HVG selection: {hvg_time:.2f}s")

        # Step 4: PCA
        pca_start = time.time()
        adata_pca, pca_stats = services["clustering"].compute_pca(
            adata_hvg.copy(), n_comps=50
        )
        pca_time = time.time() - pca_start
        data_manager.modalities["stress_50k_pca"] = adata_pca
        logger.info(f"PCA: {pca_time:.2f}s")

        # Step 5: Neighbors
        neighbors_start = time.time()
        adata_neighbors, neighbors_stats = services["clustering"].compute_neighbors(
            adata_pca.copy(), n_neighbors=15, n_pcs=30
        )
        neighbors_time = time.time() - neighbors_start
        data_manager.modalities["stress_50k_neighbors"] = adata_neighbors
        logger.info(f"Neighbors: {neighbors_time:.2f}s")

        # Step 6: Clustering
        cluster_start = time.time()
        adata_clustered, cluster_stats = services["clustering"].cluster(
            adata_neighbors.copy(), resolution=0.5
        )
        cluster_time = time.time() - cluster_start
        data_manager.modalities["stress_50k_clustered"] = adata_clustered
        logger.info(f"Clustering: {cluster_time:.2f}s")

        # Step 7: UMAP
        umap_start = time.time()
        adata_umap, umap_stats = services["clustering"].compute_umap(
            adata_clustered.copy()
        )
        umap_time = time.time() - umap_start
        data_manager.modalities["stress_50k_umap"] = adata_umap
        logger.info(f"UMAP: {umap_time:.2f}s")

        # Memory profiling
        peak_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        # Calculate total time
        total_time = (
            creation_time
            + qc_time
            + preproc_time
            + hvg_time
            + pca_time
            + neighbors_time
            + cluster_time
            + umap_time
        )

        # Report results
        peak_mb = peak_memory / (1024 * 1024)
        start_mb = start_memory / (1024 * 1024)
        used_mb = peak_mb - start_mb

        logger.info("\n" + "=" * 70)
        logger.info("50K CELL PIPELINE RESULTS")
        logger.info("=" * 70)
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Peak memory: {peak_mb:.2f} MB (used: {used_mb:.2f} MB)")
        logger.info(f"Cells/second: {50000 / total_time:.0f}")
        logger.info(f"Memory per cell: {used_mb / 50000 * 1000:.2f} KB")
        logger.info("\nStep breakdown:")
        logger.info(f"  Creation: {creation_time:.2f}s")
        logger.info(f"  QC: {qc_time:.2f}s")
        logger.info(f"  Preprocessing: {preproc_time:.2f}s")
        logger.info(f"  HVG: {hvg_time:.2f}s")
        logger.info(f"  PCA: {pca_time:.2f}s")
        logger.info(f"  Neighbors: {neighbors_time:.2f}s")
        logger.info(f"  Clustering: {cluster_time:.2f}s")
        logger.info(f"  UMAP: {umap_time:.2f}s")
        logger.info("=" * 70)

        # Assertions
        assert adata_umap.n_obs > 40000, "Too many cells filtered"
        assert "leiden" in adata_clustered.obs.columns
        assert "X_umap" in adata_umap.obsm

        # Performance benchmarks
        assert total_time < 600, f"Pipeline too slow: {total_time:.0f}s"
        assert used_mb < 10000, f"Memory usage too high: {used_mb:.0f} MB"

    def test_100k_cell_memory_profile(self, data_manager):
        """Test memory usage with 100K cells (creation and basic ops only)."""
        logger.info("\n" + "=" * 70)
        logger.info("STRESS TEST: 100K Cell Memory Profiling")
        logger.info("=" * 70)

        tracemalloc.start()
        gc.collect()
        start_memory = tracemalloc.get_traced_memory()[0]

        # Create 100K cell dataset
        creation_start = time.time()
        adata = create_large_singlecell_adata(
            n_obs=100000, n_vars=2000, use_sparse=True, name="stress_100k"
        )
        creation_time = time.time() - creation_start

        creation_memory = tracemalloc.get_traced_memory()[1]

        # Store in data manager
        data_manager.modalities["stress_100k"] = adata
        stored_memory = tracemalloc.get_traced_memory()[1]

        # Basic operations
        ops_start = time.time()

        # Operation 1: Access data
        _ = adata.X.shape
        _ = adata.obs.head()
        _ = adata.var.head()

        # Operation 2: Compute basic stats
        if sparse.issparse(adata.X):
            _ = adata.X.mean()
            _ = adata.X.std()

        # Operation 3: Filter genes
        adata_filtered = adata[:, :1000].copy()

        ops_time = time.time() - ops_start
        peak_memory = tracemalloc.get_traced_memory()[1]

        # Delete and check memory reclamation
        del adata
        del adata_filtered
        gc.collect()

        final_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        # Calculate metrics
        creation_mb = (creation_memory - start_memory) / (1024 * 1024)
        stored_mb = (stored_memory - start_memory) / (1024 * 1024)
        peak_mb = (peak_memory - start_memory) / (1024 * 1024)
        final_mb = (final_memory - start_memory) / (1024 * 1024)
        reclaimed_mb = peak_mb - final_mb
        reclaimed_pct = (reclaimed_mb / peak_mb) * 100 if peak_mb > 0 else 0

        logger.info("\n" + "=" * 70)
        logger.info("100K CELL MEMORY PROFILING RESULTS")
        logger.info("=" * 70)
        logger.info(f"Creation time: {creation_time:.2f}s")
        logger.info(f"Operations time: {ops_time:.2f}s")
        logger.info(f"Creation memory: {creation_mb:.2f} MB")
        logger.info(f"Stored memory: {stored_mb:.2f} MB")
        logger.info(f"Peak memory: {peak_mb:.2f} MB")
        logger.info(f"Final memory: {final_mb:.2f} MB")
        logger.info(f"Reclaimed: {reclaimed_mb:.2f} MB ({reclaimed_pct:.1f}%)")
        logger.info(f"Memory per cell: {peak_mb / 100000 * 1000:.2f} KB")
        logger.info("=" * 70)

        # Assertions
        assert creation_time < 60, f"Creation too slow: {creation_time:.0f}s"
        assert peak_mb < 5000, f"Memory usage too high: {peak_mb:.0f} MB"

        # Check for memory leak (should reclaim >50% of peak memory)
        if reclaimed_pct < 50:
            logger.warning(
                f"⚠️ POTENTIAL MEMORY LEAK: Only {reclaimed_pct:.1f}% "
                f"memory reclaimed after deletion"
            )


# ============================================================================
# Test 2: Large-Scale Proteomics
# ============================================================================


@pytest.mark.slow
class TestLargeScaleProteomics:
    """Test large-scale proteomics workflows."""

    def test_10k_sample_proteomics_workflow(self, data_manager):
        """Test proteomics workflow with 10K samples."""
        logger.info("\n" + "=" * 70)
        logger.info("STRESS TEST: 10K Sample Proteomics Workflow")
        logger.info("=" * 70)

        tracemalloc.start()

        # Create dataset
        creation_start = time.time()
        adata = create_large_proteomics_adata(
            n_obs=10000, n_vars=5000, missing_rate=0.3, name="stress_10k_prot"
        )
        creation_time = time.time() - creation_start

        data_manager.modalities["stress_10k_prot"] = adata
        logger.info(f"Dataset creation: {creation_time:.2f}s")

        # Basic statistics
        stats_start = time.time()

        # Calculate missing value statistics
        missing_per_sample = np.isnan(adata.X).sum(axis=1)
        missing_per_protein = np.isnan(adata.X).sum(axis=0)

        # Filter proteins with >70% missing
        proteins_to_keep = missing_per_protein < (0.7 * adata.n_obs)
        adata_filtered = adata[:, proteins_to_keep].copy()

        # Calculate correlations (on subset)
        subset_size = min(1000, adata_filtered.n_obs)
        subset_data = adata_filtered.X[:subset_size, :100]

        # Remove NaNs for correlation
        subset_data_clean = np.nan_to_num(subset_data, nan=0.0)
        correlations = np.corrcoef(subset_data_clean.T)

        stats_time = time.time() - stats_start

        peak_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        # Calculate metrics
        peak_mb = peak_memory / (1024 * 1024)
        total_time = creation_time + stats_time

        logger.info("\n" + "=" * 70)
        logger.info("10K PROTEOMICS WORKFLOW RESULTS")
        logger.info("=" * 70)
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Peak memory: {peak_mb:.2f} MB")
        logger.info(f"Samples/second: {10000 / total_time:.0f}")
        logger.info(f"Memory per sample: {peak_mb / 10000 * 1000:.2f} KB")
        logger.info(f"\nStep breakdown:")
        logger.info(f"  Creation: {creation_time:.2f}s")
        logger.info(f"  Statistics: {stats_time:.2f}s")
        logger.info(f"\nData quality:")
        logger.info(f"  Original proteins: {adata.n_vars}")
        logger.info(f"  After filtering: {adata_filtered.n_vars}")
        logger.info(f"  Removed: {adata.n_vars - adata_filtered.n_vars}")
        logger.info("=" * 70)

        # Assertions
        assert adata_filtered.n_vars > 1000, "Too many proteins filtered"
        assert total_time < 120, f"Workflow too slow: {total_time:.0f}s"
        assert peak_mb < 3000, f"Memory usage too high: {peak_mb:.0f} MB"


# ============================================================================
# Test 3: Memory Leak Detection
# ============================================================================


@pytest.mark.slow
class TestMemoryLeaks:
    """Deep memory leak detection and profiling."""

    def test_repeated_large_dataset_cycles(self, data_manager):
        """Test memory leaks with repeated create/delete cycles."""
        logger.info("\n" + "=" * 70)
        logger.info("STRESS TEST: Memory Leak Detection (Repeated Cycles)")
        logger.info("=" * 70)

        tracemalloc.start()
        gc.collect()

        memory_samples = []
        cycle_times = []

        # Run 20 cycles
        for cycle in range(20):
            cycle_start = time.time()

            # Create 10 large datasets
            for i in range(10):
                adata = create_large_singlecell_adata(
                    n_obs=5000, n_vars=1000, use_sparse=True, name=f"cycle{cycle}_ds{i}"
                )
                data_manager.modalities[f"cycle{cycle}_ds{i}"] = adata

            # Delete all datasets
            for i in range(10):
                del data_manager.modalities[f"cycle{cycle}_ds{i}"]

            gc.collect()

            # Sample memory
            current_memory = tracemalloc.get_traced_memory()[1]
            memory_samples.append(current_memory / (1024 * 1024))  # MB

            cycle_time = time.time() - cycle_start
            cycle_times.append(cycle_time)

        tracemalloc.stop()

        # Analyze trends
        first_5_avg = np.mean(memory_samples[:5])
        last_5_avg = np.mean(memory_samples[-5:])
        memory_increase_pct = ((last_5_avg - first_5_avg) / first_5_avg) * 100

        # Time trend
        first_5_time_avg = np.mean(cycle_times[:5])
        last_5_time_avg = np.mean(cycle_times[-5:])
        time_increase_pct = (
            (last_5_time_avg - first_5_time_avg) / first_5_time_avg
        ) * 100

        logger.info("\n" + "=" * 70)
        logger.info("MEMORY LEAK DETECTION RESULTS")
        logger.info("=" * 70)
        logger.info(f"Cycles completed: {len(memory_samples)}")
        logger.info(f"Memory samples (MB): {[f'{m:.1f}' for m in memory_samples]}")
        logger.info(f"\nMemory analysis:")
        logger.info(f"  First 5 cycles avg: {first_5_avg:.2f} MB")
        logger.info(f"  Last 5 cycles avg: {last_5_avg:.2f} MB")
        logger.info(f"  Increase: {memory_increase_pct:.1f}%")
        logger.info(f"\nTime analysis:")
        logger.info(f"  First 5 cycles avg: {first_5_time_avg:.2f}s")
        logger.info(f"  Last 5 cycles avg: {last_5_time_avg:.2f}s")
        logger.info(f"  Increase: {time_increase_pct:.1f}%")
        logger.info("=" * 70)

        # Check for memory leak
        if memory_increase_pct > 30:
            logger.error(
                f"⚠️ MEMORY LEAK DETECTED: {memory_increase_pct:.1f}% increase "
                f"from first to last 5 cycles"
            )

        # Check for performance degradation
        if time_increase_pct > 20:
            logger.warning(
                f"⚠️ PERFORMANCE DEGRADATION: {time_increase_pct:.1f}% slower "
                f"in last 5 cycles"
            )

        # Assertions (allow some increase but not excessive)
        assert (
            memory_increase_pct < 50
        ), f"Excessive memory increase: {memory_increase_pct:.1f}%"
        assert (
            time_increase_pct < 30
        ), f"Excessive performance degradation: {time_increase_pct:.1f}%"


# ============================================================================
# Test 4: Concurrent Operations at Scale
# ============================================================================


@pytest.mark.slow
class TestConcurrentScaling:
    """Test concurrent operations with large datasets."""

    def test_32_workers_concurrent_operations(self, data_manager):
        """Test 32 workers performing concurrent operations on large datasets."""
        logger.info("\n" + "=" * 70)
        logger.info("STRESS TEST: 32 Workers Concurrent Operations")
        logger.info("=" * 70)

        # Setup: Create base datasets
        n_base_datasets = 50
        logger.info(f"Creating {n_base_datasets} base datasets...")

        for i in range(n_base_datasets):
            adata = create_large_singlecell_adata(
                n_obs=1000, n_vars=500, use_sparse=True, name=f"base_{i}"
            )
            data_manager.modalities[f"base_{i}"] = adata

        # Concurrent operations
        n_workers = 32
        n_ops_per_worker = 200
        errors = []

        stats = {
            "reads": 0,
            "writes": 0,
            "copies": 0,
        }
        stats_lock = threading.Lock()

        def worker(worker_id: int):
            try:
                for i in range(n_ops_per_worker):
                    op_type = np.random.choice(
                        ["read", "write", "copy"], p=[0.6, 0.2, 0.2]
                    )

                    if op_type == "read":
                        # Read random dataset
                        dataset_id = np.random.randint(0, n_base_datasets)
                        adata = data_manager.get_modality(f"base_{dataset_id}")
                        assert adata.n_obs == 1000
                        with stats_lock:
                            stats["reads"] += 1

                    elif op_type == "write":
                        # Write new dataset
                        new_name = f"worker{worker_id}_temp{i}"
                        adata = create_large_singlecell_adata(
                            n_obs=500, n_vars=300, use_sparse=True, name=new_name
                        )
                        data_manager.modalities[new_name] = adata
                        with stats_lock:
                            stats["writes"] += 1

                    elif op_type == "copy":
                        # Copy and modify dataset
                        dataset_id = np.random.randint(0, n_base_datasets)
                        adata = data_manager.get_modality(f"base_{dataset_id}").copy()
                        # Simple modification
                        adata.obs["worker_id"] = worker_id
                        new_name = f"worker{worker_id}_copy{i}"
                        data_manager.modalities[new_name] = adata
                        with stats_lock:
                            stats["copies"] += 1

            except Exception as e:
                errors.append((worker_id, str(e)))

        # Run workers
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(worker, i) for i in range(n_workers)]
            concurrent.futures.wait(futures)

        duration = time.time() - start_time

        # Calculate metrics
        total_ops = sum(stats.values())
        ops_per_sec = total_ops / duration

        logger.info("\n" + "=" * 70)
        logger.info("32 WORKERS CONCURRENT OPERATIONS RESULTS")
        logger.info("=" * 70)
        logger.info(f"Duration: {duration:.2f}s")
        logger.info(f"Total operations: {total_ops}")
        logger.info(f"Operations/second: {ops_per_sec:.0f}")
        logger.info(f"\nOperation breakdown:")
        logger.info(f"  Reads: {stats['reads']}")
        logger.info(f"  Writes: {stats['writes']}")
        logger.info(f"  Copies: {stats['copies']}")
        logger.info(f"\nErrors: {len(errors)}")
        logger.info(f"Final modality count: {len(data_manager.list_modalities())}")
        logger.info("=" * 70)

        # Assertions
        assert len(errors) == 0, f"Concurrent operation errors: {errors}"
        assert ops_per_sec > 50, f"Performance too slow: {ops_per_sec:.0f} ops/sec"


# ============================================================================
# Test 5: Provenance Scaling
# ============================================================================


@pytest.mark.slow
class TestProvenanceScaling:
    """Test provenance tracking with 10K+ activities."""

    def test_10k_provenance_activities(self):
        """Test provenance tracker with 10K activities."""
        logger.info("\n" + "=" * 70)
        logger.info("STRESS TEST: 10K Provenance Activities")
        logger.info("=" * 70)

        tracker = ProvenanceTracker()

        n_activities = 10000
        start_time = time.time()

        for i in range(n_activities):
            tracker.create_activity(
                activity_type=f"stress_test_activity_{i}",
                agent="stress_test_agent",
                parameters={
                    "iteration": i,
                    "data_size": np.random.randint(1000, 100000),
                    "method": np.random.choice(["PCA", "UMAP", "clustering"]),
                },
                description=f"Stress test activity {i}",
            )

        duration = time.time() - start_time
        activities_per_sec = n_activities / duration

        # Test retrieval performance
        retrieval_start = time.time()
        activities = tracker.get_all_activities()
        retrieval_time = time.time() - retrieval_start

        # Test export performance
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = Path(temp_dir) / "provenance.json"
            export_start = time.time()
            tracker.export_to_json(str(export_path))
            export_time = time.time() - export_start

            # Check file size
            file_size_mb = export_path.stat().st_size / (1024 * 1024)

        logger.info("\n" + "=" * 70)
        logger.info("10K PROVENANCE ACTIVITIES RESULTS")
        logger.info("=" * 70)
        logger.info(f"Creation time: {duration:.2f}s")
        logger.info(f"Activities/second: {activities_per_sec:.0f}")
        logger.info(f"Retrieval time: {retrieval_time:.4f}s")
        logger.info(f"Export time: {export_time:.2f}s")
        logger.info(f"Export file size: {file_size_mb:.2f} MB")
        logger.info(f"Size per activity: {file_size_mb / n_activities * 1000:.2f} KB")
        logger.info("=" * 70)

        # Assertions
        assert len(activities) == n_activities
        assert (
            activities_per_sec > 1000
        ), f"Creation too slow: {activities_per_sec:.0f}/sec"
        assert retrieval_time < 1.0, f"Retrieval too slow: {retrieval_time:.2f}s"
        assert export_time < 10.0, f"Export too slow: {export_time:.2f}s"


# ============================================================================
# Test 6: File I/O Performance
# ============================================================================


@pytest.mark.slow
class TestFileIOPerformance:
    """Test file I/O performance with large datasets."""

    def test_large_h5ad_save_load_performance(self, temp_workspace):
        """Test save/load performance with large H5AD files."""
        logger.info("\n" + "=" * 70)
        logger.info("STRESS TEST: Large H5AD File I/O Performance")
        logger.info("=" * 70)

        # Create large dataset
        adata = create_large_singlecell_adata(
            n_obs=50000, n_vars=2000, use_sparse=True, name="io_test"
        )

        # Save performance
        save_path = temp_workspace / "large_test.h5ad"
        save_start = time.time()
        adata.write_h5ad(save_path)
        save_time = time.time() - save_start

        # File size
        file_size_mb = save_path.stat().st_size / (1024 * 1024)

        # Load performance
        load_start = time.time()
        adata_loaded = anndata.read_h5ad(save_path)
        load_time = time.time() - load_start

        # Calculate metrics
        save_mb_per_sec = file_size_mb / save_time
        load_mb_per_sec = file_size_mb / load_time

        logger.info("\n" + "=" * 70)
        logger.info("FILE I/O PERFORMANCE RESULTS")
        logger.info("=" * 70)
        logger.info(f"Dataset: 50K cells × 2K genes")
        logger.info(f"File size: {file_size_mb:.2f} MB")
        logger.info(f"\nSave performance:")
        logger.info(f"  Time: {save_time:.2f}s")
        logger.info(f"  Speed: {save_mb_per_sec:.2f} MB/s")
        logger.info(f"\nLoad performance:")
        logger.info(f"  Time: {load_time:.2f}s")
        logger.info(f"  Speed: {load_mb_per_sec:.2f} MB/s")
        logger.info("=" * 70)

        # Verify data integrity
        assert adata_loaded.n_obs == adata.n_obs
        assert adata_loaded.n_vars == adata.n_vars

        # Performance benchmarks
        assert save_time < 60, f"Save too slow: {save_time:.0f}s"
        assert load_time < 30, f"Load too slow: {load_time:.0f}s"
        assert save_mb_per_sec > 1, f"Write speed too slow: {save_mb_per_sec:.1f} MB/s"
        assert load_mb_per_sec > 5, f"Read speed too slow: {load_mb_per_sec:.1f} MB/s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--log-cli-level=INFO", "-m", "slow"])
