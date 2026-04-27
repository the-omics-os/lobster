"""
Production-scale validation for NEXUS Data Layer backend + LRU eviction.

Tests use REAL memory pressure, REAL threading, REAL scanpy pipelines.
Not toy data — production-representative 100K+ cell modalities.

Run with: pytest tests/integration/test_nexus_backend_prod_validation.py -v -s
Requires: ~4GB free RAM. Skip on CI with: pytest -m "not prod_validation"

Measures:
- Memory (RSS) before/after operations
- Lock contention under concurrent access
- LRU eviction reclamation effectiveness
- Backed mode materialize latency at scale
- End-to-end agent mutation survival through evict/reload cycle
"""

import gc
import os
import shutil
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import psutil
import pytest
from scipy import sparse as sp

from lobster.core.backends.h5ad_modality_backend import H5ADModalityBackend
from lobster.core.data_manager_v2 import DataManagerV2

pytestmark = [pytest.mark.prod_validation, pytest.mark.timeout(1800)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    """Current process RSS in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _generate_adata(n_obs: int, n_vars: int, density: float = 0.1) -> ad.AnnData:
    """Generate sparse AnnData at specified scale.

    Uses row-by-row lil_matrix construction to avoid scipy.sparse.random's
    O(n*m) rng.choice which times out at 100K×20K.
    """
    from scipy.sparse import lil_matrix
    nnz_per_row = int(n_vars * density)
    X = lil_matrix((n_obs, n_vars), dtype=np.float32)
    rng = np.random.default_rng(42)
    for i in range(n_obs):
        cols = rng.choice(n_vars, size=nnz_per_row, replace=False)
        X[i, cols] = rng.random(nnz_per_row, dtype=np.float32)
    X = X.tocsr()
    obs = pd.DataFrame(
        {
            "batch": np.random.choice(["A", "B", "C"], n_obs),
            "n_counts": np.random.rand(n_obs).astype(np.float32),
            "cell_type": np.random.choice(
                ["T cell", "B cell", "Monocyte", "NK", "DC"], n_obs
            ),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_symbol": [f"gene_{i}" for i in range(n_vars)]},
        index=[f"ENSG{i:05d}" for i in range(n_vars)],
    )
    adata = ad.AnnData(X=X, obs=obs, var=var)
    return adata


def _write_h5ad(adata: ad.AnnData, path: Path) -> int:
    """Write h5ad, return file size in bytes."""
    adata.write_h5ad(path)
    return path.stat().st_size


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def prod_workspace():
    """Temp workspace with production-scale h5ad files."""
    tmpdir = Path(tempfile.mkdtemp(prefix="nexus_prod_"))
    data_dir = tmpdir / "data"
    data_dir.mkdir()

    manifest: Dict[str, dict] = {}

    # Medium modality: 100K cells x 20K genes (~200MB sparse)
    print("\n[fixture] Generating 100K x 20K modality...")
    t0 = time.time()
    adata_100k = _generate_adata(100_000, 20_000, density=0.05)
    size = _write_h5ad(adata_100k, data_dir / "scrna_100k.h5ad")
    manifest["scrna_100k"] = {
        "n_obs": 100_000, "n_vars": 20_000,
        "size_mb": size / (1024 * 1024), "gen_time": time.time() - t0,
    }
    del adata_100k
    gc.collect()
    print(f"  -> {manifest['scrna_100k']['size_mb']:.0f} MB in {manifest['scrna_100k']['gen_time']:.1f}s")

    # Small modalities for LRU pressure (5 x 30K cells)
    for i in range(5):
        name = f"batch_{i}"
        print(f"[fixture] Generating {name} (30K x 10K)...")
        t0 = time.time()
        adata = _generate_adata(30_000, 10_000, density=0.05)
        size = _write_h5ad(adata, data_dir / f"{name}.h5ad")
        manifest[name] = {
            "n_obs": 30_000, "n_vars": 10_000,
            "size_mb": size / (1024 * 1024), "gen_time": time.time() - t0,
        }
        del adata
        gc.collect()
        print(f"  -> {manifest[name]['size_mb']:.0f} MB in {manifest[name]['gen_time']:.1f}s")

    print(f"\n[fixture] Total files: {len(manifest)}")
    print(f"[fixture] Total disk: {sum(m['size_mb'] for m in manifest.values()):.0f} MB")

    yield tmpdir, data_dir, manifest

    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def dm_with_backend(prod_workspace):
    """DataManagerV2 wired to H5ADModalityBackend over prod data."""
    tmpdir, data_dir, manifest = prod_workspace
    dm = DataManagerV2(workspace_path=tmpdir)
    backend = H5ADModalityBackend(data_dir=data_dir)
    dm.set_modality_backend(backend, cache_cap=3)
    return dm, backend, manifest


# ---------------------------------------------------------------------------
# Test 1: Production-scale list + metadata (no load)
# ---------------------------------------------------------------------------

class TestProductionMetadata:
    """Backend metadata ops must be fast even with large files."""

    def test_list_modalities_returns_all(self, dm_with_backend):
        dm, backend, manifest = dm_with_backend
        names = dm.list_modalities()
        assert set(names) == set(manifest.keys())

    def test_list_modality_records_has_correct_shapes(self, dm_with_backend):
        dm, backend, manifest = dm_with_backend
        records = dm.list_modality_records()
        for rec in records:
            expected = manifest[rec["name"]]
            assert rec["n_obs"] == expected["n_obs"]
            assert rec["n_vars"] == expected["n_vars"]
            assert rec["data_status"] == "cold"

    def test_metadata_peek_under_100ms(self, dm_with_backend):
        """h5py header reads must be fast regardless of file size."""
        dm, backend, manifest = dm_with_backend
        t0 = time.time()
        for name in manifest:
            backend.get_metadata(name)
        elapsed = time.time() - t0
        per_file = elapsed / len(manifest) * 1000
        print(f"\n  Metadata peek: {per_file:.1f} ms/file ({len(manifest)} files)")
        assert per_file < 100, f"Metadata peek too slow: {per_file:.1f}ms"


# ---------------------------------------------------------------------------
# Test 2: LRU eviction with real memory reclamation
# ---------------------------------------------------------------------------

class TestLRUMemoryReclamation:
    """Verify LRU eviction actually frees memory (RSS), not just dict slots."""

    def test_lru_eviction_reclaims_rss(self, dm_with_backend):
        """Load modalities beyond cache cap, verify RSS doesn't grow unbounded."""
        dm, backend, manifest = dm_with_backend
        gc.collect()
        baseline_mb = _rss_mb()
        print(f"\n  Baseline RSS: {baseline_mb:.0f} MB")

        peak_mb = baseline_mb
        for name in ["batch_0", "batch_1", "batch_2", "batch_3", "batch_4"]:
            dm.get_modality(name)
            gc.collect()
            current = _rss_mb()
            peak_mb = max(peak_mb, current)
            cached = len(dm.modalities)
            print(f"  After {name}: RSS={current:.0f} MB, cached={cached}")

        gc.collect()
        final_mb = _rss_mb()
        cached_count = len(dm.modalities)
        print(f"  Final: RSS={final_mb:.0f} MB, cached={cached_count}")
        print(f"  Peak: {peak_mb:.0f} MB, Growth: {final_mb - baseline_mb:.0f} MB")

        assert cached_count <= dm._modality_cache_cap
        # RSS should stabilize, not grow linearly with # loaded
        # Allow 2x single-modality overhead (cache_cap modalities in memory)
        single_mod_mb = manifest["batch_0"]["size_mb"] * 3  # sparse→dense expansion
        max_expected = baseline_mb + (dm._modality_cache_cap + 1) * single_mod_mb
        assert final_mb < max_expected, (
            f"RSS {final_mb:.0f} MB exceeds expected cap {max_expected:.0f} MB"
        )

    def test_evicted_modality_visible_in_list(self, dm_with_backend):
        """Evicted modalities must still appear in list_modalities()."""
        dm, backend, manifest = dm_with_backend
        # Load 4 to evict first
        dm.get_modality("batch_0")
        dm.get_modality("batch_1")
        dm.get_modality("batch_2")
        dm.get_modality("batch_3")  # evicts batch_0

        assert "batch_0" not in dm.modalities
        names = dm.list_modalities()
        assert "batch_0" in names  # still visible via backend

        records = dm.list_modality_records()
        cold = [r for r in records if r["name"] == "batch_0"]
        assert len(cold) == 1
        assert cold[0]["data_status"] == "cold"


# ---------------------------------------------------------------------------
# Test 3: Lock contention under concurrent access
# ---------------------------------------------------------------------------

class TestConcurrentAccess:
    """Measure _state_lock contention with real threading."""

    def test_concurrent_get_modality_no_deadlock(self, dm_with_backend):
        """Multiple threads hitting get_modality() simultaneously."""
        dm, backend, manifest = dm_with_backend
        errors: List[str] = []
        timings: Dict[str, float] = {}
        barrier = threading.Barrier(4)

        def worker(name: str):
            try:
                barrier.wait(timeout=10)
                t0 = time.time()
                adata = dm.get_modality(name)
                elapsed = time.time() - t0
                timings[name] = elapsed
                assert adata.n_obs > 0
            except Exception as e:
                errors.append(f"{name}: {e}")

        threads = []
        names = ["batch_0", "batch_1", "batch_2", "batch_3"]
        for n in names:
            t = threading.Thread(target=worker, args=(n,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=120)

        assert not errors, f"Thread errors: {errors}"
        print(f"\n  Concurrent get_modality timings:")
        for name, elapsed in sorted(timings.items()):
            print(f"    {name}: {elapsed:.2f}s")

    def test_concurrent_read_write_no_corruption(self, dm_with_backend):
        """One writer (ensure_in_memory + mutate), multiple readers."""
        dm, backend, manifest = dm_with_backend
        errors: List[str] = []
        barrier = threading.Barrier(3)

        def writer():
            try:
                barrier.wait(timeout=10)
                adata = dm.ensure_in_memory("batch_0")
                adata.obs["writer_flag"] = "written"
            except Exception as e:
                errors.append(f"writer: {e}")

        def reader(name: str):
            try:
                barrier.wait(timeout=10)
                adata = dm.get_modality(name)
                _ = adata.n_obs  # access shape
            except Exception as e:
                errors.append(f"reader-{name}: {e}")

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader, args=("batch_1",)),
            threading.Thread(target=reader, args=("batch_2",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        assert not errors, f"Concurrent errors: {errors}"


# ---------------------------------------------------------------------------
# Test 4: Full agent simulation — mutation survival through evict/reload
# ---------------------------------------------------------------------------

class TestAgentMutationSurvival:
    """Simulate real agent workflow: load → analyze → mutate → evict → reload."""

    def test_scanpy_pipeline_survives_eviction(self, dm_with_backend):
        """Full scanpy pipeline results must survive LRU eviction + backend reload."""
        dm, backend, manifest = dm_with_backend

        # Step 1: Load and run scanpy pipeline
        adata = dm.ensure_in_memory("scrna_100k")
        print(f"\n  Loaded scrna_100k: {adata.n_obs} x {adata.n_vars}")

        import scanpy as sc
        t0 = time.time()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        sc.pp.pca(adata, n_comps=20)
        pipeline_time = time.time() - t0
        print(f"  Pipeline (norm→log1p→hvg→pca): {pipeline_time:.1f}s")

        # Record state to verify after reload
        pca_shape = adata.obsm["X_pca"].shape
        n_hvg = int((adata.var["highly_variable"] == True).sum())
        adata.obs["agent_annotation"] = "processed"

        # Step 2: Force eviction by loading other modalities
        dm.modalities["scrna_100k"] = adata  # update cache
        dm._modality_dirty.add("scrna_100k")

        print("  Forcing eviction...")
        for i in range(dm._modality_cache_cap + 1):
            name = f"batch_{i}"
            if name in manifest:
                dm.get_modality(name)

        assert "scrna_100k" not in dm.modalities, "Should be evicted"

        # Step 3: Reload from backend
        print("  Reloading from backend...")
        t0 = time.time()
        reloaded = dm.get_modality("scrna_100k")
        reload_time = time.time() - t0
        print(f"  Reload time: {reload_time:.1f}s")

        # Step 4: Verify everything survived
        assert reloaded.n_obs == 100_000
        assert reloaded.obsm["X_pca"].shape == pca_shape, "PCA lost"
        hvg_after = reloaded.var["highly_variable"]
        if hasattr(hvg_after, "cat"):
            n_hvg_after = int((hvg_after.astype(str) == "True").sum())
        else:
            n_hvg_after = int((hvg_after == True).sum())
        assert n_hvg_after == n_hvg, f"HVG lost: {n_hvg_after} vs {n_hvg}"
        assert (reloaded.obs["agent_annotation"] == "processed").all(), "Annotation lost"
        print("  ✓ All pipeline results + mutations survived eviction")

    def test_obs_mutation_dirty_tracking(self, dm_with_backend):
        """Agent .obs mutations via ensure_in_memory are auto-dirty and survive."""
        dm, backend, manifest = dm_with_backend

        adata = dm.ensure_in_memory("batch_0")
        original_obs_cols = set(adata.obs.columns)
        adata.obs["cell_score"] = np.random.rand(adata.n_obs).astype(np.float32)

        assert "batch_0" in dm._modality_dirty

        # Force eviction
        for i in range(1, dm._modality_cache_cap + 2):
            name = f"batch_{min(i, 4)}"
            if name in manifest:
                dm.get_modality(name)

        if "batch_0" not in dm.modalities:
            reloaded = dm.get_modality("batch_0")
            assert "cell_score" in reloaded.obs.columns, "Mutation lost after eviction"
            print(f"\n  ✓ obs mutation survived eviction (dirty flush worked)")


# ---------------------------------------------------------------------------
# Test 5: Capacity projection with real numbers
# ---------------------------------------------------------------------------

class TestCapacityProjection:
    """Measure actual memory per modality at production scale."""

    def test_memory_per_modality_measurement(self, dm_with_backend):
        """Measure real MB per cached modality for capacity planning."""
        dm, backend, manifest = dm_with_backend
        gc.collect()
        baseline = _rss_mb()

        measurements: List[Tuple[str, float, int, int]] = []

        for name in ["batch_0", "scrna_100k"]:
            gc.collect()
            before = _rss_mb()
            adata = dm.get_modality(name)
            gc.collect()
            after = _rss_mb()
            delta = after - before
            measurements.append((name, delta, manifest[name]["n_obs"], manifest[name]["n_vars"]))

        print(f"\n  Memory per modality (production scale):")
        print(f"  {'Name':<15} {'Cells':>8} {'Genes':>8} {'RSS Δ MB':>10} {'MB/10K cells':>14}")
        for name, delta_mb, n_obs, n_vars in measurements:
            per_10k = delta_mb / (n_obs / 10_000) if n_obs > 0 else 0
            print(f"  {name:<15} {n_obs:>8,} {n_vars:>8,} {delta_mb:>10.1f} {per_10k:>14.1f}")

        # Capacity projection for 120GB ECS
        usable_gb = 120 * 0.75  # 25% for OS + Python + overhead
        print(f"\n  Capacity projection (120GB ECS, 75% usable = {usable_gb:.0f} GB):")
        for name, delta_mb, n_obs, n_vars in measurements:
            if delta_mb > 0:
                sessions = (usable_gb * 1024) / delta_mb
                print(f"  {name}: {sessions:.0f} concurrent sessions (1 modality each)")


# ---------------------------------------------------------------------------
# Test 6: Graceful degradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    """Verify system fails cleanly, not catastrophically."""

    def test_remove_and_relist(self, dm_with_backend):
        """Remove modality, verify it disappears from list."""
        dm, backend, manifest = dm_with_backend
        dm.remove_modality("batch_4")
        names = dm.list_modalities()
        assert "batch_4" not in names
        assert not backend.exists("batch_4")

    def test_get_removed_raises_clean(self, dm_with_backend):
        """Accessing removed modality gives clean ValueError, no crash."""
        dm, backend, manifest = dm_with_backend
        if backend.exists("batch_4"):
            dm.remove_modality("batch_4")
        with pytest.raises(ValueError, match="not found"):
            dm.get_modality("batch_4")

    def test_backend_metadata_after_removal(self, dm_with_backend):
        """list_modality_records doesn't crash when files were externally deleted."""
        dm, backend, manifest = dm_with_backend
        records = dm.list_modality_records()
        assert isinstance(records, list)
        for rec in records:
            assert "name" in rec
            assert "data_status" in rec
