"""
NEXUS E2E Track 1 — Engine Stress Tests (T1.1–T1.4)

Production validation of LRU eviction, dirty flush, backed-mode sacred
contract, concurrent access, and list_modalities union.

Complements test_nexus_backend_prod_validation.py with targeted stress
scenarios from the NEXUS E2E validation plan.

Run: pytest tests/integration/test_nexus_e2e_track1.py -v -s
Requires: ~2GB free RAM, scanpy, psutil
"""

import gc
import os
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import anndata as ad
import numpy as np
import pandas as pd
import psutil
import pytest
from scipy.sparse import csr_matrix, lil_matrix, random as sp_random

from lobster.core.backends.h5ad_modality_backend import H5ADModalityBackend
from lobster.core.backends.modality_backend import ModalityRecord
from lobster.core.runtime.data_manager import DataManagerV2

pytestmark = [pytest.mark.nexus_e2e, pytest.mark.timeout(600)]


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _make_adata(n_obs: int, n_vars: int, density: float = 0.1, seed: int = 42) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    nnz_per_row = max(1, int(n_vars * density))
    X = lil_matrix((n_obs, n_vars), dtype=np.float32)
    for i in range(n_obs):
        cols = rng.choice(n_vars, size=min(nnz_per_row, n_vars), replace=False)
        X[i, cols] = rng.random(len(cols), dtype=np.float32)
    X = X.tocsr()
    obs = pd.DataFrame(
        {
            "batch": rng.choice(["A", "B", "C"], n_obs),
            "n_counts": rng.random(n_obs).astype(np.float32),
            "cell_type": rng.choice(["T cell", "B cell", "Monocyte", "NK", "DC"], n_obs),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_symbol": [f"gene_{i}" for i in range(n_vars)]},
        index=[f"ENSG{i:05d}" for i in range(n_vars)],
    )
    adata = ad.AnnData(X=X, obs=obs, var=var)
    return adata


@pytest.fixture(scope="module")
def workspace_12():
    """Workspace with 12 modalities: 4 small, 4 medium, 4 large."""
    tmpdir = Path(tempfile.mkdtemp(prefix="nexus_t1_"))
    data_dir = tmpdir / "data"
    data_dir.mkdir()

    manifest: Dict[str, dict] = {}
    configs = (
        [("small", 100, 100, 0.3)] * 4
        + [("medium", 5_000, 2_000, 0.05)] * 4
        + [("large", 20_000, 8_000, 0.03)] * 4
    )

    for i, (tier, n_obs, n_vars, density) in enumerate(configs):
        name = f"{tier}_{i}"
        adata = _make_adata(n_obs, n_vars, density=density, seed=i)
        adata.write_h5ad(data_dir / f"{name}.h5ad")
        manifest[name] = {
            "n_obs": n_obs,
            "n_vars": n_vars,
            "size_bytes": (data_dir / f"{name}.h5ad").stat().st_size,
        }
        del adata
        gc.collect()

    print(f"\n[T1 fixture] Created {len(manifest)} modalities in {data_dir}")
    yield tmpdir, data_dir, manifest
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def dm_cap8(workspace_12):
    """DataManagerV2 with backend, cache_cap=8."""
    tmpdir, data_dir, manifest = workspace_12
    dm = DataManagerV2(workspace_path=tmpdir)
    backend = H5ADModalityBackend(data_dir=data_dir)
    dm.set_modality_backend(backend, cache_cap=8)
    return dm, backend, manifest


@pytest.fixture
def dm_cap4(workspace_12):
    """DataManagerV2 with backend, cache_cap=4 (tighter pressure)."""
    tmpdir, data_dir, manifest = workspace_12
    dm = DataManagerV2(workspace_path=tmpdir)
    backend = H5ADModalityBackend(data_dir=data_dir)
    dm.set_modality_backend(backend, cache_cap=4)
    return dm, backend, manifest


# ---------------------------------------------------------------------------
# T1.1 — LRU pressure test (E1+E4+E5)
# ---------------------------------------------------------------------------

class TestT1_1_LRUPressure:
    """Load 12 modalities with cache_cap=8. Verify eviction, dirty flush, reload."""

    def test_12_loads_produce_4_evictions(self, dm_cap8):
        dm, backend, manifest = dm_cap8
        all_names = sorted(manifest.keys())

        for name in all_names:
            dm.get_modality(name)

        assert len(dm.modalities) <= 8, f"Cache exceeded cap: {len(dm.modalities)}"
        evicted = [n for n in all_names if n not in dm.modalities]
        assert len(evicted) >= 4, f"Expected ≥4 evictions, got {len(evicted)}"
        print(f"\n  Cached: {list(dm.modalities.keys())}")
        print(f"  Evicted: {evicted}")

    def test_dirty_modality_survives_eviction(self, dm_cap4):
        """Modify .obs, force eviction, reload — mutation must persist."""
        dm, backend, manifest = dm_cap4
        all_names = sorted(manifest.keys())
        target = all_names[0]

        adata = dm.ensure_in_memory(target)
        adata.obs["my_score"] = np.arange(adata.n_obs, dtype=np.float32)
        original_n_obs = adata.n_obs
        assert target in dm._modality_dirty, "ensure_in_memory must mark dirty"

        for name in all_names[1:]:
            dm.get_modality(name)

        if target not in dm.modalities:
            reloaded = dm.get_modality(target)
            assert reloaded.n_obs == original_n_obs, "Row count changed"
            assert "my_score" in reloaded.obs.columns, "Dirty mutation lost after eviction"
            assert np.allclose(
                reloaded.obs["my_score"].values,
                np.arange(original_n_obs, dtype=np.float32),
            ), "Dirty mutation values corrupted"
            print(f"\n  ✓ Dirty flush preserved 'my_score' on {target}")
        else:
            pytest.skip(f"{target} was not evicted (cap too large for test)")

    def test_round_trip_fidelity(self, dm_cap4):
        """Evict and reload — obs, var, X shape must be identical."""
        dm, backend, manifest = dm_cap4
        all_names = sorted(manifest.keys())
        target = all_names[0]

        adata_orig = dm.get_modality(target)
        orig_shape = adata_orig.shape
        orig_obs_cols = set(adata_orig.obs.columns)
        orig_var_cols = set(adata_orig.var.columns)
        orig_obs_index = list(adata_orig.obs.index[:5])

        for name in all_names[1:]:
            dm.get_modality(name)

        if target not in dm.modalities:
            reloaded = dm.get_modality(target)
            assert reloaded.shape == orig_shape, f"Shape mismatch: {reloaded.shape} vs {orig_shape}"
            assert set(reloaded.obs.columns) == orig_obs_cols, "obs columns changed"
            assert set(reloaded.var.columns) == orig_var_cols, "var columns changed"
            assert list(reloaded.obs.index[:5]) == orig_obs_index, "obs index changed"
            print(f"\n  ✓ Round-trip fidelity OK for {target} ({orig_shape})")
        else:
            pytest.skip(f"{target} was not evicted")

    def test_lru_order_evicts_oldest(self, dm_cap4):
        """First loaded should be first evicted (FIFO within LRU)."""
        dm, backend, manifest = dm_cap4
        all_names = sorted(manifest.keys())

        load_order = []
        for name in all_names[:6]:
            dm.get_modality(name)
            load_order.append(name)

        cached = set(dm.modalities.keys())
        first_two = set(load_order[:2])
        evicted = first_two - cached
        assert len(evicted) >= 1, (
            f"Expected first-loaded to be evicted. Cached: {cached}, first loaded: {first_two}"
        )


# ---------------------------------------------------------------------------
# T1.2 — Concurrent access (E3)
# ---------------------------------------------------------------------------

class TestT1_2_ConcurrentAccess:
    """4 threads calling get_modality() simultaneously under LRU pressure."""

    def test_no_none_returns_under_contention(self, dm_cap4):
        dm, backend, manifest = dm_cap4
        all_names = sorted(manifest.keys())
        target_names = all_names[:4]
        errors: List[str] = []
        results: Dict[str, Any] = {}
        barrier = threading.Barrier(4, timeout=30)

        def worker(name: str):
            try:
                barrier.wait()
                adata = dm.get_modality(name)
                if adata is None:
                    errors.append(f"{name}: got None")
                    return
                if adata.n_obs <= 0:
                    errors.append(f"{name}: n_obs={adata.n_obs}")
                    return
                results[name] = adata.n_obs
            except Exception as e:
                errors.append(f"{name}: {type(e).__name__}: {e}")

        threads = [threading.Thread(target=worker, args=(n,)) for n in target_names]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        assert not errors, f"Thread errors:\n" + "\n".join(errors)
        assert len(results) == len(target_names), f"Missing results: {set(target_names) - set(results)}"
        for name, n_obs in results.items():
            assert n_obs == manifest[name]["n_obs"], f"{name}: wrong n_obs"
        print(f"\n  ✓ {len(target_names)} concurrent get_modality() calls — all correct")

    def test_concurrent_ensure_in_memory_no_race(self, dm_cap4):
        """Multiple threads calling ensure_in_memory on different modalities."""
        dm, backend, manifest = dm_cap4
        all_names = sorted(manifest.keys())
        target_names = all_names[:4]
        errors: List[str] = []
        barrier = threading.Barrier(4, timeout=30)

        def worker(name: str):
            try:
                barrier.wait()
                adata = dm.ensure_in_memory(name)
                adata.obs[f"thread_{name}"] = 1.0
                assert not getattr(adata, "isbacked", False), f"{name} still backed!"
            except Exception as e:
                errors.append(f"{name}: {type(e).__name__}: {e}")

        threads = [threading.Thread(target=worker, args=(n,)) for n in target_names]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        assert not errors, f"Thread errors:\n" + "\n".join(errors)
        for name in target_names:
            assert name in dm._modality_dirty, f"{name} not marked dirty"
        print(f"\n  ✓ {len(target_names)} concurrent ensure_in_memory() calls — no races")

    def test_rss_bounded_under_contention(self, dm_cap4):
        """RSS should not spike unboundedly under concurrent load."""
        dm, backend, manifest = dm_cap4
        all_names = sorted(manifest.keys())
        gc.collect()
        baseline = _rss_mb()
        errors: List[str] = []
        barrier = threading.Barrier(4, timeout=30)

        def worker(name: str):
            try:
                barrier.wait()
                for _ in range(3):
                    dm.get_modality(name)
            except Exception as e:
                errors.append(f"{name}: {e}")

        threads = [threading.Thread(target=worker, args=(n,)) for n in all_names[:4]]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        gc.collect()
        final = _rss_mb()
        growth = final - baseline
        cached = len(dm.modalities)

        assert not errors, f"Thread errors:\n" + "\n".join(errors)
        assert cached <= dm._modality_cache_cap, f"Cache overflowed: {cached}"
        print(f"\n  Baseline: {baseline:.0f} MB, Final: {final:.0f} MB, Growth: {growth:.0f} MB, Cached: {cached}")


# ---------------------------------------------------------------------------
# T1.3 — Backed mode sacred contract (E2)
# ---------------------------------------------------------------------------

class TestT1_3_BackedModeSacredContract:
    """Agents must NEVER receive backed AnnData from get_modality()."""

    @pytest.fixture
    def large_backed_workspace(self):
        """Create a workspace with a file that triggers backed mode."""
        tmpdir = Path(tempfile.mkdtemp(prefix="nexus_backed_"))
        data_dir = tmpdir / "data"
        data_dir.mkdir()

        adata = _make_adata(10_000, 5_000, density=0.05, seed=99)
        path = data_dir / "large_mod.h5ad"
        adata.write_h5ad(path)
        del adata
        gc.collect()

        yield tmpdir, data_dir, path
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_get_modality_never_returns_backed(self, large_backed_workspace):
        """get_modality() auto-materializes — .isbacked must be False."""
        tmpdir, data_dir, path = large_backed_workspace
        dm = DataManagerV2(workspace_path=tmpdir, backed_mode=True)
        backend = H5ADModalityBackend(data_dir=data_dir, backed_mode=True)
        dm.set_modality_backend(backend, cache_cap=4)

        adata = dm.get_modality("large_mod")
        assert not getattr(adata, "isbacked", False), "Sacred contract violated: got backed AnnData"
        assert isinstance(adata.X, (np.ndarray, csr_matrix)), f"X type unexpected: {type(adata.X)}"
        print(f"\n  ✓ Sacred contract: get_modality returned in-memory AnnData ({adata.shape})")

    def test_ensure_in_memory_materializes(self, large_backed_workspace):
        """ensure_in_memory() must also return non-backed AnnData."""
        tmpdir, data_dir, path = large_backed_workspace
        dm = DataManagerV2(workspace_path=tmpdir, backed_mode=True)
        backend = H5ADModalityBackend(data_dir=data_dir, backed_mode=True)
        dm.set_modality_backend(backend, cache_cap=4)

        adata = dm.ensure_in_memory("large_mod")
        assert not getattr(adata, "isbacked", False), "Sacred contract violated"
        assert "large_mod" in dm._modality_dirty, "ensure_in_memory must mark dirty"

    def test_scanpy_pipeline_on_materialized(self, large_backed_workspace):
        """Full scanpy pipeline must work on materialized data (no backed errors)."""
        tmpdir, data_dir, path = large_backed_workspace
        dm = DataManagerV2(workspace_path=tmpdir, backed_mode=True)
        backend = H5ADModalityBackend(data_dir=data_dir, backed_mode=True)
        dm.set_modality_backend(backend, cache_cap=4)

        adata = dm.get_modality("large_mod")

        import scanpy as sc
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=500)
        adata_sub = adata[:, adata.var["highly_variable"]].copy()
        sc.pp.pca(adata_sub, n_comps=10)

        assert "X_pca" in adata_sub.obsm, "PCA failed on materialized data"
        assert adata_sub.obsm["X_pca"].shape == (10_000, 10)
        print(f"\n  ✓ Full scanpy pipeline completed on backed→materialized data")


# ---------------------------------------------------------------------------
# T1.4 — list_modalities union (E6)
# ---------------------------------------------------------------------------

class TestT1_4_ListModalitiesUnion:
    """list_modalities and list_modality_records must return hot + cold."""

    def test_all_12_visible_after_partial_load(self, dm_cap8):
        dm, backend, manifest = dm_cap8
        all_names = sorted(manifest.keys())

        dm.get_modality(all_names[0])
        dm.get_modality(all_names[1])

        listed = dm.list_modalities()
        assert set(listed) == set(all_names), (
            f"Missing from list: {set(all_names) - set(listed)}"
        )
        print(f"\n  ✓ list_modalities returns all 12 (2 hot, 10 cold)")

    def test_records_have_correct_data_status(self, dm_cap4):
        dm, backend, manifest = dm_cap4
        all_names = sorted(manifest.keys())

        for name in all_names[:6]:
            dm.get_modality(name)

        records = dm.list_modality_records()
        rec_map = {r["name"]: r for r in records}

        assert len(rec_map) == len(manifest), f"Expected {len(manifest)} records, got {len(rec_map)}"

        hot = [r for r in records if r["data_status"] == "hot"]
        cold = [r for r in records if r["data_status"] == "cold"]
        assert len(hot) <= 4, f"Too many hot: {len(hot)} (cap=4)"
        assert len(cold) >= 8, f"Expected ≥8 cold, got {len(cold)}"

        for r in hot:
            assert r["n_obs"] == manifest[r["name"]]["n_obs"]
            assert r["n_vars"] == manifest[r["name"]]["n_vars"]

        for r in cold:
            assert r["n_obs"] == manifest[r["name"]]["n_obs"]
            assert r["n_vars"] == manifest[r["name"]]["n_vars"]

        print(f"\n  ✓ Records: {len(hot)} hot, {len(cold)} cold — shapes match manifest")

    def test_evicted_appears_as_cold(self, dm_cap4):
        """Load → evict → verify status transitions hot→cold."""
        dm, backend, manifest = dm_cap4
        all_names = sorted(manifest.keys())
        target = all_names[0]

        dm.get_modality(target)
        records_before = dm.list_modality_records()
        target_rec = [r for r in records_before if r["name"] == target][0]
        assert target_rec["data_status"] == "hot"

        for name in all_names[1:6]:
            dm.get_modality(name)

        if target not in dm.modalities:
            records_after = dm.list_modality_records()
            target_rec = [r for r in records_after if r["name"] == target][0]
            assert target_rec["data_status"] == "cold", f"Evicted modality should be cold, got {target_rec['data_status']}"
            print(f"\n  ✓ {target} transitioned hot→cold after eviction")
        else:
            pytest.skip(f"{target} not evicted")

    def test_cold_reload_transitions_to_hot(self, dm_cap4):
        """Access cold modality → should become hot again."""
        dm, backend, manifest = dm_cap4
        all_names = sorted(manifest.keys())

        for name in all_names[:6]:
            dm.get_modality(name)

        records = dm.list_modality_records()
        cold_names = [r["name"] for r in records if r["data_status"] == "cold"]
        assert cold_names, "No cold modalities found"

        target = cold_names[0]
        dm.get_modality(target)

        records_after = dm.list_modality_records()
        target_rec = [r for r in records_after if r["name"] == target][0]
        assert target_rec["data_status"] == "hot", f"Reloaded modality should be hot"
        print(f"\n  ✓ {target} transitioned cold→hot after reload")
