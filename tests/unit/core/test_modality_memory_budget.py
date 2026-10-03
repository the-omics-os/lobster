"""Regression tests for bounded modality memory and spill/reload behavior.

Each pipeline operation can register a new named modality derived from its parent.
A byte budget keeps accumulated resident memory bounded, while evicted modalities
remain retrievable from disk. A count cap alone cannot bound differently sized
modalities.
"""

import os
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from lobster.core.runtime.data_manager import (
    DataManagerV2,
    estimate_anndata_bytes,
)

# 8 MiB per modality: 128 obs x 8192 vars x 8 bytes.
# float64 is declared explicitly rather than relying on float32 surviving
# AnnData construction — it does not, and a test whose expected size depends on
# AnnData's dtype policy is a test that breaks on an upgrade for no reason.
N_OBS = 128
N_VARS = 8192
ITEMSIZE = 8
BYTES_PER_MODALITY = N_OBS * N_VARS * ITEMSIZE
N_DERIVED = 6  # synthetic retained-modality count for the memory-budget test


def _shape_label(adata: "ad.AnnData") -> str:
    """Render 'N GiB on N obs x N vars' — memory is meaningless without shape."""
    size = estimate_anndata_bytes(adata)
    return f"{size / 1024**2:.2f} MiB on {adata.n_obs:,} obs x {adata.n_vars:,} vars"


def _make_modality(seed: int) -> "ad.AnnData":
    """A deterministic, dense modality of exactly BYTES_PER_MODALITY bytes."""
    rng = np.random.default_rng(seed)
    X = rng.random((N_OBS, N_VARS), dtype=np.float64)
    adata = ad.AnnData(X=X)
    assert adata.X.dtype.itemsize == ITEMSIZE
    return adata


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def budgeted_dm(workspace, monkeypatch):
    """DataManager with a budget that fits 3 of the 6 derived modalities."""
    budget_bytes = 3 * BYTES_PER_MODALITY
    monkeypatch.setenv("LOBSTER_MODALITY_MEMORY_BUDGET_GB", str(budget_bytes / 1024**3))
    dm = DataManagerV2(workspace_path=workspace, enable_provenance=False)
    assert dm._modality_memory_budget == pytest.approx(budget_bytes, rel=1e-6)
    return dm


def test_estimate_anndata_bytes_matches_dense_matrix_size():
    """The estimator must be accurate, or the budget bounds nothing."""
    adata = _make_modality(0)
    assert estimate_anndata_bytes(adata) == BYTES_PER_MODALITY, _shape_label(adata)


def test_registry_stays_within_budget_as_derived_modalities_accumulate(budgeted_dm):
    """Several derived modalities must not all remain resident.

    This test models a pipeline that stores each derived modality separately.
    """
    dm = budgeted_dm
    parent = None
    for i in range(N_DERIVED):
        name = f"pbmc_step{i}"
        dm.store_modality(
            name=name,
            adata=_make_modality(i),
            parent_name=parent,
            step_summary=f"derived step {i}",
        )
        parent = name

    resident = dm._resident_bytes()
    budget = dm._modality_memory_budget

    assert resident <= budget, (
        f"registry holds {resident / 1024**2:.2f} MiB, above the "
        f"{budget / 1024**2:.2f} MiB budget, after {N_DERIVED} derived "
        f"modalities of {N_OBS:,} obs x {N_VARS:,} vars each"
    )
    # The pre-fix behaviour: all six resident. The bug is the *accumulation*.
    assert (
        len(dm.modalities) < N_DERIVED
    ), f"all {N_DERIVED} modalities are still resident — nothing was evicted"
    # Nothing was lost: every modality is still listed.
    assert len(dm.list_modalities()) == N_DERIVED
    assert dm._modality_spilled, "expected at least one modality spilled to disk"


def test_unbounded_when_budget_disabled(workspace, monkeypatch):
    """Budget 0 restores the old unbounded behaviour — proves the test's polarity.

    Without this, a test asserting eviction could pass for the wrong reason.
    """
    monkeypatch.setenv("LOBSTER_MODALITY_MEMORY_BUDGET_GB", "0")
    dm = DataManagerV2(workspace_path=workspace, enable_provenance=False)
    assert dm._modality_memory_budget == 0

    for i in range(N_DERIVED):
        dm.store_modality(name=f"m{i}", adata=_make_modality(i))

    assert len(dm.modalities) == N_DERIVED
    assert dm._modality_spilled == {}
    assert dm._resident_bytes() == N_DERIVED * BYTES_PER_MODALITY


def test_spilled_modality_is_retrievable_and_identical(budgeted_dm):
    """Capping memory must not turn an OOM into a 'not found'."""
    dm = budgeted_dm
    originals = {}
    for i in range(N_DERIVED):
        adata = _make_modality(i)
        originals[f"m{i}"] = adata.X.copy()
        dm.store_modality(name=f"m{i}", adata=adata)

    # The oldest entries are the ones that were spilled.
    assert "m0" in dm._modality_spilled
    assert "m0" not in dm.modalities

    restored = dm.get_modality("m0")
    assert restored is not None
    np.testing.assert_allclose(np.asarray(restored.X), originals["m0"], rtol=0, atol=0)
    # Restoring it must not blow the budget either.
    assert dm._resident_bytes() <= dm._modality_memory_budget

    # Every modality is retrievable, in any order.
    for i in range(N_DERIVED):
        got = dm.get_modality(f"m{i}")
        np.testing.assert_allclose(
            np.asarray(got.X), originals[f"m{i}"], rtol=0, atol=0
        )
        assert dm._resident_bytes() <= dm._modality_memory_budget


def test_peak_resident_bytes_scales_with_budget_not_with_modality_count(
    workspace, monkeypatch
):
    """The memory assertion: resident bytes must be flat in the number of steps.

    Pre-fix this grew with the number of retained modalities.
    """
    monkeypatch.setenv(
        "LOBSTER_MODALITY_MEMORY_BUDGET_GB", str(2 * BYTES_PER_MODALITY / 1024**3)
    )
    dm = DataManagerV2(workspace_path=workspace, enable_provenance=False)

    peaks = []
    for i in range(12):
        dm.store_modality(name=f"step{i}", adata=_make_modality(i))
        peaks.append(dm._resident_bytes())

    budget = dm._modality_memory_budget
    assert max(peaks) <= budget, (
        f"peak resident {max(peaks) / 1024**2:.2f} MiB exceeded budget "
        f"{budget / 1024**2:.2f} MiB over 12 derived modalities of "
        f"{N_OBS:,} obs x {N_VARS:,} vars"
    )
    # Flat, not linear: the last peak is no worse than the third.
    assert peaks[-1] <= peaks[2] + BYTES_PER_MODALITY
    assert len(dm.list_modalities()) == 12


def test_ensure_in_memory_restores_a_spilled_modality(budgeted_dm):
    """Mutating tools call ensure_in_memory(); it must not raise on a spill."""
    dm = budgeted_dm
    for i in range(N_DERIVED):
        dm.store_modality(name=f"m{i}", adata=_make_modality(i))

    assert "m0" in dm._modality_spilled
    adata = dm.ensure_in_memory("m0")
    assert adata is not None
    assert adata.n_obs == N_OBS and adata.n_vars == N_VARS
    assert "m0" in dm._modality_dirty


def test_materialize_modality_restores_a_spilled_modality(budgeted_dm):
    dm = budgeted_dm
    for i in range(N_DERIVED):
        dm.store_modality(name=f"m{i}", adata=_make_modality(i))

    assert "m0" in dm._modality_spilled
    adata = dm.materialize_modality("m0")
    assert adata.n_obs == N_OBS and adata.n_vars == N_VARS


def test_remove_modality_removes_a_spilled_modality(budgeted_dm):
    """A spilled modality is present, so removing it must succeed, not raise."""
    dm = budgeted_dm
    for i in range(N_DERIVED):
        dm.store_modality(name=f"m{i}", adata=_make_modality(i))

    spilled_name = next(iter(dm._modality_spilled))
    spill_path = Path(dm._modality_spilled[spilled_name])
    assert spill_path.exists()

    dm.remove_modality(spilled_name)

    assert spilled_name not in dm.list_modalities()
    assert spilled_name not in dm._modality_spilled
    assert not spill_path.exists(), "spill file should be deleted with the modality"

    with pytest.raises(ValueError):
        dm.get_modality(spilled_name)


def test_remove_modality_still_raises_for_unknown_name(budgeted_dm):
    with pytest.raises(ValueError):
        budgeted_dm.remove_modality("never_existed")


def test_list_modality_records_reports_spilled_status(budgeted_dm):
    dm = budgeted_dm
    for i in range(N_DERIVED):
        dm.store_modality(name=f"m{i}", adata=_make_modality(i))

    records = {r["name"]: r for r in dm.list_modality_records()}
    assert len(records) == N_DERIVED
    statuses = {r["data_status"] for r in records.values()}
    assert "spilled" in statuses
    assert "hot" in statuses


def test_auto_save_persists_spilled_modalities(budgeted_dm):
    """A spilled modality must still reach data_dir, or autosave loses it."""
    dm = budgeted_dm
    for i in range(N_DERIVED):
        dm.store_modality(name=f"m{i}", adata=_make_modality(i))

    spilled = set(dm._modality_spilled)
    assert spilled, "expected spilled modalities"

    dm.auto_save_state()

    for name in spilled:
        expected = dm.data_dir / f"{name}_autosave.h5ad"
        assert expected.exists(), f"spilled modality '{name}' was not auto-saved"
        # And it round-trips to the right shape.
        reloaded = ad.read_h5ad(expected)
        assert (reloaded.n_obs, reloaded.n_vars) == (N_OBS, N_VARS)


def test_most_recently_used_modality_is_never_spilled(budgeted_dm):
    """The entry the caller just stored must stay resident.

    Otherwise every store is immediately followed by a reload.
    """
    dm = budgeted_dm
    for i in range(N_DERIVED):
        name = f"m{i}"
        dm.store_modality(name=name, adata=_make_modality(i))
        assert name in dm.modalities, f"'{name}' was spilled immediately after store"


def test_oversized_single_modality_is_kept_resident(workspace, monkeypatch):
    """A budget smaller than one modality bounds accumulation, not one object.

    It must degrade to "keep it and warn", never to data loss.
    """
    monkeypatch.setenv(
        "LOBSTER_MODALITY_MEMORY_BUDGET_GB", str(BYTES_PER_MODALITY / 4 / 1024**3)
    )
    dm = DataManagerV2(workspace_path=workspace, enable_provenance=False)
    dm.store_modality(name="only", adata=_make_modality(0))

    assert "only" in dm.modalities
    assert dm.get_modality("only").n_obs == N_OBS


def test_invalid_budget_env_falls_back_to_default(workspace, monkeypatch):
    monkeypatch.setenv("LOBSTER_MODALITY_MEMORY_BUDGET_GB", "not-a-number")
    dm = DataManagerV2(workspace_path=workspace, enable_provenance=False)
    assert dm._modality_memory_budget > 0


def test_default_budget_does_not_evict_ordinary_workloads(workspace, monkeypatch):
    """The default must not perturb normal use, or it becomes a new bug."""
    monkeypatch.delenv("LOBSTER_MODALITY_MEMORY_BUDGET_GB", raising=False)
    dm = DataManagerV2(workspace_path=workspace, enable_provenance=False)

    for i in range(N_DERIVED):
        dm.store_modality(name=f"m{i}", adata=_make_modality(i))

    assert dm._modality_spilled == {}
    assert len(dm.modalities) == N_DERIVED
