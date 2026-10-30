"""A refused MuData backup leaves the session as it was.

Saving a backed modality first loads its matrix into memory. That load closes the
file handle, replaces the registered object, marks it modified and re-checks the
memory budget, which can spill an older modality to disk. When the backup the
MuData writer makes is then refused (its name links into the managed figure
folder), those changes used to stay behind. The backup name is now fixed and
checked before the load, for direct saves (with or without a given backup name)
and in the auto-save plan, including a target that only comes to exist because
the load spills another modality to the file it links to.

Real ``DataManagerV2``, real backends, real backed AnnData files and a real memory
budget. The only seam is the clock the backup name reads.
"""

from __future__ import annotations

import logging
from datetime import datetime as _real_datetime
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pytest

from lobster.core.figures.errors import ManagedPathError
from lobster.core.runtime.data_manager import DataManagerV2, estimate_anndata_bytes
from tests.unit.export_guard_fixtures import alias, saved_revision, workspace_tree

NAME = "pbmc_sample"
SHAPE = (50, 40)
NOT_SAVED = "Not saved: 1 modalities refused (managed figure folder)"
LATE_LOG = "Auto-save skipped: the target is inside the managed figure folder."
T0 = _real_datetime(2026, 10, 30, 9, 0, 0)
STAMP0 = T0.strftime("%Y%m%d_%H%M%S")
GIVEN = "20261030_120000"


def _adata(seed: int) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    adata = ad.AnnData(X=rng.random(SHAPE))
    adata.obs_names = [f"cell{i}" for i in range(SHAPE[0])]
    adata.var_names = [f"gene{i}" for i in range(SHAPE[1])]
    return adata


def _session(tmp_path, monkeypatch, **kwargs):
    """Room for one and a half modalities of SHAPE: loading a backed modality spills
    the older resident one."""
    cwd = tmp_path / "cwd"
    cwd.mkdir(exist_ok=True)
    monkeypatch.chdir(cwd)
    size = estimate_anndata_bytes(_adata(0))
    monkeypatch.setenv("LOBSTER_MODALITY_MEMORY_BUDGET_GB", str(1.5 * size / 1024**3))
    dm = DataManagerV2(workspace_path=tmp_path / "ws", **kwargs)
    rev = saved_revision(dm.workspace_path)
    return SimpleNamespace(dm=dm, ws=Path(dm.workspace_path), rev=rev, tmp=tmp_path)


@pytest.fixture
def s(tmp_path, monkeypatch):
    return _session(tmp_path, monkeypatch)


@pytest.fixture
def mu(tmp_path, monkeypatch):
    """Default backend MuData: auto-save makes backups."""
    return _session(tmp_path, monkeypatch, default_backend="mudata")


@pytest.fixture
def clock(monkeypatch):
    """The clock read for an unbound backup name: ``datetime.datetime.now()``."""
    import datetime as datetime_module

    class _Clocked(_real_datetime):
        @classmethod
        def now(cls, tz=None):
            return T0

    monkeypatch.setattr(datetime_module, "datetime", _Clocked)
    return T0


def _resident(s, name="m0", seed=0):
    s.dm.store_modality(name=name, adata=_adata(seed))
    return s.dm.modalities[name]


def _backed(s, seed=1):
    source = s.tmp / f"backed{seed}.h5ad"
    _adata(seed).write_h5ad(source)
    backed = ad.read_h5ad(source, backed="r")
    s.dm.store_modality(name=NAME, adata=backed)
    assert s.dm.modalities[NAME] is backed and backed.isbacked
    return backed


def _state(s):
    return (
        {k: id(v) for k, v in s.dm.modalities.items()},
        set(s.dm._modality_dirty),
        dict(s.dm._modality_spilled),
        dict(s.dm._modality_sources),
    )


def _rev_file(s) -> Path:
    return s.rev.directory / sorted(s.rev.files)[0]


def _sibling(path: Path, stamp: str) -> Path:
    return path.with_suffix(f".{stamp}.backup{path.suffix}")


def _spill_file(s, name="m0") -> Path:
    return s.ws / ".lobster" / "spill" / f"{name}.h5ad"


def _diagnostics(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def _assert_untouched(s, state, held, backed, before):
    assert _state(s) == state and s.dm.modalities["m0"] is held
    assert s.dm.modalities[NAME] is backed and backed.file.is_open  # not loaded
    assert not _spill_file(s).exists()
    assert workspace_tree(s.ws) == before and s.rev.unchanged()


# ------------------------------------------------------------- direct saves


@pytest.mark.parametrize("bound", [False, True], ids=["unbound", "given_name"])
def test_refused_backup_of_a_backed_save_changes_nothing(s, clock, caplog, bound):
    """An existing ordinary destination whose backup name links into a revision."""
    held = _resident(s)
    backed = _backed(s)
    destination = s.dm.data_dir / "x.h5mu"
    destination.write_bytes(b"previous")
    alias(_sibling(destination, GIVEN if bound else STAMP0), _rev_file(s))
    state, before = _state(s), workspace_tree(s.ws)
    extra = {"backup_stamp": GIVEN} if bound else {}
    with caplog.at_level(logging.INFO):
        with pytest.raises(ManagedPathError) as refused:
            s.dm.save_modality(NAME, "x.h5mu", backend="mudata", **extra)
    assert str(s.ws) not in f"{refused.value} {refused.value.args!r}"
    _assert_untouched(s, state, held, backed, before)
    assert destination.read_bytes() == b"previous"
    assert _diagnostics(caplog) == []
    backed.file.close()


def test_backup_target_created_by_the_load_is_refused_first(s, clock, caplog):
    """The destination does not exist yet: it links to the file the load would spill
    the older modality to, and the backup name next to that file links into a
    revision. Refused before the load, so the file is never created."""
    held = _resident(s)
    backed = _backed(s)
    alias(_sibling(_spill_file(s), STAMP0), _rev_file(s))
    destination = alias(s.dm.data_dir / "x.h5mu", _spill_file(s))
    assert not destination.exists()
    state, before = _state(s), workspace_tree(s.ws)
    with caplog.at_level(logging.INFO):
        with pytest.raises(ManagedPathError):
            s.dm.save_modality(NAME, "x.h5mu", backend="mudata")
    _assert_untouched(s, state, held, backed, before)
    assert _diagnostics(caplog) == []
    backed.file.close()


def test_ordinary_backed_save_still_loads_spills_and_backs_up(s, clock):
    """Control: without the link, the save loads the data (spilling the older
    modality), backs up the existing file under the name read at the call, and
    writes as before."""
    _resident(s)
    _backed(s)
    destination = s.dm.data_dir / "x.h5mu"
    destination.write_bytes(b"previous")
    s.dm.save_modality(NAME, "x.h5mu", backend="mudata")
    assert _sibling(destination, STAMP0).read_bytes() == b"previous"
    assert destination.stat().st_size > len(b"previous")
    assert not s.dm.modalities[NAME].isbacked and "m0" in s.dm._modality_spilled


def test_given_backup_name_is_the_one_written(s, clock):
    _resident(s)
    _backed(s)
    destination = s.dm.data_dir / "x.h5mu"
    destination.write_bytes(b"previous")
    s.dm.save_modality(NAME, "x.h5mu", backend="mudata", backup_stamp=GIVEN)
    assert _sibling(destination, GIVEN).read_bytes() == b"previous"
    assert not _sibling(destination, STAMP0).exists()


def test_in_memory_save_to_a_new_file_ignores_an_unused_backup_name(s, clock):
    """Nothing is loaded and the destination does not exist, so no backup is made:
    a backup name linking into a revision does not refuse the save (as before)."""
    _resident(s)
    destination = s.dm.data_dir / "new.h5mu"
    alias(_sibling(destination, STAMP0), _rev_file(s))
    s.dm.save_modality("m0", "new.h5mu", backend="mudata")
    assert destination.is_file() and s.rev.unchanged()


def test_h5ad_save_of_a_backed_modality_is_unchanged(s, clock):
    """The H5AD writer makes no backup: nothing about it is checked or changed."""
    _resident(s)
    _backed(s)
    alias(_sibling(s.dm.data_dir / "x.h5ad", STAMP0), _rev_file(s))
    s.dm.save_modality(NAME, "x.h5ad")
    assert (s.dm.data_dir / "x.h5ad").is_file() and "m0" in s.dm._modality_spilled


# ---------------------------------------------------------------- auto-save


def _progress(calls):
    def progress(current, total, name, status):
        calls.append((name, status))

    return progress


def test_autosave_target_created_by_the_load_is_excluded_in_the_plan(mu, clock, caplog):
    """The backed modality's auto-save target links to the older modality's spill
    file (not there yet); the backup name next to that file links into a revision.
    The plan excludes the backed modality, so it is never loaded: the older modality
    is saved on its own and stays resident, and nothing is spilled."""
    s = mu
    held = _resident(s)
    backed = _backed(s)
    alias(_sibling(_spill_file(s), STAMP0), _rev_file(s))
    target = alias(s.dm.data_dir / f"{NAME}_autosave.h5ad", _spill_file(s))
    assert not target.exists()
    calls = []
    with caplog.at_level(logging.INFO):
        result = s.dm.auto_save_state(progress_callback=_progress(calls))
    assert result.count(NOT_SAVED) == 1 and "Modality 'm0': m0_autosave.h5ad" in result
    assert calls == [("m0", "saving"), ("m0", "done"), (NAME, "error")]
    assert s.dm.modalities["m0"] is held and "m0" not in s.dm._modality_spilled
    assert s.dm.modalities[NAME] is backed and backed.file.is_open
    assert NAME in s.dm._modality_dirty and "m0" not in s.dm._modality_dirty
    assert not _spill_file(s).exists() and not target.exists()
    assert (s.dm.data_dir / "m0_autosave.h5ad").is_file() and s.rev.unchanged()
    # The permitted save of m0 may warn about its own content (count only). Nothing
    # comes from the refused modality: no late refusal, no error, nothing named.
    diagnostics = _diagnostics(caplog)
    assert LATE_LOG not in diagnostics
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
    assert [m for m in diagnostics if NAME in m or str(s.ws) in m] == []
    backed.file.close()


def test_autosave_new_target_of_an_in_memory_modality_is_saved(mu, clock):
    """Control: nothing to load, the target does not exist, so no backup is made and
    the unused backup name is not checked (as before)."""
    s = mu
    _resident(s)
    target = s.dm.data_dir / "m0_autosave.h5ad"
    alias(_sibling(target, STAMP0), _rev_file(s))
    result = s.dm.auto_save_state()
    assert "Modality 'm0': m0_autosave.h5ad" in result
    assert not any(i.startswith("Not saved") for i in result)
    assert target.is_file() and s.rev.unchanged()
