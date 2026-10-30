"""The data manager's native SAVE writers never write into the managed figure folder.

``plots/<figure>/...`` holds only revisions committed by the trusted save step. Every
writer here builds its destination from caller input (a path, a modality name, an
identifier), so each is checked with the export guard before anything is created:
the local backend's actual destination, the MuData backup, the memory spill, both
autosave branches, the data package and the publication cache.

Real ``DataManagerV2``, real registered backends and a revision committed through the
real store. Nothing in the guard or in the writers is patched; the only test seams are
the clock read by the backend backup, the system temp folder and existing callbacks.
"""

from __future__ import annotations

import logging
import os
import tempfile
import zipfile
from datetime import datetime as _real_datetime
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pytest

from lobster.core.figures.errors import ManagedPathError
from lobster.core.figures.export_guard import MANAGED_EXPORT_REASON
from lobster.core.runtime import data_manager as dm_module
from lobster.core.runtime.data_manager import DataManagerV2, estimate_anndata_bytes
from tests.unit.export_guard_fixtures import alias, saved_revision, workspace_tree

NAME = "pbmc_sample"
NOT_SAVED = "Not saved: 1 modalities refused (managed figure folder)"
LATE_LOG = "Auto-save skipped: the target is inside the managed figure folder."
SPILL_LOG = "Memory spill skipped: the spill file is inside the managed figure folder."
T0 = _real_datetime(2026, 10, 28, 9, 0, 0)
T1 = _real_datetime(2026, 10, 28, 9, 0, 1)
T2 = _real_datetime(2026, 10, 28, 9, 0, 2)
# The name the folder check classifies (never created) inside a folder.
PROBE = ".export-dir-check"


# ------------------------------------------------------------------------- helpers


def _adata(seed: int = 0, n_obs: int = 6, n_vars: int = 5) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    adata = ad.AnnData(X=rng.random((n_obs, n_vars)))
    adata.obs_names = [f"cell{i}" for i in range(n_obs)]
    adata.var_names = [f"gene{i}" for i in range(n_vars)]
    return adata


SPILL_SHAPE = (50, 40)


def _session(tmp_path, monkeypatch, budget_of=None, **kwargs):
    cwd = tmp_path / "cwd"
    cwd.mkdir(exist_ok=True)
    monkeypatch.chdir(cwd)
    if budget_of is None:
        monkeypatch.delenv("LOBSTER_MODALITY_MEMORY_BUDGET_GB", raising=False)
    else:  # room for one and a half modalities of SPILL_SHAPE
        size = estimate_anndata_bytes(_adata(0, *SPILL_SHAPE))
        monkeypatch.setenv(
            "LOBSTER_MODALITY_MEMORY_BUDGET_GB", str(budget_of * size / 1024**3)
        )
    dm = DataManagerV2(workspace_path=tmp_path / "ws", **kwargs)
    rev = saved_revision(dm.workspace_path)
    return SimpleNamespace(dm=dm, ws=Path(dm.workspace_path), rev=rev, tmp=tmp_path)


@pytest.fixture
def s(tmp_path, monkeypatch):
    return _session(tmp_path, monkeypatch)


@pytest.fixture
def any_name(monkeypatch):
    """Admit a name the modality-name check (1.2.1) refuses, as a name arriving by
    another route would: the downstream guard under test must still hold."""
    monkeypatch.setattr(
        "lobster.core.runtime.data_manager.check_modality_name", lambda name: None
    )


@pytest.fixture
def mu(tmp_path, monkeypatch):
    """A data manager whose default backend is MuData (its backup is active)."""
    return _session(tmp_path, monkeypatch, default_backend="mudata")


@pytest.fixture
def spill(tmp_path, monkeypatch):
    return _session(tmp_path, monkeypatch, budget_of=1.5)


def _store(s, name=NAME, seed=0, shape=(6, 5)):
    s.dm.store_modality(name=name, adata=_adata(seed, *shape))
    return s.dm.modalities[name]


def _rev_file(s) -> Path:
    return s.rev.directory / sorted(s.rev.files)[0]


def _figure_new(s) -> Path:
    """A folder that does not exist yet inside an existing figure folder."""
    return s.rev.directory.parent / "new"


def _detour(s, start: Path, end: Path) -> Path:
    """``start/x/../..`` + a walk through ``plots/<figure>/new`` + back to ``end``.

    Classified as ``end``; created recursively, it creates ``plots/<figure>/new``.
    """
    up_to_ws = [".."] * len(start.relative_to(s.ws).parts)
    fig = s.rev.directory.parent.name
    spelled = Path(start, "x", "..", *up_to_ws, "plots", fig, "new", "..", "..", "..")
    return Path(spelled, *end.relative_to(s.ws).parts)


def _key(s, path: Path) -> str:
    return f"{s.ws}/{Path(path).relative_to(s.ws).as_posix()}"


def _changed(before, after):
    return {k for k in set(before) | set(after) if before.get(k) != after.get(k)}


def _only_ordinary_changes(s, before, after, *files):
    """Every change is one of ``files`` (new or rewritten) or a folder's time."""
    allowed = {_key(s, f) for f in files} | {
        _key(s, s.dm.exports_dir / "processing_log.json")
    }
    for k in _changed(before, after):
        if k in allowed:
            continue
        assert before.get(k, ("?",))[0] == "dir" == after.get(k, ("?",))[0], k


def _no_plots_change(s, before, after):
    plots = f"{s.ws}/plots"
    assert [k for k in _changed(before, after) if k.startswith(plots)] == []


def _refusal_is_path_free(s, error):
    text = f"{error} {error.args!r}"
    assert str(s.ws) not in text and "plots/" + s.rev.directory.parent.name not in text
    assert error.details == {}


def _diagnostics(caplog, level=logging.WARNING):
    return [r for r in caplog.records if r.levelno >= level]


def _named(records, *needles):
    return [
        r.getMessage() for r in records if any(n in r.getMessage() for n in needles)
    ]


class _BackupClock:
    """The clock the backend backup reads: ``datetime.datetime.now()`` of the stdlib
    ``datetime`` module that base.py imports (looked up at call time, at the base and
    after the fix alike). Modules holding their own ``datetime`` class are unaffected.
    """

    def __init__(self, monkeypatch, at=T0):
        import datetime as datetime_module

        self.at = at
        clock = self

        class _Clocked(_real_datetime):
            @classmethod
            def now(cls, tz=None):
                return clock.at

        monkeypatch.setattr(datetime_module, "datetime", _Clocked)


def _sibling(destination: Path, at) -> Path:
    stamp = at.strftime("%Y%m%d_%H%M%S")
    return destination.with_suffix(f".{stamp}.backup{destination.suffix}")


def _backups(folder: Path):
    return sorted(p.name for p in folder.iterdir() if ".backup." in p.name)


# ==================================================== A1: the local destination


@pytest.mark.parametrize("writer", ["save_modality", "save_mudata"])
def test_a1_rerooted_destination_through_a_data_alias_is_refused(s, writer):
    """``W/exports/x`` is allowed as spelled, but the backend writes ``W/data/exports/x``."""
    _store(s)
    alias(s.dm.data_dir / "exports", s.rev.directory)
    before = workspace_tree(s.ws)
    with pytest.raises(ManagedPathError) as refused:
        if writer == "save_modality":
            s.dm.save_modality(NAME, s.ws / "exports" / "x.h5ad")
        else:
            s.dm.save_mudata(s.ws / "exports" / "x.h5mu")
    _refusal_is_path_free(s, refused.value)
    assert workspace_tree(s.ws) == before and s.rev.unchanged()


@pytest.mark.parametrize("writer", ["save_modality", "save_mudata"])
def test_a1_ordinary_rerooted_destination_is_unchanged(s, writer):
    _store(s)
    if writer == "save_modality":
        returned = s.dm.save_modality(NAME, s.ws / "exports" / "x.h5ad")
        spelled, physical = (
            s.ws / "exports" / "x.h5ad",
            s.dm.data_dir / "exports/x.h5ad",
        )
    else:
        returned = s.dm.save_mudata(s.ws / "exports" / "x.h5mu")
        spelled, physical = (
            s.ws / "exports" / "x.h5mu",
            s.dm.data_dir / "exports/x.h5mu",
        )
    assert returned == str(spelled)  # the caller spelling is returned, as before
    assert physical.is_file() and not spelled.exists()


def test_a1_final_file_alias_stays_refused(s):
    _store(s)
    alias(s.dm.data_dir / "x.h5ad", _rev_file(s))
    before = workspace_tree(s.ws)
    with pytest.raises(ManagedPathError):
        s.dm.save_modality(NAME, "x.h5ad")
    assert workspace_tree(s.ws) == before and s.rev.unchanged()


def test_a1_recursive_folder_through_a_figure_folder_is_refused(s):
    _store(s)
    target = _detour(s, s.dm.data_dir, s.dm.data_dir / "deep" / "y.h5ad")
    before = workspace_tree(s.ws)
    with pytest.raises(ManagedPathError):
        s.dm.save_modality(NAME, target)
    assert workspace_tree(s.ws) == before
    assert not _figure_new(s).exists()


def test_a1_existing_folder_alias_into_a_revision_is_refused(s):
    _store(s)
    alias(s.dm.data_dir / "sub", s.rev.directory)
    before = workspace_tree(s.ws)
    with pytest.raises(ManagedPathError):
        s.dm.save_modality(NAME, "sub/x.h5ad")
    assert workspace_tree(s.ws) == before and s.rev.unchanged()


def test_a1_ordinary_nested_folder_is_created_as_before(s):
    _store(s)
    s.dm.save_modality(NAME, "a/b/x.h5ad")
    assert (s.dm.data_dir / "a" / "b" / "x.h5ad").is_file()


@pytest.mark.parametrize("spelling", ["relative", "absolute"])
def test_a1_workspace_root_destination_stays_allowed(s, spelling):
    _store(s)
    if spelling == "relative":  # data/../root.h5ad: written in the workspace root
        s.dm.save_modality(NAME, "../root.h5ad")
        assert (s.ws / "root.h5ad").is_file()
    else:  # W/root.h5ad is re-rooted under data/ by the backend, as before
        assert s.dm.save_modality(NAME, s.ws / "root.h5ad") == str(s.ws / "root.h5ad")
        assert (s.dm.data_dir / "root.h5ad").is_file()


def test_a1_new_figure_shaped_folder_is_refused(s):
    _store(s)
    with pytest.raises(ManagedPathError):
        s.dm.save_modality(NAME, s.ws / "plots" / "newname" / "x.h5ad")
    assert not (s.ws / "plots" / "newname").exists()


def test_a1_unrelated_backend_error_keeps_its_type(s):
    _store(s)
    (s.dm.data_dir / "afile").write_text("x")
    with pytest.raises(OSError) as failed:
        s.dm.save_modality(NAME, "afile/x.h5ad")
    assert not isinstance(failed.value, ManagedPathError)


def _backed(s, seed=1):
    """A modality whose matrix stays in its file (backed mode) until it is saved."""
    source = s.tmp / f"backed{seed}.h5ad"
    _adata(seed, *SPILL_SHAPE).write_h5ad(source)
    backed = ad.read_h5ad(source, backed="r")
    s.dm.store_modality(name=NAME, adata=backed)
    assert s.dm.modalities[NAME] is backed and backed.isbacked
    return backed


def _registry(s):
    return (
        {k: id(v) for k, v in s.dm.modalities.items()},
        set(s.dm._modality_dirty),
        dict(s.dm._modality_spilled),
        dict(s.dm._modality_sources),
    )


def test_a1_rerooted_refusal_comes_before_a_backed_modality_is_loaded(spill):
    """Loading backed data before the save can spill another modality to disk: the
    destination the backend writes is refused before that (no state, no file)."""
    s = spill
    held = _store(s, "m0", 0, SPILL_SHAPE)
    backed = _backed(s)
    alias(s.dm.data_dir / "exports", s.rev.directory)
    state = _registry(s)
    before = workspace_tree(s.ws)
    with pytest.raises(ManagedPathError) as refused:
        s.dm.save_modality(NAME, s.ws / "exports" / "x.h5ad")
    _refusal_is_path_free(s, refused.value)
    assert _registry(s) == state and s.dm.modalities["m0"] is held
    assert s.dm.modalities[NAME] is backed and backed.file.is_open  # not loaded
    assert workspace_tree(s.ws) == before and s.rev.unchanged()
    backed.file.close()


def test_a1_ordinary_backed_save_is_unchanged(spill):
    """Control: the same save without the alias loads the data first, which spills
    the older modality (the budget fits one and a half), and writes as before."""
    s = spill
    _store(s, "m0", 0, SPILL_SHAPE)
    _backed(s)
    returned = s.dm.save_modality(NAME, s.ws / "exports" / "x.h5ad")
    assert returned == str(s.ws / "exports" / "x.h5ad")
    assert (s.dm.data_dir / "exports" / "x.h5ad").is_file()
    assert not s.dm.modalities[NAME].isbacked and "m0" in s.dm._modality_spilled


@pytest.mark.parametrize("probe", ["regular", "outward_link", "lstat_error"])
def test_a1_folder_with_its_check_name_taken_is_refused_before_mkdir(
    s, probe, monkeypatch
):
    """The folder check classifies a name it never creates. If that name already
    exists there (it could point the check elsewhere) or cannot be looked up, the
    folder is refused with the fixed reason before anything is created."""
    _store(s)
    folder = s.dm.data_dir / ("new" if probe == "lstat_error" else "sub")
    if probe == "regular":
        folder.mkdir()
        (folder / PROBE).write_text("x")
    elif probe == "outward_link":  # dangling, pointing out of the workspace
        folder.mkdir()
        alias(folder / PROBE, s.tmp / "outside" / "missing")
    else:  # an unexpected lookup error for the check name only
        real_lstat = os.lstat

        def lstat(path, *args, **kwargs):
            if os.path.basename(os.fspath(path)) == PROBE:
                raise PermissionError(13, "Permission denied")
            return real_lstat(path, *args, **kwargs)

        monkeypatch.setattr(os, "lstat", lstat)
    before = workspace_tree(s.ws)
    with pytest.raises(ManagedPathError) as refused:
        s.dm.save_modality(NAME, f"{folder.name}/x.h5ad")
    assert str(refused.value) == MANAGED_EXPORT_REASON
    _refusal_is_path_free(s, refused.value)
    assert workspace_tree(s.ws) == before and not (folder / "x.h5ad").exists()
    assert folder.exists() is (probe != "lstat_error")


# ==================================================== A1b: the MuData backup file


def test_a1b_backup_alias_into_a_revision_is_refused(s, monkeypatch):
    _BackupClock(monkeypatch, T0)
    _store(s)
    s.dm.save_mudata("x.h5mu")  # an ordinary existing destination (no backup yet)
    destination = s.dm.data_dir / "x.h5mu"
    alias(_sibling(destination, T0), _rev_file(s))
    before = workspace_tree(s.ws)
    with pytest.raises(ManagedPathError) as refused:
        s.dm.save_mudata("x.h5mu")
    _refusal_is_path_free(s, refused.value)
    assert workspace_tree(s.ws) == before and s.rev.unchanged()


def test_a1b_ordinary_backup_is_written_as_before(s):
    _store(s)
    s.dm.save_mudata("x.h5mu")
    destination = s.dm.data_dir / "x.h5mu"
    previous = destination.read_bytes()
    s.dm.save_mudata("x.h5mu")
    [backup] = _backups(s.dm.data_dir)
    assert (s.dm.data_dir / backup).read_bytes() == previous
    assert backup.startswith("x.") and backup.endswith(".backup.h5mu")


def test_a1b_backup_name_taken_by_a_folder_is_refused(s, monkeypatch):
    """A copy into an existing folder would write an unchecked name inside it."""
    _BackupClock(monkeypatch, T0)
    _store(s)
    s.dm.save_mudata("x.h5mu")
    destination = s.dm.data_dir / "x.h5mu"
    folder = _sibling(destination, T0)
    folder.mkdir()
    alias(folder / destination.name, _rev_file(s))
    before = workspace_tree(s.ws)
    with pytest.raises(ManagedPathError) as refused:
        s.dm.save_mudata("x.h5mu")
    _refusal_is_path_free(s, refused.value)
    assert workspace_tree(s.ws) == before and s.rev.unchanged()


def _ordinary_folder_holding(s, name):
    """An ordinary folder under exports/ whose ``name`` entry links into a revision."""
    folder = s.ws / "exports" / "ordinary_folder"
    folder.mkdir()
    alias(folder / name, _rev_file(s))
    return folder


def test_a1b_backup_name_linking_to_a_folder_is_refused(s, monkeypatch):
    """The backup name is a link to an ordinary folder: the copy would land on an
    unchecked name inside it (the variant of the folder case above)."""
    _BackupClock(monkeypatch, T0)
    _store(s)
    s.dm.save_mudata("x.h5mu")
    destination = s.dm.data_dir / "x.h5mu"
    alias(_sibling(destination, T0), _ordinary_folder_holding(s, destination.name))
    before = workspace_tree(s.ws)
    with pytest.raises(ManagedPathError) as refused:
        s.dm.save_mudata("x.h5mu")
    _refusal_is_path_free(s, refused.value)
    assert workspace_tree(s.ws) == before and s.rev.unchanged()


def _warns_on_save(held):
    held.uns["qc_count"] = 1  # a value the pre-save validation reports
    return held


def test_a1_early_refusal_comes_before_the_validation_warning(s, caplog):
    _warns_on_save(_store(s))
    alias(s.dm.data_dir / "exports", s.rev.directory)
    with caplog.at_level(logging.INFO):
        with pytest.raises(ManagedPathError):
            s.dm.save_modality(NAME, s.ws / "exports" / "x.h5ad")
    assert _diagnostics(caplog) == [] and s.rev.unchanged()


def test_a1_validation_warning_gives_the_count_only(s, caplog):
    _warns_on_save(_store(s))
    with caplog.at_level(logging.DEBUG, logger="lobster.core.runtime.data_manager"):
        s.dm.save_modality(NAME, "x.h5ad")
    warnings = [r.getMessage() for r in _diagnostics(caplog)]
    assert any(
        m.startswith("Pre-save validation found ")
        and m.endswith(
            " potential serialization issues. These will be sanitized "
            "automatically during save."
        )
        for m in warnings
    )
    assert _named(_diagnostics(caplog), NAME, "qc_count", str(s.ws)) == []
    debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert any(NAME in m and "qc_count" in m for m in debug)  # details at DEBUG


def test_a1b_direct_mudata_backup_refusal_after_the_warning_names_nothing(
    s, monkeypatch, caplog
):
    _BackupClock(monkeypatch, T0)
    _warns_on_save(_store(s))
    s.dm.save_modality(NAME, "m.h5mu", backend="mudata")
    alias(_sibling(s.dm.data_dir / "m.h5mu", T0), _rev_file(s))
    before = workspace_tree(s.ws)
    with caplog.at_level(logging.INFO):
        with pytest.raises(ManagedPathError):
            s.dm.save_modality(NAME, "m.h5mu", backend="mudata")
    assert workspace_tree(s.ws) == before and s.rev.unchanged()
    assert _named(_diagnostics(caplog), NAME, "qc_count", str(s.ws)) == []


def test_a1b_directly_constructed_backend_backup_is_unguarded(tmp_path):
    """Only the data manager's own instances carry the workspace guard."""
    from lobster.core.backends.mudata_backend import MuDataBackend

    backend = MuDataBackend(base_path=tmp_path)
    assert getattr(backend, "guard_workspace", None) is None


def test_queues_folder_through_a_revision_is_refused_before_mkdir(s):
    """``.lobster -> revision``: the first queue use would create ``queues`` in it."""
    alias(s.ws / ".lobster", s.rev.directory)
    before = workspace_tree(s.ws)
    with pytest.raises(ManagedPathError):
        s.dm.download_queue
    assert workspace_tree(s.ws) == before and s.rev.unchanged()
    assert not (s.rev.directory / "queues").exists()


# ======================================================== A2: the memory spill


@pytest.mark.parametrize("case", ["folder", "file"])
def test_a2_spill_into_a_revision_keeps_the_modality_resident(spill, case, caplog):
    s = spill
    held = _store(s, "m0", 0, SPILL_SHAPE)
    lobster_dir = s.ws / ".lobster"
    if case == "folder":
        assert not lobster_dir.exists()
        alias(lobster_dir, s.rev.directory)
        assert not (s.rev.directory / "spill").exists()
    else:
        (lobster_dir / "spill").mkdir(parents=True)
        alias(lobster_dir / "spill" / "m0.h5ad", _rev_file(s))
    dirty, sources = set(s.dm._modality_dirty), dict(s.dm._modality_sources)
    before = workspace_tree(s.ws)
    with caplog.at_level(logging.INFO):
        _store(s, "m1", 1, SPILL_SHAPE)  # over budget: m0 (not the newest) is evicted
    assert s.dm.modalities["m0"] is held  # failed eviction: still resident, intact
    assert "m0" not in s.dm._modality_spilled
    assert s.dm._modality_dirty == dirty | {"m1"}
    assert {k: v for k, v in s.dm._modality_sources.items() if k != "m1"} == sources
    assert workspace_tree(s.ws) == before and s.rev.unchanged()
    assert SPILL_LOG in [r.getMessage() for r in _diagnostics(caplog)]
    assert _named(_diagnostics(caplog), "m0", str(s.ws)) == []


def test_a2_recursive_spill_folder_through_a_figure_folder_is_refused(spill):
    s = spill
    held = _store(s, "m0", 0, SPILL_SHAPE)
    s.dm._modality_spill_dir = _detour(s, s.ws, s.ws / ".lobster" / "spill")
    before = workspace_tree(s.ws)
    _store(s, "m1", 1, SPILL_SHAPE)
    assert s.dm.modalities["m0"] is held and s.dm._modality_spilled == {}
    _no_plots_change(s, before, workspace_tree(s.ws))
    assert not _figure_new(s).exists()


def test_a2_spill_file_alias_behind_a_missing_folder_creates_nothing(spill):
    """``W/.lobster/new/../spill``: ``new`` does not exist (creating the folder would
    create it) and the spill file there links into a revision. The file is refused
    before any folder is created."""
    s = spill
    held = _store(s, "m0", 0, SPILL_SHAPE)
    (s.ws / ".lobster" / "spill").mkdir(parents=True)
    alias(s.ws / ".lobster" / "spill" / "m0.h5ad", _rev_file(s))
    s.dm._modality_spill_dir = s.ws / ".lobster" / "new" / ".." / "spill"
    before = workspace_tree(s.ws)
    _store(s, "m1", 1, SPILL_SHAPE)
    assert s.dm.modalities["m0"] is held and s.dm._modality_spilled == {}
    assert workspace_tree(s.ws) == before and s.rev.unchanged()
    assert not (s.ws / ".lobster" / "new").exists()


def test_a2_ordinary_spill_is_unchanged(spill):
    s = spill
    _store(s, "m0", 0, SPILL_SHAPE)
    _store(s, "m1", 1, SPILL_SHAPE)
    assert "m0" not in s.dm.modalities
    assert s.dm._modality_spilled["m0"] == s.ws / ".lobster" / "spill" / "m0.h5ad"
    assert s.dm._modality_spilled["m0"].is_file()


# ============================================================ A3: autosave


def _progress(calls, on_saving=None):
    def progress(current, total, name, status):
        calls.append((name, status))
        if status == "saving" and on_saving is not None:
            on_saving(name)

    return progress


def test_a3_resident_target_alias_is_excluded_before_any_write(s, caplog):
    _store(s)
    alias(s.dm.data_dir / f"{NAME}_autosave.h5ad", _rev_file(s))
    _store(s, "ordinary", 1)
    before = workspace_tree(s.ws)
    calls = []
    with caplog.at_level(logging.INFO):
        result = s.dm.auto_save_state(progress_callback=_progress(calls))
    assert result.count(NOT_SAVED) == 1 and NAME not in " ".join(result)
    assert "Modality 'ordinary': ordinary_autosave.h5ad" in result
    assert calls == [(NAME, "error"), ("ordinary", "saving"), ("ordinary", "done")]
    assert NAME in s.dm._modality_dirty and "ordinary" not in s.dm._modality_dirty
    after = workspace_tree(s.ws)
    _only_ordinary_changes(s, before, after, s.dm.data_dir / "ordinary_autosave.h5ad")
    assert s.rev.unchanged()
    assert _named(_diagnostics(caplog), NAME, str(s.ws)) == []


def test_a3_resident_target_refused_late_is_accounted(s, caplog):
    """The target turns into an alias after the plan (at the existing callback)."""
    _store(s)
    target = s.dm.data_dir / f"{NAME}_autosave.h5ad"
    calls = []
    progress = _progress(calls, on_saving=lambda name: alias(target, _rev_file(s)))
    with caplog.at_level(logging.INFO):
        result = s.dm.auto_save_state(progress_callback=progress)
    assert calls == [(NAME, "saving"), (NAME, "error")]
    assert result.count(NOT_SAVED) == 1 and NAME not in " ".join(result)
    assert NAME in s.dm._modality_dirty and s.rev.unchanged()
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
    assert LATE_LOG in [r.getMessage() for r in _diagnostics(caplog)]
    assert _named(_diagnostics(caplog), NAME, str(s.ws)) == []


@pytest.mark.usefixtures("any_name")
def test_a3_resident_recursive_folder_through_a_figure_folder_is_excluded(s):
    name = f"x/../../plots/{s.rev.directory.parent.name}/new/../../../data/{NAME}"
    _store(s, name)
    before = workspace_tree(s.ws)
    result = s.dm.auto_save_state()
    assert result.count(NOT_SAVED) == 1
    _no_plots_change(s, before, workspace_tree(s.ws))
    assert not _figure_new(s).exists()


@pytest.mark.parametrize("name", ["pbmc", "pbmc_autosave"])
def test_a3_spilled_target_alias_is_not_saved(spill, name, caplog):
    s = spill
    _store(s, name, 0, SPILL_SHAPE)
    _store(s, "other", 1, SPILL_SHAPE)  # spills `name` (a successful spill)
    assert name in s.dm._modality_spilled
    alias(s.dm.data_dir / "pbmc_autosave.h5ad", _rev_file(s))
    state = (
        set(s.dm._modality_dirty),
        dict(s.dm._modality_spilled),
        dict(s.dm._modality_sources),
    )
    before = workspace_tree(s.ws)
    with caplog.at_level(logging.INFO):
        result = s.dm.auto_save_state()
    after = workspace_tree(s.ws)
    assert result.count(NOT_SAVED) == 1 and "pbmc" not in " ".join(result)
    assert (
        set(s.dm._modality_dirty),
        dict(s.dm._modality_spilled),
        dict(s.dm._modality_sources),
    ) == (state[0] - {"other"}, state[1], state[2])
    _only_ordinary_changes(s, before, after, s.dm.data_dir / "other_autosave.h5ad")
    assert s.rev.unchanged()
    assert _named(_diagnostics(caplog), "pbmc", str(s.ws)) == []


def test_a3_spilled_target_taken_by_a_folder_is_not_saved(spill):
    """A copy into an existing folder would write an unchecked name inside it."""
    s = spill
    _store(s, "pbmc", 0, SPILL_SHAPE)
    _store(s, "other", 1, SPILL_SHAPE)  # spills pbmc
    spilled = Path(s.dm._modality_spilled["pbmc"])
    folder = s.dm.data_dir / "pbmc_autosave.h5ad"
    folder.mkdir()
    alias(folder / spilled.name, _rev_file(s))
    state = (dict(s.dm._modality_spilled), set(s.dm._modality_dirty) - {"other"})
    before = workspace_tree(s.ws)
    result = s.dm.auto_save_state()
    assert result.count(NOT_SAVED) == 1
    assert (dict(s.dm._modality_spilled), set(s.dm._modality_dirty)) == state
    _only_ordinary_changes(
        s, before, workspace_tree(s.ws), s.dm.data_dir / "other_autosave.h5ad"
    )
    assert s.rev.unchanged()


def test_a3_spilled_target_linking_to_a_folder_is_not_saved(spill):
    """The variant of the folder case above: the target links to an ordinary folder."""
    s = spill
    _store(s, "pbmc", 0, SPILL_SHAPE)
    _store(s, "other", 1, SPILL_SHAPE)  # spills pbmc
    spilled = Path(s.dm._modality_spilled["pbmc"])
    folder = _ordinary_folder_holding(s, spilled.name)
    alias(s.dm.data_dir / "pbmc_autosave.h5ad", folder)
    state = (dict(s.dm._modality_spilled), set(s.dm._modality_dirty) - {"other"})
    before = workspace_tree(s.ws)
    result = s.dm.auto_save_state()
    assert result.count(NOT_SAVED) == 1
    assert (dict(s.dm._modality_spilled), set(s.dm._modality_dirty)) == state
    _only_ordinary_changes(
        s, before, workspace_tree(s.ws), s.dm.data_dir / "other_autosave.h5ad"
    )
    assert s.rev.unchanged()


@pytest.mark.usefixtures("any_name")
def test_a3_spilled_recursive_folder_through_a_figure_folder_is_not_saved(spill):
    s = spill
    name = f"x/../../plots/{s.rev.directory.parent.name}/new/../../../data/{NAME}"
    _store(s, name, 0, SPILL_SHAPE)
    _store(s, "other", 1, SPILL_SHAPE)
    assert name in s.dm._modality_spilled
    before = workspace_tree(s.ws)
    result = s.dm.auto_save_state()
    assert result.count(NOT_SAVED) == 1
    _no_plots_change(s, before, workspace_tree(s.ws))
    assert not _figure_new(s).exists()


@pytest.mark.parametrize("name", ["pbmc", "pbmc_autosave"])
def test_a3_ordinary_autosave_is_unchanged(spill, name):
    s = spill
    _store(s, name, 0, SPILL_SHAPE)
    _store(s, "other", 1, SPILL_SHAPE)
    result = s.dm.auto_save_state(force=True)
    assert result == [
        "Modality 'other': other_autosave.h5ad",
        f"Modality '{name}': pbmc_autosave.h5ad (from spill)",
        "Processing log",
    ]
    assert (s.dm.data_dir / "pbmc_autosave.h5ad").is_file()
    assert not any(i.startswith("Not saved") for i in s.dm.auto_save_state())


def test_a3_has_saveable_modalities_counts_resident_and_spilled(spill):
    s = spill
    assert s.dm.has_saveable_modalities() is False
    _store(s, "a", 0, SPILL_SHAPE)
    assert s.dm.has_saveable_modalities() is True
    _store(s, "b", 1, SPILL_SHAPE)
    s.dm.remove_modality("b")
    assert s.dm.modalities == {} and "a" in s.dm._modality_spilled
    assert s.dm.has_saveable_modalities() is True and s.dm.has_data() is False


# -------------------------------- A3 + A1b: one backup name from plan to execution


def _mu_saved_once(s):
    _store(s)
    s.dm.auto_save_state()  # writes the destination (nothing to back up yet)
    destination = s.dm.data_dir / f"{NAME}_autosave.h5ad"
    assert destination.is_file() and _backups(s.dm.data_dir) == []
    s.dm.mark_modality_dirty(NAME)
    return destination


def _advance(clock, to):
    def on_saving(name):
        clock.at = to

    return on_saving


def test_a3_backup_alias_at_the_planned_name_is_excluded(mu, monkeypatch):
    s = mu
    clock = _BackupClock(monkeypatch, T0)
    destination = _mu_saved_once(s)
    alias(_sibling(destination, T0), _rev_file(s))
    later = s.ws / "exports" / "later.bin"
    later.write_bytes(b"later")
    alias(_sibling(destination, T1), later)
    before = workspace_tree(s.ws)
    calls = []
    result = s.dm.auto_save_state(
        progress_callback=_progress(calls, _advance(clock, T1))
    )
    assert calls == [(NAME, "error")]
    assert result.count(NOT_SAVED) == 1 and NAME in s.dm._modality_dirty
    _only_ordinary_changes(s, before, workspace_tree(s.ws))
    assert later.read_bytes() == b"later" and s.rev.unchanged()


def test_a3_backup_name_taken_by_a_folder_is_excluded(mu, monkeypatch):
    s = mu
    _BackupClock(monkeypatch, T0)
    destination = _mu_saved_once(s)
    folder = _sibling(destination, T0)
    folder.mkdir()
    alias(folder / destination.name, _rev_file(s))
    before = workspace_tree(s.ws)
    calls = []
    result = s.dm.auto_save_state(progress_callback=_progress(calls))
    assert calls == [(NAME, "error")]
    assert result.count(NOT_SAVED) == 1 and NAME in s.dm._modality_dirty
    _only_ordinary_changes(s, before, workspace_tree(s.ws))
    assert s.rev.unchanged()


def test_a3_backup_name_linking_to_a_folder_is_excluded(mu, monkeypatch):
    """The variant of the folder case above: the planned name links to a folder."""
    s = mu
    _BackupClock(monkeypatch, T0)
    destination = _mu_saved_once(s)
    alias(_sibling(destination, T0), _ordinary_folder_holding(s, destination.name))
    before = workspace_tree(s.ws)
    calls = []
    result = s.dm.auto_save_state(progress_callback=_progress(calls))
    assert calls == [(NAME, "error")]
    assert result.count(NOT_SAVED) == 1 and NAME in s.dm._modality_dirty
    _only_ordinary_changes(s, before, workspace_tree(s.ws))
    assert s.rev.unchanged()


def _aliased_target(s, where):
    """An auto-save target that is an ordinary link into exports/: the file itself
    (``file``) or its folder (``parent``). Saved once, then marked modified."""
    inner = s.ws / "exports" / "inner"
    inner.mkdir()
    if where == "file":
        name = NAME
        _store(s, name)
        s.dm.auto_save_state()
        saved = s.dm.data_dir / f"{name}_autosave.h5ad"
        source = inner / "src.h5ad"
        saved.rename(source)
        alias(saved, source)
    else:
        name = f"sub/{NAME}"
        alias(s.dm.data_dir / "sub", inner)
        _store(s, name)
        s.dm.auto_save_state()
        source = inner / f"{NAME}_autosave.h5ad"
    assert source.is_file() and _backups(inner) == []
    s.dm.mark_modality_dirty(name)
    return name, source


@pytest.mark.parametrize("where", ["file", "parent"])
@pytest.mark.usefixtures("any_name")
def test_a3_backup_of_a_linked_target_lands_next_to_the_file_it_copies(
    mu, monkeypatch, where
):
    """The backup copies the file the target resolves to and is named next to that
    file: the name planned at t0 is the one written (deviations v1 section 1)."""
    s = mu
    clock = _BackupClock(monkeypatch, T0)
    name, source = _aliased_target(s, where)
    previous = source.read_bytes()
    calls = []
    s.dm.auto_save_state(progress_callback=_progress(calls, _advance(clock, T1)))
    assert calls == [(name, "saving"), (name, "done")]
    assert _backups(source.parent) == [_sibling(source, T0).name]
    assert _sibling(source, T0).read_bytes() == previous
    assert _backups(s.dm.data_dir) == []


@pytest.mark.parametrize("where", ["file", "parent"])
@pytest.mark.usefixtures("any_name")
def test_a3_backup_alias_next_to_the_linked_file_is_excluded(mu, monkeypatch, where):
    """That actual backup name links into a revision: refused in the plan, before
    the save starts (no "saving" call) and before any payload is written."""
    s = mu
    _BackupClock(monkeypatch, T0)
    name, source = _aliased_target(s, where)
    previous = source.read_bytes()
    alias(_sibling(source, T0), _rev_file(s))
    before = workspace_tree(s.ws)
    calls = []
    result = s.dm.auto_save_state(progress_callback=_progress(calls))
    assert calls == [(name, "error")]
    assert result.count(NOT_SAVED) == 1 and name in s.dm._modality_dirty
    assert source.read_bytes() == previous
    _only_ordinary_changes(s, before, workspace_tree(s.ws))
    assert s.rev.unchanged()


def test_a3_ordinary_backup_uses_the_planned_name(mu, monkeypatch):
    s = mu
    clock = _BackupClock(monkeypatch, T0)
    destination = _mu_saved_once(s)
    previous = destination.read_bytes()
    s.dm.auto_save_state(progress_callback=_progress([], _advance(clock, T1)))
    assert _backups(s.dm.data_dir) == [_sibling(destination, T0).name]
    assert _sibling(destination, T0).read_bytes() == previous


def test_a3_overlapping_direct_save_keeps_its_own_backup_name(mu, monkeypatch):
    s = mu
    clock = _BackupClock(monkeypatch, T0)
    _store(s, "second", 1)
    destination = _mu_saved_once(s)
    s.dm.save_modality("second", "other.h5mu", backend="mudata")
    other = s.dm.data_dir / "other.h5mu"

    def on_saving(name):
        if name == NAME:
            clock.at = T1
            # unbound, through the same DM-owned MuData backend instance
            s.dm.save_modality("second", "other.h5mu", backend="mudata")

    s.dm.auto_save_state(progress_callback=_progress([], on_saving))
    assert _backups(s.dm.data_dir) == sorted(
        [_sibling(destination, T0).name, _sibling(other, T1).name]
    )


def test_a3_unbound_direct_saves_read_the_clock_as_before(s, monkeypatch):
    _BackupClock(monkeypatch, T2)
    _store(s)
    s.dm.save_modality(NAME, "m.h5mu", backend="mudata")
    s.dm.save_modality(NAME, "m.h5mu", backend="mudata")
    assert _backups(s.dm.data_dir) == [_sibling(s.dm.data_dir / "m.h5mu", T2).name]


def test_a3_refusal_leaves_no_state_behind(mu, monkeypatch):
    s = mu
    clock = _BackupClock(monkeypatch, T0)
    destination = _mu_saved_once(s)
    backends = {k: dict(vars(b)) for k, b in s.dm.backends.items()}
    refused = alias(_sibling(destination, T0), _rev_file(s))
    assert s.dm.auto_save_state().count(NOT_SAVED) == 1
    refused.unlink()
    clock.at = T2
    result = s.dm.auto_save_state()
    assert not any(i.startswith("Not saved") for i in result)
    assert _backups(s.dm.data_dir) == [_sibling(destination, T2).name]
    assert {k: dict(vars(b)) for k, b in s.dm.backends.items()} == backends


def test_a3_save_provenance_parameters_are_unchanged(mu, monkeypatch):
    """The backup name reaches the backend only; the saved parameters are as before."""
    s = mu
    _BackupClock(monkeypatch, T0)
    _mu_saved_once(s)
    logged = []
    real = s.dm.log_tool_usage
    monkeypatch.setattr(
        s.dm, "log_tool_usage", lambda *a, **k: (logged.append(k), real(*a, **k))[1]
    )
    s.dm.auto_save_state()
    [saved] = [k for k in logged if k.get("tool_name") == "save_dataset"]
    assert saved["parameters"] == {
        "name": NAME,
        "path": str(s.dm.data_dir / f"{NAME}_autosave.h5ad"),
        "backend": "mudata",
    }


# ======================================================== A4: the data package


@pytest.fixture
def systmp(tmp_path, monkeypatch):
    """The system temp folder the package payload is built in (a known place)."""
    folder = tmp_path / "systmp"
    folder.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(folder))
    return folder


def _payload(systmp) -> Path:
    [build] = list(systmp.iterdir())
    return build / "modalities"


class _FixedNow(_real_datetime):
    @classmethod
    def now(cls, tz=None):
        return T0


ZIP = "lobster_analysis_package_20261028_090000.zip"


def _packages(folder: Path):
    return sorted(p.name for p in Path(folder).glob("lobster_analysis_package_*.zip"))


def _hostile(s):
    """A modality name whose payload file lands in the revision (3 levels up)."""
    fig, rev = s.rev.directory.parent.name, s.rev.directory.name
    return f"../../../ws/plots/{fig}/{rev}/x"


@pytest.mark.parametrize("order", ["safe_first", "hostile_first"])
@pytest.mark.usefixtures("any_name")
def test_a4_payload_name_escaping_into_a_revision_refuses_the_whole_package(
    s, systmp, order
):
    names = [NAME, _hostile(s)] if order == "safe_first" else [_hostile(s), NAME]
    for i, name in enumerate(names):
        _store(s, name, i)
    before = workspace_tree(s.ws)
    with pytest.raises(ManagedPathError):
        s.dm.create_data_package()
    assert workspace_tree(s.ws) == before and s.rev.unchanged()
    assert _packages(s.dm.exports_dir) == [] and list(systmp.iterdir()) == []


@pytest.mark.parametrize("where", ["revision", "new_in_figure", "detour"])
def test_a4_managed_output_folder_is_refused_before_anything(s, systmp, where):
    _store(s)
    output = {
        "revision": s.rev.directory,
        "new_in_figure": _figure_new(s),
        "detour": _detour(s, s.dm.exports_dir, s.dm.exports_dir / "pkg"),
    }[where]
    before = workspace_tree(s.ws)
    with pytest.raises(ManagedPathError):
        s.dm.create_data_package(output_dir=str(output))
    assert workspace_tree(s.ws) == before and s.rev.unchanged()
    assert not _figure_new(s).exists()


@pytest.mark.parametrize("content", ["modality", "figures_only"])
def test_a4_staging_folder_inside_a_revision_is_refused_before_anything(
    s, monkeypatch, content
):
    if content == "modality":
        _store(s)
    monkeypatch.setattr(tempfile, "tempdir", str(s.rev.directory))
    output = s.dm.exports_dir / "pkg"
    before = workspace_tree(s.ws)
    with pytest.raises(ManagedPathError):
        s.dm.create_data_package(output_dir=str(output))
    assert workspace_tree(s.ws) == before and s.rev.unchanged()


def test_a4_cold_temp_choice_in_a_revision_writes_nothing_there(s, monkeypatch):
    """No temp folder chosen yet and TMPDIR in a revision: nothing is probed there."""
    _store(s)
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setenv("TMPDIR", str(s.rev.directory))
    before = workspace_tree(s.ws)
    with pytest.raises(ManagedPathError):
        s.dm.create_data_package()
    assert workspace_tree(s.ws) == before and s.rev.unchanged()  # incl. folder times
    assert tempfile.tempdir is None


def test_a4_cold_temp_choice_follows_tmpdir_as_before(s, tmp_path, monkeypatch):
    _store(s)
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setenv("TMPDIR", str(staging))
    built = []
    real = tempfile.TemporaryDirectory

    def recording(*args, **kwargs):
        made = real(*args, **kwargs)
        built.append(Path(made.name).parent)
        return made

    monkeypatch.setattr(tempfile, "TemporaryDirectory", recording)
    assert Path(s.dm.create_data_package()).is_file()
    assert built == [staging] and list(staging.iterdir()) == []


def test_a4_staging_parent_is_the_folder_tempfile_would_choose(tmp_path, monkeypatch):
    """With no temp folder chosen yet, the parent is the first of tempfile's own
    candidates that is a writable folder (nothing is written to decide)."""
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.chdir(tmp_path)
    expected = next(
        c
        for c in tempfile._candidate_tempdir_list()
        if os.path.isdir(c) and os.access(c, os.W_OK | os.X_OK)
    )
    assert dm_module._package_staging_parent() == os.path.abspath(expected)
    assert tempfile.tempdir is None


def test_a4_package_is_staged_in_a_new_private_folder(s, systmp):
    """The parent holds only a new folder created for this package by tempfile
    (an unpredictable name, owner-only access), removed afterwards."""
    _store(s)
    seen = []

    def progress(message):
        for entry in systmp.iterdir():
            seen.append((entry.name, entry.is_dir(), entry.stat().st_mode & 0o777))

    assert Path(s.dm.create_data_package(progress_callback=progress)).is_file()
    names = {name for name, _, _ in seen}
    assert len(names) == 1 and {(d, m) for _, d, m in seen} == {(True, 0o700)}
    assert list(systmp.iterdir()) == []


def test_a4_package_file_alias_is_refused_before_anything(s, systmp, monkeypatch):
    monkeypatch.setattr(dm_module, "datetime", _FixedNow)
    _store(s)
    alias(s.dm.exports_dir / ZIP, _rev_file(s))
    before = workspace_tree(s.ws)
    with pytest.raises(ManagedPathError):
        s.dm.create_data_package()
    assert workspace_tree(s.ws) == before and s.rev.unchanged()
    assert list(systmp.iterdir()) == []


def _late(on_message, action):
    def progress(message):
        if message.startswith(on_message):
            action()

    return progress


def test_a4_package_file_alias_created_late_is_refused(s, systmp, monkeypatch):
    monkeypatch.setattr(dm_module, "datetime", _FixedNow)
    _store(s)
    link = s.dm.exports_dir / ZIP
    before = workspace_tree(s.ws)
    progress = _late("Creating final package", lambda: alias(link, _rev_file(s)))
    with pytest.raises(ManagedPathError):
        s.dm.create_data_package(progress_callback=progress)
    after = workspace_tree(s.ws)
    # The figure store's own view of the revision (trusted, written on export) only.
    trusted = _key(s, s.rev.directory / "renditions")
    for k in _changed(before, after) - {_key(s, link)}:
        if not k.startswith(trusted):
            assert before.get(k, ("?",))[0] == "dir" == after.get(k, ("?",))[0], k
    assert s.rev.unchanged() and list(systmp.iterdir()) == []


@pytest.mark.parametrize(
    "branch,file",
    [
        ("direct", "h5ad"),
        ("reuse", "h5ad"),
        ("direct", "metadata"),
        ("reuse", "folder"),
    ],
)
def test_a4_payload_file_alias_created_late_is_refused(s, systmp, branch, file):
    _store(s)
    if branch == "reuse":  # a clean modality with its autosave: the copy branch
        s.dm.auto_save_state()
        assert NAME not in s.dm._modality_dirty
    output = s.dm.exports_dir / "a" / "b" / "pkg"
    leaf = f"{NAME}_metadata.json" if file == "metadata" else f"{NAME}.h5ad"

    def plant():
        if file == "folder":  # a copy into it would write an unchecked name
            (_payload(systmp) / leaf).mkdir()
            alias(_payload(systmp) / leaf / f"{NAME}_autosave.h5ad", _rev_file(s))
        else:
            alias(_payload(systmp) / leaf, _rev_file(s))

    before = workspace_tree(s.ws)
    progress = _late("Saving modality", plant)
    with pytest.raises(ManagedPathError):
        s.dm.create_data_package(output_dir=str(output), progress_callback=progress)
    assert s.rev.unchanged() and list(systmp.iterdir()) == []
    assert not output.exists()  # created by this call and empty: removed
    assert (output.parent).is_dir()  # its new ancestors stay
    _no_plots_change(s, before, workspace_tree(s.ws))


@pytest.mark.parametrize("existing", ["empty", "with_package", "created_meanwhile"])
def test_a4_existing_output_folder_is_kept_on_a_late_refusal(
    s, systmp, monkeypatch, existing
):
    _store(s)
    output = s.dm.exports_dir / "pkg"
    if existing != "created_meanwhile":
        output.mkdir()
    if existing == "with_package":
        (output / "lobster_analysis_package_old.zip").write_bytes(b"old package")
    if existing == "created_meanwhile":  # someone else creates it just before mkdir
        real_mkdir = os.mkdir

        def mkdir(path, *args, **kwargs):
            if Path(path) == output and not output.exists():
                real_mkdir(path, *args, **kwargs)
            return real_mkdir(path, *args, **kwargs)

        monkeypatch.setattr(os, "mkdir", mkdir)
    progress = _late(
        "Saving modality",
        lambda: alias(_payload(systmp) / f"{NAME}.h5ad", _rev_file(s)),
    )
    kept = workspace_tree(output) if output.exists() else None
    with pytest.raises(ManagedPathError):
        s.dm.create_data_package(output_dir=str(output), progress_callback=progress)
    assert output.is_dir()
    if kept is not None:
        assert workspace_tree(output) == kept
    assert s.rev.unchanged()


def test_a4_ordinary_package_is_unchanged(s, systmp):
    _store(s)
    output = s.dm.exports_dir / "new" / "pkg"
    package = s.dm.create_data_package(output_dir=str(output))
    assert Path(package).parent == output and Path(package).is_file()
    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
    assert {
        "technical_summary.md",
        "README.md",
        f"modalities/{NAME}.h5ad",
        f"modalities/{NAME}_metadata.json",
    } <= names
    assert list(systmp.iterdir()) == []


def test_a4_ordinary_reused_package_is_unchanged(s, systmp):
    _store(s)
    s.dm.auto_save_state()
    package = s.dm.create_data_package()
    with zipfile.ZipFile(package) as archive:
        assert (
            archive.read(f"modalities/{NAME}.h5ad")
            == (s.dm.data_dir / f"{NAME}_autosave.h5ad").read_bytes()
        )


# ================================================== A6: the publication cache


def _cache(s, fmt):
    return s.dm.cache_publication_content(
        "PMID:12345678",
        {"source": "PMID:12345678", "methods_text": "m", "content": "c"},
        format=fmt,
    )


def _cache_file(s, fmt):
    import hashlib

    folder = s.dm.literature_cache_dir / "publications"
    if fmt == "markdown":
        return folder / "PMID_12345678.md"
    return folder / (
        hashlib.md5(b"PMID:12345678", usedforsecurity=False).hexdigest() + ".json"
    )


def _activities(s):
    return len(s.dm.provenance.activities) if s.dm.provenance else 0


@pytest.mark.parametrize("fmt", ["markdown", "json"])
def test_a6_cache_file_alias_into_a_revision_is_refused(s, fmt):
    alias(_cache_file(s, fmt), s.rev.directory / "cached.bin")  # dangling: a miss
    activities = _activities(s)
    before = workspace_tree(s.ws)
    with pytest.raises(ManagedPathError) as refused:
        _cache(s, fmt)
    _refusal_is_path_free(s, refused.value)
    assert workspace_tree(s.ws) == before and s.rev.unchanged()
    assert _activities(s) == activities  # no cache provenance record


def test_a6_cache_folder_alias_into_a_revision_is_refused(s):
    folder = s.dm.literature_cache_dir / "publications"
    folder.rmdir()
    alias(folder, s.rev.directory)
    before = workspace_tree(s.ws)
    with pytest.raises(ManagedPathError):
        _cache(s, "json")
    assert workspace_tree(s.ws) == before and s.rev.unchanged()


def test_a6_recursive_cache_folder_through_a_figure_folder_is_refused(s):
    s.dm.literature_cache_dir = _detour(s, s.ws, s.ws / "literature_cache")
    before = workspace_tree(s.ws)
    with pytest.raises(ManagedPathError):
        _cache(s, "json")
    _no_plots_change(s, before, workspace_tree(s.ws))
    assert not _figure_new(s).exists()


@pytest.mark.parametrize("fmt", ["markdown", "json"])
def test_a6_cache_file_alias_behind_a_missing_folder_creates_nothing(s, fmt):
    """``W/literature_cache/new/..``: ``new`` does not exist (creating the folder
    would create it) and the cache file there links into a revision. The file is
    refused before any folder is created."""
    link = _cache_file(s, fmt)  # in the existing ordinary publications folder
    alias(link, s.rev.directory / "cached.bin")
    s.dm.literature_cache_dir = s.ws / "literature_cache" / "new" / ".."
    activities = _activities(s)
    before = workspace_tree(s.ws)
    with pytest.raises(ManagedPathError) as refused:
        _cache(s, fmt)
    _refusal_is_path_free(s, refused.value)
    assert workspace_tree(s.ws) == before and s.rev.unchanged()
    assert not (s.ws / "literature_cache" / "new").exists()
    assert _activities(s) == activities


@pytest.mark.parametrize("fmt", ["markdown", "json"])
def test_a6_ordinary_cache_is_unchanged(s, fmt):
    activities = _activities(s)
    assert _cache(s, fmt) == _cache_file(s, fmt) and _cache_file(s, fmt).is_file()
    assert _activities(s) == activities + (1 if s.dm.provenance else 0)


def test_a6_unsupported_format_keeps_its_error(s):
    with pytest.raises(ValueError, match="Unsupported cache format"):
        _cache(s, "yaml")


# ------------------------- A6: the five content-access cache writes (deviation A-D2)

CACHE_LOG = (
    "Publication cache write skipped: the cache file is inside the managed figure "
    "folder."
)
ROUTES = {
    "non_scientific": "https://github.com/example-org/example-repo",
    "pmc": "PMID:12345678",
    "preprint": "10.1101/2024.01.01.123456",
    "webpage": "https://www.nature.com/articles/s41586-020-0000-0",
    "pdf": "https://example.org/files/paper.pdf",
}


@pytest.fixture
def providers(monkeypatch):
    """Only the four provider calls ContentAccessService makes are faked (each call is
    recorded). Everything after a provider returns is real: the five cache sites, the
    data manager's cache reader and writer, provenance. The network stays blocked."""
    from lobster.services.data_access.docling_service import DoclingService
    from lobster.tools.providers.biorxiv_medrxiv_provider import (
        BioRxivMedRxivProvider,
    )
    from lobster.tools.providers.pmc_provider import (
        PMCFullText,
        PMCNotAvailableError,
        PMCProvider,
    )
    from lobster.tools.providers.webpage_provider import WebpageProvider

    calls = []

    def full_text():
        return PMCFullText(
            pmc_id="PMC1",
            pmid=None,
            doi=None,
            title="T",
            abstract="A",
            full_text="Full text",
            methods_section="Methods",
            results_section="Results",
            discussion_section="Discussion",
            tables=[],
            figures=[],
            software_tools=[],
            github_repos=[],
        )

    def pmc(self, identifier, known_pmc_id=None):
        calls.append(("pmc", identifier))
        if identifier.startswith("10.1101/"):
            raise PMCNotAvailableError("not in PMC")
        return full_text()

    def webpage(self, url, keywords=None, max_paragraphs=100, **kwargs):
        calls.append(("webpage", url))
        if url.endswith(".pdf"):
            raise RuntimeError("403 Client Error: Forbidden")
        return {"content": "Page", "methods_text": "M", "source_type": "webpage"}

    def pdf(self, source, keywords=None, max_paragraphs=50, **kwargs):
        calls.append(("pdf", source))
        return {"methods_text": "pdf", "provenance": {"conversion_seconds": 0.5}}

    def preprint(self, doi, server=None):
        calls.append(("preprint", doi, server))
        return full_text()

    monkeypatch.setattr(PMCProvider, "extract_full_text", pmc)
    monkeypatch.setattr(WebpageProvider, "extract_with_full_metadata", webpage)
    monkeypatch.setattr(DoclingService, "extract_methods_section", pdf)
    monkeypatch.setattr(BioRxivMedRxivProvider, "get_full_text", preprint)
    return calls


def _content_route(tmp_path, monkeypatch, caplog, calls, source, key=None):
    from lobster.services.data_access.content_access_service import (
        ContentAccessService,
    )

    tmp_path.mkdir()
    s = _session(tmp_path, monkeypatch)
    service = ContentAccessService(s.dm)
    folder = s.dm.literature_cache_dir / "publications"
    if key is not None:  # dangling, into the revision: a cache miss at lookup
        alias(folder / key, s.rev.directory / "cached.bin")
    lookups, tools = [], []
    read, log = s.dm.get_cached_publication, s.dm.log_tool_usage

    def lookup(identifier):  # the real reader, its answer recorded
        found = read(identifier)
        lookups.append(found)
        return found

    def record(*args, **kwargs):
        tools.append(kwargs.get("tool_name", args[0] if args else None))
        return log(*args, **kwargs)

    monkeypatch.setattr(s.dm, "get_cached_publication", lookup)
    monkeypatch.setattr(s.dm, "log_tool_usage", record)
    names = set(os.listdir(folder))
    calls.clear()
    first = len(caplog.records)
    with caplog.at_level(logging.INFO):
        result = service.get_full_content(source)
    warned = [r.getMessage() for r in caplog.records[first:] if r.levelno >= 30]
    return SimpleNamespace(
        s=s,
        result={k: v for k, v in result.items() if k != "extraction_time"},
        calls=list(calls),
        lookups=lookups,
        tools=tools,
        warned=warned,
        written=sorted(set(os.listdir(folder)) - names),
    )


@pytest.mark.parametrize("route", list(ROUTES))
def test_a6_content_cache_refusal_returns_the_same_content_with_no_other_route(
    tmp_path, monkeypatch, caplog, providers, route
):
    source = ROUTES[route]
    ordinary = _content_route(
        tmp_path / "ordinary", monkeypatch, caplog, providers, source
    )
    [key] = ordinary.written  # the route's actual cache file
    assert "error" not in ordinary.result and ordinary.lookups == [None]
    assert "cache_publication_content" in ordinary.tools
    refused = _content_route(
        tmp_path / "refused", monkeypatch, caplog, providers, source, key=key
    )
    s = refused.s
    assert refused.lookups == [None]  # a miss: the route ran (also the 5th route)
    assert refused.calls == ordinary.calls  # no retry, no other route
    assert refused.result == ordinary.result  # the same content is returned
    assert refused.tools == [
        t for t in ordinary.tools if t != "cache_publication_content"
    ]
    assert [m for m in refused.warned if m not in ordinary.warned] == [CACHE_LOG]
    assert not (s.rev.directory / "cached.bin").exists() and s.rev.unchanged()
    assert _named([r for r in caplog.records if r.levelno >= 30], str(s.ws)) == []


# ============================================================ public names


def test_new_names_after_behaviour():
    """Imported last: the behavioural tests above are RED on their own outcomes."""
    from lobster.core.runtime.data_manager import AUTOSAVE_NOT_SAVED_PREFIX

    assert AUTOSAVE_NOT_SAVED_PREFIX == "Not saved: "
    assert NOT_SAVED.startswith(AUTOSAVE_NOT_SAVED_PREFIX)
