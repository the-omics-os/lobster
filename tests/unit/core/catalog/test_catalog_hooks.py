"""The five catalog hook sites in DataManagerV2 and AgentClient.

A synchronous RecordingCatalog stands in for the engine, so each test reads the
exact call sequence a scripted session produces.
"""

import hashlib
import json
import os
import stat
from pathlib import Path
from unittest.mock import Mock, patch

import anndata as ad
import numpy as np
import pytest

from lobster.core.catalog import factory
from lobster.core.catalog.null import NullCatalog
from lobster.core.catalog.port import SessionInfo
from lobster.core.runtime import data_manager as data_manager_module
from lobster.core.runtime.data_manager import DataManagerV2

pytestmark = pytest.mark.unit


def _adata(seed=0, n_obs=30, n_vars=12):
    rng = np.random.default_rng(seed)
    a = ad.AnnData(rng.poisson(1.0, size=(n_obs, n_vars)).astype(np.float32))
    a.obs_names = [f"c{i}" for i in range(n_obs)]
    a.var_names = [f"g{i}" for i in range(n_vars)]
    return a


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def ws(tmp_path):
    path = tmp_path / "ws"
    path.mkdir()
    return path


@pytest.fixture
def dm(ws):
    return DataManagerV2(workspace_path=ws)


@pytest.fixture
def engine(recording_catalog_cls):
    return recording_catalog_cls()


@pytest.fixture
def recorded_dm(dm, engine):
    dm.attach_catalog(engine)
    return dm


class SpyNull(NullCatalog):
    def __init__(self):
        super().__init__()
        self.touched = []

    def __getattribute__(self, item):
        if item in {
            "keep_path",
            "register_artifact",
            "link_input",
            "note_unrecorded",
            "register_transform",
            "begin_run",
            "end_run",
        }:
            object.__getattribute__(self, "touched").append(item)
        return object.__getattribute__(self, item)


# -- default: nothing changes ------------------------------------------------------


def test_default_is_null_and_saves_land_where_aimed(dm, ws):
    assert isinstance(dm.catalog, NullCatalog)
    spy = SpyNull()
    dm.attach_catalog(spy)
    dm.store_modality("pbmc", _adata())
    out = dm.save_modality("pbmc", "pbmc.h5ad")
    assert Path(out) == dm.data_dir / "pbmc.h5ad" and Path(out).is_file()
    assert spy.touched == []
    assert not (ws / ".lobster" / "catalog.lock").exists()


def test_attach_none_restores_null(recorded_dm):
    recorded_dm.attach_catalog(None)
    assert isinstance(recorded_dm.catalog, NullCatalog)


# -- save_modality -------------------------------------------------------------------


def test_explicit_save_is_registered_in_place(recorded_dm, engine):
    dm = recorded_dm
    dm.store_modality(
        "pbmc_filtered", _adata(), step_summary="Filtered 3 low-quality cells"
    )
    out = Path(dm.save_modality("pbmc_filtered", "pbmc_filtered.h5ad"))

    assert out == dm.data_dir / "pbmc_filtered.h5ad" and out.is_file()
    assert engine.ops() == ["keep_path", "register_artifact"]
    reg = engine.calls[1][1]
    assert reg["path"] == out
    assert reg["key"] == "data/pbmc_filtered.h5ad"
    assert (reg["kind"], reg["otype"]) == ("dataset", "AnnData")
    assert reg["description"] == "Filtered 3 low-quality cells"
    assert reg["meta"]["modality_name"] == "pbmc_filtered"
    assert reg["meta"]["shape"] == [30, 12]


def _keep(dm):
    return dm.workspace_path / ".lamindb" / "versions" / "data" / "x.h5ad" / "v1.h5ad"


def test_kept_version_keeps_the_old_bytes_and_the_name_takes_the_new(
    recorded_dm, engine
):
    dm = recorded_dm
    dm.store_modality("x", _adata(seed=1))
    v1 = Path(dm.save_modality("x", "x.h5ad"))
    v1_sha = _sha(v1)

    dm.store_modality("x", _adata(seed=2))
    engine.next_keep_path = _keep(dm)
    out = Path(dm.save_modality("x", "x.h5ad"))

    assert out == v1 and _sha(out) != v1_sha
    assert _sha(_keep(dm)) == v1_sha, "the recorded version was not kept"
    reg = engine.calls[-1][1]
    assert reg["key"] == "data/x.h5ad" and reg["path"] == out
    saves = [a for a in dm.provenance.activities if a.get("type") == "save_dataset"]
    assert saves[-1]["parameters"]["path"] == str(out)
    assert sorted(p.name for p in dm.data_dir.iterdir()) == ["x.h5ad"]


def test_the_file_is_never_moved_so_its_mode_survives(recorded_dm, engine):
    dm = recorded_dm
    dm.store_modality("x", _adata(seed=1))
    v1 = Path(dm.save_modality("x", "x.h5ad"))
    v1.chmod(0o640)
    dm.store_modality("x", _adata(seed=2))
    engine.next_keep_path = _keep(dm)
    dm.save_modality("x", "x.h5ad")
    assert stat.S_IMODE(v1.stat().st_mode) == 0o640


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores modes"
)
def test_a_read_only_file_is_refused_as_without_catalog(recorded_dm, engine):
    dm = recorded_dm
    dm.store_modality("x", _adata(seed=1))
    v1 = Path(dm.save_modality("x", "x.h5ad"))
    v1_sha = _sha(v1)
    v1.chmod(0o444)
    dm.store_modality("x", _adata(seed=2))
    engine.next_keep_path = _keep(dm)
    try:
        with pytest.raises(ValueError, match="read-only"):
            dm.save_modality("x", "x.h5ad")
    finally:
        v1.chmod(0o644)
    assert _sha(v1) == v1_sha and not _keep(dm).exists()


def test_a_failed_write_drops_the_kept_name(recorded_dm, engine, monkeypatch):
    dm = recorded_dm
    dm.store_modality("x", _adata(seed=1))
    v1 = Path(dm.save_modality("x", "x.h5ad"))
    v1_sha = _sha(v1)
    engine.next_keep_path = _keep(dm)
    backend = dm.backends[dm.default_backend]
    monkeypatch.setattr(
        backend, "save", Mock(side_effect=RuntimeError("disk full")), raising=False
    )
    with pytest.raises(RuntimeError, match="disk full"):
        dm.save_modality("x", "x.h5ad")
    assert _sha(v1) == v1_sha and not _keep(dm).exists()
    assert engine.ops()[-1] == "keep_path"  # nothing registered


def test_a_version_that_cannot_be_kept_does_not_stop_the_save(
    recorded_dm, engine, monkeypatch, caplog
):
    dm = recorded_dm
    dm.store_modality("x", _adata(seed=1))
    v1 = Path(dm.save_modality("x", "x.h5ad"))
    engine.next_keep_path = _keep(dm)

    def refuse(*args, **kwargs):
        raise PermissionError("no")

    monkeypatch.setattr(data_manager_module, "_keep_bytes", refuse)
    dm.store_modality("x", _adata(seed=2))
    with caplog.at_level("WARNING", logger=data_manager_module.__name__):
        out = Path(dm.save_modality("x", "x.h5ad"))
    assert out == v1 and not _keep(dm).exists()
    assert any("could not be kept" in r.getMessage() for r in caplog.records)
    assert engine.ops()[-1] == "register_artifact"


@pytest.mark.parametrize("where", ["outside", "existing", "figures", "garbage"])
def test_unusable_keep_answers_are_ignored(recorded_dm, engine, tmp_path, where):
    dm = recorded_dm
    dm.store_modality("x", _adata(seed=1))
    v1 = Path(dm.save_modality("x", "x.h5ad"))
    existing = dm.workspace_path / "notes.txt"
    existing.write_text("mine")
    engine.next_keep_path = {
        "outside": tmp_path / "elsewhere" / "x.h5ad",
        "existing": existing,
        "figures": dm.workspace_path / "plots" / "fig1" / "x.h5ad",
        "garbage": 42,
    }[where]
    dm.store_modality("x", _adata(seed=2))
    assert Path(dm.save_modality("x", "x.h5ad")) == v1
    assert existing.read_text() == "mine"
    assert not (tmp_path / "elsewhere").exists()
    assert not (dm.workspace_path / "plots" / "fig1" / "x.h5ad").exists()


def test_autosave_is_not_recorded(recorded_dm, engine):
    dm = recorded_dm
    dm.store_modality("pbmc", _adata())
    dm.auto_save_state(force=True)
    assert (dm.data_dir / "pbmc_autosave.h5ad").is_file()
    assert engine.ops() == []


def test_save_outside_the_workspace_is_not_recorded(recorded_dm, engine, tmp_path):
    dm = recorded_dm
    dm.store_modality("x", _adata())
    out = dm.save_modality("x", tmp_path / "outside" / "x.h5ad")
    assert Path(out).is_file() and engine.ops() == []


@pytest.mark.parametrize("catalog_on", [False, True])
def test_a_workspace_path_outside_data_records_the_file_written(
    dm, engine, ws, catalog_on
):
    """The backend writes <ws>/results/x.h5ad under data/: that file is recorded.

    The return value keeps the caller's spelling, as before (test_dm_save_guard).
    """
    if catalog_on:
        dm.attach_catalog(engine)
    dm.store_modality("x", _adata())
    spelled = ws / "results" / "x.h5ad"
    assert dm.save_modality("x", spelled) == str(spelled)
    written = dm.data_dir / "results" / "x.h5ad"
    assert written.is_file() and not (ws / "results").exists()
    if catalog_on:
        assert engine.ops() == ["keep_path", "register_artifact"]
        reg = engine.calls[1][1]
        assert reg["path"] == written and reg["key"] == "data/results/x.h5ad"


def test_non_local_backend_is_not_recorded(recorded_dm, engine):
    dm = recorded_dm
    remote = Mock()
    remote.save = Mock()
    dm.register_backend("remote", remote)
    dm.store_modality("x", _adata())
    dm.save_modality("x", "x.h5ad", backend="remote")
    remote.save.assert_called_once()
    assert engine.ops() == []


def test_engine_errors_never_fail_a_save(recorded_dm, engine):
    dm = recorded_dm
    engine.raise_on["keep_path"] = RuntimeError("down")
    engine.raise_on["register_artifact"] = RuntimeError("down")
    dm.store_modality("x", _adata())
    out = Path(dm.save_modality("x", "x.h5ad"))
    assert out == dm.data_dir / "x.h5ad" and out.is_file()


# -- load_modality ---------------------------------------------------------------------


def test_load_from_file_links_the_input(recorded_dm, engine):
    dm = recorded_dm
    src = dm.data_dir / "input.h5ad"
    src.parent.mkdir(parents=True, exist_ok=True)
    _adata().write_h5ad(src)
    dm.load_modality(
        "input", src, adapter="transcriptomics_single_cell", validate=False
    )
    assert engine.calls[-1] == ("link_input", src.resolve())


def test_load_from_memory_links_nothing(recorded_dm, engine):
    recorded_dm.load_modality(
        "mem", _adata(), adapter="transcriptomics_single_cell", validate=False
    )
    assert "link_input" not in engine.ops()


# -- create_data_package / export_notebook ------------------------------------------------------


def test_data_package_is_registered(recorded_dm, engine):
    dm = recorded_dm
    dm.store_modality("x", _adata())

    def fake_write(staging, zip_filename, *rest):
        Path(zip_filename).parent.mkdir(parents=True, exist_ok=True)
        Path(zip_filename).write_bytes(b"PK")
        return str(zip_filename)

    with patch.object(
        DataManagerV2, "_write_data_package", side_effect=fake_write, autospec=False
    ):
        out = dm.create_data_package()

    op, reg = engine.calls[-1]
    assert op == "register_artifact"
    assert reg["path"] == Path(out) and reg["key"].startswith(
        "exports/lobster_analysis_package_"
    )
    assert reg["otype"] is None and reg["kind"] == "dataset"


def test_exported_notebook_is_a_transform(recorded_dm, engine):
    dm = recorded_dm
    nb = dm.workspace_path / "notebooks" / "qc.ipynb"
    nb.parent.mkdir(parents=True, exist_ok=True)
    nb.write_text(json.dumps({"cells": [], "nbformat": 4}))
    dm.log_tool_usage("qc", {}, description="d")
    dm._notebook_exporter = Mock(export=Mock(return_value=nb))

    assert dm.export_notebook("qc", description="QC workflow") == nb
    op, reg = engine.calls[-1]
    assert op == "register_transform"
    assert reg == {
        "key": "notebooks/qc.ipynb",
        "kind": "notebook",
        "source": nb.read_text(),
        "description": "QC workflow",
    }


# -- AgentClient: one run per session ----------------------------------------------------


@pytest.fixture
def client_graph():
    graph, metadata = Mock(), Mock()
    metadata.to_dict.return_value = {}
    with patch(
        "lobster.core.client.create_bioinformatics_graph",
        return_value=(graph, metadata),
    ):
        yield


def test_client_attaches_the_catalog_and_begins_the_session_run(
    ws, engine, client_graph
):
    from lobster.core.client import AgentClient

    with patch.object(factory, "get_catalog", return_value=engine) as get_catalog:
        client = AgentClient(workspace_path=ws, session_id="session_20261108_100000")

    get_catalog.assert_called_once_with(ws)
    assert client.data_manager.catalog is engine
    op, session = engine.calls[0]
    assert op == "begin_run" and isinstance(session, SessionInfo)
    assert session.session_id == "session_20261108_100000"
    assert session.workspace_path == ws
    assert (
        session.session_dir == ws / ".lobster" / "sessions" / "session_20261108_100000"
    )
    assert session.kind == "session"


def test_loading_a_session_begins_its_run(ws, engine, client_graph, tmp_path):
    from lobster.core.client import AgentClient

    saved = tmp_path / "saved_session.json"
    saved.write_text(
        json.dumps({"session_id": "session_20261101_090000", "conversation": []})
    )
    with patch.object(factory, "get_catalog", return_value=engine):
        client = AgentClient(workspace_path=ws, session_id="session_20261108_100000")
        client.load_session(saved)

    begins = [payload.session_id for op, payload in engine.calls if op == "begin_run"]
    assert begins == ["session_20261108_100000", "session_20261101_090000"]


def test_client_survives_a_broken_catalog(ws, client_graph):
    from lobster.core.client import AgentClient

    with patch.object(factory, "get_catalog", side_effect=RuntimeError("boom")):
        client = AgentClient(workspace_path=ws)
    assert isinstance(client.data_manager.catalog, NullCatalog)


def test_client_default_is_null(ws, client_graph, monkeypatch):
    from lobster.core.client import AgentClient

    monkeypatch.delenv("LOBSTER_LAMIN", raising=False)
    monkeypatch.setattr(
        factory, "_global_config_path", lambda: ws / "no-providers.json"
    )
    client = AgentClient(workspace_path=ws)
    assert isinstance(client.data_manager.catalog, NullCatalog)


# -- save_modality through an attached modality backend ----------------------------------


def _modality_backend():
    backend = Mock()
    saves = iter(range(1, 100))
    backend.export_h5ad.side_effect = lambda name, path: (
        Path(path).parent.mkdir(parents=True, exist_ok=True),
        Path(path).write_bytes(f"{name}-{next(saves)}".encode()),  # in place
    )
    return backend


def test_modality_backend_branch_is_recorded_and_keeps_a_copy(recorded_dm, engine):
    dm = recorded_dm
    dm.store_modality("x", _adata())
    dm._modality_backend = _modality_backend()
    first = Path(dm.save_modality("x", "x.h5ad"))
    assert first == dm.data_dir / "x.h5ad"
    engine.next_keep_path = _keep(dm)
    second = Path(dm.save_modality("x", "x.h5ad"))
    assert second == first and second.read_bytes() == b"x-2"
    assert _keep(dm).read_bytes() == b"x-1"  # a copy, not a second name
    assert os.stat(_keep(dm)).st_ino != os.stat(second).st_ino
    regs = [p for op, p in engine.calls if op == "register_artifact"]
    assert [(r["key"], r["path"]) for r in regs] == [
        ("data/x.h5ad", first),
        ("data/x.h5ad", second),
    ]


def test_modality_backend_branch_default_is_untouched(dm):
    dm.store_modality("x", _adata())
    dm._modality_backend = _modality_backend()
    spy = SpyNull()
    dm.attach_catalog(spy)
    assert Path(dm.save_modality("x", "x.h5ad")) == dm.data_dir / "x.h5ad"
    assert spy.touched == []


def test_a_writer_that_rewrites_in_place_gets_a_copy(recorded_dm, engine):
    """Only a backend that declares ``replaces_file`` gets a second name."""
    from lobster.core.backends.h5ad_backend import H5ADBackend

    class InPlace(H5ADBackend):
        replaces_file = False

        def save(self, adata, path, **kwargs):
            with open(self._resolve_path(path), "wb") as fh:  # same inode
                fh.write(f"in place {adata.X.sum():.3f}".encode())

    dm = recorded_dm
    dm.register_backend("inplace", InPlace(base_path=dm.data_dir))
    dm.store_modality("x", _adata(seed=1))
    v1 = Path(dm.save_modality("x", "x.h5ad", backend="inplace"))
    v1_bytes = v1.read_bytes()
    engine.next_keep_path = _keep(dm)
    dm.store_modality("x", _adata(seed=2))
    dm.save_modality("x", "x.h5ad", backend="inplace")
    assert v1.read_bytes() != v1_bytes
    assert _keep(dm).read_bytes() == v1_bytes, "the kept version changed with the file"


def test_the_native_h5ad_writer_replaces_the_file(recorded_dm, engine):
    """The hard link is safe only because this writer never writes in place."""
    from lobster.core.backends.h5ad_backend import H5ADBackend

    assert H5ADBackend.replaces_file is True
    dm = recorded_dm
    dm.store_modality("x", _adata(seed=1))
    v1 = Path(dm.save_modality("x", "x.h5ad"))
    inode = os.stat(v1).st_ino
    engine.next_keep_path = _keep(dm)
    dm.store_modality("x", _adata(seed=2))
    dm.save_modality("x", "x.h5ad")
    assert os.stat(_keep(dm)).st_ino == inode != os.stat(v1).st_ino
