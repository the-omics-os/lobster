"""A save never overwrites a file this session loaded (a recorded input).

Loaded means loaded by ``load_modality``, auto-loaded from ``data/`` when the
session starts, or restored. Any spelling of the file counts (a link, '..').
Auto-save may update a modality's own ``*_autosave.h5ad`` snapshot; any other
loaded target is skipped with a warning, never overwritten, never raised.
"""

import hashlib
import logging

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.core.runtime.data_manager import DataManagerV2, RecordedInputError

MESSAGE = (
    "data/x.h5ad is a file this session loaded (as modality 'x'); saving 'x' "
    "there would overwrite it. Save under a new modality name."
)


def _adata(shift=0.0):
    rng = np.random.default_rng(0)
    return ad.AnnData(
        X=(rng.poisson(3.0, size=(6, 5)) + shift).astype(np.float32),
        obs=pd.DataFrame(index=[f"s{i}" for i in range(6)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
    )


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def ws(tmp_path):
    """A workspace whose data/ already holds x.h5ad (a resumed session's file)."""
    (tmp_path / "ws" / "data").mkdir(parents=True)
    _adata().write_h5ad(tmp_path / "ws" / "data" / "x.h5ad")
    return tmp_path / "ws"


def test_an_auto_loaded_file_is_never_saved_over(ws):
    dm = DataManagerV2(workspace_path=ws)
    assert "x" in dm.list_modalities()
    target = dm.data_dir / "x.h5ad"
    before = _sha(target)
    dm.modalities["x"] = _adata(shift=1.0)  # modified in the session
    with pytest.raises(RecordedInputError) as caught:
        dm.save_modality("x", "x.h5ad")
    assert str(caught.value) == MESSAGE
    assert _sha(target) == before


def test_a_restart_protects_the_same_file(ws):
    DataManagerV2(workspace_path=ws)
    dm = DataManagerV2(workspace_path=ws)
    with pytest.raises(RecordedInputError):
        dm.save_modality("x", "x.h5ad")


@pytest.mark.parametrize("spelling", ["link", "dotdot", "absolute"])
def test_any_spelling_of_a_loaded_file_is_caught(ws, spelling):
    dm = DataManagerV2(workspace_path=ws)
    target = dm.data_dir / "x.h5ad"
    before = _sha(target)
    dm.modalities["y"] = _adata(shift=2.0)
    if spelling == "link":
        (dm.data_dir / "alias.h5ad").symlink_to(target)
        path = "alias.h5ad"
    elif spelling == "dotdot":
        (dm.data_dir / "sub").mkdir()
        path = "sub/../x.h5ad"
    else:
        path = str(target)
    with pytest.raises(RecordedInputError):
        dm.save_modality("y", path)
    assert _sha(target) == before


def test_a_file_loaded_by_load_modality_is_protected(tmp_path):
    source = tmp_path / "in.h5ad"
    _adata().write_h5ad(source)
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.load_modality("m", str(source), adapter="transcriptomics_bulk", validate=False)
    with pytest.raises(RecordedInputError):
        dm.save_modality("m", str(source))


def test_a_file_never_loaded_can_be_saved_again(tmp_path):
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.modalities["z"] = _adata()
    dm.save_modality("z", "z.h5ad")
    dm.modalities["z"] = _adata(shift=1.0)
    dm.save_modality("z", "z.h5ad")
    assert ad.read_h5ad(dm.data_dir / "z.h5ad").X[0, 0] == _adata(1.0).X[0, 0]


def test_the_legacy_store_auto_save_is_refused(ws):
    dm = DataManagerV2(workspace_path=ws)
    before = _sha(dm.data_dir / "x.h5ad")
    with pytest.raises(RecordedInputError):
        dm.store_modality(name="x", adata=_adata(shift=3.0), auto_save=True)
    assert _sha(dm.data_dir / "x.h5ad") == before


def _resumed_with_snapshot(tmp_path, snapshot_shift=5.0):
    """A workspace a first session left: x.h5ad, and x_autosave.h5ad holding x as
    that session last changed it. Restarted on, it auto-loads both."""
    data = tmp_path / "ws" / "data"
    data.mkdir(parents=True)
    _adata().write_h5ad(data / "x.h5ad")
    _adata(shift=snapshot_shift).write_h5ad(data / "x_autosave.h5ad")
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    assert {"x", "x_autosave"} <= set(dm.list_modalities())
    return dm, data / "x_autosave.h5ad"


def _first_value(path):
    return float(ad.read_h5ad(path).X[0, 0])


def test_resume_edit_exit_restart_keeps_the_edit(tmp_path):
    """Auto-save may update the snapshot a resumed session loaded: its own one."""
    dm, snapshot = _resumed_with_snapshot(tmp_path)
    dm.store_modality(name="x_autosave", adata=_adata(shift=7.0), step_summary="edit")
    saved = dm.auto_save_state()
    assert not any(i.startswith("Not saved") for i in saved), saved
    restarted = DataManagerV2(workspace_path=tmp_path / "ws")
    assert float(restarted.get_modality("x_autosave").X[0, 0]) == _first_value(snapshot)
    assert _first_value(snapshot) == float(_adata(7.0).X[0, 0])


def test_an_edit_of_x_reaches_its_own_snapshot(tmp_path):
    """x's own snapshot is x_autosave.h5ad, even while it is loaded as x_autosave."""
    dm, snapshot = _resumed_with_snapshot(tmp_path)
    dm.store_modality(name="x", adata=_adata(shift=8.0), step_summary="edit")
    dm.auto_save_state()
    assert _first_value(snapshot) == float(_adata(8.0).X[0, 0])


def test_an_unchanged_x_never_writes_over_the_loaded_snapshot(tmp_path, caplog):
    """Resume, change nothing, exit: the older x.h5ad does not replace the newer
    snapshot (it did: x was saved over x_autosave.h5ad at every exit)."""
    dm, snapshot = _resumed_with_snapshot(tmp_path)
    before = _sha(snapshot)
    with caplog.at_level(logging.INFO):
        saved = dm.auto_save_state()
    assert _sha(snapshot) == before
    assert not any(i.startswith("Not saved") for i in saved), saved
    assert (
        "Autosave skipped 'x': unchanged; x_autosave.h5ad is loaded as 'x_autosave'"
        in [r.getMessage() for r in caplog.records]
    )


def test_autosave_never_writes_another_loaded_file(ws, monkeypatch, caplog):
    """Were an autosave target ever a loaded file that is not the modality's own
    snapshot: skipped with a warning and Not saved, never written, never raised."""
    dm = DataManagerV2(workspace_path=ws)
    before = _sha(dm.data_dir / "x.h5ad")
    dm.modalities["y"] = _adata(shift=4.0)
    dm._modality_dirty.add("y")
    real = dm.save_modality
    monkeypatch.setattr(
        dm, "save_modality", lambda name, path, **kw: real(name, "x.h5ad", **kw)
    )
    with caplog.at_level(logging.WARNING):
        saved = dm.auto_save_state()
    assert _sha(dm.data_dir / "x.h5ad") == before
    assert "Not saved: Modality 'y': its file is a loaded input" in saved, saved
    assert any(
        r.getMessage().startswith("Autosave skipped 'y': data/x.h5ad is a file")
        for r in caplog.records
    )


def test_a_tool_reply_names_the_refusal(ws):
    """A resumed session imports into the name of a loaded file: what the agent
    reads, and the file stays as it was."""
    proteomics = pytest.importorskip("lobster.agents.proteomics.proteomics_expert")
    from lobster.testing.provenance_replay import factory_tools

    rng = np.random.default_rng(1)
    matrix = pd.DataFrame(
        rng.random((40, 6)) * 1e6,
        index=[f"P{i:05d}" for i in range(40)],
        columns=[f"s{i}" for i in range(6)],
    )
    matrix.index.name = "protein"
    matrix.to_csv(ws.parent / "expr.csv")
    dm = DataManagerV2(workspace_path=ws)
    before = _sha(dm.data_dir / "x.h5ad")
    tools = factory_tools(proteomics.proteomics_expert, proteomics.__name__, dm)
    reply = tools["import_proteomics_data"].invoke(
        {"file_path": str(ws.parent / "expr.csv"), "modality_name": "x"}
    )
    assert reply == "Error importing generic expression matrix: " + MESSAGE
    assert _sha(dm.data_dir / "x.h5ad") == before
