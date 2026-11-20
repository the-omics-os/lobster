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
    loaded = dm.get_modality("x")
    with pytest.raises(RecordedInputError):
        dm.store_modality(name="x", adata=_adata(shift=3.0), auto_save=True)
    assert _sha(dm.data_dir / "x.h5ad") == before
    assert dm.get_modality("x") is loaded  # memory unchanged too


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


def _scrna():
    rng = np.random.default_rng(0)
    return ad.AnnData(
        X=rng.poisson(5, (300, 400)).astype(np.float32),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(300)]),
        var=pd.DataFrame(index=[f"G{i}" for i in range(400)]),
    )


def _proteins():
    rng = np.random.default_rng(2)
    return ad.AnnData(
        X=(rng.random((6, 40)) * 1e6 + 1e3).astype(np.float32),
        obs=pd.DataFrame(index=[f"s{i}" for i in range(6)]),
        var=pd.DataFrame(index=[f"P{i:05d}" for i in range(40)]),
    )


def _clustered():
    adata = _scrna()
    adata.obs["leiden"] = pd.Categorical([str(i % 3) for i in range(300)])
    return adata


RERUNS = {
    "mark_clusters_as_debris": (
        "lobster.agents.transcriptomics.annotation_expert",
        "annotation_expert",
        _clustered,
        {"modality_name": "x", "debris_clusters": ["0"], "cluster_key": "leiden"},
        {"debris_clusters": ["1"]},
        "x_debris_marked",
    ),
    "filter_and_normalize": (
        "lobster.agents.transcriptomics.transcriptomics_expert",
        "transcriptomics_expert",
        _scrna,
        {"modality_name": "x", "min_genes_per_cell": 10, "min_cells_per_gene": 3},
        {"min_genes_per_cell": 20},
        "x_filtered_normalized",
    ),
    "normalize_proteomics_data": (
        "lobster.agents.proteomics.proteomics_expert",
        "proteomics_expert",
        _proteins,
        {"modality_name": "x", "normalization_method": "median"},
        {"normalization_method": "quantile"},
        "x_normalized",
    ),
}


@pytest.mark.parametrize("tool", list(RERUNS))
def test_a_rerun_in_a_resumed_session_takes_a_new_result_name(tmp_path, tool):
    """Session 1 saves <x>_<step>.h5ad; restarted, the same step is rerun: the
    new result is <x>_<step>_2, named in the reply; the loaded result, in
    memory and on disk, is as it was."""
    module_name, factory, data, args, changed, result = RERUNS[tool]
    module = pytest.importorskip(module_name)
    from lobster.testing.provenance_replay import factory_tools

    (tmp_path / "ws" / "data").mkdir(parents=True)
    data().write_h5ad(tmp_path / "ws" / "data" / "x.h5ad")
    first = DataManagerV2(workspace_path=tmp_path / "ws")
    reply = factory_tools(getattr(module, factory), module_name, first)[tool].invoke(
        args
    )
    saved = first.data_dir / f"{result}.h5ad"
    assert saved.is_file(), reply

    resumed = DataManagerV2(workspace_path=tmp_path / "ws")
    before = _sha(saved)
    loaded = resumed.get_modality(result).copy()
    tools = factory_tools(getattr(module, factory), module_name, resumed)
    reply = tools[tool].invoke({**args, **changed})

    assert f"{result}_2" in reply, reply
    assert "Error" not in reply
    assert (resumed.data_dir / f"{result}_2.h5ad").is_file()
    assert _sha(saved) == before
    np.testing.assert_array_equal(resumed.get_modality(result).X, loaded.X)


def test_an_import_refused_leaves_the_loaded_modality_and_provenance(ws):
    """An import into the name of a loaded file is refused before anything
    changes: the modality in memory is the loaded one, no step is logged."""
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
    loaded = dm.get_modality("x").copy()
    activities = len(dm.provenance.activities)
    tools = factory_tools(proteomics.proteomics_expert, proteomics.__name__, dm)
    reply = tools["import_proteomics_data"].invoke(
        {"file_path": str(ws.parent / "expr.csv"), "modality_name": "x"}
    )
    assert reply.endswith(MESSAGE), reply
    assert dm.get_modality("x").shape == loaded.shape
    np.testing.assert_array_equal(dm.get_modality("x").X, loaded.X)
    assert len(dm.provenance.activities) == activities


def _log_proteins(n_samples=12, n_proteins=40, prefix="P"):
    rng = np.random.default_rng(3)
    return ad.AnnData(
        X=rng.normal(20.0, 1.0, (n_samples, n_proteins)).astype(np.float32),
        obs=pd.DataFrame(index=[f"s{i}" for i in range(n_samples)]),
        var=pd.DataFrame(index=[f"{prefix}{i}" for i in range(n_proteins)]),
    )


def _batched():
    adata = _log_proteins()
    adata.obs["batch"] = ["b1"] * 6 + ["b2"] * 6
    return {"x": adata}


def _peptides():
    adata = _log_proteins(n_proteins=60, prefix="pep")
    adata.var["protein_id"] = [f"P{i // 3}" for i in range(60)]
    return {"x": adata}


def _ptm_sites():
    proteins = _log_proteins(prefix="GENE")
    sites = _log_proteins(n_proteins=30, prefix="GENE")
    sites.var_names = [f"GENE{i}_S{i + 10}" for i in range(30)]
    return {"x": sites, "prot": proteins}


def _bridged():
    adata = _log_proteins()
    adata.obs["plate_id"] = ["p1"] * 6 + ["p2"] * 6
    adata.obs["is_bridge"] = [i % 6 < 2 for i in range(12)]
    return {"x": adata}


# tool -> (modalities in data/, arguments, derived result name)
SAVED_TO = {
    "correct_batch_effects": (_batched, {"modality_name": "x"}, "x_batch_corrected"),
    "summarize_peptide_to_protein": (
        _peptides,
        {"modality_name": "x"},
        "x_protein_rollup",
    ),
    "normalize_ptm_to_protein": (
        _ptm_sites,
        {"ptm_modality_name": "x", "protein_modality_name": "prot"},
        "x_ptm_normalized",
    ),
    "normalize_bridge_samples": (
        _bridged,
        {"modality_name": "x"},
        "x_bridge_normalized",
    ),
}


@pytest.mark.parametrize("tool", list(SAVED_TO))
def test_the_reply_names_the_file_saved_and_the_modality_stored(tmp_path, tool):
    """In a resumed session whose data/ holds <x>_<step>.h5ad, the result is
    <x>_<step>_2: the reply's "Saved to" names that file, which exists, and the
    stored modality has that name (it named the loaded file)."""
    proteomics = pytest.importorskip("lobster.agents.proteomics.proteomics_expert")
    from lobster.testing.provenance_replay import factory_tools

    data, args, result = SAVED_TO[tool]
    folder = tmp_path / "ws" / "data"
    folder.mkdir(parents=True)
    for name, adata in data().items():
        adata.write_h5ad(folder / f"{name}.h5ad")
    data()["x"].write_h5ad(folder / f"{result}.h5ad")  # a first session's result
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    before = _sha(folder / f"{result}.h5ad")
    tools = factory_tools(proteomics.proteomics_expert, proteomics.__name__, dm)

    reply = tools[tool].invoke(args)

    saved = [line for line in reply.splitlines() if "Saved to" in line]
    assert saved == [f"**Saved to:** {result}_2.h5ad"], reply
    assert (folder / f"{result}_2.h5ad").is_file()
    assert f"{result}_2" in dm.list_modalities()
    assert _sha(folder / f"{result}.h5ad") == before


def _elsewhere_snapshot(tmp_path):
    """z_autosave.h5ad outside data/, loaded by load_modality as 'q'."""
    source = tmp_path / "elsewhere" / "z_autosave.h5ad"
    source.parent.mkdir()
    _adata(shift=6.0).write_h5ad(source)
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.load_modality("q", str(source), adapter="transcriptomics_bulk", validate=False)
    return dm, source


# (how the snapshot was loaded, name saved, path saved to)
NOT_OWN = {
    "another_name": ("auto", "y", "z_autosave.h5ad"),
    "loaded_as_another_modality": ("renamed", "z", "z_autosave.h5ad"),
    "outside_data": ("elsewhere", "z", None),
    "outside_data_as_itself": ("elsewhere", "z_autosave", None),
}


@pytest.mark.parametrize("case", list(NOT_OWN))
def test_own_snapshot_never_reaches_another_modalitys_loaded_file(tmp_path, case):
    """The auto-save exemption covers only the modality's own data/ snapshot,
    loaded as that modality or as its snapshot; a direct save that claims it for
    another modality's loaded file is refused and the file is unchanged."""
    how, name, path = NOT_OWN[case]
    if how == "elsewhere":
        dm, target = _elsewhere_snapshot(tmp_path)
        path = str(target)
    else:
        data = tmp_path / "ws" / "data"
        data.mkdir(parents=True)
        target = data / "z_autosave.h5ad"
        _adata(shift=6.0).write_h5ad(target)
        dm = DataManagerV2(workspace_path=tmp_path / "ws")
        if how == "renamed":  # only as 'q': the file name is z's, the modality not
            dm.load_modality("q", str(target), adapter="transcriptomics_bulk")
            del dm.modalities["z_autosave"]
    dm.modalities[name] = _adata(shift=9.0)
    before = _sha(target)

    with pytest.raises(RecordedInputError):
        dm.save_modality(name, path, own_snapshot=True)
    assert _sha(target) == before


def _case_insensitive(folder):
    probe = folder / "Case_Probe"
    probe.touch()
    try:
        return (folder / "case_probe").exists()
    finally:
        probe.unlink()


def test_a_name_differing_in_case_is_the_loaded_file(ws):
    """On a case-insensitive volume (macOS by default) X.h5ad is x.h5ad: a
    modality 'X' saved there overwrote the loaded file."""
    if not _case_insensitive(ws):
        pytest.skip("this volume tells X.h5ad from x.h5ad")
    dm = DataManagerV2(workspace_path=ws)
    target = dm.data_dir / "x.h5ad"
    before = _sha(target)
    dm.modalities["X"] = _adata(shift=2.0)

    with pytest.raises(RecordedInputError):
        dm.save_modality("X", "X.h5ad")
    assert _sha(target) == before
    assert dm.recorded_input("X.h5ad") == "x"
    assert dm.unloaded_name("X") == "X_2"


def test_a_loaded_file_moved_away_is_still_matched_whatever_its_case(ws):
    """A loaded file no longer on disk cannot be compared as a file: its path
    is compared, letter case aside (refusing is the safe side)."""
    dm = DataManagerV2(workspace_path=ws)
    (dm.data_dir / "x.h5ad").rename(ws.parent / "moved.h5ad")
    assert dm.recorded_input("x.h5ad") == "x"
    assert dm.recorded_input("X.h5ad") == "x"


def test_another_file_differing_only_in_case_is_not_the_loaded_one(ws):
    """Where the volume tells them apart, X.h5ad beside the loaded x.h5ad is
    another file and may be saved."""
    if _case_insensitive(ws):
        pytest.skip("this volume does not tell X.h5ad from x.h5ad")
    dm = DataManagerV2(workspace_path=ws)
    before = _sha(dm.data_dir / "x.h5ad")
    dm.modalities["X"] = _adata(shift=2.0)
    dm.save_modality("X", "X.h5ad")
    assert dm.recorded_input("X.h5ad") is None
    assert _sha(dm.data_dir / "x.h5ad") == before


@pytest.mark.parametrize("split", ["train", "test", "validation"])
def test_ml_splits_refused_leave_memory_provenance_and_files(ws, split):
    """create_ml_splits into a split name whose file this session loaded
    (x_<split>.h5ad from a first session) is refused before anything changes:
    no split stored, the input modality not annotated, no step logged."""
    ml = pytest.importorskip("lobster.agents.machine_learning.machine_learning_expert")
    from lobster.testing.provenance_replay import factory_tools

    rng = np.random.default_rng(4)
    adata = ad.AnnData(
        X=rng.random((20, 5)).astype(np.float32),
        obs=pd.DataFrame(index=[f"s{i}" for i in range(20)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
    )
    adata.write_h5ad(ws / "data" / "x.h5ad")
    adata[:4].copy().write_h5ad(ws / "data" / f"x_{split}.h5ad")
    dm = DataManagerV2(workspace_path=ws)
    modalities = sorted(dm.list_modalities())
    shas = {p.name: _sha(p) for p in (ws / "data").glob("*.h5ad")}
    obs, uns = list(dm.get_modality("x").obs.columns), set(dm.get_modality("x").uns)
    activities = len(dm.provenance.activities)
    tools = factory_tools(ml.machine_learning_expert, ml.__name__, dm)

    reply = tools["create_ml_splits"].invoke({"modality_name": "x"})

    assert f"data/x_{split}.h5ad is a file this session loaded" in reply, reply
    assert sorted(dm.list_modalities()) == modalities
    assert list(dm.get_modality("x").obs.columns) == obs
    assert set(dm.get_modality("x").uns) == uns
    assert len(dm.provenance.activities) == activities
    assert {p.name: _sha(p) for p in (ws / "data").glob("*.h5ad")} == shas


def _stats_for_one_autosave(tmp_path, n, monkeypatch):
    """os.stat calls made by one auto-save of a resumed session that loaded n
    files (x<i>.h5ad, none changed)."""
    import os

    data = tmp_path / f"ws{n}" / "data"
    data.mkdir(parents=True)
    tiny = ad.AnnData(X=np.zeros((2, 2), dtype=np.float32))
    for i in range(n):
        tiny.write_h5ad(data / f"x{i}.h5ad")
    dm = DataManagerV2(workspace_path=data.parent)
    assert len(dm.list_modalities()) == n
    calls = []
    real = os.stat

    def counting(*args, **kwargs):
        calls.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(os, "stat", counting)
    dm.auto_save_state()
    monkeypatch.setattr(os, "stat", real)
    return len(calls)


def test_one_autosave_costs_a_bounded_number_of_stats_per_loaded_file(
    tmp_path, monkeypatch
):
    """Matching a path against the loaded files is a lookup, not a scan: an
    auto-save of n loaded files makes O(n) stat calls, not O(n^2) (400 loaded
    files took 24 s to auto-save instead of 2.5 s)."""
    small = _stats_for_one_autosave(tmp_path, 50, monkeypatch)
    large = _stats_for_one_autosave(tmp_path, 200, monkeypatch)
    assert large <= 4 * small * 1.25, (small, large)  # linear, with slack


def test_no_validation_split_needs_no_validation_name(ws):
    """validation_size=0 makes no validation split: a loaded x_validation.h5ad
    does not stop the train/test split."""
    ml = pytest.importorskip("lobster.agents.machine_learning.machine_learning_expert")
    from lobster.testing.provenance_replay import factory_tools

    rng = np.random.default_rng(4)
    adata = ad.AnnData(
        X=rng.random((20, 5)).astype(np.float32),
        obs=pd.DataFrame(index=[f"s{i}" for i in range(20)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
    )
    adata.write_h5ad(ws / "data" / "x.h5ad")
    adata[:4].copy().write_h5ad(ws / "data" / "x_validation.h5ad")
    dm = DataManagerV2(workspace_path=ws)
    tools = factory_tools(ml.machine_learning_expert, ml.__name__, dm)

    reply = tools["create_ml_splits"].invoke(
        {"modality_name": "x", "validation_size": 0.0}
    )

    assert "Successfully created ML data splits" in reply, reply
    assert {"x_train", "x_test"} <= set(dm.list_modalities())
