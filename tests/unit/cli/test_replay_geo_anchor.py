"""A GEO import is the saved raw file: the session records its SHA-256 at the save
that follows the load, the notebook reads that file checked by SHA-256 (it does not
download), and ``lobster replay --verify`` can then say MATCH.

Only the GEO import asks for the anchor. Every other load of an object stays
not compared. Skipped without scanpy and the transcriptomics package.
"""

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tests.unit.cli import test_replay_commands as _base

globals().update(
    {
        k: v
        for k, v in vars(_base).items()
        if not k.startswith("test_") and not k.startswith("__")
    }
)

pytest.importorskip("scanpy")
tools_module = pytest.importorskip("lobster.agents.transcriptomics.shared_tools")

WORKSPACE = "workspace"  # a project's own workspace is ".lobster_workspace"
MODALITY = "geo_gse1"
RAW = f"{MODALITY}_raw.h5ad"


def _adata():
    import anndata as ad

    rng = np.random.default_rng(3)
    adata = ad.AnnData(rng.poisson(1.2, size=(240, 300)).astype("float32"))
    adata.obs_names = [f"c{i}" for i in range(240)]
    adata.var_names = [f"MT-{i}" if i < 5 else f"g{i}" for i in range(300)]
    return adata


def _manager(tmp_path, monkeypatch):
    from lobster.core.runtime.data_manager import DataManagerV2

    _clean_env(monkeypatch, tmp_path)
    return DataManagerV2(workspace_path=tmp_path / WORKSPACE, auto_scan=False)


def _import(manager, geo_import=True, between=None, save_object=None, table=False):
    """What a GEO import does: load an object (or a table), then save it."""
    source = _adata()
    if table:
        source = source.to_df()
    adata = manager.load_modality(
        MODALITY, source, adapter="transcriptomics_single_cell", validate=False
    )
    if between:
        between(manager)
    if save_object is not None:
        manager.modalities[MODALITY] = save_object(adata)
    kwargs = {"geo_import": True} if geo_import else {}
    manager.save_modality(MODALITY, RAW, **kwargs)
    return manager.data_dir / RAW


def _loaded(manager):
    from lobster.core.notebooks.exporter import NotebookExporter

    exporter = NotebookExporter(manager.provenance, manager)
    return exporter._get_input_roles({})["loaded"]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_the_import_save_records_the_sha_of_the_written_file(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    raw = _import(manager)
    (entry,) = _loaded(manager)
    assert entry["name"] == RAW
    assert entry["sha256"] == _sha(raw)
    assert entry["size"] == raw.stat().st_size
    assert replay_commands._loaded_key(entry) == ("file", _sha(raw))


def test_a_save_without_the_import_flag_does_not_anchor(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    _import(manager, geo_import=False)
    (entry,) = _loaded(manager)
    assert "sha256" not in entry and replay_commands._loaded_key(entry) is None


def test_a_different_object_saved_does_not_anchor(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    try:
        _import(manager, save_object=lambda adata: adata.copy())
    except TypeError:  # the flag does not exist yet
        pass
    (entry,) = _loaded(manager)
    assert "sha256" not in entry


def test_a_step_between_the_load_and_the_save_does_not_anchor(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)

    def step(m):
        m.log_tool_usage("some_step", {}, "a step that changes the object")

    try:
        _import(manager, between=step)
    except TypeError:
        pass
    (entry,) = _loaded(manager)
    assert "sha256" not in entry


def test_a_load_from_a_file_is_not_changed_by_the_flag(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    source = tmp_path / "in.h5ad"
    _adata().write_h5ad(source)
    manager.load_modality(MODALITY, source, adapter="transcriptomics_single_cell")
    before = _loaded(manager)
    try:
        manager.save_modality(MODALITY, RAW, geo_import=True)
    except TypeError:
        pass
    assert _loaded(manager) == before


def test_a_save_that_fails_does_not_anchor(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    manager.load_modality(
        MODALITY, _adata(), adapter="transcriptomics_single_cell", validate=False
    )
    backend = manager.backends[manager.default_backend]
    monkeypatch.setattr(
        backend, "save", lambda *a, **k: (_ for _ in ()).throw(OSError("disk"))
    )
    with pytest.raises((OSError, TypeError)):
        manager.save_modality(MODALITY, RAW, geo_import=True)
    (entry,) = _loaded(manager)
    assert "sha256" not in entry


# The six places a GEO import saves what it loaded. Only an import whose saved file
# the notebook can read back as the session's input asks for the anchor.
SITES = {
    "lobster/services/data_access/geo/download_execution.py": [True, True],
    "lobster/services/data_access/geo/concatenation.py": [False, False],
    "lobster/services/data_access/geo_fallback_service.py": [False, False],
}


def _save_calls(path):
    root = Path(__file__).resolve().parents[3]
    tree = ast.parse((root / path).read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "save_modality"
    ]
    calls.sort(key=lambda n: n.lineno)
    return [any(k.arg == "geo_import" for k in n.keywords) for n in calls]


@pytest.mark.parametrize(
    "path",
    sorted(SITES),
)
def test_only_the_imports_that_can_be_anchored_ask_for_it(path):
    assert _save_calls(path) == SITES[path]


def _concatenate_samples(manager):
    from types import SimpleNamespace

    from lobster.services.data_access.geo.concatenation import SampleConcatenator

    names = []
    for i in (1, 2):
        adata = _adata()
        adata.obs_names = [f"s{i}_{n}" for n in adata.obs_names]
        manager.load_modality(
            f"geo_sample_gsm{i}",
            adata,
            adapter="transcriptomics_single_cell",
            validate=False,
        )
        manager.save_modality(
            f"geo_sample_gsm{i}", f"gse1_gsm{i}_raw.h5ad", geo_sample=True
        )
        names.append(f"geo_sample_gsm{i}")
    SampleConcatenator(
        SimpleNamespace(data_manager=manager)
    )._concatenate_stored_samples("GSE1", names, False)


def _geo_session(
    tmp_path,
    monkeypatch,
    anchor=True,
    concatenated=False,
    table=False,
    autosave=False,
    autosave_later=False,
    on_manager=None,
):
    """A session: import, quality, filter, save; export the notebook."""
    from lobster.services.analysis.clustering_service import ClusteringService
    from lobster.services.data_access.geo_download_service import GEODownloadService
    from lobster.services.quality.preprocessing_service import PreprocessingService
    from lobster.services.quality.quality_service import QualityService

    manager = _manager(tmp_path, monkeypatch)
    if on_manager:
        on_manager(manager)
    if concatenated:  # the samples a series is made of, concatenated before the import
        _concatenate_samples(manager)
    raw = _import(manager, geo_import=anchor, table=table)
    service = GEODownloadService.__new__(GEODownloadService)
    service.data_manager = manager
    ir = service._create_analysis_step_ir(
        geo_id="GSE1",
        modality_name=MODALITY,
        strategy_name=None,
        use_intersecting_genes_only=None,
        queue_entry_id="q1",
    )
    manager.log_tool_usage("download_geo", {"geo_id": "GSE1"}, "import", ir=ir)
    if autosave:  # what the GEO import does right after its save
        manager.auto_save_state()
    tools = {
        t.name: t
        for t in tools_module.create_shared_tools(
            manager,
            QualityService(),
            PreprocessingService(),
            ClusteringService(manager),
        )
    }
    name = MODALITY
    for tool in ("assess_data_quality", "filter_and_normalize"):
        args = {"modality_name": name}
        if tool == "filter_and_normalize":
            args["save_result"] = False
        before = set(manager.list_modalities())
        result = tools[tool].invoke(args)
        assert "rror" not in result[:80], result[:300]
        made = [m for m in manager.list_modalities() if m not in before]
        name = made[-1] if made else name
    workspace = tmp_path / WORKSPACE
    manager.save_modality(name, workspace / "data" / f"{name}.h5ad")
    if autosave_later:  # an auto-save after the analysis: not the import's
        manager.auto_save_state()
    notebook = Path(manager.export_notebook("nb", "S", filter_strategy="replay"))
    return workspace, notebook, raw


@needs_kernel
def test_a_geo_session_replays_to_a_match(tmp_path, monkeypatch):
    workspace, notebook, raw = _geo_session(tmp_path, monkeypatch)
    result = _verify_real(workspace, notebook, raw)
    assert result.exit_code == 0, (result.output, result.exception)
    assert json.loads(result.stdout)["verdict"] == "MATCH"


@needs_kernel
def test_the_notebook_reads_the_raw_file_checked_by_sha_and_does_not_download(
    tmp_path, monkeypatch
):
    _workspace, notebook, raw = _geo_session(tmp_path, monkeypatch)
    import nbformat

    code = "\n".join(
        c.source for c in nbformat.read(notebook, 4).cells if c.cell_type == "code"
    )
    assert "sha256" in code and "input_data" in code
    for fetch in ("requests", "urllib", "GEOparse", "ftp", "wget", ".download("):
        assert fetch not in code, fetch


@needs_kernel
def test_a_changed_raw_file_is_never_a_match_and_is_not_downloaded_again(
    tmp_path, monkeypatch
):
    workspace, notebook, raw = _geo_session(tmp_path, monkeypatch)
    import anndata as ad

    adata = ad.read_h5ad(raw)
    adata.X = adata.X + 1
    adata.write_h5ad(raw)
    result = _verify_real(workspace, notebook, raw)
    out = json.loads(result.stdout)
    assert out["verdict"] == "REFUSED", out
    assert result.exit_code == 4


@needs_kernel
def test_an_import_that_was_not_anchored_is_not_compared(tmp_path, monkeypatch):
    workspace, notebook, raw = _geo_session(tmp_path, monkeypatch, anchor=False)
    result = _verify_real(workspace, notebook, raw)
    assert result.exit_code != 0
    assert json.loads(result.stdout)["verdict"] != "MATCH"


def _ir_for(manager, modality):
    from lobster.services.data_access.geo_download_service import GEODownloadService

    service = GEODownloadService.__new__(GEODownloadService)
    service.data_manager = manager
    return service._create_analysis_step_ir(
        geo_id="GSE1",
        modality_name=modality,
        strategy_name=None,
        use_intersecting_genes_only=None,
        queue_entry_id="q1",
    )


def _named_files(ir):
    """Every .h5ad name the IR carries: in its code, its parameters, its inputs."""
    import re

    text = ir.render() + json.dumps(ir.parameters) + json.dumps(ir.input_entities)
    return set(re.findall(r"[\w./-]+\.h5ad", text))


@pytest.mark.parametrize("case", ["anchored", "saved_not_anchored", "never_saved"])
def test_a_geo_load_cell_never_names_a_file_that_was_not_written(
    tmp_path, monkeypatch, case
):
    manager = _manager(tmp_path, monkeypatch)
    if case != "never_saved":
        _import(manager, geo_import=case == "anchored")
    else:
        manager.load_modality(
            MODALITY, _adata(), adapter="transcriptomics_single_cell", validate=False
        )
    named = _named_files(_ir_for(manager, MODALITY))
    for name in named:
        assert (manager.data_dir / Path(name).name).is_file(), (case, name)
    if case == "never_saved":
        assert not named and "raise" in _ir_for(manager, MODALITY).render()


def _verify_without_input(workspace, notebook):
    return _base.runner.invoke(
        _base.app,
        ["--verify", str(notebook), "-w", str(workspace), "--json"],
    )


@needs_kernel
def test_an_anchored_notebook_needs_no_input_option(tmp_path, monkeypatch):
    workspace, notebook, raw = _geo_session(tmp_path, monkeypatch)
    result = _verify_without_input(workspace, notebook)
    assert result.exit_code == 0, (result.output, result.exception)
    assert json.loads(result.stdout)["verdict"] == "MATCH"


@needs_kernel
def test_the_input_that_is_not_the_anchored_file_is_refused(tmp_path, monkeypatch):
    workspace, notebook, raw = _geo_session(tmp_path, monkeypatch)
    other = tmp_path / "other.h5ad"
    _adata().write_h5ad(other)
    result = _verify_real(workspace, notebook, other)
    assert result.exit_code == 4, (result.output, result.exception)
    assert json.loads(result.stdout)["verdict"] == "REFUSED"


@needs_kernel
def test_an_anchored_file_that_is_gone_is_refused_with_its_name(tmp_path, monkeypatch):
    workspace, notebook, raw = _geo_session(tmp_path, monkeypatch)
    raw.unlink()
    result = _verify_without_input(workspace, notebook)
    assert result.exit_code == 4, (result.output, result.exception)
    out = json.loads(result.stdout)
    assert out["verdict"] == "REFUSED"
    assert RAW in json.dumps(out)


@needs_kernel
def test_a_notebook_without_an_anchor_still_needs_the_input_option(
    tmp_path, monkeypatch
):
    workspace, notebook, raw = _geo_session(tmp_path, monkeypatch, anchor=False)
    result = _verify_without_input(workspace, notebook)
    assert result.exit_code == 2, (result.output, result.exception)
    assert "--input" in result.output


@needs_kernel
def test_a_series_of_samples_replays_to_a_match(tmp_path, monkeypatch):
    workspace, notebook, raw = _geo_session(tmp_path, monkeypatch, concatenated=True)
    result = _verify_real(workspace, notebook, raw)
    assert result.exit_code == 0, (result.output[-800:], result.exception)
    assert json.loads(result.stdout)["verdict"] == "MATCH"


def test_the_load_cell_of_a_series_names_the_merged_file_not_a_sample_file(
    tmp_path, monkeypatch
):
    from lobster.services.data_access.geo_download_service import GEODownloadService

    manager = _manager(tmp_path, monkeypatch)
    for i in (1, 2):  # the stored samples come first, as in a real download
        sample = manager.load_modality(
            f"geo_gse1_sample_gsm{i}",
            _adata(),
            adapter="transcriptomics_single_cell",
            validate=False,
        )
        manager.save_modality(f"geo_gse1_sample_gsm{i}", f"gse1_gsm{i}_raw.h5ad")
    merged = "geo_gse1_transcriptomics_single_cell"
    manager.load_modality(
        merged, _adata(), adapter="transcriptomics_single_cell", validate=False
    )
    written = manager.data_dir / f"{merged}_raw.h5ad"
    manager.save_modality(merged, written.name, geo_import=True)
    service = GEODownloadService.__new__(GEODownloadService)
    service.data_manager = manager
    assert service._find_stored_modality("GSE1") == merged
    named = _named_files(_ir_for(manager, merged))
    assert named == {written.name} and written.is_file()


def _rich_adata():
    import pandas as pd

    adata = _adata()
    rng = np.random.default_rng(5)
    adata.obs["title"] = rng.choice(["a b", "c d"], adata.n_obs)  # text
    adata.obs["missing_text"] = rng.choice(["x", None], adata.n_obs)
    adata.uns["geo_metadata"] = {"gse": "GSE1", "list": [1, 2], "nested": {"a": 1.5}}
    return adata


def test_after_the_import_the_session_holds_what_the_raw_file_holds(
    tmp_path, monkeypatch
):
    import anndata as ad

    manager = _manager(tmp_path, monkeypatch)
    manager.load_modality(
        MODALITY, _rich_adata(), adapter="transcriptomics_single_cell", validate=False
    )
    manager.save_modality(MODALITY, RAW, geo_import=True)
    session = manager.get_modality(MODALITY)
    file = ad.read_h5ad(manager.data_dir / RAW)
    for column in file.obs.columns:
        assert session.obs[column].dtype == file.obs[column].dtype, column
    assert session.obs.equals(file.obs)
    mine, theirs = session.uns["geo_metadata"], file.uns["geo_metadata"]
    assert mine["nested"] == theirs["nested"]
    assert list(mine["list"]) == list(theirs["list"])
    assert np.array_equal(np.asarray(session.X), np.asarray(file.X))


def test_a_raw_file_that_cannot_be_read_back_does_not_anchor(tmp_path, monkeypatch):
    import anndata as ad

    manager = _manager(tmp_path, monkeypatch)
    manager.load_modality(
        MODALITY, _adata(), adapter="transcriptomics_single_cell", validate=False
    )
    monkeypatch.setattr(
        ad, "read_h5ad", lambda *a, **k: (_ for _ in ()).throw(OSError("unreadable"))
    )
    manager.save_modality(MODALITY, RAW, geo_import=True)
    (entry,) = _loaded(manager)
    assert "sha256" not in entry


@needs_kernel
@pytest.mark.parametrize("pass_parent", [True, False])
def test_the_anchored_file_is_found_from_a_run_folder_or_its_workspace(
    tmp_path, monkeypatch, pass_parent
):
    import shutil

    workspace, notebook, raw = _geo_session(tmp_path, monkeypatch)
    run = tmp_path / "runs" / "one"
    shutil.copytree(workspace, run / ".lobster_workspace")
    shutil.rmtree(workspace)
    notebook = run / ".lobster_workspace" / notebook.relative_to(workspace)
    given = run if pass_parent else run / ".lobster_workspace"
    result = _verify_without_input(given, notebook)
    assert result.exit_code == 0, (result.output, result.exception)
    assert json.loads(result.stdout)["verdict"] == "MATCH"


@needs_kernel
def test_a_folder_with_its_own_data_is_the_workspace_even_with_a_nested_one(
    tmp_path, monkeypatch
):
    workspace, notebook, raw = _geo_session(tmp_path, monkeypatch)
    (workspace / ".lobster_workspace" / "data").mkdir(parents=True)  # not the one used
    result = _verify_without_input(workspace, notebook)
    assert result.exit_code == 0, (result.output, result.exception)
    assert json.loads(result.stdout)["verdict"] == "MATCH"


def test_an_import_loaded_from_a_table_anchors_on_the_object_the_adapter_made(
    tmp_path, monkeypatch
):
    import pandas as pd

    manager = _manager(tmp_path, monkeypatch)
    table = pd.DataFrame(
        np.arange(60, dtype="float32").reshape(6, 10),
        index=[f"c{i}" for i in range(6)],
        columns=[f"g{i}" for i in range(10)],
    )
    manager.load_modality(
        MODALITY, table, adapter="transcriptomics_single_cell", validate=False
    )
    manager.save_modality(MODALITY, RAW, geo_import=True)
    (entry,) = _loaded(manager)
    assert entry["name"] == RAW and entry["sha256"] == _sha(manager.data_dir / RAW)
    assert entry["anchored"] == "import"


def test_a_table_load_changed_before_its_save_does_not_anchor(tmp_path, monkeypatch):
    import pandas as pd

    manager = _manager(tmp_path, monkeypatch)
    table = pd.DataFrame(np.ones((6, 10), dtype="float32"))
    table.index, table.columns = [f"c{i}" for i in range(6)], [
        f"g{i}" for i in range(10)
    ]
    manager.load_modality(
        MODALITY, table, adapter="transcriptomics_single_cell", validate=False
    )
    manager.log_tool_usage("edit", {}, "changed before the save")
    manager.save_modality(MODALITY, RAW, geo_import=True)
    assert all("sha256" not in e for e in _loaded(manager))


@needs_kernel
def test_a_session_that_imported_a_table_replays_to_a_match(tmp_path, monkeypatch):
    workspace, notebook, raw = _geo_session(tmp_path, monkeypatch, table=True)
    result = _verify_real(workspace, notebook, raw)
    assert result.exit_code == 0, (result.output[-1500:], result.exception)
    assert json.loads(result.stdout)["verdict"] == "MATCH"


def _report(result):
    return json.loads(result.stdout)


@needs_kernel
def test_the_imports_own_autosave_is_covered_by_its_raw_file_and_the_row_matches(
    tmp_path, monkeypatch
):
    workspace, notebook, raw = _geo_session(tmp_path, monkeypatch, autosave=True)
    result = _verify_real(workspace, notebook, raw)
    out = _report(result)
    assert result.exit_code == 0 and out["verdict"] == "MATCH", out
    assert any("Covered by the import's raw file" in n for n in out["notes"])  # named
    assert any("geo_gse1_autosave.h5ad" in n for n in out["notes"])


@needs_kernel
def test_a_changed_import_autosave_is_a_mismatch(tmp_path, monkeypatch):
    workspace, notebook, raw = _geo_session(tmp_path, monkeypatch, autosave=True)
    import nbformat

    nb = nbformat.read(notebook, 4)
    changed = False
    for cell in nb.cells:  # the notebook writes the auto-save with one value changed
        if (
            cell.cell_type == "code"
            and "_autosave" in cell.source
            and "output_path" in cell.source
        ):
            cell.source = (
                "_keep = adata.copy()\nadata.X = adata.X + 1\n"
                + cell.source
                + "\nadata = _keep\n"
            )
            changed = True
    assert changed
    nbformat.write(nb, notebook)
    out = _report(_verify_real(workspace, notebook, raw))
    assert out["verdict"] == "MISMATCH", out


@needs_kernel
def test_a_later_autosave_of_a_changed_modality_stays_not_compared(
    tmp_path, monkeypatch
):
    workspace, notebook, raw = _geo_session(
        tmp_path, monkeypatch, autosave=True, autosave_later=True
    )
    out = _report(_verify_real(workspace, notebook, raw))
    assert out["verdict"] == "NOT_COMPARED", out
    assert any("quality_assessed" in n for n in out["notes"]), out["notes"]


@needs_kernel
def test_a_table_import_changed_after_its_save_gets_no_autosave_tag(
    tmp_path, monkeypatch
):
    manager = _manager(tmp_path, monkeypatch)
    _import(manager, table=True)
    manager.get_modality(MODALITY).obs["x"] = 1.0  # changed after the import's save
    manager.auto_save_state()
    (entry,) = _loaded(manager)
    assert entry["anchored"] == "import" and "autosave" not in entry


def test_an_unchanged_import_autosave_is_tagged_with_its_file_and_hash(
    tmp_path, monkeypatch
):
    manager = _manager(tmp_path, monkeypatch)
    _import(manager)
    manager.auto_save_state()
    (entry,) = _loaded(manager)
    written = manager.data_dir / entry["autosave"]["name"]
    assert entry["autosave"]["sha256"] == _sha(written)


@needs_kernel
def test_a_series_matches_and_names_each_sample_file_and_the_merged_input(
    tmp_path, monkeypatch
):
    workspace, notebook, raw = _geo_session(
        tmp_path, monkeypatch, autosave=True, concatenated=True
    )
    result = _verify_real(workspace, notebook, raw)
    out = _report(result)
    assert result.exit_code == 0 and out["verdict"] == "MATCH", out
    note = next(n for n in out["notes"] if "per-sample inputs" in n)
    for name in ("gse1_gsm1_raw.h5ad", "gse1_gsm2_raw.h5ad"):
        assert name in note
    report = json.loads(Path(out["replay_dir"], "replay_report.json").read_text())
    assert list(report["bindings"]["root_sha256"]) == [RAW]  # the merged file


def test_the_import_flags_are_passed_only_by_the_import_code():
    root = Path(__file__).resolve().parents[3] / "lobster"
    allowed = {"download_execution.py", "concatenation.py"}
    users = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and any(
                k.arg in ("geo_import", "geo_sample") for k in node.keywords
            ):
                users.add(path.name)
    assert users == allowed, users


def test_a_save_that_is_not_an_import_gets_no_tag_whatever_it_asks_for(
    tmp_path, monkeypatch
):
    manager = _manager(tmp_path, monkeypatch)
    manager.modalities[MODALITY] = _adata()  # put there, not loaded by an import
    manager.save_modality(MODALITY, RAW, geo_import=True, geo_sample=True)
    manager.auto_save_state()
    for entry in _loaded(manager):
        assert not {"anchored", "autosave", "samples"} & set(entry)
    assert manager._geo_sample_files == []


@needs_kernel
def test_every_tagged_file_is_named_in_the_report(tmp_path, monkeypatch):
    workspace, notebook, raw = _geo_session(
        tmp_path, monkeypatch, autosave=True, concatenated=True
    )
    out = _report(_verify_real(workspace, notebook, raw))
    text = " ".join(out["notes"])
    tagged_files = (
        "geo_gse1_autosave.h5ad",
        "gse1_gsm1_raw.h5ad",
        "gse1_gsm2_raw.h5ad",
        "geo_sample_gsm1_autosave.h5ad",
        "geo_sample_gsm2_autosave.h5ad",
    )
    for tagged in tagged_files:
        assert tagged in text, (tagged, out["notes"])


@needs_kernel
def test_an_import_autosave_changed_on_disk_after_the_session_is_never_a_match(
    tmp_path, monkeypatch
):
    import anndata as ad

    workspace, notebook, raw = _geo_session(tmp_path, monkeypatch, autosave=True)
    file = workspace / "data" / "geo_gse1_autosave.h5ad"
    adata = ad.read_h5ad(file)
    adata.X = adata.X + 1
    adata.write_h5ad(file)
    result = _verify_real(workspace, notebook, raw)
    out = _report(result)
    assert out["verdict"] == "REFUSED" and result.exit_code == 4, out


@needs_kernel
def test_an_import_autosave_gone_from_the_workspace_is_not_a_match(
    tmp_path, monkeypatch
):
    workspace, notebook, raw = _geo_session(tmp_path, monkeypatch, autosave=True)
    (workspace / "data" / "geo_gse1_autosave.h5ad").unlink()
    out = _report(_verify_real(workspace, notebook, raw))
    assert out["verdict"] == "NOT_COMPARED", out


@needs_kernel
def test_the_text_report_of_a_series_names_the_merged_raw_file_as_its_input(
    tmp_path, monkeypatch
):
    workspace, notebook, raw = _geo_session(
        tmp_path, monkeypatch, autosave=True, concatenated=True
    )
    result = _base.runner.invoke(
        _base.app, ["--verify", str(notebook), "-w", str(workspace)]
    )
    assert result.exit_code == 0, result.output[-800:]
    line = f"Input: the GEO import's raw file {RAW} (SHA-256 checked)"
    assert line in result.output, result.output[-1200:]
    assert result.output.index(line) < result.output.index("gse1_gsm1_raw.h5ad")


@needs_kernel
def test_a_series_read_from_flat_sample_files_replays_to_a_match(tmp_path, monkeypatch):
    from tests.unit.services.data_access import test_geo_flat_trio_samples as flat

    files = {}
    for gsm, tag, cells, seed in (("GSM1", "A", 80, 1), ("GSM2", "B", 90, 2)):
        files.update(flat._trio(cells, seed, f"{gsm}_{tag}_"))
    files.update(flat._trio(10, 3, "GSM3_C_"))
    files["GSM3_C_matrix.mtx.gz"] = files["GSM3_C_matrix.mtx.gz"][:40]
    merged = flat._process(tmp_path, files)
    assert merged.n_obs == 170 and len(merged.uns["geo_import"]["not_loaded"]) == 1
    monkeypatch.setitem(globals(), "_adata", lambda: merged.copy())
    workspace, notebook, raw = _geo_session(tmp_path, monkeypatch)
    result = _verify_real(workspace, notebook, raw)
    assert result.exit_code == 0, (result.output, result.exception)
    assert json.loads(result.stdout)["verdict"] == "MATCH"


@pytest.mark.parametrize("table", [False, True])
def test_what_an_import_did_not_load_reaches_the_summary_and_the_provenance(
    tmp_path, monkeypatch, table
):
    from unittest.mock import MagicMock

    from lobster.services.data_access.geo.constants import GEODataSource, GEOResult
    from lobster.services.data_access.geo.download_execution import (
        DownloadExecutor,
    )

    manager = _manager(tmp_path, monkeypatch)
    manager.metadata_store["GSE1"] = {"metadata": {}}
    data = _adata()
    statement = ["GSM2_B: the files could not be read (truncated)"]
    if table:
        data = data.to_df()
        data.attrs["not_loaded"] = statement
    else:
        data.uns["geo_import"] = {"not_loaded": statement}
    service = MagicMock()
    service.data_manager = manager
    executor = DownloadExecutor(service)
    monkeypatch.setattr(
        executor,
        "download_with_strategy",
        lambda **k: GEOResult(
            data=data,
            metadata={},
            source=GEODataSource.SUPPLEMENTARY,
            processing_info={
                "method": "supplementary_first",
                "data_type": "single_cell_rna_seq",
            },
            success=True,
        ),
    )
    message = executor.download_dataset("GSE1", adapter="transcriptomics_single_cell")
    assert "Not loaded: GSM2_B: the files could not be read" in message
    logged = [
        a for a in manager.provenance.activities if "not_loaded" in a["parameters"]
    ]
    assert logged and logged[-1]["parameters"]["not_loaded"] == statement


@needs_kernel
def test_a_project_run_with_the_catalog_on_replays_to_a_match(tmp_path):
    """The whole flow in a project: a session recorded in its catalog, then replayed."""
    pytest.importorskip("lamindb")
    import textwrap

    from lobster.core.projects import init_project
    from tests.unit.core.catalog.test_lamin_engine import _run

    root = tmp_path / "study"
    root.mkdir()
    init_project(root, "study")
    (root / ".lobster_workspace" / ".env").write_text("LOBSTER_LAMIN=1\n")
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).resolve().parents[3])!r})
        import json
        from pathlib import Path
        from _pytest.monkeypatch import MonkeyPatch
        from typer.testing import CliRunner
        from lobster.core.catalog import factory
        from tests.unit.cli import test_replay_geo_anchor as g

        proj = Path(os.environ["PROJ"])
        g.WORKSPACE = ".lobster_workspace"

        def begin(manager):
            ws = proj / ".lobster_workspace"
            sid = "session_one"
            factory.begin_session_run(manager, ws, sid, ws / ".lobster" / "sessions" / sid)

        workspace, notebook, raw = g._geo_session(
            proj, MonkeyPatch(), on_manager=begin
        )
        result = g._verify_real(workspace, notebook, raw)
        out = json.loads(result.stdout)
        emit(exit=result.exit_code, verdict=out["verdict"], notes=out["notes"], files=sorted(p.name for p in (workspace / "data").iterdir()))
        """)
    r = _run(
        code, tmp_path, env_extra={"PROJ": str(root)}, drop_enable=True, timeout=900
    )
    assert r["verdict"] == "MATCH" and r["exit"] == 0, r


def _persisted(tmp_path):
    folder = tmp_path / WORKSPACE / ".lobster" / "sessions" / "s1"
    return lambda manager: manager.attach_session_dir(folder)


def _session_folder(workspace):
    (folder,) = (workspace / ".lobster" / "sessions").iterdir()
    return folder


def _export_again(workspace, name="nb2"):
    """What a new process does: open the session's folder again and export."""
    from lobster.core.runtime.data_manager import DataManagerV2

    again = DataManagerV2(
        workspace_path=workspace,
        session_dir=_session_folder(workspace),
    )
    return Path(again.export_notebook(name, "S", filter_strategy="replay"))


@needs_kernel
def test_a_native_session_writes_the_marker_and_matches(tmp_path, monkeypatch):
    workspace, notebook, raw = _geo_session(
        tmp_path, monkeypatch, on_manager=_persisted(tmp_path)
    )
    metadata = json.loads((_session_folder(workspace) / "metadata.json").read_text())
    assert metadata["record_format"]
    assert "migrated" not in _base.replay_commands._manifest(notebook)
    result = _verify_real(workspace, notebook, raw)
    assert result.exit_code == 0, (result.output, result.exception)
    assert json.loads(result.stdout)["verdict"] == "MATCH"


@needs_kernel
def test_a_native_session_opened_again_is_not_migrated(tmp_path, monkeypatch):
    workspace, notebook, raw = _geo_session(
        tmp_path, monkeypatch, on_manager=_persisted(tmp_path)
    )
    again = _export_again(workspace)
    assert "migrated" not in _base.replay_commands._manifest(again)
    metadata = json.loads((_session_folder(workspace) / "metadata.json").read_text())
    assert metadata["record_format"]


@needs_kernel
def test_the_same_session_without_the_marker_is_migrated_and_never_matches(
    tmp_path, monkeypatch
):
    workspace, notebook, raw = _geo_session(
        tmp_path, monkeypatch, on_manager=_persisted(tmp_path)
    )
    meta = _session_folder(workspace) / "metadata.json"
    written = json.loads(meta.read_text())
    del written["record_format"]
    meta.write_text(json.dumps(written))
    migrated = _export_again(workspace)
    block = _base.replay_commands._manifest(migrated)["migrated"]
    assert block["lobster"] == written["software_versions"]["lobster"]
    result = _verify_real(workspace, migrated, raw)
    out = json.loads(result.stdout)
    assert result.exit_code == 3, (result.output, result.exception)
    assert out["verdict"] == "NOT_COMPARED"
    files = out["report"]["migrated"]["files"]
    assert files and all(f["status"] in ("MATCH", "MISMATCH") for f in files)
