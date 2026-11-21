"""A notebook whose code names a path outside the replay's folder is never run.

The replay runs the notebook in its own folder. A code cell that names an
absolute path (or a relative one that climbs out with ``..``) reads or writes a
file the replay does not own: the session's workspace, a copy's original, or a
user's file elsewhere. Such a notebook is REFUSED before anything runs, and the
reason names the cell and the path. Not covered: a path the code builds while it
runs.

Exempt: the replay's own prelude, the parameters the replay replaces, and a read
of a saved modality by its typed path through a reader the custom cell points at
this notebook's result (it never opens the file).
"""

import hashlib
import json
import shutil
from pathlib import Path

import nbformat
import pytest

from lobster.cli_internal.commands.light import replay_commands
from tests.unit.cli.test_replay_commands import app, needs_kernel, runner

pytestmark = pytest.mark.unit


def _notebook(tmp_path, *cells, parameters=None):
    nb = nbformat.v4.new_notebook()
    if parameters is not None:
        cell = nbformat.v4.new_code_cell(parameters)
        cell.metadata["tags"] = ["parameters"]
        nb.cells.append(cell)
    nb.cells.append(nbformat.v4.new_markdown_cell("## Step 1: a step"))
    nb.cells.extend(nbformat.v4.new_code_cell(source) for source in cells)
    path = tmp_path / "nb.ipynb"
    nbformat.write(nb, path)
    return path


def _refusal(tmp_path, *cells, parameters=None):
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    notebook = _notebook(tmp_path, *cells, parameters=parameters)
    return replay_commands._path_refusal(notebook, workspace.resolve())


OUTSIDE = "/srv/lab/raw/counts.csv"


@pytest.mark.parametrize(
    "code",
    [
        f"import pandas as pd\nd = pd.read_csv({OUTSIDE!r})\n",
        f"from pathlib import Path\nPath({OUTSIDE!r}).write_text('x')\n",
        f"open({OUTSIDE!r}, 'w').write('x')\n",
        "import os\np = os.path.join('/srv/lab/raw', 'counts.csv')\n",
        "n = 'counts'\np = f'/srv/lab/raw/{n}.csv'\n",
        "p = 'C:\\\\lab\\\\raw\\\\counts.csv'\n",
        "p = '~/lab/counts.csv'\n",
        "!cp /srv/lab/raw/counts.csv here.csv\n",
    ],
    ids=["read", "Path", "open", "join", "f-string", "windows", "home", "shell"],
)
def test_a_path_outside_the_replay_is_refused_and_named(tmp_path, code):
    reason = _refusal(tmp_path, "x = 1\n", code)
    assert reason is not None
    assert reason.startswith("Code cell 2 uses the path "), reason
    assert "outside this replay's folder" in reason
    assert reason.endswith("The notebook was not run."), reason


def test_a_path_in_the_workspace_is_named_from_the_workspace(tmp_path):
    workspace = (tmp_path / "ws").resolve()
    code = f"results.to_csv({str(workspace / 'exports' / 'de.csv')!r})\n"
    assert _refusal(tmp_path, code) == (
        "Code cell 1 reads or writes exports/de.csv in the workspace; a replay uses "
        "only its own folder. The notebook was not run."
    )


def test_a_relative_path_that_climbs_out_is_refused(tmp_path):
    # the replay's folder is <workspace>/replays/<run>: ../.. is the workspace
    reason = _refusal(tmp_path, "open('../../notes.txt', 'w').write('x')\n")
    assert reason == (
        "Code cell 1 reads or writes notes.txt in the workspace; a replay uses only "
        "its own folder. The notebook was not run."
    )
    assert "outside this replay's folder" in _refusal(
        tmp_path, "open('../../../elsewhere.txt').read()\n"
    )


@pytest.mark.parametrize(
    "code",
    [
        "res = run(adata, formula='~condition')\n",
        "p = f'{root}/data/{name}.h5ad'\n",
        "p = 'data/a/../b.h5ad'\n",
        "print('...')\nkey = '/'.join(['a', 'b'])\n",
        "url = f'{base}/compound/{cid}/JSON'\n",
        "x = 1\n(\n",  # not Python: the kernel cannot run it either
    ],
    ids=["formula", "f-string-root", "inner-dots", "slash", "url", "unparsable"],
)
def test_text_that_is_no_path_outside_is_admitted(tmp_path, code):
    assert _refusal(tmp_path, code) is None


def test_the_parameters_the_replay_replaces_are_admitted(tmp_path):
    assert (
        _refusal(
            tmp_path,
            "x = 1\n",
            parameters=(
                "input_data = 'in.h5ad'\noutput_prefix = 'out'\n"
                "workspace_path = '/home/me/session_ws'\n"
            ),
        )
        is None
    )
    # the same name in another cell is code the notebook runs
    assert "/home/me" in _refusal(tmp_path, "workspace_path = '/home/me/session_ws'\n")
    # any other parameter is code the notebook runs with
    assert "/srv/lab" in _refusal(
        tmp_path, "x = 1\n", parameters="table = '/srv/lab/t.csv'\n"
    )


def test_the_custom_cell_prelude_and_typed_reads_of_its_modalities_are_admitted(
    tmp_path,
):
    """The session's workspace is the prelude's argument; a typed path to a
    modality the cell reads is served this notebook's result."""
    ws = "/home/me/session_ws"
    cell = (
        f"_lobster_reads(1, {ws!r}, {{'x_norm': adata}}, ['ref'], runs_on=None)\n"
        "import anndata, h5py, scanpy as sc\n"
        f"a = anndata.read_h5ad({ws + '/data/x_norm.h5ad'!r})\n"
        f"b = sc.read({ws + '/modalities/x_norm.h5ad'!r})\n"
        f"c = h5py.File({ws + '/data/x_norm.h5ad'!r}, 'r')\n"
        "d = sc.read_h5ad('/tmp/workspace/data/ref.h5ad')\n"
    )
    from lobster.core.notebooks.exporter import CUSTOM_READS_HELPER

    assert _refusal(tmp_path, CUSTOM_READS_HELPER, cell) is None


def test_the_check_reads_the_code_not_its_text(tmp_path):
    """Names the session chose appear in comments (``# the session stored it as
    '<name>'``): a comment is never a path, and no text turns the check off."""
    assert _refusal(tmp_path, "adata = adata_qc  # stored as '/srv/x.h5ad'\n") is None
    lookalike = "# def _lobster_target(\nopen('/srv/lab/x.csv').read()\n"
    assert "/srv/lab/x.csv" in _refusal(tmp_path, lookalike)


@pytest.mark.parametrize(
    "use",
    [
        "open({p!r}, 'w').write('x')\n",  # open() is not pointed at the result
        "h5py.File({p!r}, 'w')\n",  # a write
        "anndata.read_h5ad({other!r})\n",  # a name the cell does not read
    ],
    ids=["open", "h5py-write", "other-name"],
)
def test_a_typed_path_the_prelude_does_not_serve_is_refused(tmp_path, use):
    ws = "/home/me/session_ws"
    cell = (
        f"_lobster_reads(1, {ws!r}, {{'x_norm': adata}}, [], runs_on=None)\n"
        "import anndata, h5py\n"
        + use.format(p=ws + "/data/x_norm.h5ad", other=ws + "/data/earlier.h5ad")
    )
    assert _refusal(tmp_path, cell) is not None


def test_the_replays_own_input_may_be_named(tmp_path):
    """An import step re-reads the file it imported: it is the replay's input."""
    source = tmp_path / "raw" / "counts.csv"
    source.parent.mkdir()
    source.write_text("g,a\n")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    notebook = _notebook(tmp_path, f"d = pd.read_csv({str(source)!r})\n")
    assert replay_commands._path_refusal(notebook, workspace, source) is None
    assert replay_commands._path_refusal(notebook, workspace, None) is not None


def _load_cell(path):
    from lobster.services.data_management.modality_management_service import (
        ModalityManagementService,
    )

    service = object.__new__(ModalityManagementService)
    return service._create_load_ir("x", path, "transcriptomics_bulk", "custom").render()


LOADED = "/home/me/data/pbmc.h5ad"


def test_the_load_cell_naming_where_the_session_loaded_from_is_admitted(tmp_path):
    """load_modality's cell opens the session's file read-only to print where it
    came from; its object is never used. Admitted only as the exporter writes it."""
    assert _refusal(tmp_path, _load_cell(LOADED)) is None


@pytest.mark.parametrize(
    "change",
    [
        lambda cell: cell + "\nprint('one more line')\n",
        lambda cell: cell.replace(
            "ad.read_h5ad(_src)", "ad.read_h5ad(_src, backed='r+')"
        ),
        lambda cell: cell.replace(
            "_loaded = ad.read_h5ad(_src)",
            "_loaded = ad.read_h5ad(_src); _loaded.write_h5ad(_src)",
        ),
    ],
    ids=["added-line", "backed-write", "writes"],
)
def test_a_load_cell_that_is_not_the_exporters_is_counted(tmp_path, change):
    reason = _refusal(tmp_path, change(_load_cell(LOADED)))
    assert reason is not None and LOADED in reason, reason


def test_a_load_cell_whose_object_a_later_cell_uses_is_counted(tmp_path):
    """The read is allowed because nothing uses what it read."""
    reason = _refusal(tmp_path, _load_cell(LOADED), "adata = _loaded.copy()\n")
    assert reason is not None and LOADED in reason, reason


# --- sessions, replayed --------------------------------------------------------

te = pytest.importorskip("lobster.agents.transcriptomics.transcriptomics_expert")
de_module = pytest.importorskip("lobster.agents.transcriptomics.de_analysis_expert")

from lobster.testing.provenance_replay import factory_tools  # noqa: E402
from tests.unit.cli.test_replay_custom_code import (  # noqa: E402
    _add_to_custom_code,
    _session,
)
from tests.unit.cli.test_replay_custom_named_results import _bulk  # noqa: E402


def _state(path):
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns


def _verify(notebook, source, workspace):
    result = runner.invoke(
        app,
        ["--verify", str(notebook), "--input", str(source),
         "-w", str(workspace), "--json"],
    )  # fmt: skip
    return result.exit_code, json.loads(result.stdout)


def _nothing_ran(workspace):
    replays = workspace / replay_commands.REPLAYS_DIR
    return not list(replays.glob("*/executed.ipynb"))


@needs_kernel
@pytest.mark.parametrize("where", ["workspace", "copy"])
def test_a_de_export_to_the_sessions_exports_is_refused_and_nothing_runs(
    tmp_path, monkeypatch, where
):
    """export_de_results writes the session's exports/ by its absolute path: on
    the workspace or on a copy of it, the replay runs nothing and the session's
    file is untouched."""
    dm, source = _bulk(tmp_path, monkeypatch)
    tools = factory_tools(de_module.de_analysis_expert, de_module.__name__, dm)
    tools["run_bulk_de_direct"].invoke(
        {"modality_name": "bulk", "group_key": "condition", "group1": "ctrl",
         "group2": "treat", "save_result": True}
    )  # fmt: skip
    tools["export_de_results"].invoke(
        {"modality_name": "bulk_de_ctrl_vs_treat", "output_format": "csv"}
    )
    (exported,) = (dm.workspace_path / "exports").glob("de_results_*.csv")
    notebook = dm.export_notebook("nb", "S", filter_strategy="successful")
    assert str(exported) in notebook.read_text()  # the shape this row is about
    before = _state(exported)
    workspace = dm.workspace_path
    if where == "copy":
        workspace = tmp_path / "ws_copy"
        shutil.copytree(dm.workspace_path, workspace)
        notebook = workspace / notebook.relative_to(dm.workspace_path)
    code, out = _verify(notebook, source, workspace)
    assert code == 4 and out["verdict"] == "REFUSED", out
    reason = out["report"]["refusal"]["reason"]
    if where == "workspace":
        assert f"reads or writes exports/{exported.name} in the workspace" in reason
    else:
        assert f"uses the path {exported}, outside this replay's folder" in reason
    assert _nothing_ran(workspace)
    assert _state(exported) == before


@needs_kernel
def test_an_extract_and_export_to_the_sessions_exports_is_refused(
    tmp_path, monkeypatch
):
    """extract_and_export_de_results writes exports/ by its absolute path too."""
    dm, source = _bulk(tmp_path, monkeypatch)
    tools = factory_tools(de_module.de_analysis_expert, de_module.__name__, dm)
    tools["run_bulk_de_direct"].invoke(
        {"modality_name": "bulk", "group_key": "condition", "group1": "ctrl",
         "group2": "treat", "save_result": True}
    )  # fmt: skip
    tools["extract_and_export_de_results"].invoke(
        {"modality_name": "bulk_de_ctrl_vs_treat", "shrink_lfc": False}
    )
    exports = dm.workspace_path / "exports"
    files = {p: _state(p) for p in exports.rglob("*") if p.is_file()}
    notebook = dm.export_notebook("nb", "S", filter_strategy="successful")
    assert any(str(p) in notebook.read_text() for p in files), files
    code, out = _verify(notebook, source, dm.workspace_path)
    assert code == 4 and out["verdict"] == "REFUSED", out
    assert "in the workspace; a replay uses only its own folder" in (
        out["report"]["refusal"]["reason"]
    )
    assert _nothing_ran(dm.workspace_path)
    assert {p: _state(p) for p in files} == files


@needs_kernel
def test_a_session_that_loaded_its_file_through_the_service_still_replays(
    tmp_path, monkeypatch
):
    """The replay notebook carries load_modality's cell with the session's
    absolute path; replayed on a copy of the input elsewhere, it still runs."""
    from lobster.services.data_management.modality_management_service import (
        ModalityManagementService,
    )

    monkeypatch.chdir(tmp_path)
    dm, source = _bulk(tmp_path, monkeypatch)
    dm.remove_modality("bulk")
    adata, stats, ir = ModalityManagementService(dm).load_modality(
        "bulk", str(source), "transcriptomics_bulk"
    )
    dm.store_modality("bulk", adata) if "bulk" not in dm.list_modalities() else None
    dm.log_tool_usage(
        tool_name="load_modality", parameters={"modality_name": "bulk"},
        description="load", ir=ir,
    )  # fmt: skip
    tools = factory_tools(te.transcriptomics_expert, te.__name__, dm)
    tools["filter_bulk_genes"].invoke(
        {"modality_name": "bulk", "min_counts": 10, "min_samples": 3}
    )
    dm.save_modality("bulk_filtered", "bulk_filtered.h5ad")
    notebook = dm.export_notebook("nb", "S", filter_strategy="replay")
    assert f"_src = Path({str(source)!r})" in notebook.read_text()
    copy = tmp_path / "elsewhere" / "bulk.h5ad"
    copy.parent.mkdir()
    shutil.copy(source, copy)
    code, out = _verify(notebook, copy, dm.workspace_path)
    assert out["verdict"] in ("MATCH", "MISMATCH"), out  # run, not refused
    # shortcut: the service load adds uns/dataset_id and uns/dataset_type that the
    # notebook's read of the input does not; a replay-fidelity gap of its own.
    diffs = [d for slot in out["report"]["slots"] for d in slot.get("diffs", [])]
    assert all("/uns/dataset_" in d for d in diffs), diffs


@needs_kernel
def test_custom_code_that_reads_a_workspace_file_by_its_path_is_refused(
    tmp_path, monkeypatch
):
    dm, source, notebook = _session(tmp_path, monkeypatch, "result = 1\n")
    notes = dm.workspace_path / "notes.csv"
    notes.write_text("a\n1\n")
    read = f"import pandas as pd\n_n = pd.read_csv({str(notes)!r})\n"
    code, out = _verify(
        _add_to_custom_code(notebook, read, tmp_path), source, dm.workspace_path
    )
    assert code == 4 and out["verdict"] == "REFUSED", out
    assert "reads or writes notes.csv in the workspace" in (
        out["report"]["refusal"]["reason"]
    )
    assert _nothing_ran(dm.workspace_path)


@needs_kernel
def test_custom_code_that_reads_a_folder_outside_the_workspace_is_refused(
    tmp_path, monkeypatch
):
    """A user's original outside the workspace: the replay does not read it; the
    reason says to import it in the session."""
    dm, source, notebook = _session(tmp_path, monkeypatch, "result = 1\n")
    outside = tmp_path / "raw10x"
    outside.mkdir()
    read = f"import os\n_f = os.listdir({str(outside)!r})\n"
    code, out = _verify(
        _add_to_custom_code(notebook, read, tmp_path), source, dm.workspace_path
    )
    assert code == 4 and out["verdict"] == "REFUSED", out
    reason = out["report"]["refusal"]["reason"]
    assert f"uses the path {outside}, outside this replay's folder" in reason
    assert "imported in the session" in reason
    assert _nothing_ran(dm.workspace_path)


@needs_kernel
def test_custom_code_that_climbs_out_of_the_replay_is_refused(tmp_path, monkeypatch):
    dm, source, notebook = _session(tmp_path, monkeypatch, "result = 1\n")
    notes = dm.workspace_path / "notes.txt"
    notes.write_text("SESSION\n")
    before = _state(notes)
    write = "open('../../notes.txt', 'w').write('REPLAY')\n"
    code, out = _verify(
        _add_to_custom_code(notebook, write, tmp_path), source, dm.workspace_path
    )
    assert code == 4 and out["verdict"] == "REFUSED", out
    assert "reads or writes notes.txt in the workspace" in (
        out["report"]["refusal"]["reason"]
    )
    assert _nothing_ran(dm.workspace_path) and _state(notes) == before


@needs_kernel
def test_a_custom_notebook_with_no_path_outside_still_replays(tmp_path, monkeypatch):
    """The prelude names the session's workspace; the notebook is still run."""
    dm, source, notebook = _session(tmp_path, monkeypatch, "result = 1\n")
    assert str(dm.workspace_path) in notebook.read_text()
    code, out = _verify(notebook, source, dm.workspace_path)
    assert out["verdict"] == "MATCH", out


def test_the_reasons_have_their_chinese_lines():
    root = Path(replay_commands.__file__).resolve().parents[4]
    catalogs = [
        root / "lobster" / "i18n" / "phrases.zh.json",
        root / "lobster-tui" / "internal" / "i18n" / "catalogs" / "phrases.zh.json",
    ]
    for reason in (
        "Code cell {n} reads or writes {path} in the workspace; a replay uses only "
        "its own folder. The notebook was not run.",
        "Code cell {n} uses the path {path}, outside this replay's folder; a replay "
        "reads and writes only its own folder, so a file the notebook reads must be "
        "imported in the session. The notebook was not run.",
    ):
        for catalog in catalogs:
            assert reason in json.loads(catalog.read_text()), (catalog, reason)
