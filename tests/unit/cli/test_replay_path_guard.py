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
import os
import shutil
from pathlib import Path

import nbformat
import pytest

from lobster.cli_internal.commands.light import replay_commands
from lobster.core.notebooks.exporter import CUSTOM_READS_HELPER
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


CUSTOM_HEAD = (
    "# Workspace setup (a replay runs in its own folder)\n"
    "from pathlib import Path\nimport sys\n\n"
    "WORKSPACE = Path(workspace_path)\nsys.path.append(str(WORKSPACE))\n\n"
    "OUTPUT_DIR = WORKSPACE / 'exports'\n"
    "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
    "(WORKSPACE / 'data').mkdir(exist_ok=True)\n"
    "_lobster_reads(1, '/home/me/session_ws', {'x': adata}, [], runs_on=None)\n\n"
    "_lobster_held, adata = adata, (adata.copy() if adata is not None else None)\n\n"
    "# User code\n"
)
CUSTOM_TAIL = "\n_lobster_reads_done()\nadata = _lobster_held\n"


def _custom_refusal(tmp_path, *cells):
    """The refusal for cells after the exporter's helper cell (cell 1)."""
    return _refusal(tmp_path, CUSTOM_READS_HELPER, *cells)


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
    ],
    ids=["read", "Path", "open", "join", "f-string", "windows", "home"],
)
def test_a_path_outside_the_replay_is_refused_and_named(tmp_path, code):
    reason = _refusal(tmp_path, "x = 1\n", code)
    assert reason is not None
    assert reason.startswith("Code cell 2 uses the path "), reason
    assert "outside this replay's folder" in reason
    assert reason.endswith("The notebook was not run."), reason


def test_a_path_in_a_custom_cells_shell_command_is_refused(tmp_path):
    """Custom code may run commands (the user's code); the paths it names are
    still counted, word by word."""
    command = (
        "import subprocess\n"
        "subprocess.run('cp /srv/lab/raw/counts.csv here.csv', shell=True)\n"
    )
    reason = _custom_refusal(tmp_path, CUSTOM_HEAD + command + CUSTOM_TAIL)
    assert reason is not None
    assert reason.startswith("Code cell 2 uses the path /srv/lab/raw/counts.csv")


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
    cell = CUSTOM_HEAD.replace("{'x': adata}, []", "{'x_norm': adata}, ['ref']") + (
        "import anndata, h5py, scanpy as sc\n"
        f"a = anndata.read_h5ad({ws + '/data/x_norm.h5ad'!r})\n"
        f"b = sc.read({ws + '/modalities/x_norm.h5ad'!r})\n"
        f"c = h5py.File({ws + '/data/x_norm.h5ad'!r}, 'r')\n"
        "d = sc.read_h5ad('/tmp/workspace/data/ref.h5ad')\n"
    )
    assert _custom_refusal(tmp_path, cell) is None


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
    cell = CUSTOM_HEAD.replace("{'x': adata}", "{'x_norm': adata}") + (
        "import anndata, h5py\n"
        + use.format(p=ws + "/data/x_norm.h5ad", other=ws + "/data/earlier.h5ad")
    )
    served = CUSTOM_HEAD.replace("{'x': adata}", "{'x_norm': adata}") + (
        f"import anndata\na = anndata.read_h5ad({ws + '/data/x_norm.h5ad'!r})\n"
    )
    assert _custom_refusal(tmp_path, served) is None  # the same cell, as served
    assert _custom_refusal(tmp_path, cell) is not None


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
    came from; its object is never used. Admitted only as the exporter writes it,
    and only while that file is not on this machine (the session ran elsewhere)."""
    assert not os.path.exists(LOADED)
    assert _refusal(tmp_path, _load_cell(LOADED)) is None


IMPORT_TEXT = "so a file the notebook reads must be imported in the session"


def _outside_file(tmp_path):
    outside = tmp_path / "lab" / "pbmc.h5ad"
    outside.parent.mkdir()
    outside.write_bytes(b"not read")
    return outside


def test_a_load_cell_naming_a_file_on_this_machine_is_refused(tmp_path):
    """The file the session loaded is here, outside the replay's folder: the cell
    would open it, so the notebook is refused as any read of it is."""
    outside = _outside_file(tmp_path)
    reason = _refusal(tmp_path, _load_cell(str(outside)))
    assert reason is not None and str(outside) in reason and IMPORT_TEXT in reason


def test_a_load_cell_whose_link_leads_to_a_file_here_is_refused(tmp_path):
    """Links are followed: a path in the replay's folder, or a missing one, that
    leads to a file outside it is that file."""
    outside = _outside_file(tmp_path)
    link = tmp_path / "ws" / replay_commands.REPLAYS_DIR / "run" / "pbmc.h5ad"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    elsewhere = tmp_path / "links" / "pbmc.h5ad"
    elsewhere.parent.mkdir()
    elsewhere.symlink_to(outside)
    for text in ("pbmc.h5ad", str(elsewhere)):
        reason = _refusal(tmp_path, _load_cell(text))
        assert reason is not None and IMPORT_TEXT in reason, (text, reason)


def test_a_load_cell_that_climbs_out_to_a_file_here_is_refused(tmp_path):
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "pbmc.h5ad").write_bytes(b"not read")
    outside = _outside_file(tmp_path)
    reason = _refusal(tmp_path, _load_cell("../../pbmc.h5ad"))
    assert reason is not None and "pbmc.h5ad in the workspace" in reason, reason
    reason = _refusal(tmp_path, _load_cell(f"../../../lab/{outside.name}"))
    assert reason is not None and IMPORT_TEXT in reason, reason


@pytest.mark.parametrize(
    "text",
    ["/app/workspace/data/pbmc.h5ad", "pbmc.h5ad", "data/pbmc.h5ad",
     "../../../nowhere/pbmc.h5ad", "~/nowhere/pbmc.h5ad"],
)  # fmt: skip
def test_a_load_cell_naming_a_file_not_here_is_admitted(tmp_path, text):
    """A path from the machine the session ran on (a Cloud worker), or one in the
    replay's own folder: the cell only prints that the file is not present."""
    assert _refusal(tmp_path, _load_cell(text)) is None


def test_a_load_cell_naming_a_file_the_replay_may_read_is_admitted(tmp_path):
    """A file in the replay's own folder, or the replay's input, is read as any
    cell may read it."""
    inside = tmp_path / "ws" / replay_commands.REPLAYS_DIR / "run" / "pbmc.h5ad"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"in the folder")
    assert _refusal(tmp_path, _load_cell("pbmc.h5ad")) is None
    outside = _outside_file(tmp_path)
    notebook = _notebook(tmp_path, _load_cell(str(outside)))
    workspace = (tmp_path / "ws").resolve()
    assert replay_commands._path_refusal(notebook, workspace, outside) is None


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
def test_a_session_whose_loaded_file_is_here_is_refused_and_never_opened(
    tmp_path, monkeypatch
):
    """The replay notebook carries load_modality's cell with the session's
    absolute path. On the machine that file is on, the replay refuses before any
    kernel starts: the file, made unreadable, is not opened."""
    notebook, source, dm_workspace = _service_loaded(tmp_path, monkeypatch)
    copy = tmp_path / "elsewhere" / "bulk.h5ad"
    copy.parent.mkdir()
    shutil.copy(source, copy)
    before = _state(source)
    source.chmod(0)
    try:
        code, out = _verify(notebook, copy, dm_workspace)
    finally:
        source.chmod(0o644)
    assert code == 4 and out["verdict"] == "REFUSED", out
    reason = out["report"]["refusal"]["reason"]
    assert str(source) in reason and IMPORT_TEXT in reason, reason
    assert _nothing_ran(dm_workspace) and _state(source) == before


@needs_kernel
def test_a_session_that_loaded_its_file_through_the_service_still_replays(
    tmp_path, monkeypatch
):
    """Replayed where the session's file is not (another machine), on a copy of
    the input, the notebook still runs."""
    notebook, source, workspace = _service_loaded(tmp_path, monkeypatch)
    copy = tmp_path / "elsewhere" / "bulk.h5ad"
    copy.parent.mkdir()
    shutil.move(source, copy)
    code, out = _verify(notebook, copy, workspace)
    assert out["verdict"] in ("MATCH", "MISMATCH"), out  # run, not refused
    # shortcut: the service load adds uns/dataset_id and uns/dataset_type that the
    # notebook's read of the input does not; a replay-fidelity gap of its own.
    diffs = [d for slot in out["report"]["slots"] for d in slot.get("diffs", [])]
    assert all("/uns/dataset_" in d for d in diffs), diffs


def _service_loaded(tmp_path, monkeypatch):
    """A replay notebook of a session that loaded its input through the service."""
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
    return notebook, source, dm.workspace_path


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


# --- the export cell and code that runs commands (L136) -------------------------

NOT_EXPORT = (
    "Code cell 1 is not the code the export step writes; the notebook was not run."
)
A1 = (
    "results_df = adata.uns['de_results_x']['results_df']\n"
    "results_df = results_df.sort_values('padj')\n"
    "results_df.to_csv('exports/de.csv', index=True)"
)


def _a2(padj=True, lfc=True, method="to_csv"):
    return (
        "# Extract and export DE results\n"
        "results_df = adata.uns['de_results_x']['results_df'].copy()\n"
        + ("results_df = results_df[results_df['padj'] <= 0.05]\n" if padj else "")
        + (
            "results_df = results_df[abs(results_df['log2FoldChange']) >= 1.0]\n"
            if lfc
            else ""
        )
        + "results_df = results_df.sort_values('padj')\n"
        f"results_df.{method}('exports/de.xlsx', index=True)"
    )


@pytest.mark.parametrize(
    "cell",
    [
        A1,
        A1.replace("to_csv", "to_excel"),
        _a2(),
        _a2(padj=False),
        _a2(lfc=False),
        _a2(padj=False, lfc=False, method="to_excel"),
    ],
    ids=["A1", "A1-excel", "A2", "A2-lfc", "A2-padj", "A2-none"],
)
def test_an_export_cell_as_its_step_writes_it_is_admitted(tmp_path, cell):
    assert _refusal(tmp_path, cell) is None


def test_other_cells_that_use_results_df_are_not_export_cells(tmp_path):
    """The pathway templates keep their table in results_df too."""
    pathway = (
        "results_df = enr.results\n"
        "significant = results_df[results_df['Adjusted P-value'] < 0.05]\n"
        "summary = {'results': results_df.to_dict('records')}\n"
    )
    assert _refusal(tmp_path, pathway) is None
    # filter_de_results opens like an export and writes nothing
    filtered = (
        "results_df = adata.uns['de_results_x']['results_df']\n"
        "mask = (results_df['padj'] < 0.05) & (abs(results_df['log2FoldChange']) > 1.0)\n"
        "filtered_df = results_df[mask]\n"
    )
    assert _refusal(tmp_path, filtered) is None


@pytest.mark.parametrize(
    "cell",
    [
        A1.replace("to_csv", "to_tsv"),
        # the reported payload: the method text runs two more statements
        A1.replace(
            "results_df.to_csv(",
            "results_df.to_csv('x.csv');open('marker.txt','w').write('x');"
            "results_df.to_csv(",
        ),
        # the path hidden in a comment, the cell otherwise without one
        A1.replace(
            "to_csv('exports/de.csv', index=True)",
            "to_csv('x.csv'); print(1) #('exports/de.csv', index=True)",
        ),
        A1.replace("results_df.to_csv", "(results_df.to_csv if 1 else exec)"),
        _a2().replace("<= 0.05", "<= float('0.05')"),
    ],
    ids=["method", "statements", "comment", "expression", "threshold"],
)
def test_an_export_cell_its_step_does_not_write_is_refused(tmp_path, cell):
    assert _refusal(tmp_path, cell) == NOT_EXPORT


@pytest.mark.parametrize(
    "code, call",
    [
        ("x = eval('1')\n", "eval"),
        ("exec('x = 1')\n", "exec"),
        ("m = __import__('os')\n", "__import__"),
        ("import subprocess\n", "import subprocess"),
        ("from importlib import import_module\n", "import importlib"),
        ("import os\nos.system('ls')\n", "os.system"),
        ("import os as o\no.popen('ls')\n", "os.popen"),
        ("!ls\n", "a shell command"),
    ],
    ids=["eval", "exec", "__import__", "subprocess", "importlib", "system", "alias",
         "shell"],
)  # fmt: skip
def test_a_cell_that_runs_commands_or_built_code_is_refused(tmp_path, code, call):
    assert _refusal(tmp_path, "x = 1\n", code) == (
        f"Code cell 2 runs a command or code it builds ({call}); the notebook was "
        "not run."
    )


def test_a_custom_cell_is_the_users_code_and_not_checked_for_commands(tmp_path):
    """Custom code is the user's (the session ran it): only its paths are checked."""
    user = (
        "import subprocess\nresults_df = adata.obs\n"
        "results_df.to_csv(OUTPUT_DIR / 'obs.csv')\n"
    )
    assert _custom_refusal(tmp_path, CUSTOM_HEAD + user + CUSTOM_TAIL) is None


def test_a_cell_is_custom_by_its_setup_not_by_a_call_it_holds(tmp_path):
    """A template value cannot write the custom cell's setup lines before the
    template's own: a cell holding a _lobster_reads call is not custom for it."""
    forged = (
        "_lobster_reads(1, '/home/me/session_ws', {'x': adata}, [], runs_on=None)\n"
        "x = eval('1')\n"
    )
    assert "(eval)" in _refusal(tmp_path, forged)
    # nor are its typed paths served
    read = (
        "_lobster_reads(1, '/home/me/session_ws', {'x': adata}, [], runs_on=None)\n"
        "import anndata\na = anndata.read_h5ad('/home/me/session_ws/data/x.h5ad')\n"
    )
    assert "/home/me/session_ws" in _refusal(tmp_path, read)


def _de_export_session(tmp_path, monkeypatch):
    dm, source = _bulk(tmp_path, monkeypatch)
    tools = factory_tools(de_module.de_analysis_expert, de_module.__name__, dm)
    tools["run_bulk_de_direct"].invoke(
        {"modality_name": "bulk", "group_key": "condition", "group1": "ctrl",
         "group2": "treat", "save_result": True}
    )  # fmt: skip
    tools["export_de_results"].invoke(
        {"modality_name": "bulk_de_ctrl_vs_treat", "output_format": "csv"}
    )
    notebook = dm.export_notebook("nb", "S", filter_strategy="successful")
    return dm, source, notebook


def _edit_export(notebook, payload, tmp_path):
    """The notebook as an injected output_format renders it: the text after
    results_df.to_ is the payload."""
    cells = json.loads(notebook.read_text())
    for cell in cells["cells"]:
        text = "".join(cell["source"])
        if cell["cell_type"] == "code" and "results_df.to_csv(" in text:
            cell["source"] = text.replace("results_df.to_csv(", payload, 1)
    edited = tmp_path / "injected.ipynb"
    edited.write_text(json.dumps(cells))
    return edited


@needs_kernel
@pytest.mark.parametrize(
    "payload",
    [
        # the b7 exploit (MATCH, marker written)
        "results_df.to_csv('/dev/null');open('marker.txt','w').write('x');"
        "results_df.to_csv(",
        # no path left in the code: the session's path is in a comment
        "results_df.to_csv('x.csv');open('marker.txt','w').write('x') #(",
    ],
    ids=["working-first-call", "comment"],
)
def test_an_injected_export_cell_is_refused_and_nothing_runs(
    tmp_path, monkeypatch, payload
):
    dm, source, notebook = _de_export_session(tmp_path, monkeypatch)
    code, out = _verify(
        _edit_export(notebook, payload, tmp_path), source, dm.workspace_path
    )
    assert code == 4 and out["verdict"] == "REFUSED", out
    assert "is not the code the export step writes" in (
        out["report"]["refusal"]["reason"]
    )
    assert _nothing_ran(dm.workspace_path)
    assert not list(dm.workspace_path.rglob("marker.txt"))


def test_the_new_reasons_have_their_chinese_lines():
    root = Path(replay_commands.__file__).resolve().parents[4]
    for catalog in (
        root / "lobster" / "i18n" / "phrases.zh.json",
        root / "lobster-tui" / "internal" / "i18n" / "catalogs" / "phrases.zh.json",
    ):
        phrases = json.loads(catalog.read_text())
        assert (
            "Code cell {n} is not the code the export step writes; the notebook was "
            "not run." in phrases
        )
        assert (
            "Code cell {n} runs a command or code it builds ({call}); the notebook "
            "was not run." in phrases
        )
        assert (
            "Code cell {n} is an older or changed copy of the cell that serves "
            "custom code its reads. Re-export the notebook; it was not run." in phrases
        )


# --- ways around the check (review probes) ---------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "open(b'/etc/passwd').read()\n",
        "import pandas as pd\npd.read_csv('file:///etc/passwd')\n",
        "import pandas as pd\npd.read_csv('FILE:/etc/passwd')\n",
        "import pandas as pd\npd.read_csv('~root/.profile')\n",
        "from pathlib import Path\nPath('/', 'etc', 'passwd').read_text()\n",
        "from pathlib import Path\n(Path('/') / 'etc' / 'passwd').read_text()\n",
        "import os\nopen(os.path.join('/', 'etc', 'passwd')).read()\n",
        "open('/' + 'etc/passwd').read()\n",
    ],
    ids=["bytes", "file-url", "file-url-short", "user-home", "Path-root", "Path-div",
         "join-root", "concat"],
)  # fmt: skip
def test_a_path_written_another_way_is_refused(tmp_path, code):
    reason = _refusal(tmp_path, code)
    assert reason is not None and "outside this replay's folder" in reason, reason


def test_a_lone_slash_and_a_formula_are_not_paths(tmp_path):
    cell = (
        "name = '/'.join(['a', 'b']).split('/')[0]\n"
        "design = '~condition'\nother = '~nobody_here_xyz/batch'\n"
    )
    assert _refusal(tmp_path, cell) is None


@pytest.mark.parametrize(
    "code, call",
    [
        ("%%time\nopen('/etc/passwd').read()\n", "%%time"),
        ("%%capture out\nopen('/etc/passwd').read()\n", "%%capture"),
        ("%time open('/etc/passwd').read()\n", "%time"),
        ("ip = get_ipython()\nip.system('ls')\n", "get_ipython"),
    ],
    ids=["time", "capture", "line-time", "shell-object"],
)
def test_a_magic_is_refused_in_any_cell(tmp_path, code, call):
    expected = f"runs a command or code it builds ({call})"
    assert expected in _refusal(tmp_path, code)
    if not code.startswith("%%"):  # a cell magic is one only as a cell's start
        assert expected in _custom_refusal(tmp_path, CUSTOM_HEAD + code + CUSTOM_TAIL)


@pytest.mark.parametrize(
    "code",
    [
        "import subprocess\nsubprocess.run(['bash', '-c', 'cat /etc/passwd'])\n",
        "import subprocess\nsubprocess.getstatusoutput('cat /etc/passwd')\n",
        "import subprocess\nsubprocess.Popen(args=['cat', '/etc/passwd'])\n",
    ],
    ids=["argv", "getstatusoutput", "keyword"],
)
def test_a_command_is_refused_and_a_custom_cells_command_paths_are_counted(
    tmp_path, code
):
    assert "(import subprocess)" in _refusal(tmp_path, code)
    reason = _custom_refusal(tmp_path, CUSTOM_HEAD + code + CUSTOM_TAIL)
    assert "uses the path /etc/passwd" in reason, reason


def test_only_written_values_of_the_first_parameters_cell_are_replaced(tmp_path):
    runs = "workspace_path = open('/etc/passwd').read()\n"
    assert "/etc/passwd" in _refusal(tmp_path, "x = 1\n", parameters=runs)
    built = "workspace_path = f'/etc/{1}'\n"
    assert "/etc/" in _refusal(tmp_path, "x = 1\n", parameters=built)
    workspace = tmp_path / "ws"
    second = _notebook(tmp_path, "open(workspace_path + '/passwd')\n",
                       parameters="input_data = 'x'\n")  # fmt: skip
    cells = json.loads(second.read_text())
    cells["cells"].insert(
        1,
        {"cell_type": "code", "source": "workspace_path = '/etc'\n", "outputs": [],
         "metadata": {"tags": ["parameters"]}, "execution_count": None},
    )  # fmt: skip
    second.write_text(json.dumps(cells))
    assert "/etc" in replay_commands._path_refusal(second, workspace.resolve())


SERVED_READ = (
    "import anndata\na = anndata.read_h5ad('/home/me/session_ws/data/x.h5ad')\n"
)


@pytest.mark.parametrize(
    "cells",
    [
        # no helper cell: the reads are nobody's to serve
        (CUSTOM_HEAD + SERVED_READ + CUSTOM_TAIL,),
        # a helper of its own
        ("def _lobster_reads(*a, **k):\n    pass\n",
         CUSTOM_HEAD + SERVED_READ + CUSTOM_TAIL),
        # the exporter's helper, then a cell that replaces part of it
        (CUSTOM_READS_HELPER, "def _lobster_target(path):\n    return path\n",
         CUSTOM_HEAD + SERVED_READ + CUSTOM_TAIL),
        # or undoes its patch of the readers
        (CUSTOM_READS_HELPER, "import anndata\nanndata.read_h5ad = _orig\n",
         CUSTOM_HEAD + SERVED_READ + CUSTOM_TAIL),
        # a changed copy of the helper
        (CUSTOM_READS_HELPER.replace("resolve()", "absolute()", 1),
         CUSTOM_HEAD + SERVED_READ + CUSTOM_TAIL),
    ],
    ids=["no-helper", "own-helper", "replaced", "unpatched", "changed-helper"],
)  # fmt: skip
def test_reads_are_served_only_by_the_exporters_own_helper(tmp_path, cells):
    assert _refusal(tmp_path, *cells) is not None
    assert _custom_refusal(tmp_path, CUSTOM_HEAD + SERVED_READ + CUSTOM_TAIL) is None


def test_an_older_helper_cell_is_named(tmp_path):
    older = CUSTOM_READS_HELPER.replace("resolve()", "absolute()", 1)
    assert _refusal(tmp_path, older, CUSTOM_HEAD + SERVED_READ + CUSTOM_TAIL) == (
        "Code cell 1 is an older or changed copy of the cell that serves custom "
        "code its reads. Re-export the notebook; it was not run."
    )


def test_a_served_name_that_is_a_path_is_not_served(tmp_path):
    head = CUSTOM_HEAD.replace("{'x': adata}", "{'../../../Users/me/secret': adata}")
    read = (
        "import anndata\n"
        "a = anndata.read_h5ad('/tmp/workspace/data/../../../Users/me/secret.h5ad')\n"
    )
    assert _custom_refusal(tmp_path, head + read + CUSTOM_TAIL) is not None
