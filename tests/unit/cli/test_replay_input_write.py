"""A cell may name the replay's input file only to read it.

The replay reads its input in place: a cell that names that file and writes
files is refused before the run, so the file is never changed. A cell that only
reads it (an import step re-reading the file it imported) is admitted.
"""

import json

import nbformat
import pytest

from lobster.cli_internal.commands.light import replay_commands
from tests.unit.cli import test_replay_commands as base
from tests.unit.cli.test_replay_path_guard import A1, _a2, _notebook

pytestmark = pytest.mark.unit

needs_kernel = base.needs_kernel
ws = base.ws


def _writes_input(n, name):
    return (
        f"Code cell {n} names the replay's input file {name} and writes files, "
        "so it could change that file; the notebook was not run."
    )


def _naming(cell, path):
    return cell.replace("'exports/de.csv'", repr(str(path))).replace(
        "'exports/de.xlsx'", repr(str(path))
    )


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "raw" / "counts.csv"
    path.parent.mkdir()
    path.write_text("g,a\n")
    (tmp_path / "ws").mkdir()
    return path


def _reason(tmp_path, source, *cells):
    notebook = _notebook(tmp_path, *cells)
    return replay_commands._path_refusal(notebook, tmp_path / "ws", source)


@pytest.mark.parametrize(
    "cell",
    [A1, A1.replace("to_csv", "to_excel"), _a2(), _a2(padj=False), _a2(lfc=False),
     _a2(padj=False, lfc=False, method="to_excel")],
    ids=["A1", "A1-excel", "A2", "A2-lfc", "A2-padj", "A2-none"],
)  # fmt: skip
def test_an_export_cell_writing_the_input_is_refused(tmp_path, source, cell):
    """The export step's own cell, exactly as it writes it, aimed at the input."""
    reason = _reason(tmp_path, source, _naming(cell, source))
    assert reason == _writes_input(1, "counts.csv"), reason


@pytest.mark.parametrize(
    "write",
    [
        "open({p!r}, 'w').write('x')",
        "open({p!r}, mode='a')",
        "open({p!r}, m)",
        "Path({p!r}).open('w')",
        "Path({p!r}).write_text('x')",
        "Path({p!r}).touch()",
        "Path({p!r}).unlink()",
        "adata.write_h5ad({p!r})",
        "adata.write({p!r})",
        "ad.read_h5ad({p!r}, backed='r+')",
        "h5py.File({p!r}, 'a')",
        "np.save({p!r}, x)",
        "plt.savefig({p!r})",
        "json.dump(x, open({p!r}, 'w'))",
        "os.remove({p!r})",
        "os.rename('x', {p!r})",
        "shutil.copy('x', {p!r})",
        "p = {p!r}\nd = pd.read_csv(p)\nd.to_csv(p)",
        "d = pd.read_csv({p!r})\nd.to_csv('out.csv')",
    ],
)
def test_a_cell_that_names_the_input_and_writes_is_refused(tmp_path, source, write):
    reason = _reason(tmp_path, source, write.format(p=str(source)))
    assert reason == _writes_input(1, "counts.csv"), reason


@pytest.mark.parametrize(
    "read",
    [
        "d = pd.read_csv({p!r})",
        "d = pd.read_csv({p!r}, index_col=0)\nd = d.T.copy()",
        "with open({p!r}) as f:\n    text = f.read()",
        "with open({p!r}, 'rb') as f:\n    data = f.read()",
        "Path({p!r}).read_text()",
        "a = ad.read_h5ad({p!r}, backed='r')",
        "f = h5py.File({p!r}, 'r')",
        "d = pd.read_csv({p!r})\nd = d.replace(0, 1).rename(columns=str)",
    ],
)
def test_a_cell_that_only_reads_the_input_is_admitted(tmp_path, source, read):
    assert _reason(tmp_path, source, read.format(p=str(source))) is None


def test_the_writing_cell_is_named_among_cells_that_read(tmp_path, source):
    reading = f"d = pd.read_csv({str(source)!r})"
    reason = _reason(tmp_path, source, reading, _naming(A1, source))
    assert reason == _writes_input(2, "counts.csv"), reason


def test_a_writing_cell_that_does_not_name_the_input_is_not_this_refusal(
    tmp_path, source
):
    assert _reason(tmp_path, source, A1) is None


# --- replayed ------------------------------------------------------------------

SETUP = (
    "import anndata as ad\nimport pandas as pd\n"
    "adata = ad.read_h5ad(input_data)\n"
    "adata.uns['de_results_x'] = {'results_df': pd.DataFrame("
    "{'padj': [0.1], 'log2FoldChange': [2.0]}, index=['g1'])}\n"
)


def _with_cell(notebook, cell):
    nb = nbformat.read(notebook, as_version=4)
    nb.cells.append(nbformat.v4.new_code_cell(cell))
    nbformat.write(nb, notebook)
    return notebook


@needs_kernel
@pytest.mark.parametrize("cell", [A1, _a2()], ids=["export", "extract-and-export"])
def test_an_export_to_the_input_is_refused_and_the_input_is_unchanged(ws, cell):
    target = (ws.parent / "input.h5ad").resolve()
    before = base._sha(target)
    notebook = base._notebook(ws / "notebooks" / "nb.ipynb", SETUP)
    _with_cell(notebook, _naming(cell, target))
    result = base._verify(ws, None, notebook=notebook)
    out = json.loads(result.stdout)
    assert (result.exit_code, out["verdict"]) == (4, "REFUSED"), out
    reason = base._report(ws).refusal.reason
    assert "names the replay's input file input.h5ad and writes files" in reason
    assert base._sha(target) == before
    assert not list((ws / "replays").glob("*/executed.ipynb"))


@needs_kernel
def test_an_export_to_the_imported_file_is_refused_without_input(tmp_path, monkeypatch):
    """With no --input, the replay reads the file the session imported, where the
    session saved it: a cell writing that file is refused the same way."""
    geo = pytest.importorskip("tests.unit.cli.test_replay_geo_anchor")
    workspace, notebook, _raw = geo._geo_session(tmp_path, monkeypatch)
    (imported,) = (workspace / "data").glob(f"{geo.RAW}")
    before = base._sha(imported)
    _with_cell(notebook, "import pandas as pd\n" + SETUP.split("input_data)\n")[1])
    _with_cell(notebook, _naming(A1, imported.resolve()))
    result = geo._verify_without_input(workspace, notebook)
    out = json.loads(result.stdout)
    assert (result.exit_code, out["verdict"]) == (4, "REFUSED"), out
    reason = out["report"]["refusal"]["reason"]
    assert f"names the replay's input file {geo.RAW} and writes files" in reason
    assert base._sha(imported) == before


@pytest.mark.parametrize("first", ["input", "other"])
def test_a_reading_cell_naming_the_input_still_has_its_other_paths_checked(
    tmp_path, source, first
):
    """Naming the input exempts that file only, not the cell."""
    reads = [f"d = pd.read_csv({str(source)!r})", "e = pd.read_csv('/srv/lab/x.csv')"]
    cell = "\n".join(reads if first == "input" else reads[::-1])
    reason = _reason(tmp_path, source, cell)
    assert reason is not None and "/srv/lab/x.csv" in reason, reason
