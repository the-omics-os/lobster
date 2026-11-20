"""A custom-code cell replays on the notebook's own results, never the session's files.

In the session, custom code that reads ``data/<modality>.h5ad`` gets a copy of the
modality in memory (the alias of ``_detect_required_modalities``). The exported cell
had no alias and set ``WORKSPACE`` to the session's workspace, so a replay either
failed on a relative path (the cwd is the replay folder) or read the session's
saved file through ``WORKSPACE``: a step after it then matched whatever the steps
before it computed. ``OUTPUT_DIR`` also pointed into the session's ``exports/``,
which a replay overwrote unseen.

Each session here is native: load ``x``, ``filter_and_normalize`` (saves
``x_filtered_normalized``), the custom cell on that modality, then
``filter_and_normalize`` on it (saves ``x_filtered_normalized_filtered_normalized``).
Skipped without the transcriptomics package or a Jupyter kernel.
"""

import asyncio
import json
import re
import shutil

import numpy as np
import pandas as pd
import pytest

from tests.unit.cli.test_replay_commands import app, needs_kernel, runner

te = pytest.importorskip("lobster.agents.transcriptomics.transcriptomics_expert")

import anndata as ad  # noqa: E402

from lobster.core.runtime.data_manager import DataManagerV2  # noqa: E402
from lobster.services.execution.custom_code_execution_service import (  # noqa: E402
    CustomCodeExecutionService,
)
from lobster.testing.provenance_replay import factory_tools  # noqa: E402
from lobster.tools.custom_code_tool import create_execute_custom_code_tool  # noqa: E402

pytestmark = [pytest.mark.unit, needs_kernel]

STEP_A = "x_filtered_normalized"
STEP_C = "x_filtered_normalized_filtered_normalized"
PERMISSIVE = {
    "min_genes_per_cell": 1,
    "min_cells_per_gene": 1,
    "max_genes_per_cell": 100000,
    "max_mito_percent": 100,
}


def _x(name="c"):
    rng = np.random.default_rng(3)
    return ad.AnnData(
        X=rng.poisson(3, (60, 120)).astype(np.float32),
        obs=pd.DataFrame(index=[f"{name}{i}" for i in range(60)]),
        var=pd.DataFrame(index=[f"G{i}" for i in range(120)]),
    )


def _custom(dm, code, modality_name=STEP_A):
    tool = create_execute_custom_code_tool(
        dm, CustomCodeExecutionService(dm), agent_name="transcriptomics_expert"
    )
    call = {
        "args": {
            "python_code": code,
            "modality_name": modality_name,
            "persist": True,
            "load_workspace_files": False,
            "description": "custom step",
        },
        "id": "call_1",
        "name": tool.name,
        "type": "tool_call",
    }
    reply = json.loads(asyncio.run(tool.ainvoke(call)).content)
    assert reply["success"], reply


def _session(tmp_path, monkeypatch, code, extra=None):
    """The native session; returns (data manager, input file, notebook)."""
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "in" / "x.h5ad"
    source.parent.mkdir()
    _x().write_h5ad(source)
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.load_modality("x", str(source), adapter="transcriptomics_single_cell")
    if extra:
        extra(dm)
    tools = factory_tools(te.transcriptomics_expert, te.__name__, dm)
    tools["filter_and_normalize"].invoke({"modality_name": "x", **PERMISSIVE})
    _custom(dm, code)
    tools["filter_and_normalize"].invoke({"modality_name": STEP_A, **PERMISSIVE})
    notebook = dm.export_notebook("custom", "custom", filter_strategy="replay")
    return dm, source, notebook


def _replay(dm, notebook, source):
    result = runner.invoke(
        app,
        ["--verify", str(notebook), "--input", str(source),
         "-w", str(dm.workspace_path), "--json"],
    )  # fmt: skip
    return json.loads(result.stdout)


def _diffs(out):
    return [d for slot in out["report"]["slots"] for d in slot.get("diffs", [])]


def _tamper_step_a(notebook, tmp_path):
    """A copy of the notebook whose first normalization uses another target sum."""
    cells = json.loads(notebook.read_text())
    for cell in cells["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and "normalize_total" in source:
            cell["source"] = re.sub(r"target_sum=[0-9.e+]+", "target_sum=7", source)
            break
    tampered = tmp_path / "tampered.ipynb"
    tampered.write_text(json.dumps(cells))
    return tampered


RELATIVE = (
    "import scanpy as sc\n"
    f"m = sc.read_h5ad('data/{STEP_A}.h5ad')\n"
    "adata.obs['total'] = np.asarray(m.X.sum(axis=1)).ravel()\n"
)
THROUGH_WORKSPACE = (
    "import anndata\n"
    f"adata = anndata.read_h5ad(WORKSPACE / 'data' / '{STEP_A}.h5ad')\n"
)


def test_a_relative_data_path_replays_to_match(tmp_path, monkeypatch):
    """The shape an agent writes: sc.read_h5ad('data/<modality>.h5ad')."""
    dm, source, notebook = _session(
        tmp_path, monkeypatch, "import numpy as np\n" + RELATIVE
    )
    out = _replay(dm, notebook, source)
    assert out["verdict"] == "MATCH", out


@pytest.mark.parametrize(
    "code", [RELATIVE, THROUGH_WORKSPACE], ids=["relative", "workspace"]
)
def test_a_step_after_the_custom_cell_is_recomputed_not_read(
    tmp_path, monkeypatch, code
):
    """With the step before the custom cell changed, the step after it differs too:
    the cell read the notebook's result, not the session's saved file."""
    dm, source, notebook = _session(
        tmp_path, monkeypatch, "import numpy as np\n" + code
    )
    out = _replay(dm, _tamper_step_a(notebook, tmp_path), source)
    assert out["verdict"] == "MISMATCH", out
    diffs = _diffs(out)
    assert any(f"files/{STEP_A}.h5ad/" in d for d in diffs), diffs
    assert any(f"files/{STEP_C}.h5ad/" in d for d in diffs), diffs


def _add_to_custom_code(notebook, code, tmp_path, name="edited.ipynb"):
    """A copy of the notebook with ``code`` put first in the custom cell's code
    (code a user could put there; the session never ran it)."""
    cells = json.loads(notebook.read_text())
    for cell in cells["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and "# User code\n" in source:
            cell["source"] = source.replace("# User code\n", "# User code\n" + code, 1)
    edited = tmp_path / name
    edited.write_text(json.dumps(cells))
    return edited


def test_a_typed_path_to_the_sessions_saved_file_reads_the_notebooks_result(
    tmp_path, monkeypatch
):
    """With the step before the custom cell changed, a typed path to the
    session's saved file still reads the notebook's (changed) result."""
    dm, source, notebook = _session(tmp_path, monkeypatch, "result = 1\n")
    typed = str(dm.data_dir / f"{STEP_A}.h5ad")
    check = (
        "import anndata\n"
        f"_s = anndata.read_h5ad({typed!r})\n"
        "assert abs(float(_s.X.sum()) - float(adata.X.sum())) < 1e-3, "
        "'read the session file'\n"
    )
    edited = _add_to_custom_code(_tamper_step_a(notebook, tmp_path), check, tmp_path)
    out = _replay(dm, edited, source)
    assert out["verdict"] == "MISMATCH", out
    assert "read the session file" not in json.dumps(out), out


def test_a_read_of_another_file_of_the_session_is_refused(tmp_path, monkeypatch):
    """Any other file in the session's workspace is never read: the replay stops
    and names it."""
    dm, source, notebook = _session(tmp_path, monkeypatch, "result = 1\n")
    other = dm.data_dir / "earlier.h5ad"
    _x("e").write_h5ad(other)
    edited = _add_to_custom_code(
        notebook, f"import anndata\ne = anndata.read_h5ad({str(other)!r})\n", tmp_path
    )
    out = _replay(dm, edited, source)
    assert out["verdict"] == "NOTEBOOK_FAILED", out
    notes = " ".join(out["notes"])
    assert "reads data/earlier.h5ad of the session's workspace" in notes, notes


def _upstream_read_session(tmp_path, monkeypatch):
    """Load x, filter_and_normalize x then its result, then custom code on x
    reading the first result's saved file (made two steps before), its result
    saved: the session stores it as x_custom."""
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "in" / "x.h5ad"
    source.parent.mkdir()
    _x().write_h5ad(source)
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.load_modality("x", str(source), adapter="transcriptomics_single_cell")
    tools = factory_tools(te.transcriptomics_expert, te.__name__, dm)
    tools["filter_and_normalize"].invoke({"modality_name": "x", **PERMISSIVE})
    tools["filter_and_normalize"].invoke({"modality_name": STEP_A, **PERMISSIVE})
    _custom(
        dm,
        "import anndata\n"
        f"saved = anndata.read_h5ad(WORKSPACE / 'data' / '{STEP_A}.h5ad')\n"
        "adata = saved.copy()\n"
        "adata.obs['from_upstream'] = 1\n",
        modality_name="x",
    )
    assert "x_custom" in dm.list_modalities()
    dm.save_modality("x_custom", "x_custom.h5ad")
    notebook = dm.export_notebook("custom", "custom", filter_strategy="replay")
    return dm, source, notebook


def test_a_custom_result_built_from_an_upstream_file_replays_to_match(
    tmp_path, monkeypatch
):
    dm, source, notebook = _upstream_read_session(tmp_path, monkeypatch)
    out = _replay(dm, notebook, source)
    assert out["verdict"] == "MATCH", out


def test_a_custom_result_built_from_an_upstream_file_follows_the_upstream(
    tmp_path, monkeypatch
):
    """The upstream step changed: the custom result differs too (never a slot
    compared against the session's own file)."""
    dm, source, notebook = _upstream_read_session(tmp_path, monkeypatch)
    out = _replay(dm, _tamper_step_a(notebook, tmp_path), source)
    assert out["verdict"] == "MISMATCH", out
    assert any("files/x_custom.h5ad/" in d for d in _diffs(out)), _diffs(out)


def test_a_relative_data_write_lands_in_the_replay_folder(tmp_path, monkeypatch):
    """A custom cell writing data/<name>.h5ad writes the replay's folder; the
    session's data/ is never written."""
    code = "adata.copy().write_h5ad('data/custom_out.h5ad')\n"
    dm, source, notebook = _session(tmp_path, monkeypatch, code)
    written = dm.data_dir / "custom_out.h5ad"
    before = written.read_bytes()
    out = _replay(dm, notebook, source)
    assert written.read_bytes() == before
    assert out["verdict"] == "NOT_COMPARED", out
    assert any(
        "data/custom_out.h5ad: written by the notebook" in n for n in out["notes"]
    )


def test_no_module_of_the_sessions_workspace_is_imported(tmp_path, monkeypatch):
    """A module in the session's workspace is not importable from the replay.

    Custom code imports only by import statements, which the notebook's imports
    cell runs before any custom cell; the custom cell's sys.path entry is the
    replay's folder, never the session's workspace."""

    def module(dm):
        (dm.workspace_path / "lobster_probe_mod.py").write_text("VALUE = 1\n")

    dm, source, notebook = _session(
        tmp_path,
        monkeypatch,
        "import lobster_probe_mod\n",
        extra=module,
    )
    out = _replay(dm, notebook, source)
    assert out["verdict"] == "NOTEBOOK_FAILED", out
    assert "No module named 'lobster_probe_mod'" in " ".join(out["notes"])


def test_the_bound_input_is_what_a_custom_cell_reads_for_it(tmp_path, monkeypatch):
    """The loaded input read by name reads the file given with --input (a copy in
    another folder), never the path the session loaded it from."""
    code = (
        "import numpy as np, scanpy as sc\n"
        "raw = sc.read_h5ad('data/x.h5ad')\n"
        "adata.obs['raw_total'] = np.asarray(raw[adata.obs_names].X.sum(axis=1)).ravel()\n"
    )
    dm, source, notebook = _session(tmp_path, monkeypatch, code)
    copy = tmp_path / "elsewhere" / "x.h5ad"
    copy.parent.mkdir()
    shutil.copy(source, copy)
    source.unlink()
    out = _replay(dm, notebook, copy)
    assert out["verdict"] == "MATCH", out


def test_a_name_the_notebook_does_not_compute_is_named(tmp_path, monkeypatch):
    """A modality the session loaded besides the input: the cell cannot read a
    recomputed copy, so the replay says which name, never a raw traceback and
    never the session's file."""
    ref = tmp_path / "ref.h5ad"
    _x("r").write_h5ad(ref)
    code = (
        "import numpy as np, scanpy as sc\n"
        "r = sc.read_h5ad('data/ref.h5ad')\n"
        "adata.uns['ref_cells'] = int(r.n_obs)\n"
    )

    def load_ref(dm):
        dm.load_modality("ref", str(ref), adapter="transcriptomics_single_cell")

    dm, source, notebook = _session(tmp_path, monkeypatch, code, extra=load_ref)
    out = _replay(dm, notebook, source)
    assert out["verdict"] in ("NOT_COMPARED", "NOTEBOOK_FAILED"), out
    notes = " ".join(out["notes"])
    assert "custom code reads 'ref'" in notes, notes
    assert "Traceback" not in notes, notes


def test_the_replay_writes_no_file_into_the_sessions_exports(tmp_path, monkeypatch):
    """OUTPUT_DIR is the replay's folder: the session's exports/ is untouched."""
    code = (
        "import pandas as pd\n"
        "pd.DataFrame({'n': [adata.n_obs]}).to_csv(OUTPUT_DIR / 'markers.csv')\n"
    )
    dm, source, notebook = _session(tmp_path, monkeypatch, code)
    exports = dm.workspace_path / "exports"
    (exports / "markers.csv").write_text("SESSION\n")
    out = _replay(dm, notebook, source)
    assert out["verdict"] == "MATCH", out
    assert (exports / "markers.csv").read_text() == "SESSION\n"
    assert sorted(p.name for p in exports.iterdir()) == ["markers.csv"]


def test_a_write_into_the_sessions_exports_is_refused(tmp_path, monkeypatch):
    """Custom code writing the session's exports/ by a typed path is detected."""
    dm, source, notebook = _session(tmp_path, monkeypatch, "result = 1\n")
    exports = dm.workspace_path / "exports"
    exports.mkdir(exist_ok=True)
    (exports / "markers.csv").write_text("SESSION\n")
    cells = json.loads(notebook.read_text())
    for cell in cells["cells"]:
        if cell["cell_type"] == "code" and "result = 1" in "".join(cell["source"]):
            cell["source"] = (
                "".join(cell["source"])
                + f"\nopen({str(exports / 'markers.csv')!r}, 'w').write('REPLAY')\n"
            )
    notebook.write_text(json.dumps(cells))
    out = _replay(dm, notebook, source)
    assert out["verdict"] == "REFUSED", out
    assert "exports/markers.csv" in json.dumps(out), out


def _input_read_session(tmp_path, monkeypatch):
    """A resumed session: x loaded from data/x.h5ad, filtered, then custom code on
    the result reading the loaded input through WORKSPACE, its result saved."""
    monkeypatch.chdir(tmp_path)
    pristine = tmp_path / "in" / "x.h5ad"
    pristine.parent.mkdir()
    _x().write_h5ad(pristine)
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(pristine, dm.data_dir / "x.h5ad")
    dm.load_modality(
        "x", str(dm.data_dir / "x.h5ad"), adapter="transcriptomics_single_cell"
    )
    tools = factory_tools(te.transcriptomics_expert, te.__name__, dm)
    tools["filter_and_normalize"].invoke({"modality_name": "x", **PERMISSIVE})
    _custom(
        dm,
        "import numpy as np, anndata\n"
        "raw = anndata.read_h5ad(WORKSPACE / 'data' / 'x.h5ad')\n"
        "adata.obs['raw_total'] = np.asarray(raw[adata.obs_names].X.sum(axis=1)).ravel()\n",
    )
    dm.save_modality(f"{STEP_A}_custom", f"{STEP_A}_custom.h5ad")
    notebook = dm.export_notebook("custom", "custom", filter_strategy="replay")
    return dm, pristine, notebook


def test_the_input_read_through_workspace_is_the_bound_input(tmp_path, monkeypatch):
    """The workspace's input file changed after the session (values doubled):
    the replay on the pristine --input reads only that input, so the custom
    result matches (the changed workspace file itself is not compared)."""
    dm, pristine, notebook = _input_read_session(tmp_path, monkeypatch)
    assert _replay(dm, notebook, dm.data_dir / "x.h5ad")["verdict"] == "MATCH"
    doubled = ad.read_h5ad(dm.data_dir / "x.h5ad")
    doubled.X = doubled.X * 2
    doubled.write_h5ad(dm.data_dir / "x.h5ad")
    out = _replay(dm, notebook, pristine)
    assert out["verdict"] == "NOT_COMPARED", out
    assert _diffs(out) == [], out
    compared = out["report"]["slots"][0]["compared"]
    assert compared > 0, out


def test_a_changed_copy_of_the_input_is_refused(tmp_path, monkeypatch):
    dm, pristine, notebook = _input_read_session(tmp_path, monkeypatch)
    changed = ad.read_h5ad(pristine)
    changed.X = changed.X + 1
    copy = tmp_path / "changed" / "x.h5ad"
    copy.parent.mkdir()
    changed.write_h5ad(copy)
    out = _replay(dm, notebook, copy)
    assert out["verdict"] == "REFUSED", out
