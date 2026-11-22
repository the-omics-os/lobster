"""A notebook that exports a DE table replays without touching the session's export.

The export cell wrote the table to the session's absolute ``exports/`` path, so a
replay rewrote the session's own file and was refused: "The notebook changed
files it should only read". The cell now writes the path relative to the
workspace, which a replay roots in its own folder; the table is listed among the
files the notebook wrote, not compared. Skipped without the transcriptomics
package or a Jupyter kernel.
"""

import json

import pytest

from tests.unit.cli.test_replay_bulk_de import RESULT, _bulk, _run_de, _sha, _verify
from tests.unit.cli.test_replay_commands import needs_kernel

de_module = pytest.importorskip("lobster.agents.transcriptomics.de_analysis_expert")

from lobster.core.notebooks.written_templates import template_hash  # noqa: E402
from lobster.core.runtime.data_manager import DataManagerV2  # noqa: E402
from lobster.testing.provenance_replay import factory_tools  # noqa: E402

EXPORTS = {
    "export_de_results": ({}, "de_results_ctrl_vs_treat.csv"),
    "export_de_results-subfolder": (
        {"output_path": "tables/de.csv"},
        "tables/de.csv",
    ),
    "extract_and_export_de_results": (
        {"shrink_lfc": False, "padj_threshold": 1.0},
        "de_publication_ctrl_vs_treat.csv",
    ),
}


@needs_kernel
@pytest.mark.parametrize("case", list(EXPORTS))
def test_a_de_export_replays_in_its_own_folder(tmp_path, monkeypatch, case):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "bulk.h5ad"
    _bulk().write_h5ad(source)
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.load_modality("bulk", str(source), adapter="transcriptomics_bulk")
    assert "Complete" in _run_de(dm, "run_differential_expression", "t_test")
    extra, name = EXPORTS[case]
    tools = factory_tools(de_module.de_analysis_expert, de_module.__name__, dm)
    reply = tools[case.split("-")[0]].invoke({"modality_name": RESULT, **extra})
    exported = dm.exports_dir / name
    assert exported.is_file(), reply
    before = _sha(exported)

    notebook = dm.export_notebook("de", "bulk DE", filter_strategy="successful")
    result, out = _verify(notebook, source, dm.workspace_path)

    assert result.exit_code == 0, (result.output, result.exception)
    assert out["verdict"] == "MATCH", out
    assert _sha(exported) == before
    wrote = [n for n in out["notes"] if n.startswith("Other files the notebook wrote")]
    assert wrote and f"exports/{name}" in wrote[0], out["notes"]
    code = [
        "".join(cell["source"])
        for cell in json.loads(notebook.read_text())["cells"]
        if cell["cell_type"] == "code"
    ]
    assert not any(str(dm.workspace_path.resolve()) in c for c in code)


def test_the_notebook_path_is_relative_to_the_workspace(tmp_path):
    workspace = tmp_path / "ws"
    (workspace / "exports" / "tables").mkdir(parents=True)
    linked = tmp_path / "linked"
    linked.symlink_to(workspace)
    table = str((workspace / "exports" / "tables" / "de.csv").resolve())

    path = de_module._notebook_export_path
    assert path(workspace, table) == "exports/tables/de.csv"
    assert path(linked, table) == "exports/tables/de.csv"
    elsewhere = str((tmp_path / "root" / "de.csv").resolve())
    assert path(workspace, elsewhere) == "exports/de.csv"


# The exact templates 1.3.0 wrote, and their hashes: a session it recorded
# keeps one of these.
_WRITE_1_3_0 = (
    "results_df.{% if output_format == 'xlsx' %}to_excel{% else %}to_csv{% endif %}"
    "({{ output_path | string | repr }}, index=True)"
)
TEMPLATES_1_3_0 = {
    "export_de_results": (
        "results_df = adata.uns[{{ de_results_key | string | repr }}]['results_df']\n"
        "results_df = results_df.sort_values({{ sort_by | string | repr }})\n"
        + _WRITE_1_3_0,
        "aff7d99fdfe066a84667009ea59ee8744bc50425e85c6262e4a903b2a1425bfd",
    ),
    "extract_and_export_de_results": (
        "# Extract and export DE results\n"
        "results_df = adata.uns[{{ de_results_key | string | repr }}]['results_df'].copy()\n"
        "{% if padj_threshold < 1.0 %}results_df = results_df[results_df['padj'] <= {{ padj_threshold }}]\n{% endif %}"
        "{% if lfc_threshold > 0 %}results_df = results_df[abs(results_df['log2FoldChange']) >= {{ lfc_threshold }}]\n{% endif %}"
        "results_df = results_df.sort_values('padj')\n" + _WRITE_1_3_0,
        "35440cc8c975149226867dad1bf76d30c615af8f555c781dc9510c745b10729a",
    ),
}


def _as_recorded_by_1_3_0(dm, tool):
    """The session's export step as 1.3.0 stored it: no folder line, and the
    session's file by its absolute path."""
    (activity,) = [
        a for a in dm.provenance.activities if (a["ir"] or {}).get("tool_name") == tool
    ]
    ir = activity["ir"]
    template, expected = TEMPLATES_1_3_0[tool]
    assert template_hash(template) == expected
    ir["code_template"] = template
    ir["imports"] = ["import pandas as pd"]
    relative = ir["parameters"]["output_path"]
    ir["parameters"]["output_path"] = str(dm.workspace_path.resolve() / relative)


@needs_kernel
@pytest.mark.parametrize("tool", list(TEMPLATES_1_3_0))
def test_a_de_export_step_recorded_by_1_3_0_is_shown_and_its_replay_refused(
    tmp_path, monkeypatch, tool
):
    """An upgrade keeps a 1.3.0 session's DE export step in its notebook. The
    cell writes the session's own file by its absolute path, so a replay runs
    nothing and the file is untouched."""
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "bulk.h5ad"
    _bulk().write_h5ad(source)
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.load_modality("bulk", str(source), adapter="transcriptomics_bulk")
    assert "Complete" in _run_de(dm, "run_differential_expression", "t_test")
    extra, name = EXPORTS[tool]
    tools = factory_tools(de_module.de_analysis_expert, de_module.__name__, dm)
    reply = tools[tool].invoke({"modality_name": RESULT, **extra})
    exported = dm.exports_dir / name
    assert exported.is_file(), reply
    _as_recorded_by_1_3_0(dm, tool)
    before = _sha(exported)

    notebook = dm.export_notebook("de", "bulk DE", filter_strategy="successful")
    text = notebook.read_text()
    assert "Not shown" not in text
    assert repr(str(exported.resolve()))[1:-1] in text
    result, out = _verify(notebook, source, dm.workspace_path)

    assert result.exit_code == 4 and out["verdict"] == "REFUSED", out
    assert f"exports/{name}" in out["report"]["refusal"]["reason"]
    assert _sha(exported) == before
