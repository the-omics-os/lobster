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
