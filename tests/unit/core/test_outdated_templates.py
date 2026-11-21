"""A stored step whose code an earlier template wrote is left out of the notebook.

Five tools put values into their cell as code: a value rendered raw
(``results_df.to_{{ output_format }}``) or put into the template with an
f-string (the concatenation's sample names). Their templates now keep every
value text, but a session recorded before keeps the old template, and its
export rendered the value as code again. Such a step is now named in a cell in
place of its code, in the header and in the export record, and replay refuses
the notebook.
"""

import json

import nbformat
import pytest

from lobster.core.notebooks import exporter as exporter_module
from lobster.core.notebooks.exporter import NotebookExporter
from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.data_management.concatenation_service import (
    ConcatenationService,
)

pytestmark = pytest.mark.unit

HOSTILE = """{{ "', __import__('os').getpid(), '" }}"""
OLD = {"scanpy": "1.12.1", "lobster": "1.2.2+release.f32a8be"}


@pytest.fixture(autouse=True)
def _no_host_dependency_scan(monkeypatch):
    monkeypatch.setattr(NotebookExporter, "_snapshot_dependencies", lambda self: {})


def _pre_fix_concat(sample_ids):
    """The concatenation step as 1.2.x recorded it: the names put into the
    template with str(), then rendered by Jinja."""
    template = f"""# Concatenate samples
# Join type: {{{{ join_type | string | repr }}}} (inner join = common genes only)
concatenated_adata = ad.concat(
    sample_adatas,
    axis=0,
    join={{{{ join_type }}}},
    merge='unique',
    label={{{{ batch_key | string | repr }}}},
    keys={sample_ids},
)
"""
    return AnalysisStep(
        operation="anndata.concat",
        tool_name="concatenate_samples",
        description="concat",
        library="anndata",
        code_template=template,
        imports=["import anndata as ad"],
        parameters={"join_type": "inner", "batch_key": "batch", "n_samples": 2},
        parameter_schema={},
        input_entities=["sample_adatas"],
        output_entities=["concatenated_adata"],
    )


def _pre_fix_export():
    """export_de_results as 1.2.x recorded it: the format rendered raw."""
    return AnalysisStep(
        operation="de_results.export",
        tool_name="export_de_results",
        description="export",
        library="pandas",
        code_template=(
            "results_df = adata.uns[{{ de_results_key | string | repr }}]['results_df']\n"
            "results_df.to_{{ output_format }}({{ output_path | string | repr }}, index=True)"
        ),
        imports=["import pandas as pd"],
        parameters={
            "de_results_key": "de_results",
            "output_format": "csv(__import__('os').getpid()) or results_df.to_csv",
            "output_path": "t.csv",
        },
        parameter_schema={},
    )


def _current_concat():
    service = ConcatenationService.__new__(ConcatenationService)
    return service._create_concatenation_ir(2, "inner", "batch", True, [HOSTILE, "b"])


def _plain():
    return AnalysisStep(
        operation="n",
        tool_name="n",
        description="n",
        library="custom",
        code_template="x = 1",
        imports=[],
        parameters={},
        parameter_schema={},
    )


def _export(tmp_path, ir, migrated=False):
    """The export of a stored session: one plain step, then *ir* (step 2)."""
    ws = tmp_path / "ws"
    folder = ws / ".lobster" / "sessions" / "s1"
    dm = DataManagerV2(workspace_path=ws, session_dir=folder)
    dm.provenance.create_activity("normalize", "a", description="n", ir=_plain())
    dm.provenance.create_activity(ir.tool_name, "a", description="step", ir=ir)
    if migrated:
        (folder / "metadata.json").write_text(json.dumps({"software_versions": OLD}))
    dm = DataManagerV2(workspace_path=ws, session_dir=folder)
    path = dm.notebook_exporter.export("nb", filter_strategy="successful")
    return dm.notebook_exporter, path, nbformat.read(path, as_version=4)


def _record(nb) -> dict:
    cell = next(c for c in nb.cells if "data_integrity_manifest" in c.source)
    record = json.loads(cell.source.split("```json\n", 1)[1].split("\n```", 1)[0])
    return record["data_integrity_manifest"]


STEPS = {
    "concat": (lambda: _pre_fix_concat(str([HOSTILE, "b"])), "concatenate_samples"),
    "export": (_pre_fix_export, "export_de_results"),
}
NOTE = (
    "Steps recorded with code this Lobster no longer writes are left out "
    "(steps: 2); replay cannot check this notebook."
)


@pytest.mark.parametrize("migrated", [False, True], ids=["stored", "migrated"])
@pytest.mark.parametrize("step", list(STEPS))
def test_a_stored_pre_fix_step_is_left_out_and_named(tmp_path, step, migrated):
    build, tool = STEPS[step]

    exporter, path, nb = _export(tmp_path, build(), migrated=migrated)

    assert "__import__" not in path.read_text(), "the stored value became code"
    (cell,) = [c for c in nb.cells if c.source.startswith("**Not shown")]
    assert cell.cell_type == "markdown"
    assert cell.source == (
        f"**Not shown: a {tool} step recorded with code this Lobster no longer "
        "writes (run the step again to include it)**\n"
    )
    assert cell.metadata["lobster"] == {
        exporter_module.OMITTED_CUSTOM_CODE_KEY: {
            "step": 2,
            "reason": "outdated_template",
        }
    }
    assert NOTE in nb.cells[0].source
    assert _record(nb)["outdated_steps"] == [2]
    assert exporter.outdated_steps == [2]


def test_a_step_the_current_builder_wrote_is_exported_as_code(tmp_path):
    exporter, path, nb = _export(tmp_path, _current_concat())

    assert any("ad.concat(" in c.source for c in nb.cells if c.cell_type == "code")
    assert not any("no longer writes" in c.source for c in nb.cells)
    assert "outdated_steps" not in _record(nb)
    assert exporter.outdated_steps == []


def test_a_second_export_lists_each_left_out_step_once(tmp_path):
    exporter, _, _ = _export(tmp_path, _pre_fix_export())

    path = exporter.export("again", filter_strategy="successful")

    assert exporter.outdated_steps == [2]
    assert NOTE in nbformat.read(path, as_version=4).cells[0].source


@pytest.mark.parametrize("template", [None, 7], ids=["none", "int"])
def test_a_listed_step_whose_template_is_not_text_is_left_out(tmp_path, template):
    ir = _pre_fix_export()
    ir.code_template = template

    exporter, _, _ = _export(tmp_path, ir)

    assert exporter.outdated_steps == [2]


def test_a_left_out_step_brings_none_of_its_imports(tmp_path):
    ir = _pre_fix_export()
    ir.imports = ["import os; os.getpid()"]

    _, path, _ = _export(tmp_path, ir)

    assert "getpid" not in path.read_text()
