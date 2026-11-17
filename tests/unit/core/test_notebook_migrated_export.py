"""An export over a session an earlier version wrote says so and runs none of its custom code."""

import json
import os

import nbformat
import pytest

from lobster.core.notebooks.exporter import NotebookExporter
from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.core.runtime.data_manager import DataManagerV2

pytestmark = pytest.mark.unit

OLD = {"scanpy": "1.12.1", "lobster": "1.2.2+release.f32a8be"}
CUSTOM = "open('/tmp/scbench_task/exports/out.csv', 'w').write('x')"


@pytest.fixture(autouse=True)
def _no_host_dependency_scan(monkeypatch):
    monkeypatch.setattr(NotebookExporter, "_snapshot_dependencies", lambda self: {})


def _step(operation, tool, code, imports=(), **kwargs):
    return AnalysisStep(
        operation=operation,
        tool_name=tool,
        description=operation,
        library="scanpy",
        code_template=code,
        imports=list(imports),
        parameters={},
        parameter_schema={},
        input_entities=["adata"],
        output_entities=["adata"],
        exportable=True,
        replayable=True,
        **kwargs,
    )


def _session(
    tmp_path, migrated: bool, custom: str = CUSTOM, metadata="written", imports=()
):
    ws = tmp_path / "ws"
    folder = ws / ".lobster" / "sessions" / "s1"
    dm = DataManagerV2(workspace_path=ws, session_dir=folder)
    dm.provenance.create_activity(
        "normalize",
        "a",
        description="normalize",
        ir=_step("n", "normalize", "x = 1", imports=list(imports)),
    )
    dm.provenance.create_activity(
        "execute_custom_code",
        "a",
        description="custom",
        ir=_step("custom_code_execution", "execute_custom_code", custom),
    )
    if migrated:
        meta = folder / "metadata.json"
        if metadata == "written":
            meta.write_text(json.dumps({"software_versions": OLD}))
        elif metadata == "unreadable":
            meta.write_text("{nope")
        else:
            meta.unlink()
        dm = DataManagerV2(workspace_path=ws, session_dir=folder)
    return ws, dm


def _export(dm, mode="successful"):
    path = dm.notebook_exporter.export("nb", filter_strategy=mode)
    return nbformat.read(path, as_version=4)


def _record(nb) -> dict:
    cell = next(
        c
        for c in nb.cells
        if c.cell_type == "markdown" and "data_integrity_manifest" in c.source
    )
    return json.loads(cell.source.split("```json\n", 1)[1].split("\n```", 1)[0])[
        "data_integrity_manifest"
    ]


@pytest.mark.parametrize("mode", ["successful", "replay"])
def test_migrated_custom_code_is_text_never_a_code_cell(tmp_path, mode):
    _, dm = _session(tmp_path, migrated=True)
    nb = _export(dm, mode)
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    assert not any("scbench_task" in s for s in code)
    assert any("x = 1" in s for s in code)  # the template step still runs
    shown = next(
        c.source
        for c in nb.cells
        if c.cell_type == "markdown" and "scbench_task" in c.source
    )
    assert (
        "Not executed: custom code from a Lobster 1.2.2 session "
        "(it may use paths of the machine it ran on)" in shown
    )
    assert f"```python\n{CUSTOM}" in shown


def test_the_export_record_and_header_say_migrated(tmp_path):
    _, dm = _session(tmp_path, migrated=True)
    nb = _export(dm)
    block = _record(nb)["migrated"]
    assert block["lobster"] == "1.2.2+release.f32a8be"
    assert block["software_versions"] == OLD
    assert block["input"] == "not recorded by the source version"
    assert _record(nb)["system"]["lobster_version"]  # the exporting build, unchanged
    assert (
        "Migrated from a Lobster 1.2.2 session: the cells were rebuilt from its "
        "record; the input file the session read was not recorded."
    ) in nb.cells[0].source


@pytest.mark.parametrize("metadata", ["unreadable", "absent"])
def test_an_unreadable_source_is_named_unknown(tmp_path, metadata):
    _, dm = _session(tmp_path, migrated=True, metadata=metadata)
    nb = _export(dm)
    block = _record(nb)["migrated"]
    assert block["source"] == "unknown (metadata unreadable)"
    assert block["lobster"] is None
    assert "Migrated from a Lobster session of unknown version" in nb.cells[0].source


def test_a_native_session_exports_without_any_migrated_mark(tmp_path):
    _, dm = _session(tmp_path, migrated=False)
    nb = _export(dm)
    assert "migrated" not in _record(nb)
    text = "\n".join(c.source for c in nb.cells)
    assert "Migrated from" not in text and "Not executed" not in text
    assert any(
        c.cell_type == "code" and "scbench_task" in c.source for c in nb.cells
    )  # native custom code is unchanged


def test_code_with_backticks_stays_inside_its_fence(tmp_path):
    _, dm = _session(
        tmp_path, migrated=True, custom="s = '''\n```\n'''  # scbench_task"
    )
    nb = _export(dm)
    shown = next(
        c.source
        for c in nb.cells
        if c.cell_type == "markdown" and "scbench_task" in c.source
    )
    assert "````python" in shown


def test_dangling_links_in_data_are_listed(tmp_path):
    ws, dm = _session(tmp_path, migrated=True)
    (ws / "data").mkdir(exist_ok=True)
    os.symlink("/nonexistent/cache/in.h5ad", ws / "data" / "in.h5ad")
    exporter = dm.notebook_exporter
    nb = _export(dm)
    assert exporter.dangling_links == ["data/in.h5ad"]
    assert _record(nb)["migrated"]["dangling_links"] == ["data/in.h5ad"]
    assert "restore or remove before replay" in nb.cells[0].source
    assert "data/in.h5ad" in nb.cells[0].source


def _imports_cell(nb) -> str:
    return next(c.source for c in nb.cells if c.source.startswith("# Required imports"))


def test_a_bare_name_in_a_migrated_record_is_left_out_of_the_imports_cell(tmp_path):
    _, dm = _session(tmp_path, migrated=True, imports=("import json", "pydeseq2"))
    source = _imports_cell(_export(dm))
    assert "# Not an import statement in the 1.2.x record, left out: pydeseq2" in source
    assert not any(line.strip() == "pydeseq2" for line in source.splitlines())
    assert "import json" in source
    exec(compile(source, "<imports>", "exec"), {})  # nosec B102


def test_a_native_imports_cell_is_not_changed(tmp_path):
    _, dm = _session(tmp_path, migrated=False, imports=("import json", "pydeseq2"))
    source = _imports_cell(_export(dm))
    assert "Not an import statement" not in source
    assert any(line.strip() == "pydeseq2" for line in source.splitlines())
