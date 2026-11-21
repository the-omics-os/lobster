"""Custom code is shown and kept exactly as it was written, never read as a template.

The custom-code step put the code into its Jinja template with an f-string, so
Lobster read the user's code as template source: ``{{ 7*7 }}`` came out as
``49``, CSS braces or ``{% endraw %}`` broke the render, and a Jinja expression
ran in Lobster's own process when a notebook showed the step. A new step keeps
the code as a template value. A step an earlier version wrote is read by its
shape: the workspace setup Lobster wrote, then the code, kept as recorded.
"""

import json

import nbformat
import pytest

from lobster.core.notebooks.exporter import NotebookExporter
from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.execution.custom_code_execution_service import (
    CustomCodeExecutionService,
)

pytestmark = pytest.mark.unit

CODES = [
    "adata.uns['t'] = '{{ 7*7 }}'",
    "adata.uns['t'] = '{% if x %}y{% endif %}'",
    "style = f'.c {{ color: red }}'",
    "adata.uns['t'] = '{% endraw %}'",
    "adata.uns['t'] = '{{ cycler.__init__.__globals__.os.getpid() }}'",
]

SETUP = """# Workspace setup (for notebook execution)
from pathlib import Path
import sys

WORKSPACE = Path({workspace})
sys.path.append(str(WORKSPACE))

# OUTPUT_DIR: Recommended directory for CSV/TSV/Excel exports
OUTPUT_DIR = WORKSPACE / 'exports'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
"""
# The setups earlier versions recorded in front of the code
RECORDED_SETUPS = [
    SETUP.format(workspace="'{{ workspace_path }}'"),
    SETUP.format(workspace="{{ workspace_path | string | repr }}"),
]
OLD = {"scanpy": "1.12.1", "lobster": "1.2.2+release.f32a8be"}


@pytest.fixture(autouse=True)
def _no_host_dependency_scan(monkeypatch):
    monkeypatch.setattr(NotebookExporter, "_snapshot_dependencies", lambda self: {})


@pytest.mark.parametrize("code", CODES)
def test_a_new_custom_step_renders_its_code_exactly(tmp_path, code):
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    ir = CustomCodeExecutionService(dm)._create_ir(
        code,
        "custom",
        None,
        False,
        None,
        True,
        {"success": True, "duration_seconds": 0.0, "warnings": []},
    )

    rendered = ir.render()

    assert rendered.endswith(f"\n# User code\n{code}"), rendered
    assert rendered.startswith(
        SETUP.format(workspace=repr(str(tmp_path / "ws")))
    ), rendered


def _migrated(tmp_path, template, code):
    ws = tmp_path / "ws"
    folder = ws / ".lobster" / "sessions" / "s1"
    dm = DataManagerV2(workspace_path=ws, session_dir=folder)
    dm.provenance.create_activity(
        "execute_custom_code",
        "a",
        description="custom",
        ir=AnalysisStep(
            operation="custom_code_execution",
            tool_name="execute_custom_code",
            description="custom",
            library="custom",
            code_template=template,
            imports=[],
            parameters={"workspace_path": "/old/ws", "code": code},
            parameter_schema={},
            exportable=True,
            replayable=True,
        ),
    )
    (folder / "metadata.json").write_text(json.dumps({"software_versions": OLD}))
    dm = DataManagerV2(workspace_path=ws, session_dir=folder)
    path = dm.notebook_exporter.export("nb", filter_strategy="successful")
    return [c.source for c in nbformat.read(path, as_version=4).cells]


@pytest.mark.parametrize("setup", RECORDED_SETUPS, ids=["quoted", "repr"])
@pytest.mark.parametrize("code", CODES)
def test_an_earlier_custom_step_shows_its_code_as_recorded(tmp_path, setup, code):
    cells = _migrated(tmp_path, f"{setup}\n# User code\n{code}", code)

    shown = next(s for s in cells if "Not executed: custom code" in s)
    assert "WORKSPACE = Path('/old/ws')\n" in shown
    assert f"\n# User code\n{code}\n```" in shown, shown


@pytest.mark.parametrize(
    "template",
    [
        "WORKSPACE = Path('{{ workspace_path }}')\n# User code\nprint(1)",
        RECORDED_SETUPS[1] + "\nprint(1)",
        RECORDED_SETUPS[1] + "\n# User code\nx = 1\n# User code\nprint(1)",
    ],
    ids=["other setup", "no code marker", "two code markers"],
)
def test_an_earlier_step_of_another_shape_is_not_shown(tmp_path, template):
    cells = _migrated(tmp_path, template, "print(1)")

    assert not any("print(1)" in s for s in cells)
    assert any(
        "Not shown: custom code from a Lobster 1.2.2 session whose record has "
        "a form Lobster did not write" in s
        for s in cells
    ), cells
