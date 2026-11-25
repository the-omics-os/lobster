"""``lobster replay --verify`` on a notebook with a step that reads two or more
modalities: refused before it runs, naming the step.

A replay starts from one input and no cell binds the modalities a concatenation
or a cross-platform concordance reads, so their cells stopped every replay with a
NameError. The session here is made by the product: DataManagerV2 loads two
modalities, the step's own service or tool records it, and ``export_notebook``
writes the default notebook.
"""

import json
from pathlib import Path

import nbformat
import numpy as np
import pytest

from lobster.core.notebooks.exporter import NotebookExporter
from tests.unit.cli.test_replay_commands import (
    _adata,
    _clean_env,
    _save,
    _verify_real,
    needs_kernel,
)
from tests.unit.core.test_multi_input_not_replayed import _concordance

pytestmark = pytest.mark.unit

# the export record's list and the code cell's mark (exporter.KEY)
KEY = "not_replayed_steps"


def _concatenate(manager):
    from lobster.services.data_management.concatenation_service import (
        ConcatenationService,
    )

    merged, _, ir = ConcatenationService(manager).concatenate_from_modalities(
        ["s1", "s2"], "merged"
    )
    manager.modalities["merged"] = merged
    return "merged", ir


def _assess_concordance(manager):
    return "s1", _concordance()


STEPS = {
    "concatenate_samples": (_concatenate, 2),
    "assess_cross_platform_concordance": (_assess_concordance, 2),
}


@pytest.fixture(params=list(STEPS))
def session(request, tmp_path, monkeypatch):
    """``(tool, workspace, notebook, input)`` of a session with the step."""
    from lobster.core.runtime.data_manager import DataManagerV2

    _clean_env(monkeypatch, tmp_path)
    monkeypatch.setattr(NotebookExporter, "_snapshot_dependencies", lambda self: {})
    workspace = tmp_path / "workspace"
    source = _adata()
    source.X = np.abs(source.X)
    _save(source, tmp_path / "input.h5ad")
    manager = DataManagerV2(workspace_path=workspace, auto_scan=False)
    for name in ("s1", "s2"):
        manager.load_modality(
            name, tmp_path / "input.h5ad", adapter="transcriptomics_single_cell"
        )
    tool = request.param
    saved, ir = STEPS[tool][0](manager)
    manager.log_tool_usage(tool, {}, "a step that reads two modalities", ir=ir)
    manager.save_modality(saved, workspace / "data" / f"{saved}.h5ad")
    notebook = Path(manager.export_notebook("nb", "S"))
    return tool, workspace, notebook, tmp_path / "input.h5ad"


def _step_number(notebook: Path, tool: str) -> int:
    for cell in nbformat.read(notebook, as_version=4).cells:
        if cell.source.startswith("## Step ") and tool in cell.source.splitlines()[0]:
            return int(cell.source.split()[2].rstrip(":"))
    raise AssertionError(f"no step cell for {tool}")


@needs_kernel
def test_the_replay_refuses_it_before_it_runs_and_names_the_step(session):
    tool, workspace, notebook, source = session
    step = _step_number(notebook, tool)

    result = _verify_real(workspace, notebook, source)

    out = json.loads(result.stdout)
    assert result.exit_code == 4, out
    assert out["verdict"] == "REFUSED", out
    reason = out["report"]["refusal"]["reason"]
    assert f"step {step}: {tool}" in reason, reason
    assert "a replay starts from one input" in reason, reason
    assert not any("NameError" in note for note in out["notes"]), out["notes"]
    assert not list((workspace / "replays").glob("*/executed.ipynb")), "it ran"


@needs_kernel
@pytest.mark.parametrize("keep", ["record", "mark"])
def test_either_sign_alone_refuses_it(session, keep):
    tool, workspace, notebook, source = session
    nb = nbformat.read(notebook, as_version=4)
    for cell in nb.cells:
        if keep == "mark" and cell.cell_type == "markdown":
            cell.source = cell.source.replace(f'"{KEY}"', '"x"')
        if keep == "record":
            cell.metadata.get("lobster", {}).pop(KEY, None)
    nbformat.write(nb, notebook)

    result = _verify_real(workspace, notebook, source)

    assert result.exit_code == 4, result.stdout
    assert json.loads(result.stdout)["verdict"] == "REFUSED"
