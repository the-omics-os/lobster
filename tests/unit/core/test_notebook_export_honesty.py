"""Regression tests: an exported notebook never promises more than it does.

One test group per finding:

- parameters the notebook does not read are not offered, and overrides the
  notebook does not define are refused (instead of silently doing nothing);
- a step whose generated code does not parse fails the export;
- the fallback save keeps the in-memory data and records what it could not write;
- failed steps are refused in replay mode and listed in the default mode;
- the export mode and "verification: not run" are stated, and the CLI passes
  the requested mode through.
"""

import ast
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import anndata
import nbformat
import numpy as np
import pandas as pd
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.notebooks.executor import NotebookExecutor
from lobster.core.notebooks.exporter import NotebookExporter
from lobster.core.provenance.analysis_ir import AnalysisStep, ParameterSpec
from lobster.core.provenance.provenance import ProvenanceTracker


def _adata():
    return anndata.AnnData(
        X=np.ones((4, 3)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
    )


@pytest.fixture
def data_manager():
    dm = Mock(spec=DataManagerV2)
    dm.modalities = {"m": _adata()}
    workspace = Path(tempfile.mkdtemp())
    dm.workspace_dir = workspace
    dm.workspace_path = workspace
    return dm


def _step(template, parameters=None, injectable=()):
    parameters = parameters or {}
    return AnalysisStep(
        operation="test.step",
        tool_name="test_step",
        description="test step",
        library="scanpy",
        code_template=template,
        imports=[],
        parameters=parameters,
        parameter_schema={
            name: ParameterSpec(
                param_type=type(value).__name__,
                papermill_injectable=name in injectable,
                default_value=value,
                required=False,
                description=name,
            )
            for name, value in parameters.items()
        },
        exportable=True,
    )


def _export(data_manager, *steps, mode="successful", failed=()):
    tracker = ProvenanceTracker()
    for index, step in enumerate(steps):
        tracker.create_activity(activity_type=f"step_{index}", agent="test", ir=step)
    for activity_type in failed:
        tracker.create_activity(activity_type=activity_type, agent="test")
        tracker.activities[-1]["error"] = "boom"
    exporter = NotebookExporter(tracker, data_manager)
    path = exporter.export("nb", filter_strategy=mode)
    with open(path) as handle:
        return path, nbformat.read(handle, as_version=4)


def _parameters_cell(notebook):
    return next(c for c in notebook.cells if "parameters" in c.metadata.get("tags", []))


# --- parameters that do nothing are not offered ----------------------------


def test_baked_in_setting_is_not_offered_as_a_parameter(data_manager):
    """target_sum is written into the step code, so overriding it would do nothing."""
    step = _step(
        "adata.uns['target_sum'] = {{ target_sum }}",
        {"target_sum": 10000.0},
        injectable={"target_sum"},
    )
    _, notebook = _export(data_manager, step)
    source = _parameters_cell(notebook).source

    assert "target_sum =" not in source
    assert "input_data = 'dataset.h5ad'" in source
    assert "random_seed" not in source  # nothing reads it


def test_setting_the_code_reads_is_offered_as_a_python_literal(data_manager):
    step = _step(
        "label = {{ label | repr }}\nadata.uns['label'] = label_override",
        {"label_override": 'it\'s "quoted"'},
        injectable={"label_override"},
    )
    step.parameters["label"] = "x"
    _, notebook = _export(data_manager, step)
    source = _parameters_cell(notebook).source

    assignment = next(
        line for line in source.splitlines() if line.startswith("label_override =")
    )
    value = ast.literal_eval(assignment.split("=", 1)[1].strip())
    assert value == 'it\'s "quoted"'


def _input_file():
    path = Path(tempfile.mkdtemp()) / "in.h5ad"
    _adata().write_h5ad(path)
    return path


@patch("lobster.core.notebooks.executor.papermill")
def test_runner_refuses_override_the_notebook_does_not_define(papermill, data_manager):
    step = _step(
        "adata.uns['target_sum'] = {{ target_sum }}",
        {"target_sum": 10000.0},
        injectable={"target_sum"},
    )
    path, _ = _export(data_manager, step)
    executor = NotebookExecutor(data_manager)

    result = executor.execute(path, _input_file(), parameters={"target_sum": 5000.0})

    assert result["status"] == "validation_failed"
    assert "target_sum" in result["errors"][0]
    papermill.execute_notebook.assert_not_called()


@patch("lobster.core.notebooks.executor.papermill")
def test_runner_passes_defined_override(papermill, data_manager):
    path, _ = _export(data_manager, _step("adata.uns['x'] = 1"))
    executor = NotebookExecutor(data_manager)

    result = executor.execute(path, _input_file(), parameters={"output_prefix": "b"})

    assert result["status"] == "success"
    assert (
        papermill.execute_notebook.call_args.kwargs["parameters"]["output_prefix"]
        == "b"
    )


# --- a cell that does not parse fails the export ----------------------------


def test_syntax_error_fails_the_export_and_writes_no_notebook(data_manager):
    step = _step("adata.uns['x'] = (1,")

    with pytest.raises(ValueError, match="not valid Python"):
        _export(data_manager, step)

    assert not (data_manager.workspace_path / "notebooks" / "nb.ipynb").exists()


# --- the fallback save keeps data and records the loss ----------------------


def test_fallback_save_keeps_live_data_and_records_unwritten_keys(
    data_manager, monkeypatch, capsys
):
    # Without the session writer (lobster not importable) the cell's fallback runs.
    monkeypatch.setitem(sys.modules, "lobster.core.backends.h5ad_backend", None)
    _, notebook = _export(data_manager, _step("adata.uns['x'] = 1"))
    save_cell = next(
        c.source for c in notebook.cells if c.source.startswith("# Save processed data")
    )
    adata = _adata()
    adata.uns["fine"] = 1
    adata.uns["bad"] = object()  # h5ad cannot write this
    prefix = str(Path(tempfile.mkdtemp()) / "out")
    namespace = {"adata": adata, "output_prefix": prefix}

    exec(compile(save_cell, "save_cell", "exec"), namespace)  # nosec B102

    written = anndata.read_h5ad(prefix + "_processed.h5ad")
    assert "bad" not in written.uns
    assert written.uns["fine"] == 1
    assert list(written.uns["lobster_unsaved_uns_keys"]) == ["bad"]
    assert "bad" in adata.uns  # in-memory result is untouched
    assert "Written without the session writer" in capsys.readouterr().out


def test_session_save_ir_uses_the_same_guarded_save(data_manager):
    """A recorded save_dataset IR keeps its file name but not a bare write.

    Seen in a manual CLI run: clustering stores uns['clustering_results'] with
    float keys, h5ad raises, and the whole notebook failed on its last cell.
    """
    from lobster.core.provenance.analysis_ir import create_data_saving_ir

    tracker = ProvenanceTracker()
    exporter = NotebookExporter(tracker, data_manager)
    tracker.activities = [
        {
            "type": "save_dataset",
            "parameters": {"name": "pbmc_clustered"},
            "ir": create_data_saving_ir(filename_suffix="pbmc_clustered").to_dict(),
        }
    ]
    save_cell = exporter._create_data_saving_cell().source
    adata = _adata()
    adata.uns["clustering_results"] = {0.4: {"n_clusters": 2}}
    prefix = str(Path(tempfile.mkdtemp()) / "out")
    namespace = {"adata": adata, "output_prefix": prefix}

    exec(compile(save_cell, "save_cell", "exec"), namespace)  # nosec B102

    written = anndata.read_h5ad(prefix + "_pbmc_clustered.h5ad")
    # the session's writer saves float keys; nothing is left out
    assert "lobster_unsaved_uns_keys" not in written.uns
    assert "clustering_results" in written.uns
    assert "write_session_h5ad(adata, output_path, compression='gzip')" in save_cell


# --- failed steps -------------------------------------------------------------


def test_replay_refuses_a_session_with_failed_steps(data_manager):
    with pytest.raises(ValueError, match="failed steps.*broken_step"):
        _export(
            data_manager,
            _step("adata.uns['x'] = 1"),
            mode="replay",
            failed=["broken_step"],
        )


def test_default_mode_lists_failed_steps(data_manager):
    _, notebook = _export(
        data_manager, _step("adata.uns['x'] = 1"), failed=["broken_step"]
    )
    header, footer = notebook.cells[0].source, notebook.cells[-1].source

    assert "`broken_step`" in header
    assert "Failed steps not in this notebook:** `broken_step`" in footer


# --- the header states what the notebook is -----------------------------------


@pytest.mark.parametrize("mode", ["successful", "replay"])
def test_header_states_mode_and_that_verification_did_not_run(data_manager, mode):
    _, notebook = _export(data_manager, _step("adata.uns['x'] = 1"), mode=mode)
    header = notebook.cells[0].source

    assert f"**Export mode:** `{mode}`" in header
    assert "**Verification:** not run." in header
    assert "**Steps included:** 1 of 1 recorded activities" in header


@pytest.mark.parametrize(
    ("command", "mode", "name", "description"),
    [
        ("/pipeline export nb my run", "successful", "nb", "my run"),
        ("/pipeline export --replay nb", "replay", "nb", None),
        ("/pipeline export nb --replay my run", "replay", "nb", "my run"),
    ],
)
def test_cli_passes_requested_mode(command, mode, name, description):
    from lobster.cli_internal.commands.heavy import slash_commands

    client = Mock()
    with patch.object(slash_commands, "pipeline_export") as export:
        slash_commands._execute_command(command, client, output=Mock())

    export.assert_called_once()
    args, kwargs = export.call_args
    assert args[2:] == (name, description)
    assert kwargs["mode"] == mode


def test_cli_export_calls_data_manager_with_mode(tmp_path):
    from lobster.cli_internal.commands.light.pipeline_commands import pipeline_export

    client = Mock()
    client.data_manager.workspace_path = tmp_path
    client.data_manager.export_notebook.return_value = Path("/x/nb.ipynb")
    pipeline_export(client, Mock(), "nb", "", mode="replay")

    client.data_manager.export_notebook.assert_called_once_with(
        "nb", "", filter_strategy="replay"
    )
