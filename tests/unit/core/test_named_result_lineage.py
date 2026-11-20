"""The modality a step made under its own name is known to a later custom cell.

A step whose IR names its result by a variable (``adata_clustered``, ...) records
which modality it made in two ways: a result-name parameter
(``result_modality_name``, ...), or the modalities the session stored from the
step's input while the tool ran (``stored_modalities``). A custom cell that reads
that modality gets a copy kept right after the step, from the variable the step's
code assigns, else from ``adata``.
"""

import anndata as ad
import numpy as np
import pytest

from lobster.core.notebooks.exporter import NotebookExporter
from lobster.core.runtime.data_manager import DataManagerV2

pytestmark = pytest.mark.unit


def _step(source, output, parameters=None, stored=None, template="adata = f(adata)"):
    activity = {
        "inputs": [{"name": source, "ir_name": "adata"}],
        "outputs": [{"name": output}],
        "parameters": parameters or {"modality_name": source},
        "ir": {"output_entities": [output], "code_template": template},
    }
    if stored is not None:
        activity["stored_modalities"] = stored
    return activity


def _custom_reading(name, steps, modality="x"):
    custom = {len(steps): (modality, [name], None)}
    steps = steps + [{"inputs": [], "outputs": [], "parameters": {}}]
    return NotebookExporter._lineage(steps, [], custom)


def test_a_result_named_by_a_parameter_is_kept_for_the_read():
    step = _step("x", "adata_with_markers", {"modality_name": "x",
                 "result_modality_name": "x_markers"})  # fmt: skip
    keep, _, _, plans, keep_vars = _custom_reading("x_markers", [step])
    assert plans[1]["x_markers"] == "kept"
    assert keep[0] == ["x_markers"]
    assert keep_vars.get(0, {}) == {}  # the template changed adata in place


def test_a_result_the_session_stored_from_the_input_is_kept_from_its_variable():
    step = _step(
        "x",
        "adata_clustered",
        stored=[{"name": "x_clustered", "parent": "x"}],
        template="adata_clustered, stats, _ = service.cluster(adata)",
    )
    keep, _, _, plans, keep_vars = _custom_reading("x_clustered", [step])
    assert plans[1]["x_clustered"] == "kept"
    assert keep_vars[0] == {"x_clustered": "adata_clustered"}


@pytest.mark.parametrize(
    "stored, name",
    [
        ([{"name": "y_other", "parent": "y"}], "y_other"),  # not from this input
        ([{"name": "x_a", "parent": "x"}, {"name": "x_b", "parent": "x"}], "x_a"),
        ([{"name": "x_b", "parent": None}], "x_b"),  # no parent recorded
    ],
    ids=["other-input", "two-made", "no-parent"],
)
def test_a_store_that_does_not_say_what_the_step_made_stays_unknown(stored, name):
    step = _step("x", "adata_scored", stored=stored)
    _, _, _, plans, _ = _custom_reading(name, [step])
    assert plans[1][name] == "missing"


def test_the_activity_records_what_the_tool_stored_and_not_its_save(tmp_path):
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    x = ad.AnnData(X=np.ones((3, 2), dtype=np.float32))
    dm.store_modality(name="x", adata=x)
    dm.log_tool_usage(tool_name="load", parameters={})
    dm.store_modality(name="x_done", adata=x.copy(), parent_name="x")
    dm.save_modality("x_done", "x_done.h5ad")
    activity = dm.log_tool_usage(tool_name="step", parameters={"modality_name": "x"})
    assert activity["stored_modalities"] == [{"name": "x_done", "parent": "x"}]
    later = dm.log_tool_usage(tool_name="other", parameters={})
    assert "stored_modalities" not in later
