"""A step that stores its result under a new name, from a variable of its own.

The exported cell binds ``adata`` to that variable, so the next cell and the
save see the result. The lineage knows the stored name (``stored_modalities``,
or a result-name parameter): a later step on an earlier modality starts from a
copy kept where it was made, and the session's save of the result is written
right after the step.
"""

import pytest

from lobster.core.notebooks.exporter import NotebookExporter

pytestmark = pytest.mark.unit


def _step(source, var, made=None, template=None):
    activity = {
        "inputs": [{"name": source, "ir_name": "adata", "resolved": True}],
        "outputs": [{"name": var, "resolved": True}],
        "parameters": {"modality_name": source},
        "ir": {
            "output_entities": [var],
            "code_template": template or f"{var}, stats, _ = service.run(adata)",
        },
    }
    if made:
        activity["stored_modalities"] = [{"name": made, "parent": source}]
    return activity


# A proteomics session's shape: QC on pr, a filter on pr (a side branch), then
# normalize on the QC result and impute.
STEPS = [
    _step("pr", "adata_qc", "pr_qa"),
    _step("pr", "adata_filtered", "pr_f"),
    _step("pr_qa", "adata_norm", "pr_qa_n"),
    _step("pr_qa_n", "adata_imputed", "pr_qa_n_i"),
]


def test_a_step_on_an_earlier_result_starts_from_its_kept_copy():
    keep, rebind, _, _, _ = NotebookExporter._lineage(STEPS, [])
    # the filter runs on the input as loaded, normalize on the QC result
    assert rebind == {1: "pr", 2: "pr_qa"}
    assert keep == {-1: ["pr"], 0: ["pr_qa"]}


def test_the_sessions_saves_of_stored_results_are_written_where_made():
    _, _, save_at, _, _ = NotebookExporter._lineage(
        STEPS, ["pr_f", "pr_qa_n", "pr_qa_n_i"]
    )
    assert save_at == {1: ["pr_f"], 2: ["pr_qa_n"], 3: ["pr_qa_n_i"]}


def test_the_result_variable_is_bound_to_adata_after_its_step():
    assert NotebookExporter._result_binding(STEPS[0]) == ("pr_qa", "adata_qc")


@pytest.mark.parametrize(
    "activity",
    [
        _step("pr", "adata_stats"),  # stored nothing: the session kept pr as it was
        _step("pr", "adata_x", "pr_x", template="service.run(adata)"),  # in place
    ],
    ids=["nothing-stored", "no-variable"],
)
def test_no_binding_without_a_stored_result_in_a_variable(activity):
    assert NotebookExporter._result_binding(activity) is None


def _unrecorded(source, var, made, parent="same"):
    """A step whose IR records no input: the tool named it by modality_name."""
    activity = _step(source, var)
    activity["inputs"] = []
    activity["stored_modalities"] = [
        {"name": made, "parent": source if parent == "same" else parent}
    ]
    return activity


def test_a_result_of_a_step_with_no_recorded_input_starts_from_its_parent():
    """The reported case: QC on X records no input; a later step on X must start from
    the loaded X, not from the QC result adata now holds."""
    steps = [
        _unrecorded("X", "adata_qc", "X_qc"),
        {"inputs": [{"name": "X", "ir_name": "adata", "resolved": True}],
         "outputs": [{"name": "X_norm", "ir_name": "adata", "resolved": True}],
         "parameters": {}},
    ]  # fmt: skip
    keep, rebind, _, _, _ = NotebookExporter._lineage(steps, [])
    assert rebind == {1: "X"}
    assert keep == {-1: ["X"]}


def test_a_custom_step_on_the_parent_of_such_a_result_reads_it():
    steps = [
        _unrecorded("X", "adata_qc", "X_qc"),
        {"inputs": [], "outputs": [], "parameters": {}},
    ]
    _, _, _, plans, _ = NotebookExporter._lineage(steps, [], {1: ("X", ["X"], None)})
    assert plans[1]["X"] != "missing", plans


def test_no_binding_when_the_stored_result_names_no_parent():
    assert (
        NotebookExporter._result_binding(
            _unrecorded("X", "adata_qc", "X_qc", parent=None)
        )
        is None
    )


@pytest.mark.parametrize("kind", ["execute_custom_code", "custom_code_execution"])
def test_a_custom_code_step_is_never_bound(kind):
    """Its cell is the custom cell, or markdown when it is not run: the custom
    path keeps its names, never a binding."""
    activity = _step("pr", "adata_x", "pr_x")
    activity["type"] = kind
    assert NotebookExporter._result_binding(activity) is None
