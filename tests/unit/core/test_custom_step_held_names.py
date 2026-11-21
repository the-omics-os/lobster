"""A custom step after a step that wrote several modalities reads each from adata.

A step may write its input in place and a saved copy (deseq2: ``X`` and
``X_deseq2``); ``adata`` is each of them after it. A custom step that follows is
served every name ``adata`` holds, never a kept copy of one, and starts from
``adata`` whichever of them it names as its modality.
"""

import pytest

from lobster.core.notebooks.exporter import NotebookExporter

pytestmark = pytest.mark.unit


def _ent(*names):
    return [{"name": n, "ir_name": "adata", "resolved": True} for n in names]


def _step(source, *results):
    return {
        "inputs": _ent(source) if source else [],
        "outputs": _ent(*results),
        "parameters": {},
    }


STEPS = [_step("X", "X"), _step("X", "X", "X_deseq2"), _step(None)]


def test_every_name_adata_holds_is_served_to_a_custom_step():
    plans = NotebookExporter._lineage(STEPS, [], {2: ("X", [], None)})[3]
    assert plans[2] == {"X": "adata", "X_deseq2": "adata"}


def test_a_custom_step_reads_the_saved_copy_from_adata_not_a_kept_copy():
    keep, rebind, _, plans, _ = NotebookExporter._lineage(
        STEPS, [], {2: ("X", ["X_deseq2"], None)}
    )
    assert plans[2]["X_deseq2"] == "adata"
    assert keep == {} and rebind == {}


def test_a_custom_step_on_the_saved_copy_starts_from_adata():
    keep, rebind, _, plans, _ = NotebookExporter._lineage(
        STEPS, [], {2: ("X_deseq2", [], None)}
    )
    assert rebind == {} and keep == {}
    assert plans[2] == {"X": "adata", "X_deseq2": "adata"}
    assert "X_deseq2" not in plans[-1]  # computed by this notebook
