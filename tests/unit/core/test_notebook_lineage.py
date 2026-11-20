"""The exported notebook follows each step's input, not only the step order.

Every step cell works on one ``adata``. A session can branch: two steps can
start from the same modality (assess the raw data, then filter the raw data),
or a step can follow the last save (assess the filtered data again after it was
saved). Before, the notebook ran the steps in a line on one object, so the
filtered file carried the assessment's columns, or the saved file was written
after a later step had changed it. Either way the replay did not reproduce the
file the session saved.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock

import anndata
import numpy as np
import pandas as pd
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.notebook_exporter import NotebookExporter
from lobster.core.notebooks.exporter import UNRECORDED_INPUT_NOTE
from lobster.core.provenance import ProvenanceTracker
from lobster.core.provenance.analysis_ir import AnalysisStep, create_data_saving_ir

# Each step changes the object in place, as the QC templates do.
ASSESS = "adata.obs['assessed'] = {mark}"
FILTER = "adata = adata[:5].copy()\nadata.obs['filtered'] = 1"
CLUSTER = "adata.obs['cluster'] = 1"
MARKERS = "adata.uns['markers'] = 'g0'"


def step(tracker, tool, source, result, code):
    tracker.create_activity(
        activity_type=tool,
        agent="transcriptomics_expert",
        description=tool,
        inputs=[{"name": source, "ir_name": "adata"}],
        outputs=[{"name": result, "ir_name": "adata"}],
        ir=AnalysisStep(
            operation=tool,
            tool_name=tool,
            description=tool,
            library="anndata",
            code_template=code,
            imports=[],
            parameters={},
            parameter_schema={},
            input_entities=["adata"],
            output_entities=["adata"],
            exportable=True,
        ),
    )


def save(tracker, name):
    tracker.create_activity(
        activity_type="save_dataset",
        agent="data_manager",
        description=f"Saved modality '{name}'",
        parameters={"name": name, "path": f"/ws/data/{name}.h5ad"},
        inputs=[{"name": name, "ir_name": "adata"}],
        outputs=[{"name": "file"}],
        ir=create_data_saving_ir(filename_suffix=name),
    )


def raw_data():
    return anndata.AnnData(
        X=np.ones((10, 3), dtype=np.float32),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(10)]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )


def export(tracker, tmp: Path) -> str:
    dm = Mock(spec=DataManagerV2)
    dm.modalities = {}
    dm.workspace_path = dm.workspace_dir = Path(tempfile.mkdtemp(dir=tmp))
    return Path(NotebookExporter(tracker, dm).export("lineage")).read_text()


def replay_all(tracker, tmp: Path) -> dict:
    """Export, run every cell after the loading cell on ``raw_data()``, and read
    back each file the save cells write, by modality name."""
    import nbformat

    notebook = nbformat.reads(export(tracker, tmp), as_version=4)
    code = [c.source for c in notebook.cells if c.cell_type == "code"]
    start = next(i for i, s in enumerate(code) if "ad.read_h5ad(input_data)" in s)
    namespace = {"adata": raw_data(), "output_prefix": str(tmp / "replayed")}
    for source in code[start + 1 :]:
        exec(compile(source, "<cell>", "exec"), namespace)  # noqa: S102
    return {
        path.name[len("replayed_") : -len(".h5ad")]: anndata.read_h5ad(path)
        for path in tmp.glob("replayed_*.h5ad")
    }


def replay(tracker, tmp: Path) -> anndata.AnnData:
    (written,) = replay_all(tracker, tmp).values()
    return written


def test_a_step_that_starts_from_an_earlier_modality_gets_that_modality(tmp_path):
    tracker = ProvenanceTracker()
    step(tracker, "assess_data_quality", "raw", "raw_assessed", ASSESS.format(mark=1))
    save(tracker, "raw_filtered")
    step(tracker, "filter_and_normalize", "raw", "raw_filtered", FILTER)

    written = replay(tracker, tmp_path)

    assert written.n_obs == 5 and "filtered" in written.obs
    assert "assessed" not in written.obs  # the session filtered the raw data


def test_a_branch_after_a_step_that_reads_no_data_is_still_followed(tmp_path):
    # A run starts with the host's question: a note cell with no data.
    tracker = ProvenanceTracker()
    tracker.create_activity(
        activity_type="host_question",
        agent="mcp",
        description="question",
        ir=AnalysisStep(
            operation="host_question",
            tool_name="host_question",
            description="question",
            library="lobster",
            code_template="# Question: 'Run QC.'",
            imports=[],
            parameters={},
            parameter_schema={},
            input_entities=[],
            output_entities=[],
            exportable=True,
        ),
    )
    step(tracker, "assess_data_quality", "raw", "raw_assessed", ASSESS.format(mark=1))
    step(tracker, "filter_and_normalize", "raw", "raw_filtered", FILTER)
    save(tracker, "raw_filtered")

    written = replay(tracker, tmp_path)

    assert written.n_obs == 5 and "assessed" not in written.obs


def test_the_saved_file_is_the_object_at_the_save_not_after_later_steps(tmp_path):
    tracker = ProvenanceTracker()
    step(tracker, "assess_data_quality", "raw", "a", ASSESS.format(mark=1))
    step(tracker, "filter_and_normalize", "a", "a_f", FILTER)
    save(tracker, "a_f")
    step(tracker, "assess_data_quality", "a_f", "a_f_a", ASSESS.format(mark=2))

    written = replay(tracker, tmp_path)

    assert written.n_obs == 5
    assert set(written.obs["assessed"]) == {1}  # not the later re-assessment


def test_a_straight_line_keeps_one_object_and_copies_nothing(tmp_path):
    tracker = ProvenanceTracker()
    step(tracker, "assess_data_quality", "raw", "a", ASSESS.format(mark=1))
    step(tracker, "filter_and_normalize", "a", "a_f", FILTER)
    save(tracker, "a_f")

    written = replay(tracker, tmp_path)

    assert written.n_obs == 5 and set(written.obs["assessed"]) == {1}
    text = export(tracker, tmp_path)
    assert "_kept_modalities" not in text and UNRECORDED_INPUT_NOTE not in text


@pytest.mark.parametrize("recorded", [False, True])
def test_steps_without_recorded_inputs_run_in_a_line_as_before(tmp_path, recorded):
    tracker = ProvenanceTracker()
    for tool, code in (("assess", ASSESS.format(mark=1)), ("filter", FILTER)):
        tracker.create_activity(
            activity_type=tool,
            agent="x",
            description=tool,
            inputs=[{"name": "raw"}] if recorded else None,  # no ir_name: not adata
            ir=AnalysisStep(
                operation=tool,
                tool_name=tool,
                description=tool,
                library="anndata",
                code_template=code,
                imports=[],
                parameters={},
                parameter_schema={},
                input_entities=["adata"],
                output_entities=["adata"],
                exportable=True,
            ),
        )
    save(tracker, "whatever")

    written = replay(tracker, tmp_path)

    assert written.n_obs == 5 and "assessed" in written.obs


def test_each_saved_modality_is_written_as_it_was_made(tmp_path):
    # load, assess, filter, save, cluster, save, markers, save: the session
    # saved three files, and the notebook writes all three.
    tracker = ProvenanceTracker()
    step(tracker, "assess_data_quality", "raw", "a", ASSESS.format(mark=1))
    save(tracker, "a_f")  # recorded just before the step whose result it saves
    step(tracker, "filter_and_normalize", "a", "a_f", FILTER)
    step(tracker, "cluster_cells", "a_f", "a_f_c", CLUSTER)
    save(tracker, "a_f_c")
    step(tracker, "find_marker_genes", "a_f_c", "a_f_c_m", MARKERS)
    save(tracker, "a_f_c_m")

    written = replay_all(tracker, tmp_path)

    assert sorted(written) == ["a_f", "a_f_c", "a_f_c_m"]
    assert "cluster" not in written["a_f"].obs
    assert "markers" not in written["a_f_c"].uns and "cluster" in written["a_f_c"].obs
    assert written["a_f_c_m"].uns["markers"] == "g0"
    assert "_kept_modalities" not in export(tracker, tmp_path)  # no branch


def test_a_save_no_step_made_is_not_written_so_replay_cannot_match_it(tmp_path):
    # Replay reports a saved file the notebook does not write as not produced
    # (not compared): it never counts as a match.
    tracker = ProvenanceTracker()
    step(tracker, "assess_data_quality", "raw", "a", ASSESS.format(mark=1))
    save(tracker, "made_elsewhere")
    step(tracker, "filter_and_normalize", "a", "a_f", FILTER)
    save(tracker, "a_f")

    assert sorted(replay_all(tracker, tmp_path)) == ["a_f"]


def test_a_step_that_changed_its_input_in_place_makes_the_replay_differ(tmp_path):
    # The record says the step read x and made y, but the tool also changed x
    # in place. The session's z, made from the changed x, has 'changed'; the
    # notebook's z starts from the copy of x kept before, so it does not. The
    # file differs from the saved one, and replay reports a mismatch: wrong
    # lineage never agrees by accident.
    tracker = ProvenanceTracker()
    step(tracker, "assess_data_quality", "raw", "x", ASSESS.format(mark=1))
    step(tracker, "mutating_tool", "x", "y", "adata.obs['changed'] = 1")
    step(tracker, "filter_and_normalize", "x", "z", FILTER)
    save(tracker, "z")

    written = replay(tracker, tmp_path)

    assert "changed" not in written.obs  # the session's z has it


def test_a_step_without_a_recorded_input_says_so_when_the_notebook_branches(
    tmp_path,
):
    tracker = ProvenanceTracker()
    step(tracker, "assess_data_quality", "raw", "raw_assessed", ASSESS.format(mark=1))
    step(tracker, "filter_and_normalize", "raw", "raw_filtered", FILTER)
    tracker.create_activity(
        activity_type="plot",
        agent="visualization_expert",
        description="plot",
        ir=AnalysisStep(
            operation="plot",
            tool_name="plot",
            description="plot",
            library="anndata",
            code_template="_ = adata.n_obs",
            imports=[],
            parameters={},
            parameter_schema={},
            input_entities=["adata"],
            output_entities=[],
            exportable=True,
        ),
    )
    save(tracker, "raw_filtered")

    text = export(tracker, tmp_path)

    assert text.count(UNRECORDED_INPUT_NOTE) == 1


DESEQ2_LIKE = "adata.uns['de'] = 'deseq2'"
RANK_LIKE = "adata = adata.copy()\nadata.uns['rank'] = 'wilcoxon'"


def logged_step(tracker, tool, parameters, code):
    """A step as a tool logs it: its inputs and outputs resolved from the IR."""
    tracker.create_activity(
        activity_type=tool,
        agent="de_analysis_expert",
        description=tool,
        parameters=parameters,
        ir=AnalysisStep(
            operation=tool,
            tool_name=tool,
            description=tool,
            library="lobster",
            code_template=code,
            imports=[],
            parameters={},
            parameter_schema={},
            input_entities=["adata"],
            output_entities=["adata"],
            exportable=True,
        ),
    )


def test_an_output_list_names_every_modality_the_step_wrote():
    tracker = ProvenanceTracker()
    logged_step(
        tracker,
        "de",
        {"modality_name": "raw", "result_modality_names": ["raw", "raw_de"]},
        DESEQ2_LIKE,
    )
    assert tracker.activities[-1]["outputs"] == [
        {"name": "raw", "resolved": True, "ir_name": "adata"},
        {"name": "raw_de", "resolved": True, "ir_name": "adata"},
    ]


def test_a_later_step_on_a_modality_changed_in_place_starts_from_that_change(
    tmp_path,
):
    """deseq2 writes its results into its input and saves a copy; a later rank
    test on the input sees those results, as the session's input had them."""
    tracker = ProvenanceTracker()
    logged_step(
        tracker,
        "deseq2",
        {"modality_name": "raw", "result_modality_names": ["raw", "raw_de"]},
        DESEQ2_LIKE,
    )
    save(tracker, "raw_de")
    logged_step(
        tracker,
        "wilcoxon",
        {"modality_name": "raw", "result_modality_name": "raw_rank"},
        RANK_LIKE,
    )
    save(tracker, "raw_rank")

    written = replay_all(tracker, tmp_path)

    assert written["raw_de"].uns["de"] == "deseq2"
    assert written["raw_rank"].uns["de"] == "deseq2"
    assert written["raw_rank"].uns["rank"] == "wilcoxon"
    assert "rank" not in written["raw_de"].uns


def test_a_step_after_a_result_starts_from_its_input_changed_in_place(tmp_path):
    """wilcoxon makes a result; deseq2 on the input then starts from the input."""
    tracker = ProvenanceTracker()
    logged_step(
        tracker,
        "wilcoxon",
        {"modality_name": "raw", "result_modality_name": "raw_rank"},
        RANK_LIKE,
    )
    save(tracker, "raw_rank")
    logged_step(
        tracker,
        "deseq2",
        {"modality_name": "raw", "result_modality_names": ["raw", "raw_de"]},
        DESEQ2_LIKE,
    )
    save(tracker, "raw_de")

    written = replay_all(tracker, tmp_path)

    assert "rank" not in written["raw_de"].uns
    assert written["raw_de"].uns["de"] == "deseq2"
    assert "de" not in written["raw_rank"].uns
