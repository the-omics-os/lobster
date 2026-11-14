"""An instruction whose turn was cancelled or failed is not lost: the next
turn's record and status list it, and the notebook says what became of it."""

from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from lobster.cli_internal import mcp_record
from tests.unit.mcp.test_continue import (  # noqa: F401  (fixtures)
    COMPLETE,
    ERROR,
    Hold,
    WaitForCancel,
    call,
    go_on,
    started,
    wait_terminal,
)
from tests.unit.mcp.test_publish import H5AD, scan  # noqa: F401
from tests.unit.mcp.test_run import data, engine  # noqa: F401

SCHEMA = json.loads(
    (
        Path(__file__).resolve().parents[3]
        / "lobster/schemas/lobster.result.v1.schema.json"
    ).read_text()
)


def cancel_turn(server, opened, instruction, **target):
    step = WaitForCancel()
    opened[0]["client"].events = [step]
    run_id = go_on(server, instruction, **target)["run_id"]
    assert step.entered.wait(30)
    call(server, "lobster_cancel", {"run_id": run_id})
    done = wait_terminal(server, run_id)
    assert done["status"] == "cancelled"
    return done


def fail_turn(server, opened, instruction, **target):
    opened[0]["client"].events = [ERROR]
    run_id = go_on(server, instruction, **target)["run_id"]
    assert wait_terminal(server, run_id)["status"] == "failed"


def complete_turn(server, opened, instruction, out, **target):
    opened[0]["client"].events = [COMPLETE]
    done = wait_terminal(server, go_on(server, instruction, **target)["run_id"])
    assert done["status"] == "completed"
    record = json.loads((out / mcp_record.result_name(done["turn"])).read_text())
    Draft202012Validator(SCHEMA).validate(record)
    return done, record


def test_the_next_turn_lists_the_instructions_that_did_not_finish(started):
    server, opened, _script, out, first_id = started
    cancel_turn(server, opened, "recluster at resolution 2", run_id=first_id)
    fail_turn(server, opened, "annotate the T cells", run_id=first_id)
    done, record = complete_turn(
        server, opened, "now just count the cells", out, run_id=first_id
    )
    expected = [
        {"instruction": "recluster at resolution 2", "outcome": "cancelled"},
        {"instruction": "annotate the T cells", "outcome": "failed"},
    ]
    assert record["request"]["unfinished_instructions"] == expected  # oldest first
    assert done["unfinished_instructions"] == expected
    assert record["request"]["instruction"] == "now just count the cells"
    # They are reported once: the turn after this one has none to report.
    again, later = complete_turn(
        server, opened, "and again", out, run_id=done["run_id"]
    )
    assert "unfinished_instructions" not in later["request"]
    assert "unfinished_instructions" not in again


def test_a_turn_with_nothing_unfinished_says_nothing_of_it(started):
    server, opened, _script, out, first_id = started
    done, record = complete_turn(server, opened, "count them", out, run_id=first_id)
    assert "unfinished_instructions" not in done
    assert "unfinished_instructions" not in record["request"]


def test_an_unfinished_instruction_does_not_count_as_a_step(started):
    server, opened, _script, out, first_id = started
    fail_turn(server, opened, "annotate", run_id=first_id)
    _done, record = complete_turn(server, opened, "count", out, run_id=first_id)
    assert [s["tool"] for s in record["steps"]] == []


def test_the_list_is_bounded_and_says_how_many_there_were(started):
    server, opened, _script, out, first_id = started
    for n in range(mcp_record.MAX_UNFINISHED + 3):
        fail_turn(server, opened, f"try number {n}", run_id=first_id)
    _done, record = complete_turn(server, opened, "count", out, run_id=first_id)
    listed = record["request"]["unfinished_instructions"]
    assert len(listed) == mcp_record.MAX_UNFINISHED
    assert record["request"]["unfinished_instructions_total"] == len(listed) + 3
    assert listed[-1]["instruction"] == f"try number {mcp_record.MAX_UNFINISHED + 2}"
    assert listed == sorted(listed, key=lambda e: int(e["instruction"].split()[-1]))


def test_the_notebook_note_is_comment_only_and_not_replayable():
    for outcome in ("cancelled", "failed"):
        ir = mcp_record.outcome_ir(outcome)
        assert ir.render() == (
            f"# The turn for this instruction was {outcome} before it finished."
        )
        assert ir.replayable is False
        assert ast.parse(ir.render()).body == []
    with pytest.raises(ValueError):
        mcp_record.outcome_ir("completed")


def test_the_note_follows_the_instruction_cell_in_a_real_notebook():
    from lobster.core.runtime.data_manager import DataManagerV2

    with tempfile.TemporaryDirectory() as tmp:
        dm = DataManagerV2(workspace_path=Path(tmp) / "ws")
        assert mcp_record.record_instruction(dm, "recluster", "mcp:claude-code")
        assert mcp_record.record_outcome(dm, "cancelled")
        assert mcp_record.record_instruction(dm, "count", "mcp:claude-code")
        notebook = json.loads(Path(dm.export_notebook("nb")).read_text())
    # Each step has a markdown heading before its code cell.
    sources = [
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    ]
    first = next(i for i, s in enumerate(sources) if "Instruction: 'recluster'" in s)
    assert sources[first + 1] == (
        "# The turn for this instruction was cancelled before it finished."
    )
    assert "Instruction: 'count'" in sources[first + 2]


def test_the_schema_allows_the_new_request_fields_only_as_optional():
    request = SCHEMA["properties"]["request"]
    assert "unfinished_instructions" not in request.get("required", [])
    item = request["properties"]["unfinished_instructions"]["items"]
    assert set(item["required"]) == {"instruction", "outcome"}
    assert item["properties"]["outcome"]["enum"] == ["cancelled", "failed"]


# a cancel that took effect late --------------------------------------------


def late_cancel(server, opened, instruction, steps, **target):
    """A turn whose *steps* completed before the cancel took effect."""
    ran = Hold(*steps)
    ran.release.set()
    wait = WaitForCancel()
    opened[0]["client"].events = [ran, wait]
    run_id = go_on(server, instruction, **target)["run_id"]
    assert wait.entered.wait(30)
    call(server, "lobster_cancel", {"run_id": run_id})
    return wait_terminal(server, run_id)


def test_a_cancelled_turn_lists_the_steps_that_completed_before_it_stopped(started):
    server, opened, _script, out, first_id = started
    cancelled = late_cancel(
        server,
        opened,
        "cluster them",
        ["compute_neighbors_and_embed", "cluster_cells"],
        run_id=first_id,
    )
    assert cancelled["status"] == "cancelled"
    assert cancelled["steps_completed"] == [
        "compute_neighbors_and_embed",
        "cluster_cells",
    ]
    summary = cancelled["summary"]
    assert summary.startswith("Stopped before the analysis finished. 2 analysis steps")
    assert "completed before it stopped" in summary
    assert "stay in the session until the server stops" in summary
    assert "never saved to a file is not kept across a restart" in summary
    assert "remain in the session folder" not in summary
    assert "not carried out" not in summary


def test_a_cancelled_turn_without_steps_says_none_completed(started):
    server, opened, _script, out, first_id = started
    status = cancel_turn(server, opened, "cluster them", run_id=first_id)
    assert "steps_completed" not in status
    assert "No analysis step had completed" in status["summary"]


def test_the_steps_list_is_bounded(started):
    server, opened, _script, out, first_id = started
    many = [f"step_{n}" for n in range(mcp_record.MAX_STEPS_LISTED + 5)]
    cancelled = late_cancel(server, opened, "go", many, run_id=first_id)
    assert len(cancelled["steps_completed"]) == mcp_record.MAX_STEPS_LISTED
    assert cancelled["steps_completed_total"] == len(many)


def test_the_next_turn_lists_the_steps_of_the_unfinished_instruction(started):
    server, opened, _script, out, first_id = started
    late_cancel(server, opened, "cluster them", ["cluster_cells"], run_id=first_id)
    cancel_turn(server, opened, "and annotate", run_id=first_id)
    done, record = complete_turn(server, opened, "count", out, run_id=first_id)
    expected = [
        {
            "instruction": "cluster them",
            "outcome": "cancelled",
            "steps_completed": ["cluster_cells"],
        },
        {"instruction": "and annotate", "outcome": "cancelled"},
    ]
    assert record["request"]["unfinished_instructions"] == expected
    assert done["unfinished_instructions"] == expected


def test_a_failed_turn_lists_its_steps_too(started):
    server, opened, _script, out, first_id = started
    ran = Hold("cluster_cells")
    ran.release.set()
    opened[0]["client"].events = [ran, ERROR]
    run_id = go_on(server, "cluster them", run_id=first_id)["run_id"]
    failed = wait_terminal(server, run_id)
    assert failed["status"] == "failed" and failed["steps_completed"] == [
        "cluster_cells"
    ]


def test_the_notebook_note_says_the_turn_was_stopped_and_what_stayed():
    ir = mcp_record.outcome_ir("cancelled", ["a", "b"])
    assert ir.render() == (
        "# The turn for this instruction was cancelled before it finished. "
        "Steps that completed before it stopped stay in the session until the "
        "server stops, and data they made that was never saved to a file is not "
        "kept across a restart: a, b."
    )
    assert ast.parse(ir.render()).body == []
    hostile = mcp_record.outcome_ir("failed", ["x\nimport os", "{{ y }}"])
    assert "\n" not in hostile.render() and ast.parse(hostile.render()).body == []
    assert "{" not in hostile.render().split("restart:")[1]


def test_the_steps_survive_a_real_provenance_log():
    from lobster.core.runtime.data_manager import DataManagerV2

    with tempfile.TemporaryDirectory() as tmp:
        dm = DataManagerV2(workspace_path=Path(tmp) / "ws")
        assert mcp_record.record_instruction(dm, "cluster", "mcp:x")
        assert mcp_record.record_outcome(dm, "cancelled", ["cluster_cells", "umap"])
        assert mcp_record.record_instruction(dm, "count", "mcp:x")
        entries, total = mcp_record.unfinished_instructions(dm)
        notebook = json.loads(Path(dm.export_notebook("nb")).read_text())
    assert entries == [
        {
            "instruction": "cluster",
            "outcome": "cancelled",
            "steps_completed": ["cluster_cells", "umap"],
        }
    ]
    text = "".join("".join(c["source"]) for c in notebook["cells"])
    assert "not kept across a restart: cluster_cells, umap." in text


def test_the_schema_allows_steps_completed_only_as_optional():
    item = SCHEMA["properties"]["request"]["properties"]["unfinished_instructions"][
        "items"
    ]
    assert "steps_completed" not in item["required"]
    assert item["properties"]["steps_completed"]["items"] == {"type": "string"}


def test_a_summary_that_ends_in_a_question_mark_gets_no_full_stop_after_it():
    text = mcp_record.stopped_summary(
        "Stopped while waiting for an answer to: Which?", 0
    )
    assert text == (
        "Stopped while waiting for an answer to: Which? No analysis step had completed."
    )
    assert mcp_record.stopped_summary("Stopped.", 1).startswith(
        "Stopped. 1 analysis step "
    )


# the steps after a restart ---------------------------------------------------


def reread(tracker):
    """The provenance a new process reads from the folder the tracker wrote."""
    from types import SimpleNamespace

    from lobster.core.provenance.provenance import ProvenanceTracker

    return SimpleNamespace(
        provenance=ProvenanceTracker(session_dir=tracker.session_dir)
    )


@pytest.mark.parametrize(
    "steps",
    [["assess_data_quality"], ["assess_data_quality", "cluster_cells", "umap"]],
    ids=["one step", "several steps"],
)
def test_the_steps_of_a_stopped_turn_are_names_again_after_a_restart(steps):
    from lobster.core.runtime.data_manager import DataManagerV2

    with tempfile.TemporaryDirectory() as tmp:
        dm = DataManagerV2(workspace_path=Path(tmp) / "ws")
        dm.provenance.attach_session_dir(Path(tmp) / "session")
        assert mcp_record.record_instruction(dm, "cluster", "mcp:x")
        assert mcp_record.record_outcome(dm, "cancelled", steps)
        assert mcp_record.record_instruction(dm, "count", "mcp:x")
        entries, total = mcp_record.unfinished_instructions(reread(dm.provenance))
    assert total == 1
    assert entries[0]["steps_completed"] == steps


@pytest.mark.parametrize(
    "stored",
    [
        "['assess_data_quality' 'cluster_cells']",
        "['assess_data_quality', 'cluster_cells']",
        "array(['assess_data_quality', 'cluster_cells'], dtype=object)",
        'array(["assess_data_quality" "cluster_cells"], dtype="<U19")',
        "assess_data_quality,cluster_cells",
    ],
    ids=["an array as text", "a list as text", "an array repr", "quotes", "now"],
)
def test_steps_that_an_earlier_build_stored_as_text_are_still_read(stored):
    from types import SimpleNamespace

    activities = [
        {"type": "host_instruction", "parameters": {"instruction": "go"}},
        {
            "type": "host_instruction_outcome",
            "parameters": {"outcome": "failed", "steps_completed": stored},
        },
        {"type": "host_instruction", "parameters": {"instruction": "next"}},
    ]
    dm = SimpleNamespace(provenance=SimpleNamespace(activities=activities))
    (entry,), _total = mcp_record.unfinished_instructions(dm)
    assert entry["steps_completed"] == ["assess_data_quality", "cluster_cells"]
