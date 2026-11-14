"""A status fits the limit it states, in the worst case the rest of the code allows."""

from __future__ import annotations

import json

from lobster.cli_internal import mcp_decision, mcp_record, mcp_stdio
from tests.unit.mcp.test_run import _completed

LIMIT = mcp_stdio.STATUS_MAX_CHARS


def unfinished(steps: int, name_chars: int = 80):
    return [
        {
            "instruction": "i" * mcp_record.INSTRUCTION_MAX_CHARS,
            "outcome": "cancelled",
            "steps_completed": [
                f"{n:02d}".ljust(name_chars, "s") for n in range(steps)
            ],
        }
        for _ in range(mcp_record.MAX_UNFINISHED)
    ]


def worst_case(steps: int = mcp_record.MAX_STEPS_LISTED, warnings: int = 11):
    """Every list at the size its own limit allows: ten unfinished instructions
    of 500 characters, each with 20 step names of 80, 50 files, 11 warnings of
    1,000 characters, a summary at its limit."""
    return {
        **_completed("s" * mcp_stdio.SUMMARY_MAX_CHARS),
        "metrics": {"cells_before": 3},
        "warnings": [
            {"code": "no_qc_plots", "message": "w" * mcp_stdio.FIELD_MAX_CHARS}
            for _ in range(warnings)
        ],
        "files": [
            {"path": f"/out/{n}/" + "f" * 200, "category": "plot", "size_bytes": 1}
            for n in range(mcp_stdio.MAX_FILES)
        ],
        "unfinished_instructions": unfinished(steps),
    }


def size(status) -> int:
    return len(json.dumps(status, default=str))


def test_the_worst_case_is_over_the_limit_before_it_is_bounded():
    assert size(worst_case()) >= LIMIT  # else the test below shows nothing


def test_the_worst_case_fits_and_keeps_what_must_stay():
    status = mcp_stdio._bounded(worst_case())
    assert size(status) < LIMIT
    assert status["metrics"] == {"cells_before": 3}
    assert len(status["warnings"]) == 11 and len(status["files"]) == 50
    assert status["result_path"]
    # The instructions stay (they are what the host reads); the step names go,
    # and say so; result.json has them.
    assert len(status["unfinished_instructions"]) == 10
    assert all("steps_completed" not in e for e in status["unfinished_instructions"])
    assert status["unfinished_steps_omitted"] is True


def test_step_names_stay_while_the_status_fits():
    status = mcp_stdio._bounded(worst_case(steps=2, warnings=1))
    assert size(status) < LIMIT
    assert status["unfinished_instructions"][0]["steps_completed"]
    assert "unfinished_steps_omitted" not in status


def test_a_status_without_unfinished_instructions_is_left_alone():
    status = mcp_stdio._bounded(_completed("done"))
    assert "unfinished_steps_omitted" not in status


def decisions(count: int = mcp_decision.MAX_RECORDED):
    """Answers as the record keeps them: question cut to 500, an answer of
    4,000 and a comment cut to 1,000."""
    return [
        mcp_decision.recorded_entry(
            {"question": "q" * 2000},
            "a" * mcp_stdio.ANSWER_MAX_CHARS,
            "human",
            "mcp:claude-code",
            comment="c" * mcp_stdio.ANSWER_MAX_CHARS,
        )
        for _ in range(count)
    ]


def test_the_worst_case_with_every_answer_fits_and_says_what_went():
    status = {
        **worst_case(),
        "decisions": decisions(),
        "decisions_total": 60,
        "decisions_note": "n",
    }
    bounded = mcp_stdio._bounded(status)
    assert size(bounded) < LIMIT
    assert "decisions" not in bounded and bounded["decisions_omitted"] is True
    assert bounded["decisions_total"] == 60
    assert len(bounded["warnings"]) == 11 and len(bounded["files"]) == 50


def test_answers_stay_while_the_status_fits():
    bounded = mcp_stdio._bounded({**_completed("done"), "decisions": decisions(3)})
    assert len(bounded["decisions"]) == 3 and "decisions_omitted" not in bounded
    assert "decisions_total" not in bounded
    alone = mcp_stdio._bounded({**worst_case(), "decisions": decisions()})
    assert alone["decisions_total"] == mcp_decision.MAX_RECORDED
