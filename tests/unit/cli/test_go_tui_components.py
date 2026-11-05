"""Tests for the Go TUI adapter: decision fields <-> Go components.

A fake render plays the Go TUI: it records what would be drawn and replies
with each component's own reply shape.
"""

import json
import os
from pathlib import Path

import pytest

from lobster.cli_internal.go_tui_components import (
    build_answer,
    cannot_show,
    collect,
    dashboard_data,
    fallback_prompt,
    unsupported,
)
from lobster.services.interaction.component_schemas import (
    AskUserContext,
    DecisionAnswer,
    DecisionRequest,
    check_answer,
)


def _request(*fields, view="default", **envelope):
    context = AskUserContext.model_validate({"view": view, "fields": list(fields)})
    request = DecisionRequest(
        question="Review", view=context.view, fields=context.fields, **envelope
    )
    return request.to_wire()


class FakeTUI:
    """Replies in order; each reply is (action, component reply)."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.drawn = []

    def __call__(self, component, data):
        self.drawn.append((component, data))
        return self.replies.pop(0)


def _ask(request, *replies):
    """Collect, then check the answer exactly as ask_user would."""
    tui = FakeTUI(*replies)
    action, answer = collect(request, tui)
    envelope = build_answer(request, action, answer)
    result = check_answer(
        DecisionRequest.model_validate(request), DecisionAnswer.model_validate(envelope)
    )
    return tui.drawn, result


SOURCE = {
    "key": "source",
    "type": "choice",
    "details": "Recommended: SRC-A",
    "options": [
        {"id": "SRC-A", "label": "Author matrix", "details": "Raw counts · Build-A"},
        "SRC-B",
        {"id": "SRC-C", "unavailable": "Normalized; build unknown"},
    ],
}


# --- choice ---


def test_choice_draws_options_with_ids_and_unavailable_reason():
    drawn, result = _ask(_request(SOURCE), ("submit", {"selected": [{"id": "SRC-B"}]}))
    ((component, data),) = drawn
    assert component == "choice"
    assert data["question"] == "Review"
    assert data["details"] == "Recommended: SRC-A"
    assert (data["min"], data["max"]) == (1, 1)
    assert data["options"][2] == {
        "id": "SRC-C",
        "label": "SRC-C",
        "unavailable": "Normalized; build unknown",
    }
    assert result["answer"] == {"source": {"id": "SRC-B"}}


def test_multi_pick_choice_answers_a_list():
    field = {
        "key": "treated",
        "type": "choice",
        "count": {"min": 1},
        "options": ["S05", "S06", "S07"],
        "default": [{"id": "S05"}],
    }
    reply = {"selected": [{"id": "S05"}, {"id": "S07"}]}
    drawn, result = _ask(_request(field), ("submit", reply))
    data = drawn[0][1]
    assert (data["min"], data["max"], data["default"]) == (1, 3, ["S05"])
    assert result["answer"] == {"treated": [{"id": "S05"}, {"id": "S07"}]}


def test_option_with_fields_asks_its_follow_up():
    field = {
        "key": "qc",
        "type": "choice",
        "question": "Keep Donor B (S02)?",
        "details": "Low library size",
        "options": [
            "retain",
            {"id": "exclude", "fields": [{"key": "reason", "type": "text"}]},
        ],
    }
    drawn, result = _ask(
        _request(field),
        ("submit", {"selected": [{"id": "exclude"}]}),
        ("submit", {"answer": " library too small "}),
    )
    assert [component for component, _ in drawn] == ["choice", "text_input"]
    assert drawn[0][1]["question"] == "Keep Donor B (S02)?"
    assert drawn[0][1]["details"] == "Low library size"
    assert result["answer"] == {
        "qc": {"id": "exclude", "fields": {"reason": "library too small"}}
    }


def test_for_each_asks_once_per_available_item():
    """for_each shapes other than a choice list keep one render per item."""
    field = {
        "key": "note",
        "type": "text",
        "for_each": ["S01", {"id": "S02", "unavailable": "excluded"}, "S03"],
    }
    drawn, result = _ask(
        _request(field),
        ("submit", {"answer": "fine"}),
        ("submit", {"answer": "re-run"}),
    )
    assert [d["question"] for _, d in drawn] == ["S01: Review", "S03: Review"]
    assert result["answer"] == {"note": {"S01": "fine", "S03": "re-run"}}


def test_for_each_choice_with_count_keeps_one_render_per_item():
    field = {
        "key": "marks",
        "type": "choice",
        "count": {"min": 1},
        "options": ["a", "b"],
        "for_each": ["S1", "S2"],
    }
    drawn, result = _ask(
        _request(field),
        ("submit", {"selected": [{"id": "a"}]}),
        ("submit", {"selected": [{"id": "a"}, {"id": "b"}]}),
    )
    assert [component for component, _ in drawn] == ["choice", "choice"]
    assert result["answer"]["marks"]["S2"] == [{"id": "a"}, {"id": "b"}]


# --- number ---


def test_slider_number():
    field = {
        "key": "mito",
        "type": "number",
        "view": "slider",
        "min": 0,
        "max": 0.5,
        "unit": "fraction",
        "default": 0.2,
    }
    drawn, result = _ask(_request(field), ("submit", {"value": 0.15}))
    assert drawn == [
        (
            "threshold_slider",
            {
                "label": "Review",
                "min": 0,
                "max": 0.5,
                "default": 0.2,
                "unit": "fraction",
            },
        )
    ]
    assert result["answer"] == {"mito": 0.15}


def test_integer_slider_steps_by_one_and_answers_int():
    field = {
        "key": "n",
        "type": "number",
        "view": "slider",
        "min": 1,
        "max": 9,
        "integer": True,
    }
    drawn, result = _ask(_request(field), ("submit", {"value": 4.0}))
    assert drawn[0][1]["step"] == 1
    assert result["answer"] == {"n": 4}


def test_typed_number_is_parsed_and_re_asked_when_invalid():
    field = {"key": "reads", "type": "number", "integer": True, "min": 0}
    drawn, result = _ask(
        _request(field),
        ("submit", {"answer": "lots"}),
        ("submit", {"answer": "2.5"}),
        ("submit", {"answer": "10000"}),
    )
    assert [component for component, _ in drawn] == ["text_input"] * 3
    assert "Enter a whole number, min 0" in drawn[0][1]["question"]
    assert drawn[1][1]["question"].startswith("'lots' is not a number.")
    assert drawn[2][1]["question"].startswith("2.5 is not a whole number.")
    assert result["answer"] == {"reads": 10000}


# --- text, record ---


def test_text_with_default_keeps_it_on_empty_enter():
    field = {"key": "note", "type": "text", "default": "looks fine"}
    drawn, result = _ask(_request(field), ("submit", {"answer": ""}))
    assert drawn[0][1]["placeholder"] == "Enter = keep looks fine"
    assert result["answer"] == {"note": "looks fine"}


def test_record_asks_each_sub_field_in_order():
    field = {
        "key": "mito",
        "type": "record",
        "profile": "interval",
        "fields": [
            {"key": "from", "type": "number", "view": "slider", "min": 0, "max": 1},
            {"key": "to", "type": "number", "view": "slider", "min": 0, "max": 1},
        ],
    }
    drawn, result = _ask(
        _request(field), ("submit", {"value": 0.0}), ("submit", {"value": 0.2})
    )
    assert [c for c, _ in drawn] == ["threshold_slider", "threshold_slider"]
    assert result["answer"] == {"mito": {"from": 0.0, "to": 0.2}}


# --- plate map ---

PLATE = {
    "key": "groups",
    "type": "choice",
    "view": "plate_map",
    "question": "Group of each sample?",
    "for_each": [
        {"id": "S01", "label": "Donor A"},
        {"id": "S05", "label": "Donor A", "details": "metadata: treated"},
        {"id": "S08", "label": "Donor H", "details": "condition unknown"},
    ],
    "options": ["control", "treated"],
    "default": {"S01": {"id": "control"}, "S05": {"id": "treated"}},
}


def test_plate_map_lays_items_out_as_wells():
    reply = {
        "assignments": {"S01": "control", "S05": "treated", "S08": "treated"},
        "changes": [],
    }
    drawn, result = _ask(_request(PLATE), ("submit", reply))
    ((component, data),) = drawn
    assert component == "plate_map"
    assert data["choices"] == ["control", "treated"]
    assert data["wells"][1] == {
        "row": 0,
        "col": 1,
        "sample_id": "S05",
        "label": "Donor A",
        "condition": "treated",
        "details": {"note": "metadata: treated"},
    }
    assert data["wells"][2]["condition"] == ""  # no default: must be resolved
    assert result["answer"]["groups"]["S08"] == {"id": "treated"}


# --- dashboard ---

DASHBOARD = [
    {**SOURCE, "question": "Count source?"},
    {**PLATE, "depends_on": ["source"]},
    {
        "key": "mito",
        "type": "number",
        "question": "Mito cutoff?",
        "view": "slider",
        "min": 0,
        "max": 0.5,
        "depends_on": ["groups"],
    },
]


def test_dashboard_is_one_render_with_a_section_per_field():
    reply = {
        "answers": {
            "source": {"selected": [{"id": "SRC-A"}]},
            "groups": {
                "assignments": {"S01": "control", "S05": "treated", "S08": "treated"}
            },
            "mito": {"value": 0.2},
        }
    }
    drawn, result = _ask(_request(*DASHBOARD, view="dashboard"), ("submit", reply))
    ((component, data),) = drawn
    assert component == "dashboard"
    assert [(s["key"], s["component"], s["depends_on"]) for s in data["sections"]] == [
        ("source", "choice", []),
        ("groups", "plate_map", ["source"]),
        ("mito", "threshold_slider", ["groups"]),
    ]
    assert result["answer"]["mito"] == 0.2


def test_dashboard_bad_typed_number_is_reported_not_crashed():
    sections = [DASHBOARD[0], {"key": "n", "type": "number", "question": "N?"}]
    reply = {
        "answers": {"source": {"selected": [{"id": "SRC-A"}]}, "n": {"answer": "x"}}
    }
    tui = FakeTUI(("submit", reply))
    request = _request(*sections, view="dashboard")
    action, answer = collect(request, tui)
    assert (action, answer["n"]) == ("submit", "x")


# --- choice_list: one render for a per-item choice ---

QC = {
    "key": "qc",
    "type": "choice",
    "question": "Keep each flagged sample?",
    "details": "Warnings are information, not a decision.",
    "for_each": [
        {"id": "S02", "label": "Donor B", "details": "Low library size"},
        {"id": "S06", "label": "Donor F", "details": "High mitochondrial fraction"},
        {"id": "S07", "unavailable": "already removed upstream"},
    ],
    "options": [
        "retain",
        {
            "id": "exclude",
            "fields": [{"key": "reason", "type": "text", "question": "Why?"}],
        },
        "defer",
    ],
}


def test_qc_review_is_one_choice_list_render():
    reply = {
        "answers": {
            "S02": {"id": "retain"},
            "S06": {"id": "exclude", "fields": {"reason": " high mito "}},
        }
    }
    drawn, result = _ask(_request(QC), ("submit", reply))
    assert drawn == [
        (
            "choice_list",
            {
                "question": "Keep each flagged sample?",
                "details": "Warnings are information, not a decision.",
                "items": [
                    {"id": "S02", "label": "Donor B", "details": "Low library size"},
                    {
                        "id": "S06",
                        "label": "Donor F",
                        "details": "High mitochondrial fraction",
                    },
                    {
                        "id": "S07",
                        "label": "S07",
                        "unavailable": "already removed upstream",
                    },
                ],
                "options": [
                    {"id": "retain", "label": "retain"},
                    {
                        "id": "exclude",
                        "label": "exclude",
                        "reason": {"key": "reason", "question": "Why?"},
                    },
                    {"id": "defer", "label": "defer"},
                ],
            },
        )
    ]
    assert result["answer"] == {
        "qc": {
            "S02": {"id": "retain"},
            "S06": {"id": "exclude", "fields": {"reason": "high mito"}},
        }
    }
    assert "S07" not in result["answer"]["qc"]  # shown, never answered


def test_choice_list_default_is_its_default_by_item():
    field = {**QC, "default": {"S02": {"id": "retain"}}}
    drawn, _ = _ask(
        _request(field),
        ("submit", {"answers": {"S02": {"id": "retain"}, "S06": {"id": "defer"}}}),
    )
    assert drawn[0][1]["default"] == {"S02": {"id": "retain"}}


def test_option_with_two_fields_keeps_one_render_per_item():
    field = {
        **QC,
        "for_each": ["S02"],
        "options": [
            "retain",
            {
                "id": "exclude",
                "fields": [
                    {"key": "reason", "type": "text"},
                    {"key": "by", "type": "text"},
                ],
            },
        ],
    }
    drawn, _ = _ask(_request(field), ("submit", {"selected": [{"id": "retain"}]}))
    assert [component for component, _ in drawn] == ["choice"]


def test_dashboard_qc_section_is_a_choice_list():
    request = _request(
        {**SOURCE, "question": "Count source?"},
        {**QC, "depends_on": ["source"]},
        view="dashboard",
    )
    reply = {
        "answers": {
            "source": {"selected": [{"id": "SRC-A"}]},
            "qc": {
                "answers": {
                    "S02": {"id": "defer"},
                    "S06": {"id": "exclude", "fields": {"reason": "mito"}},
                }
            },
        }
    }
    drawn, result = _ask(request, ("submit", reply))
    sections = drawn[0][1]["sections"]
    assert [s["component"] for s in sections] == ["choice", "choice_list"]
    assert result["answer"]["qc"]["S06"] == {
        "id": "exclude",
        "fields": {"reason": "mito"},
    }


def test_plate_lists_unavailable_items_below_it_never_as_wells():
    field = {
        **PLATE,
        "for_each": [
            *PLATE["for_each"],
            {"id": "S09", "label": "Donor J", "unavailable": "consent withdrawn"},
        ],
    }
    reply = {"assignments": {"S01": "control", "S05": "treated", "S08": "treated"}}
    drawn, result = _ask(_request(field), ("submit", reply))
    data = drawn[0][1]
    assert [w["sample_id"] for w in data["wells"]] == ["S01", "S05", "S08"]
    assert data["unavailable"] == [
        {"sample_id": "S09", "label": "Donor J", "reason": "consent withdrawn"}
    ]
    assert "S09" not in result["answer"]["groups"]


# --- re-ask: problems shown, answers kept ---


def _reasked(request, problems, **defaults):
    """The request as ask_user re-asks it: next version, problems, defaults."""
    from lobster.services.interaction.component_schemas import Problem, reask

    parsed = DecisionRequest.model_validate(request)
    return reask(
        parsed, [Problem(path=p, message=m) for p, m in problems], defaults
    ).to_wire()


_KEPT = (
    "Asked again (attempt 2 of 5): fix what is marked; your earlier answers are kept."
)


def test_reask_shows_problems_in_details_and_prefills():
    request = _reasked(
        _request(SOURCE),
        [("source", "option 'SRC-C' is unavailable (Normalized; build unknown)")],
        source={"id": "SRC-B"},
    )
    drawn, _ = _ask(request, ("submit", {"selected": [{"id": "SRC-A"}]}))
    data = drawn[0][1]
    assert data["details"] == (
        "⚠ Asked again (attempt 2 of 5): fix what is marked; your earlier answers"
        " are kept.\n"
        "⚠ option 'SRC-C' is unavailable (Normalized; build unknown)\n"
        "Recommended: SRC-A"
    )
    assert data["default"] == ["SRC-B"]


def test_reask_of_a_choice_list_names_the_item_and_keeps_good_rows():
    request = _reasked(
        _request(QC),
        [("qc.S06.fields", "option 'exclude' needs its fields")],
        qc={"S02": {"id": "retain"}, "S06": {"id": "exclude"}},
    )
    drawn, _ = _ask(
        request,
        ("submit", {"answers": {"S02": {"id": "retain"}, "S06": {"id": "defer"}}}),
    )
    data = drawn[0][1]
    assert data["details"].startswith(
        f"⚠ {_KEPT}\n⚠ S06: option 'exclude' needs its fields\n"
    )
    assert data["default"] == {"S02": {"id": "retain"}}  # the bad row is dropped


DIRECTION = {
    "key": "direction",
    "type": "choice",
    "question": "Compare which with which?",
    "options": [
        {"id": "t_vs_c", "label": "treated compared with control"},
        {"id": "c_vs_t", "label": "control compared with treated"},
    ],
}


def _review_dashboard():
    return _request(
        {**SOURCE, "question": "Count source?"},
        {**PLATE, "depends_on": ["source"]},
        {**DIRECTION, "depends_on": ["groups"]},
        {**QC, "depends_on": ["groups"]},
        {"key": "notes", "type": "text", "question": "Anything to note?"},
        view="dashboard",
    )


def test_first_dashboard_has_no_answer_and_no_problems():
    sections = dashboard_data(DecisionRequest.model_validate(_review_dashboard()))
    assert "problems" not in sections
    for section in sections["sections"]:
        assert "answer" not in section and "problems" not in section


def test_reasked_dashboard_routes_problems_and_sends_every_previous_answer():
    message = "'control' has no items left after removing S01 (qc: exclude)"
    request = _reasked(
        _review_dashboard(),
        [
            ("groups", message),
            ("qc", message),
            ("qc.S06.fields.reason", "expected non-empty text"),
            ("data_files", "the data changed while you were reviewing it: c.csv"),
        ],
        source={"id": "SRC-A"},
        groups={"S01": {"id": "control"}, "S05": {"id": "treated"}},
        direction={"id": "t_vs_c"},
        qc={"S02": {"id": "exclude", "fields": {"reason": "small"}}},
        notes="ok",
    )
    data = dashboard_data(DecisionRequest.model_validate(request))
    assert data["problems"] == [
        _KEPT,
        "the data changed while you were reviewing it: c.csv",
    ]
    by_key = {s["key"]: s for s in data["sections"]}
    assert by_key["groups"]["problems"] == [message]
    assert by_key["qc"]["problems"] == [message, "S06 reason: expected non-empty text"]
    assert "problems" not in by_key["source"]
    # Every section with a previous answer gets it, in its component's shape.
    assert by_key["source"]["answer"] == {"selected": [{"id": "SRC-A"}]}
    assert by_key["groups"]["answer"] == {
        "assignments": {"S01": "control", "S05": "treated"}
    }
    assert by_key["direction"]["answer"] == {"selected": [{"id": "t_vs_c"}]}
    assert by_key["qc"]["answer"] == {
        "answers": {"S02": {"id": "exclude", "fields": {"reason": "small"}}}
    }
    assert by_key["notes"]["answer"] == {"answer": "ok"}
    # Section data carries no ⚠ lines: the dashboard shows problems itself.
    assert "⚠" not in json.dumps(data["sections"][3]["data"], ensure_ascii=False)


def test_reasked_dashboard_section_without_a_previous_answer_gets_none():
    groups = {"S01": {"id": "control"}, "S05": {"id": "treated"}}
    request = _reasked(
        _review_dashboard(), [("notes", "no answer")], groups=groups, notes=None
    )
    by_key = {
        s["key"]: s
        for s in dashboard_data(DecisionRequest.model_validate(request))["sections"]
    }
    assert "answer" not in by_key["notes"]
    assert by_key["groups"]["answer"]["assignments"] == {
        "S01": "control",
        "S05": "treated",
    }  # the groups the user submitted


def _proposed_dashboard():
    """A dashboard where the supervisor proposes a value for every section."""
    return _request(
        {**SOURCE, "question": "Count source?", "default": {"id": "SRC-A"}},
        {**PLATE, "depends_on": ["source"]},
        {**DIRECTION, "default": {"id": "t_vs_c"}},
        view="dashboard",
    )


def test_misrouted_reask_shows_problems_but_no_section_is_answered():
    """The supervisor's proposals must never look like the user's answer."""
    from lobster.services.interaction.component_schemas import Problem, reask

    request = DecisionRequest.model_validate(_proposed_dashboard())
    message = "this answer is for another question"
    again = reask(request, [Problem(path="responds_to", message=message)], None)
    assert again.prefilled_from_answer is False
    data = dashboard_data(again)
    # Not "answers are kept": the defaults are still the supervisor's.
    assert data["problems"] == [
        "Asked again (attempt 2 of 5): fix what is marked.",
        message,
    ]
    for section in data["sections"]:
        assert "answer" not in section
    by_key = {s["key"]: s for s in data["sections"]}
    # The proposals still pre-fill the controls, as on the first ask.
    assert by_key["source"]["data"]["default"] == ["SRC-A"]
    assert by_key["groups"]["data"]["wells"][0]["condition"] == "control"


def test_rejected_reask_marks_the_submitted_sections_answered():
    from lobster.services.interaction.component_schemas import Problem, reask

    request = DecisionRequest.model_validate(_proposed_dashboard())
    submitted = {
        "source": {"id": "SRC-B"},
        "groups": {"S01": {"id": "control"}, "S05": {"id": "treated"}},
        "direction": {"id": "c_vs_t"},
    }
    problem = Problem(path="groups.S08", message="no answer")
    data = dashboard_data(reask(request, [problem], submitted))
    by_key = {s["key"]: s for s in data["sections"]}
    assert by_key["groups"]["problems"] == ["S08: no answer"]
    assert by_key["source"]["answer"] == {"selected": [{"id": "SRC-B"}]}
    assert by_key["direction"]["answer"] == {"selected": [{"id": "c_vs_t"}]}
    assert by_key["groups"]["answer"] == {
        "assignments": {"S01": "control", "S05": "treated"}
    }


# --- older TUI: the whole question in words, free-text answer ---


def test_fallback_prompt_states_the_whole_question():
    prompt = fallback_prompt(_request(SOURCE))
    assert prompt == (
        "Review\n"
        "\n"
        "Recommended: SRC-A\n"
        "Pick one:\n"
        "- SRC-A — Author matrix: Raw counts · Build-A\n"
        "- SRC-B\n"
        "- SRC-C (unavailable: Normalized; build unknown)\n"
        "\n"
        "Write your answer in your own words."
    )


def test_fallback_prompt_lists_items_and_dashboard_sections_in_order():
    request = _reasked(
        _review_dashboard(), [("groups", "'control' has no items")], notes="ok"
    )
    prompt = fallback_prompt(request)
    order = [
        "Review",
        "⚠ groups: 'control' has no items",
        "1. Count source?",
        "- SRC-C (unavailable: Normalized; build unknown)",
        "2. Group of each sample?",
        "For each of:",
        "- S01 — Donor A",
        "- S08 — Donor H: condition unknown",
        "Pick one per item:",
        "- control",
        "3. Compare which with which?",
        "- t_vs_c — treated compared with control",
        "4. Keep each flagged sample?",
        "- S02 — Donor B: Low library size",
        "- S07 (unavailable: already removed upstream)",
        "- exclude (then give: Why?)",
        "5. Anything to note?",
        "Answer in text.",
        "Write your answer in your own words.",
    ]
    positions = [prompt.index(line) for line in order]
    assert positions == sorted(positions)
    assert prompt.endswith("Write your answer in your own words.")


@pytest.mark.parametrize(
    "request_",
    [
        _request(SOURCE),
        _request(PLATE),
        _request(QC),
        _request(*DASHBOARD, view="dashboard"),
    ],
    ids=["choice", "plate_map", "choice_list", "dashboard"],
)
def test_text_box_reply_from_an_older_tui_is_free_text(request_):
    tui = FakeTUI(("submit", {"answer": "  SRC-A please  "}))
    action, answer = collect(request_, tui)
    assert (action, answer) == ("free_text", {"text": "SRC-A please"})
    envelope = build_answer(request_, action, answer)
    assert envelope["text"] == "SRC-A please" and "answer" not in envelope
    result = check_answer(
        DecisionRequest.model_validate(request_),
        DecisionAnswer.model_validate(envelope),
    )
    assert result["action"] == "free_text"  # passed on, not checked


def test_text_box_reply_ends_a_per_item_question_at_once():
    field = {
        "key": "marks",
        "type": "choice",
        "count": {"min": 1},
        "options": ["a", "b"],
        "for_each": ["S1", "S2"],
    }
    tui = FakeTUI(("submit", {"answer": "S1 a, S2 b"}))
    assert collect(_request(field), tui) == ("free_text", {"text": "S1 a, S2 b"})
    assert len(tui.drawn) == 1


def test_empty_text_box_reply_shows_the_question_again():
    tui = FakeTUI(
        ("submit", {"answer": " "}), ("submit", {"selected": [{"id": "SRC-A"}]})
    )
    assert collect(_request(SOURCE), tui) == ("submit", {"source": {"id": "SRC-A"}})
    assert len(tui.drawn) == 2


def test_a_text_question_answered_in_text_is_not_free_text():
    tui = FakeTUI(("submit", {"answer": "looks fine"}))
    request = _request({"key": "note", "type": "text"})
    assert collect(request, tui) == ("submit", {"note": "looks fine"})


# --- stopping, envelope, limits ---


@pytest.mark.parametrize("action", ["cancel", "defer"])
def test_stopping_part_way_returns_no_answer(action):
    field = {"key": "note", "type": "text", "for_each": ["S1", "S2"]}
    tui = FakeTUI(("submit", {"answer": "fine"}), (action, {}))
    assert collect(_request(field), tui) == (action, {})


def test_answer_envelope_echoes_identity():
    request = _request(SOURCE)
    envelope = build_answer(request, "submit", {"source": {"id": "SRC-A"}}, "ok")
    assert envelope["responds_to"] == {
        "decision_id": request["decision_id"],
        "request_version": 1,
    }
    assert envelope["answer_id"].startswith("ans-")
    assert envelope["comment"] == "ok"
    assert "answer" not in build_answer(request, "cancel")


def test_unsupported_names_what_the_terminal_cannot_draw():
    lists = {"key": "times", "type": "number", "count": {"min": 1}, "unit": "h"}
    nested = {
        "key": "qc",
        "type": "choice",
        "options": [
            {
                "id": "x",
                "fields": [{"key": "notes", "type": "text", "count": {"min": 0}}],
            }
        ],
    }
    assert unsupported(_request(SOURCE)) == []
    assert unsupported(_request(lists)) == ["times: a list of number values (count)"]
    assert unsupported(_request(nested)) == [
        "qc.x.notes: a list of text values (count)"
    ]


def _samples(count, unavailable=0):
    open_ = [f"S{n:03d}" for n in range(1, count + 1)]
    gone = [{"id": f"X{n:02d}", "unavailable": "removed"} for n in range(unavailable)]
    return open_ + gone


@pytest.mark.parametrize(
    "field, problem",
    [
        (
            {**PLATE, "options": [f"g{n}" for n in range(10)], "default": None},
            "groups: a plate with 10 options (at most 9)",
        ),
        (
            {**PLATE, "for_each": _samples(97), "default": None},
            "groups: a plate with 97 samples (at most 96)",
        ),
        (
            {
                **PLATE,
                "for_each": [{"id": "S09", "unavailable": "gone"}],
                "default": None,
            },
            "groups: a plate where every item is unavailable",
        ),
        (
            {**QC, "for_each": [{"id": "S07", "unavailable": "already removed"}]},
            "qc: a list where every item is unavailable",
        ),
        (
            {
                "key": "mito",
                "type": "number",
                "question": "Mito cutoff?",
                "view": "slider",
                "min": 0.2,
                "max": 0.2,
            },
            "mito: a slider whose min and max are both 0.2",
        ),
    ],
)
def test_unsupported_lists_the_shapes_the_go_screens_refuse(field, problem):
    """Go's Init refuses these, so they are cancelled before any render; as a
    dashboard section too. The classic UI asks in text and can show them."""
    assert unsupported(_request(field)) == [problem]
    first = {**SOURCE, "question": "Count source?"}
    dashboard = _request(first, field, view="dashboard")
    assert unsupported(dashboard) == [problem]
    assert unsupported(_request(field), go_screens=False) == []


def test_unavailable_options_and_samples_do_not_count_against_the_plate():
    options = [f"g{n}" for n in range(9)] + [{"id": "g9", "unavailable": "none left"}]
    field = {**PLATE, "options": options, "for_each": _samples(96, 3), "default": None}
    assert unsupported(_request(field)) == []


def test_plate_of_96_samples_fills_all_96_wells():
    field = {**PLATE, "for_each": _samples(96), "default": None}
    reply = {"assignments": {sample: "control" for sample in _samples(96)}}
    drawn, result = _ask(_request(field), ("submit", reply))
    data = drawn[0][1]
    assert data["format"] == 96
    assert (data["wells"][-1]["row"], data["wells"][-1]["col"]) == (7, 11)
    assert result["action"] == "submit"


def test_cannot_show_comment_tells_the_supervisor_not_to_retry_the_shape():
    comment = cannot_show(["times: a list of number values (count)"])
    assert comment == (
        "Not answered: this terminal can't show times: a list of number values "
        "(count). Don't ask again with the same shape; ask in chat or with a shape "
        "it can show."
    )


# --- Cross-language fixtures: Python adapter <-> Go dashboard ---
#
# Python writes the dashboard data it sends; a Go test
# (lobster-tui/internal/biocomp/dashboard/fixture_test.go) loads it, plays a
# scientist's keys, and writes the dashboard's reply; the test below checks
# that reply against the request. Regenerate both files with:
#   LOBSTER_UPDATE_DASHBOARD_FIXTURE=1 pytest tests/unit/cli/test_go_tui_components.py
#   cd lobster-tui && LOBSTER_UPDATE_DASHBOARD_FIXTURE=1 go test ./internal/biocomp/dashboard/
#   pytest tests/unit/cli/test_go_tui_components.py

GO_FIXTURES = (
    Path(__file__).resolve().parents[3]
    / "lobster-tui"
    / "internal"
    / "biocomp"
    / "dashboard"
    / "testdata"
)

# A synthetic example (all values fictional): count source, sample groups,
# comparison direction and QC review, with the checks a supervisor attaches.
_SAMPLES = [
    ("S01", "Donor A", "control", None),
    ("S02", "Donor B", "control", "Low library size"),
    ("S03", "Donor C", "control", None),
    ("S04", "Donor D", "control", None),
    ("S05", "Donor A", "treated", None),
    ("S06", "Donor F", "treated", "High mitochondrial fraction"),
    ("S07", "Donor G", "treated", None),
    ("S08", "Donor H", "unknown", "Condition label unresolved"),
]
EXAMPLE_DASHBOARD = [
    {
        "key": "source",
        "type": "choice",
        "question": "Which count source?",
        "options": [
            {
                "id": "SRC-A",
                "label": "Author matrix",
                "details": "Raw counts · Build-A",
            },
            {
                "id": "SRC-B",
                "label": "Repository matrix",
                "details": "Raw counts · Build-A",
            },
            {
                "id": "SRC-C",
                "label": "Author summary",
                "unavailable": "Normalized; reference build unknown",
            },
        ],
    },
    {
        "key": "groups",
        "type": "choice",
        "question": "Group of each sample?",
        "view": "plate_map",
        "depends_on": ["source"],
        "for_each": [
            {"id": i, "label": label, **({"details": warn} if warn else {})}
            for i, label, _, warn in _SAMPLES
        ],
        "options": [
            "control",
            "treated",
            {"id": "leave_out", "label": "leave out of the comparison"},
        ],
        # S08 is unknown: no default, the scientist resolves it.
        "default": {i: {"id": c} for i, _, c, _ in _SAMPLES if c != "unknown"},
    },
    {
        "key": "direction",
        "type": "choice",
        "question": "Compare which with which?",
        "depends_on": ["groups"],
        "options": [
            {"id": "treated_vs_control", "label": "treated compared with control"},
            {"id": "control_vs_treated", "label": "control compared with treated"},
        ],
    },
    {
        "key": "qc",
        "type": "choice",
        "question": "Keep each flagged sample?",
        "depends_on": ["groups"],
        "for_each": [
            {"id": i, "label": label, "details": warn}
            for i, label, _, warn in _SAMPLES
            if warn and i != "S08"
        ],
        "options": [
            "retain",
            {
                "id": "exclude",
                "fields": [{"key": "reason", "type": "text", "question": "Why?"}],
            },
            "defer",
        ],
    },
]
EXAMPLE_CHECKS = [
    {
        "check": "each_option_kept",
        "field": "groups",
        "options": ["control", "treated"],
        "minus": {"field": "qc", "options": ["exclude", "defer"]},
    },
    {"check": "reason_required", "field": "qc", "options": ["exclude"]},
    {"check": "no_default_for", "field": "qc", "options": ["exclude"]},
]


def _example_request():
    return _request(
        *EXAMPLE_DASHBOARD,
        view="dashboard",
        decision_id="dec-fixture",
        checks=EXAMPLE_CHECKS,
    )


def test_example_dashboard_passes_its_request_time_checks():
    from lobster.services.interaction.component_schemas import check_request

    context = AskUserContext.model_validate(
        {"view": "dashboard", "fields": EXAMPLE_DASHBOARD, "checks": EXAMPLE_CHECKS}
    )
    assert check_request(context) == []


def test_dashboard_data_matches_the_go_fixture():
    produced = dashboard_data(DecisionRequest.model_validate(_example_request()))
    path = GO_FIXTURES / "from_python.json"
    if os.environ.get("LOBSTER_UPDATE_DASHBOARD_FIXTURE") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(produced, indent=2, ensure_ascii=False) + "\n")
    assert json.loads(path.read_text()) == produced
    sections = {s["key"]: s for s in produced["sections"]}
    assert [s["component"] for s in produced["sections"]] == [
        "choice",
        "plate_map",
        "choice",
        "choice_list",
    ]
    wells = {w["sample_id"]: w for w in sections["groups"]["data"]["wells"]}
    assert wells["S08"]["condition"] == ""  # unresolved, never guessed


def test_go_dashboard_reply_becomes_a_valid_answer():
    """The Go reply passes the shape checks and every answer-time check."""
    reply = json.loads((GO_FIXTURES / "reply_from_go.json").read_text())
    _, result = _ask(_example_request(), ("submit", reply))
    answer = result["answer"]
    assert answer["source"] == {"id": "SRC-A"}
    assert answer["groups"]["S08"] == {"id": "treated"}  # resolved by the scientist
    assert answer["groups"]["S01"] == {"id": "control"}
    assert answer["direction"] == {"id": "treated_vs_control"}
    assert answer["qc"] == {
        "S02": {"id": "retain"},
        "S06": {"id": "exclude", "fields": {"reason": "high mito fraction"}},
    }


# The same dashboard asked again: the scientist left S01, S03 and S04 out and
# excluded S02, the only control left. Python refuses it (each_option_kept) and
# re-asks; Go loads the re-ask (reask_from_python.json), the scientist puts S01
# back into control, re-checks the direction and keeps QC, and Go writes the
# reply (reask_reply_from_go.json), which must now be accepted.
_EMPTIED = {
    "source": {"id": "SRC-A"},
    "groups": {
        **{i: {"id": c} for i, _, c, _ in _SAMPLES if c != "unknown"},
        "S01": {"id": "leave_out"},
        "S03": {"id": "leave_out"},
        "S04": {"id": "leave_out"},
        "S08": {"id": "treated"},
    },
    "direction": {"id": "treated_vs_control"},
    "qc": {
        "S02": {"id": "exclude", "fields": {"reason": "low library size"}},
        "S06": {"id": "retain"},
    },
}


def _example_reask():
    from lobster.services.interaction.component_schemas import AnswerRejected, reask

    request = DecisionRequest.model_validate(_example_request())
    envelope = build_answer(request.to_wire(), "submit", _EMPTIED)
    with pytest.raises(AnswerRejected) as refused:
        check_answer(request, DecisionAnswer.model_validate(envelope))
    return reask(request, refused.value.problems, _EMPTIED).to_wire()


def test_reasked_dashboard_matches_the_go_fixture():
    request = DecisionRequest.model_validate(_example_reask())
    produced = dashboard_data(request)
    path = GO_FIXTURES / "reask_from_python.json"
    if os.environ.get("LOBSTER_UPDATE_DASHBOARD_FIXTURE") == "1":
        path.write_text(json.dumps(produced, indent=2, ensure_ascii=False) + "\n")
    assert json.loads(path.read_text()) == produced
    sections = {s["key"]: s for s in produced["sections"]}
    message = "'control' has no items left after removing S02 (Donor B) (qc: exclude)"
    assert sections["groups"]["problems"] == [message]
    assert sections["qc"]["problems"] == [message]
    assert all("answer" in s for s in produced["sections"])  # answers kept


def test_go_dashboard_reply_to_the_reask_is_accepted():
    reply = json.loads((GO_FIXTURES / "reask_reply_from_go.json").read_text())
    _, result = _ask(_example_reask(), ("submit", reply))
    answer = result["answer"]
    assert answer["groups"]["S01"] == {"id": "control"}  # the fix made in Go
    assert answer["groups"]["S03"] == {"id": "leave_out"}  # kept from before
    assert answer["qc"]["S02"]["fields"] == {"reason": "low library size"}
    assert answer["direction"] == {"id": "treated_vs_control"}


# --- Full path: fake TUI, real checkpointed LangGraph, real ask_user ---


def _one_question_graph(question, context):
    """An agent that calls ask_user once, then stops (no LLM)."""
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    from lobster.tools.user_interaction import create_ask_user_tool

    def agent(state):
        if any(m.type == "tool" for m in state["messages"]):
            return {"messages": [AIMessage(content="done")]}
        call = {
            "name": "ask_user",
            "id": "call_1",
            "args": {"question": question, "context": context},
        }
        return {"messages": [AIMessage(content="", tool_calls=[call])]}

    def route(state):
        return "tools" if state["messages"][-1].tool_calls else END

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode([create_ask_user_tool()]))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, ["tools", END])
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=InMemorySaver())


def _run_with_tui(graph, render, tamper=None):
    """Answer every pause through the adapter, as the launcher does; return
    the tool result and the requests shown. ``tamper(envelope, n)`` may
    change the n-th answer (from 1) before it is sent."""
    from langgraph.types import Command

    config = {"configurable": {"thread_id": "full-path"}}
    shown, graph_input = [], {"messages": []}
    while True:
        for _ in graph.stream(graph_input, config, stream_mode="updates"):
            pass
        pending = [i for t in graph.get_state(config).tasks for i in t.interrupts]
        if not pending:
            break
        ((interrupt,),) = [pending]
        request = interrupt.value
        shown.append(request)
        envelope = build_answer(request, *collect(request, render))
        if tamper is not None:
            envelope = tamper(envelope, len(shown))
        graph_input = Command(resume={interrupt.id: envelope})
    messages = graph.get_state(config).values["messages"]
    (message,) = [m for m in messages if m.type == "tool"]
    return json.loads(message.content), shown


_CONTROLS_DASHBOARD = {
    "view": "dashboard",
    "fields": [
        {
            "key": "groups",
            "type": "choice",
            "question": "Group of each sample?",
            "view": "plate_map",
            "for_each": ["S01", "S02", "S03", "S04"],
            "options": ["control", "treated"],
            "default": {
                "S01": {"id": "control"},
                "S02": {"id": "control"},
                "S03": {"id": "treated"},
                "S04": {"id": "treated"},
            },
        },
        {
            "key": "qc",
            "type": "choice",
            "question": "Keep each flagged sample?",
            "depends_on": ["groups"],
            "for_each": [
                {"id": "S01", "details": "Low library size"},
                {"id": "S02", "details": "High mitochondrial fraction"},
            ],
            "options": [
                "retain",
                {"id": "exclude", "fields": [{"key": "reason", "type": "text"}]},
                "defer",
            ],
        },
    ],
    "checks": [
        {
            "check": "each_option_kept",
            "field": "groups",
            "options": ["control", "treated"],
            "minus": {"field": "qc", "options": ["exclude", "defer"]},
        },
        {"check": "reason_required", "field": "qc", "options": ["exclude"]},
        {"check": "no_default_for", "field": "qc", "options": ["exclude"]},
    ],
}


def test_full_path_every_control_excluded_is_reasked_then_accepted():
    excluded = {
        "S01": {"id": "exclude", "fields": {"reason": "too few reads"}},
        "S02": {"id": "exclude", "fields": {"reason": "degraded"}},
    }
    seen = []

    def tui(component, data):
        assert component == "dashboard"
        sections = {s["key"]: s for s in data["sections"]}
        seen.append(sections)
        if len(seen) == 1:  # first ask: exclude every control
            wells = sections["groups"]["data"]["wells"]
            groups = {"assignments": {w["sample_id"]: w["condition"] for w in wells}}
            qc = {"answers": excluded}
            return "submit", {"answers": {"groups": groups, "qc": qc}}
        # The re-ask: Go would start every section from its "answer".
        kept = {key: s["answer"] for key, s in sections.items()}
        kept["qc"] = {"answers": {**excluded, "S02": {"id": "retain"}}}
        return "submit", {"answers": kept}

    result, shown = _run_with_tui(
        _one_question_graph("Review groups and QC", _CONTROLS_DASHBOARD), tui
    )

    first, second = seen
    assert "answer" not in first["groups"] and "problems" not in first["qc"]
    message = "'control' has no items left after removing S01, S02 (qc: exclude)"
    assert second["groups"]["problems"] == [message]
    assert second["qc"]["problems"] == [message]
    assert second["qc"]["answer"] == {"answers": excluded}  # answers kept
    assert second["groups"]["answer"]["assignments"]["S01"] == "control"
    assert [r["request_version"] for r in shown] == [1, 2]
    assert (result["action"], result["attempts"]) == ("submit", 2)
    assert result["answer"]["qc"]["S02"] == {"id": "retain"}
    assert result["answer"]["qc"]["S01"]["fields"] == {"reason": "too few reads"}


def test_full_path_misrouted_answer_reasks_without_ticking_proposals():
    """A reply for another question is re-asked: the problem is shown, the
    supervisor's proposals stay proposals (no section starts answered), and
    the next valid answer is accepted."""
    seen = []
    valid = {
        "groups": {
            "assignments": {
                "S01": "control",
                "S02": "control",
                "S03": "treated",
                "S04": "treated",
            }
        },
        "qc": {"answers": {"S01": {"id": "retain"}, "S02": {"id": "retain"}}},
    }

    def tui(component, data):
        assert component == "dashboard"
        seen.append(data)
        return "submit", {"answers": valid}

    def misroute_first(envelope, n):
        if n == 1:
            envelope["responds_to"]["decision_id"] = "dec-someone-else"
        return envelope

    result, shown = _run_with_tui(
        _one_question_graph("Review groups and QC", _CONTROLS_DASHBOARD),
        tui,
        tamper=misroute_first,
    )

    first, second = seen
    assert [r["request_version"] for r in shown] == [1, 2]
    assert [r["prefilled_from_answer"] for r in shown] == [False, False]
    assert second["problems"]  # the misroute is shown
    for section in second["sections"]:
        assert "answer" not in section  # proposals are not the user's answer
    groups = {s["key"]: s for s in second["sections"]}["groups"]
    assert groups["data"] == {s["key"]: s for s in first["sections"]}["groups"]["data"]
    assert (result["action"], result["attempts"]) == ("submit", 2)
    assert result["answer"]["groups"]["S03"] == {"id": "treated"}


def test_full_path_older_tui_text_answer_is_unchecked_free_text():
    """An older TUI shows the fallback text box; the text reaches the
    supervisor as free_text, marked unchecked, without a re-ask."""

    def old_tui(component, data):
        return "submit", {"answer": "S01 and S02 stay; S03, S04 treated"}

    result, shown = _run_with_tui(
        _one_question_graph("Review groups and QC", _CONTROLS_DASHBOARD), old_tui
    )
    assert len(shown) == 1
    assert result["action"] == "free_text"
    assert result["text"] == "S01 and S02 stay; S03, S04 treated"
    assert result["checked"] is False
    assert result["attempts"] == 1


def test_first_ask_has_no_reask_banner():
    data = dashboard_data(DecisionRequest.model_validate(_proposed_dashboard()))
    assert "problems" not in data


def test_last_attempt_says_a_refusal_closes_the_question():
    """FX w9b: the user must learn that a fifth refusal ends the question."""
    from lobster.services.interaction.component_schemas import Problem, reask

    request = DecisionRequest.model_validate(_request(SOURCE))
    for _ in range(4):
        request = reask(request, [Problem(path="source", message="no")], None)
    assert request.request_version == 5
    drawn, _ = _ask(request.to_wire(), ("submit", {"selected": [{"id": "SRC-A"}]}))
    assert (
        "This is the last attempt: if this answer is refused too, the question"
        " is closed and nothing is decided." in drawn[0][1]["details"]
    )


def test_integer_slider_step_scales_with_the_range():
    """FX w5: one key may not move 1 read in an 18,000,000-read range."""
    field = {
        "key": "floor",
        "type": "number",
        "view": "slider",
        "integer": True,
        "min": 2_000_000,
        "max": 20_000_000,
        "default": 2_000_000,
    }
    drawn, _ = _ask(_request(field), ("submit", {"value": 2_180_000}))
    assert drawn[0][1]["step"] == 180_000
    small = {**field, "min": 0, "max": 7, "default": 3}
    drawn, _ = _ask(_request(small), ("submit", {"value": 4}))
    assert drawn[0][1]["step"] == 1  # never below one whole unit


def test_typed_number_bounds_never_use_exponent_form():
    """FX w8/x3: 2e+06 must reach the screen as 2,000,000."""
    field = {"key": "floor", "type": "number", "integer": True}
    field.update(min=2_000_000, max=20_000_000, unit="reads")
    drawn, _ = _ask(_request(field), ("submit", {"answer": "2500000"}))
    assert "min 2,000,000, max 20,000,000" in drawn[0][1]["question"]
    assert "e+" not in drawn[0][1]["question"]


def test_a_reply_naming_an_unknown_option_goes_to_the_check():
    """FX w14: an option id the question doesn't offer must not crash the
    adapter (it used to raise StopIteration, an empty ✖ ERROR); it reaches the
    contract check, which names it, and ask_user asks again."""
    from lobster.services.interaction.component_schemas import AnswerRejected

    with pytest.raises(AnswerRejected, match="unknown option 'SRC-9'"):
        _ask(_request(SOURCE), ("submit", {"selected": [{"id": "SRC-9"}]}))


def _flag_plate(n=4):
    return {
        "key": "review",
        "type": "choice",
        "question": "Flag samples for review?",
        "view": "plate_map",
        "for_each": [f"S{i:02d}" for i in range(1, n + 1)],
        "options": [
            "no_flag",
            {"id": "flag", "fields": [{"key": "reason", "type": "text"}]},
        ],
    }


def test_plate_whose_option_needs_a_reason_is_drawn_as_the_list():
    """FX W1: a well can't collect a reason, so such a plate was unanswerable.
    It is drawn as the per-item list, which can; the answer is the same."""
    reply = {
        "answers": {
            "S01": {"id": "flag", "fields": {"reason": "odd profile"}},
            "S02": {"id": "no_flag"},
            "S03": {"id": "no_flag"},
            "S04": {"id": "no_flag"},
        }
    }
    drawn, result = _ask(_request(_flag_plate()), ("submit", reply))
    assert [component for component, _ in drawn] == ["choice_list"]
    assert result["answer"]["review"]["S01"] == {
        "id": "flag",
        "fields": {"reason": "odd profile"},
    }


def test_plate_without_option_fields_is_still_a_plate():
    field = {**_flag_plate(), "options": ["no_flag", "flag"]}
    drawn, _ = _ask(
        _request(field),
        ("submit", {"assignments": {f"S0{i}": "no_flag" for i in range(1, 5)}}),
    )
    assert [component for component, _ in drawn] == ["plate_map"]


def test_plate_drawn_as_the_list_is_not_held_to_the_plate_limits():
    """More than 96 samples is a plate limit, not a list limit."""
    assert unsupported(_request(_flag_plate(n=120))) == []


def test_dashboard_plate_section_with_a_reason_is_drawn_as_the_list():
    request = _request(
        {**SOURCE, "question": "Source?"},
        {**_flag_plate(), "depends_on": ["source"]},
        view="dashboard",
    )
    sections = {
        s["key"]: s
        for s in dashboard_data(DecisionRequest.model_validate(request))["sections"]
    }
    assert sections["review"]["component"] == "choice_list"


# --- v3 findings ---


def test_typed_number_may_use_thousands_commas():
    """Numbers are shown as 2,000,000, so the same form must be accepted."""
    field = {"key": "floor", "type": "number", "integer": True, "min": 0}
    _, result = _ask(_request(field), ("submit", {"answer": "2,000,000"}))
    assert result["answer"] == {"floor": 2_000_000}


@pytest.mark.parametrize(
    "typed, message",
    [
        ("1,5", "'1,5' is not a number."),
        ("nan", "'nan' is not a number."),
        ("inf", "'inf' is not a number."),
        ("1e309", "'1e309' is not a number."),
        ("123456789012345678901", "123456789012345678901 is too large."),
    ],
)
def test_typed_number_that_cannot_be_an_answer_is_asked_again(typed, message):
    """A 21-digit whole number used to pass here and crash the graph's
    checkpoint (it holds 64-bit integers at most)."""
    field = {"key": "reads", "type": "number", "integer": True}
    drawn, result = _ask(
        _request(field), ("submit", {"answer": typed}), ("submit", {"answer": "7"})
    )
    assert drawn[1][1]["question"].startswith(message + "\n")
    assert result["answer"] == {"reads": 7}


def test_typed_number_retry_keeps_the_whole_question():
    """The retry adds the reason above the question; its bounds stay."""
    field = {"key": "reads", "type": "number", "integer": True, "min": 0}
    drawn, _ = _ask(
        _request(field), ("submit", {"answer": "lots"}), ("submit", {"answer": "5"})
    )
    first, retry = drawn[0][1], drawn[1][1]
    assert retry["question"] == "'lots' is not a number.\n" + first["question"]
    assert {k: v for k, v in retry.items() if k != "question"} == {
        k: v for k, v in first.items() if k != "question"
    }


def test_plate_with_an_unavailable_option_is_drawn_as_the_list():
    """The plate's legend has no place for an option shown greyed with its
    reason, so it would vanish; the list shows it."""
    field = {
        **_flag_plate(),
        "options": ["no_flag", "flag", {"id": "drop", "unavailable": "locked"}],
    }
    answers = {f"S0{i}": {"id": "no_flag"} for i in range(1, 5)}
    drawn, _ = _ask(_request(field), ("submit", {"answers": answers}))
    assert [component for component, _ in drawn] == ["choice_list"]
    assert drawn[0][1]["options"][2]["unavailable"] == "locked"


def test_problem_for_an_item_id_with_a_dot_names_the_whole_id():
    """ENSG00000141510.17 must not be split into "ENSG00000141510 17"."""
    field = {
        "key": "keep",
        "type": "choice",
        "for_each": ["ENSG00000141510.17", "ENSG00000141510"],
        "options": [
            "yes",
            {"id": "no", "fields": [{"key": "reason", "type": "text"}]},
        ],
    }
    request = _reasked(
        _request(field), [("keep.ENSG00000141510.17.fields", "needs its fields")]
    )
    drawn, _ = _ask(
        request,
        (
            "submit",
            {
                "answers": {
                    "ENSG00000141510.17": {"id": "yes"},
                    "ENSG00000141510": {"id": "yes"},
                }
            },
        ),
    )
    assert "⚠ ENSG00000141510.17: needs its fields" in drawn[0][1]["details"]


def test_fallback_prompt_says_it_is_asked_again():
    """The older TUI's text prompt carries the same re-ask line as the
    components, so a re-ask doesn't look like the first question."""
    request = _reasked(_request(SOURCE), [("source", "pick one")])
    assert fallback_prompt(request).splitlines()[:3] == [
        "Review",
        "Asked again (attempt 2 of 5): fix what is marked.",
        "⚠ pick one",
    ]


def test_follow_up_names_the_item_and_option_and_keeps_an_earlier_answer():
    """A sub-field on its own screen used to read just "reason", with no
    sample named, and a re-ask lost what the user had typed."""
    field = {
        "key": "qc",
        "type": "choice",
        "question": "Keep each sample?",
        "for_each": [{"id": "S02", "label": "Donor B"}, "S03"],
        "options": [
            "retain",
            {
                "id": "exclude",
                "label": "Exclude",
                "fields": [
                    {"key": "reason", "type": "text", "question": "Why?"},
                    {"key": "by", "type": "text", "question": "Who decided?"},
                ],
            },
        ],
    }
    kept = {"id": "exclude", "fields": {"reason": "low reads", "by": "Ana"}}
    request = _reasked(_request(field), [("qc.S03", "no answer")], qc={"S02": kept})
    drawn, result = _ask(
        request,
        ("submit", {"selected": [{"id": "exclude"}]}),
        ("submit", {"answer": ""}),
        ("submit", {"answer": ""}),
        ("submit", {"selected": [{"id": "retain"}]}),
    )
    reason, by = drawn[1][1], drawn[2][1]
    assert reason["question"] == "Donor B (S02), Exclude: Why?"
    assert reason["placeholder"] == "Enter = keep low reads"
    assert by["question"] == "Donor B (S02), Exclude: Who decided?"
    assert result["answer"] == {"qc": {"S02": kept, "S03": {"id": "retain"}}}


def test_record_sub_fields_name_the_record():
    field = {
        "key": "mito",
        "type": "record",
        "question": "Mito fraction range",
        "fields": [
            {"key": "from", "type": "number", "question": "From?"},
            {"key": "to", "type": "number", "question": "To?"},
        ],
    }
    drawn, _ = _ask(
        _request(field), ("submit", {"answer": "0"}), ("submit", {"answer": "0.2"})
    )
    assert [d["question"].splitlines()[0] for _, d in drawn] == [
        "Mito fraction range: From?",
        "Mito fraction range: To?",
    ]
