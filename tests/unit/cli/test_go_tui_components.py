"""Tests for the Go TUI adapter: decision fields <-> Go components.

A fake render plays the Go TUI: it records what would be drawn and replies
with each component's own reply shape.
"""

import pytest

from lobster.cli_internal.go_tui_components import (
    build_answer,
    collect,
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
        "for_each": [{"id": "S02", "label": "Donor B", "details": "Low library size"}],
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
    assert drawn[0][1]["question"] == "Donor B (S02): Review"
    assert drawn[0][1]["details"] == "Low library size"
    assert result["answer"] == {
        "qc": {"S02": {"id": "exclude", "fields": {"reason": "library too small"}}}
    }


def test_for_each_asks_once_per_available_item():
    field = {
        "key": "cond",
        "type": "choice",
        "options": ["control", "treated"],
        "for_each": ["S01", {"id": "S02", "unavailable": "excluded"}, "S03"],
    }
    drawn, result = _ask(
        _request(field),
        ("submit", {"selected": [{"id": "control"}]}),
        ("submit", {"selected": [{"id": "treated"}]}),
    )
    assert [d["question"] for _, d in drawn] == ["S01: Review", "S03: Review"]
    assert result["answer"] == {
        "cond": {"S01": {"id": "control"}, "S03": {"id": "treated"}}
    }


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


# --- stopping, envelope, limits ---


@pytest.mark.parametrize("action", ["cancel", "defer"])
def test_stopping_part_way_returns_no_answer(action):
    field = {
        "key": "cond",
        "type": "choice",
        "options": ["a", "b"],
        "for_each": ["S1", "S2"],
    }
    tui = FakeTUI(("submit", {"selected": [{"id": "a"}]}), (action, {}))
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

