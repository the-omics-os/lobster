"""Tests for the HITL decision contract (envelope, fields, answer checks)."""

import pytest
from pydantic import ValidationError

from lobster.services.interaction.component_schemas import (
    SCHEMA_VERSION,
    AnswerRejected,
    AnswerStale,
    AskUserContext,
    DecisionAnswer,
    DecisionRequest,
    check_answer,
)

SOURCE = {
    "key": "source",
    "type": "choice",
    "question": "Which count source?",
    "options": [
        {"id": "SRC-A", "label": "Author matrix", "details": "Raw counts · Build-A"},
        "SRC-B",
        {"id": "SRC-C", "unavailable": "Normalized; reference build unknown"},
    ],
}
QC = {
    "key": "qc",
    "type": "choice",
    "question": "Review flagged samples",
    "for_each": [{"id": "S02", "label": "Donor B"}, {"id": "S06", "label": "Donor F"}],
    "options": [
        {"id": "retain"},
        {"id": "exclude", "fields": [{"key": "reason", "type": "text"}]},
        {"id": "defer"},
    ],
}
MITO = {
    "key": "mito",
    "type": "record",
    "question": "Mito fraction range",
    "profile": "interval",
    "fields": [
        {"key": "from", "type": "number", "min": 0, "max": 1},
        {"key": "to", "type": "number", "min": 0, "max": 1},
    ],
}
GROUPS = {
    "key": "treated",
    "type": "choice",
    "question": "Treated samples",
    "count": {"min": 1},
    "options": ["S05", "S06", "S07", "S08"],
}


def _request(*fields, **envelope):
    context = AskUserContext.model_validate({"fields": list(fields)})
    return DecisionRequest(question="Review", fields=context.fields, **envelope)


def _answer(request, answer=None, action="submit", **envelope):
    return DecisionAnswer(
        responds_to={
            "decision_id": request.decision_id,
            "request_version": request.request_version,
        },
        action=action,
        answer=answer or {},
        **envelope,
    )


# --- Items ---


def test_plain_string_item_is_id_and_label():
    field = AskUserContext.model_validate({"fields": [SOURCE]}).fields[0]
    assert (field.options[1].id, field.options[1].label) == ("SRC-B", "SRC-B")


def test_repeated_labels_keep_distinct_ids():
    field = {
        "key": "donor",
        "type": "choice",
        "options": [
            {"id": "S01", "label": "Donor A"},
            {"id": "S05", "label": "Donor A"},
        ],
    }
    request = _request(field)
    result = check_answer(request, _answer(request, {"donor": {"id": "S05"}}))
    assert result["answer"] == {"donor": {"id": "S05"}}


# --- Request validation ---


@pytest.mark.parametrize(
    "context, message",
    [
        ({"fields": []}, "at least 1 item"),
        ({"fields": [{"key": "a", "type": "text"}] * 2}, "duplicate keys"),
        (
            {
                "fields": [
                    {"key": "a", "type": "text", "question": "A"},
                    {"key": "b", "type": "text"},
                ]
            },
            "need a question",
        ),
        (
            {"fields": [{"key": "a", "type": "choice", "options": ["x", "x"]}]},
            "duplicate ids",
        ),
        (
            {
                "fields": [
                    {
                        "key": "a",
                        "type": "choice",
                        "options": [{"id": "x", "unavailable": "no"}],
                    }
                ]
            },
            "every option is unavailable",
        ),
        (
            {
                "fields": [
                    {
                        "key": "a",
                        "type": "choice",
                        "options": ["x"],
                        "count": {"min": 2},
                    }
                ]
            },
            "more picks than",
        ),
        (
            {"fields": [{"key": "a", "type": "number", "min": 2, "max": 1}]},
            "larger than max",
        ),
        (
            {"fields": [{"key": "a", "type": "number", "view": "slider", "min": 0}]},
            "needs max",
        ),
        (
            {"fields": [{"key": "a", "type": "text", "view": "slider"}]},
            "applies to number",
        ),
        ({"fields": [{"key": "a", "type": "text", "view": "nope"}]}, "unknown view"),
        (
            {"fields": [{"key": "a", "type": "text", "profile": "nope"}]},
            "unknown profile",
        ),
        (
            {
                "fields": [
                    {
                        "key": "a",
                        "type": "choice",
                        "options": ["x"],
                        "default": {"id": "y"},
                    }
                ]
            },
            "default is not a valid answer",
        ),
        (
            {"fields": [{"key": "a", "type": "number", "max": 1, "default": 5}]},
            "above max",
        ),
        (
            {"fields": [{"key": "a", "type": "text", "depends_on": ["zz"]}]},
            "not a sibling field",
        ),
        (
            {
                "fields": [
                    {"key": "a", "type": "text", "question": "A", "depends_on": ["b"]},
                    {"key": "b", "type": "text", "question": "B", "depends_on": ["a"]},
                ]
            },
            "cycle",
        ),
        ({"fields": [{"key": "a", "type": "text", "colour": "red"}]}, "Extra inputs"),
        (
            {"view": "slider", "fields": [{"key": "a", "type": "text"}]},
            "applies to number",
        ),
    ],
)
def test_invalid_requests_are_refused(context, message):
    with pytest.raises(ValidationError, match=message):
        AskUserContext.model_validate(context)


def test_dashboard_view_applies_to_the_whole_question():
    context = AskUserContext.model_validate(
        {"view": "dashboard", "fields": [SOURCE, {**GROUPS, "depends_on": ["source"]}]}
    )
    assert context.view == "dashboard"
    assert context.fields[1].depends_on == ["source"]


def test_dashboard_refuses_sections_it_cannot_draw():
    with pytest.raises(ValidationError, match="dashboard section .qc."):
        AskUserContext.model_validate({"view": "dashboard", "fields": [SOURCE, QC]})


def test_envelope_wire_format():
    request = _request(SOURCE)
    wire = request.to_wire()
    assert wire["schema"] == SCHEMA_VERSION
    assert wire["decision_id"].startswith("dec-")
    assert (wire["request_version"], wire["actions"]) == (
        1,
        ["submit", "defer", "cancel"],
    )
    assert "details" not in wire["fields"][0]  # None values are left out
    assert DecisionRequest.model_validate(wire) == request  # round trip


def test_lone_field_inherits_the_main_question():
    request = _request({"key": "a", "type": "text"})
    assert request.fields[0].question == "Review"


# --- Answer checks ---


def test_valid_answer_is_cleaned():
    request = _request(SOURCE, QC, MITO, GROUPS)
    answer = {
        "source": {"id": "SRC-A"},
        "qc": {
            "S02": {"id": "exclude", "fields": {"reason": "  low library size "}},
            "S06": {"id": "retain"},
        },
        "mito": {"from": 0.0, "to": 0.2},
        "treated": [{"id": "S05"}, {"id": "S07"}],
    }
    result = check_answer(request, _answer(request, answer, comment="ok"))
    assert result["action"] == "submit"
    assert result["comment"] == "ok"
    assert result["answer"]["qc"]["S02"]["fields"]["reason"] == "low library size"


def test_every_problem_is_reported_with_its_path():
    request = _request(SOURCE, QC, MITO, GROUPS)
    answer = {
        "source": {"id": "SRC-C"},
        "qc": {"S02": {"id": "exclude"}},
        "mito": {"from": 0.5, "to": 0.2},
        "treated": [{"id": "S05"}, {"id": "S05"}],
        "extra": 1,
    }
    with pytest.raises(AnswerRejected) as caught:
        check_answer(request, _answer(request, answer))
    assert caught.value.problems == [
        "extra: not a field in this question",
        "source: option 'SRC-C' is unavailable (Normalized; reference build unknown)",
        "qc.S06: no answer",
        "qc.S02.fields: option 'exclude' needs its fields",
        "mito: from (0.5) is larger than to (0.2)",
        "treated: the same option was picked twice",
    ]


@pytest.mark.parametrize(
    "field, value, problem",
    [
        (
            {"key": "n", "type": "number", "integer": True},
            2.5,
            "must be a whole number",
        ),
        ({"key": "n", "type": "number"}, True, "expected a number"),
        ({"key": "n", "type": "number"}, "3", "expected a number"),
        ({"key": "t", "type": "text"}, "   ", "expected non-empty text"),
        ({"key": "c", "type": "choice", "options": ["a"]}, "a", 'expected {"id": ...}'),
        (
            {"key": "c", "type": "choice", "options": ["a"]},
            {"id": "a", "fields": {"x": 1}},
            "has no fields",
        ),
        (
            {**GROUPS, "count": {"min": 1, "max": 2}},
            [{"id": "S05"}] * 3,
            "give between 1 and 2",
        ),
        (GROUPS, {"id": "S05"}, "expected a list"),
    ],
)
def test_bad_values_are_rejected(field, value, problem):
    request = _request(field)
    with pytest.raises(AnswerRejected, match=problem):
        check_answer(request, _answer(request, {field["key"]: value}))


def test_integer_answers_come_back_as_int():
    request = _request({"key": "n", "type": "number", "integer": True})
    assert check_answer(request, _answer(request, {"n": 3.0}))["answer"] == {"n": 3}


def test_optional_field_may_be_left_out():
    optional = {
        "key": "note",
        "type": "text",
        "count": {"min": 0, "max": 1},
        "question": "Note",
    }
    request = _request(SOURCE, optional)
    result = check_answer(request, _answer(request, {"source": {"id": "SRC-A"}}))
    assert result["answer"] == {"source": {"id": "SRC-A"}}


def test_unavailable_rows_are_not_answered():
    field = {
        **QC,
        "for_each": ["S02", {"id": "S06", "unavailable": "already excluded"}],
    }
    request = _request(field)
    with pytest.raises(AnswerRejected, match="S06: not an item here"):
        check_answer(
            request,
            _answer(
                request, {"qc": {"S02": {"id": "retain"}, "S06": {"id": "retain"}}}
            ),
        )


# --- Envelope checks ---


@pytest.mark.parametrize("action", ["defer", "cancel"])
def test_defer_and_cancel_carry_no_answer(action):
    request = _request(SOURCE)
    result = check_answer(request, _answer(request, action=action))
    assert result["action"] == action and "answer" not in result


def test_action_must_be_allowed():
    request = _request(SOURCE, actions=["submit"])
    with pytest.raises(AnswerRejected, match="'defer' is not allowed"):
        check_answer(request, _answer(request, action="defer"))


def test_answer_for_another_decision_is_rejected():
    request, other = _request(SOURCE), _request(SOURCE)
    with pytest.raises(AnswerRejected, match="answer is for"):
        check_answer(request, _answer(other, {"source": {"id": "SRC-A"}}))


def test_answer_to_an_old_question_version_is_stale():
    request = _request(SOURCE, request_version=2)
    old = DecisionAnswer(
        responds_to={"decision_id": request.decision_id, "request_version": 1},
        action="submit",
        answer={"source": {"id": "SRC-A"}},
    )
    with pytest.raises(AnswerStale, match="question version 1"):
        check_answer(request, old)


def test_for_each_default_may_prefill_only_some_rows():
    field = {
        "key": "cond",
        "type": "choice",
        "options": ["control", "treated"],
        "for_each": ["S01", "S08"],
        "default": {"S01": {"id": "control"}},
    }
    assert AskUserContext.model_validate({"fields": [field]}).fields[0].default
    with pytest.raises(ValidationError, match="S09: not an item here"):
        AskUserContext.model_validate(
            {"fields": [{**field, "default": {"S09": {"id": "control"}}}]}
        )
    request = _request(field)
    with pytest.raises(AnswerRejected, match="cond.S08: no answer"):
        check_answer(request, _answer(request, {"cond": {"S01": {"id": "control"}}}))
