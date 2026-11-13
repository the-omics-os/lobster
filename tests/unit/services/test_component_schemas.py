"""Tests for the HITL decision contract (envelope, fields, answer checks)."""

import pytest
from pydantic import ValidationError

from lobster.services.interaction import component_schemas
from lobster.services.interaction.component_schemas import (
    CHECKS,
    SCHEMA_VERSION,
    AnswerMisrouted,
    AnswerRejected,
    AnswerStale,
    AskUserContext,
    DecisionAnswer,
    DecisionRequest,
    FileSnapshot,
    Problem,
    check_answer,
    check_request,
    fingerprint,
    reask,
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
    with pytest.raises(ValidationError, match="dashboard section .mito."):
        AskUserContext.model_validate({"view": "dashboard", "fields": [SOURCE, MITO]})


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
        Problem(path="extra", message="not a field in this question"),
        Problem(
            path="source",
            message="option 'SRC-C' is unavailable "
            "(Normalized; reference build unknown)",
        ),
        Problem(path="qc.S06", message="no answer"),
        Problem(path="qc.S02.fields", message="option 'exclude' needs its fields"),
        Problem(path="mito", message="from (0.5) is larger than to (0.2)"),
        Problem(path="treated", message="the same option was picked twice"),
    ]
    assert str(caught.value).startswith(
        "extra: not a field in this question; source: option 'SRC-C'"
    )


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


def test_answer_for_another_decision_is_misrouted():
    request, other = _request(SOURCE), _request(SOURCE)
    with pytest.raises(AnswerMisrouted, match="answer is for decision") as caught:
        check_answer(request, _answer(other, {"source": {"id": "SRC-A"}}))
    assert isinstance(caught.value, AnswerRejected)
    assert caught.value.problems[0].path == "responds_to.decision_id"


def test_answer_to_an_old_question_version_is_stale():
    request = _request(SOURCE, request_version=2)
    old = DecisionAnswer(
        responds_to={"decision_id": request.decision_id, "request_version": 1},
        action="submit",
        answer={"source": {"id": "SRC-A"}},
    )
    with pytest.raises(AnswerStale, match="version 1 of the question") as caught:
        check_answer(request, old)
    assert caught.value.problems[0].path == "responds_to.request_version"


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


# --- U-17: a choice default may name an option by its bare id ---


def _default(field):
    return AskUserContext.model_validate({"fields": [field]}).fields[0].default


def test_bare_id_default_is_read_as_the_option():
    assert _default({**SOURCE, "default": "SRC-A"}) == {"id": "SRC-A"}


def test_bare_id_default_per_item_may_mix_forms():
    field = {**QC, "default": {"S02": "retain", "S06": {"id": "defer"}}}
    assert _default(field) == {"S02": {"id": "retain"}, "S06": {"id": "defer"}}


def test_bare_id_default_with_count_is_a_list_of_ids():
    assert _default({**GROUPS, "default": ["S05", "S07"]}) == [
        {"id": "S05"},
        {"id": "S07"},
    ]
    per_item = {
        "key": "marks",
        "type": "choice",
        "for_each": ["S01", "S02"],
        "count": {"min": 0},
        "options": ["a", "b", "c"],
        "default": {"S01": ["a", {"id": "c"}], "S02": []},
    }
    assert _default(per_item) == {"S01": [{"id": "a"}, {"id": "c"}], "S02": []}


@pytest.mark.parametrize(
    "field, message",
    [
        ({**SOURCE, "default": "SRC-Z"}, 'source.default: expected {"id": ...}'),
        ({**GROUPS, "default": ["S05", "S99"]}, r"treated.default\[1\]: expected"),
        ({**QC, "default": {"S02": "keep"}}, 'qc.default.S02: expected {"id": ...}'),
        ({**SOURCE, "default": "SRC-C"}, "option 'SRC-C' is unavailable"),
        ({**QC, "default": {"S02": "exclude"}}, "option 'exclude' needs its fields"),
    ],
)
def test_bare_id_default_still_fails_for_unknown_or_unusable_options(field, message):
    with pytest.raises(ValidationError, match=message):
        AskUserContext.model_validate({"fields": [field]})


def test_bare_id_is_only_for_defaults_never_for_answers():
    request = _request(SOURCE)
    with pytest.raises(AnswerRejected, match='source: expected {"id": ...}'):
        check_answer(request, _answer(request, {"source": "SRC-A"}))
    qc = _request(QC)
    with pytest.raises(AnswerRejected, match='qc.S02: expected {"id": ...}'):
        check_answer(qc, _answer(qc, {"qc": {"S02": "retain", "S06": "retain"}}))


def test_bare_id_default_goes_on_the_wire_in_full():
    request = _request(
        {**SOURCE, "default": "SRC-A"}, {**QC, "default": {"S02": "defer"}}
    )
    defaults = [field["default"] for field in request.to_wire()["fields"]]
    assert defaults == [{"id": "SRC-A"}, {"S02": {"id": "defer"}}]


def test_bare_id_does_not_change_the_callers_context():
    field = {**SOURCE, "default": "SRC-A"}
    AskUserContext.model_validate({"fields": [field]})
    assert field["default"] == "SRC-A"


# --- Removed from the MVP: revise, Workspace, Snapshot ---


def test_revise_is_not_an_action():
    request = _request(SOURCE)
    with pytest.raises(ValidationError, match="action"):
        DecisionAnswer.model_validate(
            {
                "responds_to": {
                    "decision_id": request.decision_id,
                    "request_version": 1,
                },
                "action": "revise",
            }
        )
    with pytest.raises(ValidationError):
        _request(SOURCE, actions=["submit", "revise"])


def test_workspace_and_mvp_snapshot_are_gone():
    assert not hasattr(component_schemas, "Workspace")
    assert not hasattr(component_schemas, "Snapshot")
    with pytest.raises(ValidationError, match="Extra inputs"):
        _request(SOURCE, workspace={"id": "w", "type": "rnaseq", "stage": "qc"})
    with pytest.raises(ValidationError, match="Extra inputs"):
        DecisionAnswer.model_validate(
            {
                "responds_to": {"decision_id": "dec-x", "request_version": 1},
                "action": "submit",
                "reviewed_snapshot": "snap-1",
            }
        )


# --- Problems ---


def test_problem_is_a_path_and_a_message():
    problem = Problem(path="qc.S02", message="no answer")
    assert problem.model_dump() == {"path": "qc.S02", "message": "no answer"}
    assert str(problem) == "qc.S02: no answer"
    error = AnswerRejected([problem, Problem(path="groups", message="empty")])
    assert str(error) == "qc.S02: no answer; groups: empty"


def test_invalid_default_names_the_path_of_each_problem():
    field = {"key": "a", "type": "choice", "options": ["x"], "default": {"id": "y"}}
    with pytest.raises(ValidationError, match="a.default: unknown option 'y'"):
        AskUserContext.model_validate({"fields": [field]})


def test_request_carries_checks_problems_and_snapshot_on_the_wire():
    request = _request(SOURCE)
    wire = request.to_wire()
    assert wire["checks"] == [] and wire["problems"] == []
    assert "snapshot" not in wire
    again = reask(request, [Problem(path="source", message="pick one")], None)
    assert again.to_wire()["problems"] == [{"path": "source", "message": "pick one"}]
    assert DecisionRequest.model_validate(again.to_wire()) == again


# --- CHECKS: registry and spec validation ---

GROUPS_8 = {
    "key": "groups",
    "type": "choice",
    "question": "Group of each sample?",
    "for_each": [f"S0{i}" for i in range(1, 9)],
    "options": ["control", "treated", {"id": "leave_out", "label": "Leave out"}],
}
QC_REVIEW = {
    "key": "qc",
    "type": "choice",
    "question": "Review flagged samples",
    "for_each": ["S01", "S02", "S06"],
    "options": [
        "retain",
        {"id": "exclude", "fields": [{"key": "reason", "type": "text"}]},
        "defer",
    ],
}
KEPT = {
    "check": "each_option_kept",
    "field": "groups",
    "options": ["control", "treated"],
    "minus": {"field": "qc", "options": ["exclude", "defer"]},
}


def _context(*fields, checks=(), **extra):
    return AskUserContext.model_validate(
        {"fields": list(fields), "checks": list(checks), **extra}
    )


def _checked_request(*fields, checks=()):
    context = _context(*fields, checks=checks)
    return DecisionRequest(
        question="Review", fields=context.fields, checks=context.checks
    )


def test_checks_registry_holds_exactly_the_v1_checks():
    assert sorted(CHECKS) == ["each_option_kept", "no_default_for", "reason_required"]
    assert {name: entry["on"] for name, entry in CHECKS.items()} == {
        "each_option_kept": "answer",
        "reason_required": "request",
        "no_default_for": "request",
    }
    for entry in CHECKS.values():
        assert entry["use_when"] and "\n" not in entry["use_when"]
        assert callable(entry["validate"]) and callable(entry["run"])


@pytest.mark.parametrize(
    "spec, message",
    [
        ({"field": "groups"}, 'needs a "check"'),
        ({"check": "every_group_balanced", "field": "groups"}, "unknown check"),
        ({**KEPT, "field": "nope"}, "no field 'nope'"),
        ({**KEPT, "options": ["control", "placebo"]}, "unknown option 'placebo'"),
        ({**KEPT, "options": []}, "at least one option"),
        ({**KEPT, "colour": "red"}, "unknown parameter"),
        ({**KEPT, "minus": {"field": "qc"}}, "minus needs"),
        (
            {**KEPT, "minus": {"field": "qc", "options": ["drop"]}},
            "unknown option 'drop'",
        ),
        ({**KEPT, "minus": {**KEPT["minus"], "extra": 1}}, "unknown parameter"),
        ({**KEPT, "minus": {"field": "groups", "options": ["leave_out"]}}, "another"),
        ({**KEPT, "field": "note"}, "per-item choice"),
        ({**KEPT, "field": "source"}, "per-item choice"),
        ({"check": "reason_required", "field": "qc", "options": ["skip"]}, "'skip'"),
        ({"check": "reason_required", "field": "note", "options": ["x"]}, "choice"),
        ({"check": "reason_required", "field": "qc"}, "needs options"),
        ({"check": "no_default_for", "field": "qc", "items": ["S99"]}, "'S99'"),
        ({"check": "no_default_for", "field": "source", "items": ["S01"]}, "for_each"),
        ({"check": "no_default_for", "field": "note", "options": ["x"]}, "choice"),
        ({"check": "no_default_for", "field": "qc", "items": "S01"}, "list"),
    ],
)
def test_invalid_check_specs_are_refused(spec, message):
    note = {"key": "note", "type": "text", "question": "Note"}
    with pytest.raises(ValidationError, match=message):
        _context(GROUPS_8, QC_REVIEW, {**SOURCE}, note, checks=[spec])


def test_check_specs_are_validated_on_the_request_too():
    context = _context(GROUPS_8, QC_REVIEW)
    with pytest.raises(ValidationError, match="no field 'nope'"):
        DecisionRequest(
            question="Review",
            fields=context.fields,
            checks=[{**KEPT, "field": "nope"}],
        )


# --- each_option_kept ---


def _groups(control, treated, leave_out=()):
    answer = {s: {"id": "control"} for s in control}
    answer.update({s: {"id": "treated"} for s in treated})
    answer.update({s: {"id": "leave_out"} for s in leave_out})
    return answer


def test_each_option_kept_rejects_an_empty_group():
    request = _checked_request(GROUPS_8, checks=[{**KEPT, "minus": None}])
    answer = {"groups": _groups([], [f"S0{i}" for i in range(1, 9)])}
    with pytest.raises(AnswerRejected) as caught:
        check_answer(request, _answer(request, answer))
    assert [(p.path, p.message) for p in caught.value.problems] == [
        ("groups", "'control' has no items; give it at least one")
    ]


def test_each_option_kept_rejects_a_group_emptied_by_qc_with_one_problem_per_field():
    request = _checked_request(GROUPS_8, QC_REVIEW, checks=[KEPT])
    answer = {
        "groups": _groups(["S01", "S02"], ["S03", "S04", "S05", "S06", "S07", "S08"]),
        "qc": {
            "S01": {"id": "exclude", "fields": {"reason": "low library size"}},
            "S02": {"id": "exclude", "fields": {"reason": "degraded"}},
            "S06": {"id": "retain"},
        },
    }
    with pytest.raises(AnswerRejected) as caught:
        check_answer(request, _answer(request, answer))
    message = "'control' has no items left after removing S01, S02 (qc: exclude)"
    assert [(p.path, p.message) for p in caught.value.problems] == [
        ("groups", message),
        ("qc", message),
    ]


def test_each_option_kept_passes_when_every_group_keeps_an_item():
    request = _checked_request(GROUPS_8, QC_REVIEW, checks=[KEPT])
    answer = {
        "groups": _groups(["S01", "S02", "S03"], ["S04", "S05", "S06", "S07", "S08"]),
        "qc": {
            "S01": {"id": "exclude", "fields": {"reason": "low library size"}},
            "S02": {"id": "retain"},
            "S06": {"id": "retain"},
        },
    }
    assert check_answer(request, _answer(request, answer))["action"] == "submit"


def test_items_picked_with_options_outside_the_check_do_not_count():
    """A left-out sample keeps no group alive."""
    request = _checked_request(GROUPS_8, checks=[{**KEPT, "minus": None}])
    answer = {
        "groups": _groups(
            [], ["S02", "S03"], leave_out=["S01", *"S04 S05 S06 S07 S08".split()]
        )
    }
    with pytest.raises(AnswerRejected, match="'control' has no items"):
        check_answer(request, _answer(request, answer))


def test_deferring_the_last_control_is_refused():
    """U-13: a deferred sample does not count toward its group."""
    request = _checked_request(GROUPS_8, QC_REVIEW, checks=[KEPT])
    answer = {
        "groups": _groups(["S01", "S02"], ["S03", "S04", "S05", "S06", "S07", "S08"]),
        "qc": {
            "S01": {"id": "exclude", "fields": {"reason": "low library size"}},
            "S02": {"id": "defer"},
            "S06": {"id": "retain"},
        },
    }
    with pytest.raises(AnswerRejected) as caught:
        check_answer(request, _answer(request, answer))
    assert caught.value.problems[0].message == (
        "'control' has no items left after removing S01, S02 (qc: exclude, defer)"
    )


def test_answer_checks_run_alongside_shape_problems():
    request = _checked_request(GROUPS_8, QC_REVIEW, checks=[{**KEPT, "minus": None}])
    answer = {
        "groups": _groups([], [f"S0{i}" for i in range(1, 9)]),
        "qc": {
            "S01": {"id": "exclude"},
            "S02": {"id": "retain"},
            "S06": {"id": "retain"},
        },
    }
    with pytest.raises(AnswerRejected) as caught:
        check_answer(request, _answer(request, answer))
    assert [p.path for p in caught.value.problems] == ["qc.S01.fields", "groups"]


def test_answer_checks_skip_a_field_whose_shape_failed():
    request = _checked_request(GROUPS_8, checks=[{**KEPT, "minus": None}])
    with pytest.raises(AnswerRejected) as caught:
        check_answer(request, _answer(request, {"groups": "everything in control"}))
    assert [p.path for p in caught.value.problems] == ["groups"]
    assert "has no items" not in str(caught.value)


# --- Request-time checks: reason_required, no_default_for ---


def test_reason_required_needs_a_required_text_field_on_each_option():
    spec = {"check": "reason_required", "field": "qc", "options": ["exclude"]}
    assert check_request(_context(QC_REVIEW, checks=[spec])) == []
    no_reason = {**QC_REVIEW, "options": ["retain", "exclude", "defer"]}
    optional = {
        **QC_REVIEW,
        "options": [
            "retain",
            {
                "id": "exclude",
                "fields": [{"key": "reason", "type": "text", "count": {"min": 0}}],
            },
        ],
    }
    for qc in (no_reason, optional):
        # The removal rule also names 'exclude'; this test is about the check.
        (problem,) = _without_removal_rule(check_request(_context(qc, checks=[spec])))
        assert problem.path == "qc"
        assert "'exclude' needs a required text field" in problem.message


def test_no_default_for_refuses_prefilled_items():
    field = {
        **GROUPS_8,
        "default": {"S01": {"id": "control"}, "S08": {"id": "treated"}},
    }
    spec = {"check": "no_default_for", "field": "groups", "items": ["S08"]}
    (problem,) = _without_removal_rule(check_request(_context(field, checks=[spec])))
    assert problem.path == "groups"
    assert "pre-fills S08" in problem.message
    allowed = {**GROUPS_8, "default": {"S01": {"id": "control"}}}
    assert _without_removal_rule(check_request(_context(allowed, checks=[spec]))) == []


def test_no_default_for_refuses_a_listed_option_for_listed_items():
    """U-13: a warned sample may not be pre-filled as exclude."""
    spec = {
        "check": "no_default_for",
        "field": "qc",
        "items": ["S02"],
        "options": ["exclude"],
    }
    excluded = {
        **QC_REVIEW,
        "default": {"S02": {"id": "exclude", "fields": {"reason": "low counts"}}},
    }
    (problem,) = check_request(_context(excluded, checks=[spec]))
    assert "pre-fills S02 with 'exclude'" in problem.message
    retained = {**QC_REVIEW, "default": {"S02": {"id": "retain"}}}
    assert check_request(_context(retained, checks=[spec])) == []
    any_item = {"check": "no_default_for", "field": "qc", "options": ["exclude"]}
    assert check_request(_context(excluded, checks=[any_item]))


def test_no_default_for_without_items_or_options_means_no_default():
    spec = {"check": "no_default_for", "field": "source"}
    with_default = {**SOURCE, "default": {"id": "SRC-A"}}
    assert check_request(_context(with_default, checks=[spec]))
    assert check_request(_context(SOURCE, checks=[spec])) == []


def test_no_default_for_sees_a_bare_id_default():
    """U-17: the check runs on the normalised default."""
    items = {"check": "no_default_for", "field": "groups", "items": ["S08"]}
    (problem,) = _without_removal_rule(
        check_request(
            _context({**GROUPS_8, "default": {"S08": "treated"}}, checks=[items])
        )
    )
    assert "pre-fills S08" in problem.message
    options = {"check": "no_default_for", "field": "groups", "options": ["leave_out"]}
    (problem,) = _without_removal_rule(
        check_request(
            _context({**GROUPS_8, "default": {"S03": "leave_out"}}, checks=[options])
        )
    )
    assert "pre-fills S03 with 'Leave out'" in problem.message
    single = {"check": "no_default_for", "field": "source", "options": ["SRC-A"]}
    (problem,) = check_request(
        _context({**SOURCE, "default": "SRC-A"}, checks=[single])
    )
    assert "pre-selects 'Author matrix' (SRC-A)" in problem.message


def test_request_checks_never_run_inside_model_validators():
    """A14: building the context or the request does not run request-time checks."""
    field = {**GROUPS_8, "default": {"S08": {"id": "treated"}}}
    spec = {"check": "no_default_for", "field": "groups", "items": ["S08"]}
    context = _context(field, checks=[spec])  # no ValidationError
    request = DecisionRequest(
        question="Groups", fields=context.fields, checks=context.checks
    )
    assert DecisionRequest.model_validate(request.to_wire()).checks == [spec]
    assert check_request(context)


# --- Dashboard: per-item list sections ---


def test_dashboard_allows_a_per_item_list_with_reason_fields():
    context = AskUserContext.model_validate(
        {"view": "dashboard", "fields": [SOURCE, QC_REVIEW]}
    )
    assert context.fields[1].view == "default"


@pytest.mark.parametrize(
    "options, message",
    [
        (
            [
                "retain",
                {
                    "id": "exclude",
                    "fields": [
                        {"key": "reason", "type": "text"},
                        {"key": "note", "type": "text"},
                    ],
                },
            ],
            "one required text field",
        ),
        (
            ["retain", {"id": "exclude", "fields": [{"key": "n", "type": "number"}]}],
            "one required text field",
        ),
        (
            [
                "retain",
                {
                    "id": "exclude",
                    "fields": [{"key": "reason", "type": "text", "count": {"min": 0}}],
                },
            ],
            "one required text field",
        ),
    ],
)
def test_dashboard_list_section_refuses_other_option_fields(options, message):
    qc = {**QC_REVIEW, "options": options}
    with pytest.raises(ValidationError, match=message):
        AskUserContext.model_validate({"view": "dashboard", "fields": [SOURCE, qc]})


@pytest.mark.parametrize("view", [None, "dashboard"])
def test_plate_map_refuses_a_count(view):
    """Finding 4a: a well holds one option, so the answer could never pass."""
    plate = {**GROUPS_8, "view": "plate_map", "count": {"min": 1, "max": 2}}
    context = {"fields": [plate] if view is None else [SOURCE, plate]}
    if view:
        context["view"] = view
    with pytest.raises(ValidationError, match="'plate_map' takes one option"):
        AskUserContext.model_validate(context)


def test_dashboard_still_refuses_counts_and_reasons_it_cannot_draw():
    counted = {**QC_REVIEW, "options": ["retain", "defer"], "count": {"min": 1}}
    with pytest.raises(ValidationError, match="one option per item"):
        AskUserContext.model_validate(
            {"view": "dashboard", "fields": [SOURCE, counted]}
        )
    # A plate whose option asks for one reason is drawn as the per-item list
    # (FX W1), so it is accepted; an option asking for more is not.
    on_plate = {**QC_REVIEW, "view": "plate_map"}
    AskUserContext.model_validate({"view": "dashboard", "fields": [SOURCE, on_plate]})
    two = [{"key": "reason", "type": "text"}, {"key": "by", "type": "text"}]
    too_much = {**on_plate, "options": ["retain", {"id": "exclude", "fields": two}]}
    with pytest.raises(ValidationError, match="only have one required text field"):
        AskUserContext.model_validate(
            {"view": "dashboard", "fields": [SOURCE, too_much]}
        )
    texts = {"key": "names", "type": "text", "question": "Names?", "for_each": ["a"]}
    with pytest.raises(ValidationError, match="for_each"):
        AskUserContext.model_validate({"view": "dashboard", "fields": [SOURCE, texts]})
    number = {"key": "n", "type": "number", "question": "N?", "count": {"min": 1}}
    with pytest.raises(ValidationError, match="count is only supported on a choice"):
        AskUserContext.model_validate({"view": "dashboard", "fields": [SOURCE, number]})


# --- reask ---


def test_reask_bumps_the_version_and_keeps_the_submitted_answer():
    request = _request(SOURCE, GROUPS_8, decision_id="dec-1")
    submitted = {
        "source": {"id": "SRC-B"},
        "groups": {
            "S01": {"id": "control"},
            "S02": {"id": "placebo"},  # bad row: dropped
            "S03": {"id": "treated"},
            "S99": {"id": "control"},  # not an item: dropped
        },
    }
    problems = [Problem(path="groups.S02", message="unknown option 'placebo'")]
    again = reask(request, problems, submitted)
    assert (again.decision_id, again.request_version) == ("dec-1", 2)
    assert again.problems == problems
    assert again.fields[0].default == {"id": "SRC-B"}
    assert again.fields[1].default == {
        "S01": {"id": "control"},
        "S03": {"id": "treated"},
    }
    assert request.fields[1].default is None and request.problems == []


def test_reask_keeps_the_previous_default_where_the_answer_is_unusable():
    groups = {
        **GROUPS_8,
        "default": {"S01": {"id": "control"}, "S02": {"id": "control"}},
    }
    request = _request({**SOURCE, "default": {"id": "SRC-A"}}, groups)
    submitted = {"source": {"id": "SRC-C"}, "groups": {"S02": {"id": "treated"}}}
    again = reask(request, [], submitted)
    assert again.fields[0].default == {"id": "SRC-A"}  # SRC-C is unavailable
    assert again.fields[1].default == {
        "S01": {"id": "control"},
        "S02": {"id": "treated"},
    }


def test_reask_without_a_submitted_answer_keeps_every_default():
    request = _request({**SOURCE, "default": {"id": "SRC-A"}}, GROUPS_8)
    again = reask(request, [Problem(path="responds_to.decision_id", message="x")], None)
    assert [f.default for f in again.fields] == [{"id": "SRC-A"}, None]
    assert again.request_version == 2


def test_reask_says_whether_the_defaults_came_from_the_user():
    request = _request({**SOURCE, "default": {"id": "SRC-A"}}, GROUPS_8)
    assert request.prefilled_from_answer is False
    assert request.to_wire()["prefilled_from_answer"] is False
    problem = [Problem(path="groups", message="no answer")]
    from_user = reask(request, problem, {"source": {"id": "SRC-B"}})
    assert from_user.prefilled_from_answer is True
    assert from_user.to_wire()["prefilled_from_answer"] is True
    assert DecisionRequest.model_validate(from_user.to_wire()) == from_user
    # Misrouted or malformed: the defaults are still the supervisor's.
    misrouted = reask(request, problem, None)
    assert misrouted.prefilled_from_answer is False
    assert misrouted.to_wire()["prefilled_from_answer"] is False


@pytest.mark.parametrize("action", ["defer", "cancel", "free_text"])
def test_reask_with_an_empty_answer_keeps_the_defaults_the_supervisors(action):
    """Finding 3: defer, cancel and free_text carry an empty answer."""
    request = _request({**SOURCE, "default": {"id": "SRC-A"}}, GROUPS_8)
    stale = [Problem(path="responds_to.request_version", message="old version")]
    text = {"text": "SRC-B, I think"} if action == "free_text" else {}
    again = reask(request, stale, _answer(request, action=action, **text).answer)
    assert again.fields[0].default == {"id": "SRC-A"}
    assert again.prefilled_from_answer is False


def test_reask_with_an_empty_answer_and_no_defaults_is_not_prefilled():
    request = _request(SOURCE, GROUPS_8)
    assert reask(request, [], {}).prefilled_from_answer is False


def test_reask_after_a_partial_answer_keeps_but_does_not_claim_the_rest():
    """A field left out keeps the supervisor's default, not as the user's."""
    from lobster.cli_internal.decision_screens import dashboard_data

    request = _request(
        {**SOURCE, "default": {"id": "SRC-A"}},
        {**GROUPS_8, "default": {"S01": {"id": "control"}}},
        view="dashboard",
    )
    problem = [Problem(path="source", message="no answer")]
    again = reask(request, problem, {"groups": {"S02": {"id": "treated"}}})
    assert again.fields[0].default == {"id": "SRC-A"}
    # S01 is still the supervisor's proposal, so groups isn't all the user's.
    assert again.fields[1].default == {
        "S01": {"id": "control"},
        "S02": {"id": "treated"},
    }
    assert again.prefilled_from_answer is False
    for section in dashboard_data(again)["sections"]:
        assert "answer" not in section


def test_reask_with_every_default_from_the_user_is_prefilled():
    request = _request(
        {**SOURCE, "default": {"id": "SRC-A"}},
        {**GROUPS_8, "default": {"S01": {"id": "control"}}},
    )
    submitted = {"source": {"id": "SRC-B"}, "groups": {"S01": {"id": "treated"}}}
    again = reask(request, [], submitted)
    assert again.prefilled_from_answer is True
    # An unusable value keeps the supervisor's default: not the user's.
    bad = {"source": {"id": "SRC-C"}, "groups": {"S01": {"id": "treated"}}}
    assert reask(request, [], bad).prefilled_from_answer is False


def test_reask_after_the_users_answer_keeps_their_defaults_theirs():
    """Once every default is the user's, a later partial answer keeps it so."""
    request = _request({**SOURCE, "default": {"id": "SRC-A"}}, GROUPS_8)
    problem = [Problem(path="groups", message="no answer")]
    refused = reask(request, problem, {"source": {"id": "SRC-B"}})
    assert reask(refused, problem, {}).prefilled_from_answer is True
    partial = reask(refused, problem, {"groups": {"S01": {"id": "control"}}})
    assert partial.prefilled_from_answer is True


def test_reask_without_an_answer_keeps_where_the_defaults_came_from():
    """Defaults that are unchanged keep their origin: after the user's answer
    was refused, a misrouted reply still leaves the user's answers in place."""
    request = _request({**SOURCE, "default": {"id": "SRC-A"}}, GROUPS_8)
    problem = [Problem(path="groups", message="no answer")]
    refused = reask(request, problem, {"source": {"id": "SRC-B"}})
    misrouted = reask(refused, problem, None)
    assert misrouted.fields[0].default == {"id": "SRC-B"}
    assert misrouted.prefilled_from_answer is True


# --- free_text ---


def test_free_text_needs_text_and_only_free_text_has_text():
    responds_to = {"decision_id": "dec-x", "request_version": 1}
    with pytest.raises(ValidationError, match="non-empty text"):
        DecisionAnswer.model_validate(
            {"responds_to": responds_to, "action": "free_text", "text": "  "}
        )
    with pytest.raises(ValidationError, match="only for the free_text action"):
        DecisionAnswer.model_validate(
            {"responds_to": responds_to, "action": "submit", "text": "hello"}
        )


def test_free_text_is_returned_without_shape_or_answer_checks():
    request = _checked_request(GROUPS_8, QC_REVIEW, checks=[KEPT])
    answer = _answer(request, action="free_text", text=" S01 and S02 control ")
    result = check_answer(request, answer)
    assert result == {
        "decision_id": request.decision_id,
        "answer_id": answer.answer_id,
        "action": "free_text",
        "text": "S01 and S02 control",
    }
    assert "free_text" not in request.actions


# --- data_files fingerprints ---


def _workspace(tmp_path, **files):
    for name, content in files.items():
        path = tmp_path / "data" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return tmp_path


def test_fingerprint_hashes_each_workspace_file(tmp_path):
    import hashlib

    workspace = _workspace(tmp_path, **{"counts.csv": "gene,S01\nG1,5\n"})
    snapshot = fingerprint(["data/counts.csv"], workspace)
    assert snapshot.model_dump() == {
        "files": [
            {
                "path": "data/counts.csv",
                "sha256": hashlib.sha256(b"gene,S01\nG1,5\n").hexdigest(),
            }
        ]
    }


@pytest.mark.parametrize(
    "path, message",
    [
        ("../outside.csv", "inside the workspace"),
        ("data/../../outside.csv", "inside the workspace"),
        ("/etc/hostname", "inside the workspace"),
        ("data/missing.csv", "can't read"),
        ("data", "can't read"),
    ],
)
def test_fingerprint_refuses_paths_it_cannot_trust(tmp_path, path, message):
    workspace = _workspace(tmp_path / "ws", **{"counts.csv": "x"})
    (tmp_path / "outside.csv").write_text("secret")
    with pytest.raises(ValueError, match=message):
        fingerprint([path], workspace)


def test_fingerprint_refuses_a_link_out_of_the_workspace(tmp_path):
    workspace = _workspace(tmp_path / "ws", **{"counts.csv": "x"})
    (tmp_path / "outside.csv").write_text("secret")
    (workspace / "data" / "link.csv").symlink_to(tmp_path / "outside.csv")
    with pytest.raises(ValueError, match="outside the workspace"):
        fingerprint(["data/link.csv"], workspace)


def test_fingerprint_may_report_a_missing_file_instead_of_refusing(tmp_path):
    workspace = _workspace(tmp_path, **{"counts.csv": "x"})
    snapshot = fingerprint(
        ["data/counts.csv", "data/gone.csv"], workspace, missing_ok=True
    )
    assert [f.sha256 is None for f in snapshot.files] == [False, True]


def test_fingerprint_at_answer_time_reports_a_link_out_as_changed(tmp_path):
    """Finding 9: any failure at answer time means the data changed."""
    workspace = _workspace(tmp_path / "ws", **{"counts.csv": "x"})
    (tmp_path / "outside.csv").write_text("secret")
    link = workspace / "data" / "counts.csv"
    link.unlink()
    link.symlink_to(tmp_path / "outside.csv")
    snapshot = fingerprint(["data/counts.csv"], workspace, missing_ok=True)
    assert snapshot.files[0].sha256 is None


def test_data_files_are_workspace_relative_strings():
    with pytest.raises(ValidationError, match="duplicate"):
        _context(SOURCE, data_files=["data/a.csv", "data/a.csv"])
    with pytest.raises(ValidationError):
        _context(SOURCE, data_files=[""])
    assert _context(SOURCE, data_files=["data/a.csv"]).data_files == ["data/a.csv"]


def _snapshot_request(workspace):
    context = _context(SOURCE)
    return DecisionRequest(
        question="Which count source?",
        fields=context.fields,
        snapshot=fingerprint(["data/a.csv", "data/b.csv"], workspace),
    )


def test_changed_data_makes_a_submitted_answer_stale(tmp_path):
    workspace = _workspace(tmp_path, **{"a.csv": "1", "b.csv": "2"})
    request = _snapshot_request(workspace)
    (workspace / "data" / "a.csv").write_text("changed")
    (workspace / "data" / "b.csv").unlink()
    now = fingerprint(["data/a.csv", "data/b.csv"], workspace, missing_ok=True)
    with pytest.raises(AnswerStale) as caught:
        check_answer(request, _answer(request, {"source": {"id": "SRC-A"}}), now)
    assert [(p.path, p.message) for p in caught.value.problems] == [
        ("data_files", "the data changed while you were reviewing it: data/a.csv"),
        ("data_files", "the data changed while you were reviewing it: data/b.csv"),
    ]
    assert caught.value.snapshot == now


def test_unchanged_data_is_accepted(tmp_path):
    workspace = _workspace(tmp_path, **{"a.csv": "1", "b.csv": "2"})
    request = _snapshot_request(workspace)
    now = fingerprint(["data/a.csv", "data/b.csv"], workspace, missing_ok=True)
    result = check_answer(request, _answer(request, {"source": {"id": "SRC-A"}}), now)
    assert result["answer"] == {"source": {"id": "SRC-A"}}


def test_reask_takes_fresh_fingerprints_but_keeps_a_gone_file_stale(tmp_path):
    workspace = _workspace(tmp_path, **{"a.csv": "1", "b.csv": "2"})
    request = _snapshot_request(workspace)
    (workspace / "data" / "a.csv").write_text("changed")
    (workspace / "data" / "b.csv").unlink()
    now = fingerprint(["data/a.csv", "data/b.csv"], workspace, missing_ok=True)
    again = reask(request, [], {"source": {"id": "SRC-A"}}, snapshot=now)
    assert again.snapshot.files[0].sha256 == now.files[0].sha256
    assert again.snapshot.files[1].sha256 == request.snapshot.files[1].sha256
    with pytest.raises(AnswerStale, match="data/b.csv"):
        check_answer(again, _answer(again, {"source": {"id": "SRC-A"}}), now)
    assert isinstance(again.snapshot, FileSnapshot)


def test_plain_number_never_uses_exponent_form():
    from lobster.services.interaction.component_schemas import plain_number

    assert plain_number(2e6) == "2,000,000"
    assert plain_number(2_000_519) == "2,000,519"
    assert plain_number(0.31) == "0.31"
    assert plain_number(0.0001) == "0.0001"
    assert plain_number(-1500) == "-1,500"


# --- v3 findings: numbers, names and files the checks must refuse ---


@pytest.mark.parametrize(
    "value, message",
    [
        (float("nan"), "expected a finite number"),
        (float("inf"), "expected a finite number"),
        (float("-inf"), "expected a finite number"),
        pytest.param(10**400, "the number is too large", id="huge-int"),
    ],
)
def test_a_number_that_is_not_finite_is_refused(value, message):
    """NaN passes every bound and isn't valid JSON; an int past a float's
    range used to raise OverflowError instead of a problem."""
    field = {"key": "mito", "type": "number", "min": 0, "max": 1}
    request = _request(field)
    with pytest.raises(AnswerRejected) as error:
        check_answer(request, _answer(request, {"mito": value}))
    assert [(p.path, p.message) for p in error.value.problems] == [("mito", message)]


@pytest.mark.parametrize("key", ["a.b", "qc[0]", "x]", " mito", "mito "])
def test_a_key_that_would_break_problem_paths_is_refused(key):
    """Problem paths join keys with "." and "[i]": a key holding them would
    point at the wrong field."""
    with pytest.raises(ValidationError, match="can't contain"):
        AskUserContext.model_validate(
            {"fields": [{"key": key, "type": "text", "question": "Note?"}]}
        )


@pytest.mark.parametrize("item_id", ["S01 ", " S01", "   "])
def test_an_item_id_with_spaces_at_an_end_is_refused(item_id):
    """ "S01" and "S01 " look the same on screen but are different answers."""
    with pytest.raises(ValidationError, match="spaces at an end or is blank"):
        AskUserContext.model_validate(
            {"fields": [{"key": "pick", "type": "choice", "options": [item_id, "S02"]}]}
        )


def test_fingerprint_refuses_a_pipe_instead_of_waiting_on_it(tmp_path):
    """Reading a FIFO blocks until something writes to it; only regular
    files are read. At answer time it counts as changed data."""
    import os

    workspace = _workspace(tmp_path, **{"counts.csv": "x"})
    os.mkfifo(workspace / "data" / "pipe.csv")
    with pytest.raises(ValueError, match="not a file"):
        fingerprint(["data/pipe.csv"], workspace)
    snapshot = fingerprint(["data/pipe.csv"], workspace, missing_ok=True)
    assert snapshot.files[0].sha256 is None


def _without_removal_rule(problems):
    """The problems other than "takes something out: give it a reason field"."""
    return [p for p in problems if "takes something out" not in p.message]


@pytest.mark.parametrize(
    "option",
    [
        "remove",
        "exclude",
        "drop_sample",
        "discard",
        "leave_out",
        {"id": "x", "label": "Leave out of the comparison"},
        {"id": "removed", "label": "Removed"},
    ],
)
def test_an_option_that_takes_something_out_needs_a_reason(option):
    """Without depending on a reason_required check (v4 live test 1.1)."""
    field = {
        "key": "qc",
        "type": "choice",
        "for_each": ["S01"],
        "options": ["retain", option],
    }
    (problem,) = check_request(_context(field))
    option_id = option if isinstance(option, str) else option["id"]
    assert problem.path == "qc"
    assert problem.message == (
        f"option '{option_id}' sounds like it takes something out: give it a "
        'reason field, "fields": [{"key": "reason", "type": "text"}], or mark it '
        '"removes": false if it doesn\'t'
    )


def _one_option(option):
    return _context(
        {
            "key": "qc",
            "type": "choice",
            "for_each": ["S01"],
            "options": ["keep", option],
        }
    )


@pytest.mark.parametrize("option", ["omit", "not_included", "dropout_cells"])
def test_unmarked_names_that_sound_like_a_removal_are_refused(option):
    """omit and not included were missed before; dropout_cells still sounds
    like one until it is marked removes: false."""
    (problem,) = check_request(_one_option(option))
    assert "sounds like it takes something out" in problem.message


def test_removes_false_lets_a_name_that_only_sounds_like_one_through():
    assert check_request(_one_option({"id": "dropout_cells", "removes": False})) == []


def test_removes_true_needs_a_reason_whatever_the_name():
    (problem,) = check_request(_one_option({"id": "batch_2", "removes": True}))
    assert problem.message == (
        "option 'batch_2' takes something out (removes: true): give it a reason "
        'field, "fields": [{"key": "reason", "type": "text"}]'
    )
    reasoned = {
        "id": "batch_2",
        "removes": True,
        "fields": [{"key": "reason", "type": "text"}],
    }
    assert check_request(_one_option(reasoned)) == []


def test_an_unavailable_removal_needs_no_reason():
    option = {"id": "omit", "removes": True, "unavailable": "locked"}
    assert check_request(_one_option(option)) == []


@pytest.mark.parametrize(
    "option",
    [
        {"id": "exclude", "fields": [{"key": "reason", "type": "text"}]},
        {"id": "drop", "unavailable": "locked"},
        "retain",
        "control",
    ],
)
def test_a_removal_with_a_reason_or_no_removal_passes(option):
    field = {
        "key": "qc",
        "type": "choice",
        "for_each": ["S01"],
        "options": ["keep", option],
    }
    assert check_request(_context(field)) == []


def test_a_removal_inside_a_record_is_found():
    field = {
        "key": "plan",
        "type": "record",
        "fields": [{"key": "step", "type": "choice", "options": ["run", "discard"]}],
    }
    (problem,) = check_request(_context(field))
    assert problem.path == "plan.step"


# --- one definition of a reason box (review: three rules disagreed) ----------

_COUNTED_REASON = {"key": "reason", "type": "text", "count": {"min": 1, "max": 1}}


def _exclude_with(*fields, checks=False, for_each=True):
    field = {
        "key": "qc",
        "type": "choice",
        "options": ["retain", {"id": "exclude", "fields": list(fields)}],
    }
    if for_each:
        field["for_each"] = ["S01", "S02"]
    context = {"fields": [field]}
    if checks:
        context["checks"] = [
            {"check": "reason_required", "field": "qc", "options": ["exclude"]}
        ]
    return AskUserContext.model_validate(context)


def test_a_counted_reason_is_no_reason_for_the_removal_rule():
    """count makes the answer a list, which no screen draws: before, this
    rule passed it and the terminal then cancelled the question."""
    (problem,) = check_request(_exclude_with(_COUNTED_REASON))
    assert "sounds like it takes something out" in problem.message
    assert '"fields": [{"key": "reason", "type": "text"}]' in problem.message


def test_a_counted_reason_is_refused_the_same_way_by_reason_required():
    problems = check_request(_exclude_with(_COUNTED_REASON, checks=True))
    messages = [p.message for p in problems]
    assert any("needs a required text field for the reason" in m for m in messages)
    assert all('"fields": [{"key": "reason", "type": "text"}]' in m for m in messages)


def test_a_plain_reason_passes_both_rules():
    reason = {"key": "reason", "type": "text"}
    assert check_request(_exclude_with(reason, checks=True)) == []


def test_a_reason_beside_another_field_passes_in_a_single_choice():
    """Only the plate, list and dashboard need the reason alone; a single
    choice asks the fields one after another."""
    reason = {"key": "reason", "type": "text"}
    when = {"key": "when", "type": "number"}
    assert check_request(_exclude_with(reason, when, checks=True, for_each=False)) == []


def test_the_screens_and_the_rules_agree_on_a_reason_box():
    from lobster.services.interaction.component_schemas import (
        _asks_a_reason,
        _is_one_required_text,
    )

    plain = _exclude_with({"key": "reason", "type": "text"}).fields[0].options[1]
    counted = _exclude_with(_COUNTED_REASON).fields[0].options[1]
    assert _asks_a_reason(plain) and _is_one_required_text(plain.fields)
    assert not _asks_a_reason(counted) and not _is_one_required_text(counted.fields)


# --- number limits written as text (review) --------------------------------


def test_number_limits_written_as_text_are_compared_as_numbers():
    """ "min": "0" crashed the build; "10" > "9" was accepted as text."""
    ok = AskUserContext.model_validate(
        {"fields": [{"key": "n", "type": "number", "min": "0", "max": 1}]}
    )
    assert (ok.fields[0].min, ok.fields[0].max) == (0, 1)
    with pytest.raises(ValidationError, match=r"min \(10\) is larger than max \(9\)"):
        AskUserContext.model_validate(
            {"fields": [{"key": "n", "type": "number", "min": "10", "max": "9"}]}
        )
    with pytest.raises(ValidationError, match="valid number"):
        AskUserContext.model_validate(
            {"fields": [{"key": "n", "type": "number", "min": "abc", "max": 1}]}
        )
