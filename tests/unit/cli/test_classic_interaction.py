"""Tests for classic CLI HITL fallback renderers (Phase 2.5)."""

from unittest.mock import patch

import pytest

from lobster.cli_internal.classic_interaction import handle_interrupt_classic


def test_confirm_yes():
    data = {
        "component": "confirm",
        "data": {"question": "Proceed?", "default": False},
        "fallback_prompt": "Proceed? [y/N]",
    }
    with patch("builtins.input", return_value="y"):
        result = handle_interrupt_classic(data)
    assert result == {"confirmed": True}


def test_confirm_no():
    data = {
        "component": "confirm",
        "data": {"question": "Proceed?", "default": False},
        "fallback_prompt": "Proceed? [y/N]",
    }
    with patch("builtins.input", return_value="n"):
        result = handle_interrupt_classic(data)
    assert result == {"confirmed": False}


def test_confirm_empty_uses_default_true():
    data = {
        "component": "confirm",
        "data": {"question": "Continue?", "default": True},
        "fallback_prompt": "Continue? [Y/n]",
    }
    with patch("builtins.input", return_value=""):
        result = handle_interrupt_classic(data)
    assert result == {"confirmed": True}


def test_confirm_empty_uses_default_false():
    data = {
        "component": "confirm",
        "data": {"question": "Delete?", "default": False},
        "fallback_prompt": "Delete? [y/N]",
    }
    with patch("builtins.input", return_value=""):
        result = handle_interrupt_classic(data)
    assert result == {"confirmed": False}


def test_select_valid_choice():
    data = {
        "component": "select",
        "data": {
            "question": "Method?",
            "options": ["DESeq2", "CPM", "TMM"],
        },
        "fallback_prompt": "Method?\n  1. DESeq2\n  2. CPM\n  3. TMM",
    }
    with patch("builtins.input", return_value="2"):
        result = handle_interrupt_classic(data)
    assert result == {"selected": "CPM", "index": 1}


def test_select_first_option():
    data = {
        "component": "select",
        "data": {"question": "Pick", "options": ["A", "B"]},
        "fallback_prompt": "Pick",
    }
    with patch("builtins.input", return_value="1"):
        result = handle_interrupt_classic(data)
    assert result == {"selected": "A", "index": 0}


def test_threshold_default_on_empty():
    data = {
        "component": "threshold_slider",
        "data": {"label": "p-value", "min": 0.0, "max": 1.0, "default": 0.05},
        "fallback_prompt": "p-value (0.0-1.0, default: 0.05)",
    }
    with patch("builtins.input", return_value=""):
        result = handle_interrupt_classic(data)
    assert result == {"value": 0.05}


def test_threshold_custom_value():
    data = {
        "component": "threshold_slider",
        "data": {"label": "cutoff", "min": 0.0, "max": 1.0, "default": 0.5},
        "fallback_prompt": "cutoff",
    }
    with patch("builtins.input", return_value="0.3"):
        result = handle_interrupt_classic(data)
    assert result == {"value": 0.3}


def test_text_input_fallback():
    data = {
        "component": "text_input",
        "data": {"question": "What genes?"},
        "fallback_prompt": "What genes?",
    }
    with patch("builtins.input", return_value="TP53, BRCA1"):
        result = handle_interrupt_classic(data)
    assert result == {"answer": "TP53, BRCA1"}


def test_unknown_component_falls_back_to_text():
    data = {
        "component": "unknown_widget",
        "data": {"question": "Something?"},
        "fallback_prompt": "Something?",
    }
    with patch("builtins.input", return_value="answer"):
        result = handle_interrupt_classic(data)
    assert result == {"answer": "answer"}


def test_cell_type_selector():
    data = {
        "component": "cell_type_selector",
        "data": {
            "clusters": [
                {"id": 0, "size": 500, "markers": ["CD3D"]},
                {"id": 1, "size": 200, "markers": ["MS4A1"]},
            ]
        },
        "fallback_prompt": "Annotate clusters",
    }
    inputs = iter(["T cells", "B cells"])
    with patch("builtins.input", side_effect=inputs):
        result = handle_interrupt_classic(data)
    assert result == {"assignments": {"0": "T cells", "1": "B cells"}}


@pytest.mark.parametrize("options", [None, [], "A", {}, [None], [1], [""], ["A", " "]])
def test_invalid_select_options_fall_back_to_text(options, capsys):
    data = {"component": "select", "data": {"question": "Choose?", "options": options}}
    with patch("builtins.input", return_value="custom answer") as prompt:
        assert handle_interrupt_classic(data) == {"answer": "custom answer"}
    prompt.assert_called_once_with("\nChoose?: ")
    assert "between" not in capsys.readouterr().out


@pytest.mark.parametrize("data", [{}, {"options": []}, None, []])
def test_missing_or_malformed_select_data_falls_back_to_text(data):
    payload = {"component": "select", "fallback_prompt": "Choose?", "data": data}
    with patch("builtins.input", return_value="answer") as prompt:
        assert handle_interrupt_classic(payload) == {"answer": "answer"}
    prompt.assert_called_once_with("\nChoose?: ")


@pytest.mark.parametrize("payload", [None, [], "Choose?"])
def test_malformed_payload_falls_back_to_text(payload):
    with patch("builtins.input", return_value="answer") as prompt:
        assert handle_interrupt_classic(payload) == {"answer": "answer"}
    prompt.assert_called_once()


# --- ask_user decision requests: structured plain-text prompts ---


def _decision(*fields, view="default", checks=None, **envelope):
    from lobster.services.interaction.component_schemas import (
        AskUserContext,
        DecisionRequest,
    )

    context = AskUserContext.model_validate(
        {"view": view, "fields": list(fields), "checks": checks or []}
    )
    return DecisionRequest(
        question="Review",
        view=context.view,
        fields=context.fields,
        checks=context.checks,
        **envelope,
    ).to_wire()


def _classic(request, typed, capsys):
    """Answer a request with the typed lines; return (result, printed text).

    The result is what ask_user's contract check makes of the answer.
    """
    from lobster.services.interaction.component_schemas import (
        DecisionAnswer,
        DecisionRequest,
        check_answer,
    )

    with patch("builtins.input", side_effect=list(typed)) as prompt:
        reply = handle_interrupt_classic(request)
    assert prompt.call_count == len(typed)  # every typed line was asked for
    result = check_answer(
        DecisionRequest.model_validate(request), DecisionAnswer.model_validate(reply)
    )
    return result, capsys.readouterr().out


SOURCE = {
    "key": "source",
    "type": "choice",
    "details": "Recommended: SRC-A",
    "options": [
        {"id": "SRC-A", "label": "Author matrix", "details": "Raw counts"},
        "SRC-B",
        {"id": "SRC-C", "unavailable": "Normalized; build unknown"},
    ],
}
QC = {
    "key": "qc",
    "type": "choice",
    "question": "Keep each flagged sample?",
    "for_each": [
        {"id": "S02", "label": "Donor B", "details": "Low library size"},
        "S06",
        {"id": "S07", "unavailable": "already removed"},
    ],
    "options": [
        "retain",
        {"id": "exclude", "fields": [{"key": "reason", "type": "text"}]},
        "defer",
    ],
}


def test_choice_lists_numbered_options_and_refuses_unavailable(capsys):
    result, shown = _classic(_decision(SOURCE), ["3", "7", "1"], capsys)
    assert result["answer"] == {"source": {"id": "SRC-A"}}
    assert "1. Author matrix (SRC-A) — Raw counts" in shown
    assert "3. SRC-C (unavailable: Normalized; build unknown)" in shown
    assert "SRC-C can't be picked: Normalized; build unknown." in shown
    assert "7 is not an option (1 to 3)." in shown


def test_count_takes_comma_separated_numbers(capsys):
    field = {
        "key": "treated",
        "type": "choice",
        "count": {"min": 1, "max": 2},
        "options": ["S05", "S06", "S07"],
    }
    result, shown = _classic(_decision(field), ["1, 2, 3", "1, 3"], capsys)
    assert result["answer"] == {"treated": [{"id": "S05"}, {"id": "S07"}]}
    assert "Pick between 1 and 2." in shown


def test_for_each_asks_each_available_item_and_a_reason_to_exclude(capsys):
    result, shown = _classic(
        _decision(QC), ["1", "2", "", "high mito fraction"], capsys
    )
    assert result["answer"] == {
        "qc": {
            "S02": {"id": "retain"},
            "S06": {"id": "exclude", "fields": {"reason": "high mito fraction"}},
        }
    }
    assert "S07: not answered here (already removed)" in shown
    assert "2. exclude (asks for a reason)" in shown
    assert "An answer is needed." in shown  # an empty reason is refused


def test_plate_asks_each_sample_and_lists_unavailable_ones(capsys):
    plate = {
        "key": "groups",
        "type": "choice",
        "view": "plate_map",
        "for_each": [
            {"id": "S01", "label": "Donor A"},
            {"id": "S08", "label": "Donor H", "details": "condition unknown"},
            {"id": "S09", "unavailable": "consent withdrawn"},
        ],
        "options": ["control", "treated"],
        "default": {"S01": {"id": "control"}},
    }
    with patch("builtins.input", side_effect=["", "2"]) as prompt:
        reply = handle_interrupt_classic(_decision(plate))
    asked = [call.args[0] for call in prompt.call_args_list]
    assert asked == ["S01 Donor A [Enter = 1]: ", "S08 Donor H (condition unknown): "]
    assert reply["answer"] == {
        "groups": {"S01": {"id": "control"}, "S08": {"id": "treated"}}
    }
    assert "Not assignable: S09 — consent withdrawn" in capsys.readouterr().out


def test_typed_number_is_asked_again_until_it_is_a_number(capsys):
    field = {"key": "reads", "type": "number", "integer": True, "min": 0}
    result, shown = _classic(_decision(field), ["lots", "2.5", "10000"], capsys)
    assert result["answer"] == {"reads": 10000}
    assert "'lots' is not a number." in shown
    assert "2.5 is not a whole number." in shown


def test_slider_number_is_kept_within_its_bounds(capsys):
    field = {
        "key": "mito",
        "type": "number",
        "view": "slider",
        "min": 0,
        "max": 0.5,
        "default": 0.2,
    }
    result, shown = _classic(_decision(field), ["0.9", "x", "0.15"], capsys)
    assert result["answer"] == {"mito": 0.15}
    assert "Enter a value from 0 to 0.5." in shown and "Enter a number." in shown
    result, _ = _classic(_decision(field), [""], capsys)
    assert result["answer"] == {"mito": 0.2}  # the default, kept on Enter


def test_text_needs_an_answer(capsys):
    result, shown = _classic(
        _decision({"key": "note", "type": "text"}), ["", "looks fine"], capsys
    )
    assert result["answer"] == {"note": "looks fine"}
    assert "An answer is needed." in shown


def _small_dashboard():
    return _decision(
        {**SOURCE, "question": "Count source?"},
        {"key": "notes", "type": "text", "question": "Anything to note?"},
        view="dashboard",
    )


def test_dashboard_asks_sections_in_order_then_confirms(capsys):
    result, shown = _classic(_small_dashboard(), ["2", "ok", "y"], capsys)
    assert result["action"] == "submit"
    assert result["answer"] == {"source": {"id": "SRC-B"}, "notes": "ok"}
    assert shown.index("Count source?") < shown.index("Anything to note?")
    assert "Count source?: SRC-B" in shown  # the summary before confirming


def test_dashboard_not_confirmed_is_deferred(capsys):
    result, _ = _classic(_small_dashboard(), ["2", "ok", ""], capsys)
    assert result["action"] == "defer"
    assert "answer" not in result


@pytest.mark.parametrize("key, action", [("d", "defer"), ("q", "cancel")])
def test_d_defers_and_q_cancels_at_any_prompt(key, action, capsys):
    result, _ = _classic(_decision(SOURCE), [key], capsys)
    assert result["action"] == action
    result, _ = _classic(_decision(QC), ["2", key], capsys)  # at the reason prompt
    assert result["action"] == action


def test_reask_prints_the_problems_first_and_keeps_the_answers(capsys):
    from lobster.services.interaction.component_schemas import (
        DecisionRequest,
        Problem,
        reask,
    )

    first = DecisionRequest.model_validate(_decision(SOURCE))
    problem = Problem(path="source", message="pick a raw count table")
    request = reask(first, [problem], {"source": {"id": "SRC-B"}}).to_wire()
    with patch("builtins.input", side_effect=[""]) as prompt:
        reply = handle_interrupt_classic(request)
    shown = capsys.readouterr().out
    assert reply["responds_to"]["request_version"] == 2
    assert reply["answer"] == {"source": {"id": "SRC-B"}}  # kept on Enter
    assert prompt.call_args.args[0] == "Number [Enter = 2]: "
    assert shown.index("⚠ pick a raw count table") < shown.index("1. Author matrix")


def test_shape_the_terminal_cannot_show_is_cancelled_with_a_reason(capsys):
    request = _decision({"key": "times", "type": "number", "count": {"min": 1}})
    with patch("builtins.input") as typed:
        reply = handle_interrupt_classic(request)
    typed.assert_not_called()  # never asks for (and never guesses) an answer
    assert reply["action"] == "cancel"
    assert "Don't ask again with the same shape" in reply["comment"]
    assert "can't show" in capsys.readouterr().out


def test_classic_answers_go_through_the_checks_and_reask():
    """Real checkpointed graph and ask_user: an answer that empties a group is
    asked again with the problem shown and the answers kept, then accepted."""
    from langgraph.types import Command

    from tests.unit.cli.test_go_tui_components import (
        _CONTROLS_DASHBOARD,
        _one_question_graph,
    )

    graph = _one_question_graph("Review groups and QC", _CONTROLS_DASHBOARD)
    config = {"configurable": {"thread_id": "full-path"}}
    rounds = [
        # groups kept as pre-filled; both controls excluded; confirmed.
        ["", "", "", "", "2", "few reads", "2", "degraded", "y"],
        # asked again: groups kept, S01 still excluded (reason kept), S02 kept.
        ["", "", "", "", "", "", "1", "y"],
    ]
    graph_input, printed = {"messages": []}, []
    for typed in rounds:
        for _ in graph.stream(graph_input, config, stream_mode="updates"):
            pass
        (task,) = [t for t in graph.get_state(config).tasks if t.interrupts]
        (pending,) = task.interrupts
        with patch("builtins.input", side_effect=typed):
            with patch("builtins.print", side_effect=lambda *a: printed.append(a)):
                reply = handle_interrupt_classic(pending.value)
        graph_input = Command(resume={pending.id: reply})
    for _ in graph.stream(graph_input, config, stream_mode="updates"):
        pass
    messages = graph.get_state(config).values["messages"]
    (tool,) = [m for m in messages if m.type == "tool"]
    import json

    result = json.loads(tool.content)
    assert (result["action"], result["attempts"]) == ("submit", 2)
    assert result["answer"]["qc"] == {
        "S01": {"id": "exclude", "fields": {"reason": "few reads"}},
        "S02": {"id": "retain"},
    }
    text = "\n".join(" ".join(map(str, line)) for line in printed)
    assert "⚠ 'control' has no items left after removing S01, S02" in text
