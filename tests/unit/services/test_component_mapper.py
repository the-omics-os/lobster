"""Tests for the HITL component mapper: question + context -> decision fields."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from lobster.services.interaction.component_mapper import map_question
from lobster.services.interaction.component_schemas import AskUserContext


def _only_field(result):
    assert isinstance(result, AskUserContext)
    assert len(result.fields) == 1
    return result.fields[0]


def test_options_list_maps_to_select():
    field = _only_field(
        map_question(
            "Which normalization method?",
            context={"options": ["DESeq2", "CPM", "TMM"]},
        )
    )
    assert field.type == "choice"
    assert [option.id for option in field.options] == ["DESeq2", "CPM", "TMM"]


def test_clusters_context_maps_to_cell_type_selector():
    clusters = [
        {"id": 0, "size": 500, "markers": ["CD3D", "CD8A"]},
        {"id": 1, "size": 200, "markers": ["MS4A1", "CD79A"]},
    ]
    field = _only_field(
        map_question("Annotate clusters", context={"clusters": clusters})
    )
    # One text answer (the cell type) per cluster.
    assert (field.key, field.type) == ("cell_type", "text")
    assert [item.id for item in field.for_each] == ["0", "1"]
    assert field.for_each[0].details == "500 cells · CD3D, CD8A"


def test_threshold_context_maps_to_threshold_slider():
    field = _only_field(
        map_question(
            "Set p-value threshold",
            context={"min": 0.001, "max": 0.1, "default": 0.05},
        )
    )
    assert (field.type, field.view) == ("number", "slider")
    assert (field.min, field.max, field.default) == (0.001, 0.1, 0.05)


@pytest.mark.parametrize(
    "question",
    [
        "Which sample has the max library size?",  # code review: got a number box
        "Do you want me to continue with SRC-B or SRC-A?",  # code review: got yes/no
        "What log2FC cutoff should I use?",
        "Should I remove these outlier samples?",
    ],
)
def test_the_wording_of_a_question_never_decides_its_screen(question):
    """Words like "max" or "do you want" used to pick a number box or yes/no;
    without fields or hints, a question is free text (or shaped by the llm)."""
    field = _only_field(map_question(question))
    assert (field.type, field.question) == ("text", question)


def test_threshold_hint_without_bounds_keeps_its_default():
    """FX x2: a 5.0 mmol/L cutoff must stay reachable and suggested."""
    field = _only_field(
        map_question(
            "Which concentration flags an outlier?",
            context={"threshold": 5.0, "unit": "mmol/L"},
        )
    )
    assert (field.view, field.min, field.max, field.default) == (
        "default",
        None,
        None,
        5.0,
    )
    assert field.unit == "mmol/L"


def test_threshold_hint_with_both_bounds_is_a_slider():
    field = _only_field(
        map_question("Set cutoff", context={"min": 0, "max": 10, "default": 2.5})
    )
    assert (field.view, field.min, field.max, field.default) == ("slider", 0, 10, 2.5)


def test_fallback_maps_to_text_input():
    field = _only_field(map_question("What cell types are you expecting?"))
    assert (field.type, field.question) == (
        "text",
        "What cell types are you expecting?",
    )


def test_empty_options_falls_through():
    field = _only_field(map_question("Choose one", context={"options": []}))
    # An empty options list must not produce a choice.
    assert field.type == "text"


def test_a_bare_default_hint_makes_no_yes_no_question():
    """A yes/no question is asked explicitly, with options."""
    field = _only_field(map_question("Continue?", context={"default": True}))
    assert field.type == "text"


def test_threshold_with_unit():
    field = _only_field(
        map_question(
            "Adjust significance",
            context={"min": 0.0, "max": 1.0, "default": 0.05, "unit": "p-value"},
        )
    )
    assert field.view == "slider"
    assert field.unit == "p-value"


# --- Fields given directly ---


def test_fields_context_is_validated_and_kept():
    context = {
        "fields": [
            {"key": "source", "type": "choice", "options": ["SRC-A", "SRC-B"]},
        ]
    }
    field = _only_field(map_question("Which source?", context=context))
    assert field.key == "source"
    assert [option.id for option in field.options] == ["SRC-A", "SRC-B"]


def test_invalid_fields_context_raises_for_the_caller():
    with pytest.raises(ValidationError, match="duplicate ids"):
        map_question(
            "Which source?",
            context={"fields": [{"key": "s", "type": "choice", "options": ["A", "A"]}]},
        )


def test_fields_context_skips_the_llm():
    llm = MagicMock()
    map_question("Q?", context={"fields": [{"key": "a", "type": "text"}]}, llm=llm)
    llm.with_structured_output.assert_not_called()


def test_dict_options_keep_their_ids():
    """Stable ids survive; the old mapper flattened them to labels."""
    field = _only_field(
        map_question(
            "Which donor?",
            context={
                "options": [
                    {"id": "S01", "label": "Donor A"},
                    {"id": "S05", "label": "Donor A"},
                ]
            },
        )
    )
    assert [(o.id, o.label) for o in field.options] == [
        ("S01", "Donor A"),
        ("S05", "Donor A"),
    ]


def test_hint_options_keep_unavailable_and_details():
    """An ineligible hint option stays unavailable; it must not become pickable."""
    field = _only_field(
        map_question(
            "Which count source?",
            context={
                "options": [
                    {"id": "SRC-A", "details": "raw counts"},
                    {"id": "SRC-C", "unavailable": "normalized; build unknown"},
                ]
            },
        )
    )
    assert [(o.id, o.details, o.unavailable) for o in field.options] == [
        ("SRC-A", "raw counts", None),
        ("SRC-C", None, "normalized; build unknown"),
    ]


def test_threshold_default_out_of_range_is_dropped():
    field = _only_field(map_question("Set cutoff", context={"min": 0, "max": 0.1}))
    assert field.default is None


# --- LLM path ---


def _make_mock_llm(result=None, error=None):
    """A mock LLM whose with_structured_output(...).invoke returns or raises."""
    mock_llm = MagicMock()
    mock_structured = mock_llm.with_structured_output.return_value
    if error is not None:
        mock_structured.invoke.side_effect = error
    else:
        mock_structured.invoke.return_value = result or AskUserContext(
            fields=[{"key": "answer", "type": "choice", "options": ["yes", "no"]}]
        )
    return mock_llm


def test_fast_path_options_with_llm_still_uses_rules():
    """Options context must use rule-based path even when LLM is provided."""
    mock_llm = _make_mock_llm()
    field = _only_field(
        map_question(
            "Which normalization?",
            context={"options": ["DESeq2", "CPM"]},
            llm=mock_llm,
        )
    )
    assert [option.id for option in field.options] == ["DESeq2", "CPM"]
    mock_llm.with_structured_output.assert_not_called()


def test_fast_path_clusters_with_llm_still_uses_rules():
    """Clusters context must use rule-based path even when LLM is provided."""
    mock_llm = _make_mock_llm()
    clusters = [{"id": 0, "size": 100, "markers": ["CD3D"]}]
    field = _only_field(
        map_question("Annotate", context={"clusters": clusters}, llm=mock_llm)
    )
    assert field.key == "cell_type"
    mock_llm.with_structured_output.assert_not_called()


def test_fast_path_threshold_with_llm_still_uses_rules():
    """Threshold context must use rule-based path even when LLM is provided."""
    mock_llm = _make_mock_llm()
    field = _only_field(
        map_question("Set cutoff", context={"min": 0.0, "max": 1.0}, llm=mock_llm)
    )
    assert field.view == "slider"
    mock_llm.with_structured_output.assert_not_called()


def test_ambiguous_question_with_llm_uses_llm_path():
    """Ambiguous question + LLM provided should use the LLM's fields."""
    formatted = AskUserContext(
        fields=[{"key": "strategy", "type": "choice", "options": ["drop", "impute"]}]
    )
    mock_llm = _make_mock_llm(result=formatted)
    result = map_question("How should we handle the missing values?", llm=mock_llm)
    assert result is formatted
    mock_llm.with_structured_output.assert_called_once_with(AskUserContext)


def test_ambiguous_question_no_llm_falls_back_to_rules():
    """Ambiguous question without LLM should fall back to free text."""
    field = _only_field(map_question("How should we handle the missing values?"))
    assert field.type == "text"


def test_llm_invalid_component_falls_back_to_text_input():
    """LLM output that doesn't fit the contract falls back to free text."""
    try:
        AskUserContext.model_validate(
            {"fields": [{"key": "a", "type": "nonexistent_widget"}]}
        )
    except ValidationError as error:
        invalid_output = error
    mock_llm = _make_mock_llm(error=invalid_output)
    field = _only_field(map_question("Some ambiguous question here", llm=mock_llm))
    assert (field.type, field.question) == ("text", "Some ambiguous question here")


def test_llm_exception_falls_back_to_text_input():
    """LLM raising exception should gracefully fall back to free text."""
    mock_llm = _make_mock_llm(error=RuntimeError("LLM unavailable"))
    field = _only_field(map_question("Some ambiguous question here", llm=mock_llm))
    assert (field.type, field.question) == ("text", "Some ambiguous question here")


def test_llm_select_prompt_includes_component_descriptions():
    """The LLM prompt names every field type and carries the question."""
    mock_llm = _make_mock_llm()
    map_question("Pick the best approach", llm=mock_llm)
    mock_llm.with_structured_output.assert_called_once()
    prompt = mock_llm.with_structured_output.return_value.invoke.call_args[0][0]
    assert "Pick the best approach" in prompt
    for field_type in ("choice", "number", "text", "record"):
        assert field_type in prompt


# --- v3 findings ---


def test_hint_context_keeps_data_files_checks_and_view():
    """A hint-style context (no "fields") used to drop data_files silently,
    so the answer was never checked against the data it was about."""
    context = {"options": ["DESeq2", "CPM"], "data_files": ["data/counts.csv"]}
    result = map_question("Which normalization method?", context=context)
    assert result.data_files == ["data/counts.csv"]
    assert [option.id for option in result.fields[0].options] == ["DESeq2", "CPM"]


def test_hint_context_keys_are_still_judged():
    """Kept keys are validated, never passed through unchecked."""
    with pytest.raises(ValidationError, match="duplicate paths"):
        map_question("Pick one", context={"options": ["a"], "data_files": ["x", "x"]})


def test_placeholder_hint_is_shown_never_filled_in():
    field = _only_field(
        map_question("Describe the batch", context={"placeholder": "e.g. run 2"})
    )
    assert (field.type, field.details, field.default) == ("text", "e.g. run 2", None)


def test_threshold_hint_numbers_written_as_text_are_read():
    field = _only_field(
        map_question(
            "Set p-value threshold", {"min": "0", "max": "0.1", "default": "0.05"}
        )
    )
    assert (field.min, field.max, field.default, field.view) == (0, 0.1, 0.05, "slider")


@pytest.mark.parametrize("bad", ["nan", "inf", float("nan"), "lots", True])
def test_threshold_hint_values_that_are_not_finite_numbers_are_dropped(bad):
    field = _only_field(map_question("Set p-value threshold", {"min": 0, "max": bad}))
    assert field.max is None and field.view == "default"
