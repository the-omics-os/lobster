"""Tests for the HITL component mapper (Phase 2.3, extended Phase 5.1)."""

from unittest.mock import MagicMock, patch

from lobster.services.interaction.component_mapper import map_question
from lobster.services.interaction.component_schemas import (
    COMPONENT_SCHEMAS,
    ComponentSelection,
)


def test_options_list_maps_to_select():
    result = map_question(
        "Which normalization method?",
        context={"options": ["DESeq2", "CPM", "TMM"]},
    )
    assert isinstance(result, ComponentSelection)
    assert result.component == "select"
    assert result.data["options"] == ["DESeq2", "CPM", "TMM"]
    assert "1." in result.fallback_prompt


def test_clusters_context_maps_to_cell_type_selector():
    clusters = [
        {"id": 0, "size": 500, "markers": ["CD3D", "CD8A"]},
        {"id": 1, "size": 200, "markers": ["MS4A1", "CD79A"]},
    ]
    result = map_question("Annotate clusters", context={"clusters": clusters})
    assert result.component == "cell_type_selector"
    assert len(result.data["clusters"]) == 2


def test_threshold_context_maps_to_threshold_slider():
    result = map_question(
        "Set p-value threshold",
        context={"min": 0.001, "max": 0.1, "default": 0.05},
    )
    assert result.component == "threshold_slider"
    assert result.data["min"] == 0.001
    assert result.data["max"] == 0.1
    assert result.data["default"] == 0.05


def test_threshold_keyword_in_question_maps_to_threshold():
    result = map_question("What log2FC cutoff should I use?")
    assert result.component == "threshold_slider"


def test_confirm_pattern_maps_to_confirm():
    result = map_question("Do you want to proceed with batch correction?")
    assert result.component == "confirm"
    assert result.data["question"] == "Do you want to proceed with batch correction?"


def test_confirm_should_i_pattern():
    result = map_question("Should I remove these outlier samples?")
    assert result.component == "confirm"


def test_fallback_maps_to_text_input():
    result = map_question("What cell types are you expecting?")
    assert result.component == "text_input"
    assert result.fallback_prompt == "What cell types are you expecting?"


def test_empty_options_falls_through():
    result = map_question("Choose one", context={"options": []})
    # Empty options list shouldn't trigger select.
    assert result.component != "select"


def test_confirm_default_propagated():
    result = map_question("Continue?", context={"default": True})
    assert result.component == "confirm"
    assert result.data["default"] is True


def test_threshold_with_unit():
    result = map_question(
        "Adjust significance",
        context={"min": 0.0, "max": 1.0, "default": 0.05, "unit": "p-value"},
    )
    assert result.component == "threshold_slider"
    assert result.data["unit"] == "p-value"
    assert "p-value" in result.fallback_prompt


# --- Phase 5.1: LLM-driven component selection ---


def _make_mock_llm(component: str = "confirm", fallback_prompt: str = "Confirm?"):
    """Create a mock LLM that returns a ComponentSelection via with_structured_output."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = ComponentSelection(
        component=component,
        data={"question": "test"},
        fallback_prompt=fallback_prompt,
    )
    mock_llm.with_structured_output.return_value = mock_structured
    return mock_llm


def test_fast_path_options_with_llm_still_uses_rules():
    """Options context must use rule-based path even when LLM is provided."""
    mock_llm = _make_mock_llm()
    result = map_question(
        "Which normalization?",
        context={"options": ["DESeq2", "CPM"]},
        llm=mock_llm,
    )
    assert result.component == "select"
    mock_llm.with_structured_output.assert_not_called()


def test_fast_path_clusters_with_llm_still_uses_rules():
    """Clusters context must use rule-based path even when LLM is provided."""
    mock_llm = _make_mock_llm()
    clusters = [{"id": 0, "size": 100, "markers": ["CD3D"]}]
    result = map_question("Annotate", context={"clusters": clusters}, llm=mock_llm)
    assert result.component == "cell_type_selector"
    mock_llm.with_structured_output.assert_not_called()


def test_fast_path_threshold_with_llm_still_uses_rules():
    """Threshold context must use rule-based path even when LLM is provided."""
    mock_llm = _make_mock_llm()
    result = map_question(
        "Set cutoff", context={"min": 0.0, "max": 1.0}, llm=mock_llm
    )
    assert result.component == "threshold_slider"
    mock_llm.with_structured_output.assert_not_called()


def test_ambiguous_question_with_llm_uses_llm_path():
    """Ambiguous question + LLM provided should use LLM selection."""
    mock_llm = _make_mock_llm(component="confirm", fallback_prompt="Please confirm")
    result = map_question("How should we handle the missing values?", llm=mock_llm)
    assert result.component == "confirm"
    mock_llm.with_structured_output.assert_called_once()


def test_ambiguous_question_no_llm_falls_back_to_rules():
    """Ambiguous question without LLM should fall back to text_input."""
    result = map_question("How should we handle the missing values?", llm=None)
    assert result.component == "text_input"


def test_llm_invalid_component_falls_back_to_text_input():
    """LLM returning invalid component name should fall back via validator."""
    mock_llm = _make_mock_llm(component="nonexistent_widget", fallback_prompt="Test")
    result = map_question("Some ambiguous question here", llm=mock_llm)
    assert result.component == "text_input"


def test_llm_exception_falls_back_to_text_input():
    """LLM raising exception should gracefully fall back to text_input."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = RuntimeError("LLM unavailable")
    mock_llm.with_structured_output.return_value = mock_structured
    result = map_question("Some ambiguous question here", llm=mock_llm)
    assert result.component == "text_input"
    assert result.fallback_prompt == "Some ambiguous question here"


def test_component_selection_empty_fallback_prompt_gets_default():
    """ComponentSelection with empty fallback_prompt should get a default."""
    sel = ComponentSelection(
        component="confirm",
        data={"question": "Are you sure?"},
        fallback_prompt="",
    )
    assert sel.fallback_prompt != ""
    assert sel.fallback_prompt == "Are you sure?"


def test_component_selection_whitespace_fallback_prompt_gets_default():
    """ComponentSelection with whitespace fallback_prompt should get a default."""
    sel = ComponentSelection(
        component="confirm",
        data={"question": "Continue?"},
        fallback_prompt="   ",
    )
    assert sel.fallback_prompt.strip() != ""


def test_component_selection_invalid_component_corrected():
    """ComponentSelection with invalid component name should be corrected to text_input."""
    sel = ComponentSelection(
        component="does_not_exist",
        data={"question": "Test?"},
        fallback_prompt="Test?",
    )
    assert sel.component == "text_input"


def test_llm_select_prompt_includes_component_descriptions():
    """_llm_select_component should build a prompt with component schema info."""
    mock_llm = _make_mock_llm(component="select", fallback_prompt="Choose")
    result = map_question("Pick the best approach", llm=mock_llm)
    # Verify LLM was called and the prompt includes component info
    mock_llm.with_structured_output.assert_called_once()
    structured_chain = mock_llm.with_structured_output.return_value
    call_args = structured_chain.invoke.call_args[0][0]
    # The prompt should mention at least some component names
    for comp_name in COMPONENT_SCHEMAS:
        assert comp_name in call_args
