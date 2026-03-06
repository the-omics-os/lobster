"""Tests for the HITL component mapper (Phase 2.3)."""

from lobster.services.interaction.component_mapper import map_question
from lobster.services.interaction.component_schemas import ComponentSelection


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
