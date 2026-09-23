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
