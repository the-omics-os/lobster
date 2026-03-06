"""Tests for the ask_user HITL tool (Phase 2.4)."""

from unittest.mock import patch

from lobster.tools.user_interaction import ask_user


def test_ask_user_has_aquadif_metadata():
    assert hasattr(ask_user, "metadata")
    assert ask_user.metadata["categories"] == ["UTILITY"]
    assert ask_user.metadata["provenance"] is False
    assert ask_user.tags == ["UTILITY"]


def test_ask_user_calls_interrupt_with_component_selection():
    """ask_user should map the question and call interrupt()."""
    fake_response = {"confirmed": True}

    with patch("langgraph.types.interrupt") as mock_interrupt:
        mock_interrupt.return_value = fake_response
        result = ask_user.invoke(
            {"question": "Proceed?", "context": None}
        )

    # Verify interrupt was called with the mapped ComponentSelection.
    mock_interrupt.assert_called_once()
    call_arg = mock_interrupt.call_args[0][0]
    assert isinstance(call_arg, dict)
    assert "component" in call_arg
    assert "fallback_prompt" in call_arg

    # Result should be JSON-serialized response.
    import json
    parsed = json.loads(result)
    assert parsed == {"confirmed": True}


def test_ask_user_with_options_context():
    """ask_user with options context should produce a select component."""
    fake_response = {"selected": "CPM", "index": 1}

    with patch("langgraph.types.interrupt") as mock_interrupt:
        mock_interrupt.return_value = fake_response
        ask_user.invoke(
            {
                "question": "Which normalization?",
                "context": {"options": ["DESeq2", "CPM", "TMM"]},
            }
        )

    call_arg = mock_interrupt.call_args[0][0]
    assert call_arg["component"] == "select"
    assert call_arg["data"]["options"] == ["DESeq2", "CPM", "TMM"]
