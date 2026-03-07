"""Tests for the ask_user HITL tool (Phase 2.4, extended Phase 5.1)."""

from unittest.mock import MagicMock, patch

from lobster.tools.user_interaction import ask_user, create_ask_user_tool


def test_ask_user_has_aquadif_metadata():
    """Module-level ask_user (backward compat) has AQUADIF metadata."""
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


# --- Phase 5.1: Factory pattern tests ---


def test_create_ask_user_tool_returns_tool_with_metadata():
    """Factory-created tool has AQUADIF metadata."""
    tool = create_ask_user_tool(llm=None)
    assert hasattr(tool, "metadata")
    assert tool.metadata["categories"] == ["UTILITY"]
    assert tool.metadata["provenance"] is False
    assert tool.tags == ["UTILITY"]


def test_create_ask_user_tool_with_llm_passes_llm_to_mapper():
    """Factory-created tool passes LLM to map_question."""
    mock_llm = MagicMock()
    tool = create_ask_user_tool(llm=mock_llm)

    with (
        patch("langgraph.types.interrupt") as mock_interrupt,
        patch(
            "lobster.services.interaction.component_mapper.map_question"
        ) as mock_map,
    ):
        from lobster.services.interaction.component_schemas import ComponentSelection

        mock_map.return_value = ComponentSelection(
            component="text_input",
            data={"question": "test"},
            fallback_prompt="test",
        )
        mock_interrupt.return_value = {"answer": "hello"}
        tool.invoke({"question": "How to proceed?", "context": None})

    mock_map.assert_called_once()
    _, kwargs = mock_map.call_args
    assert kwargs.get("llm") is mock_llm


def test_create_ask_user_tool_calls_interrupt_with_selection():
    """Factory-created tool calls interrupt() with ComponentSelection dict."""
    tool = create_ask_user_tool(llm=None)

    with patch("langgraph.types.interrupt") as mock_interrupt:
        mock_interrupt.return_value = {"confirmed": True}
        tool.invoke({"question": "Continue?", "context": None})

    mock_interrupt.assert_called_once()
    call_arg = mock_interrupt.call_args[0][0]
    assert isinstance(call_arg, dict)
    assert "component" in call_arg


def test_create_ask_user_tool_no_llm_backward_compat():
    """create_ask_user_tool(llm=None) works like the old standalone ask_user."""
    tool = create_ask_user_tool(llm=None)

    with (
        patch("langgraph.types.interrupt") as mock_interrupt,
        patch(
            "lobster.services.interaction.component_mapper.map_question"
        ) as mock_map,
    ):
        from lobster.services.interaction.component_schemas import ComponentSelection

        mock_map.return_value = ComponentSelection(
            component="confirm",
            data={"question": "ok?"},
            fallback_prompt="ok?",
        )
        mock_interrupt.return_value = {"confirmed": True}
        tool.invoke({"question": "ok?", "context": None})

    _, kwargs = mock_map.call_args
    assert kwargs.get("llm") is None
