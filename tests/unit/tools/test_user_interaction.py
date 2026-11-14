"""Tests for the ask_user HITL tool (Phase 2.4, extended Phase 5.1 and Phase 0 A2/A3)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from lobster.tools.user_interaction import ask_user, create_ask_user_tool


def call(tool, args, call_id="call_1"):
    """Invoke the tool the way the graph's ToolNode does; return the content."""
    message = tool.invoke(
        {"name": "ask_user", "args": args, "id": call_id, "type": "tool_call"}
    )
    return message.content


def envelope(answer, *, comment=None, source="human", channel=None):
    decision = {
        "decision_id": "dec_1",
        "request_version": 1,
        "answer": answer,
        "source": source,
        "comment": comment,
    }
    if channel is not None:  # absent, not null, when no channel is given
        decision["channel"] = channel
    return {"__lobster_decision__": decision}


def test_ask_user_has_aquadif_metadata():
    """Module-level ask_user (backward compat) has AQUADIF metadata."""
    assert hasattr(ask_user, "metadata")
    assert ask_user.metadata["categories"] == ["UTILITY"]
    assert ask_user.metadata["provenance"] is True
    assert ask_user.tags == ["UTILITY"]


def test_ask_user_calls_interrupt_with_component_selection():
    """ask_user should map the question and call interrupt()."""
    with patch("langgraph.types.interrupt") as mock_interrupt:
        mock_interrupt.return_value = {"confirmed": True}
        result = call(ask_user, {"question": "Proceed?", "context": None})

    mock_interrupt.assert_called_once()
    call_arg = mock_interrupt.call_args[0][0]
    assert isinstance(call_arg, dict)
    assert "component" in call_arg
    assert "fallback_prompt" in call_arg
    # A bare answer (older callers) is still returned as JSON.
    assert json.loads(result) == {"confirmed": True}


def test_interrupt_value_carries_the_tool_call_id_as_ask_id():
    with patch("langgraph.types.interrupt") as mock_interrupt:
        mock_interrupt.return_value = envelope(0.1)
        call(ask_user, {"question": "Threshold?"}, call_id="call_42")
    assert mock_interrupt.call_args[0][0]["ask_id"] == "call_42"


def test_envelope_answer_reaches_the_model_unchanged_without_comment():
    with patch("langgraph.types.interrupt") as mock_interrupt:
        mock_interrupt.return_value = envelope({"threshold": 0.1})
        result = call(ask_user, {"question": "Threshold?"})
    assert result == json.dumps({"threshold": 0.1})


def test_envelope_comment_reaches_the_model():
    with patch("langgraph.types.interrupt") as mock_interrupt:
        mock_interrupt.return_value = envelope(0.1, comment="low-quality run")
        result = call(ask_user, {"question": "Threshold?"})
    assert json.loads(result) == {"answer": 0.1, "comment": "low-quality run"}


def test_decision_is_logged_with_ir_comment_verbatim_and_source_as_told():
    data_manager = MagicMock()
    tool = create_ask_user_tool(llm=None, data_manager=data_manager)
    comment = 'Cells < 200 genes; "per lab SOP" {{ not jinja }}'
    with patch("langgraph.types.interrupt") as mock_interrupt:
        mock_interrupt.return_value = envelope(0.1, comment=comment, source="model")
        call(tool, {"question": "Which threshold?"})

    data_manager.log_tool_usage.assert_called_once()
    kwargs = data_manager.log_tool_usage.call_args.kwargs
    ir = kwargs["ir"]
    assert ir is not None
    assert kwargs["tool_name"] == "ask_user"
    assert ir.parameters["comment"] == comment
    assert ir.parameters["source"] == "model"
    assert ir.parameters["answer"] == 0.1
    assert ir.parameters["question"] == "Which threshold?"
    assert ir.parameters["decision_id"] == "dec_1"
    assert ir.replayable is False
    # The notebook cell is comments only, one line per field.
    rendered = ir.code_template
    for line in rendered.splitlines():
        assert line.startswith("#")


def test_decision_cell_without_a_channel_is_unchanged():
    from lobster.tools.user_interaction import _decision_ir

    data_manager = MagicMock()
    tool = create_ask_user_tool(llm=None, data_manager=data_manager)
    with patch("langgraph.types.interrupt") as mock_interrupt:
        mock_interrupt.return_value = envelope(0.1, comment="low-quality run")
        call(tool, {"question": "Which threshold?"})
    ir = data_manager.log_tool_usage.call_args.kwargs["ir"]
    assert "channel" not in ir.parameters
    assert ir.render() == (
        "# Decision recorded during the analysis (source: 'human')\n"
        "# Question: 'Which threshold?'\n"
        "# Answer: 0.1\n"
        "# Comment: 'low-quality run'"
    )
    assert _decision_ir("Q", {"source": "model"}).render().splitlines()[0] == (
        "# Decision recorded during the analysis (source: 'model')"
    )


def test_decision_cell_with_a_channel_names_the_mcp_host():
    data_manager = MagicMock()
    tool = create_ask_user_tool(llm=None, data_manager=data_manager)
    with patch("langgraph.types.interrupt") as mock_interrupt:
        mock_interrupt.return_value = envelope(
            {"value": 0.1}, comment="low-quality run", channel="mcp:claude-code"
        )
        call(tool, {"question": "Which threshold?"})
    ir = data_manager.log_tool_usage.call_args.kwargs["ir"]
    assert ir.parameters["channel"] == "mcp:claude-code"
    assert ir.render() == (
        "# Decision recorded during the analysis (source: 'human', as reported "
        "by the MCP host 'claude-code')\n"
        "# Lobster cannot verify who answered through an MCP host.\n"
        "# Question: 'Which threshold?'\n"
        "# Answer: {'value': 0.1}\n"
        "# Comment: 'low-quality run'"
    )
    assert ir.replayable is False
    for line in ir.render().splitlines():
        assert line.startswith("#")


@pytest.mark.parametrize(
    "channel",
    [
        "mcp:x'); import os; os.system('id') #\nimport os",
        "mcp:a\r\nb\u2028c\u2029d\x0be\x85f",
        "mcp:{{ 7*7 }}{% raw %}",
        "mcp:trailing backslash\\",
        "no colon at all\nimport os",
    ],
)
def test_a_hostile_channel_stays_one_comment_line(channel):
    """``channel`` is sanitised by the MCP server, but ``runs.answer`` takes any
    string from other callers: the cell must stay comments only."""
    import ast

    from lobster.tools.user_interaction import _decision_ir

    rendered = _decision_ir("Q?", {"source": "human", "channel": channel}).render()
    for line in rendered.split("\n"):
        assert line.startswith("#"), line
    assert rendered.splitlines() == rendered.split("\n")
    assert ast.parse(rendered).body == []


def test_channel_does_not_change_what_the_model_receives():
    with patch("langgraph.types.interrupt") as mock_interrupt:
        mock_interrupt.return_value = envelope(0.1, channel="mcp:claude-code")
        result = call(ask_user, {"question": "Threshold?"})
    assert result == json.dumps(0.1)


def test_bare_answer_without_envelope_logs_nothing():
    data_manager = MagicMock()
    tool = create_ask_user_tool(llm=None, data_manager=data_manager)
    with patch("langgraph.types.interrupt") as mock_interrupt:
        mock_interrupt.return_value = {"confirmed": True}
        call(tool, {"question": "Proceed?"})
    data_manager.log_tool_usage.assert_not_called()


def test_ask_user_with_options_context():
    """ask_user with options context should produce a select component."""
    with patch("langgraph.types.interrupt") as mock_interrupt:
        mock_interrupt.return_value = {"selected": "CPM", "index": 1}
        call(
            ask_user,
            {
                "question": "Which normalization?",
                "context": {"options": ["DESeq2", "CPM", "TMM"]},
            },
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
    assert tool.metadata["provenance"] is True
    assert tool.tags == ["UTILITY"]


def test_create_ask_user_tool_with_llm_passes_llm_to_mapper():
    """Factory-created tool passes LLM to map_question."""
    mock_llm = MagicMock()
    tool = create_ask_user_tool(llm=mock_llm)

    with (
        patch("langgraph.types.interrupt") as mock_interrupt,
        patch("lobster.services.interaction.component_mapper.map_question") as mock_map,
    ):
        from lobster.services.interaction.component_schemas import ComponentSelection

        mock_map.return_value = ComponentSelection(
            component="text_input",
            data={"question": "test"},
            fallback_prompt="test",
        )
        mock_interrupt.return_value = {"answer": "hello"}
        call(tool, {"question": "How to proceed?", "context": None})

    mock_map.assert_called_once()
    _, kwargs = mock_map.call_args
    assert kwargs.get("llm") is mock_llm


def test_create_ask_user_tool_calls_interrupt_with_selection():
    """Factory-created tool calls interrupt() with ComponentSelection dict."""
    tool = create_ask_user_tool(llm=None)

    with patch("langgraph.types.interrupt") as mock_interrupt:
        mock_interrupt.return_value = {"confirmed": True}
        call(tool, {"question": "Continue?", "context": None})

    mock_interrupt.assert_called_once()
    call_arg = mock_interrupt.call_args[0][0]
    assert isinstance(call_arg, dict)
    assert "component" in call_arg


def test_create_ask_user_tool_no_llm_backward_compat():
    """create_ask_user_tool(llm=None) works like the old standalone ask_user."""
    tool = create_ask_user_tool(llm=None)

    with (
        patch("langgraph.types.interrupt") as mock_interrupt,
        patch("lobster.services.interaction.component_mapper.map_question") as mock_map,
    ):
        from lobster.services.interaction.component_schemas import ComponentSelection

        mock_map.return_value = ComponentSelection(
            component="confirm",
            data={"question": "ok?"},
            fallback_prompt="ok?",
        )
        mock_interrupt.return_value = {"confirmed": True}
        call(tool, {"question": "ok?", "context": None})

    _, kwargs = mock_map.call_args
    assert kwargs.get("llm") is None


def test_decision_cell_renders_comment_lines_only():
    data_manager = MagicMock()
    tool = create_ask_user_tool(llm=None, data_manager=data_manager)
    comment = "two\nlines {{ x }}"
    with patch("langgraph.types.interrupt") as mock_interrupt:
        mock_interrupt.return_value = envelope(0.1, comment=comment)
        call(tool, {"question": "Which threshold?"})
    code = data_manager.log_tool_usage.call_args.kwargs["ir"].render()
    lines = code.splitlines()
    assert len(lines) == 4 and all(line.startswith("#") for line in lines)
    assert repr(comment) in code


# Frozen answer forms (Phase 0 A2): per component kind, the Go TUI's structured
# form and a plain string. The engine does not check the answer against the
# question; it is passed verbatim to the supervisor model, which reads it.
ANSWER_FORMS = [
    ("confirm", "Shall I proceed?", None, {"confirmed": True, "action": "submit"}),
    (
        "select",
        "Which normalization?",
        {"options": ["CPM", "TPM"]},
        {"selected": "TPM", "index": 1, "action": "submit"},
    ),
    (
        "text_input",
        "What should the output folder be called?",
        None,
        {"answer": "run_a", "action": "submit"},
    ),
    (
        "cell_type_selector",
        "Which cell types are these clusters?",
        {"clusters": [{"id": 0, "size": 120, "markers": ["CD3E"]}]},
        {"assignments": {"0": "T cells"}, "action": "submit"},
    ),
    (
        "threshold_slider",
        "Which mitochondrial threshold?",
        {"min": 0, "max": 1, "threshold": 0.05},
        {"value": 0.1, "action": "submit"},
    ),
]


@pytest.mark.parametrize("kind, question, context, structured", ANSWER_FORMS)
@pytest.mark.parametrize("form", ["structured", "plain"])
def test_every_kind_delivers_both_answer_forms_verbatim(
    kind, question, context, structured, form
):
    answer = structured if form == "structured" else "cluster 0 are T cells, 0.1"
    data_manager = MagicMock()
    tool = create_ask_user_tool(llm=None, data_manager=data_manager)
    args = {"question": question}
    if context is not None:
        args["context"] = context
    with patch("langgraph.types.interrupt") as mock_interrupt:
        mock_interrupt.return_value = envelope(answer)
        content = call(tool, args)
    assert mock_interrupt.call_args.args[0]["component"] == kind
    assert content == json.dumps(answer)
    ir = data_manager.log_tool_usage.call_args.kwargs["ir"]
    assert ir.parameters["answer"] == answer


def test_plain_string_answer_to_a_slider_is_recorded_as_given():
    data_manager = MagicMock()
    tool = create_ask_user_tool(llm=None, data_manager=data_manager)
    with patch("langgraph.types.interrupt") as mock_interrupt:
        mock_interrupt.return_value = envelope(
            "about 0.1", comment="as in the pilot", source="model"
        )
        call(
            tool,
            {
                "question": "Which mitochondrial threshold?",
                "context": {"min": 0, "max": 1, "threshold": 0.05},
            },
        )
    assert mock_interrupt.call_args.args[0]["component"] == "threshold_slider"
    params = data_manager.log_tool_usage.call_args.kwargs["ir"].parameters
    assert params["answer"] == "about 0.1"
    assert params["comment"] == "as in the pilot"
    assert params["source"] == "model"


@pytest.mark.parametrize("selection", [None, "Just ask them for a number."])
def test_plain_text_component_selection_still_asks_with_ask_id(selection):
    """A model that answers the component selection in plain text yields no
    ComponentSelection; the question still pauses as a text_input decision
    that carries the tool call id as ask_id."""
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = selection
    tool = create_ask_user_tool(llm=llm)
    with patch("langgraph.types.interrupt") as mock_interrupt:
        mock_interrupt.return_value = envelope("HeLa")
        call(tool, {"question": "Which cell line did you use?"}, call_id="call_77")
    value = mock_interrupt.call_args[0][0]
    assert value["ask_id"] == "call_77"
    assert value["component"] == "text_input"
    assert value["data"]["question"] == "Which cell line did you use?"
