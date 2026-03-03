"""Tests for context management module (pre_model_hook, token counting, budget resolution)."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool


class TestApproximateTokenCount:
    """Test the approximate token counter (chars/4 + 3 heuristic)."""

    def test_empty_string(self):
        from lobster.agents.context_management import approximate_token_count

        assert approximate_token_count("") == 0

    def test_short_string(self):
        from lobster.agents.context_management import approximate_token_count

        # "hello" = 5 chars -> int(5/4 + 3) = int(4.25) = 4
        assert approximate_token_count("hello") == 4

    def test_long_string(self):
        from lobster.agents.context_management import approximate_token_count

        text = "a" * 1000
        result = approximate_token_count(text)
        assert result == int(1000 / 4.0 + 3.0)  # 253

    def test_none_returns_zero(self):
        from lobster.agents.context_management import approximate_token_count

        assert approximate_token_count(None) == 0


class TestMeasureToolSchemaTokens:
    """Test tool schema token measurement."""

    def test_single_tool(self):
        from lobster.agents.context_management import measure_tool_schema_tokens

        @tool
        def my_tool(query: str) -> str:
            """Search for something."""
            return query

        tokens = measure_tool_schema_tokens([my_tool])
        assert tokens > 0

    def test_empty_tools(self):
        from lobster.agents.context_management import measure_tool_schema_tokens

        assert measure_tool_schema_tokens([]) == 0

    def test_multiple_tools(self):
        from lobster.agents.context_management import measure_tool_schema_tokens

        @tool
        def tool_a(x: str) -> str:
            """Tool A."""
            return x

        @tool
        def tool_b(x: str, y: int) -> str:
            """Tool B with more params."""
            return f"{x}{y}"

        single = measure_tool_schema_tokens([tool_a])
        double = measure_tool_schema_tokens([tool_a, tool_b])
        assert double > single


class TestResolveContextBudget:
    """Test context budget resolution."""

    def test_default_budget(self):
        from lobster.agents.context_management import (
            DEFAULT_CONTEXT_WINDOW,
            resolve_context_budget,
        )

        budget = resolve_context_budget()
        assert 0 < budget < DEFAULT_CONTEXT_WINDOW

    def test_explicit_context_window(self):
        from lobster.agents.context_management import resolve_context_budget

        budget = resolve_context_budget(context_window=100_000)
        assert budget < 100_000
        assert budget > 0

    def test_tools_reduce_budget(self):
        from lobster.agents.context_management import resolve_context_budget

        @tool
        def my_tool(x: str) -> str:
            """A tool."""
            return x

        no_tools = resolve_context_budget(context_window=100_000)
        with_tools = resolve_context_budget(context_window=100_000, tools=[my_tool])
        assert with_tools < no_tools

    def test_minimum_budget_floor(self):
        from lobster.agents.context_management import resolve_context_budget

        budget = resolve_context_budget(context_window=100)
        assert budget >= 4096

    def test_small_model_budget(self):
        from lobster.agents.context_management import resolve_context_budget

        budget = resolve_context_budget(context_window=40_000)
        assert budget > 4096
        assert budget < 40_000


class TestCreateSupervisorPreModelHook:
    """Test the pre_model_hook factory."""

    def test_returns_callable(self):
        from lobster.agents.context_management import create_supervisor_pre_model_hook

        hook = create_supervisor_pre_model_hook(max_tokens=100_000)
        assert callable(hook)

    def test_empty_messages(self):
        from lobster.agents.context_management import create_supervisor_pre_model_hook

        hook = create_supervisor_pre_model_hook(max_tokens=100_000)
        result = hook({"messages": []})
        assert "llm_input_messages" in result
        assert result["llm_input_messages"] == []

    def test_no_trimming_when_under_budget(self):
        from lobster.agents.context_management import create_supervisor_pre_model_hook

        hook = create_supervisor_pre_model_hook(max_tokens=100_000)
        messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="Hello"),
        ]
        result = hook({"messages": messages})
        assert len(result["llm_input_messages"]) == len(messages)

    def test_trims_when_over_budget(self):
        from lobster.agents.context_management import create_supervisor_pre_model_hook

        hook = create_supervisor_pre_model_hook(max_tokens=100)
        messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="First message " * 100),
            AIMessage(content="First response " * 100),
            HumanMessage(content="Second message " * 100),
            AIMessage(content="Second response " * 100),
            HumanMessage(content="Latest question"),
        ]
        result = hook({"messages": messages})
        trimmed = result["llm_input_messages"]
        assert len(trimmed) < len(messages)

    def test_preserves_system_message(self):
        from lobster.agents.context_management import create_supervisor_pre_model_hook

        hook = create_supervisor_pre_model_hook(max_tokens=100)
        messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="x " * 500),
            AIMessage(content="y " * 500),
            HumanMessage(content="Latest"),
        ]
        result = hook({"messages": messages})
        trimmed = result["llm_input_messages"]
        assert any(isinstance(m, SystemMessage) for m in trimmed)

    def test_keeps_most_recent_messages(self):
        from lobster.agents.context_management import create_supervisor_pre_model_hook

        hook = create_supervisor_pre_model_hook(max_tokens=200)
        messages = [
            SystemMessage(content="Sys"),
            HumanMessage(content="old " * 200),
            AIMessage(content="old reply " * 200),
            HumanMessage(content="recent question"),
        ]
        result = hook({"messages": messages})
        trimmed = result["llm_input_messages"]
        contents = [m.content for m in trimmed if isinstance(m, HumanMessage)]
        assert "recent question" in contents

    def test_returns_llm_input_messages_key(self):
        """Verify hook uses non-destructive llm_input_messages bypass."""
        from lobster.agents.context_management import create_supervisor_pre_model_hook

        hook = create_supervisor_pre_model_hook(max_tokens=100_000)
        result = hook({"messages": [HumanMessage(content="hi")]})
        assert "llm_input_messages" in result
        assert "messages" not in result
