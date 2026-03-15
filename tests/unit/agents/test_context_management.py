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
        assert result["context_compaction"]["before_count"] == len(messages)
        assert result["context_compaction"]["after_count"] == len(trimmed)
        assert result["context_compaction"]["budget_tokens"] == 100

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
        assert "context_compaction" not in result


class TestResolveContextWindow:
    """Test context window resolution with provider/model overrides."""

    def test_both_overrides_skip_resolver(self, monkeypatch):
        """Bug #1: When both overrides are given, resolver should not be called."""
        from lobster.agents.context_management import resolve_context_window

        class _FakeProvider:
            def get_model_info(self, model_id):
                from types import SimpleNamespace

                if model_id == "test-model":
                    return SimpleNamespace(context_window=32768)
                return None

        from lobster.config import providers

        monkeypatch.setattr(
            providers,
            "get_provider",
            lambda name: _FakeProvider() if name == "test-provider" else None,
        )

        result = resolve_context_window(
            provider_override="test-provider",
            model_override="test-model",
        )
        assert result == 32768

    def test_both_overrides_no_config_required(self, monkeypatch):
        """Bug #1 core: overrides must work even when no persistent config exists."""
        from lobster.agents.context_management import resolve_context_window

        class _FakeProvider:
            def get_model_info(self, model_id):
                from types import SimpleNamespace

                return SimpleNamespace(context_window=40960)

        from lobster.config import providers

        monkeypatch.setattr(providers, "get_provider", lambda name: _FakeProvider())

        # ConfigResolver.resolve_provider() would throw — but should never be called
        result = resolve_context_window(
            provider_override="ollama",
            model_override="qwen3:8b",
        )
        assert result == 40960

    def test_decision_source_not_used_as_model_id(self, monkeypatch):
        """Bug #2: resolve_provider returns (name, decision_source), not (name, model_id)."""
        from lobster.agents.context_management import resolve_context_window

        call_log = []

        class _FakeProvider:
            def get_model_info(self, model_id):
                call_log.append(model_id)
                from types import SimpleNamespace

                if model_id == "real-model":
                    return SimpleNamespace(context_window=16384)
                return None

        class _FakeResolver:
            @classmethod
            def get_instance(cls, path):
                return cls()

            def resolve_provider(self):
                return (
                    "test-provider",
                    "workspace_config",
                )  # second element is source, NOT model

        from lobster.config import providers
        from lobster.core import config_resolver

        monkeypatch.setattr(config_resolver, "ConfigResolver", _FakeResolver)
        monkeypatch.setattr(providers, "get_provider", lambda name: _FakeProvider())

        # With model_override, decision_source should NOT be passed to get_model_info
        result = resolve_context_window(model_override="real-model")
        assert result == 16384
        assert (
            "workspace_config" not in call_log
        ), "decision_source was incorrectly used as model_id"

    def test_no_overrides_no_config_returns_none_with_warning(
        self, monkeypatch, caplog
    ):
        """Bug #3: should warn (not silently swallow) when resolution fails."""
        import logging

        from lobster.agents.context_management import resolve_context_window
        from lobster.core.config_resolver import ConfigurationError

        class _FakeResolver:
            @classmethod
            def get_instance(cls, path):
                return cls()

            def resolve_provider(self):
                raise ConfigurationError("No provider configured.")

        from lobster.core import config_resolver

        monkeypatch.setattr(config_resolver, "ConfigResolver", _FakeResolver)

        with caplog.at_level(
            logging.WARNING, logger="lobster.agents.context_management"
        ):
            result = resolve_context_window()

        assert result is None
        assert any("Could not resolve provider" in r.message for r in caplog.records)

    def test_bogus_provider_returns_none(self, monkeypatch):
        from lobster.agents.context_management import resolve_context_window
        from lobster.config import providers

        monkeypatch.setattr(providers, "get_provider", lambda name: None)

        result = resolve_context_window(
            provider_override="nonexistent",
            model_override="foo",
        )
        assert result is None

    def test_provider_override_only_falls_back_to_resolver(self, monkeypatch):
        """Only provider given — should still try resolver for provider, skip model."""
        from lobster.agents.context_management import resolve_context_window

        class _FakeProvider:
            def get_model_info(self, model_id):
                from types import SimpleNamespace

                # model_id will be None since no model_override and resolver doesn't give model
                return (
                    SimpleNamespace(context_window=8192) if model_id is None else None
                )

        from lobster.config import providers

        monkeypatch.setattr(
            providers,
            "get_provider",
            lambda name: _FakeProvider() if name == "ollama" else None,
        )

        result = resolve_context_window(provider_override="ollama")
        assert result == 8192
