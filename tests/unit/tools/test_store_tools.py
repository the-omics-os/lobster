"""Tests for store-backed context management tools."""

import pytest
from langgraph.store.memory import InMemoryStore


class TestRetrieveAgentResultTool:
    """Test the supervisor's retrieval tool."""

    def _make_tool(self, store=None):
        from lobster.tools.store_tools import create_retrieve_agent_result_tool

        if store is None:
            store = InMemoryStore()
        return create_retrieve_agent_result_tool(store), store

    def test_has_invoke(self):
        tool_fn, _ = self._make_tool()
        assert hasattr(tool_fn, "invoke")

    def test_has_aquadif_metadata(self):
        tool_fn, _ = self._make_tool()
        assert hasattr(tool_fn, "metadata")
        assert tool_fn.metadata["categories"] == ["UTILITY"]
        assert tool_fn.metadata["provenance"] is False
        assert tool_fn.tags == ["UTILITY"]

    def test_retrieves_stored_result(self):
        store = InMemoryStore()
        store.put(
            ("agent_results",),
            "test_key_123",
            {
                "content": "Full analysis data here",
                "agent": "transcriptomics_expert",
            },
        )
        tool_fn, _ = self._make_tool(store)
        result = tool_fn.invoke({"store_key": "test_key_123"})
        assert "Full analysis data here" in result
        assert "transcriptomics_expert" in result

    def test_returns_not_found_for_missing_key(self):
        tool_fn, _ = self._make_tool()
        result = tool_fn.invoke({"store_key": "nonexistent"})
        assert "No results found" in result

    def test_truncates_large_results(self):
        store = InMemoryStore()
        large_content = "x" * 50_000
        store.put(
            ("agent_results",),
            "big_key",
            {
                "content": large_content,
                "agent": "test_agent",
            },
        )
        tool_fn, _ = self._make_tool(store)
        result = tool_fn.invoke({"store_key": "big_key"})
        assert len(result) < 50_000
        assert "truncated" in result.lower()

    def test_tool_name(self):
        tool_fn, _ = self._make_tool()
        assert tool_fn.name == "retrieve_agent_result"


class TestStoreResultInDelegation:
    """Test the dual-write helper used by delegation tools."""

    def test_stores_content(self):
        from lobster.tools.store_tools import store_delegation_result

        store = InMemoryStore()
        key = store_delegation_result(store, "test_agent", "Full result content")
        assert key is not None
        item = store.get(("agent_results",), key)
        assert item is not None
        assert item.value["content"] == "Full result content"
        assert item.value["agent"] == "test_agent"

    def test_returns_unique_keys(self):
        from lobster.tools.store_tools import store_delegation_result

        store = InMemoryStore()
        key1 = store_delegation_result(store, "agent_a", "result 1")
        key2 = store_delegation_result(store, "agent_a", "result 2")
        assert key1 != key2

    def test_returns_none_when_store_is_none(self):
        from lobster.tools.store_tools import store_delegation_result

        key = store_delegation_result(None, "agent_a", "result")
        assert key is None

    def test_key_contains_agent_name(self):
        from lobster.tools.store_tools import store_delegation_result

        store = InMemoryStore()
        key = store_delegation_result(store, "transcriptomics_expert", "data")
        assert "transcriptomics_expert" in key
