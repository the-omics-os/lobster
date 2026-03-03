"""Integration tests for context management across delegation + store + trimming."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.store.memory import InMemoryStore


class TestDualWriteInDelegation:
    """Test that delegation tools store results and append store_key."""

    def test_agent_tool_stores_result(self):
        """Verify _create_agent_tool stores content when store is provided."""
        from unittest.mock import MagicMock

        from lobster.agents.graph import _create_agent_tool

        store = InMemoryStore()

        # Create a mock agent that returns a fixed response
        mock_agent = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "Full analysis: 50K chars of data here"
        mock_agent.invoke.return_value = {"messages": [mock_msg]}

        tool_fn = _create_agent_tool(
            agent_name="test_agent",
            agent=mock_agent,
            tool_name="handoff_to_test_agent",
            description="Test agent",
            store=store,
        )

        result = tool_fn.invoke({"task_description": "Analyze data"})

        # Result should contain store_key reference
        assert "[store_key=" in result
        assert "Full analysis: 50K chars of data here" in result

        # Store should have the full content
        ns = store.search(("agent_results",))
        assert len(ns) == 1
        assert ns[0].value["content"] == "Full analysis: 50K chars of data here"
        assert ns[0].value["agent"] == "test_agent"

    def test_agent_tool_no_store_graceful(self):
        """Verify _create_agent_tool works without store (backward compat)."""
        from unittest.mock import MagicMock

        from lobster.agents.graph import _create_agent_tool

        mock_agent = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "Analysis result"
        mock_agent.invoke.return_value = {"messages": [mock_msg]}

        tool_fn = _create_agent_tool(
            agent_name="test_agent",
            agent=mock_agent,
            tool_name="handoff_to_test",
            description="Test",
            store=None,
        )

        result = tool_fn.invoke({"task_description": "Do something"})
        assert "Analysis result" in result
        assert "[store_key=" not in result

    def test_lazy_delegation_tool_stores_result(self):
        """Verify _create_lazy_delegation_tool stores content when store is provided."""
        from unittest.mock import MagicMock

        from lobster.agents.graph import _create_lazy_delegation_tool

        store = InMemoryStore()
        agents_dict = {}

        # Create the tool before the agent exists (lazy resolution)
        tool_fn = _create_lazy_delegation_tool(
            agent_name="child_agent",
            agents_dict=agents_dict,
            description="Analyzes things",
            store=store,
        )

        # Register the mock agent after tool creation
        mock_agent = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "Detailed child analysis result"
        mock_agent.invoke.return_value = {"messages": [mock_msg]}
        agents_dict["child_agent"] = mock_agent

        result = tool_fn.invoke({"task_description": "Run child analysis"})

        assert "[store_key=" in result
        assert "Detailed child analysis result" in result

        ns = store.search(("agent_results",))
        assert len(ns) == 1
        assert ns[0].value["agent"] == "child_agent"

    def test_lazy_delegation_tool_no_store_graceful(self):
        """Verify _create_lazy_delegation_tool works without store."""
        from unittest.mock import MagicMock

        from lobster.agents.graph import _create_lazy_delegation_tool

        agents_dict = {}

        tool_fn = _create_lazy_delegation_tool(
            agent_name="child_agent",
            agents_dict=agents_dict,
            description="Analyzes things",
            store=None,
        )

        mock_agent = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "Child result"
        mock_agent.invoke.return_value = {"messages": [mock_msg]}
        agents_dict["child_agent"] = mock_agent

        result = tool_fn.invoke({"task_description": "Run analysis"})
        assert "Child result" in result
        assert "[store_key=" not in result


class TestRetrievalRoundTrip:
    """Test store -> retrieve round-trip."""

    def test_store_then_retrieve(self):
        from lobster.tools.store_tools import (
            create_retrieve_agent_result_tool,
            store_delegation_result,
        )

        store = InMemoryStore()

        # Simulate delegation storing a result
        key = store_delegation_result(store, "research_agent", "PubMed search results: 42 papers found")

        # Create retrieval tool and invoke
        retrieve_tool = create_retrieve_agent_result_tool(store)
        result = retrieve_tool.invoke({"store_key": key})

        assert "PubMed search results: 42 papers found" in result
        assert "research_agent" in result

    def test_delegation_then_retrieve_end_to_end(self):
        """Full round-trip: delegation tool stores, retrieval tool reads back."""
        import re
        from unittest.mock import MagicMock

        from lobster.agents.graph import _create_agent_tool
        from lobster.tools.store_tools import create_retrieve_agent_result_tool

        store = InMemoryStore()

        # Create delegation tool
        mock_agent = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "Gene list: BRCA1, TP53, EGFR with p-values 0.001, 0.003, 0.01"
        mock_agent.invoke.return_value = {"messages": [mock_msg]}

        delegation_tool = _create_agent_tool(
            agent_name="de_analysis_expert",
            agent=mock_agent,
            tool_name="handoff_to_de_analysis",
            description="DE analysis",
            store=store,
        )

        # Step 1: Delegation stores result
        delegation_result = delegation_tool.invoke({"task_description": "Run DE analysis"})

        # Extract store_key from delegation result
        match = re.search(r"\[store_key=([^\]]+)\]", delegation_result)
        assert match is not None, f"No store_key found in: {delegation_result}"
        store_key = match.group(1)

        # Step 2: Retrieval tool reads it back
        retrieve_tool = create_retrieve_agent_result_tool(store)
        retrieved = retrieve_tool.invoke({"store_key": store_key})

        assert "BRCA1" in retrieved
        assert "de_analysis_expert" in retrieved

    def test_retrieve_nonexistent_key(self):
        """Retrieval tool handles missing keys gracefully."""
        from lobster.tools.store_tools import create_retrieve_agent_result_tool

        store = InMemoryStore()
        retrieve_tool = create_retrieve_agent_result_tool(store)
        result = retrieve_tool.invoke({"store_key": "nonexistent_key_abc123"})

        assert "No results found" in result or "not found" in result.lower()

    def test_multiple_delegations_independent_keys(self):
        """Multiple delegation results get distinct store keys."""
        from unittest.mock import MagicMock

        from lobster.agents.graph import _create_agent_tool

        store = InMemoryStore()

        results = []
        for i in range(3):
            mock_agent = MagicMock()
            mock_msg = MagicMock()
            mock_msg.content = f"Result from agent {i}"
            mock_agent.invoke.return_value = {"messages": [mock_msg]}

            tool_fn = _create_agent_tool(
                agent_name=f"agent_{i}",
                agent=mock_agent,
                tool_name=f"handoff_to_agent_{i}",
                description=f"Agent {i}",
                store=store,
            )
            results.append(tool_fn.invoke({"task_description": f"Task {i}"}))

        # All should have distinct store keys
        ns = store.search(("agent_results",))
        assert len(ns) == 3

        # Each result should reference its own store_key
        for r in results:
            assert "[store_key=" in r


class TestPreModelHookWithRealMessages:
    """Test pre_model_hook behavior with realistic supervisor messages."""

    def test_trims_large_tool_results(self):
        from lobster.agents.context_management import create_supervisor_pre_model_hook

        hook = create_supervisor_pre_model_hook(max_tokens=500)

        # Simulate a session with large tool results
        messages = [
            SystemMessage(content="You are a supervisor."),
            HumanMessage(content="Search for CRISPR papers"),
            AIMessage(content="I'll delegate to research_agent."),
            # Large tool result (simulating sub-agent response)
            HumanMessage(content="x" * 5000),  # Oversized tool result
            AIMessage(content="Found 42 papers. Key findings: ..."),
            HumanMessage(content="Now analyze the first dataset"),
        ]

        result = hook({"messages": messages})
        trimmed = result["llm_input_messages"]

        # Should be trimmed but preserve system + latest messages
        assert len(trimmed) < len(messages)
        assert any(isinstance(m, SystemMessage) for m in trimmed)
        # Latest user message should be preserved
        assert trimmed[-1].content == "Now analyze the first dataset"

    def test_no_trimming_when_within_budget(self):
        from lobster.agents.context_management import create_supervisor_pre_model_hook

        hook = create_supervisor_pre_model_hook(max_tokens=100_000)

        messages = [
            SystemMessage(content="You are a supervisor."),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi, how can I help?"),
            HumanMessage(content="Search for CRISPR papers"),
        ]

        result = hook({"messages": messages})
        trimmed = result["llm_input_messages"]

        assert len(trimmed) == len(messages)

    def test_system_message_always_preserved(self):
        """System message must survive even aggressive trimming."""
        from lobster.agents.context_management import create_supervisor_pre_model_hook

        # Very tight budget forces aggressive trimming
        hook = create_supervisor_pre_model_hook(max_tokens=200)

        messages = [
            SystemMessage(content="You are a supervisor."),
            HumanMessage(content="old message " * 200),
            AIMessage(content="old response " * 200),
            HumanMessage(content="another old " * 200),
            AIMessage(content="another reply " * 200),
            HumanMessage(content="latest question"),
        ]

        result = hook({"messages": messages})
        trimmed = result["llm_input_messages"]

        system_msgs = [m for m in trimmed if isinstance(m, SystemMessage)]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "You are a supervisor."

    def test_hook_returns_llm_input_messages_key(self):
        """Verify non-destructive bypass: returns llm_input_messages, not messages."""
        from lobster.agents.context_management import create_supervisor_pre_model_hook

        hook = create_supervisor_pre_model_hook(max_tokens=100_000)
        result = hook({"messages": [HumanMessage(content="hi")]})

        assert "llm_input_messages" in result
        assert "messages" not in result


class TestEndToEndContextFlow:
    """Test the full flow: delegation stores result, trimming drops it, retrieval recovers it."""

    def test_store_survives_trimming(self):
        """Results stored in InMemoryStore remain accessible even after trimming drops the message."""
        import re
        from unittest.mock import MagicMock

        from lobster.agents.context_management import create_supervisor_pre_model_hook
        from lobster.agents.graph import _create_agent_tool
        from lobster.tools.store_tools import create_retrieve_agent_result_tool

        store = InMemoryStore()

        # Step 1: Delegation stores a large result
        mock_agent = MagicMock()
        mock_msg = MagicMock()
        large_content = "Comprehensive analysis: " + ("data " * 2000)
        mock_msg.content = large_content
        mock_agent.invoke.return_value = {"messages": [mock_msg]}

        delegation_tool = _create_agent_tool(
            agent_name="research_agent",
            agent=mock_agent,
            tool_name="handoff_to_research",
            description="Research",
            store=store,
        )

        delegation_result = delegation_tool.invoke({"task_description": "Search PubMed"})
        match = re.search(r"\[store_key=([^\]]+)\]", delegation_result)
        assert match is not None
        store_key = match.group(1)

        # Step 2: Build a conversation that would be trimmed
        hook = create_supervisor_pre_model_hook(max_tokens=300)

        messages = [
            SystemMessage(content="You are a supervisor."),
            HumanMessage(content="Search for papers"),
            AIMessage(content="Delegating to research_agent..."),
            # The large delegation result as a message (this will be trimmed)
            HumanMessage(content=delegation_result),
            AIMessage(content="Found results. Summary: ..."),
            HumanMessage(content="Give me the full gene list"),
        ]

        result = hook({"messages": messages})
        trimmed = result["llm_input_messages"]

        # The large delegation result message should have been trimmed
        assert len(trimmed) < len(messages)

        # Step 3: But the store still has the full content
        retrieve_tool = create_retrieve_agent_result_tool(store)
        retrieved = retrieve_tool.invoke({"store_key": store_key})

        assert "Comprehensive analysis:" in retrieved
        assert "research_agent" in retrieved
