"""
Regression tests for the stream narrator boundary.

These tests verify the event-stream contract between _stream_query() (producer)
and any consumer (Go TUI, web UI, cloud backend). They are frontend-agnostic.

The contract:
- Supervisor content → ``content_delta`` events (user-visible, accumulated)
- Specialist content → ``agent_content`` events (available for future UI lanes)
- Specialist activity → ``agent_change`` events (tool/activity telemetry)
- Tool messages → always filtered
- ``__end__`` node → always filtered
"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)

from lobster.core.client import AgentClient, _classify_stream_source
from lobster.ui.callbacks.protocol_callback import ProtocolCallbackHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(tmp_path, stream_events):
    """Build an AgentClient wired to yield *stream_events* from graph.stream."""
    mock_graph = Mock()

    def _stream(*args, **kwargs):
        yield from stream_events

    mock_graph.stream.side_effect = _stream

    mock_metadata = Mock()
    mock_metadata.subscription_tier = "free"
    mock_metadata.available_agents = []
    mock_metadata.supervisor_accessible_agents = []
    mock_metadata.filtered_out_agents = []
    mock_metadata.to_dict.return_value = {}

    mock_dm = Mock()
    mock_dm.profile_timings_enabled = False
    mock_dm.get_workspace_status.return_value = {}
    mock_dm.has_data.return_value = False
    mock_dm.plot_manager = Mock()
    mock_dm.plot_manager.get_latest_plots.return_value = ([], {}, None)
    mock_dm.provenance = None

    with patch("lobster.core.client.create_bioinformatics_graph") as mock_create:
        mock_create.return_value = (mock_graph, mock_metadata)
        client = AgentClient(data_manager=mock_dm, workspace_path=tmp_path)

    client._save_session_json = lambda: None
    return client


def _collect_events(client):
    """Drive _stream_query and collect all yielded event dicts."""
    graph_input = {"messages": [HumanMessage(content="test")]}
    config = {"configurable": {"thread_id": "test"}}
    return list(client._stream_query(graph_input, config))


def _ai_chunk(
    content: str,
    node: str,
    msg_id: str = "msg-1",
    namespace: tuple = (),
) -> tuple:
    """Build a (namespace, 'messages', (AIMessageChunk, metadata)) tuple.

    *namespace* is the LangGraph subgraph path tuple.  With
    ``subgraphs=True`` the real values are e.g. ``()`` for the top-level
    graph, ``("supervisor",)`` for the supervisor subgraph, and
    ``("research_agent",)`` for a specialist subgraph.
    """
    chunk = AIMessageChunk(content=content, id=msg_id)
    metadata = {"langgraph_node": node}
    return (namespace, "messages", (chunk, metadata))


def _ai_complete(
    content: str,
    node: str,
    msg_id: str = "msg-1",
    namespace: tuple = (),
) -> tuple:
    """Build a complete AIMessage stream tuple."""
    msg = AIMessage(content=content, id=msg_id)
    metadata = {"langgraph_node": node}
    return (namespace, "messages", (msg, metadata))


def _tool_msg(
    content: str,
    node: str,
    tool_call_id: str = "tc-1",
    namespace: tuple = (),
) -> tuple:
    """Build a ToolMessage stream tuple."""
    msg = ToolMessage(content=content, tool_call_id=tool_call_id)
    metadata = {"langgraph_node": node}
    return (namespace, "messages", (msg, metadata))


def _updates_event(node: str, payload: dict) -> tuple:
    """Build an updates stream tuple."""
    return ("", "updates", {node: payload})


def _events_of_type(events, event_type):
    return [e for e in events if e.get("type") == event_type]


# ===========================================================================
# TestClassifyStreamSource — namespace-based source classification
# ===========================================================================

class TestClassifyStreamSource:
    """_classify_stream_source uses the namespace tuple to determine which
    top-level agent owns a stream event."""

    def test_empty_namespace_is_supervisor(self):
        assert _classify_stream_source(()) == "supervisor"

    def test_supervisor_namespace(self):
        assert _classify_stream_source(("supervisor",)) == "supervisor"

    def test_specialist_expert(self):
        assert _classify_stream_source(("transcriptomics_expert",)) == "transcriptomics_expert"

    def test_specialist_agent(self):
        assert _classify_stream_source(("research_agent",)) == "research_agent"

    def test_specialist_assistant(self):
        assert _classify_stream_source(("metadata_assistant",)) == "metadata_assistant"

    def test_non_tuple_fallback(self):
        """Non-tuple namespace (legacy/unexpected) falls back to supervisor."""
        assert _classify_stream_source("") == "supervisor"
        assert _classify_stream_source("some_string") == "supervisor"


# ===========================================================================
# TestSingleSpeakerPolicy — only supervisor content reaches the user
# ===========================================================================

class TestSingleSpeakerPolicy:
    """The stream contract: only the supervisor's AIMessage content is emitted
    as ``content_delta``.  Specialist content goes to ``agent_content``."""

    def test_supervisor_content_is_emitted(self, tmp_path):
        """Supervisor AIMessageChunks produce content_delta events."""
        events_in = [
            _ai_chunk("Hello ", "agent", "sup-1", namespace=("supervisor",)),
            _ai_chunk("world", "agent", "sup-1", namespace=("supervisor",)),
        ]
        client = _make_client(tmp_path, events_in)
        events = _collect_events(client)

        deltas = _events_of_type(events, "content_delta")
        assert len(deltas) == 2
        assert deltas[0]["delta"] == "Hello "
        assert deltas[0]["source"] == "supervisor"
        assert deltas[1]["delta"] == "world"

    def test_sub_agent_content_yields_agent_content(self, tmp_path):
        """Specialist content produces agent_content events (not content_delta)
        so future consumers can opt into showing it."""
        events_in = [
            _ai_chunk("supervisor says hi", "agent", "sup-1", namespace=("supervisor",)),
            _ai_chunk("research internal", "agent", "ra-1", namespace=("research_agent",)),
            _ai_chunk("data expert internal", "agent", "de-1", namespace=("data_expert_agent",)),
        ]
        client = _make_client(tmp_path, events_in)
        events = _collect_events(client)

        # Only supervisor → content_delta
        deltas = _events_of_type(events, "content_delta")
        delta_texts = [d["delta"] for d in deltas]
        assert delta_texts == ["supervisor says hi"]

        # Specialists → agent_content (not discarded, just routed differently)
        agent_content = _events_of_type(events, "agent_content")
        assert len(agent_content) == 2
        assert agent_content[0]["delta"] == "research internal"
        assert agent_content[0]["source"] == "research_agent"
        assert agent_content[1]["delta"] == "data expert internal"
        assert agent_content[1]["source"] == "data_expert_agent"

    def test_sub_agent_activity_still_tracked(self, tmp_path):
        """Sub-agent nodes still emit agent_change events even when their
        content is routed to the agent_content lane."""
        events_in = [
            _ai_chunk("thinking...", "agent", "ra-1", namespace=("research_agent",)),
        ]
        client = _make_client(tmp_path, events_in)
        events = _collect_events(client)

        agent_changes = _events_of_type(events, "agent_change")
        working = [e for e in agent_changes if e["status"] == "working"]
        assert len(working) >= 1
        assert working[0]["agent"] == "research_agent"

    def test_tool_messages_always_filtered(self, tmp_path):
        """ToolMessages never produce content_delta or agent_content."""
        events_in = [
            _tool_msg("tool output", "tools", namespace=("supervisor",)),
            _tool_msg("tool output", "tools", namespace=("research_agent",)),
        ]
        client = _make_client(tmp_path, events_in)
        events = _collect_events(client)

        deltas = _events_of_type(events, "content_delta")
        agent_content = _events_of_type(events, "agent_content")
        assert len(deltas) == 0
        assert len(agent_content) == 0

    def test_top_level_namespace_is_supervisor(self, tmp_path):
        """Messages from the top-level graph (empty namespace) are treated
        as supervisor content — this covers edge cases where the graph
        structure doesn't nest the supervisor in a subgraph."""
        events_in = [
            _ai_chunk("top level", "supervisor", "tl-1", namespace=()),
        ]
        client = _make_client(tmp_path, events_in)
        events = _collect_events(client)

        deltas = _events_of_type(events, "content_delta")
        assert len(deltas) == 1
        assert deltas[0]["delta"] == "top level"


# ===========================================================================
# TestStreamingDedup — message replay suppression
# ===========================================================================

class TestStreamingDedup:
    """LangGraph with subgraphs=True re-emits complete messages at the parent
    graph level after streaming chunks. The dedup logic must catch these."""

    def test_dedup_with_message_id(self, tmp_path):
        """Complete message replay with same ID as prior chunks is suppressed."""
        events_in = [
            _ai_chunk("Hello", "agent", "msg-42", namespace=("supervisor",)),
            _ai_chunk(" world", "agent", "msg-42", namespace=("supervisor",)),
            _ai_complete("Hello world", "agent", "msg-42", namespace=("supervisor",)),
        ]
        client = _make_client(tmp_path, events_in)
        events = _collect_events(client)

        deltas = _events_of_type(events, "content_delta")
        texts = [d["delta"] for d in deltas]
        assert texts == ["Hello", " world"]

    def test_dedup_without_message_id(self, tmp_path):
        """Content-hash fallback catches ID-less replays."""
        events_in = [
            _ai_chunk("Hello", "agent", None, namespace=("supervisor",)),
            _ai_chunk(" world", "agent", None, namespace=("supervisor",)),
            _ai_complete("Hello world", "agent", None, namespace=("supervisor",)),
        ]
        client = _make_client(tmp_path, events_in)
        events = _collect_events(client)

        deltas = _events_of_type(events, "content_delta")
        texts = [d["delta"] for d in deltas]
        assert texts == ["Hello", " world"], (
            f"ID-less replay was not deduplicated: {texts}"
        )


# ===========================================================================
# TestUnknownToolSuppression — "unknown_tool" must not reach operators
# ===========================================================================

class TestUnknownToolSuppression:
    """The ProtocolCallbackHandler defaults tool_name to 'unknown_tool' when
    the serialized dict has no 'name' key or current_tool is None. These
    events should be suppressed rather than forwarded."""

    def test_unknown_tool_start_suppressed(self):
        emitted = []

        def capture(msg_type, payload):
            emitted.append((msg_type, payload))

        handler = ProtocolCallbackHandler(emit_event=capture)
        handler.on_tool_start(serialized={}, input_str="some input")

        tool_events = [
            (t, p) for t, p in emitted
            if t == "tool_execution" and p.get("tool") == "unknown_tool"
        ]
        assert len(tool_events) == 0, (
            f"'unknown_tool' event was emitted to operators: {tool_events}"
        )

    def test_unknown_tool_end_suppressed(self):
        emitted = []

        def capture(msg_type, payload):
            emitted.append((msg_type, payload))

        handler = ProtocolCallbackHandler(emit_event=capture)
        handler.on_tool_end(output="some output")

        tool_events = [
            (t, p) for t, p in emitted
            if t == "tool_execution" and p.get("tool") == "unknown_tool"
        ]
        assert len(tool_events) == 0, (
            f"'unknown_tool' event was emitted on tool_end: {tool_events}"
        )

    def test_real_tool_still_emitted(self):
        emitted = []

        def capture(msg_type, payload):
            emitted.append((msg_type, payload))

        handler = ProtocolCallbackHandler(emit_event=capture)
        handler.on_tool_start(
            serialized={"name": "search_pubmed"},
            input_str="CRISPR",
        )
        handler.on_tool_end(output="3 results found")

        tool_events = [
            p for t, p in emitted if t == "tool_execution"
        ]
        assert len(tool_events) == 2
        assert tool_events[0]["tool"] == "search_pubmed"
        assert tool_events[0]["status"] == "running"
        assert tool_events[1]["tool"] == "search_pubmed"
        assert tool_events[1]["status"] == "complete"


# ===========================================================================
# TestEndNodeFiltered — __end__ node never emits content
# ===========================================================================

class TestEndNodeFiltered:
    """The __end__ sentinel node must be silently dropped."""

    def test_end_node_silent(self, tmp_path):
        """__end__ node content never produces content_delta events."""
        events_in = [
            _ai_chunk("final summary", "agent", "sup-1", namespace=("supervisor",)),
            _ai_chunk("end leak", "__end__", "end-1", namespace=()),
        ]
        client = _make_client(tmp_path, events_in)
        events = _collect_events(client)

        deltas = _events_of_type(events, "content_delta")
        texts = [d["delta"] for d in deltas]
        assert texts == ["final summary"]
