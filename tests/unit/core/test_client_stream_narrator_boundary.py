"""
Regression tests for the stream narrator boundary.

These tests verify the event-stream contract between _stream_query() (producer)
and any consumer (Go TUI, web UI, cloud backend). They are frontend-agnostic.

5 bugs captured:
1. Narrator boundary breach — sub-agent content leaks into content_delta events
2. Streaming dedup weakness — no content-hash fallback when message_id is absent
3. unknown_tool telemetry — "unknown_tool" events emitted to operators
4. Corrupted markdown — downstream of #1, interleaved sub-agent JSON in stream
5. Badge duplication — downstream of #1, agent_change events from both messages
   and updates paths

Tests marked with the bug they prove should FAIL against the current code.
"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)

from lobster.core.client import AgentClient
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


def _ai_chunk(content: str, node: str, msg_id: str = "msg-1") -> tuple:
    """Build a (namespace, 'messages', (AIMessageChunk, metadata)) tuple."""
    chunk = AIMessageChunk(content=content, id=msg_id)
    metadata = {"langgraph_node": node}
    return ("", "messages", (chunk, metadata))


def _ai_complete(content: str, node: str, msg_id: str = "msg-1") -> tuple:
    """Build a (namespace, 'messages', (AIMessage, metadata)) tuple for a
    complete (non-chunk) message."""
    msg = AIMessage(content=content, id=msg_id)
    metadata = {"langgraph_node": node}
    return ("", "messages", (msg, metadata))


def _tool_msg(content: str, node: str, tool_call_id: str = "tc-1") -> tuple:
    """Build a ToolMessage stream tuple."""
    msg = ToolMessage(content=content, tool_call_id=tool_call_id)
    metadata = {"langgraph_node": node}
    return ("", "messages", (msg, metadata))


def _updates_event(node: str, payload: dict) -> tuple:
    """Build an updates stream tuple."""
    return ("", "updates", {node: payload})


def _events_of_type(events, event_type):
    return [e for e in events if e.get("type") == event_type]


# ===========================================================================
# TestSingleSpeakerPolicy — only supervisor content reaches the user
# ===========================================================================

class TestSingleSpeakerPolicy:
    """The stream contract: only the supervisor node's AIMessage content
    should be emitted as content_delta events. Sub-agent content belongs
    in the activity/telemetry lane, never the narrative lane."""

    def test_supervisor_content_is_emitted(self, tmp_path):
        """Supervisor AIMessageChunks produce content_delta events."""
        events_in = [
            _ai_chunk("Hello ", "supervisor", "sup-1"),
            _ai_chunk("world", "supervisor", "sup-1"),
        ]
        client = _make_client(tmp_path, events_in)
        events = _collect_events(client)

        deltas = _events_of_type(events, "content_delta")
        assert len(deltas) == 2
        assert deltas[0]["delta"] == "Hello "
        assert deltas[1]["delta"] == "world"

    def test_sub_agent_content_is_silent(self, tmp_path):
        """BUG #1 — Sub-agent AIMessageChunks must NOT produce content_delta.

        Current code emits content_delta for ALL nodes. This test MUST FAIL
        against the current implementation, proving the narrator boundary breach.
        """
        events_in = [
            _ai_chunk("supervisor says hi", "supervisor", "sup-1"),
            _ai_chunk("research internal thought", "research_agent", "ra-1"),
            _ai_chunk("data expert internal", "data_expert_agent", "de-1"),
        ]
        client = _make_client(tmp_path, events_in)
        events = _collect_events(client)

        deltas = _events_of_type(events, "content_delta")
        # Only the supervisor delta should be present
        delta_texts = [d["delta"] for d in deltas]
        assert delta_texts == ["supervisor says hi"], (
            f"Sub-agent content leaked into content_delta stream: {delta_texts}"
        )

    def test_sub_agent_activity_still_tracked(self, tmp_path):
        """Sub-agent nodes still emit agent_change events even when their
        content is filtered from the narrative lane."""
        events_in = [
            _ai_chunk("thinking...", "research_agent", "ra-1"),
        ]
        client = _make_client(tmp_path, events_in)
        events = _collect_events(client)

        agent_changes = _events_of_type(events, "agent_change")
        working = [e for e in agent_changes if e["status"] == "working"]
        assert len(working) >= 1
        assert working[0]["agent"] == "research_agent"

    def test_tool_messages_always_filtered(self, tmp_path):
        """ToolMessages never produce content_delta, regardless of node."""
        events_in = [
            _tool_msg("tool output from supervisor", "supervisor"),
            _tool_msg("tool output from sub-agent", "research_agent"),
        ]
        client = _make_client(tmp_path, events_in)
        events = _collect_events(client)

        deltas = _events_of_type(events, "content_delta")
        assert len(deltas) == 0


# ===========================================================================
# TestStreamingDedup — message replay suppression
# ===========================================================================

class TestStreamingDedup:
    """LangGraph with subgraphs=True re-emits complete messages at the parent
    graph level after streaming chunks. The dedup logic must catch these."""

    def test_dedup_with_message_id(self, tmp_path):
        """Complete message replay with same ID as prior chunks is suppressed."""
        events_in = [
            _ai_chunk("Hello", "supervisor", "msg-42"),
            _ai_chunk(" world", "supervisor", "msg-42"),
            # LangGraph re-emits the full message at parent level
            _ai_complete("Hello world", "supervisor", "msg-42"),
        ]
        client = _make_client(tmp_path, events_in)
        events = _collect_events(client)

        deltas = _events_of_type(events, "content_delta")
        texts = [d["delta"] for d in deltas]
        # Should have exactly 2 deltas (chunks), not 3 (replay suppressed)
        assert texts == ["Hello", " world"]

    def test_dedup_without_message_id(self, tmp_path):
        """BUG #2 — Content-hash fallback catches ID-less replays.

        When message_id is None, the current code has no dedup mechanism.
        This test MUST FAIL, proving the dedup weakness.
        """
        events_in = [
            _ai_chunk("Hello", "supervisor", None),
            _ai_chunk(" world", "supervisor", None),
            # Replay without ID — current code cannot detect this
            _ai_complete("Hello world", "supervisor", None),
        ]
        client = _make_client(tmp_path, events_in)
        events = _collect_events(client)

        deltas = _events_of_type(events, "content_delta")
        texts = [d["delta"] for d in deltas]
        # Should only have the 2 chunk deltas, not the replayed complete message
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
        """BUG #3 — on_tool_start with missing name emits 'unknown_tool'.

        Current code defaults to "unknown_tool" and emits. This test MUST FAIL.
        """
        emitted = []

        def capture(msg_type, payload):
            emitted.append((msg_type, payload))

        handler = ProtocolCallbackHandler(emit_event=capture)
        # serialized dict without 'name' key
        handler.on_tool_start(serialized={}, input_str="some input")

        tool_events = [
            (t, p) for t, p in emitted
            if t == "tool_execution" and p.get("tool") == "unknown_tool"
        ]
        assert len(tool_events) == 0, (
            f"'unknown_tool' event was emitted to operators: {tool_events}"
        )

    def test_unknown_tool_end_suppressed(self):
        """BUG #3b — on_tool_end with no prior on_tool_start emits 'unknown_tool'.

        Current code falls back to "unknown_tool" via self.current_tool. This
        test MUST FAIL.
        """
        emitted = []

        def capture(msg_type, payload):
            emitted.append((msg_type, payload))

        handler = ProtocolCallbackHandler(emit_event=capture)
        # on_tool_end called without prior on_tool_start
        handler.on_tool_end(output="some output")

        tool_events = [
            (t, p) for t, p in emitted
            if t == "tool_execution" and p.get("tool") == "unknown_tool"
        ]
        assert len(tool_events) == 0, (
            f"'unknown_tool' event was emitted on tool_end: {tool_events}"
        )

    def test_real_tool_still_emitted(self):
        """Named tools still emit start + end events normally."""
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
            _ai_chunk("final summary", "supervisor", "sup-1"),
            _ai_chunk("end leak", "__end__", "end-1"),
        ]
        client = _make_client(tmp_path, events_in)
        events = _collect_events(client)

        deltas = _events_of_type(events, "content_delta")
        texts = [d["delta"] for d in deltas]
        assert texts == ["final summary"]
