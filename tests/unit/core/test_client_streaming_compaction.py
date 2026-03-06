import threading
from unittest.mock import Mock, call, patch

from langchain_core.messages import AIMessage, HumanMessage

from lobster.core.client import AgentClient


def _make_client_with_stream_events(tmp_path, stream_events):
    mock_graph = Mock()

    def _stream(*args, **kwargs):
        for event in stream_events:
            yield event

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


def test_stream_query_emits_context_compaction_event(tmp_path):
    events = [
        (
            ("supervisor:abc",),
            "updates",
            {
                "supervisor": {
                    "context_compaction": {
                        "before_count": 48,
                        "after_count": 21,
                        "budget_tokens": 8192,
                    }
                }
            },
        ),
        (
            ("supervisor:abc",),
            "messages",
            (
                AIMessage(content="done", id="msg-1"),
                {"langgraph_node": "supervisor"},
            ),
        ),
    ]

    client = _make_client_with_stream_events(tmp_path, events)
    result = list(client.query("analyze", stream=True))

    compaction_events = [e for e in result if e["type"] == "context_compaction"]
    assert len(compaction_events) == 1
    assert compaction_events[0]["agent"] == "supervisor"
    assert compaction_events[0]["before_count"] == 48
    assert compaction_events[0]["after_count"] == 21
    assert compaction_events[0]["budget_tokens"] == 8192


def test_stream_query_dedupes_repeated_context_compaction_event(tmp_path):
    repeated_payload = {
        "supervisor": {
            "context_compaction": {
                "before_count": 48,
                "after_count": 21,
                "budget_tokens": 8192,
            }
        }
    }
    events = [
        (("supervisor:abc",), "updates", repeated_payload),
        (("supervisor:abc",), "updates", repeated_payload),
        (
            ("supervisor:abc",),
            "messages",
            (
                AIMessage(content="done", id="msg-1"),
                {"langgraph_node": "supervisor"},
            ),
        ),
    ]

    client = _make_client_with_stream_events(tmp_path, events)
    result = list(client.query("analyze", stream=True))

    compaction_events = [e for e in result if e["type"] == "context_compaction"]
    assert len(compaction_events) == 1


# ---------------------------------------------------------------------------
# Cancel tests (Phase 1: Query Cancellation)
# ---------------------------------------------------------------------------


def test_cancel_event_breaks_stream(tmp_path):
    """When cancel_event is set, _stream_query breaks the loop early."""
    events = [
        (
            (),
            "messages",
            (
                AIMessage(content="first chunk", id="msg-1"),
                {"langgraph_node": "supervisor"},
            ),
        ),
        (
            (),
            "messages",
            (
                AIMessage(content="second chunk", id="msg-2"),
                {"langgraph_node": "supervisor"},
            ),
        ),
    ]

    cancel_event = threading.Event()

    client = _make_client_with_stream_events(tmp_path, events)
    client.graph.update_state = Mock()

    result = []
    for event in client.query("test", stream=True, cancel_event=cancel_event):
        result.append(event)
        # Set cancel after first content event — next iteration should break.
        if event["type"] == "content_delta":
            cancel_event.set()

    # Should have gotten at most the first content_delta, no "complete" event.
    content_events = [e for e in result if e["type"] == "content_delta"]
    assert len(content_events) == 1
    assert content_events[0]["delta"] == "first chunk"

    complete_events = [e for e in result if e["type"] == "complete"]
    assert len(complete_events) == 0, "stream should NOT emit 'complete' on cancel"


def test_cancel_saves_partial_state(tmp_path):
    """Cancellation writes accumulated text and system marker to graph state."""
    events = [
        (
            (),
            "messages",
            (
                AIMessage(content="partial response", id="msg-1"),
                {"langgraph_node": "supervisor"},
            ),
        ),
        (
            (),
            "messages",
            (
                AIMessage(content=" never seen", id="msg-2"),
                {"langgraph_node": "supervisor"},
            ),
        ),
    ]

    cancel_event = threading.Event()

    client = _make_client_with_stream_events(tmp_path, events)
    client.graph.update_state = Mock()

    result = []
    for event in client.query("test", stream=True, cancel_event=cancel_event):
        result.append(event)
        if event["type"] == "content_delta":
            cancel_event.set()

    # Verify update_state was called with accumulated text and system message.
    assert client.graph.update_state.call_count == 2
    first_call_messages = client.graph.update_state.call_args_list[0][0][1]["messages"]
    assert len(first_call_messages) == 1
    assert isinstance(first_call_messages[0], AIMessage)
    assert first_call_messages[0].content == "partial response"

    second_call_messages = client.graph.update_state.call_args_list[1][0][1]["messages"]
    assert len(second_call_messages) == 1
    assert isinstance(second_call_messages[0], HumanMessage)
    assert "[SYSTEM] Query cancelled" in second_call_messages[0].content


def test_cancel_preserves_session_messages(tmp_path):
    """After cancellation, client.messages includes partial text + system marker."""
    events = [
        (
            (),
            "messages",
            (
                AIMessage(content="partial", id="msg-1"),
                {"langgraph_node": "supervisor"},
            ),
        ),
    ]

    cancel_event = threading.Event()
    cancel_event.set()  # Pre-set: cancel immediately.

    client = _make_client_with_stream_events(tmp_path, events)
    client.graph.update_state = Mock()

    # Drain the generator.
    list(client.query("test", stream=True, cancel_event=cancel_event))

    # The HumanMessage from query() + system cancel marker should be in messages.
    human_msgs = [m for m in client.messages if isinstance(m, HumanMessage)]
    assert any("[SYSTEM] Query cancelled" in m.content for m in human_msgs)


def test_next_query_works_after_cancel(tmp_path):
    """Session remains usable after a cancelled query."""
    cancel_events_data = [
        (
            (),
            "messages",
            (
                AIMessage(content="will cancel", id="msg-1"),
                {"langgraph_node": "supervisor"},
            ),
        ),
    ]

    cancel_event = threading.Event()
    cancel_event.set()

    client = _make_client_with_stream_events(tmp_path, cancel_events_data)
    client.graph.update_state = Mock()

    # First query: cancelled.
    list(client.query("first", stream=True, cancel_event=cancel_event))

    # Second query: normal (no cancel_event).
    normal_events = [
        (
            (),
            "messages",
            (
                AIMessage(content="normal response", id="msg-2"),
                {"langgraph_node": "supervisor"},
            ),
        ),
    ]

    def _stream2(*args, **kwargs):
        for e in normal_events:
            yield e

    client.graph.stream.side_effect = _stream2

    result = list(client.query("second", stream=True))
    content_events = [e for e in result if e["type"] == "content_delta"]
    assert len(content_events) == 1
    assert content_events[0]["delta"] == "normal response"
    # Complete event should be emitted for non-cancelled query.
    complete_events = [e for e in result if e["type"] == "complete"]
    assert len(complete_events) == 1
