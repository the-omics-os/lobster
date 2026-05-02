"""Tests for HITL interrupt detection and resume (Phase 2)."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage

from lobster.core.client import AgentClient


def _make_client_with_stream_events(tmp_path, stream_events):
    """Shared helper: create an AgentClient with a mocked graph."""
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


# ---------------------------------------------------------------------------
# 2.1 __interrupt__ detection in _stream_query
# ---------------------------------------------------------------------------


def test_interrupt_in_updates_stream_yields_interrupt_event(tmp_path):
    """__interrupt__ in updates stream -> yields interrupt event and stops."""
    interrupt_obj = SimpleNamespace(
        value={"component": "confirm", "data": {"question": "Proceed?"}},
        id="intr-001",
    )
    events = [
        (
            (),
            "messages",
            (
                AIMessage(content="thinking...", id="msg-1"),
                {"langgraph_node": "supervisor"},
            ),
        ),
        (
            (),
            "updates",
            {"__interrupt__": [interrupt_obj]},
        ),
    ]

    client = _make_client_with_stream_events(tmp_path, events)
    # Mock get_state to avoid lingering interrupt check interfering.
    client.graph.get_state = Mock(return_value=SimpleNamespace(tasks=None))

    result = list(client.query("test", stream=True))

    interrupt_events = [e for e in result if e["type"] == "interrupt"]
    assert len(interrupt_events) == 1
    assert interrupt_events[0]["data"]["component"] == "confirm"
    assert interrupt_events[0]["interrupt_id"] == "intr-001"

    # Stream should end at interrupt — no "complete" event.
    complete_events = [e for e in result if e["type"] == "complete"]
    assert len(complete_events) == 0


def test_interrupt_with_multiple_objects(tmp_path):
    """Multiple interrupt objects in a single __interrupt__ list."""
    intr1 = SimpleNamespace(value={"component": "confirm"}, id="intr-1")
    intr2 = SimpleNamespace(value={"component": "select"}, id="intr-2")

    events = [
        ((), "updates", {"__interrupt__": [intr1, intr2]}),
    ]

    client = _make_client_with_stream_events(tmp_path, events)
    client.graph.get_state = Mock(return_value=SimpleNamespace(tasks=None))

    result = list(client.query("test", stream=True))

    interrupt_events = [e for e in result if e["type"] == "interrupt"]
    assert len(interrupt_events) == 2
    assert interrupt_events[0]["interrupt_id"] == "intr-1"
    assert interrupt_events[1]["interrupt_id"] == "intr-2"


def test_lingering_interrupt_detected_after_stream(tmp_path):
    """Interrupt that didn't appear in stream is caught via get_state."""
    # Normal stream with no __interrupt__ in events.
    events = [
        (
            (),
            "messages",
            (
                AIMessage(content="done", id="msg-1"),
                {"langgraph_node": "supervisor"},
            ),
        ),
    ]

    client = _make_client_with_stream_events(tmp_path, events)

    # Simulate lingering interrupt in graph state.
    lingering_intr = SimpleNamespace(
        value={"component": "text_input", "data": {"question": "Name?"}},
        id="intr-linger",
    )
    task = SimpleNamespace(interrupts=[lingering_intr])
    client.graph.get_state = Mock(return_value=SimpleNamespace(tasks=[task]))

    result = list(client.query("test", stream=True))

    interrupt_events = [e for e in result if e["type"] == "interrupt"]
    assert len(interrupt_events) == 1
    assert interrupt_events[0]["interrupt_id"] == "intr-linger"

    # No complete event since we returned on interrupt.
    complete_events = [e for e in result if e["type"] == "complete"]
    assert len(complete_events) == 0


def test_no_lingering_interrupt_normal_completion(tmp_path):
    """Normal completion when get_state has no tasks/interrupts."""
    events = [
        (
            (),
            "messages",
            (
                AIMessage(content="all good", id="msg-1"),
                {"langgraph_node": "supervisor"},
            ),
        ),
    ]

    client = _make_client_with_stream_events(tmp_path, events)
    client.graph.get_state = Mock(return_value=SimpleNamespace(tasks=None))

    result = list(client.query("test", stream=True))

    interrupt_events = [e for e in result if e["type"] == "interrupt"]
    assert len(interrupt_events) == 0

    complete_events = [e for e in result if e["type"] == "complete"]
    assert len(complete_events) == 1


# ---------------------------------------------------------------------------
# 2.2 resume_from_interrupt
# ---------------------------------------------------------------------------


def test_resume_from_interrupt_streams_resumed_events(tmp_path):
    """resume_from_interrupt() passes Command(resume=...) to graph."""
    # First query triggers interrupt.
    interrupt_obj = SimpleNamespace(value={"component": "confirm"}, id="intr-001")
    initial_events = [
        ((), "updates", {"__interrupt__": [interrupt_obj]}),
    ]

    client = _make_client_with_stream_events(tmp_path, initial_events)
    client.graph.get_state = Mock(return_value=SimpleNamespace(tasks=None))

    # Trigger initial query.
    result = list(client.query("test", stream=True))
    assert any(e["type"] == "interrupt" for e in result)

    # Now set up resumed stream.
    resumed_events = [
        (
            (),
            "messages",
            (
                AIMessage(content="Resumed!", id="msg-resumed"),
                {"langgraph_node": "supervisor"},
            ),
        ),
    ]

    def _resumed_stream(*args, **kwargs):
        for e in resumed_events:
            yield e

    client.graph.stream.side_effect = _resumed_stream
    client.graph.get_state = Mock(return_value=SimpleNamespace(tasks=None))

    resumed_result = list(
        client.resume_from_interrupt({"confirmed": True}, stream=True)
    )

    content_events = [e for e in resumed_result if e["type"] == "content_delta"]
    assert len(content_events) == 1
    assert content_events[0]["delta"] == "Resumed!"

    # Verify graph.stream was called with Command input.
    call_args = client.graph.stream.call_args
    from langgraph.types import Command

    assert isinstance(call_args[0][0], Command)
