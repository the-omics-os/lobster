"""A reply that ended at the output-token limit is never presented as complete.

The model's final call can end with ``stopReason == "max_tokens"`` in the middle
of a sentence or table row. The turn then ends with a plain note saying the
answer was cut off at the length limit, in both the streamed and the one-shot
path, so neither the reader nor the saved history takes the cut text for a full
answer.
"""

from langchain_core.messages import AIMessage, AIMessageChunk

from lobster.config.output_bound import truncation_note
from tests.unit.core.test_client_stream_narrator_boundary import (
    _collect_events,
    _events_of_type,
    _make_client,
)


def _final_chunk(stop, key="stopReason", msg_id="sup-1"):
    chunk = AIMessageChunk(content="", id=msg_id, response_metadata={key: stop})
    return ((), "messages", (chunk, {"langgraph_node": "agent"}))


def _text_chunk(text, msg_id="sup-1"):
    chunk = AIMessageChunk(content=text, id=msg_id)
    return ((), "messages", (chunk, {"langgraph_node": "agent"}))


def test_streamed_reply_cut_at_the_limit_ends_with_the_note(tmp_path):
    client = _make_client(
        tmp_path,
        [_text_chunk("| gene | logFC |\n| TP53 |"), _final_chunk("max_tokens")],
    )
    events = _collect_events(client)

    complete = _events_of_type(events, "complete")[0]
    assert complete["response"].startswith("| gene | logFC |\n| TP53 |")
    assert complete["response"].endswith(truncation_note())
    deltas = "".join(e["delta"] for e in _events_of_type(events, "content_delta"))
    assert deltas.endswith(truncation_note())
    assert client.messages[-1].content.endswith(truncation_note())


def test_streamed_reply_with_the_other_provider_key(tmp_path):
    client = _make_client(
        tmp_path,
        [_text_chunk("cut"), _final_chunk("max_tokens", key="stop_reason")],
    )
    complete = _events_of_type(_collect_events(client), "complete")[0]
    assert complete["response"].endswith(truncation_note())


def test_streamed_reply_that_ended_normally_has_no_note(tmp_path):
    client = _make_client(tmp_path, [_text_chunk("done"), _final_chunk("end_turn")])
    complete = _events_of_type(_collect_events(client), "complete")[0]
    assert complete["response"] == "done"


def test_a_cut_call_followed_by_a_complete_one_has_no_note(tmp_path):
    client = _make_client(
        tmp_path,
        [
            _text_chunk("partial", "sup-1"),
            _final_chunk("max_tokens", msg_id="sup-1"),
            _text_chunk("the whole answer", "sup-2"),
            _final_chunk("end_turn", msg_id="sup-2"),
        ],
    )
    complete = _events_of_type(_collect_events(client), "complete")[0]
    assert truncation_note() not in complete["response"]


def test_one_shot_reply_cut_at_the_limit_ends_with_the_note(tmp_path):
    client = _make_client(tmp_path, [])
    message = AIMessage(
        content="| gene | logFC |\n| TP53 |",
        response_metadata={"stopReason": "max_tokens"},
    )
    events = [{"supervisor": {"messages": [message]}}]
    text = client._extract_response(events)
    assert text.startswith("| gene | logFC |\n| TP53 |")
    assert text.endswith(truncation_note())


def test_one_shot_reply_that_ended_normally_is_unchanged(tmp_path):
    client = _make_client(tmp_path, [])
    message = AIMessage(content="done", response_metadata={"stopReason": "end_turn"})
    assert client._extract_response([{"supervisor": {"messages": [message]}}]) == "done"
