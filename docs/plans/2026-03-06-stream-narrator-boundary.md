# Stream Narrator Boundary Fix — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 5 critical streaming bugs that make Go TUI chat output unreadable: sub-agent content leaking into user stream, duplicate fragments, unknown_tool telemetry, corrupted markdown, and badge duplication.

**Architecture:** The root cause is `_stream_query()` in `core/client.py` emitting `content_delta` events for ALL graph nodes (supervisor + sub-agents). LangGraph's `stream_mode=["messages", "updates"]` with `subgraphs=True` yields AIMessage chunks from every node in every subgraph. The fix enforces a **single-speaker policy**: only the supervisor node's content reaches the user. Sub-agent content stays in the tool/activity telemetry lane (already handled by `ProtocolCallbackHandler`). Secondary fixes harden dedup and suppress garbage telemetry events.

**Tech Stack:** Python 3.12, LangGraph (langgraph-core), LangChain message types, pytest

**Seam Map (read these files before starting):**
| Seam | File | Role |
|------|------|------|
| Content stream | `lobster/core/client.py:322-497` | `_stream_query()` — yields events to Go event loop |
| Event loop | `lobster/cli_internal/go_tui_launcher.py:694-744` | `_handle_user_query()` — maps events to protocol messages |
| Tool telemetry | `lobster/ui/callbacks/protocol_callback.py` | `ProtocolCallbackHandler` — emits tool/agent events |
| Go renderer | `lobster-tui/internal/chat/model.go` | Receives protocol messages, renders to terminal |

---

## Task 1: Regression Test — Capture the Failure Pattern

**Files:**
- Create: `tests/unit/core/test_client_stream_narrator_boundary.py`

**Why first:** We need a failing test that reproduces the exact bugs before writing any fix. This test will assert the contract: only supervisor content reaches the user stream; sub-agent content is silent.

**Step 1: Write the failing test**

```python
"""Regression tests for stream narrator boundary.

Validates the single-speaker policy: only supervisor node content
reaches the user-visible stream. Sub-agent nodes (research_agent,
data_expert, etc.) must NOT emit content_delta events.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, patch

import pytest

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage


# ---------------------------------------------------------------------------
# Helpers: fake LangGraph stream tuples
# ---------------------------------------------------------------------------


def _ai_chunk(content: str, node: str, msg_id: str = "msg-1") -> tuple:
    """Build a (namespace, 'messages', (AIMessageChunk, metadata)) tuple."""
    chunk = AIMessageChunk(content=content, id=msg_id)
    metadata = {"langgraph_node": node}
    return ("", "messages", (chunk, metadata))


def _ai_complete(content: str, node: str, msg_id: str = "msg-1") -> tuple:
    """Build a complete AIMessage tuple (non-chunk)."""
    msg = AIMessage(content=content, id=msg_id)
    metadata = {"langgraph_node": node}
    return ("", "messages", (msg, metadata))


def _tool_msg(content: str, node: str) -> tuple:
    """Build a ToolMessage tuple (should always be filtered)."""
    msg = ToolMessage(content=content, tool_call_id="tc-1")
    metadata = {"langgraph_node": node}
    return ("", "messages", (msg, metadata))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_client(stream_tuples: list) -> Any:
    """Build a minimal mock of AgentClient with a fake graph.stream()."""
    mock_graph = MagicMock()
    mock_graph.stream.return_value = iter(stream_tuples)

    client = MagicMock()
    client.graph = mock_graph
    client.session_id = "test-session"
    client.messages = []
    client.token_tracker = MagicMock()
    client.token_tracker.get_latest_cost.return_value = {}
    client._save_session_json = MagicMock()
    return client


def _collect_stream_events(client: Any) -> List[Dict]:
    """Run _stream_query and collect all yielded events."""
    from lobster.core.client import AgentClient

    # Call _stream_query as an unbound method with our mock
    events = list(AgentClient._stream_query(client, {}, {}))
    return events


def _content_deltas(events: List[Dict]) -> List[str]:
    """Extract content_delta text from events."""
    return [e["delta"] for e in events if e["type"] == "content_delta"]


def _agent_changes(events: List[Dict]) -> List[str]:
    """Extract agent names from agent_change events."""
    return [e["agent"] for e in events if e["type"] == "agent_change"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingleSpeakerPolicy:
    """Only supervisor content reaches user stream."""

    def test_supervisor_content_is_emitted(self):
        """Supervisor AIMessageChunks produce content_delta events."""
        tuples = [
            _ai_chunk("Hello from supervisor", "supervisor", "sup-1"),
        ]
        client = _make_mock_client(tuples)
        events = _collect_stream_events(client)
        deltas = _content_deltas(events)
        assert deltas == ["Hello from supervisor"]

    def test_sub_agent_content_is_silent(self):
        """Sub-agent AIMessageChunks must NOT produce content_delta events."""
        tuples = [
            _ai_chunk("internal reasoning from research", "research_agent", "ra-1"),
            _ai_chunk("```json\n{\"modality\": \"scrna_10x\"}\n```", "data_expert_agent", "de-1"),
            _ai_chunk("User-facing answer", "supervisor", "sup-1"),
        ]
        client = _make_mock_client(tuples)
        events = _collect_stream_events(client)
        deltas = _content_deltas(events)
        # ONLY supervisor content should appear
        assert deltas == ["User-facing answer"]

    def test_sub_agent_activity_still_tracked(self):
        """Sub-agent nodes still emit agent_change events for the activity lane."""
        tuples = [
            _ai_chunk("internal work", "research_agent", "ra-1"),
            _ai_chunk("answer", "supervisor", "sup-1"),
        ]
        client = _make_mock_client(tuples)
        events = _collect_stream_events(client)
        agents = _agent_changes(events)
        assert "research_agent" in agents

    def test_tool_messages_always_filtered(self):
        """ToolMessages are never emitted regardless of node."""
        tuples = [
            _tool_msg("tool output", "supervisor"),
            _tool_msg("tool output", "research_agent"),
            _ai_chunk("answer", "supervisor", "sup-1"),
        ]
        client = _make_mock_client(tuples)
        events = _collect_stream_events(client)
        deltas = _content_deltas(events)
        assert deltas == ["answer"]


class TestStreamingDedup:
    """Dedup prevents content duplication."""

    def test_dedup_with_message_id(self):
        """Complete message replay with same ID is suppressed."""
        tuples = [
            _ai_chunk("streamed", "supervisor", "msg-1"),
            _ai_complete("streamed", "supervisor", "msg-1"),  # replay
        ]
        client = _make_mock_client(tuples)
        events = _collect_stream_events(client)
        deltas = _content_deltas(events)
        assert deltas == ["streamed"]

    def test_dedup_without_message_id(self):
        """Content-hash fallback dedup catches ID-less replays."""
        tuples = [
            _ai_chunk("duplicated text", "supervisor", ""),
            _ai_complete("duplicated text", "supervisor", ""),
        ]
        client = _make_mock_client(tuples)
        events = _collect_stream_events(client)
        deltas = _content_deltas(events)
        # Should appear at most once
        assert deltas.count("duplicated text") == 1


class TestEndNodeFiltered:
    """__end__ and __start__ nodes never produce content."""

    def test_end_node_silent(self):
        tuples = [
            _ai_chunk("phantom", "__end__", "end-1"),
            _ai_chunk("real", "supervisor", "sup-1"),
        ]
        client = _make_mock_client(tuples)
        events = _collect_stream_events(client)
        deltas = _content_deltas(events)
        assert deltas == ["real"]
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/tyo/Omics-OS/lobster && uv run pytest tests/unit/core/test_client_stream_narrator_boundary.py -v`

Expected: `test_sub_agent_content_is_silent` FAILS (currently sub-agent content leaks through). `test_dedup_without_message_id` FAILS (no content-hash fallback exists).

**Step 3: Commit the failing tests**

```bash
git add tests/unit/core/test_client_stream_narrator_boundary.py
git commit -m "test: add failing regression tests for stream narrator boundary breach"
```

---

## Task 2: Enforce Single-Speaker Policy in `_stream_query()`

**Files:**
- Modify: `lobster/core/client.py:377-425` (inside `_stream_query`, the messages handler)

**What changes:** Gate content emission on `node_name == "supervisor"`. Sub-agent nodes still emit `agent_change` events (activity lane), but their AIMessage content is silenced. This is the primary fix — it resolves Bugs 1, 4, and 5.

**Step 1: Apply the single-speaker gate**

In `_stream_query()`, after the `node_name == "__end__"` check (line 384) and after the agent_change emission block (lines 386-397), add a supervisor-only gate before the content extraction block (lines 399-425).

The modified section (lines 382-425) becomes:

```python
                    # Get node name from metadata for agent tracking
                    node_name = metadata.get("langgraph_node", "")
                    if node_name == "__end__":
                        continue

                    # Emit agent change events for specialist agents
                    if node_name != last_agent and node_name != "supervisor":
                        if node_name.endswith("_expert") or node_name.endswith(
                            "_agent"
                        ) or node_name.endswith("_assistant"):
                            yield {
                                "type": "agent_change",
                                "agent": node_name,
                                "status": "working",
                                "timestamp": datetime.now().isoformat(),
                            }
                            last_agent = node_name

                    # ── SINGLE-SPEAKER POLICY ──
                    # Only supervisor content is user-visible. Sub-agent
                    # content stays in tool/activity lanes (handled by
                    # ProtocolCallbackHandler). This prevents internal
                    # reasoning, JSON fragments, and tool outputs from
                    # leaking into the user-facing stream.
                    if node_name != "supervisor":
                        continue

                    # Extract text content from message
                    content = message_chunk.content
                    # ... rest unchanged ...
```

**Step 2: Run the narrator boundary tests**

Run: `cd /Users/tyo/Omics-OS/lobster && uv run pytest tests/unit/core/test_client_stream_narrator_boundary.py::TestSingleSpeakerPolicy -v`

Expected: `test_supervisor_content_is_emitted` PASS, `test_sub_agent_content_is_silent` PASS, `test_sub_agent_activity_still_tracked` PASS.

**Step 3: Run existing client tests to check for regressions**

Run: `cd /Users/tyo/Omics-OS/lobster && uv run pytest tests/unit/core/test_client.py tests/unit/core/test_client_streaming_compaction.py -v`

Expected: All pass. If any existing test assumes sub-agent content reaches the stream, update that test to reflect the correct contract.

**Step 4: Commit**

```bash
git add lobster/core/client.py
git commit -m "fix(critical): enforce single-speaker policy — only supervisor content reaches user stream"
```

---

## Task 3: Harden Streaming Dedup (Content-Hash Fallback)

**Files:**
- Modify: `lobster/core/client.py:342-376` (dedup logic in `_stream_query`)

**What changes:** Add a content-hash based fallback dedup for messages without `message_id`. This catches provider-specific edge cases where LangGraph re-emits complete messages without consistent IDs.

**Step 1: Add content-hash dedup**

Add a `seen_content_hashes` set alongside the existing `seen_message_ids`. When a complete (non-chunk) message arrives without a message_id, hash its content and check for duplicates.

Modified dedup section (after line 343):

```python
            seen_message_ids: set = set()
            seen_content_hashes: set = set()
            seen_compaction_signatures: set[tuple[str, Any, Any, Any]] = set()
```

Modified dedup logic (lines 365-376):

```python
                    # Dedup: skip complete messages we already streamed as chunks.
                    # When subgraphs=True, LangGraph re-emits the subgraph's final
                    # complete message at the parent graph level. Without dedup,
                    # the entire response appears twice.
                    if not is_chunk:
                        # Primary dedup: message ID (when available)
                        if message_id and message_id in seen_message_ids:
                            logger.debug(
                                f"Streaming: dedup skipped complete msg {message_id}"
                            )
                            continue
                        # Fallback dedup: content hash (for ID-less replays)
                        raw = message_chunk.content
                        content_str = raw if isinstance(raw, str) else str(raw)
                        if content_str:
                            h = hashlib.md5(content_str.encode(), usedforsecurity=False).hexdigest()
                            if h in seen_content_hashes:
                                logger.debug("Streaming: dedup skipped content-hash replay")
                                continue
                            seen_content_hashes.add(h)
                    if is_chunk and message_id:
                        seen_message_ids.add(message_id)
```

**Step 2: Run dedup tests**

Run: `cd /Users/tyo/Omics-OS/lobster && uv run pytest tests/unit/core/test_client_stream_narrator_boundary.py::TestStreamingDedup -v`

Expected: Both tests PASS.

**Step 3: Commit**

```bash
git add lobster/core/client.py
git commit -m "fix: add content-hash fallback dedup for ID-less stream replays"
```

---

## Task 4: Suppress `unknown_tool` Telemetry Events

**Files:**
- Modify: `lobster/ui/callbacks/protocol_callback.py:40-103`
- Test: `tests/unit/core/test_client_stream_narrator_boundary.py` (add new class)

**What changes:** Suppress tool_execution protocol events when tool name resolves to "unknown_tool". These are noise from LangGraph internal chains that don't correspond to real tool invocations.

**Step 1: Write the failing test**

Add to `tests/unit/core/test_client_stream_narrator_boundary.py`:

```python
class TestUnknownToolSuppression:
    """unknown_tool events must not reach the protocol."""

    def test_unknown_tool_start_suppressed(self):
        from lobster.ui.callbacks.protocol_callback import ProtocolCallbackHandler

        emitted: list = []
        handler = ProtocolCallbackHandler(
            emit_event=lambda t, p: emitted.append((t, p))
        )
        # Simulate a tool_start with missing name
        handler.on_tool_start({}, "some input")
        # Should NOT have emitted a tool_execution event
        tool_events = [(t, p) for t, p in emitted if t == "tool_execution"]
        assert len(tool_events) == 0

    def test_unknown_tool_end_suppressed(self):
        from lobster.ui.callbacks.protocol_callback import ProtocolCallbackHandler

        emitted: list = []
        handler = ProtocolCallbackHandler(
            emit_event=lambda t, p: emitted.append((t, p))
        )
        # on_tool_end with no prior on_tool_start -> current_tool is None -> "unknown_tool"
        handler.on_tool_end("some output")
        tool_events = [(t, p) for t, p in emitted if t == "tool_execution"]
        assert len(tool_events) == 0

    def test_real_tool_still_emitted(self):
        from lobster.ui.callbacks.protocol_callback import ProtocolCallbackHandler

        emitted: list = []
        handler = ProtocolCallbackHandler(
            emit_event=lambda t, p: emitted.append((t, p))
        )
        handler.on_tool_start({"name": "search_pubmed"}, "query")
        handler.on_tool_end("results")
        tool_events = [(t, p) for t, p in emitted if t == "tool_execution"]
        assert len(tool_events) == 2  # start + end
        assert all(p["tool"] == "search_pubmed" for _, p in tool_events)
```

**Step 2: Run to verify failure**

Run: `cd /Users/tyo/Omics-OS/lobster && uv run pytest tests/unit/core/test_client_stream_narrator_boundary.py::TestUnknownToolSuppression -v`

Expected: `test_unknown_tool_start_suppressed` and `test_unknown_tool_end_suppressed` FAIL.

**Step 3: Fix protocol_callback.py**

In `on_tool_start` (line 48), after resolving tool_name, add early return:

```python
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        inputs: Dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        tool_name = (serialized or {}).get("name", "unknown_tool")
        if tool_name == "unknown_tool":
            return
        self.current_tool = tool_name
        # ... rest unchanged ...
```

In `on_tool_end` (line 81), add early return:

```python
    def on_tool_end(self, output: Any, **kwargs) -> None:
        tool_name = self.current_tool or "unknown_tool"
        if tool_name == "unknown_tool":
            self.current_tool = None
            return
        # ... rest unchanged ...
```

In `on_tool_error` (line 106), same pattern:

```python
    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs
    ) -> None:
        tool_name = self.current_tool or kwargs.get("name", "unknown_tool")
        if tool_name == "unknown_tool":
            self.current_tool = None
            return
        # ... rest unchanged ...
```

**Step 4: Run tests**

Run: `cd /Users/tyo/Omics-OS/lobster && uv run pytest tests/unit/core/test_client_stream_narrator_boundary.py::TestUnknownToolSuppression -v`

Expected: All PASS.

**Step 5: Commit**

```bash
git add lobster/ui/callbacks/protocol_callback.py tests/unit/core/test_client_stream_narrator_boundary.py
git commit -m "fix: suppress unknown_tool telemetry events from protocol emission"
```

---

## Task 5: Full Regression Pass

**Files:**
- No new files — validation only

**Step 1: Run all narrator boundary tests**

Run: `cd /Users/tyo/Omics-OS/lobster && uv run pytest tests/unit/core/test_client_stream_narrator_boundary.py -v`

Expected: All tests PASS.

**Step 2: Run existing client + streaming tests**

Run: `cd /Users/tyo/Omics-OS/lobster && uv run pytest tests/unit/core/test_client.py tests/unit/core/test_client_streaming_compaction.py -v`

Expected: All PASS. No regressions.

**Step 3: Run Go TUI regression suite**

Run: `cd /Users/tyo/Omics-OS/lobster && uv run pytest tests/unit/cli/test_go_tui_launcher_completions.py tests/unit/cli/test_slash_commands_go_tui_regressions.py tests/integration/test_slash_command_golden_transcripts.py -v`

Expected: All PASS.

**Step 4: Run protocol callback tests (if any exist)**

Run: `cd /Users/tyo/Omics-OS/lobster && uv run pytest tests/ -k "protocol_callback or protocol_smoke" -v`

Expected: All PASS.

**Step 5: Build Go TUI binary and smoke test**

Run: `cd /Users/tyo/Omics-OS/lobster/lobster-tui && go build -o lobster-tui ./cmd/lobster-tui && go test ./internal/chat ./internal/protocol`

Expected: Build succeeds, all Go tests PASS.

**Step 6: Manual PTY validation**

Run: `cd /Users/tyo/Omics-OS/lobster && uv run lobster chat --ui go`

Send: "search for human lung adenocarcinoma scRNA-seq datasets"

Verify:
1. Sub-agent internal reasoning (JSON fragments, tool parsing) does NOT appear in the chat stream
2. Tool activity shows real tool names (never "unknown_tool")
3. Only the supervisor's synthesized response appears as chat text
4. No duplicated content fragments
5. Markdown renders cleanly without broken fences

---

## Summary of Changes by File

| File | Change | Bugs Fixed |
|------|--------|------------|
| `lobster/core/client.py:382-425` | Add `if node_name != "supervisor": continue` gate before content extraction | #1 narrator breach, #4 corrupted markdown, #5 badge duplication |
| `lobster/core/client.py:342-376` | Add `seen_content_hashes` set + md5 fallback dedup for ID-less messages | #2 streaming dedup |
| `lobster/ui/callbacks/protocol_callback.py:48,81,106` | Early-return when tool_name is "unknown_tool" | #3 unknown_tool telemetry |
| `tests/unit/core/test_client_stream_narrator_boundary.py` | New: 9 regression tests (single-speaker, dedup, unknown_tool, __end__ filtering) | All — prevents regression |

**Execution order matters:** Task 2 (single-speaker) is the highest-impact fix and resolves 3 of 5 bugs. Task 3 (dedup) and Task 4 (unknown_tool) are independent and can run in parallel after Task 2.
