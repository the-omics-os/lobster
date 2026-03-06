"""Lightweight Go TUI chat launcher -- zero heavy dependencies at module level.

WHY THIS EXISTS
---------------
The Go TUI must appear on screen in <200ms.  Importing the normal chat
machinery (``chat_commands.py``) pulls in Rich, LangChain, pandas, and the
entire lobster agent graph, which takes 1-2 seconds on a cold start.

This module sidesteps that entirely.  At module level it imports **only**
Python stdlib.  Heavy imports (lobster core, LangChain, UI callbacks) happen
*after* the Go binary is already on-screen and showing a spinner, so the
user perceives instant startup.

The module inlines the minimal IPC bridge code (~80 lines) rather than
importing ``lobster.ui.bridge``, because that package transitively imports
Rich via ``lobster.ui.__init__``.

PROTOCOL
--------
Communication with the Go binary uses a JSON-lines protocol over inherited
file-descriptor pipes.  Each message is a single JSON object terminated by
``\\n``.  See ``_make_message`` and ``_LightBridge`` for details.
"""
from __future__ import annotations

import json
import inspect
import logging
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lobster.cli_internal.startup_diagnostics import (
    StartupDiagnostic,
    StartupDiagnosticError,
    format_startup_diagnostic_text,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

_PROTOCOL_VERSION = 1
_SUPPORTED_PROTOCOL_VERSIONS = {1}
_FEEDBACK_URL = "https://forms.cloud.microsoft/e/AkNk8J8nE8"
_ISSUES_URL = "https://github.com/the-omics-os/lobster/issues"
_SUPPORT_EMAIL = "info@omics-os.com"
_STARTUP_DIAGNOSTIC_CONFIRM_ID = "__startup_diagnostic_exit__"


# ---------------------------------------------------------------------------
# Binary discovery (inlined from lobster/ui/bridge/binary_finder.py)
# ---------------------------------------------------------------------------


def find_tui_binary_fast() -> Optional[str]:
    """Locate the ``lobster-tui`` binary using only stdlib.

    Search order:
    1. ``LOBSTER_TUI_BINARY`` override path (if set).
    2. Development build -- walk up from this file looking for
       ``lobster-tui/lobster-tui``.
    3. Platform wheel package (``lobster_ai_tui``).
    4. User cache directory (``~/.cache/lobster/bin/lobster-tui``).
    5. System PATH via ``shutil.which``.

    Returns the absolute path to the binary, or ``None`` if not found.
    """
    # 1. Explicit override (useful for local dev/debugging).
    override = os.environ.get("LOBSTER_TUI_BINARY", "").strip()
    if override:
        override_path = Path(override).expanduser()
        if override_path.is_file() and os.access(override_path, os.X_OK):
            return str(override_path.resolve())

    # 2. Development build -- walk up from this file.
    current = Path(__file__).resolve().parent
    for _ in range(8):
        candidate = current / "lobster-tui" / "lobster-tui"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        if current.parent == current:
            break
        current = current.parent

    # 3. Platform wheel package (tiny or absent -- safe to attempt)
    try:
        from lobster_ai_tui import get_binary_path  # type: ignore[import-untyped]

        candidate = get_binary_path()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    except (ImportError, FileNotFoundError, AttributeError, TypeError):
        pass

    # 4. User cache
    cache_bin = Path.home() / ".cache" / "lobster" / "bin" / "lobster-tui"
    if cache_bin.is_file() and os.access(cache_bin, os.X_OK):
        return str(cache_bin)

    # 5. System PATH
    on_path = shutil.which("lobster-tui")
    if on_path:
        return on_path

    return None


# ---------------------------------------------------------------------------
# Inline IPC bridge (stdlib only -- avoids importing lobster.ui.bridge)
# ---------------------------------------------------------------------------


def _make_message(
    msg_type: str,
    payload: Optional[dict] = None,
    msg_id: str = "",
) -> str:
    """Create a JSON-lines protocol message.

    Parameters
    ----------
    msg_type:
        Message type string (e.g. ``"spinner"``, ``"text"``, ``"quit"``).
    payload:
        Optional dict payload.
    msg_id:
        Optional message ID for request/response correlation.
    """
    msg: Dict[str, Any] = {
        "version": _PROTOCOL_VERSION,
        "type": msg_type,
        "payload": payload or {},
    }
    if msg_id:
        msg["id"] = msg_id
    return json.dumps(msg)


class _LightBridge:
    """Minimal IPC bridge to the Go TUI -- stdlib only, no lobster imports.

    Communicates over two inherited file-descriptor pipes using the
    JSON-lines protocol.  A background thread reads incoming messages
    and enqueues them for the main thread to consume via ``recv_event``.
    """

    def __init__(
        self,
        process: subprocess.Popen,  # type: ignore[type-arg]
        writer_fd: int,
        reader_fd: int,
    ) -> None:
        self.process = process
        self._writer = os.fdopen(writer_fd, "w", encoding="utf-8", buffering=1)
        self._reader = os.fdopen(reader_fd, "r", encoding="utf-8", buffering=1)
        self._events: queue.Queue[dict] = queue.Queue()
        self._running = True
        self._write_lock = threading.Lock()
        self._read_thread = threading.Thread(
            target=self._read_loop, daemon=True, name="go-tui-reader"
        )
        self._read_thread.start()

    def send(
        self,
        msg_type: str,
        payload: Optional[dict] = None,
        msg_id: str = "",
    ) -> None:
        """Send a message to the Go TUI.  No-op if bridge is closed."""
        if not self._running:
            return
        line = _make_message(msg_type, payload, msg_id=msg_id)
        with self._write_lock:
            try:
                self._writer.write(line + "\n")
                self._writer.flush()
            except (OSError, ValueError):
                # Pipe closed or writer already closed.
                self._running = False

    def recv_event(self, *, timeout: Optional[float] = None) -> Optional[dict]:
        """Block until a message arrives or *timeout* expires.

        Returns ``None`` on timeout or if the bridge is closed.
        """
        try:
            return self._events.get(timeout=timeout)
        except queue.Empty:
            return None

    def _read_loop(self) -> None:
        """Background thread: read JSON-lines from the Go TUI."""
        while self._running:
            try:
                line = self._reader.readline()
            except Exception:
                break
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                self._events.put(msg)
            except json.JSONDecodeError:
                logger.debug("Ignoring malformed message from Go TUI: %s", line[:200])
                continue

    def close(self) -> None:
        """Shut down the bridge and terminate the Go process."""
        if not self._running:
            return

        # Send quit BEFORE clearing _running (send() checks the flag).
        try:
            self.send("quit", {})
        except Exception:
            pass

        self._running = False

        # Close writer/reader.
        for stream in (self._writer, self._reader):
            try:
                stream.close()
            except Exception:
                pass

        # Terminate the process group.
        proc = self.process
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


def _heartbeat_loop(bridge: _LightBridge, stop_event: threading.Event) -> None:
    """Send periodic heartbeats so the Go TUI knows Python is alive."""
    while not stop_event.is_set():
        bridge.send("heartbeat", {"timestamp": time.time()})
        stop_event.wait(3.0)


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


def _normalize_tool_payload(payload: dict) -> dict:
    """Normalize tool-execution event payloads to a consistent shape."""
    event = {
        "running": "start",
        "complete": "finish",
        "error": "error",
    }.get(payload.get("status", ""), payload.get("event", ""))

    summary = payload.get("summary", "")

    return {
        "tool_name": payload.get("tool", payload.get("tool_name", "")),
        "event": event,
        "agent": payload.get("agent", ""),
        "summary": summary,
    }


def _format_usage(client: Any) -> str:
    """Format token/cost usage from the client's tracker."""
    try:
        tracker = client.token_tracker
        return f"Tokens: {tracker.total_tokens:,} | Cost: ${tracker.total_cost:.4f}"
    except Exception:
        return "Ready"


def _format_context_compaction_notice(event: dict) -> str:
    """Build operator-facing copy for context compaction stream events."""
    before = event.get("before_count")
    after = event.get("after_count")
    budget = event.get("budget_tokens")

    if all(isinstance(v, int) for v in (before, after, budget)):
        return (
            f"Context compacted: {before}->{after} messages (budget {budget} tokens).\n"
            "Full delegated outputs remain available via store keys and retrieve_agent_result."
        )

    return (
        "Context compacted to stay within model budget.\n"
        "Full delegated outputs remain available via store keys and retrieve_agent_result."
    )


def _safe_minimal_token_summary(client: Any) -> str:
    """Return a minimal token summary string, or empty string on failure."""
    try:
        tracker = getattr(client, "token_tracker", None)
        if not tracker:
            return ""
        if hasattr(tracker, "get_minimal_summary"):
            return str(tracker.get_minimal_summary() or "")
        if hasattr(tracker, "get_verbose_summary"):
            return str(tracker.get_verbose_summary() or "")
    except Exception:
        return ""
    return ""


def _resolve_active_provider_name(client: Any) -> str:
    """Resolve the active provider name for local clients, if available."""
    runtime_override = str(
        getattr(client, "provider_override", None) or ""
    ).strip()

    # Only local AgentClient instances have a resolver-backed provider choice.
    if hasattr(client, "data_manager"):
        workspace_path = getattr(client, "workspace_path", None)
        if workspace_path:
            try:
                from lobster.core.config_resolver import ConfigResolver

                resolver = ConfigResolver(workspace_path=Path(workspace_path))
                provider_name, _ = resolver.resolve_provider(
                    runtime_override=runtime_override or None
                )
                resolved = str(provider_name or "").strip()
                if resolved:
                    return resolved
            except Exception:
                pass

    fallback = runtime_override or str(
        getattr(client, "provider", None) or ""
    ).strip()
    return fallback


def _emit_provider_status(bridge: _LightBridge, client: Any) -> None:
    """Push the current provider into Go status state."""
    provider_name = _resolve_active_provider_name(client) or "auto"
    bridge.send("status", {"text": f"Provider: {provider_name}"})


def _emit_startup_diagnostic(bridge: _LightBridge, diagnostic: StartupDiagnostic) -> None:
    """Render a fatal startup diagnostic through the Go protocol surface."""
    bridge.send("spinner", {"active": False})
    bridge.send(
        "alert",
        {
            "level": diagnostic.level,
            "message": format_startup_diagnostic_text(diagnostic),
        },
    )
    bridge.send("status", {"text": "Initialization failed"})


def _await_startup_diagnostic_ack(bridge: _LightBridge) -> None:
    """Keep a fatal startup diagnostic visible until the operator exits it."""
    bridge.send(
        "confirm",
        {
            "title": "Startup failed",
            "message": "Press Enter to exit.",
            "default": True,
        },
        msg_id=_STARTUP_DIAGNOSTIC_CONFIRM_ID,
    )

    while True:
        event = bridge.recv_event(timeout=None)
        if event is None:
            return

        msg_type = str(event.get("type", "") or "")
        if msg_type == "quit":
            return
        if (
            msg_type == "confirm_response"
            and str(event.get("id", "") or "") == _STARTUP_DIAGNOSTIC_CONFIRM_ID
        ):
            return


def _build_go_tui_exit_footer_lines(client: Any) -> List[str]:
    """Build post-exit footer lines for Go chat mode."""
    lines: List[str] = []

    session = str(getattr(client, "session_id", "") or "").strip()
    if session:
        lines.append(f"Session: {session} (use --session-id latest to continue)")

    lines.append(f"Feedback: {_FEEDBACK_URL}")
    lines.append(f"Issues: {_ISSUES_URL}")
    lines.append(f"Contact: {_SUPPORT_EMAIL}")

    provider_name = _resolve_active_provider_name(client)
    model_name = str(
        getattr(client, "model_override", None)
        or getattr(client, "model", None)
        or ""
    ).strip()
    runtime_parts: List[str] = []
    if provider_name:
        runtime_parts.append(f"provider={provider_name}")
    if model_name:
        runtime_parts.append(f"model={model_name}")
    if runtime_parts:
        lines.append(f"Runtime: {' '.join(runtime_parts)}")

    token_summary = _safe_minimal_token_summary(client)
    if token_summary:
        lines.append(token_summary)

    lines.append("Next steps:")
    lines.append("  Resume: lobster chat --ui go --session-id latest")
    lines.append("  New chat: lobster chat --ui go")
    lines.append("  Help: lobster --help")
    return lines


def _emit_go_tui_exit_footer(client: Any) -> None:
    """Print Go chat exit footer to stdout after TUI teardown."""
    try:
        lines = _build_go_tui_exit_footer_lines(client)
        if not lines:
            return
        sys.stdout.write("\n")
        for line in lines:
            sys.stdout.write(f"{line}\n")
        sys.stdout.write("\n")
        sys.stdout.flush()
    except Exception:
        # Never let footer rendering block CLI exit.
        return


def _path_completion_suggestions(prefix: str, limit: int = 50) -> List[str]:
    """Return path-like completion candidates for the provided prefix."""
    prefix = (prefix or "").strip()
    expanded = os.path.expanduser(prefix) if prefix else ""

    if prefix.endswith(("/", "\\")):
        base_input = prefix
        base_expanded = expanded or "."
        typed = ""
    else:
        slash_idx = max(prefix.rfind("/"), prefix.rfind("\\"))
        base_input = prefix[: slash_idx + 1] if slash_idx >= 0 else ""
        base_expanded = expanded[: slash_idx + 1] if slash_idx >= 0 else "."
        typed = prefix[slash_idx + 1 :] if slash_idx >= 0 else prefix

    try:
        base_path = Path(base_expanded or ".")
        entries = list(base_path.iterdir())
    except Exception:
        return []

    out: List[str] = []
    typed_lower = typed.lower()
    for entry in entries:
        name = entry.name
        if typed and not name.lower().startswith(typed_lower):
            continue
        candidate = f"{base_input}{name}"
        try:
            if entry.is_dir():
                candidate += "/"
        except OSError:
            continue
        out.append(candidate)

    out = sorted(set(out), key=lambda s: s.lower())
    return out[:limit]


def _workspace_load_suggestions(client: Any, prefix: str, limit: int = 50) -> List[str]:
    """Suggest workspace dataset names plus fallback filesystem paths."""
    suggestions: List[str] = []
    try:
        data_manager = getattr(client, "data_manager", None)
        available = getattr(data_manager, "available_datasets", None)
        if isinstance(available, dict):
            pfx = (prefix or "").lower()
            for name in available.keys():
                if not isinstance(name, str):
                    continue
                if pfx and not name.lower().startswith(pfx):
                    continue
                suggestions.append(name)
    except Exception:
        pass

    suggestions.extend(_path_completion_suggestions(prefix, limit=limit))
    deduped = sorted(set(suggestions), key=lambda s: s.lower())
    return deduped[:limit]


def _quote_completion_token(token: str) -> str:
    """Quote completion token when it contains whitespace or quotes."""
    if not token:
        return token
    if any(ch.isspace() for ch in token) or '"' in token:
        escaped = token.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return token


def _prefix_completion_suggestions(
    command: str, items: List[str], limit: int = 50
) -> List[str]:
    """Format candidate tokens as full command-line completion suggestions."""
    command = (command or "").strip()
    if not command:
        return []
    formatted: List[str] = []
    for item in items:
        token = _quote_completion_token((item or "").strip())
        if not token:
            continue
        formatted.append(f"{command} {token}")
    deduped = sorted(set(formatted), key=lambda s: s.lower())
    return deduped[:limit]


def _infer_completion_from_input(text: str) -> Tuple[str, str]:
    """Best-effort fallback parser for completion requests."""
    text = (text or "").lstrip()
    if text.startswith("/read "):
        return "/read", text[len("/read ") :]
    if text == "/read":
        return "/read", ""
    if text.startswith("/open "):
        return "/open", text[len("/open ") :]
    if text == "/open":
        return "/open", ""
    if text.startswith("/workspace load "):
        return "/workspace load", text[len("/workspace load ") :]
    if text == "/workspace load":
        return "/workspace load", ""
    return "", ""


def _handle_completion_request(bridge: _LightBridge, client: Any, event: dict) -> None:
    """Handle completion_request by returning best-effort suggestions quickly."""
    payload = event.get("payload", {}) or {}
    req_id = str(event.get("id", "") or "")
    command = str(payload.get("command", "") or "").strip().lower()
    prefix = str(payload.get("prefix", "") or "")
    if not command:
        command, inferred_prefix = _infer_completion_from_input(
            str(payload.get("input", "") or "")
        )
        if not prefix:
            prefix = inferred_prefix

    try:
        if command in {"/read", "/open"}:
            raw = _path_completion_suggestions(prefix)
            suggestions = _prefix_completion_suggestions(command, raw)
        elif command == "/workspace load":
            raw = _workspace_load_suggestions(client, prefix)
            suggestions = _prefix_completion_suggestions(command, raw)
        else:
            suggestions = []
        bridge.send(
            "completion_response",
            {"suggestions": suggestions},
            msg_id=req_id,
        )
    except Exception as exc:
        bridge.send(
            "completion_response",
            {"suggestions": [], "error": str(exc)},
            msg_id=req_id,
        )


def _save_session_json_if_available(client: Any) -> None:
    """Persist session transcript when the client supports it."""
    if hasattr(client, "_save_session_json"):
        try:
            client._save_session_json()
        except Exception:
            pass


def _resolve_go_chat_session_target(
    workspace: Optional[str],
    session_id: Optional[str],
) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve session continuation target for Go chat.

    Returns:
        ``(session_file_to_load, session_id_for_client)``
    """
    if not session_id:
        return None, None

    from lobster.core.workspace import resolve_workspace
    from lobster.cli_internal.commands.heavy.session_infra import (
        resolve_session_continuation,
    )

    workspace_path = resolve_workspace(explicit_path=workspace, create=True)
    (
        session_file_to_load,
        session_id_for_client,
        found_existing_session,
    ) = resolve_session_continuation(workspace_path, session_id)

    # Match classic chat behavior: unresolved "latest" creates a new session.
    if session_id == "latest" and not found_existing_session:
        return None, None

    return session_file_to_load, session_id_for_client


def _maybe_restore_go_session_transcript(
    bridge: _LightBridge,
    client: Any,
    session_file_to_load: Optional[Path],
) -> None:
    """Load persisted session transcript when available."""
    if session_file_to_load is None:
        return

    try:
        load_result = client.load_session(session_file_to_load)
        messages_loaded = load_result.get("messages_loaded", 0)
        if messages_loaded:
            bridge.send("status", {"text": f"Loaded {messages_loaded} previous messages"})
    except Exception as exc:
        bridge.send(
            "alert",
            {
                "level": "warning",
                "message": f"Failed to restore session transcript: {exc}",
            },
        )


# ---------------------------------------------------------------------------
# Event loop
# ---------------------------------------------------------------------------


def _go_tui_event_loop(bridge: _LightBridge, client: Any) -> None:
    """Main event loop: dispatch messages from the Go TUI to the client."""
    cancel_event = threading.Event()
    while True:
        event = bridge.recv_event(timeout=None)
        if event is None:
            break
        msg_type = event.get("type", "")
        if msg_type == "quit":
            _save_session_json_if_available(client)
            break
        elif msg_type == "cancel":
            cancel_event.set()
        elif msg_type == "input":
            cancel_event.clear()
            content = event.get("payload", {}).get("content", "")
            _dispatch_user_query(bridge, client, content, cancel_event)
        elif msg_type == "slash_command":
            payload = event.get("payload", {})
            command = payload.get("command", "")
            args = payload.get("args", "")
            _handle_slash_command(bridge, client, command, args)
        elif msg_type == "completion_request":
            _handle_completion_request(bridge, client, event)


def _dispatch_user_query(
    bridge: _LightBridge,
    client: Any,
    text: str,
    cancel_event: threading.Event,
) -> None:
    """Call _handle_user_query with optional cancel support when available."""
    try:
        params = inspect.signature(_handle_user_query).parameters
    except (TypeError, ValueError):
        params = {}

    if "cancel_event" in params:
        _handle_user_query(bridge, client, text, cancel_event=cancel_event)
        return

    _handle_user_query(bridge, client, text)


def _handle_user_query(
    bridge: _LightBridge,
    client: Any,
    text: str,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """Forward a user query to the Lobster client and stream results back.

    Implements the resume loop pattern for HITL interrupts:
    stream → detect interrupt → render component → collect response → resume → stream.
    """
    bridge.send("spinner", {"active": True})
    query_start = time.time()
    is_first = True
    stream_source: Any = None  # Will be set to the generator each iteration.

    try:
        while True:
            # Build the stream: first iteration uses query(), subsequent use resume.
            if is_first:
                stream_source = client.query(text, stream=True, cancel_event=cancel_event)
                is_first = False
            # else: stream_source was set by the resume path below.

            interrupt_event = None
            cancelled = False

            for event in stream_source:
                if cancel_event and cancel_event.is_set():
                    cancelled = True
                    break
                etype = event["type"]
                if etype == "interrupt":
                    interrupt_event = event
                    break  # Exit inner loop to handle interrupt.
                _forward_stream_event(bridge, client, event, query_start)

            if cancelled:
                bridge.send(
                    "alert",
                    {"level": "warning", "message": "Query cancelled"},
                )
                bridge.send("done", {"summary": "cancelled"})
                duration = time.time() - query_start
                bridge.send("status", {"text": f"Cancelled after {duration:.1f}s"})
                bridge.send("spinner", {"active": False})
                return

            if interrupt_event is None:
                # Stream completed normally (no interrupt).
                return

            # --- HITL interrupt: render component and await user response ---
            response = _handle_interrupt(bridge, interrupt_event)
            if response is None:
                # Interrupt was cancelled or timed out.
                bridge.send("spinner", {"active": False})
                return

            # Resume the graph with the user's response.
            bridge.send("spinner", {"active": True, "label": "resuming"})
            stream_source = client.resume_from_interrupt(
                response, stream=True, cancel_event=cancel_event
            )
            # Continue the while loop to process the resumed stream.

    except KeyboardInterrupt:
        bridge.send(
            "alert", {"level": "warning", "message": "Query interrupted"}
        )
        bridge.send("spinner", {"active": False})
    except Exception as exc:
        bridge.send("alert", {"level": "error", "message": f"Error: {exc}"})
        bridge.send("spinner", {"active": False})


# Response types the bridge accepts as HITL answers from the Go TUI.
_COMPONENT_RESPONSE_TYPES = frozenset(
    {"component_response", "confirm_response", "select_response"}
)


def _handle_interrupt(bridge: _LightBridge, interrupt_event: dict) -> Optional[dict]:
    """Send component_render to Go TUI and wait for the user's response.

    Returns the response data dict, or None if the interrupt timed out or
    was cancelled (quit).
    """
    data = interrupt_event.get("data", {})
    msg_id = str(uuid.uuid4())

    bridge.send("spinner", {"active": False})
    bridge.send("component_render", data, msg_id=msg_id)

    # Block waiting for the Go TUI to send back the response.
    while True:
        resp = bridge.recv_event(timeout=300)
        if resp is None:
            # Timeout — treat as cancelled.
            return None

        resp_type = resp.get("type", "")
        if resp_type in _COMPONENT_RESPONSE_TYPES:
            payload = resp.get("payload", {})
            # confirm_response → {"confirmed": bool}
            if resp_type == "confirm_response":
                return {"confirmed": payload.get("confirm", False)}
            # select_response → {"selected": str, "index": int}
            if resp_type == "select_response":
                return {
                    "selected": payload.get("value", ""),
                    "index": payload.get("index", 0),
                }
            # component_response → pass through data
            return payload.get("data", payload)

        if resp_type == "quit":
            return None
        if resp_type == "cancel":
            return None
        # Other message types (heartbeat, resize, etc.) — ignore and keep waiting.


def _forward_stream_event(
    bridge: _LightBridge, client: Any, event: dict, query_start: float
) -> None:
    """Forward a single stream event to the Go TUI."""
    etype = event["type"]
    if etype == "content_delta":
        bridge.send("text", {"content": event["delta"]})
    elif etype == "agent_change":
        bridge.send(
            "agent_transition",
            {
                "to": event.get("agent", ""),
                "kind": "activity",
                "status": event.get("status", "working"),
                "reason": event.get("status", "working"),
            },
        )
    elif etype == "complete":
        bridge.send("done", {"summary": ""})
        duration = time.time() - query_start
        status_parts = [_format_usage(client)]
        status_parts.append(f"Duration: {duration:.1f}s")
        bridge.send("status", {"text": " · ".join(status_parts)})
        bridge.send("spinner", {"active": False})
    elif etype == "context_compaction":
        bridge.send(
            "alert",
            {
                "level": "info",
                "message": _format_context_compaction_notice(event),
            },
        )
    elif etype == "error":
        bridge.send(
            "alert",
            {
                "level": "error",
                "message": str(event.get("error", "Unknown error")),
            },
        )
        bridge.send("spinner", {"active": False})


def _handle_slash_command(
    bridge: _LightBridge,
    client: Any,
    cmd: str,
    args: str = "",
) -> None:
    """Execute a slash command via Python and stream results to the Go TUI.

    Go handles /help, /clear, /exit, /data natively -- they never arrive here.
    Everything else is executed through the OutputAdapter pipeline.
    """
    full_cmd = cmd if cmd.startswith("/") else f"/{cmd}"
    args = (args or "").strip()
    if args:
        full_cmd = f"{full_cmd} {args}"

    # Show spinner while heavy imports load (~1-2s first time).
    bridge.send("spinner", {"active": True, "label": "running command"})

    try:
        from lobster.cli_internal.commands.output_adapter import ProtocolOutputAdapter
        output = ProtocolOutputAdapter(bridge.send)

        from lobster.cli_internal.commands.heavy.slash_commands import _execute_command
        from lobster.cli_internal.commands.heavy.session_infra import (
            _add_command_to_history,
        )

        command_summary = _execute_command(
            full_cmd,
            client,
            original_command=full_cmd,
            output=output,
        )
        if command_summary:
            _add_command_to_history(client, full_cmd, command_summary)
    except Exception as exc:
        try:
            from lobster.cli_internal.commands.heavy.session_infra import (
                _add_command_to_history,
            )

            error_summary = f"Failed: {type(exc).__name__}: {str(exc)[:100]}"
            _add_command_to_history(
                client,
                full_cmd,
                error_summary,
                is_error=True,
            )
        except Exception:
            pass
        bridge.send("alert", {"level": "error", "message": str(exc)})
    finally:
        _emit_provider_status(bridge, client)
        bridge.send("spinner", {"active": False})
        bridge.send("done", {"summary": ""})


# ---------------------------------------------------------------------------
# Main launcher
# ---------------------------------------------------------------------------


def launch_go_tui_chat(
    binary: str,
    *,
    workspace: Optional[str] = None,
    session_id: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    debug: bool = False,
    profile_timings: Any = None,
    no_intro: bool = False,
    stream: bool = True,
) -> None:
    """Launch the Go TUI and wire it to the Lobster agent client.

    This function is the main entry point.  It spawns the Go binary
    **immediately** (using only stdlib), then performs heavy imports
    while the TUI displays a loading spinner.  This achieves <200ms
    perceived startup time.

    Parameters
    ----------
    binary:
        Absolute path to the ``lobster-tui`` executable.
    workspace:
        Working directory / data workspace path.
    session_id:
        Session identifier for conversation continuity.
    provider:
        LLM provider override (e.g. ``"openrouter"``).
    model:
        LLM model override (e.g. ``"claude-sonnet-4-20250514"``).
    debug:
        If ``True``, capture stderr from the Go process.
    profile_timings:
        Optional profiling configuration.
    stream:
        Whether to use streaming mode (default ``True``).
    """
    # -----------------------------------------------------------------
    # 1. Create pipes and spawn the Go binary (stdlib only)
    # -----------------------------------------------------------------
    p2g_r, p2g_w = os.pipe()  # Python-to-Go
    g2p_r, g2p_w = os.pipe()  # Go-to-Python

    theme_name = os.environ.get("LOBSTER_TUI_THEME", "lobster-dark").strip() or "lobster-dark"
    inline_env = os.environ.get("LOBSTER_TUI_INLINE", "1").strip().lower()
    fullscreen_env = os.environ.get("LOBSTER_TUI_FULLSCREEN", "").strip().lower()
    inline_mode = inline_env not in {"0", "false", "no"}
    if fullscreen_env in {"1", "true", "yes"}:
        inline_mode = False

    cmd = [
        binary,
        "chat",
        "--proto-fd-in",
        str(p2g_r),
        "--proto-fd-out",
        str(g2p_w),
        "--theme",
        theme_name,
    ]
    if inline_mode:
        cmd.append("--inline")

    child_env = _prepare_go_tui_chat_env(
        os.environ,
        workspace=workspace,
        provider=provider,
        no_intro=no_intro,
    )

    proc = subprocess.Popen(
        cmd,
        pass_fds=(p2g_r, g2p_w),
        # stdout inherited — BubbleTea needs the real terminal to render.
        # Protocol communication uses separate FDs (the pipes), not stdout.
        stderr=subprocess.PIPE if debug else subprocess.DEVNULL,
        text=False,
        preexec_fn=os.setsid,
        env=child_env,
    )

    # Close the child-side FDs in the parent process.
    os.close(p2g_r)
    os.close(g2p_w)

    bridge = _LightBridge(proc, p2g_w, g2p_r)
    _saved_stderr = sys.stderr
    client: Any = None
    should_emit_exit_footer = False
    startup_diagnostic: Optional[StartupDiagnostic] = None

    try:
        # -----------------------------------------------------------------
        # 2. Wait for handshake from Go (5-second timeout)
        # -----------------------------------------------------------------
        handshake = bridge.recv_event(timeout=5.0)
        if handshake is None or handshake.get("type") != "handshake":
            raise RuntimeError(
                "Go TUI handshake failed -- "
                "did not receive handshake within 5 seconds"
            )

        # Validate protocol compatibility
        hs_payload = handshake.get("payload", {})
        tui_protocol = hs_payload.get("protocol_version", 0)
        tui_version = hs_payload.get("client_version", "unknown")
        if tui_protocol not in _SUPPORTED_PROTOCOL_VERSIONS:
            from lobster.core.component_registry import get_install_command

            bridge.close()
            raise RuntimeError(
                f"Go TUI protocol version {tui_protocol} (binary v{tui_version}) "
                f"is incompatible with this Lobster AI release "
                f"(supported protocols: {_SUPPORTED_PROTOCOL_VERSIONS}). "
                f"Install with: {get_install_command('lobster-ai-tui')}"
            )

        # -----------------------------------------------------------------
        # 3. Tell the TUI to show a loading spinner
        # -----------------------------------------------------------------
        bridge.send(
            "spinner", {"active": True, "label": "Initializing agents"}
        )

        # -----------------------------------------------------------------
        # 4. Start heartbeat thread
        # -----------------------------------------------------------------
        heartbeat_stop = threading.Event()
        heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            args=(bridge, heartbeat_stop),
            daemon=True,
            name="go-tui-heartbeat",
        )
        heartbeat_thread.start()

        # -----------------------------------------------------------------
        # 5. HEAVY IMPORTS (Go is already visible and showing spinner)
        #    Suppress Python logging/stderr so it doesn't corrupt the TUI.
        # -----------------------------------------------------------------
        _saved_stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")
        logging.disable(logging.CRITICAL)

        from lobster.cli_internal.commands.heavy.session_infra import (
            init_client_or_raise_startup_diagnostic,
            set_go_tui_active,
        )

        set_go_tui_active(True)
        session_file_to_load, session_id_for_client = _resolve_go_chat_session_target(
            workspace,
            session_id,
        )

        client = init_client_or_raise_startup_diagnostic(
            workspace=workspace,
            reasoning=False,
            verbose=False,
            debug=debug,
            profile_timings=profile_timings,
            provider_override=provider,
            model_override=model,
            session_id=session_id_for_client,
        )
        _maybe_restore_go_session_transcript(bridge, client, session_file_to_load)

        _emit_provider_status(bridge, client)

        # -----------------------------------------------------------------
        # 6. Wire the protocol callback handler
        # -----------------------------------------------------------------
        from lobster.ui.callbacks.protocol_callback import (
            ProtocolCallbackHandler,
        )

        proto_callback = ProtocolCallbackHandler(
            emit_event=lambda msg_type, payload: bridge.send(
                msg_type,
                _normalize_tool_payload(payload)
                if msg_type == "tool_execution"
                else payload,
            )
        )
        client.callbacks.append(proto_callback)

        # Wire modality loaded events to the Go TUI.
        def _on_modality_loaded(name, adata):
            shape = ""
            if hasattr(adata, "shape"):
                shape = f"{adata.shape[0]} obs x {adata.shape[1]} vars"
            bridge.send("modality_loaded", {"name": name, "shape": shape})

        if hasattr(client, "data_manager") and client.data_manager is not None:
            client.data_manager.on_modality_loaded = _on_modality_loaded

        # -----------------------------------------------------------------
        # 7. Restore stderr/logging, stop heartbeat, signal ready
        # -----------------------------------------------------------------
        sys.stderr = _saved_stderr
        logging.disable(logging.NOTSET)

        heartbeat_stop.set()
        bridge.send("spinner", {"active": False})

        # Send session ID to Go TUI before ready signal.
        try:
            sid = getattr(client, "session_id", None)
            if sid:
                bridge.send("status", {"text": f"Session: {sid}"})
        except Exception:
            pass

        bridge.send("ready", {})
        if not inline_mode:
            bridge.send("status", {"text": "Ready"})

        # -----------------------------------------------------------------
        # 8. Enter event loop
        # -----------------------------------------------------------------
        _go_tui_event_loop(bridge, client)
        should_emit_exit_footer = True
    except StartupDiagnosticError as exc:
        startup_diagnostic = exc.diagnostic
        _emit_startup_diagnostic(bridge, startup_diagnostic)
        _await_startup_diagnostic_ack(bridge)

    finally:
        # Restore stderr/logging in case init crashed mid-suppression.
        sys.stderr = _saved_stderr
        logging.disable(logging.NOTSET)

        bridge.close()
        try:
            from lobster.cli_internal.commands.heavy.session_infra import (
                set_go_tui_active,
            )

            set_go_tui_active(False)
        except Exception:
            pass

        if client is not None:
            _save_session_json_if_available(client)

        if should_emit_exit_footer and client is not None:
            _emit_go_tui_exit_footer(client)

    if startup_diagnostic is not None:
        raise SystemExit(startup_diagnostic.exit_code)


def _prepare_go_tui_chat_env(
    base_env: Dict[str, str],
    *,
    workspace: Optional[str] = None,
    provider: Optional[str] = None,
    no_intro: bool = False,
) -> Dict[str, str]:
    """Build the child environment for a Go TUI chat session."""
    child_env = dict(base_env)
    child_env["LOBSTER_TUI_PROVIDER"] = (
        provider or child_env.get("LOBSTER_TUI_PROVIDER", "auto")
    ).strip() or "auto"
    child_env["LOBSTER_TUI_WORKSPACE"] = str(workspace or Path.cwd())
    if not child_env.get("LOBSTER_TUI_APP_VERSION"):
        try:
            from lobster.version import __version__

            child_env["LOBSTER_TUI_APP_VERSION"] = __version__
        except Exception:
            pass
    if no_intro:
        child_env["LOBSTER_TUI_NO_INTRO"] = "1"
    return child_env
