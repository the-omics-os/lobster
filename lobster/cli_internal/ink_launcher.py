"""Ink TUI chat launcher — zero heavy dependencies at module level.

Mirrors go_tui_launcher.py's design philosophy: stdlib only at module level,
heavy imports deferred until the TUI binary is already displaying.

Architecture:
  1. Start a lightweight local HTTP server on a random port
  2. Spawn the lobster-chat binary with --api-url=http://localhost:{port}
  3. The HTTP server bridges DataStream SSE to the LangGraph agent

The server implements the ui-message-stream protocol (SSE with JSON payloads),
the same format consumed by the Cloud web app.

DataStream event types emitted:
  - text-delta: assistant text chunks (user-visible supervisor output)
  - tool-call-begin: tool invocation start (name + ID)
  - tool-call-delta: tool argument streaming
  - tool-result: tool execution result
  - data: custom state patches (active_agent, token_usage, activity)
  - finish: stream completion signal
"""

from __future__ import annotations

import json
import logging
import os
import queue as queue_mod
import shutil
import signal
import socket
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote

logger = logging.getLogger(__name__)


@dataclass
class _PythonTerminalQuarantine:
    """Mute Python-owned terminal output while the Ink UI owns the screen."""

    devnull_fd: int
    stdout_dup_fd: int
    stderr_dup_fd: int
    logging_disable_level: int

    @classmethod
    def activate(cls) -> "_PythonTerminalQuarantine":
        stdout_fd = 1
        stderr_fd = 2

        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            sys.stderr.flush()
        except Exception:
            pass

        previous_disable = logging.root.manager.disable
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        stdout_dup_fd = os.dup(stdout_fd)
        stderr_dup_fd = os.dup(stderr_fd)

        try:
            os.dup2(devnull_fd, stdout_fd)
            os.dup2(devnull_fd, stderr_fd)
            logging.disable(logging.CRITICAL)
        except Exception:
            for fd in (stdout_dup_fd, stderr_dup_fd, devnull_fd):
                try:
                    os.close(fd)
                except Exception:
                    pass
            raise

        return cls(
            devnull_fd=devnull_fd,
            stdout_dup_fd=stdout_dup_fd,
            stderr_dup_fd=stderr_dup_fd,
            logging_disable_level=previous_disable,
        )

    def restore(self) -> None:
        stdout_fd = 1
        stderr_fd = 2

        try:
            os.dup2(self.stdout_dup_fd, stdout_fd)
            os.dup2(self.stderr_dup_fd, stderr_fd)
        finally:
            for fd in (self.stdout_dup_fd, self.stderr_dup_fd, self.devnull_fd):
                try:
                    os.close(fd)
                except Exception:
                    pass
            logging.disable(self.logging_disable_level)


def _resolve_active_provider_name(client: Any) -> str:
    """Resolve the active provider name for local clients, if available."""
    runtime_override = str(
        getattr(client, "provider_override", None) or ""
    ).strip()

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


def _log_slash_history_async(client: Any, command: str, summary: str) -> None:
    """Record slash-command history without blocking the HTTP response."""

    def _worker() -> None:
        try:
            from lobster.cli_internal.commands.heavy.session_infra import (
                _add_command_to_history,
            )

            _add_command_to_history(client, command, summary)
        except Exception:
            logger.exception("Failed to record Ink slash command history")

    threading.Thread(
        target=_worker,
        name="ink-slash-history",
        daemon=True,
    ).start()


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------


def find_ink_binary() -> Optional[str]:
    """Locate the lobster-chat binary.

    Search order:
    1. LOBSTER_INK_BINARY env var
    2. lobster-tui-ink/dist/lobster-chat (dev build)
    3. ~/.cache/lobster/bin/lobster-chat (downloaded)
    4. PATH lookup
    """
    env = os.environ.get("LOBSTER_INK_BINARY")
    if env and os.path.isfile(env):
        return env

    # Dev build in monorepo
    dev_path = (
        Path(__file__).resolve().parents[2]
        / "lobster-tui-ink"
        / "dist"
        / "lobster-chat"
    )
    if dev_path.is_file():
        return str(dev_path)

    # Cached download
    cache_path = Path.home() / ".cache" / "lobster" / "bin" / "lobster-chat"
    if cache_path.is_file():
        return str(cache_path)

    # PATH
    found = shutil.which("lobster-chat")
    if found:
        return found

    return None


# ---------------------------------------------------------------------------
# Local DataStream HTTP server
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Find a random available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _DataStreamHandler(BaseHTTPRequestHandler):
    """HTTP handler that bridges DataStream SSE to the LangGraph agent.

    Translates LangGraph streaming events from ``client.query(stream=True)``
    into the ui-message-stream SSE protocol consumed by @assistant-ui/react.

    State patch events use transient ``data-*`` chunks, for example
    ``{"type": "data-activity_events", "data": [...], "transient": true}``,
    so assistant-ui dispatches them through ``onData`` without polluting the
    visible assistant message content. Keys match stateHandlers.ts: plots,
    modalities, active_agent, agent_status, activity_events, token_usage,
    session_title.

    LangGraph event mapping:
      content_delta  -> text-delta  (supervisor text, user-visible)
      agent_content  -> data        (activity_events state patch)
      agent_change   -> data        (active_agent + agent_status state patches)
      tool_execution -> data        (activity_events: tool_start / tool_complete)
      complete       -> data        (token_usage state patch)
      error          -> text-delta  (inline error) + finish(error)
      interrupt      -> data        (HITL interrupt payload)
      context_compaction -> data    (activity_events: compaction notice)
    """

    client: Any = None  # Set after heavy imports
    _tool_event_queue: Optional[queue_mod.Queue] = None  # Shared tool event queue
    _event_counter: int = 0  # SSE event ID counter for reconnection
    _session_id: str = ""  # Set by launch_ink_chat
    _init_error: Optional[str] = None  # Stores init failure message
    _runtime_provider: str = ""
    _runtime_model: str = ""

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress default HTTP logging
        pass

    @staticmethod
    def _state_chunk(name: str, data: Any, *, transient: bool = True) -> dict:
        return {
            "type": f"data-{name}",
            "data": data,
            "transient": transient,
        }

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("Ink client disconnected during request handling")

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    # ------------------------------------------------------------------
    # GET endpoints
    # ------------------------------------------------------------------

    def do_GET(self) -> None:
        if self.path == "/health":
            ready = _DataStreamHandler.client is not None
            error = _DataStreamHandler._init_error
            self._send_json({
                "ready": ready,
                "error": error,
            })
        elif self.path == "/bootstrap":
            self._handle_bootstrap()
        elif self.path == "/config/flags":
            self._send_json(self._feature_flags())
        elif self.path == "/config/templates":
            self._send_json({"templates": []})
        elif self.path == "/resources":
            self._send_json({"resources": []})
        elif self.path.endswith("/messages"):
            self._send_json({"messages": []})
        else:
            self.send_response(404)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

    # ------------------------------------------------------------------
    # POST endpoints
    # ------------------------------------------------------------------

    def do_POST(self) -> None:
        if self.path.endswith("/chat/stream"):
            self._handle_chat_stream()
        elif self.path.endswith("/sessions"):
            self._send_json({"session_id": _DataStreamHandler._session_id})
        elif self.path.startswith("/sessions/") and "/commands/" in self.path:
            self._handle_slash_command()
        else:
            self.send_response(404)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

    # ------------------------------------------------------------------
    # Endpoint handlers
    # ------------------------------------------------------------------

    def _handle_chat_stream(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length)) if content_length > 0 else {}
        messages = body.get("messages", [])

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        try:
            self._stream_response(messages)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            # The launcher uses a single-threaded HTTPServer, so each request must
            # close promptly after completion instead of idling on keep-alive.
            self.close_connection = True

    def _handle_bootstrap(self) -> None:
        self._send_json({
            "flags": self._feature_flags(),
            "templates": [],
            "resources": [],
            "session": {"session_id": _DataStreamHandler._session_id},
            "runtime": {
                "provider": _DataStreamHandler._runtime_provider,
                "model": _DataStreamHandler._runtime_model,
                "mode": "local",
            },
        })

    def _handle_slash_command(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length)) if content_length > 0 else {}
        args = str(body.get("args", "")).strip()

        parts = self.path.split("/commands/", 1)
        command_path = unquote(parts[-1] if len(parts) > 1 else "").strip("/")
        command_tokens = [token for token in command_path.split("/") if token]

        if not command_tokens:
            self._send_json({
                "blocks": [{
                    "kind": "alert",
                    "level": "error",
                    "message": "No slash command was provided.",
                }],
            })
            return

        if self.client is None:
            self._send_json({
                "blocks": [{
                    "kind": "alert",
                    "level": "error",
                    "message": _DataStreamHandler._init_error or "Agent client not initialized.",
                }],
            })
            return

        normalized_command = "/" + " ".join(token.lower() for token in command_tokens)
        original_command = "/" + " ".join(command_tokens)
        if args:
            normalized_command = f"{normalized_command} {args}"
            original_command = f"{original_command} {args}"

        try:
            from lobster.cli_internal.commands import JsonOutputAdapter
            from lobster.cli_internal.commands.heavy.slash_commands import _execute_command

            output = JsonOutputAdapter()
            summary = _execute_command(
                normalized_command,
                self.client,
                original_command=original_command,
                output=output,
            )
            if summary:
                _log_slash_history_async(self.client, original_command, summary)

            payload = output.to_dict()
            payload["command"] = original_command
            if summary:
                payload["summary"] = summary
            self._send_json(payload)
        except Exception as exc:
            logger.exception("Failed to execute Ink slash command")
            self._send_json({
                "command": original_command,
                "error": str(exc),
                "blocks": [{
                    "kind": "alert",
                    "level": "error",
                    "message": str(exc),
                }],
            })

    @staticmethod
    def _feature_flags() -> dict:
        return {
            "projects_datasets": False,
            "session_state_read": False,
            "session_state_write": False,
            "message_ddb_primary": False,
            "curated_agent_store": False,
            "playground": False,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _send_json(self, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _stream_response(self, messages: list) -> None:
        """Stream LangGraph agent response as SSE events."""
        message_id = f"msg_{uuid.uuid4().hex[:8]}"

        self._send_sse({"type": "start", "messageId": message_id})

        if self.client is None:
            error_msg = _DataStreamHandler._init_error or "Agent client not initialized"
            self._send_sse(
                {
                    "type": "text-delta",
                    "textDelta": f"Error: {error_msg}\n\nPlease check your configuration with 'lobster init'.",
                }
            )
            self._send_sse({"type": "finish", "finishReason": "error"})
            self._send_done()
            return

        user_msg = messages[-1].get("content", "") if messages else ""

        try:
            for chunk in self._run_agent(user_msg):
                self._send_sse(chunk)
        except Exception as exc:
            logger.exception("Agent error during streaming")
            self._send_sse(
                {
                    "type": "text-delta",
                    "textDelta": f"\n\nError: {exc}",
                }
            )

        self._send_sse({"type": "finish", "finishReason": "stop"})
        self._send_done()

    def _run_agent(self, user_message: str):
        """Run the LangGraph agent and yield ui-message-stream SSE chunks.

        Invokes ``client.query(user_message, stream=True)`` and translates
        each LangGraph event into the SSE protocol expected by
        @assistant-ui/react-data-stream.

        Tool execution events arrive asynchronously via the
        ProtocolCallbackHandler -> _tool_event_queue pipeline and are
        drained between each agent stream event.
        """
        client = self.client
        if client is None:
            yield {
                "type": "text-delta",
                "textDelta": "Agent client not available.",
            }
            return

        cancel_event = threading.Event()
        active_agent = "supervisor"

        # Stream the agent response
        try:
            stream = client.query(user_message, stream=True, cancel_event=cancel_event)
        except Exception as exc:
            logger.exception("Failed to start agent query")
            yield {
                "type": "text-delta",
                "textDelta": f"Failed to start agent: {exc}",
            }
            return

        for event in stream:
            # Drain any pending tool events from the callback handler
            yield from self._drain_tool_events()

            etype = event.get("type", "")

            if etype == "content_delta":
                # Supervisor text — user-visible
                delta = event.get("delta", "")
                if delta:
                    yield {"type": "text-delta", "textDelta": delta}

            elif etype == "agent_content":
                # Specialist agent text — emit as activity event
                delta = event.get("delta", "")
                source = event.get("source", "specialist")
                if delta:
                    yield self._state_chunk(
                        "activity_events",
                        [{"type": "agent_content", "agent": source, "delta": delta}],
                    )

            elif etype == "agent_change":
                agent_name = event.get("agent", "")
                status = event.get("status", "working")
                if agent_name:
                    previous_agent = active_agent
                    active_agent = agent_name
                    yield self._state_chunk("active_agent", agent_name)
                    yield self._state_chunk("agent_status", status)
                    if agent_name != previous_agent:
                        yield self._state_chunk(
                            "activity_events",
                            [{
                                "type": "agent_transition",
                                "from_agent": previous_agent,
                                "to_agent": agent_name,
                                "status": "complete"
                                if agent_name == "supervisor"
                                else status,
                                "task_description": event.get("reason")
                                or event.get("task_description"),
                            }],
                        )

            elif etype == "complete":
                # Final completion — emit token usage and session title
                token_usage = event.get("token_usage", {})
                duration = event.get("duration", 0)
                if duration:
                    token_usage["duration"] = duration
                yield self._state_chunk("token_usage", token_usage)
                yield self._state_chunk("active_agent", "supervisor")
                session_title = event.get("session_title", "")
                if session_title:
                    yield self._state_chunk("session_title", session_title)

            elif etype == "error":
                error_msg = str(event.get("error", "Unknown error"))
                is_rate_limit = event.get("is_rate_limit", False)
                yield {
                    "type": "text-delta",
                    "textDelta": f"\n\nError: {error_msg}",
                }
                if is_rate_limit:
                    yield self._state_chunk(
                        "activity_events",
                        [{"type": "error", "error_type": "rate_limit", "message": error_msg}],
                    )

            elif etype == "interrupt":
                # HITL interrupt — emit as activity event so the
                # frontend can render an interactive component
                yield self._state_chunk(
                    "activity_events",
                    [{
                        "type": "interrupt",
                        "interrupt_data": event.get("data", {}),
                        "interrupt_id": event.get("interrupt_id"),
                    }],
                )

            elif etype == "context_compaction":
                yield self._state_chunk(
                    "activity_events",
                    [{
                        "type": "context_compaction",
                        "agent": event.get("agent", ""),
                        "before_count": event.get("before_count"),
                        "after_count": event.get("after_count"),
                        "budget_tokens": event.get("budget_tokens"),
                    }],
                )

        # Drain any remaining tool events after stream completes
        yield from self._drain_tool_events()

    def _drain_tool_events(self):
        """Drain pending tool execution events from the callback queue.

        The ProtocolCallbackHandler pushes tool_execution and
        agent_transition events into ``_tool_event_queue``.  We convert
        them to the ui-message-stream format here.
        """
        eq = _DataStreamHandler._tool_event_queue
        if eq is None:
            return

        while True:
            try:
                msg_type, payload = eq.get_nowait()
            except queue_mod.Empty:
                break

            if msg_type == "tool_execution":
                status = payload.get("status", "")
                tool_name = payload.get("tool", payload.get("tool_name", ""))
                tool_call_id = payload.get("tool_call_id", "") or f"tc_{uuid.uuid4().hex[:8]}"
                agent = payload.get("agent", "")

                if status == "running":
                    yield self._state_chunk(
                        "activity_events",
                        [{
                            "type": "tool_start",
                            "tool_name": tool_name,
                            "tool_call_id": tool_call_id,
                            "agent": agent,
                        }],
                    )
                elif status in ("complete", "error"):
                    duration_ms = payload.get("duration_ms")
                    evt: Dict[str, Any] = {
                        "type": "tool_error" if status == "error" else "tool_complete",
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "agent": agent,
                    }
                    if duration_ms is not None:
                        evt["duration_ms"] = duration_ms
                    summary = payload.get("summary")
                    if summary:
                        evt["summary"] = summary
                    if status == "error":
                        evt["error"] = payload.get("summary") or True
                    yield self._state_chunk("activity_events", [evt])

            elif msg_type == "agent_transition":
                from_agent = payload.get("from", "")
                to_agent = payload.get("to", "")
                if to_agent:
                    yield self._state_chunk("active_agent", to_agent)
                    yield self._state_chunk(
                        "activity_events",
                        [{
                            "type": "agent_handoff",
                            "from_agent": from_agent,
                            "to_agent": to_agent,
                            "task_description": payload.get("reason"),
                            "kind": payload.get("kind"),
                            "status": payload.get("status"),
                        }],
                    )

            elif msg_type == "progress":
                label = str(payload.get("label", "")).strip()
                if label:
                    yield self._state_chunk(
                        "progress",
                        {
                            label: {
                                "label": label,
                                "current": payload.get("current", 0),
                                "total": payload.get("total", -1),
                                "done": payload.get("done", False),
                            },
                        },
                    )

    def _send_sse(self, data: dict) -> None:
        _DataStreamHandler._event_counter += 1
        self.wfile.write(f"id: {_DataStreamHandler._event_counter}\n".encode())
        self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
        self.wfile.flush()

    def _send_done(self) -> None:
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


def launch_ink_chat(
    *,
    workspace: Optional[str] = None,
    session_id: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    debug: bool = False,
) -> None:
    """Launch the Ink TUI and wire it to the Lobster agent client.

    1. Start local DataStream HTTP server on random port (stdlib only)
    2. Spawn lobster-chat binary with --api-url
    3. Heavy-import the agent client while TUI displays
    4. Wire agent client into the HTTP handler
    5. Wait for the TUI process to exit
    """
    binary = find_ink_binary()
    if binary is None:
        print("Error: lobster-chat binary not found.", file=sys.stderr)
        print("Run: cd lobster-tui-ink && bun run build", file=sys.stderr)
        sys.exit(1)

    # Start HTTP server
    port = _find_free_port()
    sid = session_id or uuid.uuid4().hex[:12]
    api_url = f"http://127.0.0.1:{port}"

    # Store session ID for handler access
    _DataStreamHandler._session_id = sid

    handler = partial(_DataStreamHandler)
    server = HTTPServer(("127.0.0.1", port), handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    logger.debug("Local DataStream server on port %d", port)

    # Spawn the TUI binary
    cmd = [binary, f"--api-url={api_url}", f"--session-id={sid}"]
    if session_id:
        cmd.append("--resume-session")

    proc = subprocess.Popen(
        cmd,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=subprocess.PIPE if debug else subprocess.DEVNULL,
    )
    terminal_quarantine: Optional[_PythonTerminalQuarantine] = None
    terminal_quarantine = _PythonTerminalQuarantine.activate()

    # Drain stderr in a background thread to prevent pipe deadlock
    if debug:
        def _drain_stderr(p: subprocess.Popen) -> None:
            assert p.stderr is not None
            for line in p.stderr:
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                logger.debug("TUI stderr: %s", line.rstrip())

        stderr_thread = threading.Thread(target=_drain_stderr, args=(proc,), daemon=True)
        stderr_thread.start()

    # Heavy imports happen here, while TUI is already displaying
    try:
        from lobster.cli_internal.commands.heavy.session_infra import (
            init_client_or_raise_startup_diagnostic,
            set_go_tui_active,
        )
        from lobster.cli_internal.startup_diagnostics import StartupDiagnosticError

        # Suppress Rich progress indicators — the Ink TUI has its own.
        set_go_tui_active(True)

        client = init_client_or_raise_startup_diagnostic(
            workspace=Path(workspace) if workspace else None,
            session_id=sid,
            provider_override=provider,
            model_override=model,
        )

        # Wire the ProtocolCallbackHandler so tool execution events
        # flow through to the DataStream SSE bridge.
        tool_event_queue: queue_mod.Queue = queue_mod.Queue()
        _DataStreamHandler._tool_event_queue = tool_event_queue

        try:
            from lobster.ui.callbacks.protocol_callback import (
                ProtocolCallbackHandler,
            )

            proto_callback = ProtocolCallbackHandler(
                emit_event=lambda msg_type, payload: tool_event_queue.put(
                    (msg_type, payload)
                )
            )
            client.callbacks.append(proto_callback)
            logger.debug("ProtocolCallbackHandler wired to tool event queue")
        except ImportError:
            logger.debug("ProtocolCallbackHandler not available, tool events disabled")

        # Wire the client into the handler
        _DataStreamHandler.client = client
        _DataStreamHandler._runtime_provider = _resolve_active_provider_name(client)
        _DataStreamHandler._runtime_model = str(
            getattr(client, "model_override", None)
            or getattr(client, "model", None)
            or ""
        ).strip()
        logger.debug("Agent client initialized, handler wired")
    except Exception as exc:
        logger.error("Failed to initialize agent client: %s", exc)
        _DataStreamHandler._init_error = str(exc)

    # Wait for TUI to exit
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
    finally:
        if terminal_quarantine is not None:
            terminal_quarantine.restore()
        server.shutdown()
        # Reset Go TUI active flag so Rich output works again
        try:
            from lobster.cli_internal.commands.heavy.session_infra import (
                set_go_tui_active,
            )
            set_go_tui_active(False)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Init wizard launcher
# ---------------------------------------------------------------------------


def run_ink_init_wizard() -> dict:
    """Launch the Ink TUI init wizard and return the result dict.

    1. Find lobster-chat binary
    2. Build WizardManifest and write to a temp file
    3. Run lobster-chat --init with the real terminal attached
    4. Read JSON result from a temp file
    5. Convert WizardResult -> legacy dict format for apply_tui_init_result()

    Raises:
        ImportError: if binary not found
        RuntimeError: if wizard fails
    """
    import tempfile

    binary = find_ink_binary()
    if binary is None:
        raise ImportError("lobster-chat binary not found")

    from lobster.ui.wizard.manifest import build_init_manifest

    manifest = build_init_manifest()
    manifest_json = manifest.to_json()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(manifest_json)
        manifest_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        result_path = f.name

    try:
        proc = subprocess.run(
            [
                binary,
                "--init",
                "--manifest-file",
                manifest_path,
                "--result-file",
                result_path,
            ],
            stdin=sys.stdin,
            text=True,
            timeout=300,
        )

        try:
            raw_payload = Path(result_path).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            raw_payload = ""

        if not raw_payload:
            if proc.returncode in (0, 130, -signal.SIGINT):
                return {"cancelled": True}
            raise RuntimeError(f"Ink wizard exited with code {proc.returncode}")

        try:
            result = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ink wizard returned invalid JSON") from exc

        if not isinstance(result, dict):
            raise RuntimeError("Ink wizard result must be a JSON object")

        if result.get("cancelled", False):
            return {"cancelled": True}
        return _convert_ink_result_to_legacy(result)

    finally:
        try:
            os.unlink(manifest_path)
        except FileNotFoundError:
            pass
        try:
            os.unlink(result_path)
        except FileNotFoundError:
            pass


def _convert_ink_result_to_legacy(ink_result: dict) -> dict:
    """Convert Ink WizardResult to the legacy dict format expected by _postprocess_tui_init_result."""
    legacy = {
        "provider": ink_result.get("provider", ""),
        "api_key": "",
        "api_key_secondary": "",
        "profile": ink_result.get("profile") or "",
        "agents": ink_result.get("selectedPackages", []),
        "ncbi_key": ink_result.get("optionalKeys", {}).get("NCBI_API_KEY", ""),
        "cloud_key": ink_result.get("optionalKeys", {}).get("OMICS_OS_CLOUD_KEY", ""),
        "ollama_model": "",
        "model_id": ink_result.get("model") or "",
        "smart_standardization_enabled": ink_result.get("smartStandardization", False),
        "smart_standardization_openai_key": "",
        "cancelled": False,
    }

    creds = ink_result.get("credentials", {})
    provider = ink_result.get("provider", "")

    # Map credentials to legacy api_key/api_key_secondary
    if provider == "anthropic":
        legacy["api_key"] = creds.get("ANTHROPIC_API_KEY", "")
    elif provider == "bedrock":
        legacy["api_key"] = creds.get("AWS_ACCESS_KEY_ID", "")
        legacy["api_key_secondary"] = creds.get("AWS_SECRET_ACCESS_KEY", "")
    elif provider == "gemini":
        legacy["api_key"] = creds.get("GOOGLE_API_KEY", "")
    elif provider == "azure":
        legacy["api_key"] = creds.get("AZURE_OPENAI_API_KEY", "")
        legacy["api_key_secondary"] = creds.get("AZURE_OPENAI_ENDPOINT", "")
    elif provider == "openai":
        legacy["api_key"] = creds.get("OPENAI_API_KEY", "")
    elif provider == "openrouter":
        legacy["api_key"] = creds.get("OPENROUTER_API_KEY", "")
    elif provider == "omics-os":
        legacy["api_key"] = creds.get("OMICS_OS_API_KEY", "")

    return legacy
