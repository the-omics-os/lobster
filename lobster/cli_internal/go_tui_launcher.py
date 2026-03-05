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
import logging
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

_PROTOCOL_VERSION = 1


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
        import lobster_ai_tui  # type: ignore[import-untyped]

        pkg_dir = Path(lobster_ai_tui.__file__).parent
        candidate = pkg_dir / "lobster-tui"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    except (ImportError, AttributeError, TypeError):
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

    def send(self, msg_type: str, payload: Optional[dict] = None) -> None:
        """Send a message to the Go TUI.  No-op if bridge is closed."""
        if not self._running:
            return
        line = _make_message(msg_type, payload)
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
        "summary": summary,
    }


def _format_usage(client: Any) -> str:
    """Format token/cost usage from the client's tracker."""
    try:
        tracker = client.token_tracker
        return f"Tokens: {tracker.total_tokens:,} | Cost: ${tracker.total_cost:.4f}"
    except Exception:
        return "Ready"


# ---------------------------------------------------------------------------
# Event loop
# ---------------------------------------------------------------------------


def _go_tui_event_loop(bridge: _LightBridge, client: Any) -> None:
    """Main event loop: dispatch messages from the Go TUI to the client."""
    while True:
        event = bridge.recv_event(timeout=None)
        if event is None:
            break
        msg_type = event.get("type", "")
        if msg_type == "quit":
            break
        elif msg_type == "input":
            content = event.get("payload", {}).get("content", "")
            _handle_user_query(bridge, client, content)
        elif msg_type == "slash_command":
            payload = event.get("payload", {})
            command = payload.get("command", "")
            args = payload.get("args", "")
            _handle_slash_command(bridge, client, command, args)
        elif msg_type == "cancel":
            pass  # Phase 2: cancellation support


def _handle_user_query(bridge: _LightBridge, client: Any, text: str) -> None:
    """Forward a user query to the Lobster client and stream results back."""
    bridge.send("spinner", {"active": True})
    query_start = time.time()
    try:
        for event in client.query(text, stream=True):
            etype = event["type"]
            if etype == "content_delta":
                bridge.send("text", {"content": event["delta"]})
            elif etype == "agent_change":
                bridge.send(
                    "agent_transition",
                    {
                        "from": event.get("from_agent", "supervisor"),
                        "to": event.get("agent", ""),
                        "reason": event.get("status", "working"),
                    },
                )
            elif etype == "complete":
                bridge.send("done", {"summary": ""})
                # Build rich status with tokens, cost, and duration.
                duration = time.time() - query_start
                status_parts = [_format_usage(client)]
                status_parts.append(f"Duration: {duration:.1f}s")
                bridge.send("status", {"text": " · ".join(status_parts)})
                bridge.send("spinner", {"active": False})
            elif etype == "error":
                bridge.send(
                    "alert",
                    {
                        "level": "error",
                        "message": str(event.get("error", "Unknown error")),
                    },
                )
                bridge.send("spinner", {"active": False})
    except KeyboardInterrupt:
        bridge.send(
            "alert", {"level": "warning", "message": "Query interrupted"}
        )
        bridge.send("spinner", {"active": False})
    except Exception as exc:
        bridge.send("alert", {"level": "error", "message": f"Error: {exc}"})
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
        _execute_command(full_cmd, client, original_command=full_cmd, output=output)
    except Exception as exc:
        bridge.send("alert", {"level": "error", "message": str(exc)})
    finally:
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

    cmd = [
        binary,
        "chat",
        "--proto-fd-in",
        str(p2g_r),
        "--proto-fd-out",
        str(g2p_w),
        "--theme",
        "lobster-dark",
    ]

    proc = subprocess.Popen(
        cmd,
        pass_fds=(p2g_r, g2p_w),
        # stdout inherited — BubbleTea needs the real terminal to render.
        # Protocol communication uses separate FDs (the pipes), not stdout.
        stderr=subprocess.PIPE if debug else subprocess.DEVNULL,
        text=False,
        preexec_fn=os.setsid,
    )

    # Close the child-side FDs in the parent process.
    os.close(p2g_r)
    os.close(g2p_w)

    bridge = _LightBridge(proc, p2g_w, g2p_r)

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
            init_client,
            set_go_tui_active,
        )

        set_go_tui_active(True)

        client = init_client(
            workspace=workspace,
            reasoning=False,
            verbose=False,
            debug=debug,
            profile_timings=profile_timings,
            provider_override=provider,
            model_override=model,
            session_id=session_id,
        )

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
        bridge.send("status", {"text": "Ready"})

        # -----------------------------------------------------------------
        # 8. Enter event loop
        # -----------------------------------------------------------------
        _go_tui_event_loop(bridge, client)

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
