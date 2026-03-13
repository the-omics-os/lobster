"""Ink TUI chat launcher — zero heavy dependencies at module level.

Mirrors go_tui_launcher.py's design philosophy: stdlib only at module level,
heavy imports deferred until the TUI binary is already displaying.

Architecture:
  1. Start a lightweight local HTTP server on a random port
  2. Spawn the lobster-chat binary with --api-url=http://localhost:{port}
  3. The HTTP server bridges DataStream SSE to the LangGraph agent

The server implements the ui-message-stream protocol (SSE with JSON payloads),
the same format consumed by the Cloud web app.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import uuid
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


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
    dev_path = Path(__file__).resolve().parents[2] / "lobster-tui-ink" / "dist" / "lobster-chat"
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
    """HTTP handler that bridges DataStream SSE to the LangGraph agent."""

    client: Any = None  # Set after heavy imports

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress default HTTP logging
        pass

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_POST(self) -> None:
        if not self.path.endswith("/chat/stream"):
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length)) if content_length > 0 else {}
        messages = body.get("messages", [])

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        try:
            self._stream_response(messages)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _stream_response(self, messages: list) -> None:
        """Stream LangGraph agent response as SSE events."""
        message_id = f"msg_{uuid.uuid4().hex[:8]}"

        self._send_sse({"type": "start", "messageId": message_id})

        if self.client is None:
            self._send_sse({
                "type": "text-delta",
                "textDelta": "Error: Agent client not initialized. Please wait for startup to complete.",
            })
            self._send_sse({"type": "finish", "finishReason": "error"})
            self._send_done()
            return

        user_msg = messages[-1].get("content", "") if messages else ""

        try:
            # Delegate to the agent client's stream method
            # This will be wired to the LangGraph agent in a future step
            for chunk in self._run_agent(user_msg):
                self._send_sse(chunk)
        except Exception as exc:
            logger.exception("Agent error")
            self._send_sse({
                "type": "text-delta",
                "textDelta": f"\n\nError: {exc}",
            })

        self._send_sse({"type": "finish", "finishReason": "stop"})
        self._send_done()

    def _run_agent(self, user_message: str):
        """Run the LangGraph agent and yield SSE chunks.

        TODO: Wire to actual LangGraph agent via AgentClient.
        Currently yields a placeholder response.
        """
        yield {"type": "text-delta", "textDelta": f"Received: {user_message}\n\n"}
        yield {"type": "text-delta", "textDelta": "(Local DataStream server running — agent wiring pending)"}

    def _send_sse(self, data: dict) -> None:
        line = f"data: {json.dumps(data)}\n\n"
        self.wfile.write(line.encode())
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

    handler = partial(_DataStreamHandler)
    server = HTTPServer(("127.0.0.1", port), handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    logger.debug("Local DataStream server on port %d", port)

    # Spawn the TUI binary
    cmd = [binary, f"--api-url={api_url}", f"--session-id={sid}"]

    proc = subprocess.Popen(
        cmd,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=subprocess.PIPE if debug else subprocess.DEVNULL,
    )

    # Heavy imports happen here, while TUI is already displaying
    try:
        from lobster.cli_internal.commands.heavy.session_infra import (
            init_client_or_raise_startup_diagnostic,
        )
        from lobster.cli_internal.startup_diagnostics import StartupDiagnosticError

        client = init_client_or_raise_startup_diagnostic(
            workspace=Path(workspace) if workspace else None,
            session_id=sid,
            provider_override=provider,
            model_override=model,
        )
        # Wire the client into the handler
        _DataStreamHandler.client = client
        logger.debug("Agent client initialized, handler wired")
    except Exception as exc:
        logger.error("Failed to initialize agent client: %s", exc)

    # Wait for TUI to exit
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
    finally:
        server.shutdown()
