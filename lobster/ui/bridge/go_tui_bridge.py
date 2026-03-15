"""Synchronous subprocess bridge for lobster-tui."""

from __future__ import annotations

import json
import logging
import os
import queue
import signal
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

from lobster.ui.bridge.protocol import (
    PROTOCOL_VERSION,
    MessageType,
    ProtocolMessage,
    new_message,
    new_request,
)

logger = logging.getLogger(__name__)


class BridgeError(RuntimeError):
    """Raised when the Go TUI bridge fails."""


class GoTUIBridge:
    """Manages JSON-lines IPC with `lobster-tui`."""

    def __init__(
        self,
        binary_path: str,
        *,
        mode: str = "chat",
        theme: str = "",
        debug: bool = False,
    ):
        self.binary_path = str(Path(binary_path))
        self.mode = mode
        self.theme = theme
        self.debug = debug

        self._process: Optional[subprocess.Popen] = None
        self._writer = None
        self._reader = None
        self._read_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._events: "queue.Queue[ProtocolMessage]" = queue.Queue()
        self._pending: dict[str, "queue.Queue[ProtocolMessage]"] = {}
        self._pending_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._running = False

    def start(self) -> None:
        """Start the Go TUI process and protocol loops."""
        if self._running:
            return

        p2g_r, p2g_w = os.pipe()
        g2p_r, g2p_w = os.pipe()

        cmd = [
            self.binary_path,
            self.mode,
            "--proto-fd-in",
            str(p2g_r),
            "--proto-fd-out",
            str(g2p_w),
        ]
        if self.theme:
            cmd.extend(["--theme", self.theme])

        stderr_pipe = subprocess.PIPE
        try:
            self._process = subprocess.Popen(
                cmd,
                pass_fds=(p2g_r, g2p_w),
                stdout=subprocess.DEVNULL,
                stderr=stderr_pipe,
                text=False,
                preexec_fn=os.setsid,
            )
        except OSError as e:
            raise BridgeError(f"Failed to start lobster-tui: {e}") from e
        finally:
            # Child owns these now.
            os.close(p2g_r)
            os.close(g2p_w)

        self._writer = os.fdopen(p2g_w, "w", encoding="utf-8", buffering=1)
        self._reader = os.fdopen(g2p_r, "r", encoding="utf-8", buffering=1)

        self._running = True
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()

        if self._process.stderr:
            self._stderr_thread = threading.Thread(
                target=self._stderr_loop, daemon=True
            )
            self._stderr_thread.start()

        handshake = self.recv_event(timeout=5.0)
        if handshake is None or handshake.type != MessageType.HANDSHAKE.value:
            raise BridgeError("Did not receive valid handshake from lobster-tui")
        version = (handshake.payload or {}).get("protocol_version")
        if version != PROTOCOL_VERSION:
            raise BridgeError(
                f"Protocol mismatch: python={PROTOCOL_VERSION}, go={version}"
            )

    def close(self) -> None:
        """Close bridge and terminate child process."""
        if not self._running:
            return

        self._running = False

        try:
            self.send(MessageType.QUIT.value, {})
        except Exception:
            pass

        try:
            if self._writer:
                self._writer.close()
        except Exception:
            pass
        try:
            if self._reader:
                self._reader.close()
        except Exception:
            pass

        proc = self._process
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

        self._process = None

    def send(
        self,
        msg_type: str,
        payload: Optional[dict] = None,
        *,
        msg_id: str | None = None,
    ) -> None:
        """Send a non-blocking message to the Go process."""
        msg = ProtocolMessage(type=msg_type, id=msg_id, payload=payload)
        self._send_message(msg)

    def request(
        self, msg_type: str, payload: Optional[dict] = None, *, timeout: float = 30.0
    ) -> Optional[dict]:
        """Send a blocking request and wait for response with matching id."""
        request = new_request(msg_type, payload)
        slot: "queue.Queue[ProtocolMessage]" = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[request.id or ""] = slot

        self._send_message(request)
        try:
            response = slot.get(timeout=timeout)
            return response.payload or {}
        except queue.Empty as e:
            raise BridgeError(f"Timed out waiting for response to {msg_type}") from e
        finally:
            with self._pending_lock:
                self._pending.pop(request.id or "", None)

    def recv_event(
        self, *, timeout: Optional[float] = None
    ) -> Optional[ProtocolMessage]:
        """Receive next asynchronous event."""
        try:
            return self._events.get(timeout=timeout)
        except queue.Empty:
            return None

    def _send_message(self, message: ProtocolMessage) -> None:
        if not self._running or not self._writer:
            raise BridgeError("Bridge is not running")

        line = message.to_json_line()
        with self._write_lock:
            self._writer.write(line + "\n")
            self._writer.flush()

    def _read_loop(self) -> None:
        if not self._reader:
            return

        while self._running:
            try:
                line = self._reader.readline()
            except Exception:
                return

            if not line:
                return

            line = line.strip()
            if not line:
                continue

            try:
                msg = ProtocolMessage.from_json_line(line)
            except json.JSONDecodeError:
                logger.warning("Invalid protocol line from lobster-tui: %s", line[:120])
                continue

            handled = False
            if msg.id:
                with self._pending_lock:
                    slot = self._pending.get(msg.id)
                if slot is not None:
                    slot.put(msg)
                    handled = True

            if not handled:
                self._events.put(msg)

    def _stderr_loop(self) -> None:
        if not self._process or not self._process.stderr:
            return

        while self._running:
            line = self._process.stderr.readline()
            if not line:
                break
            if self.debug:
                try:
                    decoded = line.decode("utf-8", errors="replace").rstrip()
                except Exception:
                    decoded = str(line).rstrip()
                logger.debug("lobster-tui: %s", decoded)


def run_init_wizard(binary_path: str, *, theme: str = "", timeout: int = 300) -> dict:
    """Run one-shot Go init wizard and parse its JSON result file.

    The Go wizard needs the real terminal for interactive rendering, so the
    result is written to a temporary file instead of stdout capture.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", prefix="lobster-init-", delete=False
    ) as result_file:
        result_path = result_file.name

    cmd = [binary_path, "init", "--result-file", result_path]
    if theme:
        cmd.extend(["--theme", theme])
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            timeout=timeout,
        )

        try:
            raw_payload = Path(result_path).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            raw_payload = ""

        if raw_payload:
            try:
                payload = json.loads(raw_payload)
            except json.JSONDecodeError as e:
                raise BridgeError("lobster-tui init returned invalid JSON") from e
        else:
            payload = None
    finally:
        try:
            Path(result_path).unlink()
        except FileNotFoundError:
            pass

    if payload is None:
        if proc.returncode != 0:
            raise BridgeError(f"lobster-tui init failed (exit code {proc.returncode})")
        raise BridgeError("lobster-tui init returned empty output")

    if not isinstance(payload, dict):
        raise BridgeError("lobster-tui init JSON payload must be an object")
    return payload
