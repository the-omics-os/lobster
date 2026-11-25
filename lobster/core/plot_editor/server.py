"""Loopback-only HTTP adapter with scoped bearer sessions and no filesystem routes."""

from __future__ import annotations

import atexit
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lobster.core.figures.store import LocalFigureStore

from urllib.parse import parse_qs, urlencode, urlsplit

from lobster.core.figures.errors import FigureError, safe_reason
from lobster.core.figures.ids import FigureRef
from lobster.core.figures.payload import MAX_FIGURE_BYTES, strict_json_loads
from lobster.core.figures.records import json_blob_bytes

from .browser import open_browser
from .service import EditorService

_sessions: dict[tuple[int, str], EditorSession] = {}
_registry_lock = threading.RLock()
IDLE_SECONDS = 300


class EditorSession:
    def __init__(self, service: EditorService) -> None:
        from plotly.offline import get_plotlyjs

        self.service = service
        self.token = secrets.token_urlsafe(32)
        self.last_seen = time.monotonic()
        self._stop = threading.Event()
        self.connected = threading.Event()
        self._state_lock = threading.Lock()
        self._active = 0
        self._plotly = get_plotlyjs().encode("utf-8")
        # Bundled frontend assets are explicitly packaged and whitelisted below.
        assets = {
            name: (Path(__file__).with_name("assets") / name).read_text(
                encoding="utf-8"
            )
            for name in ("index.html", "editor.js", "editor.css")
        }
        session = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self) -> None:
                super().setup()
                self.connection.settimeout(15)

            def log_message(self, format: str, *args: Any) -> None:
                pass  # no tokens, user data or request URLs in logs

            def reply(
                self,
                status: int,
                value: dict[str, Any] | bytes,
                mime: str = "application/json",
            ) -> None:
                data = value if isinstance(value, bytes) else json_blob_bytes(value)
                self.send_response(status)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'none'; script-src 'self' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; worker-src blob:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
                )
                self.end_headers()
                self.wfile.write(data)

            def dispatch(self) -> None:
                active = False
                try:
                    if self.headers.get_all("Host") != [session.authority]:
                        return self.reply(403, {"error": "Invalid Host"})
                    origins = self.headers.get_all("Origin")
                    if origins is not None and origins != [session.origin]:
                        return self.reply(403, {"error": "Invalid Origin"})
                    if self.headers.get("Sec-Fetch-Site") == "cross-site":
                        return self.reply(403, {"error": "Cross-site request refused"})
                    url = urlsplit(self.path)
                    routes = {
                        "/": ("index.html", "text/html; charset=utf-8"),
                        "/editor.js": ("editor.js", "text/javascript; charset=utf-8"),
                        "/editor.css": ("editor.css", "text/css; charset=utf-8"),
                    }
                    if self.command == "GET" and url.path in routes:
                        name, mime = routes[url.path]
                        return self.reply(200, assets[name].encode("utf-8"), mime)
                    if self.command == "GET" and url.path == "/plotly.js":
                        return self.reply(
                            200, session._plotly, "text/javascript; charset=utf-8"
                        )
                    auth = self.headers.get_all("Authorization") or []
                    if len(auth) != 1 or not secrets.compare_digest(
                        auth[0], "Bearer " + session.token
                    ):
                        return self.reply(
                            403, {"error": "Editor authorization required"}
                        )
                    with session._state_lock:
                        if session.closed:
                            return self.reply(410, {"error": "Editor session expired"})
                        session._active += 1
                        session.last_seen = time.monotonic()
                    active = True
                    if self.command == "GET":
                        if url.path == "/api/figure":
                            refs = parse_qs(url.query).get("ref", [service.initial_ref])
                            if len(refs) != 1:
                                raise ValueError("Ambiguous reference")
                            loaded = service.load(refs[0])
                            session.connected.set()
                            return self.reply(200, loaded)
                        if url.path == "/api/history":
                            return self.reply(200, service.history())
                        if url.path == "/api/data":
                            query = parse_qs(url.query)
                            if not set(query) <= {"ref", "offset", "limit"} or any(
                                len(v) != 1 for v in query.values()
                            ):
                                raise ValueError("Invalid data page query")
                            return self.reply(
                                200,
                                service.data_page(
                                    query.get("ref", [service.initial_ref])[0],
                                    offset=int(query.get("offset", ["0"])[0]),
                                    limit=int(query.get("limit", ["100"])[0]),
                                ),
                            )
                        if url.path == "/api/attachment":
                            query = parse_qs(url.query)
                            return self.reply(
                                200, service.attachment(query["ref"][0], query["id"][0])
                            )
                        if url.path == "/api/heartbeat":
                            return self.reply(200, {"ok": True})
                    if self.command == "POST":
                        if origins != [session.origin] or self.headers.get_all(
                            "Content-Type"
                        ) != ["application/json"]:
                            return self.reply(
                                403, {"error": "Same-origin JSON required"}
                            )
                        lengths = self.headers.get_all("Content-Length") or []
                        if len(lengths) != 1 or self.headers.get("Transfer-Encoding"):
                            return self.reply(400, {"error": "Invalid request framing"})
                        size = int(lengths[0])
                        if not 0 <= size <= MAX_FIGURE_BYTES + 28_000_000:
                            return self.reply(
                                413, {"error": "Editor request too large"}
                            )
                        data = self.rfile.read(size)
                        if len(data) != size:
                            raise ValueError("Incomplete request")
                        draft = strict_json_loads(data, what="editor draft")
                        if url.path == "/api/delete":
                            return self.reply(200, service.delete(draft))
                        if url.path == "/api/validate":
                            service.validate(draft)
                            return self.reply(200, {"valid": True})
                        if url.path == "/api/save":
                            return self.reply(200, service.save(draft))
                    self.reply(404, {"error": "Unknown editor operation"})
                except FigureError as exc:
                    self.reply(400, {"error": safe_reason(exc)})
                except ValueError as exc:
                    self.reply(409, {"error": str(exc)})
                except (TypeError, KeyError):
                    self.reply(400, {"error": "Invalid editor request"})
                except (OSError, TimeoutError):
                    pass
                except Exception:
                    self.reply(
                        500, {"error": "Editor operation failed; reopen or retry"}
                    )
                finally:
                    if active:
                        with session._state_lock:
                            session._active -= 1
                            session.last_seen = time.monotonic()

            do_GET = dispatch
            do_POST = dispatch

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.http.daemon_threads = True
        self.authority = f"127.0.0.1:{self.http.server_port}"
        self.origin = "http://" + self.authority
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        try:
            self.thread.start()
            threading.Thread(target=self._watchdog, daemon=True).start()
        except Exception:
            self._stop.set()
            if self.thread.is_alive():
                self.http.shutdown()
            self.http.server_close()
            raise

    @property
    def closed(self) -> bool:
        return self._stop.is_set()

    def url(self, ref: str) -> str:
        ref = self.service.reference(ref)
        return self.origin + "/?" + urlencode({"ref": ref}) + "#" + self.token

    def _watchdog(self) -> None:
        while not self._stop.wait(10):
            with self._state_lock:
                expired = (
                    self._active == 0
                    and time.monotonic() - self.last_seen > IDLE_SECONDS
                )
            if expired:
                self.close(only_if_idle=True)
                if self.closed:
                    return

    def close(self, *, only_if_idle: bool = False) -> None:
        with self._state_lock:
            if self.closed:
                return
            if only_if_idle and (
                self._active or time.monotonic() - self.last_seen <= IDLE_SECONDS
            ):
                return
            self._stop.set()
        self.http.shutdown()
        # Finish an in-flight trusted save before normal shutdown returns.
        self.service.wait_for_saves()
        self.http.server_close()
        with _registry_lock:
            for key, value in list(_sessions.items()):
                if value is self:
                    _sessions.pop(key, None)


def launch_editor(store: LocalFigureStore, ref: str) -> dict[str, Any]:
    """Reuse one server per store/figure; tabs may independently edit any revision."""
    ref = str(FigureRef.parse(ref))
    key = (id(store), FigureRef.parse(ref).figure_id)
    with _registry_lock:
        session = _sessions.get(key)
        created = session is None or session.closed
        if session is None or session.closed:
            session = EditorSession(EditorService(store, ref))
            _sessions[key] = session
        else:
            session.service.load(ref)  # recheck current integrity and availability
        session.last_seen = time.monotonic()
        try:
            session.connected.clear()
            mode = open_browser(session.url(ref))
            connected = session.connected.wait(3)
            return {
                "status": "editor_connected" if connected else "launch_requested",
                "mode": mode,
                "message": (
                    "Editor loaded the saved figure."
                    if connected
                    else "Browser launch requested; no editor connection observed yet. Keep Lobster running."
                ),
            }
        except Exception:
            # Existing windows keep their working server if a repeat launch fails.
            if created:
                session.close()
            raise


@atexit.register
def close_editors() -> None:
    for session in list(_sessions.values()):
        session.close()
