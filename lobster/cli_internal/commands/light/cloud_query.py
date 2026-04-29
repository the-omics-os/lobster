"""Cloud query — single-turn query against Omics-OS Cloud backend (ECS Fargate)."""

import json
import logging
import re
import threading
from typing import Callable, Optional

import httpx

from lobster.config.credentials import get_api_key, get_endpoint

logger = logging.getLogger(__name__)

REST_API_BASE = "https://app.omics-os.com"
STREAM_API_BASE = "https://stream.omics-os.com"

_ALLOWED_HOSTS = frozenset({
    "app.omics-os.com",
    "stream.omics-os.com",
    "localhost",
    "127.0.0.1",
    "::1",
})

# Covers CSI, OSC (BEL and ST terminated), C1 controls, single-char ESC sequences
_ANSI_ESCAPE_RE = re.compile(
    r"\x1b\[[0-9;?]*[A-Za-z]"       # CSI (including private-mode ?-prefixed)
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC (BEL or ST terminated)
    r"|\x1b[PX^_][^\x1b]*\x1b\\"    # DCS/SOS/PM/APC strings
    r"|\x1b[A-Z@-_]"                # single-char ESC sequences
    r"|[\x00-\x08\x0e-\x1f\x7f]"   # C0 controls (except \t \n \r)
    r"|[\x80-\x9f]"                 # C1 controls
)

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

MAX_RESPONSE_SIZE = 10 * 1024 * 1024  # 10 MB


# P0-MIGRATE: replace with lobster.cloud.errors.CloudError hierarchy
class CloudQueryError(Exception):
    """Typed error for JSON-safe error reporting."""
    pass


# P0-MIGRATE: replace with CloudClient endpoint validation (P0 amendment A9)
def _validate_endpoint(endpoint: str) -> None:
    """Reject endpoints not in allowlist to prevent token exfiltration."""
    from urllib.parse import urlparse

    parsed = urlparse(endpoint)
    hostname = parsed.hostname or ""
    if hostname not in _ALLOWED_HOSTS:
        raise CloudQueryError(
            f"Endpoint '{hostname}' not in allowlist. "
            f"Allowed: {', '.join(sorted(_ALLOWED_HOSTS))}. "
            f"Use --unsafe-endpoint to override."
        )
    if parsed.scheme not in ("https", "http"):
        raise CloudQueryError("Only https:// and http:// endpoints supported.")
    if parsed.scheme == "http" and hostname not in ("localhost", "127.0.0.1", "::1"):
        raise CloudQueryError("HTTP only allowed for localhost. Use HTTPS.")
    if parsed.path and parsed.path != "/":
        raise CloudQueryError(
            f"Endpoint must be an origin (no path). Got: {endpoint}"
        )
    if parsed.query or parsed.fragment:
        raise CloudQueryError(
            f"Endpoint must be an origin (no query/fragment). Got: {endpoint}"
        )


# P0-MIGRATE: move to lobster.cloud.output or shared terminal utils
def strip_ansi(text: str) -> str:
    """Remove ANSI/OSC/C0/C1 escape sequences from text."""
    return _ANSI_ESCAPE_RE.sub("", text)


def _validate_session_id(sid: str) -> str:
    """Validate a session ID is a proper UUID. Raises CloudQueryError."""
    if not isinstance(sid, str) or not _UUID_RE.match(sid):
        raise CloudQueryError(f"Backend returned invalid session ID: {sid!r:.80}")
    return sid


# P0-MIGRATE: replace with CloudClient._auth_header() + API key detection (P0 amendment A3)
def resolve_auth(token_override: Optional[str] = None) -> dict:
    """Resolve auth token and build headers. No os.environ mutation.

    Returns:
        headers_dict
    """
    token = token_override.strip() if token_override else None
    if token_override and not token:
        raise CloudQueryError("Provided token is empty after stripping whitespace.")

    token = token or get_api_key()
    if not token:
        raise CloudQueryError(
            "Not authenticated. Run 'lobster cloud login' first."
        )

    if token.startswith("omk_"):
        return {"X-API-Key": token}
    return {"Authorization": f"Bearer {token}"}


def resolve_rest_base(endpoint_override: Optional[str] = None) -> str:
    """Resolve REST base URL from override or stored credentials."""
    if endpoint_override:
        return endpoint_override.rstrip("/")
    return get_endpoint()


def derive_stream_base(
    rest_base: str, stream_override: Optional[str] = None
) -> str:
    """Derive stream base from explicit override or REST base.

    If REST was overridden but stream was not, use REST base for streaming
    to avoid cross-environment credential leakage.
    """
    if stream_override:
        return stream_override.rstrip("/")
    if rest_base == REST_API_BASE:
        return STREAM_API_BASE
    # Custom REST → use same host for streaming (avoids cross-env token leak)
    return rest_base


def create_cloud_session(
    rest_base: str, headers: dict, name: str = "Cloud Query",
    project_id: Optional[str] = None, client_source: str = "cli",
) -> str:
    """POST /api/v1/sessions → session_id (UUID)."""
    url = f"{rest_base}/api/v1/sessions"
    body: dict = {"name": name, "client_source": client_source}
    if project_id:
        body["project_id"] = project_id
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                url,
                json=body,
                headers={**headers, "Content-Type": "application/json"},
            )
    except httpx.HTTPError as e:
        raise CloudQueryError(f"Cannot reach cloud: {e}") from e

    if resp.status_code == 401:
        raise CloudQueryError("Authentication failed (401). Run 'lobster cloud login'.")
    if resp.status_code == 402:
        raise CloudQueryError(
            "Monthly budget exhausted. Check usage: lobster cloud status"
        )
    if resp.status_code == 429:
        raise CloudQueryError("Rate limited (5/min for session creation). Wait and retry.")
    if not resp.is_success:
        raise CloudQueryError(f"Failed to create session: {resp.status_code}")

    try:
        data = resp.json()
    except (json.JSONDecodeError, ValueError) as e:
        raise CloudQueryError(f"Invalid JSON from session create: {e}") from e

    sid = None
    if isinstance(data, dict):
        session_obj = data.get("session")
        if isinstance(session_obj, dict):
            sid = session_obj.get("session_id")
        if not sid:
            sid = data.get("session_id") or data.get("id")

    if not sid:
        raise CloudQueryError("Backend returned no session_id.")
    return _validate_session_id(sid)


def resolve_cloud_session(
    rest_base: str, headers: dict, session_id: Optional[str],
    project_id: Optional[str] = None, client_source: str = "cli",
) -> str:
    """Resolve session_id: create new, use provided UUID, or resolve 'latest'."""
    # P0-MIGRATE: replace with lobster.cloud.sessions.resolve_session_id()
    if not session_id:
        return create_cloud_session(rest_base, headers, project_id=project_id, client_source=client_source)

    if session_id.lower() == "latest":
        url = f"{rest_base}/api/v1/sessions"
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(
                    url, headers={**headers, "Accept": "application/json"}
                )
        except httpx.HTTPError as e:
            raise CloudQueryError(f"Cannot reach cloud: {e}") from e

        if not resp.is_success:
            raise CloudQueryError(f"Failed to list sessions: {resp.status_code}")

        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as e:
            raise CloudQueryError(f"Invalid JSON from session list: {e}") from e

        sessions = data if isinstance(data, list) else data.get("sessions", []) if isinstance(data, dict) else []
        valid = [
            s for s in sessions
            if isinstance(s, dict) and isinstance(s.get("session_id"), str)
            and _UUID_RE.match(s["session_id"])
        ]
        if not valid:
            raise CloudQueryError("No existing sessions found. Start a new one first.")

        latest = max(
            valid,
            key=lambda s: s.get("last_activity") or s.get("created_at", ""),
        )
        return _validate_session_id(latest["session_id"])

    if not _UUID_RE.match(session_id):
        raise CloudQueryError(
            f'Invalid session ID: "{session_id}". Expected a UUID or "latest".'
        )
    return session_id


class CloudStreamResult:
    """Accumulated result from DataStream."""

    def __init__(self):
        self._text_chunks: list[str] = []
        self._reasoning_chunks: list[str] = []
        self._text_size: int = 0
        self.active_agent: Optional[str] = None
        self.agent_status: Optional[str] = None
        self.token_usage: Optional[dict] = None
        self.session_title: Optional[str] = None
        self.error_detail: Optional[str] = None
        self.finish_reason: Optional[str] = None
        self.tool_calls: list = []
        self._saw_valid_line: bool = False

    def append_text(self, chunk: str) -> None:
        if self._text_size + len(chunk) > MAX_RESPONSE_SIZE:
            return
        self._text_chunks.append(chunk)
        self._text_size += len(chunk)

    def append_reasoning(self, chunk: str) -> None:
        self._reasoning_chunks.append(chunk)

    @property
    def text(self) -> str:
        return "".join(self._text_chunks)

    @property
    def reasoning(self) -> str:
        return "".join(self._reasoning_chunks)

    @property
    def success(self) -> bool:
        return self.agent_status != "error" and self.error_detail is None

    @property
    def display_text(self) -> str:
        t = self.text.strip()
        return t if t else ""

    def set_error(self, detail: str) -> None:
        """Set error_detail. Once set, cannot be cleared (B5 fix)."""
        if self.error_detail is None:
            self.error_detail = detail


def stream_cloud_query(
    stream_base: str,
    headers: dict,
    session_id: str,
    question: str,
    on_text_delta: Optional[Callable[[str], None]] = None,
) -> CloudStreamResult:
    """POST /api/v1/sessions/{id}/chat/stream via DataStream protocol.

    Uses stream.omics-os.com to bypass CloudFront OriginReadTimeout.
    DataStream is newline-delimited prefix:payload (NOT standard SSE).
    """
    url = f"{stream_base}/api/v1/sessions/{session_id}/chat/stream"
    body = {"messages": [{"role": "user", "content": question}]}

    result = CloudStreamResult()

    try:
        with httpx.Client(timeout=httpx.Timeout(600.0, connect=15.0)) as client:
            with client.stream(
                "POST", url, json=body,
                headers={**headers, "Content-Type": "application/json"},
            ) as resp:
                if resp.status_code == 401:
                    result.set_error("Authentication failed (401). Run 'lobster cloud login'.")
                    return result
                if resp.status_code == 402:
                    result.set_error("Monthly budget exhausted. Check: lobster cloud status")
                    return result
                if resp.status_code == 429:
                    result.set_error("Rate limited (10/min). Wait 10-15 seconds and retry.")
                    return result
                if resp.status_code == 404:
                    result.set_error("Session not found. It may have been deleted or expired.")
                    return result
                if not resp.is_success:
                    result.set_error(f"Stream failed: {resp.status_code}")
                    return result

                for line in resp.iter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    _parse_datastream_line(line, result, on_text_delta)

    except httpx.ConnectError as e:
        result.set_error(f"Cannot reach cloud: {e}")
    except httpx.ReadTimeout:
        result.set_error("Stream read timeout (600s). Backend may still be processing.")
    except httpx.HTTPError as e:
        result.set_error(f"Stream error: {e}")

    # B4: detect empty/truncated streams that would falsely report success
    if not result._saw_valid_line and result.error_detail is None:
        result.set_error("Empty response from cloud (no valid DataStream parts received).")

    return result


# P0-MIGRATE: replace with CloudClient.post() fire-and-forget wrapper
def cancel_cloud_run(rest_base: str, headers: dict, session_id: str) -> None:
    """Fire-and-forget POST /sessions/{id}/chat/cancel on Ctrl+C."""
    def _do_cancel():
        url = f"{rest_base}/api/v1/sessions/{session_id}/chat/cancel"
        try:
            with httpx.Client(timeout=3.0) as client:
                client.post(url, headers={**headers, "Content-Type": "application/json"})
        except Exception:
            logger.debug("Cancel request failed (fire-and-forget)", exc_info=True)

    t = threading.Thread(target=_do_cancel, daemon=True)
    t.start()


def fetch_workspace_files(rest_base: str, headers: dict, session_id: str) -> list:
    # P0-MIGRATE: replace with CloudClient.request()
    """Best-effort GET /sessions/{id}/workspace/files/metadata. Returns [] on any error."""
    url = f"{rest_base}/api/v1/sessions/{session_id}/workspace/files/metadata"
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(url, headers={**headers, "Accept": "application/json"})
        if resp.is_success:
            data = resp.json()
            files = data.get("files", data) if isinstance(data, dict) else data if isinstance(data, list) else []
            return [f for f in files if isinstance(f, dict)]
    except Exception:
        logger.debug("Failed to fetch workspace files", exc_info=True)
    return []


def _parse_datastream_line(
    line: str, result: CloudStreamResult, on_text_delta: Optional[Callable] = None
) -> None:
    """Parse a single DataStream line (newline-delimited prefix:payload).

    Prefixes: 0: text, g: reasoning, b: tool start, c: tool args (skipped),
    a: tool result, 2: finish/data, 3: error, d: final usage,
    aui-state: state patches. Unknown prefixes silently ignored.
    """
    if line.startswith("aui-state:"):
        patch_str = line[len("aui-state:"):]
        try:
            patches = json.loads(patch_str)
        except (json.JSONDecodeError, TypeError):
            return

        if not isinstance(patches, list):
            return

        result._saw_valid_line = True
        for patch in patches:
            if not isinstance(patch, dict):
                continue
            path = patch.get("path")
            value = patch.get("value")
            if not isinstance(path, list) or not path:
                continue

            key = path[0]
            if key == "active_agent" and (value is None or isinstance(value, str)):
                result.active_agent = value
            elif key == "agent_status" and (value is None or isinstance(value, str)):
                result.agent_status = value
            elif key == "token_usage" and isinstance(value, dict):
                result.token_usage = value
            elif key == "session_title" and isinstance(value, str):
                result.session_title = value
            elif key == "error_detail" and isinstance(value, str):
                result.set_error(value)
            # B5: error_detail=null patches are ignored (don't clear errors)
        return

    colon_idx = line.find(":")
    if colon_idx == -1:
        return

    prefix = line[:colon_idx]
    payload = line[colon_idx + 1:]

    if prefix == "0":
        try:
            text = json.loads(payload)
            if isinstance(text, str):
                result._saw_valid_line = True
                result.append_text(text)
                if on_text_delta:
                    on_text_delta(strip_ansi(text))
        except (json.JSONDecodeError, TypeError):
            pass

    elif prefix == "g":
        try:
            text = json.loads(payload)
            if isinstance(text, str):
                result._saw_valid_line = True
                result.append_reasoning(text)
        except (json.JSONDecodeError, TypeError):
            pass

    elif prefix == "3":
        # P1: DataStream error part — treat as terminal failure
        result._saw_valid_line = True
        try:
            data = json.loads(payload)
            if isinstance(data, dict):
                msg = data.get("message") or data.get("error") or str(data)
            elif isinstance(data, str):
                msg = data
            else:
                msg = str(data)
        except (json.JSONDecodeError, TypeError):
            msg = payload[:200] if payload else "Unknown stream error"
        result.set_error(f"Stream error: {msg[:200]}")

    elif prefix == "b":
        try:
            data = json.loads(payload)
            if isinstance(data, dict):
                result._saw_valid_line = True
                result.tool_calls.append({
                    "id": data.get("toolCallId"),
                    "name": data.get("toolName"),
                    "status": "started",
                })
        except (json.JSONDecodeError, TypeError):
            pass

    elif prefix == "c":
        pass

    elif prefix == "a":
        try:
            data = json.loads(payload)
            if isinstance(data, dict):
                result._saw_valid_line = True
                tc_id = data.get("toolCallId")
                for tc in result.tool_calls:
                    if tc.get("id") == tc_id:
                        tc["status"] = "complete"
                        tc["result_preview"] = str(data.get("result", ""))[:200]
                        break
        except (json.JSONDecodeError, TypeError):
            pass

    elif prefix == "2":
        try:
            data = json.loads(payload)
            result._saw_valid_line = True
            if isinstance(data, list):
                return
            if isinstance(data, dict):
                if "total_cost_usd" in data or "input_tokens" in data:
                    result.token_usage = data
        except (json.JSONDecodeError, TypeError):
            pass

    elif prefix == "d":
        try:
            data = json.loads(payload)
            if isinstance(data, dict):
                result._saw_valid_line = True
                result.finish_reason = data.get("finishReason")
                usage = data.get("usage")
                if isinstance(usage, dict) and not result.token_usage:
                    result.token_usage = usage
        except (json.JSONDecodeError, TypeError):
            pass
