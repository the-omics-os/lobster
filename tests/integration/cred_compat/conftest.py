"""Fixtures for CRED_COMPAT_V2 E2E tests.

Provides:
  * ``fake_gateway``    — stdlib recording HTTP server standing in for the cloud.
  * ``attacker_server`` — a SECOND recording server; the exfil gate asserts it
                          receives ZERO requests under poisoned-endpoint variants.
  * ``cred_factory``    — builds the 5 credential shapes (V1 oauth/apikey,
                          V2 single/multi, malformed) with PLACEHOLDER tokens.
  * ``isolated_home``   — a tmp HOME with a ``write(shape)`` helper; running the
                          real ``lobster`` CLI under it reads/writes the tmp
                          credentials file, not the developer's real one.
  * ``run_cli``         — subprocess runner for ``python -m lobster.cli`` under
                          the isolated env.

NO real tokens. NO live network — the servers are localhost stdlib.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Recording HTTP server
# ---------------------------------------------------------------------------


class _RecordingServer:
    """A localhost HTTP server that records every request and serves canned JSON.

    ``requests`` accumulates dicts: {method, path, headers, body, host}. The
    canned response is chosen by path suffix via ``routes`` (path-suffix → dict);
    unmatched paths return 200 ``{}``.
    """

    def __init__(self, routes: dict | None = None):
        self.requests: list[dict] = []
        self.routes = routes or {}
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        assert self._server is not None
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def port(self) -> int:
        assert self._server is not None
        return self._server.server_address[1]

    def _match(self, path: str) -> dict:
        for suffix, resp in self.routes.items():
            if path.endswith(suffix) or suffix in path:
                return resp
        return {}

    def start(self) -> None:
        recorder = self

        class _Handler(BaseHTTPRequestHandler):
            def _record_and_respond(self, method: str) -> None:
                length = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(length) if length else b""
                body = None
                if raw:
                    try:
                        body = json.loads(raw)
                    except (ValueError, json.JSONDecodeError):
                        body = raw.decode("utf-8", "replace")
                recorder.requests.append(
                    {
                        "method": method,
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": body,
                        "host": self.headers.get("Host", ""),
                    }
                )
                payload = json.dumps(recorder._match(self.path)).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):  # noqa: N802
                self._record_and_respond("GET")

            def do_POST(self):  # noqa: N802
                self._record_and_respond("POST")

            def log_message(self, *args):  # silence
                pass

        self._server = HTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def token_bearing_requests(self) -> list[dict]:
        """Requests that carried a credential (Authorization or X-API-Key header)."""
        out = []
        for r in self.requests:
            h = r["headers"]
            if "authorization" in h or "x-api-key" in h:
                out.append(r)
        return out


@pytest.fixture
def fake_gateway():
    """Recording server standing in for the Omics-OS gateway."""
    server = _RecordingServer(
        routes={
            "/api/v1/gateway/usage": {
                "tier": "pro",
                "user_id": "u_TEST",
                "email": "test@x.com",
                "budget": {
                    "remaining_usd": 42,
                    "monthly_budget_usd": 100,
                    "utilization_pct": 0.0,
                },
                "usage": {},
                "limits": {"max_tokens_per_request": 4096},
            },
            "/api/v1/gateway/oauth/cli/token": {
                "cli_token": "omc_ROTATEDxxxxxxxxxxxxxxxxxxxxxxxx",
                "refresh_token": "omr_ROTATEDxxxxxxxxxxxxxxxxxxxxxx",
                "refresh_expiry": "2099-06-01T00:00:00+00:00",
            },
            "/api/v1/gateway/token/refresh": {"access_token": "COGNITO_ROTATED"},
            "/api/v1/sessions": {"session_id": "00000000-0000-4000-8000-000000000000"},
        }
    )
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def attacker_server():
    """Second recording server — must receive ZERO token-bearing requests."""
    server = _RecordingServer()
    server.start()
    try:
        yield server
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Credential shape factory (PLACEHOLDER tokens only)
# ---------------------------------------------------------------------------


class _CredFactory:
    """Builds the five credential shapes used across the matrix."""

    @staticmethod
    def v1_oauth(endpoint: str = "https://app.omics-os.com") -> dict:
        return {
            "auth_mode": "oauth",
            "access_token": "COGNITO_ACCESS_TEST",
            "refresh_token": "COGNITO_REFRESH_TEST",
            "client_id": "cid",
            "endpoint": endpoint,
            "token_expiry": "2099-01-01T00:00:00+00:00",
        }

    @staticmethod
    def v1_apikey(endpoint: str = "https://app.omics-os.com") -> dict:
        return {"auth_mode": "api_key", "api_key": "omk_TESTKEY", "endpoint": endpoint}

    @staticmethod
    def v2_single(endpoint: str = "https://app.omics-os.com") -> dict:
        return {
            "version": 2,
            "active_profile": "default",
            "profiles": {
                "default": {
                    "credential_type": "omics_cli",
                    "cli_token": "omc_SINGLExxxxxxxxxxxxxxxxxxxxxxxx",
                    "refresh_token": "omr_SINGLExxxxxxxxxxxxxxxxxxxxxx",
                    "credential_id": "clicred_SINGLE",
                    "endpoint": endpoint,
                    "auth_mode": "oauth",
                    "token_expiry": "2099-01-01T00:00:00+00:00",
                    "email": "test@x.com",
                    "label": "default",
                }
            },
        }

    @staticmethod
    def v2_multi(
        default_endpoint: str = "https://app.omics-os.com",
        tenant_endpoint: str = "https://databiomix.omics-os.com",
    ) -> dict:
        def _prof(token, endpoint, label):
            return {
                "credential_type": "omics_cli",
                "cli_token": token,
                "refresh_token": "omr_" + label + "xxxxxxxxxxxxxxxxxx",
                "credential_id": "clicred_SHARED",
                "endpoint": endpoint,
                "auth_mode": "oauth",
                "token_expiry": "2099-01-01T00:00:00+00:00",
                "email": "test@x.com",
                "label": label,
            }

        return {
            "version": 2,
            "active_profile": "databiomix",
            "profiles": {
                "default": _prof(
                    "omc_DEFAULTxxxxxxxxxxxxxxxxxxxxx", default_endpoint, "default"
                ),
                "databiomix": _prof(
                    "omc_TENANTxxxxxxxxxxxxxxxxxxxxxx", tenant_endpoint, "databiomix"
                ),
            },
        }

    @staticmethod
    def malformed_v2() -> dict:
        # version 2 but no active_profile → resolver must fall back to default.
        return {
            "version": 2,
            "profiles": {
                "default": {
                    "credential_type": "omics_cli",
                    "cli_token": "omc_FALLBACKxxxxxxxxxxxxxxxxxxxx",
                    "credential_id": "clicred_FB",
                    "endpoint": "https://app.omics-os.com",
                    "auth_mode": "oauth",
                    "token_expiry": "2099-01-01T00:00:00+00:00",
                }
            },
        }


@pytest.fixture
def cred_factory():
    return _CredFactory


# ---------------------------------------------------------------------------
# Isolated HOME + CLI runner
# ---------------------------------------------------------------------------


class _IsolatedHome:
    def __init__(self, tmp_path: Path):
        self.home = tmp_path / "home"
        self.xdg_config = tmp_path / "xdg-config"
        self.xdg_cache = tmp_path / "xdg-cache"
        self.workspace = tmp_path / "workspace"
        for p in (self.home, self.xdg_config, self.xdg_cache, self.workspace):
            p.mkdir(parents=True, exist_ok=True)
        self.cred_file = self.home / ".config" / "omics-os" / "credentials.json"

    def write(self, shape: dict) -> Path:
        self.cred_file.parent.mkdir(parents=True, exist_ok=True)
        self.cred_file.write_text(json.dumps(shape, indent=2) + "\n", encoding="utf-8")
        os.chmod(self.cred_file, 0o600)
        return self.cred_file

    def read(self) -> dict | None:
        if not self.cred_file.exists():
            return None
        return json.loads(self.cred_file.read_text(encoding="utf-8"))

    def env(
        self, *, endpoint_override: str | None = None, extra: dict | None = None
    ) -> dict:
        env = os.environ.copy()
        env.update(
            {
                "HOME": str(self.home),
                "XDG_CONFIG_HOME": str(self.xdg_config),
                "XDG_CACHE_HOME": str(self.xdg_cache),
                "LOBSTER_WORKSPACE": str(self.workspace),
                "NO_COLOR": "1",
                "PYTHONUNBUFFERED": "1",
                "TERM": "xterm-256color",
            }
        )
        for k in (
            "LOBSTER_CLI_BINARY",
            "LOBSTER_ENDPOINT",
            "LOBSTER_CLOUD_KEY",
            "OMICS_OS_API_KEY",
        ):
            env.pop(k, None)
        if endpoint_override is not None:
            env["OMICS_OS_ENDPOINT"] = endpoint_override
        else:
            env.pop("OMICS_OS_ENDPOINT", None)
        # Hide any npm lobster on PATH so `python -m lobster.cli` is authoritative.
        env["PATH"] = os.pathsep.join(
            [
                str(Path(sys.executable).resolve().parent),
                "/usr/bin",
                "/bin",
                "/usr/sbin",
                "/sbin",
            ]
        )
        if extra:
            env.update(extra)
        return env

    def run_cli(
        self,
        args: list[str],
        *,
        endpoint_override: str | None = None,
        extra_env: dict | None = None,
        timeout: int = 30,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "lobster.cli", *args],
            cwd=str(REPO_ROOT),
            env=self.env(endpoint_override=endpoint_override, extra=extra_env),
            input="",
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )


@pytest.fixture
def isolated_home(tmp_path):
    return _IsolatedHome(tmp_path)
