"""
Anthropic OAuth 2.0 Authorization Code + PKCE flow.

Enables login with a Claude Pro/Max account instead of an API key.
The access token returned by Anthropic is usable as a Bearer token
for the Anthropic Messages API (same as sk-ant-* keys).

Flow:
    1. Generate PKCE verifier + challenge
    2. Open browser to claude.ai/oauth/authorize
    3. Listen on localhost for GET callback with authorization code
    4. Exchange code for access_token + refresh_token at platform.claude.com
    5. Store tokens via oauth_store
    6. Auto-refresh expired tokens transparently

Reference: pi-mono/packages/ai/src/utils/oauth/anthropic.ts
"""

import json
import logging
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

import httpx

from lobster.config.auth import oauth_store
from lobster.config.auth.oauth_store import OAuthCredentials
from lobster.config.auth.pkce import generate_pkce

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Anthropic OAuth constants
# ---------------------------------------------------------------------------
# Public OAuth client — no client_secret required (PKCE-protected).
# TODO: Register Lobster's own client ID with Anthropic. Currently using the
#       well-known public CLI client ID (same as Claude Code / pi-mono).
CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
TOKEN_URL = "https://platform.claude.com/v1/oauth/token"

CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 53692
CALLBACK_PATH = "/callback"
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}{CALLBACK_PATH}"

SCOPES = "org:create_api_key user:profile user:inference"

PROVIDER_ID = "anthropic"

# Token exchange / refresh timeout
HTTP_TIMEOUT = 30.0

# Safety buffer: refresh 5 minutes before expiry
EXPIRY_BUFFER_SECONDS = 300


# ---------------------------------------------------------------------------
# Token exchange helpers
# ---------------------------------------------------------------------------


def _post_token(body: dict) -> dict:
    """POST to Anthropic's token endpoint and return parsed JSON."""
    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        resp = client.post(
            TOKEN_URL,
            json=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
    if resp.status_code != 200:
        raise OAuthError(f"Token request failed (HTTP {resp.status_code}): {resp.text}")
    try:
        return resp.json()
    except json.JSONDecodeError as e:
        raise OAuthError(f"Token response is not valid JSON: {e}") from e


def exchange_authorization_code(
    code: str,
    state: str,
    verifier: str,
    redirect_uri: str = REDIRECT_URI,
) -> OAuthCredentials:
    """Exchange an authorization code for tokens (standard OAuth 2.0 + PKCE)."""
    data = _post_token(
        {
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "state": state,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        }
    )
    return OAuthCredentials(
        access_token=data["access_token"],
        refresh_token=data["refresh_token"],
        expires_at=time.time() + data.get("expires_in", 3600) - EXPIRY_BUFFER_SECONDS,
        provider=PROVIDER_ID,
    )


def refresh_access_token(
    refresh_token: str,
) -> OAuthCredentials:
    """Refresh an expired access token."""
    data = _post_token(
        {
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": refresh_token,
        }
    )
    return OAuthCredentials(
        access_token=data["access_token"],
        refresh_token=data.get("refresh_token", refresh_token),
        expires_at=time.time() + data.get("expires_in", 3600) - EXPIRY_BUFFER_SECONDS,
        provider=PROVIDER_ID,
    )


# ---------------------------------------------------------------------------
# High-level credential access (used by AnthropicProvider)
# ---------------------------------------------------------------------------


def get_access_token() -> Optional[str]:
    """Get a valid Anthropic OAuth access token, refreshing if needed.

    Returns:
        Access token string, or None if no OAuth credentials exist.

    Raises:
        OAuthError: If token refresh fails (credentials should be cleared).
    """
    creds = oauth_store.load(PROVIDER_ID)
    if creds is None:
        return None

    if not creds.is_expired(buffer_seconds=EXPIRY_BUFFER_SECONDS):
        return creds.access_token

    # Token expired — attempt refresh
    if not creds.refresh_token:
        logger.warning("Anthropic OAuth token expired and no refresh_token available.")
        return None

    try:
        new_creds = refresh_access_token(creds.refresh_token)
        # Preserve metadata from original creds
        new_creds.email = creds.email
        new_creds.account_id = creds.account_id
        oauth_store.save(new_creds)
        logger.debug("Anthropic OAuth token refreshed successfully.")
        return new_creds.access_token
    except OAuthError as e:
        logger.warning("Anthropic OAuth token refresh failed: %s", e)
        # Return stale token — let the API reject it with a clear error
        return creds.access_token


def is_authenticated() -> bool:
    """Check if Anthropic OAuth credentials exist (may be expired but refreshable)."""
    return oauth_store.has_credentials(PROVIDER_ID)


def logout() -> bool:
    """Remove stored Anthropic OAuth credentials."""
    return oauth_store.delete(PROVIDER_ID)


# ---------------------------------------------------------------------------
# Interactive browser login flow
# ---------------------------------------------------------------------------


class OAuthError(Exception):
    """Raised on OAuth flow failures."""


class LoginResult:
    """Result of an interactive login attempt."""

    def __init__(
        self,
        success: bool,
        credentials: Optional[OAuthCredentials] = None,
        error: Optional[str] = None,
    ):
        self.success = success
        self.credentials = credentials
        self.error = error


def build_authorize_url(verifier: str, challenge: str) -> str:
    """Build the Anthropic OAuth authorize URL with PKCE parameters."""
    from urllib.parse import urlencode

    params = {
        "code": "true",
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def login_interactive(
    timeout: int = 120,
    open_browser: bool = True,
    on_url: Optional[callable] = None,
    on_progress: Optional[callable] = None,
    on_manual_input: Optional[callable] = None,
) -> LoginResult:
    """Run the full OAuth login flow with a local callback server.

    Args:
        timeout: Seconds to wait for the browser callback.
        open_browser: Whether to auto-open the browser.
        on_url: Callback(url: str) — called with the authorize URL.
            Frontends (Go TUI, lobster-cli) use this to display/open the URL.
        on_progress: Callback(msg: str) — status updates.
        on_manual_input: Callback() -> str — called to get manual code/URL input
            as fallback when browser callback doesn't arrive.

    Returns:
        LoginResult with success flag and credentials or error.
    """
    verifier, challenge = generate_pkce()
    auth_url = build_authorize_url(verifier, challenge)

    # State for the callback server
    received: dict = {}
    server_ready = threading.Event()

    class _CallbackHandler(BaseHTTPRequestHandler):
        """Handle the OAuth redirect GET request."""

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path != CALLBACK_PATH:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found")
                return

            params = parse_qs(parsed.query)
            error = params.get("error", [None])[0]
            if error:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(f"Authentication error: {error}".encode())
                return

            code = params.get("code", [None])[0]
            state = params.get("state", [None])[0]

            if not code or not state:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing code or state parameter.")
                return

            if state != verifier:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"State mismatch - possible CSRF attack.")
                return

            received["code"] = code
            received["state"] = state

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"<html><body>"
                b"<h2>Authentication successful!</h2>"
                b"<p>You can close this tab and return to the terminal.</p>"
                b"</body></html>"
            )

        def log_message(self, format, *args):
            pass  # Suppress server access logs

    # Start local callback server
    try:
        server = HTTPServer((CALLBACK_HOST, CALLBACK_PORT), _CallbackHandler)
    except OSError as e:
        return LoginResult(
            success=False,
            error=f"Cannot bind to {CALLBACK_HOST}:{CALLBACK_PORT}: {e}",
        )

    server.timeout = 1  # Short timeout for polling loop

    # Notify frontend of the URL
    if on_url:
        on_url(auth_url)

    if open_browser:
        try:
            webbrowser.open(auth_url)
        except Exception:
            pass  # Frontends handle this via on_url

    if on_progress:
        on_progress("Waiting for browser authentication...")

    # Serve callback requests in a thread
    stop_event = threading.Event()

    def _serve():
        server_ready.set()
        while not stop_event.is_set() and not received:
            server.handle_request()

    server_thread = threading.Thread(target=_serve, daemon=True)
    server_thread.start()
    server_ready.wait()

    # Wait for callback or timeout
    deadline = time.monotonic() + timeout
    while not received and time.monotonic() < deadline:
        time.sleep(0.2)

    stop_event.set()
    server.server_close()

    # If no callback received, try manual input fallback
    if not received and on_manual_input:
        if on_progress:
            on_progress("Browser callback not received. Trying manual input...")
        try:
            manual = on_manual_input()
            parsed = _parse_authorization_input(manual, verifier)
            if parsed:
                received.update(parsed)
        except Exception:
            pass

    if not received or "code" not in received:
        return LoginResult(
            success=False,
            error="No authorization code received (timed out or cancelled).",
        )

    # Exchange authorization code for tokens
    if on_progress:
        on_progress("Exchanging authorization code for tokens...")

    try:
        creds = exchange_authorization_code(
            code=received["code"],
            state=received["state"],
            verifier=verifier,
        )
    except OAuthError as e:
        return LoginResult(success=False, error=str(e))

    # Persist credentials
    oauth_store.save(creds)

    return LoginResult(success=True, credentials=creds)


def _parse_authorization_input(raw: str, expected_state: str) -> Optional[dict]:
    """Parse manual authorization input (URL or bare code)."""
    value = raw.strip()
    if not value:
        return None

    # Try parsing as a full URL
    try:
        parsed = urlparse(value)
        params = parse_qs(parsed.query)
        code = params.get("code", [None])[0]
        state = params.get("state", [None])[0]
        if code:
            if state and state != expected_state:
                return None  # State mismatch
            return {"code": code, "state": state or expected_state}
    except Exception:
        pass

    # Try code#state format
    if "#" in value:
        code, _, state = value.partition("#")
        if state and state != expected_state:
            return None
        return {"code": code, "state": state or expected_state}

    # Bare code
    return {"code": value, "state": expected_state}
