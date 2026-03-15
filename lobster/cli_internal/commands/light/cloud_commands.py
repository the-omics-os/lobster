"""
Cloud management subcommands for Lobster CLI.

Provides commands to authenticate with and manage Omics-OS Cloud.

Commands:
    lobster cloud login   - Authenticate with Omics-OS Cloud
    lobster cloud status  - Show current usage and budget
    lobster cloud logout  - Remove stored credentials
"""

import logging
import sys
import time
from typing import Optional
from urllib.parse import urlparse

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)

cloud_app = typer.Typer(
    name="cloud",
    help="Manage Omics-OS Cloud connection and usage",
    no_args_is_help=True,
)

console = Console()


def _is_secure_endpoint(endpoint: str) -> bool:
    """Check that the endpoint is HTTPS or localhost (for dev)."""
    parsed = urlparse(endpoint)
    if parsed.scheme == "https":
        return True
    if parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1", "::1"):
        return True
    return False


def _validate_credentials(
    endpoint: str, token: str, token_type: str = "API key"
) -> Optional[dict]:
    """Validate credentials against the gateway. Returns usage data or None on failure."""
    import httpx

    usage_url = f"{endpoint}/api/v1/gateway/usage"
    console.print(f"[dim]Validating {token_type}...[/dim]")

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                usage_url,
                headers={"Authorization": f"Bearer {token}"},
            )
    except (httpx.ConnectError, httpx.TimeoutException):
        console.print(
            f"[red]Cannot reach Omics-OS Cloud at {endpoint}.[/red]\n"
            "[dim]Check your network connection and try again.[/dim]"
        )
        return None

    if resp.status_code == 401:
        console.print(f"[red]Invalid {token_type}. Please check and try again.[/red]")
        return None

    if resp.status_code != 200:
        console.print(
            f"[red]Unexpected response ({resp.status_code}): {resp.text}[/red]"
        )
        return None

    return resp.json()


def _print_login_success(data: dict) -> None:
    """Print login success panel."""
    tier = data.get("tier", "free")
    budget = data.get("budget", {})
    remaining = budget.get("remaining_usd", "?")

    console.print(
        Panel(
            f"[green]Authenticated successfully![/green]\n\n"
            f"Tier: [bold]{tier}[/bold]\n"
            f"Budget remaining: [bold]${remaining}[/bold]",
            title="Omics-OS Cloud",
            border_style="green",
        )
    )


def _api_key_login(api_key: Optional[str]) -> None:
    """Authenticate with an API key (interactive prompt if not provided)."""
    from lobster.config.credentials import get_endpoint, save_credentials

    if not api_key:
        api_key = typer.prompt(
            "Paste your Omics-OS API key (from app.omics-os.com/settings/api-keys)",
            hide_input=True,
        )

    if not api_key or not api_key.strip():
        console.print("[red]API key cannot be empty.[/red]")
        raise typer.Exit(1)

    api_key = api_key.strip()
    endpoint = get_endpoint()

    if not _is_secure_endpoint(endpoint):
        console.print(
            f"[red]Refusing to send API key over insecure connection: {endpoint}[/red]\n"
            "[dim]The endpoint must use HTTPS (or localhost for development).[/dim]"
        )
        raise typer.Exit(1)

    data = _validate_credentials(endpoint, api_key)
    if data is None:
        raise typer.Exit(1)

    creds = {
        "auth_mode": "api_key",
        "api_key": api_key,
        "endpoint": endpoint,
        "user_id": data.get("user_id", ""),
        "email": data.get("email", ""),
        "tier": data.get("tier", "free"),
    }

    save_credentials(creds)
    _print_login_success(data)


def _browser_login() -> None:
    """Authenticate via browser OAuth flow with localhost callback."""
    import json as _json
    import random
    import secrets
    import threading
    import webbrowser
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from lobster.config.credentials import get_endpoint, save_credentials

    endpoint = get_endpoint()
    if not _is_secure_endpoint(endpoint):
        console.print(
            f"[red]Refusing to authenticate over insecure connection: {endpoint}[/red]\n"
            "[dim]The endpoint must use HTTPS (or localhost for development).[/dim]"
        )
        raise typer.Exit(1)

    port = random.randint(18765, 18800)
    state = secrets.token_urlsafe(32)
    received_tokens: dict = {}
    server_error: list = []

    class CallbackHandler(BaseHTTPRequestHandler):
        def _send_cors_headers(self):
            self.send_header("Access-Control-Allow-Origin", endpoint)
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Private-Network", "true")

        def do_POST(self):
            if self.path != "/callback":
                self.send_response(404)
                self._send_cors_headers()
                self.end_headers()
                return

            # Validate Content-Type
            content_type = self.headers.get("Content-Type", "")
            if "application/json" not in content_type:
                self.send_response(415)
                self._send_cors_headers()
                self.end_headers()
                return

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                tokens = _json.loads(body)

                # Verify state parameter (CSRF protection)
                if tokens.get("state") != state:
                    self.send_response(403)
                    self._send_cors_headers()
                    self.end_headers()
                    return

                received_tokens.update(tokens)
                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<html><body><h2>Authentication successful!</h2>"
                    b"<p>You can close this tab and return to the terminal.</p></body></html>"
                )
            except (ValueError, _json.JSONDecodeError) as e:
                server_error.append(str(e))
                self.send_response(400)
                self._send_cors_headers()
                self.end_headers()

        def do_OPTIONS(self):
            self.send_response(204)
            self._send_cors_headers()
            self.end_headers()

        def log_message(self, format, *args):
            pass  # Suppress server logs

    try:
        server = HTTPServer(("127.0.0.1", port), CallbackHandler)
    except OSError:
        console.print(
            "[yellow]Could not start local server. Falling back to API key login.[/yellow]"
        )
        _api_key_login(None)
        return

    server.timeout = 120

    auth_url = f"{endpoint}/auth/cli?port={port}&state={state}"
    console.print(f"[dim]Opening browser for authentication...[/dim]")
    console.print(f"[dim]If the browser doesn't open, visit:[/dim] {auth_url}")

    try:
        webbrowser.open(auth_url)
    except Exception:
        console.print("[yellow]Could not open browser automatically.[/yellow]")

    # Serve until we receive tokens or timeout.
    # Browser sends OPTIONS preflight + POST — need to handle multiple requests.
    done_event = threading.Event()

    def serve_until_done():
        while not done_event.is_set():
            server.handle_request()

    server_thread = threading.Thread(target=serve_until_done, daemon=True)
    server_thread.start()

    console.print("[dim]Waiting for browser authentication (timeout: 120s)...[/dim]")

    # Poll for tokens with overall timeout
    deadline = time.monotonic() + 120
    while not received_tokens and time.monotonic() < deadline:
        time.sleep(0.2)

    done_event.set()
    server.server_close()

    if not received_tokens or "access_token" not in received_tokens:
        if time.monotonic() >= deadline:
            console.print("[yellow]Browser login timed out.[/yellow]")
        else:
            console.print("[yellow]Did not receive valid tokens from browser.[/yellow]")
        console.print("[dim]Falling back to API key login...[/dim]")
        _api_key_login(None)
        return

    access_token = received_tokens["access_token"]

    # Validate the token
    data = _validate_credentials(endpoint, access_token)
    if data is None:
        console.print(
            "[yellow]Token validation failed. Falling back to API key login.[/yellow]"
        )
        _api_key_login(None)
        return

    # Calculate token expiry (Cognito tokens last 1 hour by default)
    from datetime import datetime, timedelta, timezone

    token_expiry = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()

    creds = {
        "auth_mode": "oauth",
        "access_token": access_token,
        "refresh_token": received_tokens.get("refresh_token", ""),
        "id_token": received_tokens.get("id_token", ""),
        "token_expiry": token_expiry,
        "endpoint": endpoint,
        "user_id": data.get("user_id", ""),
        "email": data.get("email", ""),
        "tier": data.get("tier", "free"),
    }

    save_credentials(creds)
    _print_login_success(data)


def attempt_login_for_init() -> bool:
    """Attempt Omics-OS Cloud browser login for the init wizard.

    Runs the same browser OAuth flow as ``lobster cloud login`` but returns
    a boolean instead of raising ``typer.Exit``.  On success the credentials
    are persisted to ``~/.config/omics-os/credentials.json``.

    Returns:
        True if login succeeded and credentials were saved.
    """
    import json as _json
    import random
    import secrets
    import threading
    import webbrowser
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from lobster.config.credentials import get_endpoint, save_credentials

    endpoint = get_endpoint()
    if not _is_secure_endpoint(endpoint):
        console.print(
            f"[red]Refusing to authenticate over insecure connection: {endpoint}[/red]"
        )
        return False

    port = random.randint(18765, 18800)
    state = secrets.token_urlsafe(32)
    received_tokens: dict = {}

    class _CallbackHandler(BaseHTTPRequestHandler):
        def _send_cors_headers(self):
            self.send_header("Access-Control-Allow-Origin", endpoint)
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Private-Network", "true")

        def do_POST(self):
            if self.path != "/callback":
                self.send_response(404)
                self._send_cors_headers()
                self.end_headers()
                return
            content_type = self.headers.get("Content-Type", "")
            if "application/json" not in content_type:
                self.send_response(415)
                self._send_cors_headers()
                self.end_headers()
                return
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                tokens = _json.loads(body)
                if tokens.get("state") != state:
                    self.send_response(403)
                    self._send_cors_headers()
                    self.end_headers()
                    return
                received_tokens.update(tokens)
                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<html><body><h2>Authentication successful!</h2>"
                    b"<p>You can close this tab and return to the terminal.</p></body></html>"
                )
            except (ValueError, _json.JSONDecodeError):
                self.send_response(400)
                self._send_cors_headers()
                self.end_headers()

        def do_OPTIONS(self):
            self.send_response(204)
            self._send_cors_headers()
            self.end_headers()

        def log_message(self, format, *args):
            pass

    try:
        server = HTTPServer(("127.0.0.1", port), _CallbackHandler)
    except OSError:
        return False

    server.timeout = 120

    auth_url = f"{endpoint}/auth/cli?port={port}&state={state}"
    console.print("[dim]Opening browser for Omics-OS Cloud login...[/dim]")
    console.print(f"[dim]If browser doesn't open: {auth_url}[/dim]")

    try:
        webbrowser.open(auth_url)
    except Exception:
        console.print("[yellow]Could not open browser automatically.[/yellow]")

    done_event = threading.Event()

    def _serve_until_done():
        while not done_event.is_set():
            server.handle_request()

    server_thread = threading.Thread(target=_serve_until_done, daemon=True)
    server_thread.start()

    console.print("[dim]Waiting for browser authentication (timeout: 120s)...[/dim]")

    deadline = time.monotonic() + 120
    while not received_tokens and time.monotonic() < deadline:
        time.sleep(0.2)

    done_event.set()
    server.server_close()

    if not received_tokens or "access_token" not in received_tokens:
        if time.monotonic() >= deadline:
            console.print("[yellow]Browser login timed out.[/yellow]")
        return False

    access_token = received_tokens["access_token"]
    data = _validate_credentials(endpoint, access_token, token_type="token")
    if data is None:
        return False

    from datetime import datetime, timedelta, timezone

    creds = {
        "auth_mode": "oauth",
        "access_token": access_token,
        "refresh_token": received_tokens.get("refresh_token", ""),
        "id_token": received_tokens.get("id_token", ""),
        "token_expiry": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        "endpoint": endpoint,
        "user_id": data.get("user_id", ""),
        "email": data.get("email", ""),
        "tier": data.get("tier", "free"),
    }
    save_credentials(creds)

    tier = data.get("tier", "free")
    email = data.get("email", "")
    console.print(f"[green]Authenticated as {email} (tier: {tier})[/green]")
    return True


@cloud_app.command()
def login(
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="API key (omk_...). If provided, skips browser login.",
    ),
) -> None:
    """Authenticate with Omics-OS Cloud.

    Without --api-key, opens a browser for OAuth login.
    Falls back to interactive API key paste if the browser flow fails.

    All LLM calls route through the Omics-OS gateway for managed Bedrock access
    with billing, usage tracking, and model access control.
    """
    if api_key:
        _api_key_login(api_key)
    else:
        _browser_login()


@cloud_app.command()
def status() -> None:
    """Show current Omics-OS Cloud connection and usage."""
    import httpx

    from lobster.config.credentials import get_api_key, get_endpoint

    api_key = get_api_key()
    if not api_key:
        console.print(
            "[yellow]Not connected to Omics-OS Cloud.[/yellow]\n"
            "Run: [bold]lobster cloud login[/bold]"
        )
        raise typer.Exit(0)

    endpoint = get_endpoint()
    usage_url = f"{endpoint}/api/v1/gateway/usage"

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                usage_url,
                headers={"Authorization": f"Bearer {api_key}"},
            )
    except (httpx.ConnectError, httpx.TimeoutException):
        console.print(f"[red]Cannot reach Omics-OS Cloud at {endpoint}.[/red]")
        raise typer.Exit(1)

    if resp.status_code == 401:
        console.print(
            "[red]Credentials expired or invalid.[/red]\n"
            "Run: [bold]lobster cloud login[/bold]"
        )
        raise typer.Exit(1)

    if resp.status_code != 200:
        console.print(f"[red]Error ({resp.status_code}): {resp.text}[/red]")
        raise typer.Exit(1)

    data = resp.json()
    usage = data.get("usage", {})
    budget = data.get("budget", {})
    limits = data.get("limits", {})

    table = Table(title="Omics-OS Cloud Usage", show_header=False, border_style="cyan")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Tier", data.get("tier", "?"))
    table.add_row("Period", data.get("period", "?"))
    table.add_row("Input tokens", f"{usage.get('input_tokens', 0):,}")
    table.add_row("Output tokens", f"{usage.get('output_tokens', 0):,}")
    table.add_row("Total tokens", f"{usage.get('total_tokens', 0):,}")
    table.add_row("Cost", f"${usage.get('cost_usd', 0):.2f}")
    table.add_row("Requests", f"{usage.get('request_count', 0):,}")
    table.add_row("", "")
    table.add_row("Monthly budget", f"${budget.get('monthly_budget_usd', 0):.2f}")
    table.add_row("Remaining", f"${budget.get('remaining_usd', 0):.2f}")
    table.add_row("Utilization", f"{budget.get('utilization_pct', 0):.1f}%")
    table.add_row("", "")
    table.add_row("Max tokens/req", f"{limits.get('max_tokens_per_request', '?'):,}")

    console.print(table)


@cloud_app.command()
def logout() -> None:
    """Log out of Omics-OS Cloud."""
    from lobster.config.credentials import clear_credentials

    clear_credentials()
    console.print("Logged out of Omics-OS Cloud.")
