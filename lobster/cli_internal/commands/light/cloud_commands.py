"""
Cloud management subcommands for Lobster CLI.

Provides commands to authenticate with and manage Omics-OS Cloud.

Commands:
    lobster cloud login   - Authenticate with Omics-OS Cloud
    lobster cloud status  - Show current usage and budget
    lobster cloud logout  - Remove stored credentials
    lobster cloud chat    - Open interactive cloud chat (launches npm TUI)
"""

import logging
import sys
from typing import Optional
from urllib.parse import urlparse

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lobster.i18n import h, t

logger = logging.getLogger(__name__)

cloud_app = typer.Typer(
    name="cloud",
    help=t("light.cloud.help"),
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

    from lobster.config.endpoint_policy import EndpointPolicyError, validate_endpoint

    # Fail closed before attaching the bearer token — a poisoned endpoint would
    # otherwise exfiltrate the token during "validation". (Codex P0)
    try:
        validate_endpoint(endpoint)
    except EndpointPolicyError as e:
        console.print(f"[red]Refusing to send token to disallowed endpoint: {e}[/red]")
        return None

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
    from lobster.config.credentials import _save_active_profile, get_endpoint

    if not api_key:
        api_key = typer.prompt(
            "Paste your Omics-OS API key (from app.omics-os.com/account)",
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

    # V2-safe write: merge into the active profile (preserves siblings). Clear
    # stale omics_cli/oauth fields so the profile stays token-model-coherent.
    _save_active_profile(
        creds,
        remove=(
            "cli_token",
            "credential_id",
            "credential_type",
            "access_token",
            "refresh_token",
            "id_token",
            "token_expiry",
        ),
    )
    _print_login_success(data)


def _browser_login() -> None:
    """Redirect legacy no-key callers without browser, listener, or hidden prompt."""
    from lobster.config.cloud_setup import OMICS_OS_AUTH_GUIDANCE

    console.print(OMICS_OS_AUTH_GUIDANCE, markup=False)
    raise typer.Exit(1)


def attempt_login_for_init() -> bool:
    """Compatibility helper: describe supported paths, without authentication I/O."""
    from lobster.config.cloud_setup import OMICS_OS_AUTH_GUIDANCE

    console.print(OMICS_OS_AUTH_GUIDANCE, markup=False)
    return False


@cloud_app.command()
def account() -> None:
    """Show your Omics-OS Cloud account summary."""
    from lobster.config.credentials import (
        get_api_key,
        get_endpoint,
        load_active_profile,
    )

    creds = load_active_profile()
    api_key = get_api_key()
    if not api_key or not creds:
        console.print(
            "[yellow]Not connected to Omics-OS Cloud.[/yellow]\n"
            "Run: [bold]lobster cloud login --api-key <key>[/bold]"
        )
        raise typer.Exit(0)

    endpoint = get_endpoint()
    email = creds.get("email", "unknown")
    tier = creds.get("tier", "unknown")
    user_id = creds.get("user_id", "unknown")
    auth_mode = creds.get("auth_mode", "unknown")

    table = Table(
        title="Omics-OS Cloud Account", show_header=False, border_style="cyan"
    )
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Email", email)
    table.add_row("Tier", tier)
    table.add_row("User ID", user_id)
    table.add_row("Auth mode", auth_mode)
    table.add_row("Endpoint", endpoint)

    # Fetch live usage data
    import httpx

    from lobster.config.endpoint_policy import EndpointPolicyError, validate_endpoint

    try:
        # Fail closed before attaching the bearer token (poisoned-endpoint exfil).
        validate_endpoint(endpoint)
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                f"{endpoint}/api/v1/gateway/usage",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        if resp.status_code == 200:
            data = resp.json()
            budget = data.get("budget", {})
            table.add_row("", "")
            table.add_row("Budget remaining", f"${budget.get('remaining_usd', '?')}")
            table.add_row("Period", data.get("period", "?"))
    except EndpointPolicyError:
        table.add_row("", "")
        table.add_row("Usage", "[red]endpoint not in allowlist[/red]")
    except (httpx.ConnectError, httpx.TimeoutException):
        table.add_row("", "")
        table.add_row("Status", "[dim]Could not reach cloud[/dim]")

    console.print(table)
    console.print(
        "\n[dim]Manage API keys:   https://app.omics-os.com/account[/dim]\n"
        "[dim]Account settings:  https://app.omics-os.com/account[/dim]"
    )


keys_app = typer.Typer(
    name="keys",
    help=t("light.cloud.keys"),
    no_args_is_help=False,
    invoke_without_command=True,
)
cloud_app.add_typer(keys_app, name="keys")


@keys_app.callback(invoke_without_command=True)
def keys_callback(ctx: typer.Context) -> None:
    """Manage API keys via the Omics-OS Cloud web interface."""
    if ctx.invoked_subcommand is None:
        console.print(
            "API key management is available at:\n\n"
            "  [bold cyan]https://app.omics-os.com/account[/bold cyan]\n\n"
            "[dim]For security, API keys are managed through the web interface only.\n"
            "Terminal sessions can expose key material in scrollback and shell history.[/dim]"
        )


@cloud_app.command()
def login(
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help=h("Omics-OS API key from app.omics-os.com/account"),
    ),
) -> None:
    """Authenticate the Python local gateway provider with --api-key <key>.

    Create a key at app.omics-os.com/account. Without --api-key,
    this Python command prints setup guidance and exits nonzero; it does not
    launch a browser or prompt for a key.

    Alternative managed Cloud client (npm): npm i -g @omicsos/lobster &&
    lobster cloud login --device. Use lobster-cli cloud login --device to select
    the npm client when both are installed. Configure Python separately using
    OMICS_OS_API_KEY; do not assume npm credential sharing.
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
    from lobster.config.endpoint_policy import EndpointPolicyError, validate_endpoint

    api_key = get_api_key()
    if not api_key:
        console.print(
            "[yellow]No Omics-OS Cloud API key found.[/yellow]\n"
            "Credentials are read from the active Python profile or the "
            "[bold]OMICS_OS_API_KEY[/bold] environment variable; these are "
            "separate sources and neither is set in the current active profile.\n"
            "Set [bold]OMICS_OS_API_KEY[/bold], run [bold]lobster init[/bold] "
            "to configure a profile, or run: "
            "[bold]lobster cloud login --api-key <key>[/bold]"
        )
        raise typer.Exit(0)

    endpoint = get_endpoint()
    # Fail closed before attaching the bearer token (poisoned-endpoint exfil). (Codex P0)
    try:
        validate_endpoint(endpoint)
    except EndpointPolicyError as e:
        console.print(f"[red]Endpoint not in allowlist: {e}[/red]")
        raise typer.Exit(1)
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
            "Run: [bold]lobster cloud login --api-key <key>[/bold]"
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
    max_tokens = limits.get("max_tokens_per_request")
    table.add_row(
        "Max tokens/req", f"{max_tokens:,}" if isinstance(max_tokens, int) else "?"
    )

    console.print(table)


@cloud_app.command()
def logout() -> None:
    """Log out of Omics-OS Cloud."""
    from lobster.config.credentials import _clear_active_profile

    # Active-profile-only logout: on a V2 multi-profile file this removes the
    # active profile and preserves siblings (Codex P1). On V1 it deletes the file.
    _clear_active_profile()
    console.print("Logged out of Omics-OS Cloud.")


@cloud_app.command()
def chat(
    session_id: Optional[str] = typer.Option(
        None, "--session-id", "-s", help=h("Resume an existing cloud session")
    ),
    token: Optional[str] = typer.Option(
        None, "--token", help=h("Override stored auth token")
    ),
    endpoint: Optional[str] = typer.Option(
        None, "--endpoint", help=h("Custom cloud API endpoint")
    ),
    stream_endpoint: Optional[str] = typer.Option(
        None,
        "--stream-endpoint",
        help=h("Custom stream origin (e.g., https://stream.omics-os.com)"),
    ),
    project_id: Optional[str] = typer.Option(
        None, "--project-id", "-p", help=h("Associate session with a cloud project")
    ),
) -> None:
    """Start an interactive cloud chat session (lobster-cli, direct connection)."""
    import os
    import subprocess

    from lobster.cli_internal.npm_launcher import find_npm_binary, install_hint

    binary = find_npm_binary()
    if not binary:
        console.print(
            "[red]Error:[/red] Cloud TUI not installed.\n"
            f"Install: [bold]{install_hint()}[/bold]"
        )
        raise typer.Exit(1)

    if not os.isatty(0):
        console.print("[red]Error:[/red] Cloud chat requires an interactive terminal.")
        raise typer.Exit(1)

    api_url = endpoint or "https://app.omics-os.com/api/v1"

    from lobster.cli_internal.commands.light.cloud_query import (
        CloudQueryError,
        _validate_endpoint,
        derive_stream_base,
    )

    if endpoint:
        try:
            _validate_endpoint(endpoint.rstrip("/"))
        except CloudQueryError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
    if stream_endpoint:
        try:
            _validate_endpoint(stream_endpoint.rstrip("/"))
        except CloudQueryError as e:
            console.print(f"[red]Error:[/red] Stream endpoint: {e}")
            raise typer.Exit(1)

    stream_base = derive_stream_base(api_url, stream_endpoint)

    cmd = [binary, "chat", "--cloud", f"--api-url={api_url}"]
    if stream_base != api_url:
        cmd.append(f"--stream-endpoint={stream_base}")
    if session_id:
        cmd.append(f"--session-id={session_id}")
    if token:
        cmd.append(f"--token={token}")
    if project_id:
        cmd.append(f"--project-id={project_id}")

    proc = subprocess.Popen(cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
    try:
        proc.wait()
    except KeyboardInterrupt:
        import signal

        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
    raise typer.Exit(proc.returncode or 0)


@cloud_app.command()
def query(
    question: str = typer.Argument(..., help=h("Question to ask")),
    session_id: Optional[str] = typer.Option(
        None, "--session-id", "-s", help=h("Session UUID to continue (or 'latest')")
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        help=h("Override stored auth token (prefer OMICS_OS_API_KEY env var)"),
    ),
    endpoint: Optional[str] = typer.Option(
        None,
        "--endpoint",
        help=h("Custom REST API origin (e.g., https://app.omics-os.com)"),
    ),
    stream_endpoint: Optional[str] = typer.Option(
        None,
        "--stream-endpoint",
        help=h("Custom stream origin (e.g., https://stream.omics-os.com)"),
    ),
    unsafe_endpoint: bool = typer.Option(
        False,
        "--unsafe-endpoint",
        is_flag=True,
        hidden=True,
        help=h("Skip endpoint allowlist validation (DANGEROUS)"),
    ),
    stream: bool = typer.Option(
        False, "--stream/--no-stream", help=h("Stream text as it arrives")
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", is_flag=True, help=h("Output JSON only on stdout")
    ),
    project_id: Optional[str] = typer.Option(
        None,
        "--project-id",
        "-p",
        help=h("Associate session with a cloud project (personal projects only)"),
    ),
) -> None:
    """Send a single query to Omics-OS Cloud (agents run on ECS Fargate)."""
    import json as _json
    import math
    import sys as _sys

    from lobster.cli_internal.commands.light.cloud_query import (
        CloudQueryError,
        _validate_endpoint,
        cancel_cloud_run,
        derive_stream_base,
        fetch_workspace_files,
        resolve_auth,
        resolve_cloud_session,
        resolve_rest_base,
        stream_cloud_query,
        strip_ansi,
    )

    def _emit_json_error(error: str, sid: Optional[str] = None) -> None:
        print(_json.dumps({"success": False, "error": error, "session_id": sid}))

    try:
        if not question.strip():
            raise CloudQueryError("Question cannot be empty.")

        # S1 fix: validate endpoints BEFORE resolving auth (which may send refresh token)
        rest_base = resolve_rest_base(endpoint)
        stream_base = derive_stream_base(rest_base, stream_endpoint)

        if not unsafe_endpoint:
            _validate_endpoint(rest_base)
            _validate_endpoint(stream_base)

        headers = resolve_auth(token_override=token)

        sid = resolve_cloud_session(
            rest_base, headers, session_id, project_id=project_id, client_source="cli"
        )

    except CloudQueryError as e:
        if json_output:
            _emit_json_error(str(e))
        else:
            from rich.markup import escape

            console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1)

    if not json_output:
        console.print(f"[dim]session: {sid}[/dim]")

    saw_text_delta = False
    last_worker_status_line: Optional[str] = None
    text_callback = None
    worker_status_callback = None
    if stream and not json_output:

        def text_callback(delta: str):
            nonlocal saw_text_delta
            saw_text_delta = True
            try:
                _sys.stdout.write(delta)
                _sys.stdout.flush()
            except BrokenPipeError:
                raise KeyboardInterrupt

        def worker_status_callback(status: Optional[dict]):
            nonlocal last_worker_status_line
            if saw_text_delta or not status:
                return
            phase = str(status.get("phase") or status.get("status") or "starting")
            message = str(status.get("message") or "Starting analysis worker")
            transport = status.get("transport")
            label = "cloud worker" if transport == "ecs" else "worker"
            line = f"{label}: {phase.replace('_', ' ')} - {message}"
            if line == last_worker_status_line:
                return
            last_worker_status_line = line
            _sys.stderr.write(f"{line}\n")
            _sys.stderr.flush()

    try:
        result = stream_cloud_query(
            stream_base,
            headers,
            sid,
            question,
            on_text_delta=text_callback,
            on_worker_status=worker_status_callback,
        )
    except KeyboardInterrupt:
        if not json_output:
            console.print("\n[yellow]Cancelled.[/yellow]")
        cancel_cloud_run(rest_base, headers, sid)
        if json_output:
            _emit_json_error("Cancelled by user", sid)
        raise typer.Exit(130)

    if stream and not json_output:
        try:
            _sys.stdout.write("\n")
        except BrokenPipeError:
            pass

    if result.error_detail:
        if json_output:
            _emit_json_error(result.error_detail, sid)
        else:
            from rich.markup import escape

            console.print(f"[red]Error:[/red] {escape(result.error_detail)}")
        raise typer.Exit(1)

    workspace_files: list = []
    if result.tool_calls:
        try:
            workspace_files = fetch_workspace_files(rest_base, headers, sid)
        except Exception:
            pass

    if json_output:
        print(
            _json.dumps(
                {
                    "success": result.success,
                    "response": result.display_text,
                    "session_id": sid,
                    "active_agent": result.active_agent,
                    "token_usage": result.token_usage,
                    "session_title": result.session_title,
                    "finish_reason": result.finish_reason,
                    "worker_status": result.worker_status,
                    "workspace_files": workspace_files,
                }
            )
        )
        return

    from rich.markdown import Markdown
    from rich.markup import escape
    from rich.rule import Rule

    # S3 fix: escape backend-controlled agent name before Rich markup interpolation
    agent_label = result.active_agent or "Supervisor"
    agent_display = escape(agent_label.replace("_", " ").title())

    if not stream:
        console.print()
        console.print(
            Rule(
                f"[bold #FF6B35]Lobster · {agent_display}[/bold #FF6B35]",
                style="grey27",
            )
        )
        console.print(Markdown(strip_ansi(result.display_text)))

    footer_parts = [f"session={sid}"]
    if result.token_usage:
        cost = result.token_usage.get("total_cost_usd")
        if isinstance(cost, (int, float)) and math.isfinite(cost):
            footer_parts.append(f"cost ${cost:.4f}")
    footer_parts.append("continue with --session-id latest")
    console.print(f"\n[dim]{'  ·  '.join(footer_parts)}[/dim]")

    if workspace_files:
        file_count = len(workspace_files)
        console.print(
            f"[dim]{file_count} file{'s' if file_count != 1 else ''} in workspace · "
            f"download: lobster cloud files download {sid} <path>[/dim]"
        )


@cloud_app.command("projects")
def projects(
    json_output: bool = typer.Option(
        False, "--json", "-j", is_flag=True, help=h("Output JSON")
    ),
) -> None:
    """List your Omics-OS Cloud projects."""
    from lobster.cli_internal.commands.light.cloud_query import (
        CloudQueryError,
        _validate_endpoint,
        resolve_auth,
        resolve_rest_base,
    )

    try:
        rest_base = resolve_rest_base()
        _validate_endpoint(rest_base)
        headers = resolve_auth()
    except CloudQueryError as e:
        if json_output:
            import json as _json

            print(_json.dumps({"success": False, "error": str(e)}))
        else:
            console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    import httpx

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                f"{rest_base}/api/v1/projects",
                headers={**headers, "Accept": "application/json"},
            )
        if resp.status_code in (403, 404):
            if json_output:
                print(
                    '{"success": false, "error": "Projects not available (feature may be disabled)"}'
                )
            else:
                console.print(
                    "[yellow]Projects not available.[/yellow] Feature may not be enabled for your account."
                )
            raise typer.Exit(0)
        if not resp.is_success:
            if json_output:
                import json as _json

                print(
                    _json.dumps({"success": False, "error": f"HTTP {resp.status_code}"})
                )
            else:
                console.print(f"[red]Error:[/red] {resp.status_code}")
            raise typer.Exit(1)
        try:
            data = resp.json()
        except (ValueError, Exception):
            if json_output:
                import json as _json

                print(
                    _json.dumps({"success": False, "error": "Invalid JSON from server"})
                )
            else:
                console.print("[red]Error:[/red] Invalid response from server")
            raise typer.Exit(1)
    except httpx.HTTPError as e:
        if json_output:
            import json as _json

            print(_json.dumps({"success": False, "error": str(e)}))
        else:
            console.print(f"[red]Cannot reach cloud:[/red] {e}")
        raise typer.Exit(1)

    project_list = (
        data.get("projects", data)
        if isinstance(data, dict)
        else data
        if isinstance(data, list)
        else []
    )

    if json_output:
        import json as _json

        print(_json.dumps({"success": True, "projects": project_list}))
        return

    if not project_list:
        console.print("[dim]No projects found. Create one at app.omics-os.com[/dim]")
        return

    table = Table(title="Omics-OS Cloud Projects", border_style="cyan")
    table.add_column("ID", style="dim")
    table.add_column("Name", style="bold")
    table.add_column("Datasets")
    table.add_column("Created")
    for p in project_list:
        if not isinstance(p, dict):
            continue
        table.add_row(
            str(p.get("project_id") or p.get("id", "")),
            str(p.get("name", "")),
            str(p.get("dataset_count", "-")),
            str(p.get("created_at", ""))[:10],
        )
    console.print(table)
