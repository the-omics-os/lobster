"""
Data management subcommands for Lobster CLI.

Provides read-only provenance and audit trail access for cloud datasets and sessions.

Commands:
    lobster data events <entity_id>  - Show provenance events for a dataset or session
"""

import csv
import io
import logging
import re
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)

data_app = typer.Typer(
    name="data",
    help="Data provenance and audit trail",
    no_args_is_help=True,
)

console = Console()

# Entity IDs are UUIDs or ULID-like strings — reject anything with path/query chars
_SAFE_ENTITY_ID = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _validate_entity_id(entity_id: str) -> bool:
    """Reject entity IDs that could cause path traversal or query injection."""
    if not entity_id or len(entity_id) > 128:
        return False
    return bool(_SAFE_ENTITY_ID.match(entity_id))


def _sanitize_cell(value: str) -> str:
    """Prevent CSV formula injection by prefixing dangerous leading characters."""
    if value and value[0] in ("=", "+", "-", "@", "\t", "\r"):
        return f"'{value}"
    return value


def _strip_control_chars(text: str) -> str:
    """Remove ANSI escape sequences and control characters from untrusted text."""
    text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


def _fetch_events(
    endpoint: str, token: str, entity_id: str, limit: int
) -> Optional[dict]:
    """Try fetching events as dataset first, then session."""
    import httpx

    headers = {"Authorization": f"Bearer {token}"}

    for entity_type in ("datasets", "sessions"):
        url = f"{endpoint}/api/v1/{entity_type}/{entity_id}/events?limit={limit}"
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                data["_entity_type"] = entity_type.rstrip("s")
                return data
            if resp.status_code == 404:
                continue
            if resp.status_code == 401:
                console.print(
                    "[red]Authentication failed.[/red]\n"
                    "Run: [bold]lobster cloud login[/bold]"
                )
                return None
            if resp.status_code == 403:
                console.print(f"[red]Access denied for {entity_type[:-1]} {entity_id}.[/red]")
                return None
        except (httpx.ConnectError, httpx.TimeoutException):
            console.print("[red]Cannot reach Omics-OS Cloud.[/red]")
            return None

    console.print(
        f"[yellow]No dataset or session found with ID: {entity_id}[/yellow]\n"
        "[dim]Check the ID and ensure you have access.[/dim]"
    )
    return None


@data_app.command()
def events(
    entity_id: str = typer.Argument(help="Dataset or session ID"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max events to fetch (1-200)"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table or csv"),
) -> None:
    """Show provenance events for a dataset or session.

    Displays the audit trail including creation, file additions, and result promotions.
    Tries the ID as a dataset first, then as a session.
    """
    from lobster.config.credentials import get_api_key, get_endpoint

    if not _validate_entity_id(entity_id):
        console.print(
            "[red]Invalid entity ID.[/red]\n"
            "[dim]Entity IDs must be alphanumeric (with hyphens/underscores), max 128 chars.[/dim]"
        )
        raise typer.Exit(1)

    api_key = get_api_key()
    if not api_key:
        console.print(
            "[yellow]Not connected to Omics-OS Cloud.[/yellow]\n"
            "Run: [bold]lobster cloud login[/bold]"
        )
        raise typer.Exit(0)

    limit = max(1, min(limit, 200))
    endpoint = get_endpoint()
    data = _fetch_events(endpoint, api_key, entity_id, limit)
    if data is None:
        raise typer.Exit(1)

    events_list = data.get("events", [])
    entity_type = data.get("_entity_type", "entity")

    if not events_list:
        console.print(f"[dim]No events found for {entity_type} {entity_id}.[/dim]")
        raise typer.Exit(0)

    if format == "csv":
        _print_csv(events_list)
    else:
        _print_table(events_list, entity_type, entity_id, data.get("count", len(events_list)))


def _print_table(events_list: list, entity_type: str, entity_id: str, count: int) -> None:
    """Render events as a Rich table."""
    table = Table(
        title=f"Events for {entity_type} {entity_id} ({count} shown)",
        border_style="cyan",
    )
    table.add_column("Timestamp", style="dim", no_wrap=True)
    table.add_column("Event Type", style="bold")
    table.add_column("Actor", style="dim")
    table.add_column("Details")

    for event in events_list:
        timestamp = event.get("timestamp", "?")
        if "T" in timestamp:
            timestamp = timestamp.split("T")[0] + " " + timestamp.split("T")[1][:8]
        event_type = event.get("event_type", "?")
        actor = event.get("actor", "?")
        payload = event.get("payload", {})
        details = ", ".join(
            f"{k}={_strip_control_chars(str(v))}" for k, v in payload.items()
        ) if payload else ""
        table.add_row(timestamp, event_type, actor, details)

    console.print(table)


def _print_csv(events_list: list) -> None:
    """Render events as CSV to stdout with formula injection protection."""
    import json

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["timestamp", "event_id", "event_type", "entity_type", "actor", "payload"])
    for event in events_list:
        writer.writerow([
            _sanitize_cell(str(event.get("timestamp", ""))),
            _sanitize_cell(str(event.get("event_id", ""))),
            _sanitize_cell(str(event.get("event_type", ""))),
            _sanitize_cell(str(event.get("entity_type", ""))),
            _sanitize_cell(str(event.get("actor", ""))),
            _sanitize_cell(json.dumps(event.get("payload", {}))),
        ])
    sys.stdout.write(output.getvalue())
