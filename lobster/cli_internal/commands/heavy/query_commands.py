"""
Query command body.

Extracted from cli.py -- the `lobster query` single-turn command.
"""

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import typer
from rich import box
from rich.markdown import Markdown
from rich.panel import Panel

from lobster.cli_internal.commands.heavy.session_infra import (
    _maybe_print_timings,
    init_client,
    should_show_progress,
)
from lobster.ui.console_manager import get_console_manager

if TYPE_CHECKING:
    from lobster.core.client import AgentClient

logger = logging.getLogger(__name__)

console_manager = get_console_manager()
console = console_manager.console

def query_impl(
    question: str,
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory. Can also be set via LOBSTER_WORKSPACE env var. Default: ./.lobster_workspace",
    ),
    session_id: Optional[str] = typer.Option(
        None,
        "--session-id",
        "-s",
        help="Session ID to continue (use 'latest' for most recent session in workspace)",
    ),
    reasoning: bool = typer.Option(
        False,
        "--reasoning",
        is_flag=True,
        help="Show agent reasoning and thinking process",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed tool usage and agent activity"
    ),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Enable debug mode with detailed logging"
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    profile_timings: Optional[bool] = typer.Option(
        None,
        "--profile-timings/--no-profile-timings",
        help="Enable timing diagnostics for data manager operations",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM provider to use (bedrock, anthropic, ollama). Overrides auto-detection.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use (e.g., claude-4-sonnet, llama3:70b-instruct, mixtral:8x7b). Overrides configuration.",
    ),
    stream: bool = typer.Option(
        False,
        "--stream/--no-stream",
        help="Enable real-time text streaming (default: off for query mode)",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        is_flag=True,
        help="Output result as JSON for programmatic consumption. Suppresses all Rich formatting.",
    ),
):
    """
    Send a single query to the agent system.

    Use --session-id to continue a previous conversation:
      lobster query "follow up question" --session-id latest
      lobster query "follow up question" --session-id session_20241208_150000

    Use --json for machine-readable output (no Rich formatting):
      lobster query "analyze data" --json | jq .response

    Agent reasoning is shown by default. Use --no-reasoning to disable.
    """
    # In JSON mode, redirect Rich console to stderr so only JSON hits stdout
    if json_output:
        console.file = sys.stderr

    # Check for configuration
    env_file = Path.cwd() / ".env"
    if not env_file.exists():
        if json_output:
            print(
                json.dumps(
                    {
                        "success": False,
                        "error": "No configuration found. Run 'lobster init' first.",
                    }
                )
            )
            raise typer.Exit(1)
        console.print("[red]❌ No configuration found. Run 'lobster init' first.[/red]")
        raise typer.Exit(1)

    # Handle session loading for continuity
    # Determine session_id before client initialization
    session_file_to_load = None
    session_id_for_client = None

    if session_id:
        # Resolve workspace early to check for session files
        from lobster.core.workspace import resolve_workspace

        workspace_path = resolve_workspace(explicit_path=workspace, create=True)

        if session_id == "latest":
            # Find most recent session file
            session_files = list(workspace_path.glob("session_*.json"))
            if session_files:
                # Sort by modification time (most recent first)
                session_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                session_file_to_load = session_files[0]
                if not json_output:
                    console.print(
                        f"[cyan]📂 Loading latest session: {session_file_to_load.name}[/cyan]"
                    )
            else:
                if not json_output:
                    console.print(
                        "[yellow]⚠️  No previous sessions found - creating new session[/yellow]"
                    )
        else:
            # Explicit session ID provided
            # Try with session_ prefix first
            session_file_candidate = workspace_path / f"session_{session_id}.json"
            if not session_file_candidate.exists():
                # Try exact filename
                session_file_candidate = workspace_path / f"{session_id}.json"

            if session_file_candidate.exists():
                session_file_to_load = session_file_candidate
            else:
                # Session file doesn't exist - use this session_id for new session
                session_id_for_client = session_id
                if not json_output:
                    console.print(f"[cyan]📂 Creating new session: {session_id}[/cyan]")

    # Initialize client with custom session_id if new session
    client = init_client(
        workspace,
        reasoning,
        verbose,
        debug,
        profile_timings,
        provider,
        model,
        session_id_for_client,
    )

    # Load session if found
    if session_file_to_load:
        try:
            load_result = client.load_session(session_file_to_load)
            if not json_output:
                console.print(
                    f"[green]✓ Loaded {load_result['messages_loaded']} "
                    f"previous messages[/green]"
                )
                if load_result.get("original_session_id"):
                    console.print(
                        f"[dim]  Original session: "
                        f"{load_result['original_session_id']}[/dim]"
                    )
        except Exception as e:
            if not json_output:
                console.print(f"[red]❌ Failed to load session: {e}[/red]")
                console.print("[yellow]   Creating new session instead[/yellow]")

    # Process query
    if json_output:
        # JSON mode: no streaming, no Rich output, no spinners
        result = client.query(question)
    else:
        # Determine if streaming is appropriate
        use_streaming = stream and should_show_progress(client)

        if use_streaming:
            result = _display_streaming_response(client, question, console)
        elif should_show_progress(client):
            with console.status("[red]🦞 Processing query...[/red]"):
                result = client.query(question)
        else:
            # In verbose/reasoning mode, no progress indication
            result = client.query(question)

    # JSON output mode: emit structured JSON and exit
    if json_output:
        json_result = {
            "success": result.get("success", False),
            "response": result.get("response", ""),
            "session_id": getattr(client, "session_id", None),
        }
        if result.get("success"):
            json_result["last_agent"] = result.get("last_agent")
            token_usage = result.get("token_usage")
            if (
                not token_usage
                and hasattr(client, "token_tracker")
                and client.token_tracker
            ):
                token_usage = (
                    client.token_tracker.get_summary()
                    if hasattr(client.token_tracker, "get_summary")
                    else None
                )
            if token_usage:
                json_result["token_usage"] = token_usage
        else:
            json_result["error"] = result.get("error", "Unknown error")
        print(json.dumps(json_result))
        return

    # Display or save result (Rich mode)
    if result["success"]:
        if output:
            output.write_text(result["response"])
            console.print(
                f"[bold red]✓[/bold red] [white]Response saved to:[/white] [grey74]{output}[/grey74]"
            )
        else:
            # For streaming, response was shown live - now render final markdown
            # For non-streaming, show the panel
            if use_streaming:
                agent_name = result.get("last_agent", "supervisor")
                agent_display = (
                    agent_name.replace("_", " ").title()
                    if agent_name and agent_name != "__end__"
                    else "Lobster"
                )
                console.print(f"\n[dim]◀ {agent_display}[/dim]")
                console.print(Markdown(result["response"]))
            else:
                console.print(
                    Panel(
                        Markdown(result["response"]),
                        title="[bold white on red] 🦞 Lobster Response [/bold white on red]",
                        border_style="red",
                        box=box.DOUBLE,
                    )
                )

        # Display session ID for continuity
        console.print(
            f"\n[dim]Session: {client.session_id} "
            f"(use --session-id latest for follow-ups)[/dim]"
        )

        # Display session token usage summary
        if hasattr(client, "token_tracker") and client.token_tracker:
            if verbose:
                summary = client.token_tracker.get_verbose_summary()
            else:
                summary = client.token_tracker.get_minimal_summary()
            if summary:
                console.print(f"[dim]{summary}[/dim]")
    else:
        console.print(
            f"[bold red on white] ⚠️  Error [/bold red on white] [red]{result['error']}[/red]"
        )

    _maybe_print_timings(client, "Query")


# ============================================================================
# lobster command — fast programmatic slash command access
# ============================================================================


