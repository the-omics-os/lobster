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
from rich.console import Console
from rich.markup import escape
from rich.markdown import Markdown
from rich.rule import Rule

from lobster.cli_internal.commands.heavy.session_infra import (
    _maybe_print_timings,
    init_client,
    should_show_progress,
    validate_startup_or_raise_startup_diagnostic,
)
from lobster.cli_internal.startup_diagnostics import (
    StartupDiagnosticError,
    format_startup_diagnostic_text,
    render_startup_diagnostic_rich,
)
from lobster.ui import LobsterTheme
from lobster.ui.console_manager import get_console_manager

if TYPE_CHECKING:
    from lobster.core.client import AgentClient

logger = logging.getLogger(__name__)

console_manager = get_console_manager()
console = console_manager.console


def _query_trace_callbacks(client: "AgentClient") -> list[Any]:
    """Return callback handlers that expose query execution activity."""
    callbacks = getattr(client, "callbacks", None) or []
    return [
        callback
        for callback in callbacks
        if hasattr(callback, "events") or hasattr(callback, "has_visible_output")
    ]


def _query_callbacks_displayed_reasoning(callbacks: list[Any]) -> bool:
    """Return True when a callback already rendered reasoning during execution."""
    for callback in callbacks:
        checker = getattr(callback, "has_displayed_reasoning", None)
        if callable(checker):
            try:
                if checker():
                    return True
            except Exception:
                continue
    return False


def _query_execution_summary(callbacks: list[Any]) -> str:
    """Build a compact execution summary from callback events."""
    agent_order: list[str] = []
    tool_order: list[str] = []

    for callback in callbacks:
        for event in getattr(callback, "events", []) or []:
            event_type = getattr(getattr(event, "type", None), "name", "")
            if event_type == "HANDOFF":
                target = str(getattr(event, "metadata", {}).get("to", "") or "").strip()
                if target and target not in agent_order:
                    agent_order.append(target)
            elif event_type == "AGENT_START":
                agent_name = str(getattr(event, "agent_name", "") or "").strip()
                if (
                    agent_name
                    and agent_name != "supervisor"
                    and agent_name not in agent_order
                ):
                    agent_order.append(agent_name)
            elif event_type == "TOOL_START":
                tool_name = str(
                    getattr(event, "metadata", {}).get("tool_name", "") or ""
                ).strip()
                if tool_name:
                    tool_order.append(tool_name)

    summary_parts = []
    if agent_order:
        agents = " -> ".join(
            agent_name.replace("_", " ").title() for agent_name in agent_order
        )
        summary_parts.append(f"agents: {agents}")
    if tool_order:
        tools = " -> ".join(tool_order)
        summary_parts.append(f"tools: {tools}")
    return "  |  ".join(summary_parts)


def _display_streaming_response(
    client: "AgentClient", user_input: str, console: Console
) -> Dict[str, Any]:
    """Delegate streaming query rendering to the shared slash-command helper."""
    from lobster.cli_internal.commands.heavy.slash_commands import (
        _display_streaming_response as _impl,
    )

    return _impl(client, user_input, console)


def _compact_query_usage_summary(summary: str) -> str:
    """Normalize token/cost summaries for compact one-shot footer display."""
    summary = str(summary or "").strip()
    if not summary:
        return ""

    lowered = summary.lower()
    if lowered.startswith("session cost:"):
        return "cost " + summary.split(":", 1)[1].strip()
    if lowered.startswith("session:"):
        return summary.split(":", 1)[1].strip()
    return summary


def _query_usage_summary(client: "AgentClient", verbose: bool) -> str:
    """Return query usage summary, compact for normal mode and detailed for verbose."""
    tracker = getattr(client, "token_tracker", None)
    if not tracker:
        return ""

    try:
        if verbose and hasattr(tracker, "get_verbose_summary"):
            return str(tracker.get_verbose_summary() or "").strip()
        if hasattr(tracker, "get_minimal_summary"):
            return _compact_query_usage_summary(
                str(tracker.get_minimal_summary() or "")
            )
        if hasattr(tracker, "get_verbose_summary"):
            return _compact_query_usage_summary(
                str(tracker.get_verbose_summary() or "")
            )
    except Exception:
        return ""
    return ""


def _query_header_label(result: Dict[str, Any]) -> str:
    """Build a compact header label for human-facing query output."""
    agent_name = str(result.get("last_agent", "") or "").strip()
    if not agent_name or agent_name == "__end__":
        return "Lobster"
    return f"Lobster · {agent_name.replace('_', ' ').title()}"


def _query_answer_text(result: Dict[str, Any]) -> str:
    """Return the final user-facing answer text."""
    answer_text = str(result.get("text", "") or "").strip()
    if answer_text:
        return answer_text
    return str(result.get("response", "") or "").strip()


def _render_human_query_result(
    console: Console,
    result: Dict[str, Any],
    client: "AgentClient",
    *,
    reasoning: bool,
    verbose: bool,
) -> None:
    """Render a clean, one-shot query response for human readers."""
    trace_callbacks = _query_trace_callbacks(client)
    header = _query_header_label(result)
    usage_summary = _query_usage_summary(client, verbose)
    answer_text = _query_answer_text(result)
    reasoning_text = str(result.get("reasoning", "") or "").strip()
    execution_summary = _query_execution_summary(trace_callbacks) if verbose else ""
    footer_parts = []

    session = str(getattr(client, "session_id", "") or "").strip()
    if session:
        footer_parts.append(f"session={session}")
    if usage_summary and "\n" not in usage_summary:
        footer_parts.append(usage_summary)
    footer_parts.append("continue with --session-id latest")

    console.print()
    console.print(
        Rule(
            f"[bold {LobsterTheme.PRIMARY_ORANGE}]{header}[/bold {LobsterTheme.PRIMARY_ORANGE}]",
            style="grey27",
        )
    )
    if reasoning and reasoning_text and not _query_callbacks_displayed_reasoning(
        trace_callbacks
    ):
        console.print(f"[dim]{escape(reasoning_text)}[/dim]")
        console.print()
    console.print(Markdown(answer_text))
    console.print()
    if execution_summary:
        console.print(f"[dim]{escape(execution_summary)}[/dim]")
    if footer_parts:
        console.print(f"[dim]{'  ·  '.join(footer_parts)}[/dim]")
    if usage_summary and "\n" in usage_summary:
        console.print(f"[dim]{usage_summary}[/dim]")


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

    try:
        validate_startup_or_raise_startup_diagnostic(
            workspace=workspace,
            provider_override=provider,
        )
    except StartupDiagnosticError as exc:
        if json_output:
            print(
                json.dumps(
                    {
                        "success": False,
                        "error": format_startup_diagnostic_text(exc.diagnostic),
                        "error_code": exc.diagnostic.code,
                    }
                )
            )
        else:
            render_startup_diagnostic_rich(console, exc.diagnostic)
        raise typer.Exit(exc.diagnostic.exit_code)

    # Handle session loading for continuity
    session_file_to_load = None
    session_id_for_client = None

    if session_id:
        from lobster.core.workspace import resolve_workspace

        from lobster.cli_internal.commands.heavy.session_infra import (
            resolve_session_continuation,
        )

        workspace_path = resolve_workspace(explicit_path=workspace, create=True)
        (
            session_file_to_load,
            session_id_for_client,
            found_existing_session,
        ) = resolve_session_continuation(workspace_path, session_id)

        if session_id == "latest":
            if found_existing_session and session_id_for_client and not json_output:
                console.print(
                    f"[cyan]📂 Loading latest session: {session_id_for_client}[/cyan]"
                )
            elif not found_existing_session and not json_output:
                console.print(
                    "[yellow]⚠️  No previous sessions found - creating new session[/yellow]"
                )
                session_id_for_client = None
        elif found_existing_session:
            if session_id_for_client and not json_output:
                console.print(
                    f"[cyan]📂 Loading session: {session_id_for_client}[/cyan]"
                )
        else:
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
        trace_mode = reasoning or verbose
        use_streaming = stream and not trace_mode

        if use_streaming:
            result = _display_streaming_response(client, question, console)
        elif should_show_progress(client):
            with console.status("[red]🦞 Processing query...[/red]"):
                result = client.query(question)
        else:
            if trace_mode:
                if stream:
                    console.print(
                        "[dim]◀ Lobster  Trace mode active; final answer renders on completion.[/dim]"
                    )
                else:
                    console.print("[dim]◀ Lobster  Working…[/dim]")
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
            _render_human_query_result(
                console,
                result,
                client,
                reasoning=reasoning,
                verbose=verbose,
            )
    else:
        console.print(
            f"[bold red on white] ⚠️  Error [/bold red on white] [red]{result['error']}[/red]"
        )

    _maybe_print_timings(client, "Query")


# ============================================================================
# lobster command — fast programmatic slash command access
# ============================================================================
