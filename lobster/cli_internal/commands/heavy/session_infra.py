"""
Session infrastructure for CLI commands.

Shared session management: client initialization, adapters, history,
progress indicators. Extracted from cli.py for modularity.
"""

import json
import logging
import os
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from lobster.ui.console_manager import get_console_manager

if TYPE_CHECKING:
    from lobster.core.client import AgentClient

logger = logging.getLogger(__name__)

console_manager = get_console_manager()
console = console_manager.console

_COMMAND_HISTORY_LOCK = threading.Lock()

# Go TUI active flag: when True, Rich progress indicators are suppressed
# because the Go TUI handles its own spinner/progress display.
_go_tui_active = False


def set_go_tui_active(active: bool) -> None:
    """Set whether the Go TUI is the active UI backend."""
    global _go_tui_active
    _go_tui_active = active


# Extraction cache manager loaded lazily to avoid triggering all agent imports at startup
_ExtractionCacheManager = None
_extraction_cache_checked = False


def _get_extraction_cache_manager():
    """Lazy loader for ExtractionCacheManager (premium feature)."""
    global _ExtractionCacheManager, _extraction_cache_checked
    if not _extraction_cache_checked:
        from lobster.core.component_registry import component_registry

        _ExtractionCacheManager = component_registry.get_service("extraction_cache")
        _extraction_cache_checked = True
    return _ExtractionCacheManager


# ============================================================================
# Progress Management
# ============================================================================


class NoOpProgress:
    """No-operation progress context manager for verbose/reasoning modes."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def add_task(self, *args, **kwargs):
        """No-op task addition."""
        return None

    def update(self, *args, **kwargs):
        """No-op update."""
        pass

    def remove_task(self, *args, **kwargs):
        """No-op task removal."""
        pass


def should_show_progress(client_arg: Optional["AgentClient"] = None) -> bool:
    """
    Determine if progress indicators should be shown based on current mode.

    Returns False (no progress) when:
    - Go TUI is active (it has its own spinner)
    - Reasoning mode is enabled
    - Verbose mode is enabled
    - Any callback has verbose/show_tools enabled

    Returns True (show progress) otherwise.
    """
    if _go_tui_active:
        return False

    # Use provided client or try to get from cli module
    c = client_arg
    if c is None:
        try:
            from lobster import cli as _cli_mod

            c = _cli_mod.client
        except Exception:
            pass
    if not c:
        return True  # Default to showing progress if no client

    # Don't show progress if reasoning mode is enabled
    if hasattr(c, "enable_reasoning") and c.enable_reasoning:
        return False

    # Check callbacks for verbose settings
    if hasattr(c, "callbacks") and c.callbacks:
        for callback in c.callbacks:
            if hasattr(callback, "verbose") and callback.verbose:
                return False
            if hasattr(callback, "show_tools") and callback.show_tools:
                return False

    # Check custom_callbacks for verbose settings
    if hasattr(c, "custom_callbacks") and c.custom_callbacks:
        for callback in c.custom_callbacks:
            if hasattr(callback, "verbose") and callback.verbose:
                return False
            if hasattr(callback, "show_tools") and callback.show_tools:
                return False

    return True


def create_progress(description: str = "", client_arg: Optional["AgentClient"] = None):
    """
    Create a progress indicator that respects verbose/reasoning mode.

    In verbose/reasoning mode: Returns no-op progress manager
    In normal mode: Returns actual Progress spinner

    Args:
        description: Initial progress description
        client_arg: Optional client to check mode (uses global if not provided)

    Returns:
        Either a Progress object or NoOpProgress based on mode
    """
    if not should_show_progress(client_arg):
        return NoOpProgress()

    # Create actual progress spinner for normal mode
    try:
        progress_console = Console(stderr=True, force_terminal=True)
    except Exception:
        progress_console = console

    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=progress_console,
        transient=True,  # Always transient to clean up properly
    )


# ============================================================================
# Autocomplete Infrastructure
# ============================================================================


class LobsterClientAdapter:
    """Adapter to handle both local and cloud clients uniformly for autocomplete."""

    def __init__(self, client):
        self.client = client
        # Detect client type
        self.is_cloud = hasattr(client, "list_workspace_files") and hasattr(
            client, "session"
        )
        self.is_local = hasattr(client, "data_manager")

    def get_workspace_files(self) -> List[Dict[str, Any]]:
        """Get workspace files from either local or cloud client."""
        try:
            if self.is_cloud:
                # Cloud client has direct list_workspace_files method
                cloud_files = self.client.list_workspace_files()
                # Ensure consistent format
                return [self._normalize_file_info(f) for f in cloud_files]
            elif self.is_local and hasattr(self.client, "data_manager"):
                # Local client uses data_manager
                workspace_files = self.client.data_manager.list_workspace_files()
                return self._format_local_files(workspace_files)
            else:
                return []
        except Exception as e:
            # Graceful fallback for any errors
            console.print(f"[dim red]Error getting workspace files: {e}[/dim red]")
            return []

    def _normalize_file_info(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize file info to consistent format."""
        return {
            "name": file_info.get("name", ""),
            "path": file_info.get("path", ""),
            "size": file_info.get("size", 0),
            "type": file_info.get("type", "unknown"),
            "modified": file_info.get("modified", 0),
        }

    def _format_local_files(
        self, workspace_files: Dict[str, List]
    ) -> List[Dict[str, Any]]:
        """Format local workspace files to consistent format."""
        files = []
        for category, file_list in workspace_files.items():
            for file_info in file_list:
                files.append(
                    {
                        "name": file_info.get("name", ""),
                        "path": file_info.get("path", ""),
                        "size": file_info.get("size", 0),
                        "type": category,
                        "modified": file_info.get("modified", 0),
                    }
                )
        return files

    def can_read_files(self) -> bool:
        """Check if client supports file reading."""
        return self.is_cloud or (self.is_local and hasattr(self.client, "read_file"))


class CloudAwareCache:
    """Smart caching that adapts to client type."""

    def __init__(self, client):
        self.is_cloud = hasattr(client, "list_workspace_files") and hasattr(
            client, "session"
        )
        self.cache = {}
        self.timeouts = {
            "commands": float("inf"),  # Commands never change
            "files": 60 if self.is_cloud else 10,  # Longer cache for cloud
            "workspace": 30 if self.is_cloud else 5,
        }

    def get_or_fetch(self, key: str, fetch_func, category: str = "default"):
        """Get cached value or fetch if expired."""
        current_time = time.time()
        timeout = self.timeouts.get(category, 10)

        if (
            key not in self.cache
            or current_time - self.cache[key]["timestamp"] > timeout
        ):
            try:
                self.cache[key] = {"data": fetch_func(), "timestamp": current_time}
            except Exception as e:
                if self.is_cloud and (
                    "connection" in str(e).lower() or "timeout" in str(e).lower()
                ):
                    # For cloud connection errors, return stale cache if available
                    if key in self.cache:
                        console.print(
                            "[dim yellow]Using cached data due to connection issue[/dim yellow]"
                        )
                        return self.cache[key]["data"]
                raise e

        return self.cache[key]["data"]


# ============================================================================
# Command History
# ============================================================================


def _add_command_to_history(
    client: "AgentClient", command: str, summary: str, is_error: bool = False
) -> bool:
    """
    Add command execution to conversation history for AI context.

    BUG FIX #4: Enhanced error handling with full logging and file backup.
    """
    # 1. Validate inputs
    if not command or not summary:
        logger.warning("Empty command or summary provided to history logger")
        return False

    # 2. Check client compatibility
    if not hasattr(client, "messages") or not isinstance(client.messages, list):
        logger.info(
            f"Client type {type(client).__name__} doesn't support message history. "
            f"Commands will not be available in AI context."
        )
        return False

    # 3. Attempt primary logging (graph state)
    primary_logged = False
    try:
        from langchain_core.messages import AIMessage, HumanMessage

        human_message_command_usage = f"Command: {command}"
        status_prefix = "Error" if is_error else "Result"
        ai_message_command_response = f"Command {status_prefix}: {summary}"

        config = dict(configurable=dict(thread_id=client.session_id))
        human_msg = HumanMessage(content=human_message_command_usage)
        ai_msg = AIMessage(content=ai_message_command_response)

        client.messages.append(human_msg)
        client.messages.append(ai_msg)

        client.graph.update_state(
            config,
            dict(messages=[human_msg, ai_msg]),
        )

        logger.debug(f"Logged command to graph state: {command[:50]}")
        primary_logged = True

    except AttributeError as e:
        logger.error(
            f"Client missing required attributes for history logging: {e}. "
            f"Client type: {type(client).__name__}, "
            f"Has messages: {hasattr(client, 'messages')}, "
            f"Has graph: {hasattr(client, 'graph')}"
        )

    except Exception as e:
        logger.error(
            f"Failed to log command '{command}' to graph state: {e}",
            exc_info=True,
        )

    # 4. Backup to file (always, for audit trail and recovery)
    backup_logged = _backup_command_to_file(
        client, command, summary, is_error, primary_logged
    )

    return primary_logged or backup_logged


def _backup_command_to_file(
    client: "AgentClient",
    command: str,
    summary: str,
    is_error: bool,
    primary_logged: bool,
) -> bool:
    """Write command to backup file for audit trail and recovery."""
    try:
        from lobster.core.queue_storage import queue_file_lock

        history_dir = client.data_manager.workspace_path / ".lobster"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_file = history_dir / "command_history.jsonl"
        lock_path = history_file.with_suffix(".lock")

        from datetime import datetime

        record = {
            "timestamp": datetime.now().isoformat(),
            "session_id": client.session_id,
            "command": command,
            "summary": summary,
            "is_error": is_error,
            "logged_to_graph": primary_logged,
        }

        with queue_file_lock(_COMMAND_HISTORY_LOCK, lock_path):
            with open(history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        logger.debug(f"Backed up command to file: {command[:50]}")
        return True

    except Exception as e:
        logger.error(f"Failed to write command backup: {e}", exc_info=True)
        return False


# ============================================================================
# Lightweight client for `lobster command` (no LLM, no graph)
# ============================================================================


class CommandClient:
    """Minimal client providing only what shared command functions need.

    Skips AgentClient's expensive graph/LLM initialization. Provides
    data_manager, workspace_path, session_id, and publication_queue.
    """

    def __init__(self, workspace_path: Path, session_id: Optional[str] = None):
        from lobster.core.data_manager_v2 import DataManagerV2

        self.workspace_path = workspace_path
        self.session_id = session_id or "command"

        # Compute session_dir if real session_id provided
        session_dir = None
        if session_id and session_id != "command":
            session_dir = workspace_path / ".lobster" / "sessions" / session_id

            # Verify session exists
            if not session_dir.exists():
                available = self._list_available_sessions(workspace_path)
                raise FileNotFoundError(
                    f"Session '{session_id}' not found.\n"
                    f"Available sessions: {available}\n"
                    f"Or list all: lobster command 'workspace list'"
                )

        self.data_manager = DataManagerV2(
            workspace_path=workspace_path,
            session_dir=session_dir,
        )
        self.messages: list = []
        self.graph = None
        self.token_tracker = None
        self.provider_override = None
        self._publication_queue_ref = None
        self._publication_queue_unavailable = False

    def _list_available_sessions(self, workspace_path: Path) -> str:
        """List available sessions for error messages."""
        sessions_dir = workspace_path / ".lobster" / "sessions"
        if not sessions_dir.exists() or not list(sessions_dir.iterdir()):
            return "none (create one with: lobster query '...' --session-id <name>)"

        session_ids = sorted([d.name for d in sessions_dir.iterdir() if d.is_dir()])
        return ", ".join(session_ids[:5]) + (" ..." if len(session_ids) > 5 else "")

    @property
    def publication_queue(self):
        if self._publication_queue_unavailable:
            return None
        if self._publication_queue_ref is None:
            pq = getattr(self.data_manager, "publication_queue", None)
            if pq is None:
                self._publication_queue_unavailable = True
                return None
            self._publication_queue_ref = pq
        return self._publication_queue_ref


# ============================================================================
# Profile Timings
# ============================================================================

PROFILE_TIMINGS_ENV = "LOBSTER_PROFILE_TIMINGS"


def _str_to_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def _resolve_profile_timings_flag(cli_flag: Optional[bool]) -> bool:
    if cli_flag is not None:
        return cli_flag
    env_value = _str_to_bool(os.environ.get(PROFILE_TIMINGS_ENV))
    return bool(env_value)


def _collect_profile_timings(
    client: "AgentClient", clear: bool = True
) -> Dict[str, Dict[str, float]]:
    timings: Dict[str, Dict[str, float]] = {}
    data_manager = getattr(client, "data_manager", None)
    if data_manager and hasattr(data_manager, "get_latest_timings"):
        dm_timings = data_manager.get_latest_timings(clear=clear)
        if dm_timings:
            timings["DataManager"] = dm_timings
    return timings


def _maybe_print_timings(client: "AgentClient", context: str) -> None:
    if not getattr(client, "profile_timings_enabled", False):
        return

    timing_sources = _collect_profile_timings(client, clear=True)
    if not timing_sources:
        return

    table = Table(title=f"{context} Timings", box=box.ROUNDED)
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Step", style="white")
    table.add_column("Seconds", justify="right")

    for component, entries in timing_sources.items():
        for step, value in sorted(
            entries.items(), key=lambda item: item[1], reverse=True
        ):
            table.add_row(component, step, f"{value:.2f}")

    console.print(table)


# ============================================================================
# Display Session
# ============================================================================


def display_session(client: "AgentClient"):
    """Display current session status with enhanced orange theming."""
    from lobster.config.agent_defaults import get_current_profile

    status = client.get_status()

    current_mode = get_current_profile()

    status_data = {
        "session_id": status["session_id"],
        "mode": current_mode,
        "messages": str(status["message_count"]),
        "workspace": status["workspace"],
        "data_loaded": status["has_data"],
    }

    if status["has_data"] and status["data_summary"]:
        summary = status["data_summary"]
        status_data["data_shape"] = str(summary.get("shape", "N/A"))
        status_data["memory_usage"] = summary.get("memory_usage", "N/A")

    console_manager.print_status_panel(status_data, "Session Status")


# ============================================================================
# Client Initialization
# ============================================================================


def init_client(
    workspace: Optional[Path] = None,
    reasoning: bool = False,
    verbose: bool = False,
    debug: bool = False,
    profile_timings: Optional[bool] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    session_id: Optional[str] = None,
) -> "AgentClient":
    """Initialize either local or cloud client based on environment."""
    import typer

    from lobster.config.settings import settings
    from lobster.core.client import AgentClient
    from lobster.core.config_resolver import ConfigResolver, ConfigurationError
    from lobster.core.workspace import resolve_workspace
    from lobster.ui import LobsterTheme, setup_logging

    # Resolve workspace (create if needed for proper config loading)
    workspace_path = resolve_workspace(explicit_path=workspace, create=True)

    # Auto-inject provider config into new workspaces that lack one.
    from lobster.config.global_config import GlobalProviderConfig
    from lobster.config.workspace_config import WorkspaceProviderConfig

    if not WorkspaceProviderConfig.exists(workspace_path):
        env_provider = os.environ.get("LOBSTER_LLM_PROVIDER")
        if not env_provider and not GlobalProviderConfig.exists():
            default_workspace = Path.cwd() / ".lobster_workspace"
            if default_workspace != workspace_path and WorkspaceProviderConfig.exists(
                default_workspace
            ):
                source_config = WorkspaceProviderConfig.load(default_workspace)
                if source_config.global_provider:
                    source_config.save(workspace_path)
                    logger.info(
                        f"Auto-injected provider config into {workspace_path} "
                        f"from {default_workspace}"
                    )

    # Reload credentials for the target workspace
    settings.reload_credentials(workspace_path)

    resolver = ConfigResolver.get_instance(workspace_path)

    try:
        resolver.resolve_provider()
    except ConfigurationError as e:
        console.print(f"[red]{str(e)}[/red]")
        console.print(f"\n[yellow]{e.help_text}[/yellow]")
        raise typer.Exit(code=1)

    # Check for cloud API key
    cloud_key = os.environ.get("LOBSTER_CLOUD_KEY")
    cloud_endpoint = os.environ.get("LOBSTER_ENDPOINT")

    if cloud_key:
        console.print("[bold blue]Cloud API key detected...[/bold blue]")

        try:
            from lobster.lobster_cloud.client import CloudLobsterClient

            console.print("[bold blue]   Initializing Lobster Cloud...[/bold blue]")
            if cloud_endpoint:
                console.print(f"[dim blue]   Endpoint: {cloud_endpoint}[/dim blue]")

            client_kwargs = {"api_key": cloud_key}
            if cloud_endpoint:
                client_kwargs["endpoint"] = cloud_endpoint

            cloud_client = CloudLobsterClient(**client_kwargs)

            max_retries = 3
            retry_delay = 2

            for attempt in range(max_retries):
                try:
                    status_result = cloud_client.get_status()

                    if status_result.get("success", False):
                        console.print(
                            "[bold green]Cloud connection established[/bold green]"
                        )
                        console.print(
                            f"[dim blue]   Status: {status_result.get('status', 'unknown')}[/dim blue]"
                        )
                        if status_result.get("version"):
                            console.print(
                                f"[dim blue]   Version: {status_result.get('version')}[/dim blue]"
                            )
                        return cloud_client
                    else:
                        error_msg = status_result.get("error", "Unknown error")
                        if attempt < max_retries - 1:
                            console.print(
                                f"[yellow]Connection test failed (attempt {attempt + 1}): {error_msg}[/yellow]"
                            )
                            console.print(
                                f"[yellow]   Retrying in {retry_delay} seconds...[/yellow]"
                            )
                            time.sleep(retry_delay)
                        else:
                            console.print(
                                f"[red]Cloud connection failed after {max_retries} attempts: {error_msg}[/red]"
                            )
                            raise Exception(
                                f"Connection test failed: {error_msg}"
                            )

                except Exception as e:
                    if "timeout" in str(e).lower():
                        error_type = "Connection timeout"
                        suggestion = "Check your internet connection and endpoint URL"
                    elif "401" in str(e) or "unauthorized" in str(e).lower():
                        error_type = "Authentication failed"
                        suggestion = "Verify your LOBSTER_CLOUD_KEY is correct"
                    elif "404" in str(e) or "not found" in str(e).lower():
                        error_type = "Endpoint not found"
                        suggestion = "Check your LOBSTER_ENDPOINT URL"
                    else:
                        error_type = "Connection error"
                        suggestion = "Check network connectivity and service status"

                    if attempt < max_retries - 1:
                        console.print(
                            f"[yellow]{error_type} (attempt {attempt + 1}): {e}[/yellow]"
                        )
                        console.print(
                            f"[yellow]   Retrying in {retry_delay} seconds...[/yellow]"
                        )
                        time.sleep(retry_delay)
                    else:
                        console.print(
                            f"[red]{error_type} after {max_retries} attempts[/red]"
                        )
                        console.print(f"[red]   Error: {e}[/red]")
                        console.print(f"[yellow]   Suggestion: {suggestion}[/yellow]")
                        raise Exception(f"{error_type}: {e}")

        except ImportError:
            console.print(
                "[bold yellow]Lobster Cloud Not Available Locally[/bold yellow]"
            )
            console.print(
                "[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/cyan]"
            )
            console.print(
                "[white]You have a [bold blue]LOBSTER_CLOUD_KEY[/bold blue] set, but this is the open-source version.[/white]"
            )
            console.print("")
            console.print("[bold white]Get Lobster Cloud Access:[/bold white]")
            console.print("   Visit: [bold blue]https://cloud.lobster.ai[/bold blue]")
            console.print("   Email: [bold blue]cloud@omics-os.com[/bold blue]")
            console.print("")
            console.print(
                "[bold white]For now, using local mode with full functionality:[/bold white]"
            )
            console.print(
                "[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/cyan]"
            )

        except Exception as e:
            console.print(f"[red]Cloud connection error: {e}[/red]")
            console.print("[yellow]   Falling back to local mode...[/yellow]")

    # Use local client (existing code)
    import logging as _logging

    if debug:
        setup_logging(_logging.DEBUG)
    else:
        setup_logging(_logging.WARNING)

    workspace = resolve_workspace(explicit_path=workspace, create=True)

    from datetime import datetime

    actual_session_id = (
        session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    session_dir = workspace / ".lobster" / "sessions" / actual_session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    from lobster.core.data_manager_v2 import DataManagerV2

    data_manager = DataManagerV2(
        workspace_path=workspace, console=console, session_dir=session_dir
    )

    profile_timings_enabled = _resolve_profile_timings_flag(profile_timings)
    if profile_timings_enabled and hasattr(data_manager, "enable_timing"):
        data_manager.enable_timing(True)

    from lobster.utils import SimpleTerminalCallback, TerminalCallbackHandler

    callbacks = []

    if reasoning or verbose:
        callback = TerminalCallbackHandler(
            console=console,
            show_reasoning=reasoning,
            verbose=verbose,
            show_tools=verbose,
        )
        callbacks.append(callback)
    else:
        simple_callback = SimpleTerminalCallback(
            console=console,
            show_reasoning=False,
            minimal=True,
        )
        callbacks.append(simple_callback)

    try:
        local_client = AgentClient(
            data_manager=data_manager,
            workspace_path=workspace,
            session_id=session_id,
            enable_reasoning=reasoning,
            custom_callbacks=callbacks,
            provider_override=provider_override,
            model_override=model_override,
        )
    except ImportError as e:
        from rich.markup import escape as rich_escape

        error_msg = str(e)
        console.print("\n[red bold]Missing provider package[/red bold]")
        console.print(f"[red]  {rich_escape(error_msg)}[/red]\n")

        if "Install with:" in error_msg:
            cmd = error_msg.split("Install with:")[-1].strip()
            console.print("[yellow]How to fix:[/yellow]")
            console.print(f"  [white]Run:[/white] [dim]{rich_escape(cmd)}[/dim]")
        else:
            console.print("[yellow]How to fix:[/yellow]")
            console.print(
                "  [white]Run:[/white] [dim]lobster init[/dim] to configure your LLM provider"
            )
        console.print()
        raise typer.Exit(code=1)
    except ValueError as e:
        from rich.markup import escape as rich_escape

        error_msg = str(e)
        resolved_provider = provider_override or "configured provider"
        console.print(
            f"\n[red bold]Missing credentials for provider '{rich_escape(resolved_provider)}'[/red bold]"
        )
        console.print(f"[red]  {rich_escape(error_msg)}[/red]\n")
        console.print("[yellow]How to fix:[/yellow]")
        console.print(
            f"  1. [white]Add the key to the workspace .env file:[/white]\n"
            f"     [dim]{workspace}/.env[/dim]"
        )
        console.print(
            "  2. [white]Or add to global credentials (works everywhere):[/white]\n"
            "     [dim]lobster init --global[/dim]"
        )
        console.print(
            f"  3. [white]Or export in your shell:[/white]\n"
            f"     [dim]{error_msg.split('Set it with: ')[-1] if 'Set it with: ' in error_msg else 'export <KEY>=<value>'}[/dim]"
        )
        from lobster.core.config_resolver import _find_existing_configs

        found_configs = _find_existing_configs()
        if found_configs:
            closest = found_configs[0]
            project_root = closest.parent
            console.print(
                f"\n  [dim]Found an existing workspace at "
                f"[cyan]{rich_escape(str(closest))}[/cyan] --\n"
                f"     it may already have credentials configured.\n"
                f"     Try: [bold]export LOBSTER_WORKSPACE={rich_escape(str(project_root))}/.lobster_workspace[/bold]\n"
                f"     then re-run your command.[/dim]"
            )
        console.print()
        raise typer.Exit(code=1)

    local_client.profile_timings_enabled = profile_timings_enabled

    # Show graph visualization in debug mode
    if debug:
        try:
            if hasattr(local_client, "graph") and local_client.graph:
                mermaid_png = local_client.graph.get_graph().draw_mermaid_png()
                graph_file = workspace / "agent_graph.png"

                with open(graph_file, "wb") as f:
                    f.write(mermaid_png)

                console.print(
                    f"[green]Graph visualization saved to: {graph_file}[/green]"
                )
        except Exception as e:
            console.print(
                f"[yellow]Could not generate graph visualization: {e}[/yellow]"
            )

    return local_client


def init_client_with_animation(
    workspace: Optional[Path] = None,
    reasoning: bool = False,
    verbose: bool = False,
    debug: bool = False,
    profile_timings: Optional[bool] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    session_id: Optional[str] = None,
) -> "AgentClient":
    """
    Initialize client. Fast startup thanks to lazy imports.
    """
    get_console_manager()

    # Initialize client - lazy imports make this fast
    client = init_client(
        workspace,
        reasoning,
        verbose,
        debug,
        profile_timings,
        provider_override,
        model_override,
        session_id,
    )

    return client
