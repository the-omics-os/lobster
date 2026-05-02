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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from lobster.cli_internal.commands.output_adapter import (
    ConsoleOutputAdapter,
    OutputBlock,
    hint_block,
    kv_block,
)
from lobster.cli_internal.startup_diagnostics import (
    StartupDiagnosticError,
    raise_startup_diagnostic,
    render_startup_diagnostic_rich,
)
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


def _startup_console_print(*args, **kwargs) -> None:
    """Print startup notices only when a Rich terminal owns the screen."""
    if _go_tui_active:
        return
    console.print(*args, **kwargs)


def get_latest_session_id(workspace_path: Path) -> Optional[str]:
    """Return the most recent persisted session id for the workspace.

    Prefers real session directories under ``.lobster/sessions`` because those
    exist even when a chat session has not yet exported a JSON transcript.
    Falls back to legacy ``session_*.json`` files in the workspace root.
    """
    sessions_dir = workspace_path / ".lobster" / "sessions"
    if sessions_dir.exists():
        session_dirs = [path for path in sessions_dir.iterdir() if path.is_dir()]
        if session_dirs:
            session_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            return session_dirs[0].name

    session_files = list(workspace_path.glob("session_*.json"))
    if session_files:
        session_files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        stem = session_files[0].stem
        if stem.startswith("session_"):
            return stem.removeprefix("session_")

    return None


def resolve_session_continuation(
    workspace_path: Path, requested_session_id: Optional[str]
) -> tuple[Optional[Path], Optional[str], bool]:
    """Resolve transcript loading and runtime continuation for a session id.

    Returns:
        ``(session_file_to_load, session_id_for_client, found_existing_session)``

    Notes:
    - ``session_id_for_client`` is set for both loaded and newly created
      sessions so the checkpointer/session directory uses the intended id.
    - ``session_file_to_load`` is optional because some chat sessions persist
      only a session directory until a transcript JSON is exported.
    """
    if not requested_session_id:
        return None, None, False

    if requested_session_id == "latest":
        resolved_session_id = get_latest_session_id(workspace_path)
        if not resolved_session_id:
            return None, None, False
    else:
        resolved_session_id = requested_session_id

    session_file = workspace_path / f"session_{resolved_session_id}.json"
    if not session_file.exists():
        exact_session_file = workspace_path / f"{resolved_session_id}.json"
        if exact_session_file.exists():
            session_file = exact_session_file
        else:
            session_file = None

    session_dir = workspace_path / ".lobster" / "sessions" / resolved_session_id
    found_existing = session_dir.exists() or session_file is not None

    if not found_existing and requested_session_id != "latest":
        return None, resolved_session_id, False

    return session_file, resolved_session_id, found_existing


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


def build_session_blocks(client: "AgentClient") -> List[OutputBlock]:
    """Build structured blocks describing the current session."""
    from lobster.config.agent_defaults import get_current_profile

    status = client.get_status()
    current_mode = get_current_profile()

    rows = [
        ("Session ID", status["session_id"]),
        ("Mode", current_mode),
        ("Messages", status["message_count"]),
        ("Workspace", status["workspace"]),
        ("Data Loaded", status["has_data"]),
    ]

    if status["has_data"] and status["data_summary"]:
        summary = status["data_summary"]
        rows.append(("Data Shape", summary.get("shape", "N/A")))
        rows.append(("Memory Usage", summary.get("memory_usage", "N/A")))

    return [
        kv_block(rows, title="Session Status"),
        hint_block("Use `/workspace list` to inspect available datasets."),
    ]


def display_session(client: "AgentClient", output=None):
    """Display current session status via the configured output adapter."""
    if output is None:
        output = ConsoleOutputAdapter(console)
    output.render_blocks(build_session_blocks(client))


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
    interactive: bool = True,
) -> "AgentClient":
    """Initialize a client and render startup diagnostics via the Rich console."""
    import typer

    try:
        return init_client_or_raise_startup_diagnostic(
            workspace=workspace,
            reasoning=reasoning,
            verbose=verbose,
            debug=debug,
            profile_timings=profile_timings,
            provider_override=provider_override,
            model_override=model_override,
            session_id=session_id,
            interactive=interactive,
        )
    except StartupDiagnosticError as exc:
        render_startup_diagnostic_rich(console, exc.diagnostic)
        raise typer.Exit(code=exc.diagnostic.exit_code)


def _maybe_seed_workspace_provider_config(workspace_path: Path) -> None:
    """Copy the default workspace provider config into new workspaces when useful."""
    from lobster.config.global_config import GlobalProviderConfig
    from lobster.config.workspace_config import WorkspaceProviderConfig

    if WorkspaceProviderConfig.exists(workspace_path):
        return

    env_provider = os.environ.get("LOBSTER_LLM_PROVIDER")
    if env_provider or GlobalProviderConfig.exists():
        return

    default_workspace = Path.cwd() / ".lobster_workspace"
    if default_workspace == workspace_path:
        return
    if not WorkspaceProviderConfig.exists(default_workspace):
        return

    source_config = WorkspaceProviderConfig.load(default_workspace)
    if source_config.global_provider:
        source_config.save(workspace_path)
        logger.info(
            f"Auto-injected provider config into {workspace_path} "
            f"from {default_workspace}"
        )


def validate_startup_or_raise_startup_diagnostic(
    workspace: Optional[Path] = None,
    *,
    provider_override: Optional[str] = None,
) -> Path:
    """Validate provider startup prerequisites without binding a UI renderer."""
    from lobster.config.settings import settings
    from lobster.core.config_resolver import ConfigResolver, ConfigurationError
    from lobster.core.workspace import resolve_workspace

    workspace_path = resolve_workspace(explicit_path=workspace, create=True)
    _maybe_seed_workspace_provider_config(workspace_path)
    settings.reload_credentials(workspace_path)

    resolver = ConfigResolver.get_instance(workspace_path)
    try:
        provider_name, _ = resolver.resolve_provider(
            runtime_override=provider_override
        )
    except ConfigurationError as exc:
        raise_startup_diagnostic(
            exc,
            workspace=workspace_path,
            provider_override=provider_override,
        )

    from lobster.config.providers.registry import ProviderRegistry

    provider = ProviderRegistry.get(provider_name)
    if provider is not None:
        try:
            provider.check_dependencies()
        except ImportError as exc:
            raise_startup_diagnostic(
                exc,
                workspace=workspace_path,
                provider_override=provider_name,
            )

    return workspace_path


def _create_local_agent_client(
    *,
    data_manager: Any,
    workspace_path: Path,
    session_id: Optional[str],
    reasoning: bool,
    callbacks: list[Any],
    provider_override: Optional[str],
    model_override: Optional[str],
    interactive: bool = True,
) -> "AgentClient":
    from lobster.core.client import AgentClient

    return AgentClient(
        data_manager=data_manager,
        workspace_path=workspace_path,
        session_id=session_id,
        enable_reasoning=reasoning,
        custom_callbacks=callbacks,
        provider_override=provider_override,
        model_override=model_override,
        interactive=interactive,
    )


def init_client_or_raise_startup_diagnostic(
    workspace: Optional[Path] = None,
    reasoning: bool = False,
    verbose: bool = False,
    debug: bool = False,
    profile_timings: Optional[bool] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    session_id: Optional[str] = None,
    interactive: bool = True,
) -> "AgentClient":
    """Initialize either local or cloud client, raising structured diagnostics."""
    from lobster.core.config_resolver import ConfigurationError
    from lobster.ui import setup_logging

    workspace_path = validate_startup_or_raise_startup_diagnostic(
        workspace=workspace,
        provider_override=provider_override,
    )

    # Check for cloud API key
    cloud_key = os.environ.get("LOBSTER_CLOUD_KEY")
    cloud_endpoint = os.environ.get("LOBSTER_ENDPOINT")

    if cloud_key:
        _startup_console_print("[bold blue]Cloud API key detected...[/bold blue]")

        try:
            from lobster.lobster_cloud.client import CloudLobsterClient

            _startup_console_print(
                "[bold blue]   Initializing Lobster Cloud...[/bold blue]"
            )
            if cloud_endpoint:
                _startup_console_print(
                    f"[dim blue]   Endpoint: {cloud_endpoint}[/dim blue]"
                )

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
                        _startup_console_print(
                            "[bold green]Cloud connection established[/bold green]"
                        )
                        _startup_console_print(
                            f"[dim blue]   Status: {status_result.get('status', 'unknown')}[/dim blue]"
                        )
                        if status_result.get("version"):
                            _startup_console_print(
                                f"[dim blue]   Version: {status_result.get('version')}[/dim blue]"
                            )
                        return cloud_client
                    else:
                        error_msg = status_result.get("error", "Unknown error")
                        if attempt < max_retries - 1:
                            _startup_console_print(
                                f"[yellow]Connection test failed (attempt {attempt + 1}): {error_msg}[/yellow]"
                            )
                            _startup_console_print(
                                f"[yellow]   Retrying in {retry_delay} seconds...[/yellow]"
                            )
                            time.sleep(retry_delay)
                        else:
                            _startup_console_print(
                                f"[red]Cloud connection failed after {max_retries} attempts: {error_msg}[/red]"
                            )
                            raise Exception(f"Connection test failed: {error_msg}")

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
                        _startup_console_print(
                            f"[yellow]{error_type} (attempt {attempt + 1}): {e}[/yellow]"
                        )
                        _startup_console_print(
                            f"[yellow]   Retrying in {retry_delay} seconds...[/yellow]"
                        )
                        time.sleep(retry_delay)
                    else:
                        _startup_console_print(
                            f"[red]{error_type} after {max_retries} attempts[/red]"
                        )
                        _startup_console_print(f"[red]   Error: {e}[/red]")
                        _startup_console_print(
                            f"[yellow]   Suggestion: {suggestion}[/yellow]"
                        )
                        raise Exception(f"{error_type}: {e}")

        except ImportError:
            _startup_console_print(
                "[bold yellow]Lobster Cloud Not Available Locally[/bold yellow]"
            )
            _startup_console_print(
                "[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/cyan]"
            )
            _startup_console_print(
                "[white]You have a [bold blue]LOBSTER_CLOUD_KEY[/bold blue] set, but this is the open-source version.[/white]"
            )
            _startup_console_print("")
            _startup_console_print("[bold white]Get Lobster Cloud Access:[/bold white]")
            _startup_console_print(
                "   Visit: [bold blue]https://cloud.lobster.ai[/bold blue]"
            )
            _startup_console_print(
                "   Email: [bold blue]cloud@omics-os.com[/bold blue]"
            )
            _startup_console_print("")
            _startup_console_print(
                "[bold white]For now, using local mode with full functionality:[/bold white]"
            )
            _startup_console_print(
                "[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/cyan]"
            )

        except Exception as e:
            _startup_console_print(f"[red]Cloud connection error: {e}[/red]")
            _startup_console_print("[yellow]   Falling back to local mode...[/yellow]")

    # Use local client (existing code)
    import logging as _logging

    if debug:
        setup_logging(_logging.DEBUG)
    else:
        setup_logging(_logging.WARNING)

    workspace = workspace_path

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

    callbacks = []

    if not _go_tui_active:
        from lobster.utils import SimpleTerminalCallback, TerminalCallbackHandler

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
        local_client = _create_local_agent_client(
            data_manager=data_manager,
            workspace_path=workspace,
            session_id=actual_session_id,
            reasoning=reasoning,
            callbacks=callbacks,
            provider_override=provider_override,
            model_override=model_override,
            interactive=interactive,
        )
    except (ImportError, ValueError, ConfigurationError) as exc:
        raise_startup_diagnostic(
            exc,
            workspace=workspace,
            provider_override=provider_override,
        )

    local_client.profile_timings_enabled = profile_timings_enabled

    # Show graph visualization in debug mode
    if debug:
        try:
            if hasattr(local_client, "graph") and local_client.graph:
                mermaid_png = local_client.graph.get_graph().draw_mermaid_png()
                graph_file = workspace / "agent_graph.png"

                with open(graph_file, "wb") as f:
                    f.write(mermaid_png)

                _startup_console_print(
                    f"[green]Graph visualization saved to: {graph_file}[/green]"
                )
        except Exception as e:
            _startup_console_print(
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
