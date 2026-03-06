#!/usr/bin/env python3
"""
Modern, user-friendly CLI for the Multi-Agent Bioinformatics System.
Installable via pip or curl, with rich terminal interface.
"""

# Python 3.13 compatibility: Fix multiprocessing/tqdm file descriptor issue
# Must be done FIRST before any imports that might use multiprocessing
import sys

if sys.version_info >= (3, 13):
    import multiprocessing as _mp

    try:
        _mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass  # Already set

# Runtime Python version guard — catch unsupported versions early with a clear message
if sys.version_info < (3, 12):
    print(
        f"Error: Lobster AI requires Python 3.12+. "
        f"Found Python {sys.version_info.major}.{sys.version_info.minor}.\n"
        f"Fix: uv tool install --python 3.13 lobster-ai",
        file=sys.stderr,
    )
    raise SystemExit(1)

# Suppress harmless Pydantic v1 deprecation warning from langchain_core on Python 3.14+
import warnings

warnings.filterwarnings(
    "ignore", message="Core Pydantic V1 functionality", category=UserWarning
)

import html
import os
import random
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional

# Disable pandas Arrow-backed strings (pandas >=2.2 default)
# ArrowStringArray cannot be serialized to H5AD by anndata/h5py.
# Setting this BEFORE any pandas import ensures all .astype(str),
# categorical creation, and DataFrame operations use object dtype.
import pandas as pd

pd.options.future.infer_string = False

os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "900000"
import ast
import inspect
import json
import logging
import shutil
import time
import warnings
from functools import lru_cache
from typing import Any, Dict, Iterable, List

# =============================================================================
# Suppress noisy third-party warnings
# =============================================================================
# langchain-aws warns when both Anthropic API key and AWS creds are present
logging.getLogger("langchain_aws.utils").setLevel(logging.ERROR)
logging.getLogger("langchain_aws").setLevel(logging.ERROR)
# httpx and httpcore are verbose
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
# Suppress external API error logging that agents recover from
logging.getLogger("lobster.services.drug_discovery").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
# Suppress deprecation warnings from third-party libs
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic.*")

# Heavy imports moved to TYPE_CHECKING for lazy loading (saves ~5s startup time)
if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from lobster.core.client import AgentClient
    from lobster.utils import TerminalCallbackHandler, open_path

import typer
from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from lobster.cli_internal.commands import (
    ConsoleOutputAdapter,
    QueueFileTypeNotSupported,
    archive_queue,
    config_model_list,
    config_model_switch,
    config_provider_list,
    config_provider_switch,
    config_show,
    data_summary,
    export_data,
    file_read,
    metadata_clear,
    metadata_clear_all,
    metadata_clear_exports,
    metadata_exports,
    metadata_list,
    metadata_overview,
    metadata_publications,
    metadata_samples,
    metadata_workspace,
    modalities_list,
    modality_describe,
    pipeline_export,
    pipeline_info,
    pipeline_list,
    pipeline_run,
    plot_show,
    plots_list,
    queue_clear,
    queue_export,
    queue_import,
    queue_list,
    queue_load_file,
    show_queue_status,
    workspace_info,
    workspace_list,
    workspace_load,
    workspace_remove,
    workspace_status,
)
from lobster.cli_internal.commands.light.agent_commands import agents_app
from lobster.cli_internal.commands.light.scaffold_commands import scaffold_app
from lobster.cli_internal.commands.light.validate_commands import validate_app
from lobster.cli_internal.utils.path_resolution import (  # BUG FIX #6: Secure path resolution
    PathResolver,
)
from lobster.config import provider_setup

# Import component registry (lazy - don't trigger load_components at import time)
from lobster.core.component_registry import component_registry

# Extraction cache manager loaded lazily to avoid triggering all agent imports at startup
_ExtractionCacheManager = None
_extraction_cache_checked = False


def _get_extraction_cache_manager():
    """Lazy loader for ExtractionCacheManager (premium feature)."""
    from lobster.cli_internal.commands.heavy.session_infra import _get_extraction_cache_manager as _impl
    return _impl()


def _add_command_to_history(
    client: "AgentClient", command: str, summary: str, is_error: bool = False
) -> bool:
    """Add command execution to conversation history for AI context."""
    from lobster.cli_internal.commands.heavy.session_infra import _add_command_to_history as _impl
    return _impl(client, command, summary, is_error)


def _backup_command_to_file(
    client: "AgentClient", command: str, summary: str, is_error: bool, primary_logged: bool,
) -> bool:
    """Write command to backup file for audit trail and recovery."""
    from lobster.cli_internal.commands.heavy.session_infra import _backup_command_to_file as _impl
    return _impl(client, command, summary, is_error, primary_logged)


from lobster.core.queue_storage import queue_file_lock
from lobster.core.workspace import resolve_workspace

# Import new UI system
from lobster.ui import LobsterTheme, setup_logging
from lobster.ui.components import (
    create_file_tree,
    create_workspace_tree,
    get_multi_progress_manager,
    get_status_display,
)
from lobster.ui.console_manager import get_console_manager

# Note: SimpleTerminalCallback, TerminalCallbackHandler, open_path moved to lazy loading
from lobster.version import __version__

# Import prompt_toolkit for autocomplete functionality (optional dependency)
try:
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import (
        CompleteEvent,
        Completer,
        Completion,
        ThreadedCompleter,
    )
    from prompt_toolkit.document import Document
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style

    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

# Module logger
logger = logging.getLogger(__name__)

_COMMAND_HISTORY_LOCK = threading.Lock()


# ============================================================================
# Queue Command Exceptions
# ============================================================================
# Note: QueueFileTypeNotSupported exception now imported from shared module
# (lobster.cli_internal.commands)
# ============================================================================
# Progress Management
# ============================================================================


class NoOpProgress:
    """No-operation progress context manager for verbose/reasoning modes."""
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def add_task(self, *a, **kw): return None
    def update(self, *a, **kw): pass
    def remove_task(self, *a, **kw): pass


class CommandClient:
    """Minimal client providing only what shared command functions need.

    Skips AgentClient's expensive graph/LLM initialization. Provides
    data_manager, workspace_path, session_id, and publication_queue —
    the only attributes accessed by shared commands in cli_internal/commands/.

    When session_id is provided, loads provenance from that session's directory,
    enabling cross-session pipeline export.
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


def check_for_missing_slash_command(user_input: str) -> Optional[str]:
    """Check if user input matches a command without the leading slash."""
    if not user_input or user_input.startswith("/"):
        return None

    # Get the first word (potential command)
    first_word = user_input.split()[0].lower()

    # Check against known commands (without the slash)
    available_commands = extract_available_commands()
    for cmd in available_commands.keys():
        cmd_without_slash = cmd[1:]  # Remove the leading slash
        if first_word == cmd_without_slash:
            return cmd

    return None


def extract_available_commands() -> Dict[str, str]:
    from lobster.cli_internal.commands.heavy.slash_commands import extract_available_commands as _impl
    return _impl()



def change_mode(new_mode: str, current_client: "AgentClient") -> "AgentClient":
    from lobster.cli_internal.commands.heavy.slash_commands import change_mode as _impl
    return _impl(new_mode, current_client)




# =============================================================================
# Agent Selection Helper Functions (CLI-01, CLI-02, CONF-07, CONF-08)
# =============================================================================


# =============================================================================
# Available Agent Packages (static registry for init flow)
# =============================================================================
# These are the official lobster-* packages that can be installed.
# Format: (package_name, description, agents_provided, published_on_pypi)


# Initialize Rich console with orange theming and Typer app
console_manager = get_console_manager()
console = console_manager.console

app = typer.Typer(
    name="lobster",
    help="🦞 Lobster by Omics-OS - Multi-Agent Bioinformatics Analysis System",
    add_completion=True,
    rich_markup_mode="rich",
)

# Create a subcommand for configuration management
config_app = typer.Typer(
    name="config",
    help="Configuration management for Lobster agents",
    invoke_without_command=True,
)
app.add_typer(config_app, name="config")


# Config callback to run config_show when no subcommand is provided
@config_app.callback(invoke_without_command=True)
def config_callback(
    ctx: typer.Context,
    workspace: Optional[str] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory to use",
    ),
):
    """Display current configuration including agent composition.

    Run 'lobster config' to see configuration summary, or use subcommands
    like 'lobster config show', 'lobster config list-models', etc.
    """
    # Only run if no subcommand was invoked
    if ctx.invoked_subcommand is None:
        from lobster.core.client import AgentClient
        from lobster.core.workspace import resolve_workspace

        # Resolve workspace path
        workspace_path = resolve_workspace(workspace, create=False)

        # Create minimal client for config access
        try:
            client = AgentClient(workspace_path=str(workspace_path))
            output = ConsoleOutputAdapter(console)

            # Use unified config_show implementation from config_commands.py
            config_show(client, output)
        except Exception as e:
            console.print(f"[red]Error displaying configuration: {str(e)}[/red]")
            console.print("[dim]Run 'lobster init' if not yet configured[/dim]")
            raise typer.Exit(1)


# Register agents subcommand group
app.add_typer(agents_app, name="agents")

# Register scaffold subcommand group
app.add_typer(scaffold_app, name="scaffold")

# Register validate-plugin command
app.add_typer(validate_app, name="validate-plugin")


# App callback to show help when no subcommand is provided
@app.callback(invoke_without_command=True)
def default_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit",
        is_flag=True,
    ),
):
    """
    Show friendly help guide when lobster is invoked without subcommands.
    Also checks for updates on every invocation.
    """
    # Handle --version flag (fast path - no heavy imports)
    if version:
        console.print(f"lobster version {__version__}")
        raise typer.Exit()

    # Check for updates (non-blocking, cached, fails silently if offline)
    from lobster.config.version_check import maybe_show_update_notification

    maybe_show_update_notification(console)

    # If no subcommand was invoked, show the default help
    if ctx.invoked_subcommand is None:
        show_default_help()
        raise typer.Exit()


# Global client instance
client: Optional["AgentClient"] = None

# Global current directory tracking
current_directory = Path.cwd()

PROFILE_TIMINGS_ENV = "LOBSTER_PROFILE_TIMINGS"


def _str_to_bool(value: Optional[str]) -> Optional[bool]:
    from lobster.cli_internal.commands.heavy.session_infra import _str_to_bool as _impl
    return _impl(value)


def _resolve_profile_timings_flag(cli_flag: Optional[bool]) -> bool:
    from lobster.cli_internal.commands.heavy.session_infra import _resolve_profile_timings_flag as _impl
    return _impl(cli_flag)


def _collect_profile_timings(
    client: "AgentClient", clear: bool = True
) -> Dict[str, Dict[str, float]]:
    from lobster.cli_internal.commands.heavy.session_infra import _collect_profile_timings as _impl
    return _impl(client, clear)


def _maybe_print_timings(client: "AgentClient", context: str) -> None:
    from lobster.cli_internal.commands.heavy.session_infra import _maybe_print_timings as _impl
    _impl(client, context)


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
    from lobster.cli_internal.commands.heavy.session_infra import init_client as _init_client_impl

    global client
    client = _init_client_impl(
        workspace, reasoning, verbose, debug, profile_timings,
        provider_override, model_override, session_id,
    )
    return client




def get_user_input_with_editing(prompt_text: str, client=None) -> str:
    from lobster.cli_internal.commands.heavy.slash_commands import get_user_input_with_editing as _impl
    return _impl(prompt_text, client)



def execute_shell_command(command: str) -> bool:
    from lobster.cli_internal.commands.heavy.slash_commands import execute_shell_command as _impl
    return _impl(command)



def get_current_agent_name() -> str:
    """Get the current active agent name for display."""
    from lobster.utils import TerminalCallbackHandler

    global client
    if client and hasattr(client, "callbacks") and client.callbacks:
        for callback in client.callbacks:
            if isinstance(callback, TerminalCallbackHandler):
                if hasattr(callback, "current_agent") and callback.current_agent:
                    # Format the agent name properly
                    agent_name = callback.current_agent.replace("_", " ").title()
                    return f"🦞 {agent_name}"
                # Check if there are any recent events that might indicate the active agent
                elif hasattr(callback, "events") and callback.events:
                    # Get the most recent agent from events
                    for event in reversed(callback.events):
                        if (
                            event.agent_name
                            and event.agent_name != "system"
                            and event.agent_name != "unknown"
                        ):
                            agent_name = event.agent_name.replace("_", " ").title()
                            return f"🦞 {agent_name}"
                break
    return "🦞 Lobster"


def _dna_helix_animation(width: int, duration: float = 0.7):
    """DNA sequence animation with colorful bases."""
    from lobster.cli_internal.commands.heavy.animations import _dna_helix_animation as _impl
    _impl(width, duration)


def display_welcome():
    """Display DNA sequence animation as bioinformatics-themed startup visualization."""
    from lobster.cli_internal.commands.heavy.animations import display_welcome as _impl
    _impl()


def _dna_agent_loading_phase(
    width: int, agent_names: List[str], ready_queue=None, timeout: float = 10.0
):
    """DNA-themed agent loading animation showing real-time progress."""
    from lobster.cli_internal.commands.heavy.animations import _dna_agent_loading_phase as _impl
    _impl(width, agent_names, ready_queue, timeout)


def _dna_exit_animation(width: int, duration: float = 0.5):
    """DNA exit animation - reverse of startup animation."""
    from lobster.cli_internal.commands.heavy.animations import _dna_exit_animation as _impl
    _impl(width, duration)


def display_goodbye():
    """Display DNA exit animation as bioinformatics-themed farewell visualization."""
    from lobster.cli_internal.commands.heavy.animations import display_goodbye as _impl
    _impl()


def show_default_help():
    from lobster.cli_internal.commands.heavy.slash_commands import show_default_help as _impl
    _impl()


def _show_workspace_prompt(client):
    from lobster.cli_internal.commands.heavy.slash_commands import _show_workspace_prompt as _impl
    _impl(client)



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
    """Initialize client. Fast startup thanks to lazy imports."""
    from lobster.cli_internal.commands.heavy.session_infra import init_client_with_animation as _impl
    return _impl(workspace, reasoning, verbose, debug, profile_timings,
                 provider_override, model_override, session_id)


@app.command()
def config_test(
    output_json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output results as JSON (machine-readable)",
    ),
):
    """Test API connectivity and validate configuration."""
    from lobster.cli_internal.commands.light.config_commands import config_test_impl
    config_test_impl(output_json=output_json)


@app.command()
def status():
    """Display subscription tier, installed packages, and available agents."""
    from lobster.cli_internal.commands.heavy.display_helpers import _display_status_info

    _display_status_info()


@app.command(name="metadata")
def metadata_command(
    subcommand: Optional[str] = typer.Argument(
        None,
        help="Subcommand: overview, publications, samples, workspace, exports, list, clear",
    ),
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory. Default: ./.lobster_workspace",
    ),
    status_filter: Optional[str] = typer.Option(
        None,
        "--status",
        help="Status filter for publications subcommand",
    ),
):
    """
    Display metadata overview and statistics.

    Subcommands:
      - (none): Smart overview with key stats & next steps
      - publications: Publication queue status breakdown
      - samples: Sample statistics & disease coverage
      - workspace: File inventory across all locations
      - exports: Export files with usage hints
      - list: Legacy detailed list
      - clear: Clear metadata (with exports/all options)

    Examples:
      lobster metadata                           # Smart overview
      lobster metadata publications              # Publication queue status
      lobster metadata publications --status=handoff_ready
      lobster metadata samples                   # Sample statistics
      lobster metadata exports                   # Export files
    """
    from lobster.cli_internal.commands.heavy.slash_commands import metadata_command_impl
    metadata_command_impl(subcommand=subcommand, workspace=workspace, status_filter=status_filter)




@app.command()
def activate(
    access_code: str = typer.Argument(
        ..., help="Premium activation code from Omics-OS"
    ),
    server_url: Optional[str] = typer.Option(
        None,
        "--server",
        help="License server URL (defaults to https://licenses.omics-os.com)",
    ),
):
    """
    Activate a premium license using an access code.

    This command contacts the Omics-OS license server to validate your
    access code and activate premium features on this machine.

    Examples:
      lobster activate ABC123-XYZ789
      lobster activate ABC123-XYZ789 --server https://custom.server.com
    """
    from lobster.cli_internal.commands.heavy.slash_commands import activate_impl
    activate_impl(access_code=access_code, server_url=server_url)




@app.command()
def deactivate():
    """
    Deactivate the current premium license.

    This removes the local license file and reverts to the free tier.
    Your license can be re-activated on another machine or re-used later.
    """
    from lobster.cli_internal.commands.heavy.slash_commands import deactivate_impl
    deactivate_impl()




@app.command()
def purge(
    scope: str = typer.Option(
        "all",
        "--scope",
        "-s",
        help="What to purge: 'global' (~/.lobster, ~/.config/lobster), 'workspace' (current), or 'all'",
    ),
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory to purge (default: current workspace)",
    ),
    keep_license: bool = typer.Option(
        False,
        "--keep-license",
        help="Preserve license file (~/.lobster/license.json)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show what would be deleted without actually deleting",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt",
    ),
):
    """
    Remove all Lobster AI files from your system.

    This command removes configuration, cache, and workspace files:

    \b
    GLOBAL FILES (~/.lobster/, ~/.config/lobster/):
      • license.json - Premium license entitlement
      • providers.json - LLM provider configuration
      • credentials.env - API keys (securely stored)
      • lobster_history - Command history
      • version_check_cache - Version check cache

    \b
    WORKSPACE FILES (.lobster_workspace/):
      • data/ - Downloaded and processed data files
      • cache/ - Cached API responses and metadata
      • plots/ - Generated visualizations
      • sessions - Conversation history

    \b
    SAFETY:
      Only directories verified to contain Lobster AI files will be removed.
      Other software named "lobster" will NOT be affected.

    \b
    EXAMPLES:
      lobster purge --dry-run          # Preview what would be deleted
      lobster purge --scope global     # Remove only global config files
      lobster purge --scope workspace  # Remove only current workspace
      lobster purge --keep-license     # Preserve your license file
      lobster purge --force            # Skip confirmation prompt
    """
    from lobster.cli_internal.commands import purge as purge_cmd
    from lobster.cli_internal.commands.output_adapter import ConsoleOutputAdapter

    console.print()
    console.print(
        Panel.fit(
            f"[bold {LobsterTheme.PRIMARY_ORANGE}]🧹 Lobster AI Purge[/bold {LobsterTheme.PRIMARY_ORANGE}]",
            border_style=LobsterTheme.PRIMARY_ORANGE,
            padding=(0, 2),
        )
    )

    output = ConsoleOutputAdapter(console)
    result = purge_cmd(
        output=output,
        scope=scope,
        workspace_path=workspace,
        keep_license=keep_license,
        dry_run=dry_run,
        force=force,
    )

    if result is None:
        raise typer.Exit(1)



@app.command()
def init(
    global_config: bool = typer.Option(
        False,
        "--global",
        "-g",
        help="Save configuration globally (~/.config/lobster/) for all workspaces",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing configuration"
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Non-interactive mode for CI/CD (requires API key flags)",
    ),
    anthropic_key: Optional[str] = typer.Option(
        None, "--anthropic-key", help="Claude API key (non-interactive mode)"
    ),
    bedrock_access_key: Optional[str] = typer.Option(
        None,
        "--bedrock-access-key",
        help="AWS Bedrock access key (non-interactive mode)",
    ),
    bedrock_secret_key: Optional[str] = typer.Option(
        None,
        "--bedrock-secret-key",
        help="AWS Bedrock secret key (non-interactive mode)",
    ),
    use_ollama: bool = typer.Option(
        False,
        "--use-ollama",
        help="Use Ollama (local LLM) provider (non-interactive mode)",
    ),
    ollama_model: Optional[str] = typer.Option(
        None,
        "--ollama-model",
        help="Ollama model name (default: llama3:8b-instruct, non-interactive mode)",
    ),
    gemini_key: Optional[str] = typer.Option(
        None, "--gemini-key", help="Google API key (non-interactive mode)"
    ),
    openai_key: Optional[str] = typer.Option(
        None, "--openai-key", help="OpenAI API key (non-interactive mode)"
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Agent profile (development, production, performance, max). Only for Anthropic/Bedrock providers.",
    ),
    ncbi_key: Optional[str] = typer.Option(
        None, "--ncbi-key", help="NCBI API key (optional, non-interactive mode)"
    ),
    cloud_key: Optional[str] = typer.Option(
        None,
        "--cloud-key",
        help="Lobster Cloud API key (optional, enables premium tier)",
    ),
    cloud_endpoint: Optional[str] = typer.Option(
        None,
        "--cloud-endpoint",
        help="Custom cloud endpoint URL (optional)",
    ),
    skip_ssl_test: bool = typer.Option(
        False,
        "--skip-ssl-test",
        help="Skip SSL connectivity test during init",
    ),
    ssl_verify: Optional[bool] = typer.Option(
        None,
        "--ssl-verify/--no-ssl-verify",
        help="Enable/disable SSL verification (non-interactive mode)",
    ),
    ssl_cert_path: Optional[str] = typer.Option(
        None,
        "--ssl-cert-path",
        help="Path to custom CA certificate bundle",
    ),
    # Agent selection parameters (CLI-01, CLI-02, CONF-07, CONF-08)
    agents: Optional[str] = typer.Option(
        None,
        "--agents",
        help="Comma-separated list of agents to enable (non-interactive mode)",
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help="Agent preset name (scrna-basic, scrna-full, multiomics-full)",
    ),
    auto_agents: bool = typer.Option(
        False,
        "--auto-agents",
        help="Use LLM to suggest agents based on description (requires --agents-description)",
    ),
    agents_description: Optional[str] = typer.Option(
        None,
        "--agents-description",
        help="Workflow description for LLM agent suggestion (with --auto-agents)",
    ),
    skip_docling: bool = typer.Option(
        False,
        "--skip-docling",
        help="Skip docling installation prompt (non-interactive mode)",
    ),
    install_docling: bool = typer.Option(
        False,
        "--install-docling",
        help="Install docling for PDF intelligence (non-interactive mode)",
    ),
    install_vector_search: bool = typer.Option(
        False,
        "--install-vector-search",
        help="Install Smart Standardization dependencies (backend requires lobster-metadata dev package)",
    ),
    skip_extras: bool = typer.Option(
        False,
        "--skip-extras",
        help="Skip optional package installation (provider, TUI, extended-data)",
    ),
    ui_mode: str = typer.Option(
        "auto",
        "--ui",
        help="UI mode for interactive init: auto (Go TUI if available, else questionary, else classic), go (require Go TUI), classic (Rich prompts only)",
    ),
):
    """
    Initialize Lobster AI configuration.

    By default, creates workspace-specific configuration (.env + provider_config.json).
    Use --global to set user-wide defaults that apply to all workspaces.

    Interactive mode (default):
      Guides you through provider selection and API key entry with masked input.

    Non-interactive mode (CI/CD):
      Provide API keys via command-line flags for automated deployment.

    Global mode (--global):
      Saves provider settings to ~/.config/lobster/providers.json AND
      credentials to ~/.config/lobster/credentials.env (mode 0o600).
      External workspaces without their own .env will use these global credentials.

    Credential Resolution Priority:
      1. Workspace .env (highest priority)
      2. Global credentials.env (~/.config/lobster/)
      3. Environment variables

    Examples:
      lobster init                                    # Interactive setup (workspace)
      lobster init --global                           # Interactive setup (global defaults)
      lobster init --force                            # Reconfigure (overwrite existing)
      lobster init --global --force                   # Reconfigure global defaults
      lobster init --non-interactive \\
        --anthropic-key=sk-ant-xxx                   # CI/CD: Claude API (default: production)
      lobster init --non-interactive --global \\
        --use-ollama                                 # CI/CD: Set Ollama as global default
      lobster init --non-interactive \\
        --anthropic-key=sk-ant-xxx \\
        --profile=development                        # CI/CD: Claude with dev profile
      lobster init --non-interactive \\
        --bedrock-access-key=AKIA... \\
        --bedrock-secret-key=xxx \\
        --profile=performance                        # CI/CD: Bedrock with performance profile
      lobster init --non-interactive \\
        --use-ollama                                 # CI/CD: Ollama (profile not applicable)
      lobster init --non-interactive \\
        --use-ollama --ollama-model=mixtral:8x7b-instruct  # CI/CD: Ollama with custom model
      lobster init --non-interactive \\
        --anthropic-key=sk-ant-xxx \\
        --cloud-key=cloud_xxx                        # CI/CD: With cloud access
    """
    from lobster.cli_internal.commands.heavy.init_commands import init_impl
    init_impl(
        global_config=global_config, force=force, non_interactive=non_interactive,
        anthropic_key=anthropic_key, bedrock_access_key=bedrock_access_key,
        bedrock_secret_key=bedrock_secret_key, use_ollama=use_ollama,
        ollama_model=ollama_model, gemini_key=gemini_key, openai_key=openai_key,
        profile=profile, ncbi_key=ncbi_key, cloud_key=cloud_key,
        cloud_endpoint=cloud_endpoint, skip_ssl_test=skip_ssl_test,
        ssl_verify=ssl_verify, ssl_cert_path=ssl_cert_path, agents=agents,
        preset=preset, auto_agents=auto_agents, agents_description=agents_description,
        skip_docling=skip_docling, install_docling=install_docling,
        install_vector_search=install_vector_search, skip_extras=skip_extras,
        ui_mode=ui_mode,
    )


def _display_streaming_response(client, user_input: str, console: Console) -> Dict[str, Any]:
    """Display streaming response with live updates."""
    from lobster.cli_internal.commands.heavy.slash_commands import _display_streaming_response as _impl
    return _impl(client, user_input, console)

def _validate_chat_ui_mode(ui_mode: str) -> None:
    """Validate `lobster chat --ui` values early with CLI-friendly errors."""
    if ui_mode in {"auto", "go", "classic"}:
        return
    _raise_cli_error(f"Error: Invalid --ui value '{ui_mode}'. Must be one of: auto, classic, go")


def _raise_cli_error(message: str) -> None:
    """Emit a CLI error and exit."""
    import typer as _typer

    _typer.echo(message, err=True)
    raise typer.Exit(1)


def _ensure_go_chat_tty(ui_mode: str) -> None:
    """Reject explicit Go UI requests when stdin is not interactive."""
    if ui_mode == "go" and not os.isatty(0):
        _raise_cli_error("Error: Go TUI requires an interactive terminal (stdin is not a TTY)")


def _should_try_go_chat_ui(ui_mode: str, reasoning: bool, verbose: bool) -> bool:
    """Return True when chat should attempt the Go UI fast path."""
    return ui_mode in ("auto", "go") and not reasoning and not verbose and os.isatty(0)


def _resolve_go_chat_binary(ui_mode: str) -> Optional[str]:
    """Return Go TUI binary path, or raise for explicit `--ui go`."""
    from lobster.core.component_registry import get_install_command
    from lobster.cli_internal.go_tui_launcher import find_tui_binary_fast

    binary = find_tui_binary_fast()
    if binary or ui_mode != "go":
        return binary
    _raise_cli_error(
        f"Error: Go TUI binary not found. Install with: {get_install_command('lobster-ai-tui')}"
    )
    return None


def _launch_go_chat_binary(
    binary: str,
    *,
    workspace: Optional[Path],
    session_id: Optional[str],
    provider: Optional[str],
    model: Optional[str],
    debug: bool,
    profile_timings: Optional[bool],
    no_intro: bool,
    stream: bool,
) -> bool:
    """Launch the Go chat binary once the fast-path decision is made."""
    from lobster.cli_internal.go_tui_launcher import launch_go_tui_chat

    launch_go_tui_chat(
        binary,
        workspace=workspace,
        session_id=session_id,
        provider=provider,
        model=model,
        debug=debug,
        profile_timings=profile_timings,
        no_intro=no_intro,
        stream=stream,
    )
    return True


def _maybe_launch_go_chat_ui(
    *,
    ui_mode: str,
    workspace: Optional[Path],
    session_id: Optional[str],
    reasoning: bool,
    verbose: bool,
    debug: bool,
    profile_timings: Optional[bool],
    provider: Optional[str],
    model: Optional[str],
    no_intro: bool,
    stream: bool,
) -> bool:
    """Launch Go chat UI when requested and supported, else return False."""
    _ensure_go_chat_tty(ui_mode)
    if not _should_try_go_chat_ui(ui_mode, reasoning, verbose):
        return False

    binary = _resolve_go_chat_binary(ui_mode)
    if not binary:
        if ui_mode == "auto":
            import sys
            from lobster.core.component_registry import get_install_command

            print(
                "\033[33mNote:\033[0m Go TUI not found, using classic mode. "
                f"Install with: {get_install_command('lobster-ai-tui')}",
                file=sys.stderr,
            )
        return False

    try:
        return _launch_go_chat_binary(
            binary,
            workspace=workspace,
            session_id=session_id,
            provider=provider,
            model=model,
            debug=debug,
            profile_timings=profile_timings,
            no_intro=no_intro,
            stream=stream,
        )
    except Exception as exc:
        if ui_mode == "go":
            raise
        import sys
        print(
            f"\033[33mNote:\033[0m Go TUI failed ({exc}), falling back to classic mode.",
            file=sys.stderr,
        )
        return False

@app.command()
def chat(
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
        False, "--debug", "-d", help="Enable debug mode with enhanced error reporting"
    ),
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
    no_intro: bool = typer.Option(False, "--no-intro", is_flag=True,
        help="Disable the Go TUI inline intro animation (useful for automation and PTY capture)."),
    stream: bool = typer.Option(
        True,
        "--stream/--no-stream",
        help="Enable real-time text streaming (default: on)",
    ),
    ui_mode: str = typer.Option(
        "auto",
        "--ui",
        help="UI mode: auto (Go TUI if available), go (require Go TUI), classic (Rich terminal)",
    ),
    classic: bool = typer.Option(
        False,
        "--classic",
        is_flag=True,
        help="Shorthand for --ui classic (force Rich/Textual UI)",
    ),
):
    """
    Start an interactive chat session with the multi-agent system.

    Use --session-id to continue a previous conversation:
      lobster chat --session-id latest
      lobster chat --session-id session_20241208_150000

    Use --reasoning to see agent thinking process. Use --verbose for detailed tool output.
    Use --classic to force the Rich/Textual UI instead of the Go TUI.
    """
    if classic:
        ui_mode = "classic"
    _validate_chat_ui_mode(ui_mode)
    if _maybe_launch_go_chat_ui(
        ui_mode=ui_mode,
        workspace=workspace,
        session_id=session_id,
        reasoning=reasoning,
        verbose=verbose,
        debug=debug,
        profile_timings=profile_timings,
        provider=provider,
        model=model,
        no_intro=no_intro,
        stream=stream,
    ):
        return

    # Classic Rich terminal path (heavy imports happen here).
    from lobster.cli_internal.commands.heavy.chat_commands import chat_impl
    chat_impl(workspace=workspace, session_id=session_id, reasoning=reasoning,
        verbose=verbose, debug=debug, profile_timings=profile_timings,
        provider=provider, model=model, stream=stream)


@app.command(name="dashboard")
def dashboard_command(
    workspace: Optional[Path] = typer.Option(
        None, "--workspace", "-w",
        help="Workspace directory. Can also be set via LOBSTER_WORKSPACE env var. Default: ./.lobster_workspace",
    ),
):
    """Launch interactive dashboard (Textual UI)."""
    try:
        from lobster.ui.os_app import run_lobster_os
        run_lobster_os(workspace)
    except ImportError:
        from lobster.core.component_registry import get_install_command

        console.print(
            "[yellow]TUI mode requires the textual package.[/yellow]\n"
            f"Install with: [bold]{get_install_command('classic-tui', is_extra=True)}[/bold]"
        )
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to launch dashboard: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def query(
    question: str,
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w",
        help="Workspace directory. Can also be set via LOBSTER_WORKSPACE env var. Default: ./.lobster_workspace"),
    session_id: Optional[str] = typer.Option(None, "--session-id", "-s",
        help="Session ID to continue (use 'latest' for most recent session in workspace)"),
    reasoning: bool = typer.Option(False, "--reasoning", is_flag=True,
        help="Show agent reasoning and thinking process"),
    verbose: bool = typer.Option(False, "--verbose", "-v",
        help="Show detailed tool usage and agent activity"),
    debug: bool = typer.Option(False, "--debug", "-d",
        help="Enable debug mode with detailed logging"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    profile_timings: Optional[bool] = typer.Option(None, "--profile-timings/--no-profile-timings",
        help="Enable timing diagnostics for data manager operations"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p",
        help="LLM provider to use (bedrock, anthropic, ollama). Overrides auto-detection."),
    model: Optional[str] = typer.Option(None, "--model", "-m",
        help="Model to use (e.g., claude-4-sonnet, llama3:70b-instruct). Overrides configuration."),
    stream: bool = typer.Option(False, "--stream/--no-stream",
        help="Enable real-time text streaming (default: off for query mode)"),
    json_output: bool = typer.Option(False, "--json", "-j", is_flag=True,
        help="Output result as JSON for programmatic consumption. Suppresses all Rich formatting."),
):
    """Send a single query to the agent system."""
    from lobster.cli_internal.commands.heavy.query_commands import query_impl
    query_impl(
        question=question, workspace=workspace, session_id=session_id,
        reasoning=reasoning, verbose=verbose, debug=debug, output=output,
        profile_timings=profile_timings, provider=provider, model=model,
        stream=stream, json_output=json_output,
    )


def handle_command(command: str, client: "AgentClient"):
    """Handle slash commands with enhanced error handling."""
    from lobster.cli_internal.commands.heavy.slash_commands import handle_command as _impl
    _impl(command, client)




def _command_files(client, output) -> Optional[str]:
    from lobster.cli_internal.commands.heavy.slash_commands import _command_files as _impl
    return _impl(client, output)


def _command_save(client, output, force: bool = False) -> Optional[str]:
    from lobster.cli_internal.commands.heavy.slash_commands import _command_save as _impl
    return _impl(client, output, force)


def _command_restore(client, output, pattern: str = "recent") -> Optional[str]:
    from lobster.cli_internal.commands.heavy.slash_commands import _command_restore as _impl
    return _impl(client, output, pattern)


_UNKNOWN_COMMAND = object()  # Sentinel for unrecognized commands


def _dispatch_command(cmd_str: str, client, output):
    from lobster.cli_internal.commands.heavy.slash_commands import _dispatch_command as _impl
    return _impl(cmd_str, client, output)


@app.command(name="command")
def command_cmd(
    cmd: str = typer.Argument(
        help='Command to execute, e.g. "data", "workspace list", "files"'
    ),
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory. Default: ./.lobster_workspace",
    ),
    session_id: Optional[str] = typer.Option(
        None,
        "--session-id",
        "-s",
        help="Session ID to load. Use 'latest' for most recent. Defaults to last session if not specified.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        is_flag=True,
        help="Output as JSON for programmatic consumption",
    ),
):
    """
    Execute workspace commands without an LLM session.

    Supports --session-id for accessing session-specific data like provenance.
    Defaults to most recent session when available.

    \b
    Available commands:
      data                     Show current data summary
      files                    List workspace files
      tree                     Show directory tree view
      config                   Show current configuration
      config provider          List available LLM providers
      config provider <name>   Switch to specified provider
      status                   Show tier, packages, and agents
      workspace                Show workspace status
      workspace list           List datasets in workspace
      workspace load <file>    Load dataset into workspace
      workspace remove <name>  Remove a modality
      modalities               Show detailed modality info
      describe <name>          Inspect a specific modality
      metadata                 Smart metadata overview
      queue                    Show queue status
      plots                    List all generated plots
      pipeline export          Export reproducible notebook
      pipeline list            List available pipelines
      save                     Save current state
      export                   Export workspace to ZIP

    \b
    Examples:
      lobster command data
      lobster command config
      lobster command "pipeline export" --session-id exp1
      lobster command "workspace list" --json
    """
    from lobster.cli_internal.commands.heavy.slash_commands import command_cmd_impl
    command_cmd_impl(cmd=cmd, workspace=workspace, session_id=session_id, json_output=json_output)



@app.command(name="vector-search")
def vector_search_cmd(
    query_text: str = typer.Argument(..., help="Biomedical term to search"),
    top_k: Optional[int] = typer.Option(
        None, "--top-k", "-k", help="Results per collection (default: 5)"
    ),
    pretty: bool = typer.Option(
        True, "--pretty/--compact", help="Pretty-print or compact JSON"
    ),
):
    """
    Search all ontology collections via semantic vector similarity.

    Queries MONDO (diseases), UBERON (tissues), and Cell Ontology (cell types)
    and returns top N results per collection in JSON.

    Requires vector dependencies and currently a lobster-metadata development install
    for backend modules.

    \b
    Examples:
      lobster vector-search "glioblastoma"
      lobster vector-search "CD8+ T cell" --top-k 10
      lobster vector-search "liver" --compact | jq '.results.uberon'
    """
    from lobster.cli_internal.commands.heavy.slash_commands import vector_search_cmd_impl
    vector_search_cmd_impl(query_text=query_text, top_k=top_k, pretty=pretty)



@app.command()
def serve(
    port: int = typer.Option(8000, "--port", "-p"),
    host: str = typer.Option("0.0.0.0", "--host"),
):
    """
    Start the agent system as an API server (for React UI).
    """
    from lobster.cli_internal.commands.heavy.slash_commands import serve_impl
    serve_impl(port=port, host=host)



# Config subcommands
@config_app.command(name="list-models")
def list_models():
    """List all available model presets."""
    from lobster.cli_internal.commands.light.config_commands import list_models_impl
    list_models_impl()



@config_app.command(name="list-profiles")
def list_profiles():
    """List all available testing profiles."""
    from lobster.cli_internal.commands.light.config_commands import list_profiles_impl
    list_profiles_impl()



@config_app.command(name="show")
def config_show_subcommand(
    workspace: Optional[str] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory to use",
    ),
):
    """Display current configuration including agent composition."""
    from lobster.core.client import AgentClient
    workspace_path = resolve_workspace(workspace, create=False)
    try:
        client = AgentClient(workspace_path=str(workspace_path))
        output = ConsoleOutputAdapter(console)
        config_show(client, output)
    except Exception as e:
        console.print(f"[red]Error displaying configuration: {str(e)}[/red]")
        raise typer.Exit(1)



@config_app.command(name="show-config")
def show_config(
    workspace: Optional[Path] = typer.Option(
        None, "--workspace", "-w", help="Workspace path (default: current directory)"
    ),
    show_all: bool = typer.Option(
        False,
        "--show-all",
        help="Show all configured agents regardless of license tier",
    ),
):
    """Show current runtime configuration from ConfigResolver and ProviderRegistry."""
    from lobster.cli_internal.commands.light.config_commands import show_config_impl
    show_config_impl(workspace=workspace, show_all=show_all)



@config_app.command(name="test")
def test(
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        "-p",
        help="Profile to test (optional, auto-detects if not provided)",
    ),
    agent: Optional[str] = typer.Option(
        None, "--agent", "-a", help="Specific agent to test"
    ),
):
    """Test LLM provider connectivity and configuration.

    If no --profile is specified, auto-detects the currently configured provider
    and tests basic connectivity. With --profile, tests the full agent configuration.
    """
    from lobster.cli_internal.commands.light.config_commands import test_impl
    test_impl(profile=profile, agent=agent)



@config_app.command(name="create-custom")
def create_custom():
    """Interactive creation of custom configuration."""
    from lobster.cli_internal.commands.light.config_commands import create_custom_impl
    create_custom_impl()


@config_app.command(name="models")
def config_models(
    workspace: Optional[str] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace path (default: current directory)",
    ),
):
    """Interactive per-agent model configuration."""
    from lobster.cli_internal.commands.light.config_commands import config_models_impl
    config_models_impl(workspace=workspace)


@config_app.command(name="generate-env")
def generate_env():
    """Generate .env template with all available options."""
    from lobster.cli_internal.commands.light.config_commands import generate_env_impl
    generate_env_impl()



if __name__ == "__main__":
    app()
