"""
Slash command dispatch and interactive session helpers.

Extracted from cli.py (Plan 08-02) to isolate command dispatch logic.
Contains: slash command execution, autocomplete, shell commands, streaming,
user input handling, and workspace prompt display.
"""

import ast
import html
import inspect
import json
import logging
import os
import shutil
import time
import threading
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from lobster.ui import LobsterTheme
from lobster.ui.console_manager import get_console_manager
from lobster.ui.components.file_tree import create_file_tree, create_workspace_tree
from lobster.ui.components.status_display import get_status_display
from lobster.ui.components.multi_progress import get_multi_progress_manager
from lobster.version import __version__

if TYPE_CHECKING:
    from lobster.core.client import AgentClient

# Lazy imports from other heavy modules (these are the extracted companions)
from lobster.cli_internal.commands.heavy.session_infra import (
    _add_command_to_history,
    _backup_command_to_file,
    _maybe_print_timings,
    should_show_progress,
    display_session,
    LobsterClientAdapter,
    CloudAwareCache,
)
from lobster.cli_internal.commands.heavy.animations import (
    display_welcome,
    display_goodbye,
)
from lobster.cli_internal.commands.heavy.display_helpers import (
    _display_status_info,
)
from lobster.cli_internal.utils.path_resolution import PathResolver
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

# Console manager (shared with cli.py)
console_manager = get_console_manager()
console = console_manager.console

# Global current directory tracking (shared with cli.py)
current_directory = Path.cwd()


def _is_protocol_output(output: Optional["OutputAdapter"]) -> bool:
    """Return True when command output is routed to the Go TUI protocol."""
    return output is not None and output.__class__.__name__ == "ProtocolOutputAdapter"


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


@lru_cache(maxsize=1)
def extract_available_commands() -> Dict[str, str]:
    """Extract commands dynamically from _execute_command implementation."""

    # Static command definitions with descriptions (extracted from help text)
    command_descriptions = {
        "/help": "Show this help message",
        "/session": "Show current session status",
        "/status": "Show subscription tier, packages, and agents",
        "/input-features": "Show input capabilities and navigation features",
        "/dashboard": "Switch to interactive dashboard (Textual UI)",
        "/status-panel": "Show comprehensive system status panel",
        "/workspace-info": "Show detailed workspace overview",
        "/analysis-dash": "Show analysis monitoring dashboard",
        "/progress": "Show multi-task progress monitor",
        "/files": "List workspace files",
        "/tree": "Show directory tree view",
        "/data": "Show current data summary",
        "/metadata": "Show smart metadata overview with stats & next steps",
        "/metadata publications": "Publication queue status breakdown",
        "/metadata samples": "Aggregated sample statistics & disease coverage",
        "/metadata workspace": "Categorized file inventory across all locations",
        "/metadata exports": "Export files with categories & usage hints",
        "/metadata list": "Legacy detailed list (use overview instead)",
        "/metadata clear": "Clear metadata (memory + workspace/metadata/)",
        "/metadata clear exports": "Clear export files (workspace/exports/)",
        "/metadata clear all": "Clear ALL metadata (memory + disk + exports)",
        # Queue commands (temporary, intent-driven)
        "/queue": "Show queue status",
        "/queue load": "Load file into queue (supports .ris, more coming)",
        "/queue list": "List queued items",
        "/queue clear": "Clear publication queue",
        "/queue clear download": "Clear download queue",
        "/queue clear all": "Clear all queues (publication + download)",
        "/queue export": "Export queue to workspace for persistence",
        # Workspace commands (persistent)
        "/workspace": "Show workspace status and information",
        "/workspace list": "List available datasets in workspace",
        "/workspace load": "Load dataset or file into workspace",
        "/workspace remove": "Remove modality(ies) by index, pattern, or name",
        "/workspace save": "Save modality to workspace",
        "/workspace info": "Show dataset information",
        "/restore": "Restore previous session datasets",
        "/modalities": "Show detailed modality information",
        "/describe": "Show detailed information about a specific modality",
        "/plots": "List all generated plots",
        "/plot": "Open plots directory or specific plot",
        "/open": "Open file or folder in system default application",
        "/save": "Save current state to workspace (--force to re-save all)",
        "/read": "View file contents (inspection only)",
        "/export": "Export to ZIP (--no-png, --force to re-serialize all)",
        "/reset": "Reset conversation",
        "/config": "Show current configuration (provider, profile, config files)",
        "/config provider": "List available LLM providers",
        "/config provider <name>": "Switch to specified provider (runtime only)",
        "/config provider <name> --save": "Switch provider and persist to workspace",
        "/config model": "List available Ollama models",
        "/config model <name>": "Switch to specified Ollama model (runtime only)",
        "/config model <name> --save": "Switch Ollama model and persist to workspace",
        "/vector-search": "Search all ontology collections (semantic vector similarity)",
        "/clear": "Clear screen",
        "/exit": "Exit the chat",
    }

    # Try to extract dynamically as fallback, but use static definitions as primary
    try:
        # Get the source code of _execute_command
        source = inspect.getsource(_execute_command)
        tree = ast.parse(source)

        # Walk through AST to find command comparisons
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                # Look for patterns like: cmd == "/something"
                if (
                    isinstance(node.left, ast.Name)
                    and node.left.id == "cmd"
                    and len(node.ops) == 1
                    and isinstance(node.ops[0], ast.Eq)
                    and len(node.comparators) == 1
                    and isinstance(node.comparators[0], ast.Constant)
                ):
                    cmd = node.comparators[0].value
                    if isinstance(cmd, str) and cmd.startswith("/"):
                        # If not in our static definitions, add with generic description
                        if cmd not in command_descriptions:
                            command_descriptions[cmd] = f"Execute {cmd} command"

    except Exception:
        # Fallback to static definitions if AST parsing fails (silent)
        pass

    return command_descriptions


if PROMPT_TOOLKIT_AVAILABLE:

    class LobsterCommandCompleter(Completer):
        """Completer for Lobster slash commands with rich metadata."""

        def __init__(self):
            self.commands_cache = None
            self.cache_time = 0

        def get_completions(
            self, document: Document, complete_event: CompleteEvent
        ) -> Iterable[Completion]:
            """Generate hierarchical command completions.

            Implements depth-aware filtering to show only relevant commands:
            - Typing "/" → shows top-level commands (/help, /config, /queue)
            - Typing "/config " → shows subcommands (provider, model, show)
            - Typing "/config model " → shows model names

            This prevents overwhelming users with nested subcommands at the top level.
            """
            text_before_cursor = document.text_before_cursor.lstrip()

            # Only complete if we're typing a command (starts with /)
            if not text_before_cursor.startswith("/"):
                return

            # Get available commands (cached)
            commands = self._get_cached_commands()

            # Analyze input structure for hierarchical filtering
            has_trailing_space = text_before_cursor.endswith(" ")
            typed_text = (
                text_before_cursor.rstrip()
            )  # Remove trailing space for analysis
            typed_parts = typed_text.split() if typed_text else []
            current_depth = len(typed_parts)

            # Determine target depth for completions
            if has_trailing_space:
                # User typed "/config " (with space) → show next level subcommands
                # Target depth: current_depth + 1 (e.g., depth 1 → show depth 2)
                target_depth = current_depth + 1
                prefix_match = typed_text + " "  # Must start with "/config "
            else:
                # User typed "/conf" (no space) → complete current level
                # Target depth: current_depth (e.g., depth 1 → complete depth 1)
                target_depth = current_depth
                prefix_match = typed_text  # Must start with "/conf"

            # Generate completions with depth filtering
            for cmd, description in commands.items():
                # Calculate command depth (number of space-separated parts)
                cmd_depth = len(cmd.split())

                # Apply hierarchical filter: only show commands at target depth
                if cmd_depth != target_depth:
                    continue

                # Apply prefix filter: command must start with what user typed
                if not cmd.lower().startswith(prefix_match.lower()):
                    continue

                # Calculate completion text and cursor position
                if has_trailing_space:
                    # Show only the next part (e.g., "provider" from "/config provider")
                    completion_text = cmd[len(prefix_match) :]  # Extract remaining part
                    start_position = 0
                else:
                    # Show full command and replace from beginning of current word
                    completion_text = cmd
                    current_word = typed_parts[-1] if typed_parts else ""
                    start_position = -len(current_word)

                # Escape HTML special characters to prevent XML parsing errors
                escaped_cmd = html.escape(completion_text)
                escaped_desc = html.escape(description)

                yield Completion(
                    text=completion_text,
                    start_position=start_position,
                    display=HTML(f"<ansired>{escaped_cmd}</ansired>"),
                    display_meta=HTML(f"<dim>{escaped_desc}</dim>"),
                    style="class:completion.command",
                )

        def _get_cached_commands(self) -> Dict[str, str]:
            """Get commands with caching."""
            current_time = time.time()
            # Cache commands for 5 minutes
            if self.commands_cache is None or current_time - self.cache_time > 300:
                self.commands_cache = extract_available_commands()
                self.cache_time = current_time
            return self.commands_cache

    class LobsterFileCompleter(Completer):
        """Completer for workspace files with cloud-aware caching."""

        def __init__(self, client):
            self.adapter = LobsterClientAdapter(client)
            self.cache = CloudAwareCache(client)

        def get_completions(
            self, document: Document, complete_event: CompleteEvent
        ) -> Iterable[Completion]:
            """Generate file completions."""
            word = document.get_word_before_cursor()

            # Get files with caching
            try:
                files = self.cache.get_or_fetch(
                    "workspace_files",
                    lambda: self.adapter.get_workspace_files(),
                    "files",
                )
            except Exception as e:
                # Graceful fallback
                console = console_manager.get_console()
                console.print(f"[dim red]File completion error: {e}[/dim red]")
                files = []

            # Generate completions
            for file_info in files:
                file_name = file_info.get("name", "")
                if file_name.lower().startswith(word.lower()):
                    # Format file metadata
                    file_size = file_info.get("size", 0)
                    file_type = file_info.get("type", "unknown")

                    # Format size
                    if file_size < 1024:
                        size_str = f"{file_size}B"
                    elif file_size < 1024**2:
                        size_str = f"{file_size / 1024:.1f}KB"
                    elif file_size < 1024**3:
                        size_str = f"{file_size / 1024**2:.1f}MB"
                    else:
                        size_str = f"{file_size / 1024**3:.1f}GB"

                    meta = f"{file_type} • {size_str}"

                    yield Completion(
                        text=file_name,
                        start_position=-len(word),
                        display=HTML(f"<ansicyan>{file_name}</ansicyan>"),
                        display_meta=HTML(f"<dim>{meta}</dim>"),
                        style="class:completion.file",
                    )

    class LobsterContextualCompleter(Completer):
        """Smart contextual completer that switches between commands and files."""

        def __init__(self, client):
            self.client = client
            self.adapter = LobsterClientAdapter(client)
            self.command_completer = LobsterCommandCompleter()
            self.file_completer = LobsterFileCompleter(client)

            # Commands that expect file arguments
            self.file_commands = {"/read", "/plot", "/open"}

        def get_completions(
            self, document: Document, complete_event: CompleteEvent
        ) -> Iterable[Completion]:
            """Generate context-aware completions."""
            text = document.text_before_cursor.strip()

            if not text:
                # Empty input - show all commands
                yield from self.command_completer.get_completions(
                    document, complete_event
                )

            elif text.startswith("/") and " " not in text:
                # Command completion (typing a command)
                yield from self.command_completer.get_completions(
                    document, complete_event
                )

            elif text.startswith("/workspace load "):
                # Suggest available dataset names
                prefix = text.replace("/workspace load ", "")
                try:
                    if hasattr(self.client.data_manager, "available_datasets"):
                        for (
                            name,
                            info,
                        ) in self.client.data_manager.available_datasets.items():
                            if name.lower().startswith(prefix.lower()):
                                size_mb = info.get("size_mb", 0)
                                shape = info.get("shape", (0, 0))
                                meta = f"{size_mb:.1f}MB • {shape[0]}×{shape[1]}"
                                yield Completion(
                                    text=name,
                                    start_position=-len(prefix),
                                    display=HTML(f"<ansicyan>{name}</ansicyan>"),
                                    display_meta=HTML(f"<dim>{meta}</dim>"),
                                    style="class:completion.dataset",
                                )
                except Exception:
                    pass

            elif text.startswith("/workspace remove "):
                # Suggest currently loaded modality names for removal
                prefix = text.replace("/workspace remove ", "")
                try:
                    if hasattr(self.client.data_manager, "list_modalities"):
                        modalities = self.client.data_manager.list_modalities()
                        for name in modalities:
                            if name.lower().startswith(prefix.lower()):
                                # Get modality info for metadata
                                try:
                                    adata = self.client.data_manager.get_modality(name)
                                    meta = f"{adata.n_obs}×{adata.n_vars}"
                                    yield Completion(
                                        text=name,
                                        start_position=-len(prefix),
                                        display=HTML(f"<ansired>{name}</ansired>"),
                                        display_meta=HTML(f"<dim>{meta}</dim>"),
                                        style="class:completion.modality",
                                    )
                                except Exception:
                                    # Fallback without metadata if getting modality fails
                                    yield Completion(
                                        text=name,
                                        start_position=-len(prefix),
                                        display=HTML(f"<ansired>{name}</ansired>"),
                                        style="class:completion.modality",
                                    )
                except Exception:
                    pass

            elif text.startswith("/restore "):
                # Suggest restore patterns
                patterns = ["recent", "all", "*"]
                prefix = text.replace("/restore ", "")
                for pattern in patterns:
                    if pattern.startswith(prefix):
                        yield Completion(
                            text=pattern,
                            start_position=-len(prefix),
                            display=HTML(f"<ansiyellow>{pattern}</ansiyellow>"),
                            display_meta=HTML("<dim>restore pattern</dim>"),
                            style="class:completion.pattern",
                        )

            elif text.startswith("/describe "):
                # Suggest modality names for describe command
                prefix = text.replace("/describe ", "")
                try:
                    if hasattr(self.client.data_manager, "list_modalities"):
                        modalities = self.client.data_manager.list_modalities()
                        for modality_name in modalities:
                            if modality_name.lower().startswith(prefix.lower()):
                                # Get basic info about the modality if possible
                                try:
                                    adata = self.client.data_manager.get_modality(
                                        modality_name
                                    )
                                    meta = (
                                        f"{adata.n_obs:,} obs × {adata.n_vars:,} vars"
                                    )
                                except Exception:
                                    meta = "modality"

                                yield Completion(
                                    text=modality_name,
                                    start_position=-len(prefix),
                                    display=HTML(
                                        f"<ansicyan>{modality_name}</ansicyan>"
                                    ),
                                    display_meta=HTML(f"<dim>{meta}</dim>"),
                                    style="class:completion.modality",
                                )
                except Exception:
                    pass

            elif text == "/config" or text == "/config ":
                # Show /config subcommands
                subcommands = [
                    ("provider", "Set/show LLM provider"),
                    ("model", "Set/show model for current provider"),
                    ("show", "Show current configuration"),
                    ("profile", "Set/show agent profile"),
                ]
                prefix = "" if text == "/config" else ""
                for subcmd, desc in subcommands:
                    yield Completion(
                        text=" " + subcmd if text == "/config" else subcmd,
                        start_position=0,
                        display=HTML(f"<ansicyan>{subcmd}</ansicyan>"),
                        display_meta=HTML(f"<dim>{desc}</dim>"),
                        style="class:completion.subcommand",
                    )

            elif text == "/config model" or text == "/config provider":
                # User typed /config model or /config provider without trailing space
                # Show available options
                if text == "/config model":
                    try:
                        from lobster.config.llm_factory import LLMFactory
                        from lobster.config.providers import get_provider

                        # Get current provider (from client, not adapter)
                        current_provider = (
                            getattr(self.client, "provider_override", None)
                            or LLMFactory.get_current_provider()
                        )

                        # Get models from provider
                        provider_obj = get_provider(current_provider)
                        models = provider_obj.list_models() if provider_obj else []

                        for model in models:
                            meta = f"{model.display_name}"
                            if model.is_default:
                                meta += " (default)"

                            yield Completion(
                                text=" " + model.name,  # Add space before model name
                                start_position=0,
                                display=HTML(f"<ansiyellow>{model.name}</ansiyellow>"),
                                display_meta=HTML(f"<dim>{meta}</dim>"),
                                style="class:completion.model",
                            )
                    except Exception:
                        pass

                elif text == "/config provider":
                    # Dynamically fetch providers from ProviderRegistry
                    from lobster.config.providers import ProviderRegistry

                    try:
                        providers = ProviderRegistry.get_provider_names()
                    except Exception:
                        providers = [
                            "anthropic",
                            "bedrock",
                            "ollama",
                            "gemini",
                        ]  # Fallback

                    for provider in providers:
                        meta = f"Switch to {provider}"
                        yield Completion(
                            text=" " + provider,  # Add space before provider
                            start_position=0,
                            display=HTML(f"<ansiyellow>{provider}</ansiyellow>"),
                            display_meta=HTML(f"<dim>{meta}</dim>"),
                            style="class:completion.provider",
                        )

            elif text.startswith("/config model ") or text.startswith(
                "/config provider "
            ):
                # Tab completion for model names and provider names
                prefix = text.split(" ", 2)[-1]

                if text.startswith("/config model "):
                    # Provider-aware model completion
                    try:
                        from lobster.config.llm_factory import LLMFactory
                        from lobster.config.providers import get_provider

                        # Get current provider (from client, not adapter)
                        current_provider = (
                            getattr(self.client, "provider_override", None)
                            or LLMFactory.get_current_provider()
                        )

                        # Get models from provider
                        provider_obj = get_provider(current_provider)
                        models = provider_obj.list_models() if provider_obj else []

                        for model in models:
                            if model.name.lower().startswith(prefix.lower()):
                                meta = f"{model.display_name}"
                                if model.is_default:
                                    meta += " (default)"

                                yield Completion(
                                    text=model.name,
                                    start_position=-len(prefix),
                                    display=HTML(
                                        f"<ansiyellow>{model.name}</ansiyellow>"
                                    ),
                                    display_meta=HTML(f"<dim>{meta}</dim>"),
                                    style="class:completion.model",
                                )
                    except Exception:
                        # Silent fail for tab completion
                        pass

                elif text.startswith("/config provider "):
                    # Provider completion - dynamically fetch from ProviderRegistry
                    from lobster.config.providers import ProviderRegistry

                    try:
                        providers = ProviderRegistry.get_provider_names()
                    except Exception:
                        providers = [
                            "anthropic",
                            "bedrock",
                            "ollama",
                            "gemini",
                        ]  # Fallback

                    for provider in providers:
                        if provider.lower().startswith(prefix.lower()):
                            meta = f"Switch to {provider}"
                            yield Completion(
                                text=provider,
                                start_position=-len(prefix),
                                display=HTML(f"<ansiyellow>{provider}</ansiyellow>"),
                                display_meta=HTML(f"<dim>{meta}</dim>"),
                                style="class:completion.provider",
                            )

            elif any(text.startswith(cmd + " ") for cmd in self.file_commands):
                # File completion for file-accepting commands
                if self.adapter.can_read_files():
                    # Create a modified document that only includes the file part
                    # Find where the file argument starts
                    parts = text.split(" ", 1)
                    if len(parts) > 1:
                        file_part = parts[1]
                        # Create new document for file completion
                        from prompt_toolkit.document import Document

                        file_document = Document(
                            text=file_part, cursor_position=len(file_part)
                        )
                        yield from self.file_completer.get_completions(
                            file_document, complete_event
                        )

            elif text.startswith("/") and " " in text:
                # Other commands with arguments - could be extended for more specific completions
                pass



def change_mode(new_mode: str, current_client: "AgentClient") -> "AgentClient":
    """
    Change the operation mode and reinitialize client with the new configuration.

    Args:
        new_mode: The new mode/profile to switch to
        current_client: The current AgentClient instance

    Returns:
        Updated AgentClient instance
    """
    from lobster.cli_internal.commands.heavy.session_infra import init_client

    # Store current settings before reinitializing
    current_workspace = Path(current_client.workspace_path)
    current_reasoning = current_client.enable_reasoning

    # Persist the new profile to workspace config
    from lobster.config.workspace_config import WorkspaceProviderConfig

    ws_config = WorkspaceProviderConfig.load(current_workspace)
    ws_config.profile = new_mode
    ws_config.save(current_workspace)

    # Reinitialize the client with the new profile settings
    client = init_client(workspace=current_workspace, reasoning=current_reasoning)

    return client


def get_user_input_with_editing(prompt_text: str, client=None) -> str:
    """
    Get user input with advanced arrow key navigation, command history, and autocomplete.

    Features:
    - Left/Right arrows for cursor movement
    - Up/Down arrows for command history navigation
    - Ctrl+R for reverse search through history
    - Home/End for line navigation
    - Backspace/Delete for editing
    - Tab completion for commands and files
    - Cloud-aware file completion
    - Full command history persistence
    """
    try:
        # Create history file path for persistent command history
        history_file = None
        if PROMPT_TOOLKIT_AVAILABLE:
            try:
                history_dir = Path.home() / ".lobster"
                history_dir.mkdir(exist_ok=True)
                history_file = FileHistory(str(history_dir / "lobster_history"))
            except Exception:
                # If history file creation fails, continue without it
                history_file = None

        # Try to use prompt_toolkit with autocomplete if available
        if PROMPT_TOOLKIT_AVAILABLE and client:
            # Clean prompt text - remove Rich markup, keep just the symbol
            import re

            clean_prompt = re.sub(r"\[.*?\]", "", prompt_text).strip()
            if not clean_prompt:
                clean_prompt = "❯ "

            # Create client-aware completer
            main_completer = ThreadedCompleter(LobsterContextualCompleter(client))

            # Custom style to match Rich orange theme
            style = Style.from_dict(
                {
                    "prompt": "#e45c47 bold",
                    "completion-menu.completion": "bg:#2d2d2d #00d7ff",  # Cyan text (colorblind safe, high contrast)
                    "completion-menu.completion.current": "bg:#0087ff #ffffff bold",  # Blue bg (colorblind safe)
                    "completion-menu.meta": "bg:#2d2d2d #ffd700",  # Gold text (colorblind safe, high contrast)
                    "completion-menu.meta.current": "bg:#0087ff #ffffff",  # Blue bg (colorblind safe)
                    "completion.command": "#00d7ff",  # Cyan (high contrast, colorblind safe)
                    "completion.file": "#ffd700",  # Gold/Yellow (high contrast, colorblind safe)
                }
            )

            # Use prompt_toolkit with autocomplete - orange prompt
            user_input = prompt(
                HTML(f"<style fg='#e45c47' bold='true'>{clean_prompt}</style>"),
                completer=main_completer,
                complete_while_typing=True,
                # Disable mouse support so terminal scroll remains usable
                mouse_support=False,  # FIXME change this back to True if needed. I deactivated to allow scrolling
                style=style,
                complete_style="multi-column",
                history=history_file,
            )
            return user_input.strip()

        elif PROMPT_TOOLKIT_AVAILABLE:
            # Clean prompt text for non-autocomplete mode too
            import re

            clean_prompt = re.sub(r"\[.*?\]", "", prompt_text).strip()
            if not clean_prompt:
                clean_prompt = "❯ "

            # Use prompt_toolkit without autocomplete (no client provided)
            user_input = prompt(
                HTML(f"<style fg='#e45c47' bold='true'>{clean_prompt}</style>"),
                # Disable mouse support so terminal scroll remains usable
                mouse_support=False,  # FIXME change this back to True if needed. I deactivated to allow scrolling
                history=history_file,
            )
            return user_input.strip()

        else:
            # Graceful fallback to current Rich input
            user_input = console_manager.console.input(
                prompt=prompt_text, markup=True, emoji=True
            )
            return user_input.strip()

    except (KeyboardInterrupt, EOFError):
        # Handle Ctrl+C or Ctrl+D gracefully
        raise KeyboardInterrupt
    except Exception as e:
        # Fallback on any other error (e.g., prompt_toolkit issues)
        console.print(f"[dim red]Input error, using fallback: {e}[/dim red]")
        try:
            user_input = console_manager.console.input(
                prompt=prompt_text, markup=True, emoji=True
            )
            return user_input.strip()
        except (KeyboardInterrupt, EOFError):
            raise KeyboardInterrupt



def execute_shell_command(command: str) -> bool:
    """Execute shell commands and return True if successful."""
    global current_directory

    parts = command.strip().split()
    if not parts:
        return False

    cmd = parts[0].lower()

    try:
        if cmd == "cd":
            # Handle cd command
            if len(parts) == 1:
                # cd with no arguments goes to home
                new_dir = Path.home()
            else:
                target = " ".join(parts[1:])  # Handle paths with spaces
                if target == "~":
                    new_dir = Path.home()
                elif target.startswith("~/"):
                    new_dir = Path.home() / target[2:]
                else:
                    new_dir = (
                        current_directory / target
                        if not Path(target).is_absolute()
                        else Path(target)
                    )

                new_dir = new_dir.resolve()

            if new_dir.exists() and new_dir.is_dir():
                current_directory = new_dir
                os.chdir(current_directory)
                console.print(f"[grey74]{current_directory}[/grey74]")
                return True
            else:
                console.print(f"[red]cd: no such file or directory: {target}[/red]")
                return True  # We handled it, even if it failed

        elif cmd == "pwd":
            # Print working directory
            console.print(f"[grey74]{current_directory}[/grey74]")
            return True

        elif cmd == "ls":
            # List directory contents with structured output
            target_dir = current_directory
            show_path = ""
            if len(parts) > 1:
                target_path = parts[1]
                show_path = target_path
                if target_path.startswith("~/"):
                    target_dir = Path.home() / target_path[2:]
                else:
                    target_dir = (
                        current_directory / target_path
                        if not Path(target_path).is_absolute()
                        else Path(target_path)
                    )

            if target_dir.exists() and target_dir.is_dir():
                items = list(target_dir.iterdir())
                if not items:
                    console.print(
                        f"[grey50]Empty directory: {show_path or str(target_dir)}[/grey50]"
                    )
                    return True

                # Create a structured table for ls output
                table = Table(
                    title=f"📁 Directory Contents: {show_path or target_dir.name}",
                    box=box.SIMPLE,
                    border_style="blue",
                    show_header=True,
                    title_style="bold blue",
                )
                table.add_column("Name", style="white", min_width=20)
                table.add_column("Type", style="cyan", width=10)
                table.add_column("Size", style="grey74", width=10)
                table.add_column("Modified", style="grey50", width=16)

                # Sort: directories first, then files
                dirs = [item for item in items if item.is_dir()]
                files = [item for item in items if item.is_file()]
                sorted_items = sorted(dirs, key=lambda x: x.name.lower()) + sorted(
                    files, key=lambda x: x.name.lower()
                )

                for item in sorted_items:
                    try:
                        stat = item.stat()
                        if item.is_dir():
                            name = f"[bold blue]{item.name}/[/bold blue]"
                            type_str = "📁 DIR"
                            size_str = "-"
                        else:
                            name = f"[white]{item.name}[/white]"
                            type_str = "📄 FILE"
                            size = stat.st_size
                            if size < 1024:
                                size_str = f"{size}B"
                            elif size < 1024**2:
                                size_str = f"{size / 1024:.1f}KB"
                            elif size < 1024**3:
                                size_str = f"{size / 1024**2:.1f}MB"
                            else:
                                size_str = f"{size / 1024**3:.1f}GB"

                        # Format modification time
                        from datetime import datetime

                        mod_time = datetime.fromtimestamp(stat.st_mtime)
                        mod_str = mod_time.strftime("%Y-%m-%d %H:%M")

                        table.add_row(name, type_str, size_str, mod_str)
                    except (OSError, PermissionError):
                        # If we can't get stats, just show the name
                        name = (
                            f"[bold blue]{item.name}/[/bold blue]"
                            if item.is_dir()
                            else f"[white]{item.name}[/white]"
                        )
                        table.add_row(name, "?", "?", "?")

                console.print(table)
                console.print(
                    f"\n[grey50]Total: {len(dirs)} directories, {len(files)} files[/grey50]"
                )
                return True
            else:
                console.print(
                    f"[red]ls: cannot access '{parts[1] if len(parts) > 1 else target_dir}': No such file or directory[/red]"
                )
                return True

        elif cmd == "cat":
            # Enhanced cat command with syntax highlighting
            if len(parts) < 2:
                console.print("[red]cat: missing file argument[/red]")
                return True

            file_path = " ".join(parts[1:])  # Handle paths with spaces
            if not file_path.startswith("/") and not file_path.startswith("~/"):
                file_path = current_directory / file_path
            else:
                file_path = Path(file_path).expanduser()

            try:
                if file_path.exists() and file_path.is_file():
                    content = file_path.read_text(encoding="utf-8", errors="replace")

                    # Try to guess syntax from extension for highlighting

                    ext = file_path.suffix.lower()

                    # Map common extensions to syntax highlighting
                    language_map = {
                        ".py": "python",
                        ".js": "javascript",
                        ".ts": "typescript",
                        ".html": "html",
                        ".css": "css",
                        ".json": "json",
                        ".xml": "xml",
                        ".yaml": "yaml",
                        ".yml": "yaml",
                        ".sh": "bash",
                        ".bash": "bash",
                        ".zsh": "bash",
                        ".sql": "sql",
                        ".md": "markdown",
                        ".txt": "text",
                        ".log": "text",
                        ".conf": "text",
                        ".cfg": "text",
                    }

                    language = language_map.get(ext, "text")

                    if content.strip():
                        syntax = Syntax(
                            content, language, theme="monokai", line_numbers=True
                        )
                        console.print(
                            Panel(
                                syntax,
                                title=f"[bold blue]📄 {file_path.name}[/bold blue]",
                                border_style="blue",
                                box=box.ROUNDED,
                            )
                        )
                    else:
                        console.print(
                            f"[grey50]📄 {file_path.name} (empty file)[/grey50]"
                        )
                else:
                    console.print(
                        f"[red]cat: {file_path}: No such file or directory[/red]"
                    )
            except PermissionError:
                console.print(f"[red]cat: {file_path}: Permission denied[/red]")
            except UnicodeDecodeError:
                console.print(
                    f"[red]cat: {file_path}: Binary file (cannot display)[/red]"
                )
            except Exception as e:
                console.print(f"[red]cat: {file_path}: {e}[/red]")

            return True

        elif cmd == "open":
            # Handle open command to open files or folders
            if len(parts) < 2:
                console.print("[red]open: missing file or folder argument[/red]")
                return True

            file_or_folder = " ".join(parts[1:])  # Handle paths with spaces

            # Resolve path relative to current directory if not absolute
            if not file_or_folder.startswith("/") and not file_or_folder.startswith(
                "~/"
            ):
                target_path = current_directory / file_or_folder
            else:
                target_path = Path(file_or_folder).expanduser()

            if not target_path.exists():
                console.print(
                    f"[red]open: '{file_or_folder}': No such file or directory[/red]"
                )
                return True

            # Open file or folder using centralized system utility
            from lobster.utils import open_path
            success, message = open_path(target_path)

            if success:
                # Format success message with appropriate icon
                if target_path.is_dir():
                    console.print(f"[green]📁 {message}[/green]")
                else:
                    console.print(f"[green]📄 {message}[/green]")
            else:
                console.print(f"[red]open: {message}[/red]")

            return True

        elif cmd == "mkdir":
            # Create directory using pathlib (safe, no shell injection)
            if len(parts) < 2:
                console.print("[red]mkdir: missing operand[/red]")
                return True

            dir_path = " ".join(parts[1:])  # Handle paths with spaces
            target_dir = (
                current_directory / dir_path
                if not Path(dir_path).is_absolute()
                else Path(dir_path)
            )

            try:
                target_dir.mkdir(parents=True, exist_ok=False)
                console.print(f"[green]📁 Created directory: {parts[1]}[/green]")
            except FileExistsError:
                console.print(
                    f"[red]mkdir: cannot create directory '{parts[1]}': File exists[/red]"
                )
            except Exception as e:
                console.print(f"[red]mkdir: {e}[/red]")

            return True

        elif cmd == "touch":
            # Create file using pathlib (safe, no shell injection)
            if len(parts) < 2:
                console.print("[red]touch: missing file operand[/red]")
                return True

            file_path = " ".join(parts[1:])  # Handle paths with spaces
            target_file = (
                current_directory / file_path
                if not Path(file_path).is_absolute()
                else Path(file_path)
            )

            try:
                target_file.touch()
                console.print(f"[green]📄 Created file: {parts[1]}[/green]")
            except Exception as e:
                console.print(f"[red]touch: {e}[/red]")

            return True

        elif cmd == "cp":
            # Copy file using shutil (safe, no shell injection)
            if len(parts) < 3:
                console.print("[red]cp: missing file operand[/red]")
                return True

            src = parts[1]
            dst = parts[2]
            src_path = (
                current_directory / src if not Path(src).is_absolute() else Path(src)
            )
            dst_path = (
                current_directory / dst if not Path(dst).is_absolute() else Path(dst)
            )

            try:
                if src_path.is_dir():
                    shutil.copytree(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
                console.print(f"[green]📋 Copied: {parts[1]} → {parts[2]}[/green]")
            except Exception as e:
                console.print(f"[red]cp: {e}[/red]")

            return True

        elif cmd == "mv":
            # Move file using shutil (safe, no shell injection)
            if len(parts) < 3:
                console.print("[red]mv: missing file operand[/red]")
                return True

            src = parts[1]
            dst = parts[2]
            src_path = (
                current_directory / src if not Path(src).is_absolute() else Path(src)
            )
            dst_path = (
                current_directory / dst if not Path(dst).is_absolute() else Path(dst)
            )

            try:
                shutil.move(str(src_path), str(dst_path))
                console.print(f"[green]📦 Moved: {parts[1]} → {parts[2]}[/green]")
            except Exception as e:
                console.print(f"[red]mv: {e}[/red]")

            return True

        elif cmd == "rm":
            # Remove file using pathlib (safe, no shell injection)
            if len(parts) < 2:
                console.print("[red]rm: missing operand[/red]")
                return True

            file_path = " ".join(parts[1:])  # Handle paths with spaces
            target_path = (
                current_directory / file_path
                if not Path(file_path).is_absolute()
                else Path(file_path)
            )

            try:
                if target_path.is_dir():
                    # For directories, require explicit -r flag
                    if "-r" in parts or "-rf" in parts:
                        shutil.rmtree(target_path)
                        console.print(
                            f"[green]🗑️  Removed directory: {parts[1]}[/green]"
                        )
                    else:
                        console.print(
                            f"[red]rm: cannot remove '{parts[1]}': Is a directory (use -r to remove directories)[/red]"
                        )
                else:
                    target_path.unlink()
                    console.print(f"[green]🗑️  Removed: {parts[1]}[/green]")
            except FileNotFoundError:
                console.print(
                    f"[red]rm: cannot remove '{parts[1]}': No such file or directory[/red]"
                )
            except Exception as e:
                console.print(f"[red]rm: {e}[/red]")

            return True

        else:
            # Not a recognized shell command
            return False

    except Exception as e:
        console.print(f"[red]Error executing command: {e}[/red]")
        return True  # We handled it, even if it failed


def get_current_agent_name(client=None) -> str:
    """Get the current active agent name for display."""
    from lobster.utils import TerminalCallbackHandler
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


def show_default_help():
    """Display default help guide when lobster is run without subcommands."""
    # Create branded header
    header_text = LobsterTheme.create_title_text("LOBSTER by Omics-OS", "🦞")

    help_content = f"""[bold white]Multi-Agent Bioinformatics Analysis System v{__version__}[/bold white]

[bold {LobsterTheme.PRIMARY_ORANGE}]AVAILABLE COMMANDS:[/bold {LobsterTheme.PRIMARY_ORANGE}]

  [{LobsterTheme.PRIMARY_ORANGE}]lobster chat[/{LobsterTheme.PRIMARY_ORANGE}]      [grey50]-[/grey50] Start interactive chat session with AI agents
  [{LobsterTheme.PRIMARY_ORANGE}]lobster query[/{LobsterTheme.PRIMARY_ORANGE}]     [grey50]-[/grey50] Send a single analysis query
  [{LobsterTheme.PRIMARY_ORANGE}]lobster serve[/{LobsterTheme.PRIMARY_ORANGE}]     [grey50]-[/grey50] Start API server for web services
  [{LobsterTheme.PRIMARY_ORANGE}]lobster config[/{LobsterTheme.PRIMARY_ORANGE}]    [grey50]-[/grey50] Manage agent configuration

[bold {LobsterTheme.PRIMARY_ORANGE}]QUICK START:[/bold {LobsterTheme.PRIMARY_ORANGE}]

  [white]# Start interactive analysis with enhanced autocomplete[/white]
  [{LobsterTheme.PRIMARY_ORANGE}]lobster chat[/{LobsterTheme.PRIMARY_ORANGE}]

  [white]# Show agent reasoning during analysis[/white]
  [{LobsterTheme.PRIMARY_ORANGE}]lobster chat --reasoning[/{LobsterTheme.PRIMARY_ORANGE}]

  [white]# Send a single query and get results[/white]
  [{LobsterTheme.PRIMARY_ORANGE}]lobster query "Analyze my RNA-seq data"[/{LobsterTheme.PRIMARY_ORANGE}]

  [white]# Start API server on custom port[/white]
  [{LobsterTheme.PRIMARY_ORANGE}]lobster serve --port 8080[/{LobsterTheme.PRIMARY_ORANGE}]

[bold {LobsterTheme.PRIMARY_ORANGE}]CONFIGURATION:[/bold {LobsterTheme.PRIMARY_ORANGE}]

  [{LobsterTheme.PRIMARY_ORANGE}]lobster config list-models[/{LobsterTheme.PRIMARY_ORANGE}]    [grey50]-[/grey50] List available AI models
  [{LobsterTheme.PRIMARY_ORANGE}]lobster config list-profiles[/{LobsterTheme.PRIMARY_ORANGE}]  [grey50]-[/grey50] List testing profiles
  [{LobsterTheme.PRIMARY_ORANGE}]lobster config show-config[/{LobsterTheme.PRIMARY_ORANGE}]    [grey50]-[/grey50] Show current configuration

[bold {LobsterTheme.PRIMARY_ORANGE}]KEY FEATURES:[/bold {LobsterTheme.PRIMARY_ORANGE}]
• [white]Single-Cell & Bulk RNA-seq Analysis[/white]
• [white]Mass Spectrometry & Affinity Proteomics[/white]
• [white]GEO Dataset Access & Literature Mining[/white]
• [white]Interactive Visualizations & Reports[/white]
• [white]Natural Language Interface[/white]

[bold {LobsterTheme.PRIMARY_ORANGE}]HELP & DOCUMENTATION:[/bold {LobsterTheme.PRIMARY_ORANGE}]

  [{LobsterTheme.PRIMARY_ORANGE}]lobster --help[/{LobsterTheme.PRIMARY_ORANGE}]              [grey50]-[/grey50] Show detailed help
  [{LobsterTheme.PRIMARY_ORANGE}]lobster <command> --help[/{LobsterTheme.PRIMARY_ORANGE}]    [grey50]-[/grey50] Show help for specific command

[dim grey50]🌐 Website: https://omics-os.com | 📚 Docs: https://github.com/the-omics-os[/dim grey50]
[dim grey50]Powered by LangGraph | © 2025 Omics-OS[/dim grey50]"""

    # Create branded help panel
    help_panel = LobsterTheme.create_panel(help_content, title=str(header_text))

    console_manager.print(help_panel)



def _show_workspace_prompt(client):
    """Display minimal oh-my-zsh style status with system information."""
    try:
        import shutil

        import psutil

        from lobster.config.agent_registry import AGENT_REGISTRY, _ensure_plugins_loaded
        from lobster.core.license_manager import get_current_tier
        from lobster.tools.gpu_detector import GPUDetector

        # Ensure plugins are loaded before accessing AGENT_REGISTRY
        _ensure_plugins_loaded()

        tier = get_current_tier()
        tier_color = {
            "free": "green",
            "premium": "yellow",
            "enterprise": "magenta",
        }.get(tier, "white")

        # Count agents
        child_agent_names = set()
        for config in AGENT_REGISTRY.values():
            if config.child_agents:
                child_agent_names.update(config.child_agents)
        agent_count = sum(
            1
            for name, config in AGENT_REGISTRY.items()
            if config.supervisor_accessible is not False
            and name not in child_agent_names
        )

        # Get current provider (same approach as /config command)
        try:
            from lobster.core.config_resolver import ConfigResolver

            # Use ConfigResolver with workspace path (same as /config command)
            resolver = ConfigResolver(workspace_path=Path(client.workspace_path))
            current_provider, _ = resolver.resolve_provider(
                runtime_override=getattr(client, "provider_override", None)
            )

            # Provider icon mapping
            provider_icons = {
                "anthropic": "🔵",
                "bedrock": "🟠",
                "ollama": "🦙",
                "gemini": "🔷",
            }
            provider_icon = provider_icons.get(current_provider, "⚪")
            provider_display = f"{provider_icon} {current_provider}"
        except Exception:
            # Fallback to unknown if resolution fails
            provider_display = "⚪ unknown"

        # Get system information
        # RAM
        ram_gb = round(psutil.virtual_memory().total / (1024**3))

        # GPU
        hw_rec = GPUDetector.get_hardware_recommendation()
        gpu_label = hw_rec["device"].upper()  # CUDA, MPS, or CPU

        # Disk space (workspace)
        try:
            workspace_path = (
                client.workspace_path
                if hasattr(client, "workspace_path")
                else Path.cwd()
            )
            disk_usage = shutil.disk_usage(workspace_path)
            disk_free_gb = round(disk_usage.free / (1024**3))
        except Exception:
            disk_free_gb = "?"

        # Check semantic search availability
        try:
            import chromadb  # noqa: F401

            has_semantic = True
        except ImportError:
            has_semantic = False
        semantic_badge = " [dim]│ semantic[/]" if has_semantic else ""

        # Line 1: Application status
        console.print(
            f"  [{tier_color}]●[/] lobster v{__version__} [{tier_color}]{tier}[/] [dim]│[/] {agent_count} agents{semantic_badge} [dim]│ local │[/] [dim italic]/help[/]"
        )

        # Line 2-4: System resources (stacked for clarity)
        console.print(f"[dim]  └─ Compute:[/] {ram_gb}GB RAM [dim]│[/] {gpu_label}")
        console.print(
            f"[dim]     Storage:[/] {disk_free_gb}GB free [dim](workspace)[/]"
        )
        console.print(f"[dim]     Provider:[/] {provider_display}")
        console.print()
    except Exception as e:
        # Fallback to basic prompt if system info fails
        console.print(f"[dim red]Error loading system info: {e}[/dim red]")
        console.print(f"  [green]●[/] lobster v{__version__}")
        console.print()


def _display_streaming_response(
    client,
    user_input: str,
    console: Console,
) -> Dict[str, Any]:
    """
    Display streaming response with live updates.

    Returns the final result dict for consistency with non-streaming path.
    """
    accumulated_text = ""
    last_agent = None
    final_result = {"success": False, "response": ""}

    try:
        # Show immediate feedback so the user knows work has started
        initial_status = Text()
        initial_status.append("◀ Lobster", style="dim")
        initial_status.append("  Thinking…", style="dim italic")

        with Live(
            initial_status, console=console, refresh_per_second=10, transient=True
        ) as live:
            for event in client.query(user_input, stream=True):
                event_type = event.get("type")

                if event_type == "content_delta":
                    accumulated_text += event.get("delta", "")
                    # Build display with agent indicator + text
                    display = Text()
                    if last_agent:
                        agent_display = last_agent.replace("_", " ").title()
                        display.append(f"◀ {agent_display}\n", style="dim")
                    display.append(accumulated_text)
                    live.update(display)

                elif event_type == "agent_change":
                    agent = event.get("agent", "")
                    if event.get("status") == "working":
                        last_agent = agent
                        # Update status indicator before first content arrives
                        if not accumulated_text:
                            agent_display = agent.replace("_", " ").title()
                            status = Text()
                            status.append(f"◀ {agent_display}", style="dim")
                            status.append("  Working…", style="dim italic")
                            live.update(status)

                elif event_type == "complete":
                    # Use accumulated text, or fallback to response from event
                    response_text = accumulated_text or event.get("response", "")
                    final_result = {
                        "success": True,
                        "response": response_text,
                        "last_agent": event.get("last_agent"),
                        "token_usage": event.get("token_usage"),
                        "plots": [],  # Plots handled separately
                    }

                elif event_type == "error":
                    final_result = {
                        "success": False,
                        "error": event.get("error", "Unknown error"),
                    }
                    break

        return final_result

    except KeyboardInterrupt:
        return {
            "success": False,
            "error": "Interrupted by user",
            "response": accumulated_text,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_command(command: str, client: "AgentClient"):
    """Handle slash commands with enhanced error handling."""
    cmd = command.lower().strip()

    try:
        # Execute command and capture summary for history
        command_summary = _execute_command(cmd, client, original_command=command.strip())

        # Add to conversation history if summary provided
        if command_summary:
            _add_command_to_history(client, command, command_summary)

        _maybe_print_timings(client, f"Command {cmd}")

    except Exception as e:
        # Enhanced command error handling
        error_message = str(e)
        error_type = type(e).__name__

        # Log command failure to history
        error_summary = f"Failed: {error_type}: {error_message[:100]}"
        _add_command_to_history(client, command, error_summary, is_error=True)

        # Command-specific error suggestions
        if cmd.startswith("/read"):
            suggestion = "Check if the file exists and you have read permissions"
        elif cmd.startswith("/plot"):
            suggestion = "Ensure plots have been generated and saved to workspace"
        elif cmd in ["/files", "/data", "/metadata"]:
            suggestion = "Check if workspace is properly initialized"
        else:
            suggestion = "Check command syntax with /help"

        console_manager.print_error_panel(
            f"Command failed ({error_type}): {error_message}", suggestion
        )


# ============================================================================
# Queue Command Helpers (MOVED TO SHARED MODULE)
# ============================================================================
# The following functions have been extracted to lobster/cli_internal/commands/
# for reuse across CLI and Dashboard interfaces:
#
# - show_queue_status()    (was _show_queue_status)
# - queue_load_file()      (was _queue_load_file)
# - queue_list()           (was _queue_list)
# - queue_clear()          (was _queue_clear)
# - queue_export()         (was _queue_export)
#
# Import these from lobster.cli_internal.commands instead of defining them here.
# This ensures CLI and Dashboard use the same implementation (Single Source of Truth).
# ============================================================================


def _extract_argument(original_command: str, prefix: str) -> Optional[str]:
    """Extract the argument after a command prefix, preserving case and handling quotes.

    Uses the original (not lowercased) command so file paths keep their case.
    Strips surrounding single or double quotes so quoted paths with spaces work.

    Examples:
        _extract_argument('/workspace load "My File.fcs"', '/workspace load')
        → 'My File.fcs'

        _extract_argument('/read data.h5ad', '/read')
        → 'data.h5ad'
    """
    lower = original_command.lower()
    if not lower.startswith(prefix.lower()):
        return None
    arg = original_command[len(prefix) :].strip()
    if not arg:
        return None
    # Strip surrounding quotes (single or double)
    if len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in ('"', "'"):
        arg = arg[1:-1]
    return arg


def _execute_command(
    cmd: str, client: "AgentClient", original_command: Optional[str] = None,
    output: Optional["OutputAdapter"] = None,
) -> Optional[str]:
    """Execute individual slash commands.

    Args:
        cmd: Lowercased command string (for routing/dispatch).
        client: AgentClient instance.
        original_command: Original command string with preserved case.
            Used for argument extraction so file paths keep their case.
        output: Optional OutputAdapter for rendering results. Defaults to
            ConsoleOutputAdapter(console) when not provided.

    Returns:
        Optional[str]: Summary of command execution for conversation history,
                      or None if command should not be logged to history.
    """
    if original_command is None:
        original_command = cmd

    if output is None:
        output = ConsoleOutputAdapter(console)

    # -------------------------------------------------------------------------
    # Helper function for quantification directory detection
    # -------------------------------------------------------------------------
    def _is_quantification_directory(path: Path) -> Optional[str]:
        """
        Detect if directory contains Kallisto or Salmon quantification files.

        Args:
            path: Path object to check

        Returns:
            Tool type ('kallisto' or 'salmon') if quantification directory, None otherwise

        Detection criteria:
        - Kallisto: Looks for abundance.tsv, abundance.h5, or abundance.txt files
        - Salmon: Looks for quant.sf or quant.genes.sf files
        - Requires at least 2 sample subdirectories to avoid false positives
        """
        if not path.is_dir():
            return None

        kallisto_count = 0
        salmon_count = 0

        try:
            for subdir in path.iterdir():
                if not subdir.is_dir():
                    continue

                # Check for Kallisto signatures
                if (
                    (subdir / "abundance.tsv").exists()
                    or (subdir / "abundance.h5").exists()
                    or (subdir / "abundance.txt").exists()
                ):
                    kallisto_count += 1

                # Check for Salmon signatures
                if (subdir / "quant.sf").exists() or (
                    subdir / "quant.genes.sf"
                ).exists():
                    salmon_count += 1
        except (PermissionError, OSError):
            return None

        # Require at least 2 samples to identify as quantification directory
        # This avoids false positives from directories with only 1 sample
        if kallisto_count >= 2:
            return "kallisto"
        elif salmon_count >= 2:
            return "salmon"

        return None

    if cmd == "/help":
        output.print("## Quick Start\n")
        output.print("Just ask what you want to do in natural language")
        output.print('Example: "analyze my single-cell data" or "find breast cancer datasets"')
        output.print("Load data: /read <file> (supports *.h5ad, *.csv, etc.)\n")
        output.print_table({
            "title": "Core Commands",
            "columns": [
                {"name": "Command"},
                {"name": "Description"},
            ],
            "rows": [
                ["/session", "Current session status"],
                ["/status", "Subscription tier & agents"],
                ["/data", "Current data summary"],
                ["/workspace", "Workspace info"],
                ["/queue", "Queue status & controls"],
                ["/metadata", "Metadata overview"],
                ["/config", "Provider/model settings"],
                ["/plots", "List generated plots"],
                ["/read <file>", "Load data file"],
                ["/pipeline export", "Export as Jupyter notebook"],
                ["/tokens", "Token usage & costs"],
                ["/clear", "Clear screen"],
                ["/exit", "Exit"],
            ],
        })
        output.print_table({
            "title": "Power Commands",
            "columns": [
                {"name": "Command"},
                {"name": "Description"},
            ],
            "rows": [
                ["/workspace list", "List available datasets"],
                ["/workspace load <name>", "Load data from workspace"],
                ["/queue clear [download|all]", "Clear queue(s)"],
                ["/metadata clear [exports|all]", "Clear metadata"],
                ["/describe <modality>", "Inspect a modality"],
                ["/save [--force]", "Persist loaded modalities"],
                ["/restore [pattern]", "Restore datasets from session cache"],
            ],
        })
        output.print_table({
            "title": "Admin & UI Commands",
            "columns": [
                {"name": "Command"},
                {"name": "Description"},
            ],
            "rows": [
                ["/dashboard", "Classic/Textual dashboard (Go-safe fallback shown in Go mode)"],
                ["/status-panel", "Rich status dashboard (degraded to /status in Go mode)"],
                ["/workspace-info", "Rich workspace dashboard (degraded to /workspace in Go mode)"],
                ["/analysis-dash", "Rich analysis dashboard (degraded to /plots + /metadata in Go mode)"],
                ["/progress", "Rich progress monitor (degraded in Go mode)"],
            ],
        })
        output.print("Next step: use `/workspace list` or `/read <file>` to begin.", style="dim")

    elif cmd == "/data":
        return data_summary(client, output)

    elif cmd == "/session":
        try:
            if _is_protocol_output(output):
                workspace = getattr(client, "workspace_path", None)
                if workspace is None and hasattr(client, "data_manager"):
                    workspace = getattr(client.data_manager, "workspace_path", None)
                modality_count = 0
                if hasattr(client, "data_manager"):
                    modality_count = len(getattr(client.data_manager, "modalities", {}))
                output.print_table({
                    "title": "Session",
                    "columns": [{"name": "Field"}, {"name": "Value"}],
                    "rows": [
                        ["Session ID", str(getattr(client, "session_id", "unknown"))],
                        ["Workspace", str(workspace or "unknown")],
                        ["Loaded Modalities", str(modality_count)],
                    ],
                })
                output.print("Tip: use `/workspace list` to inspect available datasets.", style="dim")
                return f"Session: {getattr(client, 'session_id', 'unknown')}"
            display_session(client)
        except Exception as e:
            output.print(f"Session info error: {e}", style="error")

    elif cmd == "/status":
        try:
            if _is_protocol_output(output):
                tier = "unknown"
                try:
                    from lobster.core.governance.license_manager import get_current_tier

                    tier = get_current_tier()
                except Exception:
                    pass
                provider = getattr(client, "provider_override", None) or "auto"
                model = getattr(client, "model_override", None) or "auto"
                output.print_table({
                    "title": "Status",
                    "columns": [{"name": "Field"}, {"name": "Value"}],
                    "rows": [
                        ["Tier", str(tier)],
                        ["Provider", str(provider)],
                        ["Model", str(model)],
                        ["Session ID", str(getattr(client, "session_id", "unknown"))],
                    ],
                })
                output.print("Tip: use `/config provider` or `/config model` to change runtime settings.", style="dim")
                return None
            _display_status_info()
        except Exception as e:
            output.print(f"Status info error: {e}", style="error")

    elif cmd == "/tokens":
        try:
            token_usage = client.get_token_usage()
            if not token_usage or "error" in token_usage:
                output.print("Token tracking not available for this client type", style="warning")
                return

            from lobster.config.llm_factory import LLMFactory
            current_provider = (
                getattr(client, "provider_override", None)
                or LLMFactory.get_current_provider()
            )
            is_ollama = current_provider == "ollama"

            title = (
                "Session Token Usage (FREE - Local)"
                if is_ollama
                else "Session Token Usage & Cost"
            )
            cost_display = (
                "FREE (local model)"
                if is_ollama
                else f"${token_usage['total_cost_usd']:.4f}"
            )
            provider_display = (
                "Ollama (Local)"
                if is_ollama
                else (current_provider.capitalize() if current_provider else "Unknown")
            )

            output.print_table({
                "title": title,
                "columns": [
                    {"name": "Metric"},
                    {"name": "Value"},
                ],
                "rows": [
                    ["Session ID", token_usage["session_id"]],
                    ["Provider", provider_display],
                    ["Total Input Tokens", f"{token_usage['total_input_tokens']:,}"],
                    ["Total Output Tokens", f"{token_usage['total_output_tokens']:,}"],
                    ["Total Tokens", f"{token_usage['total_tokens']:,}"],
                    ["Total Cost", cost_display],
                ],
            })

            if token_usage.get("by_agent"):
                agent_cols = [
                    {"name": "Agent"},
                    {"name": "Input"},
                    {"name": "Output"},
                    {"name": "Total"},
                ]
                if not is_ollama:
                    agent_cols.append({"name": "Cost (USD)"})
                agent_cols.append({"name": "Calls"})

                agent_rows = []
                for agent_name, stats in token_usage["by_agent"].items():
                    row = [
                        agent_name.replace("_", " ").title(),
                        f"{stats['input_tokens']:,}",
                        f"{stats['output_tokens']:,}",
                        f"{stats['total_tokens']:,}",
                    ]
                    if not is_ollama:
                        row.append(f"${stats['cost_usd']:.4f}")
                    row.append(str(stats["invocation_count"]))
                    agent_rows.append(row)

                output.print_table({
                    "title": "Tokens by Agent",
                    "columns": agent_cols,
                    "rows": agent_rows,
                })

        except Exception as e:
            output.print(f"Failed to retrieve token usage: {e}", style="error")

    elif cmd == "/input-features":
        if _is_protocol_output(output):
            output.print_table({
                "title": "Input Features (Go TUI)",
                "columns": [{"name": "Feature"}, {"name": "Status"}],
                "rows": [
                    ["Slash command completion", "Enabled"],
                    ["Tab accept suggestion", "Enabled"],
                    ["Up/Down suggestion cycle", "Enabled"],
                    ["PgUp/PgDn transcript scroll", "Enabled"],
                    ["Ctrl+G mouse mode toggle", "Enabled"],
                    ["Mouse select mode", "Drag to copy text"],
                    ["Mouse scroll mode", "Wheel/trackpad scrolls transcript"],
                    ["Inline dropdown list", "Not supported (inline only)"],
                ],
            })
            output.print(
                "Tip: type '/' then press Tab to accept a suggestion. Use Ctrl+G to switch mouse mode between select and scroll.",
                style="info",
            )
            return None

        # Show input capabilities and navigation features
        input_features = console_manager.get_input_features()

        if PROMPT_TOOLKIT_AVAILABLE and input_features["arrow_navigation"]:
            features_text = f"""[bold white]✨ Enhanced Input Features Active[/bold white]

[bold {LobsterTheme.PRIMARY_ORANGE}]Available Navigation:[/bold {LobsterTheme.PRIMARY_ORANGE}]
• [green]←/→ Arrow keys[/green] - Navigate within your input text
• [green]↑/↓ Arrow keys[/green] - Browse command history
• [green]Ctrl+R[/green] - Reverse search through history
• [green]Home/End[/green] - Jump to beginning/end of line
• [green]Backspace/Delete[/green] - Edit text naturally

[bold {LobsterTheme.PRIMARY_ORANGE}]Autocomplete Features:[/bold {LobsterTheme.PRIMARY_ORANGE}]
• [green]Tab completion[/green] - Complete commands and file names
• [green]Smart context[/green] - Commands when typing /, files after /read
• [green]Live preview[/green] - See completions as you type
• [green]Rich metadata[/green] - File sizes, types, and descriptions
• [green]Cloud aware[/green] - Works with both local and cloud clients

[bold {LobsterTheme.PRIMARY_ORANGE}]History Features:[/bold {LobsterTheme.PRIMARY_ORANGE}]
• [green]Persistent history[/green] - Commands saved between sessions
• [green]History file[/green] - {input_features["history_file"]}
• [green]Reverse search[/green] - Ctrl+R to find previous commands

[bold {LobsterTheme.PRIMARY_ORANGE}]Tips:[/bold {LobsterTheme.PRIMARY_ORANGE}]
• Use ↑/↓ to recall previous commands and questions
• Use Ctrl+R followed by typing to search command history
• Press Tab to see available commands or files
• Edit recalled commands with arrow keys before pressing Enter"""
        elif PROMPT_TOOLKIT_AVAILABLE:
            features_text = f"""[bold white]✨ Autocomplete Features Active[/bold white]

[bold {LobsterTheme.PRIMARY_ORANGE}]Autocomplete Features:[/bold {LobsterTheme.PRIMARY_ORANGE}]
• [green]Tab completion[/green] - Complete commands and file names
• [green]Smart context[/green] - Commands when typing /, files after /read
• [green]Live preview[/green] - See completions as you type
• [green]Rich metadata[/green] - File sizes, types, and descriptions
• [green]Cloud aware[/green] - Works with both local and cloud clients

[bold {LobsterTheme.PRIMARY_ORANGE}]Available Input:[/bold {LobsterTheme.PRIMARY_ORANGE}]
• [yellow]Basic arrow navigation[/yellow] - Limited cursor control
• [yellow]Backspace/Delete[/yellow] - Edit text
• [yellow]Enter[/yellow] - Submit commands

[bold {LobsterTheme.PRIMARY_ORANGE}]Tips:[/bold {LobsterTheme.PRIMARY_ORANGE}]
• Press Tab to see available commands or files
• Type / to see all available commands
• Type /read followed by Tab to see workspace files"""
        else:
            features_text = f"""[bold white]📝 Basic Input Mode[/bold white]

[bold {LobsterTheme.PRIMARY_ORANGE}]Current Capabilities:[/bold {LobsterTheme.PRIMARY_ORANGE}]
• [yellow]Basic text input[/yellow] - Standard terminal input
• [yellow]Backspace[/yellow] - Delete characters
• [yellow]Enter[/yellow] - Submit commands

[bold {LobsterTheme.PRIMARY_ORANGE}]Upgrade Available:[/bold {LobsterTheme.PRIMARY_ORANGE}]
🚀 [bold white]Get Enhanced Input Features & Autocomplete![/bold white]
Install prompt-toolkit for arrow key navigation, command history, and Tab completion:

[bold {LobsterTheme.PRIMARY_ORANGE}]pip install prompt-toolkit[/bold {LobsterTheme.PRIMARY_ORANGE}]

[bold white]After installation, you'll get:[/bold white]
• ←/→ Arrow keys for text navigation
• ↑/↓ Arrow keys for command history
• Ctrl+R for reverse search
• [green]Tab completion[/green] for commands and files
• [green]Smart autocomplete[/green] with file metadata
• [green]Cloud-aware completion[/green] for remote files
• Persistent command history between sessions"""

        features_panel = LobsterTheme.create_panel(
            features_text, title="🔤 Input Features & Navigation"
        )
        console_manager.print(features_panel)

    elif cmd == "/dashboard":
        # Launch interactive Textual dashboard
        try:
            from lobster.ui.os_app import run_lobster_os

            console_manager.print(
                f"[{LobsterTheme.PRIMARY_ORANGE}]Launching interactive dashboard...[/{LobsterTheme.PRIMARY_ORANGE}]"
            )
            # Get workspace from client
            workspace_path = getattr(client, "workspace_path", None)
            run_lobster_os(workspace_path)
            return True  # Exit chat loop after dashboard closes
        except ImportError as e:
            console_manager.print_error_panel(
                f"Failed to import Textual UI: {str(e)}",
                "Ensure textual>=0.79.1 is installed: pip install textual",
            )
        except Exception as e:
            console_manager.print_error_panel(
                f"Failed to launch dashboard: {e}",
                "Check system permissions and try again",
            )

    elif cmd == "/status-panel":
        if _is_protocol_output(output):
            output.print(
                "/status-panel renders Rich dashboard panels and is not available in Go TUI. Use /status instead.",
                style="warning",
            )
            return None

        # Show comprehensive system health dashboard (Rich panels)
        try:
            # Create a compact dashboard using individual panels instead of full-screen layout
            status_display = get_status_display()

            # Get individual panels from the dashboard components
            core_panel = status_display._create_core_status_panel(client)
            resource_panel = status_display._create_resource_panel()
            agent_panel = status_display._create_agent_status_panel(client)

            # Print panels individually instead of using full-screen layout
            console_manager.print(
                LobsterTheme.create_panel(
                    f"[bold {LobsterTheme.PRIMARY_ORANGE}]System Health Dashboard[/bold {LobsterTheme.PRIMARY_ORANGE}]",
                    title="🦞 Status Panel",
                )
            )
            console_manager.print(core_panel)
            console_manager.print(resource_panel)
            console_manager.print(agent_panel)

        except Exception as e:
            console_manager.print_error_panel(
                f"Failed to create status panel: {e}",
                "Check system permissions and try again",
            )

    elif cmd == "/workspace-info":
        if _is_protocol_output(output):
            output.print(
                "/workspace-info renders Rich dashboard panels and is not available in Go TUI. Use /workspace or /files instead.",
                style="warning",
            )
            return None

        # Show detailed workspace overview
        try:
            # Create a compact workspace overview using individual panels instead of full-screen layout
            status_display = get_status_display()

            # Get individual panels from the workspace dashboard components
            workspace_info_panel = status_display._create_workspace_info_panel(client)
            files_panel = status_display._create_recent_files_panel(client)
            data_panel = status_display._create_data_status_panel(client)

            # Print panels individually instead of using full-screen layout
            console_manager.print(
                LobsterTheme.create_panel(
                    f"[bold {LobsterTheme.PRIMARY_ORANGE}]Workspace Overview[/bold {LobsterTheme.PRIMARY_ORANGE}]",
                    title="🏗️ Workspace",
                )
            )
            console_manager.print(workspace_info_panel)
            console_manager.print(files_panel)
            console_manager.print(data_panel)

        except Exception as e:
            console_manager.print_error_panel(
                f"Failed to create workspace overview: {e}",
                "Check if workspace is properly initialized",
            )

    elif cmd == "/analysis-dash":
        if _is_protocol_output(output):
            output.print(
                "/analysis-dash renders Rich dashboard panels and is not available in Go TUI. Use /plots and /metadata instead.",
                style="warning",
            )
            return None

        # Show analysis monitoring dashboard
        try:
            # Create a compact analysis dashboard using individual panels instead of full-screen layout
            status_display = get_status_display()

            # Get individual panels from the analysis dashboard components
            analysis_panel = status_display._create_analysis_panel(client)
            plots_panel = status_display._create_plots_panel(client)

            # Print panels individually instead of using full-screen layout
            console_manager.print(
                LobsterTheme.create_panel(
                    f"[bold {LobsterTheme.PRIMARY_ORANGE}]Analysis Dashboard[/bold {LobsterTheme.PRIMARY_ORANGE}]",
                    title="🧬 Analysis",
                )
            )
            console_manager.print(analysis_panel)
            console_manager.print(plots_panel)

        except Exception as e:
            console_manager.print_error_panel(
                f"Failed to create analysis dashboard: {e}",
                "Check if analysis operations have been performed",
            )

    elif cmd == "/progress":
        if _is_protocol_output(output):
            output.print(
                "/progress renders Rich live panels and is not available in Go TUI.",
                style="warning",
            )
            return None

        # Show multi-task progress monitor
        try:
            progress_manager = get_multi_progress_manager()
            active_count = progress_manager.get_active_operations_count()

            if active_count > 0:
                # Create a compact progress display using individual panels instead of full-screen layout
                operations_panel = progress_manager._create_operations_panel()
                details_panel = progress_manager._create_details_panel()

                # Print panels individually instead of using full-screen layout
                console_manager.print(
                    LobsterTheme.create_panel(
                        f"[bold {LobsterTheme.PRIMARY_ORANGE}]Multi-Task Progress Monitor[/bold {LobsterTheme.PRIMARY_ORANGE}]",
                        title=f"🔄 Progress ({active_count} active)",
                    )
                )
                console_manager.print(operations_panel)
                console_manager.print(details_panel)
            else:
                # Show information about the progress system
                info_text = f"""[bold white]Multi-Task Progress Monitor[/bold white]

[{LobsterTheme.PRIMARY_ORANGE}]Status:[/{LobsterTheme.PRIMARY_ORANGE}] No active multi-task operations

[bold white]Features:[/bold white]
• Real-time progress tracking for concurrent operations
• Subtask progress monitoring with detailed status
• Live updates with orange-themed progress bars
• Operation duration and completion tracking

[bold white]Usage:[/bold white]
The progress monitor automatically tracks multi-task operations
when they are started by agents or analysis workflows.

[grey50]Multi-task operations will appear here when active.[/grey50]"""

                info_panel = LobsterTheme.create_panel(
                    info_text, title="🔄 Progress Monitor"
                )
                console_manager.print(info_panel)

        except Exception as e:
            console_manager.print_error_panel(
                f"Failed to create progress monitor: {e}",
                "Check system status and try again",
            )

    elif cmd.startswith("/metadata"):
        parts = cmd.split()
        subcommand = parts[1] if len(parts) > 1 else None
        subsubcommand = parts[2] if len(parts) > 2 else None

        if subcommand == "clear":
            # Handle /metadata clear [subtype]
            if subsubcommand == "exports":
                return metadata_clear_exports(client, output)
            elif subsubcommand == "all":
                return metadata_clear_all(client, output)
            elif subsubcommand is None:
                return metadata_clear(client, output)
            else:
                output.print(f"Unknown clear type: {subsubcommand}", style="warning")
                output.print(
                    "Available: /metadata clear, /metadata clear exports, /metadata clear all",
                    style="info",
                )
                return None
        elif subcommand in ("publications", "pub"):
            # Parse --status= flag if present
            status_filter = None
            for part in parts[2:]:
                if part.startswith("--status="):
                    status_filter = part.split("=", 1)[1]
            return metadata_publications(client, output, status_filter=status_filter)
        elif subcommand in ("samples", "sample"):
            return metadata_samples(client, output)
        elif subcommand in ("workspace", "ws"):
            return metadata_workspace(client, output)
        elif subcommand in ("exports", "export"):
            return metadata_exports(client, output)
        elif subcommand == "list":
            return metadata_list(client, output)
        elif subcommand is None:
            # Default: new smart overview
            return metadata_overview(client, output)
        else:
            output.print(f"Unknown metadata subcommand: {subcommand}", style="warning")
            output.print(
                "Available: publications, samples, workspace, exports, list, clear",
                style="info",
            )
            return None

    elif cmd == "/files":
        if _is_protocol_output(output):
            return _command_files(client, output)

        # Get categorized workspace files from data_manager
        workspace_files = client.data_manager.list_workspace_files()

        if any(workspace_files.values()):
            for category, files in workspace_files.items():
                if files:
                    # Sort files by modified date (descending: newest first)
                    files_sorted = sorted(
                        files, key=lambda f: f["modified"], reverse=True
                    )

                    # Create themed table
                    table = Table(
                        title=f"🦞 {category.title()} Files",
                        **LobsterTheme.get_table_style(),
                    )
                    table.add_column("Name", style="bold white")
                    table.add_column("Size", style="grey74")
                    table.add_column("Modified", style="grey50")
                    table.add_column("Path", style="dim grey50")

                    for f in files_sorted:
                        from datetime import datetime

                        size_kb = f["size"] / 1024
                        mod_time = datetime.fromtimestamp(f["modified"]).strftime(
                            "%Y-%m-%d %H:%M"
                        )
                        table.add_row(
                            f["name"],
                            f"{size_kb:.1f} KB",
                            mod_time,
                            Path(f["path"]).parent.name,
                        )

                    console_manager.print(table)
                    console_manager.print()  # Add spacing between categories
        else:
            console_manager.print("[grey50]No files in workspace[/grey50]")

    elif cmd == "/tree":
        if _is_protocol_output(output):
            output.print(
                "Tree rendering is not available in Go TUI. Use /files for structured listing.",
                style="info",
            )
            return None

        # Show directory tree view
        try:
            # Show current directory tree
            current_tree = create_file_tree(
                root_path=current_directory,
                title=f"Current Directory: {current_directory.name}",
                show_hidden=False,
                max_depth=3,
            )

            tree_panel = LobsterTheme.create_panel(
                current_tree, title="📁 Directory Tree"
            )
            console_manager.print(tree_panel)

            # Also show workspace tree if it exists
            from lobster.core.workspace import resolve_workspace

            workspace_path = resolve_workspace(
                explicit_path=client.workspace_path, create=False
            )
            if workspace_path.exists():
                console_manager.print()  # Add spacing
                workspace_tree = create_workspace_tree(workspace_path)

                workspace_panel = LobsterTheme.create_panel(
                    workspace_tree, title="🦞 Workspace Tree"
                )
                console_manager.print(workspace_panel)

        except Exception as e:
            console_manager.print_error_panel(
                f"Failed to create tree view: {e}",
                "Check directory permissions and try again",
            )

    elif cmd.startswith("/workspace"):
        parts = cmd.split()

        # Default to showing status summary when no args (like /config and /queue)
        if len(parts) == 1:
            return workspace_status(client, output)

        subcommand = parts[1]

        if subcommand == "list":
            force_refresh = "--refresh" in cmd.lower()
            return workspace_list(client, output, force_refresh)
        elif subcommand == "info":
            selector = _extract_argument(original_command, "/workspace info")
            return workspace_info(client, output, selector)
        elif subcommand == "load":
            selector = _extract_argument(original_command, "/workspace load")
            return workspace_load(
                client, output, selector, current_directory, PathResolver
            )
        elif subcommand == "remove":
            selector = _extract_argument(original_command, "/workspace remove")
            return workspace_remove(client, output, selector)
        elif subcommand == "status":
            return workspace_status(client, output)
        elif subcommand == "save":
            force_save = "--force" in parts
            return _command_save(client, output, force=force_save)
        else:
            output.print(f"Unknown workspace subcommand: {subcommand}", style="warning")
            output.print(
                "Available: status, list, info, load, remove, save",
                style="info",
            )
            return None

    elif cmd.startswith("/queue"):
        parts = cmd.split()

        if len(parts) == 1:
            # Show queue status
            return show_queue_status(client, output)

        subcommand = parts[1] if len(parts) > 1 else None

        if subcommand == "load":
            filename = _extract_argument(original_command, "/queue load")
            try:
                return queue_load_file(client, filename, output, current_directory)
            except QueueFileTypeNotSupported as e:
                output.print(str(e), style="warning")
                return None

        elif subcommand == "list":
            # Determine queue type from additional parameters
            queue_type = "publication"  # default
            if len(parts) > 2:
                if parts[2] == "download":
                    queue_type = "download"
                elif parts[2] == "publication":
                    queue_type = "publication"
                else:
                    output.print(f"Unknown queue type: {parts[2]}", style="warning")
                    output.print(
                        "Usage: /queue list [download|publication]",
                        style="info",
                    )
                    return None
            return queue_list(client, output, queue_type=queue_type)

        elif subcommand == "clear":
            # Determine queue type from additional parameters
            queue_type = "publication"  # default
            if len(parts) > 2:
                if parts[2] == "download":
                    queue_type = "download"
                elif parts[2] == "all":
                    queue_type = "all"
                else:
                    output.print(f"Unknown queue type: {parts[2]}", style="warning")
                    output.print("Usage: /queue clear [download|all]", style="info")
                    return None
            return queue_clear(client, output, queue_type=queue_type)

        elif subcommand == "export":
            name = parts[2] if len(parts) > 2 else None
            return queue_export(client, name, output)

        elif subcommand == "import":
            filename = _extract_argument(original_command, "/queue import")
            return queue_import(client, filename, output, current_directory)

        else:
            output.print(f"Unknown queue subcommand: {subcommand}", style="warning")
            output.print("Available: load, list, clear, export, import", style="info")
            return None

    # =========================================================================
    # /read - File Inspection Only (no state change, view-only)
    # =========================================================================
    elif cmd == "/read":
        # No argument provided - show usage help
        output.print("/read - View file contents (inspection only)", style="info")
        output.print("Usage:")
        output.print("  /read <filename>        View a single file")
        output.print("  /read *.h5ad            View all H5AD files (glob pattern)")
        output.print("  /read data/*.csv        View CSVs from a subfolder")
        output.print("Examples:")
        output.print("  /read my_data.h5ad      Inspect H5AD file structure")
        output.print("  /read config.yaml       View configuration file")
        output.print("  /read results.csv       Preview CSV contents")
        output.print("To load data for analysis, use /workspace load <name>.", style="dim")
        return None

    elif cmd.startswith("/read "):
        filename = _extract_argument(original_command, "/read")
        return file_read(client, output, filename, current_directory, PathResolver)

    elif cmd.startswith("/archive"):
        parts = cmd.split()
        subcommand = parts[1] if len(parts) > 1 else "help"
        args = parts[2:] if len(parts) > 2 else None
        return archive_queue(client, output, subcommand, args)

    elif cmd.startswith("/pipeline"):
        parts = cmd.split()
        subcommand = parts[1] if len(parts) > 1 else None

        if subcommand == "export":
            name = parts[2] if len(parts) > 2 else None
            description = " ".join(parts[3:]) if len(parts) > 3 else None
            return pipeline_export(client, output, name, description)
        elif subcommand == "list":
            return pipeline_list(client, output)
        elif subcommand == "run":
            notebook_name = parts[2] if len(parts) > 2 else None
            input_modality = parts[3] if len(parts) > 3 else None
            return pipeline_run(client, output, notebook_name, input_modality)
        elif subcommand == "info":
            return pipeline_info(client, output)
        elif subcommand is None:
            return pipeline_list(client, output)
        else:
            output.print(f"Unknown pipeline subcommand: {subcommand}", style="warning")
            output.print("Available: export, list, run, info", style="info")
            return None

    elif cmd.startswith("/export"):
        # Use shared command implementation (unified with dashboard)
        # Parse options: --no-png, --force
        include_png = "--no-png" not in cmd
        force_resave = "--force" in cmd

        return export_data(
            client,
            output,
            include_png=include_png,
            force_resave=force_resave,
            console=console,
        )

    elif cmd == "/open" or cmd.startswith("/open "):
        # Handle /open command for files and folders
        file_or_folder = _extract_argument(original_command, "/open") or ""

        if not file_or_folder:
            output.print("/open: missing file or folder argument", style="error")
            output.print("Usage: /open <file_or_folder>")
            return "No file or folder specified for /open command"

        # Resolve path relative to current working directory.
        if not file_or_folder.startswith("/") and not file_or_folder.startswith("~/"):
            target_path = (current_directory / file_or_folder).expanduser()
        else:
            target_path = Path(file_or_folder).expanduser()

        if not target_path.exists():
            output.print(
                f"/open: '{file_or_folder}': No such file or folder",
                style="error",
            )
            return None

        from lobster.utils import open_path

        success, message = open_path(target_path)
        if success:
            output.print(message, style="success")
            return f"Opened {target_path.name}"
        output.print(f"/open failed: {message}", style="error")
        return None

    elif cmd == "/plots":
        return plots_list(client, output)

    elif cmd == "/modalities":
        return modalities_list(client, output)

    elif cmd.startswith("/describe"):
        modality_name = _extract_argument(original_command, "/describe")
        return modality_describe(client, output, modality_name)

    elif cmd.startswith("/save"):
        # Auto-save current state with progress display
        # Support /save --force to force re-save all modalities
        force_save = "--force" in cmd

        if _is_protocol_output(output):
            # Go TUI protocol cannot rely on direct Rich console rendering.
            return _command_save(client, output, force=force_save)

        modality_count = len(client.data_manager.modalities)
        if modality_count == 0:
            console.print("[grey50]Nothing to save (no data loaded)[/grey50]")
            return "No data to save"

        # Progress tracking state
        save_status = {"current": 0, "total": modality_count, "current_name": ""}

        def progress_callback(current: int, total: int, name: str, status: str):
            save_status["current"] = current
            save_status["total"] = total
            save_status["current_name"] = name

        # Show progress for multi-modality saves
        if modality_count > 1:
            with console.status(
                f"[bold cyan]Saving {modality_count} modalities...[/bold cyan]",
                spinner="dots",
            ) as status:

                def update_progress(current: int, total: int, name: str, state: str):
                    status_icon = {
                        "saving": "💾",
                        "skipped": "⏭️",
                        "done": "✓",
                        "error": "❌",
                    }.get(state, "•")
                    status.update(
                        f"[bold cyan]{status_icon} [{current}/{total}] {state}: {name}[/bold cyan]"
                    )

                saved_items = client.data_manager.auto_save_state(
                    progress_callback=update_progress, force=force_save
                )
        else:
            # Single modality - no progress needed
            saved_items = client.data_manager.auto_save_state(force=force_save)

        if saved_items:
            # Count actual saves vs skips
            actual_saves = [item for item in saved_items if "Skipped" not in item]
            skipped_count = len(saved_items) - len(actual_saves)

            console.print("[bold green]✓[/bold green] [white]Save complete:[/white]")
            for item in actual_saves:
                console.print(f"  [green]•[/green] {item}")
            if skipped_count > 0:
                console.print(
                    f"  [dim]⏭️ Skipped {skipped_count} unchanged modalities[/dim]"
                )

            return f"Saved {len(actual_saves)} items, skipped {skipped_count} unchanged"
        else:
            console.print(
                "[grey50]Nothing to save (all modalities up-to-date)[/grey50]"
            )
            return "All modalities already up-to-date"

    elif cmd.startswith("/restore"):
        # Restore previous session
        parts = cmd.split()
        pattern = "recent"  # default
        if len(parts) > 1:
            pattern = parts[1]
        return _command_restore(client, output, pattern=pattern)

    elif cmd.startswith("/plot"):
        parts = cmd.split()
        plot_identifier = parts[1] if len(parts) > 1 else None
        return plot_show(client, output, plot_identifier)

    elif cmd.startswith("/config"):
        parts = cmd.split()

        if len(parts) == 1 or parts[1] == "show":
            return config_show(client, output)
        elif parts[1] == "provider":
            if len(parts) == 2 or parts[2] == "list":
                return config_provider_list(client, output)
            elif parts[2] == "switch":
                provider_name = parts[3] if len(parts) > 3 else None
                save = "--save" in parts
                return config_provider_switch(client, output, provider_name, save)
            else:
                provider_name = parts[2] if len(parts) > 2 else None
                save = "--save" in parts
                return config_provider_switch(client, output, provider_name, save)
        elif parts[1] == "model":
            if len(parts) == 2 or parts[2] == "list":
                return config_model_list(client, output)
            elif parts[2] == "switch":
                model_name = parts[3] if len(parts) > 3 else None
                save = "--save" in parts
                return config_model_switch(client, output, model_name, save)
            else:
                model_name = parts[2] if len(parts) > 2 else None
                save = "--save" in parts
                return config_model_switch(client, output, model_name, save)
        else:
            output.print(f"Unknown config subcommand: {parts[1]}", style="warning")
            output.print("Available: show, provider, model", style="info")
            return None

    elif cmd.startswith("/vector-search"):
        # Parse: /vector-search "query" [--top-k N]
        raw_args = cmd[len("/vector-search") :].strip()
        if not raw_args:
            output.print("Usage: /vector-search <query> [--top-k N]", style="info")
            output.print('Example: /vector-search "glioblastoma"', style="dim")
            output.print("Example: /vector-search CD8+ T cell --top-k 10", style="dim")
            return None

        # Parse --top-k flag
        vs_top_k = None
        if "--top-k" in raw_args:
            parts_vs = raw_args.split("--top-k")
            raw_args = parts_vs[0].strip()
            try:
                vs_top_k = int(parts_vs[1].strip().split()[0])
            except (ValueError, IndexError):
                output.print("Invalid --top-k value, using default (5)", style="warning")

        # Strip surrounding quotes from query
        query_text = raw_args.strip().strip("\"'")
        if not query_text:
            output.print("Please provide a search query.", style="warning")
            return None

        try:
            from lobster.cli_internal.commands import vector_search_all_collections

            result = vector_search_all_collections(query_text, top_k=vs_top_k)
            output.print_code_block(json.dumps(result, indent=2), language="json")
            # Summary for conversation history
            total = sum(len(v) for v in result["results"].values())
            return (
                f"Vector search for '{query_text}': {total} matches across 3 ontologies"
            )
        except ImportError as e:
            output.print(str(e), style="error")
            return None
        except Exception as e:
            output.print(f"Vector search error: {e}", style="error")
            return None

    elif cmd == "/clear":
        if _is_protocol_output(output):
            # Go TUI owns screen clearing; avoid direct Rich console calls.
            return None
        console.clear()

    elif cmd == "/reset":
        # Reset conversation state
        client.reset()
        output.print("Conversation reset", style="success")
        output.print("Messages cleared. Data and modalities retained.", style="dim")
        return "Conversation reset"

    elif cmd == "/exit":
        if _is_protocol_output(output):
            # Non-interactive protocol mode cannot use direct Rich prompts.
            if output.confirm("exit?"):
                raise KeyboardInterrupt
            return None

        if Confirm.ask("[dim]exit?[/dim]"):
            display_goodbye()

            # Display session ID for continuity
            console.print(
                f"[dim]Session: {client.session_id} "
                f"(use --session-id latest to continue)[/dim]"
            )

            console.print(
                "[dim]◆ feedback: [link=https://forms.cloud.microsoft/e/AkNk8J8nE8]forms.cloud.microsoft/e/AkNk8J8nE8[/link][/dim]"
            )
            console.print(
                "[dim]◆ issues: [link=https://github.com/the-omics-os/lobster/issues]github.com/the-omics-os/lobster/issues[/link][/dim]"
            )

            # Display session token usage summary
            if hasattr(client, "token_tracker") and client.token_tracker:
                summary = client.token_tracker.get_minimal_summary()
                if summary:
                    console.print(f"[dim]{summary}[/dim]")
            console.print()  # Empty line before exit
            raise KeyboardInterrupt

    else:
        output.print(f"Unknown command: {cmd}", style="error")


# ============================================================================
# lobster command — fast programmatic slash command access
# ============================================================================


def _command_files(client, output) -> Optional[str]:
    """List workspace files via OutputAdapter (standalone version of /files)."""
    workspace_files = client.data_manager.list_workspace_files()

    if not any(workspace_files.values()):
        output.print("No files in workspace", style="info")
        return None

    from datetime import datetime

    for category, files in workspace_files.items():
        if files:
            files_sorted = sorted(files, key=lambda f: f["modified"], reverse=True)
            rows = []
            for f in files_sorted:
                size_kb = f["size"] / 1024
                mod_time = datetime.fromtimestamp(f["modified"]).strftime(
                    "%Y-%m-%d %H:%M"
                )
                rows.append(
                    [
                        f["name"],
                        f"{size_kb:.1f} KB",
                        mod_time,
                        Path(f["path"]).parent.name,
                    ]
                )

            output.print_table(
                {
                    "title": f"{category.title()} Files",
                    "columns": [
                        {"name": "Name", "style": "bold white"},
                        {"name": "Size", "style": "grey74"},
                        {"name": "Modified", "style": "grey50"},
                        {"name": "Path", "style": "dim grey50"},
                    ],
                    "rows": rows,
                }
            )

    total = sum(len(v) for v in workspace_files.values())
    return f"Listed {total} workspace files"





def _command_save(client, output, force: bool = False) -> Optional[str]:
    """Save workspace state via OutputAdapter (standalone version of /save)."""
    modality_count = len(client.data_manager.modalities)
    if modality_count == 0:
        output.print("Nothing to save (no data loaded)", style="info")
        return "No data to save"

    saved_items = client.data_manager.auto_save_state(force=force)
    if saved_items:
        actual_saves = [item for item in saved_items if "Skipped" not in item]
        skipped_count = len(saved_items) - len(actual_saves)
        for item in actual_saves:
            output.print(f"Saved: {item}", style="info")
        if skipped_count > 0:
            output.print(f"Skipped {skipped_count} unchanged modalities", style="info")
        return f"Saved {len(actual_saves)} items, skipped {skipped_count} unchanged"
    else:
        output.print("All modalities already up-to-date", style="info")
        return "All modalities already up-to-date"


def _command_restore(client, output, pattern: str = "recent") -> Optional[str]:
    """Restore workspace state via OutputAdapter (standalone version of /restore)."""
    if not hasattr(client.data_manager, "restore_session"):
        output.print("Restore not available for this client", style="error")
        return None

    try:
        result = client.data_manager.restore_session(pattern=pattern)
        restored = result.get("restored", [])
        skipped = result.get("skipped", [])
        if restored:
            for item in restored:
                name = item if isinstance(item, str) else item.get("name", str(item))
                output.print(f"Restored: {name}", style="info")
        if skipped:
            output.print(f"Skipped {len(skipped)} items", style="info")
        if not restored and not skipped:
            output.print("Nothing to restore", style="info")
            return "Nothing to restore"
        return f"Restored {len(restored)} datasets"
    except Exception as e:
        output.print(f"Restore failed: {e}", style="error")
        return None


_UNKNOWN_COMMAND = object()  # Sentinel for unrecognized commands


def _dispatch_command(cmd_str: str, client, output):
    """Route a command string to the appropriate shared command function.

    Returns the command's summary string (may be None for valid commands with
    no data), or the _UNKNOWN_COMMAND sentinel if the command is unrecognized.
    """
    from lobster.cli_internal.commands import (
        config_model_list,
        config_provider_list,
        config_show,
        data_summary,
        export_data,
        file_read,
        metadata_clear,
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
        plots_list,
        queue_clear,
        queue_export,
        queue_list,
        show_queue_status,
        vector_search_all_collections,
        workspace_info,
        workspace_list,
        workspace_load,
        workspace_remove,
        workspace_status,
    )
    from lobster.cli_internal.utils.path_resolution import PathResolver

    parts = cmd_str.strip().split(None, 2)
    base = parts[0] if parts else ""
    sub = parts[1] if len(parts) > 1 else None
    args = parts[2] if len(parts) > 2 else None

    # --- Data & files ---

    if base == "data":
        return data_summary(client, output)

    elif base == "files":
        return _command_files(client, output)

    elif base == "read":
        filename = sub
        if args:
            filename = f"{sub} {args}"
        if not filename:
            output.print("Usage: read <file|pattern>", style="error")
            return None
        current_directory = Path(client.workspace_path)
        return file_read(client, output, filename, current_directory, PathResolver)

    elif base == "plots":
        return plots_list(client, output)

    elif base == "modalities":
        return modalities_list(client, output)

    elif base == "describe":
        name = sub  # "describe <name>" — name is the second token
        return modality_describe(client, output, name)

    # --- Workspace ---

    elif base == "workspace":
        if sub == "list":
            return workspace_list(client, output)
        elif sub == "info" and args:
            return workspace_info(client, output, args)
        elif sub == "load" and args:
            current_directory = Path(client.workspace_path)
            return workspace_load(client, output, args, current_directory, PathResolver)
        elif sub == "remove" and args:
            return workspace_remove(client, output, args)
        else:
            return workspace_status(client, output)

    # --- Queue ---

    elif base == "queue":
        if sub == "list":
            queue_type = args if args else "publication"
            return queue_list(client, output, queue_type=queue_type)
        elif sub == "clear":
            queue_type = args if args else "publication"
            return queue_clear(client, output, queue_type=queue_type)
        elif sub == "export":
            queue_type = args if args else "publication"
            return queue_export(client, output, queue_type=queue_type)
        else:
            return show_queue_status(client, output)

    # --- Metadata ---

    elif base == "metadata":
        if sub == "publications":
            return metadata_publications(client, output)
        elif sub == "samples":
            return metadata_samples(client, output)
        elif sub == "workspace":
            return metadata_workspace(client, output)
        elif sub == "exports":
            return metadata_exports(client, output)
        elif sub == "list":
            return metadata_list(client, output)
        elif sub == "clear":
            return metadata_clear(client, output)
        else:
            return metadata_overview(client, output)

    # --- Pipeline ---
    #
    # `pipeline export` and `pipeline run` require provenance (AnalysisStep IR chain).
    # With session_id support, provenance can be loaded from disk via CommandClient.
    # Check provenance availability and delegate to existing functions if present.
    #
    # `pipeline list` and `pipeline info` are read-only and work without provenance.

    elif base == "pipeline":
        if sub == "export":
            # Check if provenance available (may be loaded from session_dir)
            prov = getattr(client.data_manager, "provenance", None)

            if not prov or not prov.activities:
                # No provenance - provide guidance
                output.print(
                    "Pipeline export requires a session with analysis history.",
                    style="error",
                )
                output.print(
                    "Usage: lobster command 'pipeline export' --session-id <id>",
                    style="info",
                )

                # List available sessions if any exist
                sessions_dir = client.workspace_path / ".lobster" / "sessions"
                if sessions_dir.exists():
                    sessions = sorted(
                        [d.name for d in sessions_dir.iterdir() if d.is_dir()]
                    )
                    if sessions:
                        output.print(
                            f"Available sessions: {', '.join(sessions[:5])}"
                            + (" ..." if len(sessions) > 5 else ""),
                            style="info",
                        )

                return None

            # Provenance loaded - delegate to existing function
            # Pass defaults since `lobster command` may run non-interactively
            return pipeline_export(
                client, output, name="analysis_workflow", description=""
            )

        elif sub == "run":
            # Similar check for pipeline run
            prov = getattr(client.data_manager, "provenance", None)

            if not prov or not prov.activities:
                output.print(
                    "Pipeline run requires a session with analysis history.",
                    style="error",
                )
                output.print(
                    "Usage: lobster command 'pipeline run <notebook>' --session-id <id>",
                    style="info",
                )
                return None

            # Provenance loaded - delegate to existing function
            return pipeline_run(client, output)

        elif sub == "info":
            return pipeline_info(client, output)
        else:
            # Default to list (covers "pipeline list" and bare "pipeline")
            return pipeline_list(client, output)

    # --- Save & restore ---

    elif base == "save":
        force_save = sub == "--force" or args == "--force" if sub or args else False
        return _command_save(client, output, force=force_save)

    elif base == "restore":
        pattern = sub if sub else "recent"
        return _command_restore(client, output, pattern=pattern)

    # --- Export ---

    elif base == "export":
        include_png = True
        force_resave = False
        flag_str = f"{sub or ''} {args or ''}".strip()
        if "--no-png" in flag_str:
            include_png = False
        if "--force" in flag_str:
            force_resave = True
        return export_data(
            client, output, include_png=include_png, force_resave=force_resave
        )

    # --- Config ---

    elif base == "config":
        if sub == "provider":
            if args == "list" or args is None:
                return config_provider_list(client, output)
        elif sub == "model":
            if args == "list" or args is None:
                return config_model_list(client, output)
        else:
            return config_show(client, output)

    # --- Vector search ---

    elif base == "vector-search":
        # Reassemble query from sub + args
        query_text = ""
        if sub:
            query_text = sub
            if args:
                query_text = f"{sub} {args}"
        query_text = query_text.strip().strip("\"'")
        if not query_text:
            output.print("Usage: vector-search <query> [--top-k N]", style="error")
            return None

        # Parse --top-k flag
        vs_top_k = None
        if "--top-k" in query_text:
            tk_parts = query_text.split("--top-k")
            query_text = tk_parts[0].strip()
            try:
                vs_top_k = int(tk_parts[1].strip().split()[0])
            except (ValueError, IndexError):
                pass

        try:
            result = vector_search_all_collections(query_text, top_k=vs_top_k)
            import json as _json

            output.print(_json.dumps(result, indent=2))
            total = sum(len(v) for v in result["results"].values())
            return f"Vector search for '{query_text}': {total} matches"
        except ImportError as e:
            output.print(str(e), style="error")
            return None
        except Exception as e:
            output.print(f"Vector search error: {e}", style="error")
            return None

    else:
        available = [
            "data",
            "files",
            "read <file>",
            "workspace [list|info|load|remove]",
            "plots",
            "modalities",
            "describe <name>",
            "queue [list|clear|export]",
            "metadata [publications|samples|workspace|exports|list|clear]",
            "pipeline [list|info]",
            "save [--force]",
            "restore [pattern]",
            "export [--no-png] [--force]",
            "config [provider list|model list]",
            "vector-search <query>",
        ]
        output.print(f"Unknown command: {cmd_str}", style="error")
        output.print("Available commands: " + ", ".join(available), style="info")
        return _UNKNOWN_COMMAND



# ============================================================================
# Command impl functions (extracted from cli.py Plan 08-02)
# ============================================================================


def metadata_command_impl(subcommand=None, workspace=None, status_filter=None):
    """Metadata command dispatcher (extracted from cli.py)."""
    from lobster.core.client import AgentClient
    from lobster.core.workspace import resolve_workspace
    from lobster.cli_internal.commands import (
        ConsoleOutputAdapter,
        metadata_overview, metadata_publications, metadata_samples,
        metadata_workspace, metadata_exports, metadata_list,
    )
    import typer

    from lobster.core.client import AgentClient

    try:
        # Resolve workspace
        workspace_path = resolve_workspace(workspace, create=False)

        # Create client
        client = AgentClient(workspace_path=str(workspace_path))
        output = ConsoleOutputAdapter(console)

        # Route to appropriate command
        if subcommand is None:
            metadata_overview(client, output)
        elif subcommand in ("publications", "pub"):
            metadata_publications(client, output, status_filter=status_filter)
        elif subcommand in ("samples", "sample"):
            metadata_samples(client, output)
        elif subcommand in ("workspace", "ws"):
            metadata_workspace(client, output)
        elif subcommand in ("exports", "export"):
            metadata_exports(client, output)
        elif subcommand == "list":
            metadata_list(client, output)
        elif subcommand == "clear":
            console.print("[yellow]Use: lobster metadata clear [exports|all][/yellow]")
            console.print(
                "[cyan]  lobster metadata clear         # Clear metadata (memory + workspace/metadata/)[/cyan]"
            )
            console.print(
                "[cyan]  lobster metadata clear exports # Clear export files only[/cyan]"
            )
            console.print(
                "[cyan]  lobster metadata clear all     # Clear ALL metadata[/cyan]"
            )
        else:
            console.print(f"[red]Unknown subcommand: {subcommand}[/red]")
            console.print(
                "[cyan]Available: overview, publications, samples, workspace, exports, list, clear[/cyan]"
            )
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        raise typer.Exit(1)




def activate_impl(access_code: str, server_url=None):
    """Activate a premium license (extracted from cli.py)."""
    from rich.panel import Panel
    from rich.prompt import Confirm
    from lobster.ui import LobsterTheme
    import typer

    console.print()
    console.print(
        Panel.fit(
            f"[bold {LobsterTheme.PRIMARY_ORANGE}]🦞 License Activation[/bold {LobsterTheme.PRIMARY_ORANGE}]",
            border_style=LobsterTheme.PRIMARY_ORANGE,
            padding=(0, 2),
        )
    )
    console.print()

    # Check if already activated
    from lobster.core.license_manager import get_entitlement_status

    current = get_entitlement_status()
    if (
        current.get("tier") not in ("free", None)
        and current.get("source") == "license_file"
    ):
        console.print(
            f"[yellow]⚠️  You already have an active {current.get('tier_display', 'Premium')} license[/yellow]"
        )
        console.print(f"[dim]Source: {current.get('source')}[/dim]")
        console.print()
        if not Confirm.ask("Replace existing license?", default=False):
            console.print("[yellow]Activation cancelled[/yellow]")
            raise typer.Exit(0)

    console.print("[dim]Contacting license server...[/dim]")

    try:
        from lobster.core.license_manager import activate_license

        result = activate_license(access_code, license_server_url=server_url)

        if result.get("success"):
            entitlement = result.get("entitlement", {})
            tier = entitlement.get("tier", "premium").title()
            packages_installed = result.get("packages_installed", [])
            packages_failed = result.get("packages_failed", [])

            # Build the success message
            msg_lines = [
                "[bold green]✅ License Activated Successfully![/bold green]\n",
                f"Tier: [bold]{tier}[/bold]",
                f"Features: {', '.join(entitlement.get('features', []))}",
            ]

            # Show installed packages
            if packages_installed:
                msg_lines.append("")
                msg_lines.append(
                    f"[bold green]Custom Packages Installed ({len(packages_installed)}):[/bold green]"
                )
                for pkg in packages_installed:
                    msg_lines.append(
                        f"  [green]✓[/green] {pkg['name']} v{pkg['version']}"
                    )

            # Show failed packages with warnings
            if packages_failed:
                msg_lines.append("")
                msg_lines.append(
                    f"[bold yellow]⚠️  Package Installation Issues ({len(packages_failed)}):[/bold yellow]"
                )
                for pkg in packages_failed:
                    error = pkg.get("error", "Unknown error")
                    msg_lines.append(
                        f"  [yellow]✗[/yellow] {pkg['name']}: {error[:50]}..."
                    )
                msg_lines.append("")
                msg_lines.append(
                    "[dim]You can retry later with: pip install <package_name>[/dim]"
                )

            msg_lines.append("")
            msg_lines.append(
                f"Run [bold {LobsterTheme.PRIMARY_ORANGE}]lobster status[/bold {LobsterTheme.PRIMARY_ORANGE}] to see available agents."
            )

            console.print()
            console.print(
                Panel.fit(
                    "\n".join(msg_lines),
                    border_style="green",
                    padding=(1, 2),
                )
            )
        else:
            error = result.get("error", "Unknown error")
            console.print()
            console.print(
                Panel.fit(
                    f"[bold red]❌ Activation Failed[/bold red]\n\n"
                    f"Error: {error}\n\n"
                    f"[dim]If this problem persists, contact support@omics-os.com[/dim]",
                    border_style="red",
                    padding=(1, 2),
                )
            )
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]❌ Activation error: {e}[/red]")
        raise typer.Exit(1)




def deactivate_impl():
    """Deactivate the current premium license (extracted from cli.py)."""
    from rich.panel import Panel
    from rich.prompt import Confirm
    from lobster.ui import LobsterTheme
    import typer

    console.print()
    console.print(
        Panel.fit(
            f"[bold {LobsterTheme.PRIMARY_ORANGE}]🦞 License Deactivation[/bold {LobsterTheme.PRIMARY_ORANGE}]",
            border_style=LobsterTheme.PRIMARY_ORANGE,
            padding=(0, 2),
        )
    )
    console.print()

    # Check current status
    try:
        from lobster.core.license_manager import (
            clear_entitlement,
            get_entitlement_status,
        )

        current = get_entitlement_status()

        if current.get("source") == "cloud_key":
            console.print(
                "[yellow]⚠️  Your premium tier is from LOBSTER_CLOUD_KEY environment variable.[/yellow]"
            )
            console.print(
                "[dim]To deactivate, unset the LOBSTER_CLOUD_KEY environment variable.[/dim]"
            )
            raise typer.Exit(0)

        if current.get("source") == "environment":
            console.print(
                "[yellow]⚠️  Your tier is set via LOBSTER_SUBSCRIPTION_TIER environment variable.[/yellow]"
            )
            console.print(
                "[dim]To deactivate, unset the LOBSTER_SUBSCRIPTION_TIER environment variable.[/dim]"
            )
            raise typer.Exit(0)

        if current.get("tier") == "free" or current.get("source") == "default":
            console.print("[dim]No active license found. Already on free tier.[/dim]")
            raise typer.Exit(0)

        # Confirm deactivation
        tier_display = current.get("tier_display", "Premium")
        console.print(f"[bold]Current tier:[/bold] {tier_display}")
        console.print(f"[dim]Source: {current.get('source')}[/dim]")
        console.print()

        if not Confirm.ask(
            f"[yellow]Deactivate {tier_display} license and revert to Free tier?[/yellow]",
            default=False,
        ):
            console.print("[yellow]Deactivation cancelled[/yellow]")
            raise typer.Exit(0)

        # Remove license file
        if clear_entitlement():
            console.print()
            console.print(
                Panel.fit(
                    "[bold green]✅ License Deactivated[/bold green]\n\n"
                    "You are now on the Free tier.\n"
                    f"Run [bold {LobsterTheme.PRIMARY_ORANGE}]lobster activate <code>[/bold {LobsterTheme.PRIMARY_ORANGE}] to re-activate.",
                    border_style="green",
                    padding=(1, 2),
                )
            )
        else:
            console.print("[red]❌ Failed to remove license file[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]❌ License deactivation error: {e}[/red]")
        raise typer.Exit(1)




def command_cmd_impl(cmd: str, workspace=None, session_id=None, json_output: bool = False):
    """Execute workspace commands without an LLM session (extracted from cli.py)."""
    import json
    import sys
    from pathlib import Path
    from lobster.core.workspace import resolve_workspace
    from lobster.cli_internal.commands.heavy.session_infra import CommandClient
    import typer

    # Strip leading / if user types it out of habit
    cmd_str = cmd.lstrip("/").strip()
    if not cmd_str:
        console.print("[red]No command specified.[/red]")
        raise typer.Exit(1)

    # In JSON mode, redirect Rich console to stderr
    if json_output:
        console.file = sys.stderr

    try:
        workspace_path = resolve_workspace(explicit_path=workspace, create=True)

        # Resolve session_id: None or "latest" both resolve to most recent session
        resolved_session_id = None

        if session_id and session_id != "latest":
            # Explicit session ID provided
            resolved_session_id = session_id
        else:
            # session_id is None or "latest" - resolve to most recent session
            sessions_dir = workspace_path / ".lobster" / "sessions"

            if sessions_dir.exists():
                # Find all session directories, sort by mtime (most recent first)
                session_dirs = sorted(
                    [d for d in sessions_dir.iterdir() if d.is_dir()],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )

                if session_dirs:
                    resolved_session_id = session_dirs[0].name
                    # Show resolved session
                    from datetime import datetime

                    last_activity = datetime.fromtimestamp(
                        session_dirs[0].stat().st_mtime
                    )
                    console.print(
                        f"Using session: {resolved_session_id} "
                        f"(last activity: {last_activity.strftime('%Y-%m-%d %H:%M')})"
                    )

            # If no sessions found (either no directory or empty)
            if not resolved_session_id:
                console.print("No sessions found in workspace.")
                console.print(
                    "Create one with: lobster query '...' --session-id <name>"
                )
                raise typer.Exit(1)

        cmd_client = CommandClient(workspace_path, session_id=resolved_session_id)

        if json_output:
            from lobster.cli_internal.commands import JsonOutputAdapter

            output = JsonOutputAdapter()
        else:
            from lobster.cli_internal.commands import ConsoleOutputAdapter

            output = ConsoleOutputAdapter(console)

        result = _dispatch_command(cmd_str, cmd_client, output)
        success = result is not _UNKNOWN_COMMAND

        if json_output:
            envelope: Dict[str, Any] = {
                "success": success,
                "command": cmd_str,
                "data": output.to_dict(),
            }
            if result is not _UNKNOWN_COMMAND and result is not None:
                envelope["summary"] = result
            print(json.dumps(envelope))

    except Exception as e:
        if json_output:
            print(
                json.dumps(
                    {
                        "success": False,
                        "command": cmd_str,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )
            )
        else:
            console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)




def vector_search_cmd_impl(query_text: str, top_k=None, pretty: bool = True):
    """Search all ontology collections via semantic vector similarity (extracted from cli.py)."""
    import json
    from lobster.cli_internal.commands import vector_search_all_collections
    import typer

    try:
        from lobster.cli_internal.commands import vector_search_all_collections

        result = vector_search_all_collections(query_text, top_k=top_k)
        indent = 2 if pretty else None
        print(json.dumps(result, indent=indent))
    except ImportError as e:
        from rich.markup import escape as rich_escape

        console.print(f"[red]{rich_escape(str(e))}[/red]", stderr=True)
        raise typer.Exit(1)
    except Exception as e:
        from rich.markup import escape as rich_escape

        console.print(
            f"[red]Vector search error: {rich_escape(str(e))}[/red]", stderr=True
        )
        raise typer.Exit(1)




def serve_impl(port: int = 8000, host: str = "0.0.0.0"):
    """Start the agent system as an API server (extracted from cli.py)."""
    from typing import Optional
    import typer

    import uvicorn
    from lobster.cli_internal.commands.heavy.session_infra import init_client
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    # Create FastAPI app
    api = FastAPI(
        title="Lobster Agent API",
        description="🦞 Multi-Agent Bioinformatics System by Omics-OS",
        version="2.0",
    )

    class QueryRequest(BaseModel):
        question: str
        session_id: Optional[str] = None
        stream: bool = False

    @api.post("/query")
    async def query_endpoint(request: QueryRequest):
        try:
            client = init_client()
            result = client.query(request.question, stream=request.stream)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @api.get("/status")
    async def status_endpoint():
        client = init_client()
        return client.get_status()

    console.print(f"[red]🦞 Starting Lobster API server on {host}:{port}[/red]")
    uvicorn.run(api, host=host, port=port)
