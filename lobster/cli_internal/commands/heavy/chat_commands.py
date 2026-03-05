"""
Chat command body and supporting functions.

Extracted from cli.py -- the `lobster chat` interactive loop and all
supporting functions (input, display, welcome/goodbye, handle_command).
"""

import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import typer
from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from lobster.cli_internal.commands.heavy.animations import (
    display_goodbye,
    display_welcome,
)
from lobster.cli_internal.commands.heavy.session_infra import (
    LobsterClientAdapter,
    _add_command_to_history,
    _maybe_print_timings,
    display_session,
    init_client,
    init_client_with_animation,
    set_go_tui_active,
    should_show_progress,
)
from lobster.cli_internal.commands.heavy.slash_commands import (
    _execute_command,
    check_for_missing_slash_command,
)
from lobster.ui import LobsterTheme, setup_logging
from lobster.ui.console_manager import get_console_manager
from lobster.utils.callbacks import TerminalCallbackHandler
from lobster.utils.system import open_path
from lobster.version import __version__

try:
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import ThreadedCompleter
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style

    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

if TYPE_CHECKING:
    from lobster.core.client import AgentClient

logger = logging.getLogger(__name__)

console_manager = get_console_manager()
console = console_manager.console

# Global state shared across functions in this module
current_directory = Path.cwd()
client = None

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
            from lobster.cli_internal.commands.heavy.slash_commands import (
                LobsterContextualCompleter,
            )

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



def get_current_agent_name() -> str:
    """Get the current active agent name for display."""
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


# ============================================================================
# Go TUI Chat Integration
# ============================================================================


def _normalize_tool_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize tool_execution payload keys for the Go TUI protocol."""
    return {
        "tool_name": payload.get("tool", payload.get("tool_name", "")),
        "event": {"running": "start", "complete": "finish", "error": "error"}.get(
            payload.get("status", ""), payload.get("event", "")
        ),
        "summary": payload.get("summary", payload.get("status", "")),
        "agent": payload.get("agent", ""),
        "duration_ms": payload.get("duration_ms", 0),
    }


def _handle_slash_command(bridge, client, cmd: str) -> None:
    """Handle a slash command by suspending TUI and running in raw terminal."""
    bridge.send("suspend", {})
    import time

    time.sleep(0.1)  # Brief pause for TUI to suspend
    try:
        handle_command(cmd, client)
    except Exception as e:
        from rich.console import Console

        Console().print(f"[red]Command error:[/red] {e}")
    finally:
        input("\nPress Enter to return to chat...")
        bridge.send("resume", {})


def _handle_user_query(bridge, client, text: str) -> None:
    """Stream a user query through the Go TUI bridge."""
    bridge.send("spinner", {"active": True})
    try:
        for event in client.query(text, stream=True):
            if event["type"] == "content_delta":
                bridge.send("text", {"content": event["delta"]})
            elif event["type"] == "agent_change":
                bridge.send(
                    "agent_transition",
                    {
                        "agent": event.get("agent", ""),
                        "status": event.get("status", "working"),
                    },
                )
            elif event["type"] == "complete":
                bridge.send(
                    "done",
                    {
                        "duration": event.get("duration", 0),
                        "response_length": len(event.get("response", "")),
                    },
                )
                # Send status update with usage info
                usage = getattr(client, "_last_usage", None)
                if usage:
                    bridge.send(
                        "status",
                        {
                            "text": f"Tokens: {usage.get('total_tokens', '?')} "
                            f"· Cost: ${usage.get('total_cost', 0):.4f}"
                        },
                    )
                bridge.send("spinner", {"active": False})
            elif event["type"] == "error":
                bridge.send(
                    "alert",
                    {
                        "level": "error",
                        "message": str(event.get("error", "Unknown error")),
                    },
                )
                bridge.send("spinner", {"active": False})
    except KeyboardInterrupt:
        bridge.send(
            "alert", {"level": "warning", "message": "Query interrupted"}
        )
        bridge.send("spinner", {"active": False})
    except Exception as e:
        bridge.send("alert", {"level": "error", "message": f"Error: {e}"})
        bridge.send("spinner", {"active": False})


def _go_tui_event_loop(bridge, client) -> None:
    """Main event loop: read Go TUI events and dispatch."""
    while True:
        event = bridge.recv_event(timeout=None)
        if event is None:
            break
        if event.type == "quit":
            break
        elif event.type == "input":
            _handle_user_query(bridge, client, event.payload.get("content", ""))
        elif event.type == "slash_command":
            _handle_slash_command(
                bridge, client, event.payload.get("command", "")
            )
        elif event.type == "cancel":
            pass  # Phase 2: cancel in-progress query


def _run_go_tui_chat(
    binary: str,
    workspace: Optional[Path],
    session_id: Optional[str],
    provider: Optional[str],
    model: Optional[str],
    debug: bool,
    profile_timings: Optional[bool],
    stream: bool,
) -> None:
    """Launch the Go TUI chat interface with JSON-lines IPC."""
    from lobster.ui.bridge import GoTUIBridge
    from lobster.ui.callbacks.protocol_callback import ProtocolCallbackHandler

    # Initialize client FIRST — Rich spinner visible on normal terminal.
    # Do NOT set _go_tui_active yet; that suppresses Rich progress.
    client = init_client_with_animation(
        workspace=workspace,
        reasoning=False,
        verbose=False,
        debug=debug,
        profile_timings=profile_timings,
        provider_override=provider,
        model_override=model,
        session_id=session_id,
    )

    # Now suppress Rich and hand the terminal to Go.
    set_go_tui_active(True)
    bridge = GoTUIBridge(binary, mode="chat")
    bridge.start()

    proto_callback = ProtocolCallbackHandler(
        emit_event=lambda msg_type, payload: bridge.send(
            msg_type,
            _normalize_tool_payload(payload)
            if msg_type == "tool_execution"
            else payload,
        )
    )
    client.callbacks.append(proto_callback)

    try:
        _go_tui_event_loop(bridge, client)
    finally:
        bridge.close()
        set_go_tui_active(False)


def chat_impl(
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
    stream: bool = typer.Option(
        True,
        "--stream/--no-stream",
        help="Enable real-time text streaming (default: on)",
    ),
    ui_mode: str = "auto",
):
    """
    Start an interactive chat session with the multi-agent system.

    Use --session-id to continue a previous conversation:
      lobster chat --session-id latest
      lobster chat --session-id session_20241208_150000

    Use --reasoning to see agent thinking process. Use --verbose for detailed tool output.
    Use --ui to select the UI backend: auto, go, or classic.
    """
    # Go TUI gate: try launching Go TUI before falling through to classic Rich UI
    if ui_mode in ("auto", "go") and not reasoning and not verbose:
        from lobster.ui.bridge import find_tui_binary

        binary = find_tui_binary()
        if binary:
            _run_go_tui_chat(
                binary, workspace, session_id, provider, model,
                debug, profile_timings, stream,
            )
            return
        elif ui_mode == "go":
            console.print(
                "[red]Error:[/red] Go TUI binary not found. "
                "Install with: pip install lobster-ai-tui"
            )
            raise typer.Exit(1)

    # Enhanced error handling setup
    if debug:
        # Enable more detailed tracebacks in debug mode
        import rich.traceback

        rich.traceback.install(
            console=console_manager.error_console,
            width=None,
            extra_lines=5,
            theme="monokai",
            word_wrap=True,
            show_locals=True,  # Show local variables in debug mode
            suppress=[],
            max_frames=30,
        )

    # Configure logging level based on debug flag
    import logging

    if debug:
        setup_logging(logging.DEBUG)
    else:
        setup_logging(logging.WARNING)  # Suppress INFO logs

    # Check for configuration
    env_file = Path.cwd() / ".env"
    if not env_file.exists():
        console.print()
        console.print("[red]✗[/red] no config found")
        console.print("  run [bold #e45c47]lobster init[/bold #e45c47]")
        console.print()
        raise typer.Exit(1)

    # Handle session loading for continuity (similar to query command)
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
                console.print(
                    f"[cyan]📂 Loading session: {session_file_to_load.name}[/cyan]"
                )
            else:
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
                console.print(
                    f"[cyan]📂 Loading session: {session_file_candidate.name}[/cyan]"
                )
            else:
                # Session file doesn't exist - use this session_id for new session
                session_id_for_client = session_id
                console.print(f"[cyan]📂 Creating new session: {session_id}[/cyan]")

    # Show DNA animation (starts instantly thanks to lazy imports)
    display_welcome()

    # Eagerly preload Ollama model in background while client initializes.
    # This overlaps model weight loading with graph creation, so the model
    # is warm by the time the user types their first query.
    if provider and provider.lower() == "ollama" and model:
        try:
            from lobster.config.providers import get_provider

            _ollama = get_provider("ollama")
            if _ollama:
                _ollama.preload_model_async(model)
        except Exception:
            pass  # Non-critical — user gets normal cold-load if this fails

    # Initialize client (heavy imports happen here)
    try:
        client = init_client_with_animation(
            workspace,
            reasoning,
            verbose,
            debug,
            profile_timings,
            provider,
            model,
            session_id_for_client,
        )
    except (SystemExit, typer.Exit):
        raise  # Already handled with clean message
    except Exception as e:
        console.print(f"\n[red]✗[/red] init failed: {str(e)[:80]}")
        raise

    # Load session if found
    if session_file_to_load:
        try:
            load_result = client.load_session(session_file_to_load)
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
            console.print(f"[red]❌ Failed to load session: {e}[/red]")
            console.print("[yellow]   Starting fresh session instead[/yellow]")

    # Show compact session status
    _show_workspace_prompt(client)

    while True:
        try:
            # Get current token usage for prompt prefix
            token_prefix = ""
            try:
                token_usage = client.get_token_usage()
                if token_usage and "error" not in token_usage:
                    cost = token_usage.get("total_cost_usd", 0.0)
                    tokens = token_usage.get("total_tokens", 0)

                    # Get provider using ConfigResolver (same as /config command)
                    from lobster.core.config_resolver import ConfigResolver

                    resolver = ConfigResolver(
                        workspace_path=Path(client.workspace_path)
                    )
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

                    if current_provider == "ollama":
                        # Ollama is free - show icon, FREE, and token count
                        token_prefix = f"[dim grey42]{provider_icon} · FREE · {tokens:,}t[/dim grey42] "
                    else:
                        # Cloud providers - show icon, cost, and token count
                        token_prefix = f"[dim grey42]{provider_icon} · ${cost:.4f} · {tokens:,}t[/dim grey42] "
            except Exception:
                pass

            # Prompt with token costs on the left
            user_input = get_user_input_with_editing(
                f"\n{token_prefix}[bold #e45c47]❯[/bold #e45c47] ", client
            )

            # Skip processing if input is empty or just whitespace
            if not user_input.strip():
                continue

            # Handle commands
            if user_input.startswith("/"):
                handle_command(user_input, client)
                continue

            # Check if user forgot the slash for a command
            potential_command = check_for_missing_slash_command(user_input)
            if potential_command:
                if Confirm.ask(
                    f"[yellow]Did you mean '{potential_command}'?[/yellow]",
                    default=True,
                ):
                    # Replace first word with the slash command
                    words = user_input.split()
                    words[0] = potential_command
                    corrected_input = " ".join(words)
                    handle_command(corrected_input, client)
                    continue

            # Check if it's a shell command first
            if execute_shell_command(user_input):
                continue

            # Determine if streaming is appropriate
            # (disable when verbose/reasoning callbacks are handling output)
            use_streaming = stream and should_show_progress(client)

            if use_streaming:
                result = _display_streaming_response(client, user_input, console)

                # Show final markdown-rendered response
                if result["success"] and result.get("response"):
                    agent_name = result.get("last_agent", "supervisor")
                    agent_display = (
                        agent_name.replace("_", " ").title()
                        if agent_name and agent_name != "__end__"
                        else "Lobster"
                    )
                    console.print(f"\n[dim]◀ {agent_display}[/dim]")
                    console.print(Markdown(result["response"]))
            else:
                # Existing non-streaming path
                if should_show_progress(client):
                    console.print("[dim]...[/dim]", end="", flush=True)

                result = client.query(user_input, stream=False)

                if should_show_progress(client):
                    console.print("\r   \r", end="", flush=True)

                # Display response
                if result["success"]:
                    agent_name = result.get("last_agent", "supervisor")
                    agent_display = (
                        agent_name.replace("_", " ").title()
                        if agent_name and agent_name != "__end__"
                        else "Lobster"
                    )
                    console.print(f"\n[dim]◀ {agent_display}[/dim]")
                    console.print(Markdown(result["response"]))

            # Plots indicator (for both streaming and non-streaming)
            if result.get("success") and result.get("plots"):
                console.print(
                    f"[dim #e45c47]◆ {len(result['plots'])} plot(s)[/dim #e45c47]"
                )

            if not result.get("success"):
                console.print(f"\n[red]✗[/red] {result.get('error', 'Unknown error')}")

            _maybe_print_timings(client, "Chat Query")

        except KeyboardInterrupt:
            if Confirm.ask("\n[dim]exit?[/dim]"):
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
                    if verbose:
                        summary = client.token_tracker.get_verbose_summary()
                    else:
                        summary = client.token_tracker.get_minimal_summary()
                    if summary:
                        console.print(f"[dim]{summary}[/dim]")
                console.print()  # Empty line before exit
                break
            continue
        except Exception as e:
            # Enhanced error reporting with context
            error_message = str(e)
            error_type = type(e).__name__

            # Provide context-aware suggestions
            suggestions = {
                "FileNotFoundError": "Check if the file path is correct and the file exists",
                "PermissionError": "Check file permissions or run with appropriate privileges",
                "ConnectionError": "Check your internet connection and API keys",
                "TimeoutError": "The operation timed out. Try again or check your connection",
                "ImportError": "Required dependency missing. Try reinstalling the package",
                "ValueError": "Invalid input provided. Check your command syntax",
                "KeyError": "Missing configuration or data. Check your setup",
            }

            suggestion = suggestions.get(
                error_type, "Check the error details and try again"
            )

            console_manager.print_error_panel(
                f"{error_type}: {error_message}", suggestion
            )

            # In debug mode, also print the full traceback
            if debug:
                console_manager.error_console.print_exception(
                    width=None,
                    extra_lines=3,
                    theme="monokai",
                    word_wrap=True,
                    show_locals=True,
                )



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



