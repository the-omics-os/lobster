"""
Agent management subcommands for Lobster CLI.

Provides commands to discover, inspect, install, and manage agent packages
through the command line.

Commands:
    lobster agents list      - Display installed agents in flat sorted table
    lobster agents info      - Show concise card with agent details
    lobster agents install   - Install package via uv pip, auto-enable agents
    lobster agents uninstall - Remove package via uv pip, remove from config

Example:
    >>> # List all installed agents
    >>> lobster agents list
    >>>
    >>> # Show info for a specific agent
    >>> lobster agents info research_agent
    >>>
    >>> # Install a new agent package
    >>> lobster agents install lobster-transcriptomics
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

logger = logging.getLogger(__name__)

# Create the agents subcommand group
agents_app = typer.Typer(
    name="agents",
    help="Manage agent packages and composition",
    no_args_is_help=True,
)

# Console for Rich output
console = Console()


def _find_local_package(package: str) -> str | None:
    """
    Check if package exists locally in packages/ folder.

    Args:
        package: Package name (e.g., "lobster-transcriptomics")

    Returns:
        Path to local package if found, None otherwise
    """
    # Find the lobster repo root (where packages/ lives)
    # Walk up from this file to find pyproject.toml with [tool.uv.workspace]
    current = Path(__file__).resolve()
    for parent in current.parents:
        packages_dir = parent / "packages"
        if packages_dir.exists():
            # Check for the package
            local_pkg = packages_dir / package
            if local_pkg.exists() and (local_pkg / "pyproject.toml").exists():
                return str(local_pkg)
    return None


def _uv_pip_install(package: str) -> tuple[bool, str]:
    """
    Install a package via uv pip install (or pip as fallback).

    First checks if the package exists locally in packages/ folder.
    If found, installs from local path (editable). Otherwise installs from PyPI.

    In a uv tool environment, ``uv pip install`` installs into a venv that
    gets wiped on next ``uv tool install``/``upgrade``.  Fail with a
    helpful message so the caller can surface the right command instead.

    Args:
        package: Package name to install (e.g., "lobster-transcriptomics")

    Returns:
        Tuple of (success: bool, message: str)
    """
    import shutil

    from lobster.core.uv_tool_env import is_uv_tool_env

    if is_uv_tool_env():
        return (
            False,
            "Cannot install packages directly in a uv tool environment. "
            "Use: uv tool install lobster-ai --with <package>",
        )

    # Check for local package first (development mode)
    local_path = _find_local_package(package)

    # Build the install arguments
    if local_path:
        install_args = ["-e", local_path]
    else:
        install_args = [package]

    # Try uv first (as CLI tool), then fall back to pip
    uv_path = shutil.which("uv")
    if uv_path:
        cmd = [uv_path, "pip", "install"] + install_args
    else:
        # Fall back to pip
        cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + install_args

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    except FileNotFoundError:
        return False, "Neither uv nor pip found"


def _uv_pip_uninstall(package: str) -> tuple[bool, str]:
    """
    Uninstall a package via uv pip uninstall (or pip as fallback).

    Args:
        package: Package name to uninstall

    Returns:
        Tuple of (success: bool, message: str)
    """
    import shutil

    # Try uv first (as CLI tool), then fall back to pip
    uv_path = shutil.which("uv")
    if uv_path:
        cmd = [uv_path, "pip", "uninstall", package, "-y"]
    else:
        # Fall back to pip
        cmd = [sys.executable, "-m", "pip", "uninstall", package, "-y"]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    except FileNotFoundError:
        return False, "Neither uv nor pip found"


def _get_agents_for_package(package_name: str) -> list[str]:
    """
    Get list of agent names provided by a package.

    Filters agents from ComponentRegistry by their package_name field.

    Args:
        package_name: Package name to filter by

    Returns:
        List of agent names from the specified package
    """
    from lobster.core.component_registry import component_registry

    agents = component_registry.list_agents()
    matching_agents = []

    for name, config in agents.items():
        agent_package = getattr(config, "package_name", None)
        # Normalize package name (lobster-foo -> lobster_foo for comparison)
        if agent_package:
            normalized = agent_package.replace("-", "_")
            package_normalized = package_name.replace("-", "_")
            if normalized == package_normalized:
                matching_agents.append(name)
        elif package_name in ("lobster-ai", "lobster_ai"):
            # Core agents have no package_name (None) - they belong to lobster-ai
            if agent_package is None:
                matching_agents.append(name)

    return matching_agents


def _get_workspace_path() -> Path:
    """Get the current workspace path."""
    from lobster.core.workspace import resolve_workspace

    return resolve_workspace()


@agents_app.command("list")
def agents_list(
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Path to workspace directory",
    ),
) -> None:
    """
    List installed agents with status.

    Displays a flat sorted table with columns: Name, Package, Tier, Enabled.
    Shows only locally installed agents (entry point discovery, no network).

    CLI-03: lobster agents list shows available agents with install status and tier
    """
    from lobster.core.component_registry import component_registry
    from lobster.config.workspace_agent_config import WorkspaceAgentConfig

    # Resolve workspace
    workspace_path = workspace or _get_workspace_path()

    # Load current config to check enabled status
    config = WorkspaceAgentConfig.load(workspace_path)
    enabled_set = set(config.enabled_agents)

    # Get all installed agents (entry points only - no network)
    all_agents = component_registry.list_agents()

    if not all_agents:
        console.print("[yellow]No agents installed.[/yellow]")
        console.print(
            "[dim]Install agents with: lobster agents install <package>[/dim]"
        )
        return

    # Build table
    table = Table(
        title="Installed Agents",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
    )
    table.add_column("Name", style="cyan")
    table.add_column("Package", style="white")
    table.add_column("Tier", style="yellow")
    table.add_column("Enabled", style="green", justify="center")

    # Sort agents alphabetically by name
    for name in sorted(all_agents.keys()):
        agent_config = all_agents[name]

        # Extract tier (default to 'free')
        tier = getattr(agent_config, "tier_requirement", "free") or "free"

        # Extract package name (default to 'lobster-ai' for core agents)
        package = getattr(agent_config, "package_name", None) or "lobster-ai"

        # Check if enabled
        enabled = "[green]v[/green]" if name in enabled_set else ""

        table.add_row(name, package, tier, enabled)

    console.print(table)
    console.print(f"\n[dim]{len(all_agents)} agents installed[/dim]")


@agents_app.command("info")
def agents_info(
    agent_name: str = typer.Argument(..., help="Agent name to inspect"),
) -> None:
    """
    Show detailed agent information.

    Displays a concise card (5-8 lines) with:
    - Display name (bold cyan)
    - Description (dim)
    - Tier (yellow label)
    - Package name
    - Service dependencies
    - Version if available

    CLI-05: lobster agents info <agent> shows agent details
    """
    from lobster.core.component_registry import component_registry

    # Get agent config
    agent_config = component_registry.get_agent(agent_name)

    if not agent_config:
        console.print(f"[red]Agent '{agent_name}' not found.[/red]")
        console.print("[dim]Use 'lobster agents list' to see available agents.[/dim]")
        raise typer.Exit(1)

    # Extract agent properties
    display_name = getattr(agent_config, "display_name", agent_name)
    description = getattr(agent_config, "description", "No description available")
    tier = getattr(agent_config, "tier_requirement", "free") or "free"
    package = getattr(agent_config, "package_name", None) or "lobster-ai"
    service_deps = getattr(agent_config, "service_dependencies", []) or []

    # Format service dependencies
    deps_str = ", ".join(service_deps) if service_deps else "none"

    # Build concise card (5-8 lines)
    card_content = (
        f"[bold cyan]{display_name}[/bold cyan]\n"
        f"[dim]{description}[/dim]\n\n"
        f"[yellow]Tier:[/yellow] {tier}\n"
        f"[yellow]Package:[/yellow] {package}\n"
        f"[yellow]Dependencies:[/yellow] {deps_str}"
    )

    panel = Panel.fit(
        card_content,
        title=f"Agent: {agent_name}",
        border_style="cyan",
    )
    console.print(panel)


@agents_app.command("install")
def agents_install(
    package_name: str = typer.Argument(..., help="Package name to install"),
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Path to workspace directory",
    ),
) -> None:
    """
    Install an agent package and auto-enable its agents.

    Installs the package via uv pip install, then automatically enables
    all agents from the package in the workspace config.toml.

    CLI-04: lobster agents install <package> installs via uv pip
    """
    from lobster.core.component_registry import component_registry
    from lobster.config.workspace_agent_config import WorkspaceAgentConfig

    # Resolve workspace
    workspace_path = workspace or _get_workspace_path()

    console.print(f"[cyan]Installing {package_name}...[/cyan]")

    # Get agents before install for comparison
    component_registry.reset()  # Force reload after install
    agents_before = set(component_registry.list_agents().keys())

    # Install package with spinner
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Installing {package_name}...", total=None)
        success, message = _uv_pip_install(package_name)
        progress.remove_task(task)

    if not success:
        console.print("[red]Installation failed:[/red]")
        console.print(f"[dim]{message}[/dim]")
        raise typer.Exit(1)

    console.print("[green]v[/green] Package installed successfully")

    # Reload component registry to discover new agents
    component_registry.reset()
    agents_after = set(component_registry.list_agents().keys())

    # Find new agents
    new_agents = agents_after - agents_before

    if not new_agents:
        # No new agents discovered - try by package name
        new_agents = set(_get_agents_for_package(package_name))

    if new_agents:
        # Auto-enable new agents in config
        config = WorkspaceAgentConfig.load(workspace_path)

        # Add new agents to enabled list
        current_enabled = set(config.enabled_agents)
        updated_enabled = current_enabled | new_agents

        config.enabled_agents = sorted(updated_enabled)
        config.save(workspace_path)

        console.print(f"[green]v[/green] Enabled {len(new_agents)} agent(s):")
        for agent in sorted(new_agents):
            console.print(f"  [cyan]{agent}[/cyan]")
    else:
        console.print("[yellow]No new agents detected from package.[/yellow]")


@agents_app.command("uninstall")
def agents_uninstall(
    package_name: str = typer.Argument(..., help="Package name to uninstall"),
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Path to workspace directory",
    ),
) -> None:
    """
    Uninstall an agent package and remove its agents from config.

    Removes the package via uv pip uninstall, then removes all agents
    from the package from the workspace config.toml enabled list.
    """
    from lobster.core.component_registry import component_registry
    from lobster.config.workspace_agent_config import WorkspaceAgentConfig

    # Resolve workspace
    workspace_path = workspace or _get_workspace_path()

    # Get agents from this package before uninstall
    package_agents = set(_get_agents_for_package(package_name))

    console.print(f"[cyan]Uninstalling {package_name}...[/cyan]")

    # Uninstall package with spinner
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Uninstalling {package_name}...", total=None)
        success, message = _uv_pip_uninstall(package_name)
        progress.remove_task(task)

    if not success:
        console.print("[red]Uninstallation failed:[/red]")
        console.print(f"[dim]{message}[/dim]")
        raise typer.Exit(1)

    console.print("[green]v[/green] Package uninstalled successfully")

    # Remove package's agents from config
    if package_agents:
        config = WorkspaceAgentConfig.load(workspace_path)

        # Remove agents from enabled list
        current_enabled = set(config.enabled_agents)
        updated_enabled = current_enabled - package_agents

        config.enabled_agents = sorted(updated_enabled)
        config.save(workspace_path)

        console.print(
            f"[green]v[/green] Removed {len(package_agents)} agent(s) from config:"
        )
        for agent in sorted(package_agents):
            console.print(f"  [dim]{agent}[/dim]")

    # Reset registry to reflect changes
    component_registry.reset()
