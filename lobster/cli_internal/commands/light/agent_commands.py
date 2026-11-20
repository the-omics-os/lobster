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
import subprocess  # nosec B404 # runs the package installer with an argv list; no shell
import sys
from pathlib import Path
from typing import NoReturn, Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from lobster.i18n import h, t

logger = logging.getLogger(__name__)

# Packages whose install failed in this process; `lobster init` reports them.
FAILED_INSTALLS: list[str] = []

# Create the agents subcommand group
agents_app = typer.Typer(
    name="agents",
    help=t("light.agents.help"),
    no_args_is_help=True,
)

# Console for Rich output
console = Console()


@agents_app.callback(invoke_without_command=True)
def _agents_callback(
    ctx: typer.Context,
    stdio: bool = typer.Option(False, "--stdio", hidden=True),
):
    # `lobster agents --stdio` is a likely typo for the machine interface.
    if stdio:
        typer.echo(
            h(
                "`lobster agents` manages agent packages; the machine interface is `lobster agent --stdio`."
            ),
            err=True,
        )
        raise typer.Exit(2)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(0)


def agent_command(
    stdio: bool = typer.Option(
        False, "--stdio", help=h("Speak protocol v1 NDJSON on stdin and stdout.")
    ),
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help=h(
            "Workspace directory. Can also be set via LOBSTER_WORKSPACE env var. Default: ./.lobster_workspace"
        ),
    ),
    session_id: Optional[str] = typer.Option(
        None,
        "--session",
        "--session-id",
        "-s",
        help=h(
            "Session ID to continue (use 'latest' for most recent session in workspace)"
        ),
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help=h(
            "LLM provider to use (anthropic, bedrock, ollama, gemini, azure, openai, openrouter, nebius, ollama-cloud, omics-os). Overrides auto-detection."
        ),
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help=h(
            "Model to use (e.g., claude-4-sonnet, llama3:70b-instruct, mixtral:8x7b). Overrides configuration."
        ),
    ),
):
    if not stdio:
        print(
            h(
                "`lobster agent` is a machine interface; run it with --stdio. For agent management see `lobster agents`."
            ),
            file=sys.stderr,
        )
        raise typer.Exit(2)
    from lobster.cli_internal.agent_stdio import run_agent_stdio

    raise typer.Exit(
        run_agent_stdio(
            workspace=str(workspace) if workspace else None,
            session_id=session_id,
            provider=provider,
            model=model,
        )
    )


agent_command.__doc__ = h(
    "Run Lobster as a machine interface for other programs (NDJSON on stdin and stdout)."
)


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


def _pinned_to_lobster(package: str, op: str = "~=") -> str:
    """Pin an official agent package to this Lobster's minor version (1.3.x).

    Unpinned, a release candidate picked the previous minor from PyPI. Under
    uv tool use op=">=": the receipt keeps the pin, and ~= would stop
    `uv tool upgrade`.
    """
    from lobster.cli_internal.agent_packages import AVAILABLE_AGENT_PACKAGES
    from lobster.core.uv_tool_env import lobster_release

    if package not in {entry[0] for entry in AVAILABLE_AGENT_PACKAGES}:
        return package
    major, minor = lobster_release()[:2]
    return f"{package}{op}{major}.{minor}.0"


def _install_into_uv_tool(package: str, workspace_path: Path) -> NoReturn:
    """Add a package to this uv tool's receipt, enable its agents, and exit.

    The install replaces the tool's venv, so the agents (from the package
    catalog) are enabled once it worked and the process ends after it.
    """
    import shlex

    from lobster.cli_internal.agent_packages import AVAILABLE_AGENT_PACKAGES
    from lobster.core.uv_tool_env import (
        build_tool_install_command,
        run_tool_install,
        windows_locks_running_tool,
    )

    agents = {
        a for entry in AVAILABLE_AGENT_PACKAGES if entry[0] == package for a in entry[2]
    }
    config, said = (
        _changed_config(workspace_path, add=agents) if agents else (None, None)
    )
    after = None
    if config is not None:
        if windows_locks_running_tool():  # the install is left to the user
            if said:
                console.print(said)
            config.save(workspace_path)
        else:
            after = _save_after_install(config, workspace_path, said)

    cmd = build_tool_install_command(with_packages=[_pinned_to_lobster(package, ">=")])
    cmd.insert(3, "--reinstall")
    if not windows_locks_running_tool():  # else run_tool_install prints the command
        console.print(f"  {shlex.join(cmd)}", markup=False)
        if console.file:
            console.file.flush()
    run_tool_install(cmd, after=after)


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
            h(
                "Cannot install packages directly in a uv tool environment. "
                "Use: uv tool install lobster-ai --with <package>"
            ),
        )

    # Check for local package first (development mode)
    local_path = _find_local_package(package)

    # Build the install arguments
    if local_path:
        install_args = ["-e", local_path]
    else:
        install_args = [_pinned_to_lobster(package)]

    # Try uv first (as CLI tool), then fall back to pip. Both install into the
    # Python running Lobster, so no activated venv is needed.
    uv_path = shutil.which("uv")
    if uv_path:
        cmd = [uv_path, "pip", "install", "--python", sys.executable] + install_args
    else:
        # Fall back to pip
        cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + install_args

    try:
        result = subprocess.run(  # nosec B603 # argv list: uv/pip install of the named package; no shell
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        FAILED_INSTALLS.append(package)
        return False, e.stderr
    except FileNotFoundError:
        FAILED_INSTALLS.append(package)
        return False, h("Neither uv nor pip found")


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
        cmd = [uv_path, "pip", "uninstall", "--python", sys.executable, package, "-y"]
    else:
        # Fall back to pip
        cmd = [sys.executable, "-m", "pip", "uninstall", package, "-y"]

    try:
        result = subprocess.run(  # nosec B603 # argv list: uv/pip uninstall of the named package; no shell
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    except FileNotFoundError:
        return False, h("Neither uv nor pip found")


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


def _with_children(names: set) -> set:
    """*names* and their child agents, which the graph adds for them."""
    from lobster.core.component_registry import component_registry

    agents = component_registry.list_agents()
    present = set(names)
    for name in names:
        present.update(getattr(agents.get(name), "child_agents", None) or ())
    return present


def _changed_config(workspace_path: Path, *, add=(), remove=()):
    """(the agent config with agents added or removed, the line that says a
    preset became a list or None); (None, None) if nothing changes.

    config.toml holds a preset or a list, never both: a list saved next to a
    preset is dropped, so a preset becomes the explicit list it changes.
    """
    from lobster.config.workspace_agent_config import WorkspaceAgentConfig

    config = WorkspaceAgentConfig.load(workspace_path)
    current = set(config.effective_agents())
    present = _with_children(current)
    if set(add) <= present and not set(remove) & present:
        return None, None  # nothing to change: a preset stays a preset
    said = None
    if config.preset:
        said = h(
            "[dim]Preset '{preset_name}' will be saved as explicit agents instead.[/dim]"
        ).format(preset_name=config.preset)
    config.enabled_agents = sorted((current | set(add)) - set(remove))
    config.preset = None
    return config, said


def _change_enabled(workspace_path: Path, *, add=(), remove=()) -> None:
    """Add or remove agents in config.toml (see ``_changed_config``)."""
    config, said = _changed_config(workspace_path, add=add, remove=remove)
    if said:
        console.print(said)
    if config is not None:
        config.save(workspace_path)


def _save_after_install(config, workspace_path: Path, said: Optional[str]):
    """Stage *config* now; the returned ``after`` puts it in place, and prints
    *said*, once the install worked. It runs after the tool's venv is
    replaced: builtins and ``os`` only."""
    import os
    import tempfile

    workspace_path.mkdir(parents=True, exist_ok=True)  # as config.save does
    folder = tempfile.mkdtemp(prefix=".config-", dir=workspace_path)
    staged = str(config.save(Path(folder)))
    final = os.path.join(str(workspace_path), os.path.basename(staged))
    plain = Text.from_markup(said).plain if said else None

    def after(ok: bool) -> None:
        try:
            if ok:
                os.replace(staged, final)
                if plain:
                    print(plain)
            else:
                os.unlink(staged)
            os.rmdir(folder)
        except OSError:
            pass  # no traceback over a stale venv

    return after


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
        help=h("Path to workspace directory"),
    ),
) -> None:
    """
    List installed agents with status.

    Displays a flat sorted table with columns: Name, Package, Tier, Enabled.
    Shows only locally installed agents (entry point discovery, no network).

    CLI-03: lobster agents list shows available agents with install status and tier
    """
    from lobster.config.workspace_agent_config import WorkspaceAgentConfig
    from lobster.core.component_registry import component_registry

    # Resolve workspace
    workspace_path = workspace or _get_workspace_path()

    # Load current config to check enabled status
    config = WorkspaceAgentConfig.load(workspace_path)
    enabled_set = set(config.effective_agents())

    # Get all installed agents (entry points only - no network)
    all_agents = component_registry.list_agents()

    if not all_agents:
        console.print(h("[yellow]No agents installed.[/yellow]"))
        console.print(
            h("[dim]Install agents with: lobster agents install <package>[/dim]")
        )
        return

    # Build table
    table = Table(
        title=h("Installed Agents"),
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
    )
    table.add_column(h("Name"), style="cyan")
    table.add_column(h("Package"), style="white")
    table.add_column(h("Tier"), style="yellow")
    table.add_column(h("Enabled"), style="green", justify="center")

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
    console.print(
        h("\n[dim]{count} agents installed[/dim]").format(count=len(all_agents))
    )


@agents_app.command("info")
def agents_info(
    agent_name: str = typer.Argument(..., help=h("Agent name to inspect")),
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
        console.print(f"[red]{t('light.agent.not_found', name=agent_name)}[/red]")
        console.print(f"[dim]{t('light.agent.list_hint')}[/dim]")
        raise typer.Exit(1)

    # Extract agent properties
    display_name = getattr(agent_config, "display_name", agent_name)
    description = getattr(agent_config, "description", h("No description available"))
    tier = getattr(agent_config, "tier_requirement", "free") or "free"
    package = getattr(agent_config, "package_name", None) or "lobster-ai"
    service_deps = getattr(agent_config, "service_dependencies", []) or []

    # Format service dependencies
    deps_str = ", ".join(service_deps) if service_deps else h("none")

    # Build concise card (5-8 lines)
    card_content = (
        f"[bold cyan]{display_name}[/bold cyan]\n"
        f"[dim]{description}[/dim]\n\n"
        f"[yellow]{h('Tier:')}[/yellow] {tier}\n"
        f"[yellow]{h('Package:')}[/yellow] {package}\n"
        f"[yellow]{h('Dependencies:')}[/yellow] {deps_str}"
    )

    panel = Panel.fit(
        card_content,
        title=f"{h('Agent')}: {agent_name}",
        border_style="cyan",
    )
    console.print(panel)


@agents_app.command("install")
def agents_install(
    package_name: str = typer.Argument(..., help=h("Package name to install")),
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help=h("Path to workspace directory"),
    ),
) -> None:
    """
    Install an agent package and auto-enable its agents.

    Installs the package via uv pip install, then automatically enables
    all agents from the package in the workspace config.toml.

    CLI-04: lobster agents install <package> installs via uv pip
    """
    from lobster.core.component_registry import component_registry

    # Resolve workspace
    workspace_path = workspace or _get_workspace_path()

    console.print(
        h("[cyan]Installing {package}...[/cyan]").format(package=package_name)
    )

    from lobster.core.uv_tool_env import is_uv_tool_env

    if is_uv_tool_env():
        _install_into_uv_tool(package_name, workspace_path)

    # Get agents before install for comparison
    component_registry.reset()  # Force reload after install
    agents_before = set(component_registry.list_agents().keys())

    # Install package with spinner
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            h("Installing {package}...").format(package=package_name), total=None
        )
        success, message = _uv_pip_install(package_name)
        progress.remove_task(task)

    if not success:
        console.print(f"[red]{t('light.agent.install_failed')}[/red]")
        console.print(f"[dim]{message}[/dim]")
        raise typer.Exit(1)

    console.print(h("[green]v[/green] Package installed successfully"))

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
        _change_enabled(workspace_path, add=new_agents)

        console.print(
            h("[green]v[/green] Enabled {count} agent(s):").format(
                count=len(new_agents)
            )
        )
        for agent in sorted(new_agents):
            console.print(f"  [cyan]{agent}[/cyan]")
    else:
        console.print(h("[yellow]No new agents detected from package.[/yellow]"))


@agents_app.command("uninstall")
def agents_uninstall(
    package_name: str = typer.Argument(..., help=h("Package name to uninstall")),
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help=h("Path to workspace directory"),
    ),
) -> None:
    """
    Uninstall an agent package and remove its agents from config.

    Removes the package via uv pip uninstall, then removes all agents
    from the package from the workspace config.toml enabled list.
    """
    from lobster.core.component_registry import component_registry

    # Resolve workspace
    workspace_path = workspace or _get_workspace_path()

    # Get agents from this package before uninstall
    package_agents = set(_get_agents_for_package(package_name))

    console.print(
        h("[cyan]Uninstalling {package}...[/cyan]").format(package=package_name)
    )

    # Uninstall package with spinner
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            h("Uninstalling {package}...").format(package=package_name), total=None
        )
        success, message = _uv_pip_uninstall(package_name)
        progress.remove_task(task)

    if not success:
        console.print(f"[red]{t('light.agent.uninstall_failed')}[/red]")
        console.print(f"[dim]{message}[/dim]")
        raise typer.Exit(1)

    console.print(h("[green]v[/green] Package uninstalled successfully"))

    # Remove package's agents from config
    if package_agents:
        _change_enabled(workspace_path, remove=package_agents)

        console.print(
            h("[green]v[/green] Removed {count} agent(s) from config:").format(
                count=len(package_agents)
            )
        )
        for agent in sorted(package_agents):
            console.print(f"  [dim]{agent}[/dim]")

    # Reset registry to reflect changes
    component_registry.reset()
