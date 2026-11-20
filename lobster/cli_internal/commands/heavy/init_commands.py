"""
Init command body and helper functions.

Extracted from cli.py -- the `lobster init` wizard and all supporting
helper functions for provider setup, agent selection, and package installation.
"""

import functools
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import typer
from rich.markup import escape as _markup_escape
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from lobster.cli_internal.agent_packages import AVAILABLE_AGENT_PACKAGES as _CATALOG
from lobster.cli_internal.agent_packages import DEFAULT_AGENT_PACKAGES
from lobster.config import provider_setup
from lobster.i18n import h
from lobster.ui import LobsterTheme
from lobster.ui.console_manager import get_console_manager

if TYPE_CHECKING:
    from lobster.core.client import AgentClient

logger = logging.getLogger(__name__)

console_manager = get_console_manager()
console = console_manager.console


def change_mode(new_mode: str, current_client: "AgentClient") -> "AgentClient":
    """
    Change the operation mode and reinitialize client with the new configuration.

    Args:
        new_mode: The new mode/profile to switch to
        current_client: The current AgentClient instance

    Returns:
        Updated AgentClient instance
    """
    global client

    # Store current settings before reinitializing
    current_workspace = Path(current_client.workspace_path)
    current_reasoning = current_client.enable_reasoning

    # Persist the new profile to workspace config
    from lobster.config.workspace_config import WorkspaceProviderConfig

    ws_config = WorkspaceProviderConfig.load(current_workspace)
    ws_config.profile = new_mode
    ws_config.save(current_workspace)

    # Reinitialize the client with the new profile settings
    from lobster.cli_internal.commands.heavy.session_infra import init_client

    client = init_client(workspace=current_workspace, reasoning=current_reasoning)

    return client


# =============================================================================
# Agent Selection Helper Functions (CLI-01, CLI-02, CONF-07, CONF-08)
# =============================================================================


# =============================================================================
# Available Agent Packages (static registry for init flow)
# =============================================================================
# These are the official lobster-* packages that can be installed.
# Format: (package_name, description, agents_provided, published_on_pypi, experimental)

AVAILABLE_AGENT_PACKAGES = [
    (name, h(description), agents, published, experimental)
    for name, description, agents, published, experimental in _CATALOG
]


def _get_init_package_agents_map() -> dict[str, list[str]]:
    """Return package -> agents mapping for init selection UIs."""
    return {
        pkg_name: list(agents) for pkg_name, _, agents, _, _ in AVAILABLE_AGENT_PACKAGES
    }


def _normalize_selected_agents(selected_agents: list[str]) -> list[str]:
    """Normalize mixed agent/package selections into canonical agent ids.

    Older Go/questionary init flows returned package ids such as
    ``lobster-research``. The workspace config and install prompt expect agent
    ids such as ``research_agent``. This helper accepts either shape and
    preserves order while deduplicating.
    """
    package_agents = _get_init_package_agents_map()
    normalized: list[str] = []
    seen: set[str] = set()

    for value in selected_agents or []:
        for agent in package_agents.get(value, [value]):
            agent = str(agent).strip()
            if not agent or agent in seen:
                continue
            normalized.append(agent)
            seen.add(agent)

    return normalized


def _get_tui_agent_package_choices() -> list[tuple[str, str, list[str]]]:
    """Return published init package choices for Go/questionary UIs."""
    choices: list[tuple[str, str, list[str]]] = []
    for (
        pkg_name,
        description,
        agents,
        published,
        _experimental,
    ) in AVAILABLE_AGENT_PACKAGES:
        if not published:
            continue
        choices.append((pkg_name, description, list(agents)))
    return choices


def _get_installed_agents() -> dict[str, any]:
    """Get dict of installed agents from ComponentRegistry."""
    from lobster.core.component_registry import component_registry

    return component_registry.list_agents()


def _display_agent_selection_list(
    agents_dict: dict, enabled_set: set = None
) -> list[str]:
    """Display numbered list of agents for selection. Returns list of agent names in display order."""
    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console()
    enabled_set = enabled_set or set()

    table = Table(
        title=h("Available Agents"),
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
    )
    table.add_column("#", style="cyan", width=4)
    table.add_column(h("Name"), style="white")
    table.add_column(h("Description"), style="dim")
    table.add_column(h("Tier"), style="yellow")

    # Sort agents alphabetically
    sorted_agents = sorted(agents_dict.keys())
    for i, name in enumerate(sorted_agents, 1):
        config = agents_dict[name]
        desc = getattr(config, "description", "")[:50]
        tier = getattr(config, "tier_requirement", "free") or "free"
        table.add_row(str(i), name, desc, tier)

    console.print(table)
    return sorted_agents


def _packages_of(agents) -> list[str]:
    """The published catalog packages that provide any of ``agents``, in order."""
    agents = set(agents)
    return [
        name
        for name, _, provided, published, _ in _CATALOG
        if published and agents & set(provided)
    ]


def _starting_packages(
    workspace_path: Optional[Path], preset: Optional[str] = None
) -> Optional[list[str]]:
    """The packages an init picker starts from; None: the defaults.

    ``--preset`` first, then the experts the workspace has saved (`init --force`).
    """
    from lobster.config.agent_presets import expand_preset
    from lobster.config.workspace_agent_config import (
        CONFIG_FILE_NAME,
        WorkspaceAgentConfig,
    )

    if preset:
        return _packages_of(expand_preset(preset) or []) or None
    if workspace_path is None or not (workspace_path / CONFIG_FILE_NAME).is_file():
        return None
    saved = WorkspaceAgentConfig.load(workspace_path)
    agents = set(saved.enabled_agents)
    if saved.preset:
        agents |= set(expand_preset(saved.preset) or [])
    return _packages_of(agents) or None


def _kept_preset(preset: Optional[str], agents: list[str]) -> Optional[str]:
    """``preset`` when the picked agents cover exactly its packages, else None.

    A picker that started from a preset and was changed says the preset is
    replaced by the picked experts.
    """
    from lobster.config.agent_presets import expand_preset

    if not preset or not agents:
        return None
    if set(_packages_of(agents)) == set(_packages_of(expand_preset(preset) or [])):
        return preset
    console.print(
        h(
            "[dim]Preset '{preset_name}' will be saved as explicit agents instead.[/dim]"
        ).format(preset_name=preset)
    )
    return None


def _check_preset_or_exit(preset_flag: str, out) -> None:
    from lobster.config.agent_presets import is_valid_preset

    if not is_valid_preset(preset_flag):
        out.print(
            h("[red]Invalid preset: {preset_flag}[/red]").format(
                preset_flag=preset_flag
            )
        )
        out.print(
            h("[dim]Valid presets: scrna-basic, scrna-full, multiomics-full[/dim]")
        )
        raise typer.Exit(1)


def _prompt_manual_agent_selection(
    workspace_path: Path, preset: Optional[str] = None
) -> list[str]:
    """Interactive agent package selection. Returns list of selected agent names."""
    from rich import box
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.table import Table

    console = Console()

    # Check which packages are already installed
    installed_agents = _get_installed_agents()
    installed_agent_names = set(installed_agents.keys())

    # Build table showing available packages
    table = Table(
        title=h("Available Agent Packages"),
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
    )
    table.add_column("#", style="cyan", width=3)
    table.add_column(h("Package"), style="white")
    table.add_column(h("Description"), style="dim")
    table.add_column(h("Agents"), style="yellow")
    table.add_column(h("Status"), style="green", width=12)

    for i, (pkg_name, desc, agents, published, experimental) in enumerate(
        AVAILABLE_AGENT_PACKAGES, 1
    ):
        # Check if this package's agents are installed
        is_installed = any(a in installed_agent_names for a in agents)
        if is_installed:
            status = h("[green]installed[/green]")
        elif not published:
            status = h("[dim]coming soon[/dim]")
        elif experimental:
            status = h("[yellow]experimental[/yellow]")
        else:
            status = ""
        agents_str = ", ".join(agents[:2])  # Show first 2 agents
        if len(agents) > 2:
            agents_str += f" +{len(agents) - 2}"
        table.add_row(str(i), pkg_name, desc, agents_str, status)

    console.print(table)

    console.print(
        h("\n[bold white]Select packages to install (comma-separated):[/bold white]")
    )
    console.print(h("[dim]Example: 1,2,3 or 1-4 or 'all' for all packages[/dim]"))
    console.print(h("[dim]Already installed packages will be skipped[/dim]"))

    ticked = _starting_packages(workspace_path, preset) or DEFAULT_AGENT_PACKAGES
    default = ",".join(
        str(i)
        for i, (name, *_) in enumerate(AVAILABLE_AGENT_PACKAGES, 1)
        if name in ticked
    )
    selection = Prompt.ask(h("[bold white]Selection[/bold white]"), default=default)

    # Parse selection
    selected_indices = set()
    if selection.strip().lower() == "all":
        selected_indices = {
            index
            for index, (_, _, _, published, _) in enumerate(AVAILABLE_AGENT_PACKAGES, 1)
            if published
        }
    else:
        for part in selection.split(","):
            part = part.strip()
            if "-" in part:
                try:
                    start, end = part.split("-")
                    for i in range(int(start), int(end) + 1):
                        if 1 <= i <= len(AVAILABLE_AGENT_PACKAGES):
                            selected_indices.add(i)
                except ValueError:
                    pass
            else:
                try:
                    idx = int(part)
                    if 1 <= idx <= len(AVAILABLE_AGENT_PACKAGES):
                        selected_indices.add(idx)
                except ValueError:
                    pass

    # Collect packages to install and agents to enable
    packages_to_install = []
    all_agents = []

    for idx in sorted(selected_indices):
        pkg_name, _, agents, published, _ = AVAILABLE_AGENT_PACKAGES[idx - 1]
        # Check if already installed
        is_installed = any(a in installed_agent_names for a in agents)
        if not is_installed and published:
            packages_to_install.append(pkg_name)
        if published:
            all_agents.extend(agents)

    from lobster.core.uv_tool_env import is_uv_tool_env

    # Install missing packages. Under uv tool a direct install cannot work; the
    # reinstall at the end of init adds them.
    if packages_to_install and not is_uv_tool_env():
        from lobster.cli_internal.commands.light.agent_commands import _uv_pip_install

        console.print(
            h("\n[dim]Installing {len} package(s)...[/dim]").format(
                len=len(packages_to_install)
            )
        )
        for pkg in packages_to_install:
            console.print(h("  [dim]Installing {pkg}...[/dim]").format(pkg=pkg))
            success, msg = _uv_pip_install(pkg)
            if success:
                console.print(f"  [green]✓[/green] {pkg}")
            else:
                console.print(
                    h("  [yellow]⚠[/yellow] {pkg} failed: {msg}").format(
                        pkg=pkg, msg=msg[:50]
                    )
                )
    elif selected_indices and not packages_to_install:
        console.print(h("[dim]All selected packages already installed[/dim]"))

    return list(dict.fromkeys(all_agents))  # Remove duplicates preserving order


def _prompt_automatic_agent_selection(workspace_path: Path) -> list[str]:
    """LLM-assisted automatic agent selection. Returns list of selected agent names."""
    from rich.console import Console
    from rich.prompt import Confirm, Prompt

    console = Console()

    console.print(
        h("\n[bold white]Describe your workflow and data types:[/bold white]")
    )
    console.print(
        h(
            "[dim]Example: I have single-cell RNA-seq data and want to identify cell types and run differential expression[/dim]"
        )
    )

    description = Prompt.ask(h("[bold white]Workflow description[/bold white]"))

    if not description.strip():
        console.print(
            h(
                "[yellow]No description provided. Falling back to manual selection.[/yellow]"
            )
        )
        return _prompt_manual_agent_selection(workspace_path)

    # Call Omics-OS endpoint
    console.print(
        h("\n[dim]Contacting Omics-OS service for agent suggestions...[/dim]")
    )

    try:
        from lobster.config.agent_config_endpoint import suggest_agents

        result = suggest_agents(description)
    except ImportError:
        result = None

    if result is None:
        console.print(
            h(
                "[yellow]Can't reach Omics-OS service. Falling back to manual selection.[/yellow]"
            )
        )
        return _prompt_manual_agent_selection(workspace_path)

    # Show suggestions with reasoning
    suggested_agents = result.get("agents", [])
    reasoning = result.get("reasoning", {})

    if not suggested_agents:
        console.print(
            h("[yellow]No agents suggested. Falling back to manual selection.[/yellow]")
        )
        return _prompt_manual_agent_selection(workspace_path)

    console.print(h("\n[bold green]Suggested agents:[/bold green]"))
    for agent in suggested_agents:
        reason = reasoning.get(agent, "")
        console.print(f"  [cyan]v[/cyan] {agent}")
        if reason:
            console.print(f"      [dim]{reason}[/dim]")

    # Allow user to modify
    if not Confirm.ask(
        h("\n[bold white]Accept these suggestions?[/bold white]"), default=True
    ):
        console.print(h("[dim]Opening manual selection to modify...[/dim]"))
        return _prompt_manual_agent_selection(workspace_path)

    return suggested_agents


def _check_and_prompt_install_packages(
    selected_agents: list[str],
    workspace_path: Path,
    *,
    prompt_for_install: bool = True,
    auto_install_missing: bool = False,
) -> list[str]:
    """Check if selected agents are installed, prompt to install missing packages.

    In a uv tool environment, package installation is deferred to the end-of-init
    handoff (``_uv_tool_env_handoff``).  We return the full selected list so the
    handoff can build the correct ``uv tool install`` command.
    """
    from rich.console import Console
    from rich.prompt import Confirm

    from lobster.core.uv_tool_env import is_uv_tool_env

    console = Console()

    normalized_agents = _normalize_selected_agents(selected_agents)

    agents_dict = _get_installed_agents()
    installed_names = set(agents_dict.keys())
    confirmed_names = set(installed_names)

    # Find agents that need installation
    missing = [a for a in normalized_agents if a not in installed_names]

    if not missing:
        return normalized_agents  # All agents installed

    # In uv tool env, defer installation — return full list for handoff
    if is_uv_tool_env():
        console.print(
            h(
                "\n[dim]  Agent packages will be added via uv tool install after setup completes.[/dim]"
            )
        )
        return normalized_agents

    console.print(h("\n[yellow]These agent packages are not installed:[/yellow]"))

    # Map agents to packages using centralized mapping
    from lobster.core.component_registry import AGENT_TO_PACKAGE

    packages_to_install: dict[str, list[str]] = {}
    unmapped_agents: list[str] = []
    for agent in missing:
        pkg = AGENT_TO_PACKAGE.get(agent)
        if pkg:
            packages_to_install.setdefault(pkg, []).append(agent)
        else:
            unmapped_agents.append(agent)

    for pkg, agents in sorted(packages_to_install.items()):
        console.print(f"  [dim]- {pkg}[/dim] [dim]({', '.join(agents)})[/dim]")
    for agent in unmapped_agents:
        console.print(f"  [dim]- {agent}[/dim]")

    if packages_to_install:
        console.print(h("\n[bold white]Install these packages?[/bold white]"))
        for pkg in sorted(packages_to_install):
            console.print(f"  [cyan]{pkg}[/cyan]")

        should_install = auto_install_missing
        if prompt_for_install and not auto_install_missing:
            should_install = Confirm.ask(
                h("\n[bold white]Install now?[/bold white]"), default=True
            )

        if should_install:
            from lobster.cli_internal.commands.light.agent_commands import (
                _uv_pip_install,
            )

            for pkg in sorted(packages_to_install):
                console.print(h("[dim]Installing {pkg}...[/dim]").format(pkg=pkg))
                success, msg = _uv_pip_install(pkg)
                if success:
                    console.print(h("[green]v[/green] {pkg} installed").format(pkg=pkg))
                    confirmed_names.update(packages_to_install[pkg])
                else:
                    console.print(
                        h(
                            "[yellow]Warning: Failed to install {pkg}: {msg}[/yellow]"
                        ).format(pkg=pkg, msg=msg)
                    )
        elif prompt_for_install:
            console.print(h("[dim]Skipping package installation for now[/dim]"))
        else:
            console.print(
                h("[dim]Skipping package installation in non-interactive mode[/dim]")
            )

    unresolved = [agent for agent in normalized_agents if agent not in confirmed_names]
    if unresolved:
        console.print(
            h("\n[yellow]Skipping uninstalled agents:[/yellow] ")
            + ", ".join(unresolved)
        )

    # Avoid immediate same-process registry reload here. In development installs we
    # often install editable packages from packages/, and the running interpreter
    # does not reliably observe their new import hooks until the next process.
    return [a for a in normalized_agents if a in confirmed_names]


def _save_agent_config(
    selected_agents: list[str], workspace_path: Path, preset_name: str = None
) -> None:
    """Save selected agents to WorkspaceAgentConfig."""
    from lobster.config.workspace_agent_config import WorkspaceAgentConfig

    config = WorkspaceAgentConfig(
        enabled_agents=selected_agents,
        preset=preset_name,
    )
    config.save(workspace_path)


def _perform_agent_selection_non_interactive(
    agents_flag: str,
    preset_flag: str,
    auto_agents_flag: bool,
    agents_description_flag: str,
    workspace_path: Path,
) -> tuple[list[str], str]:
    """Handle non-interactive agent selection. Returns (selected_agents, preset_name or None)."""
    from rich.console import Console

    from lobster.config.agent_presets import expand_preset

    console = Console()

    # Priority: preset > agents > auto_agents
    if preset_flag:
        _check_preset_or_exit(preset_flag, console)

        expanded = expand_preset(preset_flag)
        expanded = _filter_unpublished_preset_agents(expanded, console, preset_flag)
        console.print(
            h("[green]v[/green] Using preset: {preset_flag} ({len} agents)").format(
                preset_flag=preset_flag, len=len(expanded)
            )
        )
        return expanded, preset_flag

    elif agents_flag:
        # Parse comma-separated list
        agents = [a.strip() for a in agents_flag.split(",") if a.strip()]
        if not agents:
            console.print(h("[red]No agents specified in --agents flag[/red]"))
            raise typer.Exit(1)

        console.print(
            h("[green]v[/green] Enabling {len} agent(s): {join}").format(
                len=len(agents), join=", ".join(agents)
            )
        )
        return agents, None

    elif auto_agents_flag:
        if not agents_description_flag:
            console.print(h("[red]--auto-agents requires --agents-description[/red]"))
            raise typer.Exit(1)

        console.print(
            h("[dim]Contacting Omics-OS service for agent suggestions...[/dim]")
        )

        try:
            from lobster.config.agent_config_endpoint import suggest_agents

            result = suggest_agents(agents_description_flag)
        except ImportError:
            result = None

        if result is None:
            console.print(h("[yellow]Can't reach Omics-OS service.[/yellow]"))
            console.print(
                h(
                    "[yellow]Use --agents or --preset instead for non-interactive mode.[/yellow]"
                )
            )
            raise typer.Exit(1)

        suggested = result.get("agents", [])
        if not suggested:
            console.print(
                h(
                    "[yellow]No agents suggested. Use --agents or --preset instead.[/yellow]"
                )
            )
            raise typer.Exit(1)

        console.print(
            h("[green]v[/green] Auto-selected {len} agents: {join}").format(
                len=len(suggested), join=", ".join(suggested)
            )
        )
        return suggested, None

    # No agent selection flags - use all installed agents (default behavior)
    agents_dict = _get_installed_agents()
    all_agents = list(agents_dict.keys())
    console.print(
        h(
            "[dim]Using all {len} installed agents (no agent selection flags provided)[/dim]"
        ).format(len=len(all_agents))
    )
    return all_agents, None


def _filter_unpublished_preset_agents(
    selected_agents: list[str] | None,
    console,
    preset_name: str,
) -> list[str]:
    """Drop preset agents whose packages are not published on public PyPI."""
    from lobster.core.component_registry import AGENT_TO_PACKAGE

    published_packages = {
        pkg_name
        for pkg_name, _, _, published, _ in AVAILABLE_AGENT_PACKAGES
        if published
    }

    filtered: list[str] = []
    skipped: list[str] = []
    for agent in _normalize_selected_agents(selected_agents or []):
        package_name = AGENT_TO_PACKAGE.get(agent)
        if package_name and package_name not in published_packages:
            skipped.append(agent)
            continue
        filtered.append(agent)

    if skipped:
        console.print(
            h(
                "[yellow]⚠️  Preset includes agents not published on public PyPI; skipping for this install surface: {join}[/yellow]"
            ).format(join=", ".join(skipped))
        )
        console.print(
            h(
                "[dim]Preset '{preset_name}' will be saved as explicit agents instead.[/dim]"
            ).format(preset_name=preset_name)
        )

    return filtered


def _build_uv_tool_init_command(
    provider_name: str | None,
    selected_agents: list[str] | None,
    *,
    include_vector_search: bool = False,
    include_lamin: bool = False,
) -> list[str]:
    """Build the uv tool install command needed after init."""
    import importlib
    import importlib.util

    from lobster.cli_internal.commands.light.agent_commands import _pinned_to_lobster
    from lobster.core.component_registry import AGENT_TO_PACKAGE
    from lobster.core.uv_tool_env import build_tool_install_command, detect_uv_tool_env

    info = detect_uv_tool_env()

    extras: list[str] = []
    if provider_name:
        module_name = _PROVIDER_IMPORT_NAMES.get(provider_name)
        if module_name:
            try:
                importlib.import_module(module_name)
            except ImportError:
                extras.append(
                    "openai" if provider_name == "ollama-cloud" else provider_name
                )

    if include_vector_search:
        try:
            importlib.import_module("chromadb")
        except ImportError:
            extras.append("vector-search")

    if include_lamin and importlib.util.find_spec("lamindb_setup") is None:
        extras.append("lamin")

    published_packages = {
        pkg_name
        for pkg_name, _, _, published, _ in AVAILABLE_AGENT_PACKAGES
        if published
    }
    installed_packages = set()
    if info:
        installed_packages = {
            pkg.lower().replace("-", "_") for pkg in info.installed_packages
        }

    with_packages: list[str] = []
    for agent in _normalize_selected_agents(selected_agents or []):
        package_name = AGENT_TO_PACKAGE.get(agent)
        if not package_name or package_name not in published_packages:
            continue
        if package_name.lower().replace("-", "_") in installed_packages:
            continue
        requirement = _pinned_to_lobster(package_name, ">=")
        if requirement not in with_packages:
            with_packages.append(requirement)

    return build_tool_install_command(
        extras=extras or None,
        with_packages=with_packages or None,
    )


def _print_deferred_agent_install(selected_agents: list[str]) -> None:
    """Record the choice without installs; show a command valid for this environment."""
    from lobster.core.component_registry import AGENT_TO_PACKAGE
    from lobster.core.uv_tool_env import command_line, is_uv_tool_env

    packages = sorted(
        {AGENT_TO_PACKAGE[a] for a in selected_agents if a in AGENT_TO_PACKAGE}
    )
    if not packages:
        return
    from lobster.cli_internal.commands.light.agent_commands import _pinned_to_lobster

    if is_uv_tool_env():
        from lobster.core.uv_tool_env import build_tool_install_command

        command = build_tool_install_command(
            with_packages=[_pinned_to_lobster(p, ">=") for p in packages]
        )
        command[0] = "uv"  # a quoted path first is an expression in PowerShell
        command.insert(3, "--reinstall")
    else:
        pinned = [_pinned_to_lobster(p) for p in packages]
        command = ["uv", "pip", "install", "--python", sys.executable, *pinned]
    console.print(
        h("Selected agents recorded; --skip-extras prevented package installation.")
    )
    # soft_wrap: a wrapped command breaks when pasted.
    console.print(
        h("Install later: ") + command_line(command), markup=False, soft_wrap=True
    )


def _perform_agent_selection_interactive(
    workspace_path: Path, *, skip_extras: bool = False, preset: Optional[str] = None
) -> tuple[list[str], str]:
    """Handle interactive agent selection. Returns (selected_agents, preset_name or None)."""
    from rich.console import Console

    console = Console()

    console.print(h("\n[bold white]Agent Selection[/bold white]"))
    console.print(h("Select which agents to enable for your workspace:"))
    console.print()

    selected = _prompt_manual_agent_selection(workspace_path, preset)

    if selected:
        if skip_extras:
            _print_deferred_agent_install(selected)
        else:
            selected = _check_and_prompt_install_packages(selected, workspace_path)
        console.print(
            h("\n[green]v[/green] {len} agent(s) configured").format(len=len(selected))
        )

    return selected, _kept_preset(preset, selected)


_SMART_STD_AGENT_DISPLAY = {
    "metadata_assistant": h("Metadata Assistant"),
    "transcriptomics_expert": h("Transcriptomics Expert"),
    "annotation_expert": h("Annotation Expert"),
    "proteomics_expert": h("Proteomics Expert"),
}


def _vector_search_install_command() -> str:
    """Return an environment-aware install command for vector dependencies.

    Escaped for Rich markup: every caller prints it inside a markup string, and
    unescaped Rich drops "[vector-search]" from the command.
    """
    from rich.markup import escape

    from lobster.core.component_registry import get_install_command

    return escape(get_install_command("vector-search", is_extra=True))


def _is_vector_search_backend_available() -> bool:
    """Check whether vector backend modules are importable."""
    try:
        from lobster.vector.service import VectorSearchService  # noqa: F401
    except ImportError:
        return False
    return True


def _print_vector_backend_unavailable() -> None:
    """Explain that Smart Standardization backend is unavailable in this install."""
    console.print(
        h(
            "  [yellow]⚠ Smart Standardization backend (the lobster-vector package) is not installed.[/yellow]"
        )
    )
    console.print(
        h(
            "  [dim]Install vector dependencies: {_vector_search_install_command}[/dim]"
        ).format(_vector_search_install_command=_vector_search_install_command())
    )


def _get_smart_standardization_beneficiaries(
    selected_agents: list[str] | None,
) -> list[str]:
    """Return display names for selected agents that benefit from embeddings."""
    beneficiaries: list[str] = []
    for agent in _normalize_selected_agents(selected_agents or []):
        if agent in _SMART_STD_AGENTS:
            beneficiaries.append(_SMART_STD_AGENT_DISPLAY.get(agent, agent))
    return beneficiaries


def _install_smart_standardization_dependencies() -> None:
    """Install vector-search dependencies and warm ontology databases."""
    if not _is_vector_search_backend_available():
        _print_vector_backend_unavailable()
        return

    console.print(h("\n  [dim]Installing Smart Standardization dependencies...[/dim]"))
    from lobster.cli_internal.commands.light.agent_commands import _uv_pip_install

    s1, _ = _uv_pip_install("chromadb>=1.0.0")
    s2, _ = _uv_pip_install("openai>=1.0.0")
    if s1 and s2:
        console.print(h("  [green]✓[/green] Dependencies installed"))
    else:
        console.print(
            h(
                "  [yellow]⚠ Install manually: {_vector_search_install_command}[/yellow]"
            ).format(_vector_search_install_command=_vector_search_install_command())
        )
        return

    console.print(
        h(
            "\n  [dim]Downloading ontology databases (MONDO, UBERON, Cell Ontology)...[/dim]"
        )
    )
    if _download_ontology_databases():
        console.print(h("  [green]✓[/green] Ontology databases ready"))
    else:
        console.print(
            h(
                "  [yellow]⚠ Some databases failed to download. They will be auto-downloaded on first use.[/yellow]"
            )
        )


def _repair_skipped_secret_permissions(path: Path) -> None:
    """Tighten an owned, regular legacy secret during init, without rewriting it."""
    import os
    import shlex
    import stat

    if not hasattr(os, "getuid"):
        return  # POSIX ownership/mode checks do not certify Windows ACLs.
    fd = None
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            return
        if not stat.S_IMODE(info.st_mode) & 0o077:
            return
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or opened.st_uid != os.getuid():
            return
        os.fchmod(fd, 0o600)
        console.print(
            h("tightened permissions on {path} to 0600").format(path=path),
            markup=False,
            soft_wrap=True,
        )
    except FileNotFoundError:
        return  # No existing secret to repair; no misleading chmod command.
    except OSError:
        command = shlex.join(["chmod", "600", str(path)])
        console.print(
            h(
                "Warning: could not tighten secret file permissions; run {command}"
            ).format(command=command),
            markup=False,
            soft_wrap=True,
        )
    finally:
        if fd is not None:
            os.close(fd)


def _backup_init_file(source: Path, destination: Path) -> None:
    """Back up a workspace secret privately; retain non-secret JSON metadata."""
    if source.name == ".env":
        from lobster.config.global_config import _open_private_file

        with source.open("rb") as original:
            with _open_private_file(destination, binary=True, exclusive=True) as backup:
                shutil.copyfileobj(original, backup)
    else:
        shutil.copy2(source, destination)


def _create_workspace_config(config_dict: Dict[str, Any], workspace_path: Path) -> None:
    """
    Create workspace-scoped provider configuration from structured config dict.

    This function creates the provider_config.json file in the workspace directory,
    enabling the ConfigResolver to properly detect provider settings via layer 2
    (workspace config) instead of relying solely on .env (layer 4).

    Args:
        config_dict: Structured configuration dictionary with keys:
            - provider: str - LLM provider ("anthropic", "bedrock", "ollama")
            - profile: Optional[str] - Agent profile (for Anthropic/Bedrock)
            - ollama_model: Optional[str] - Ollama model name
            - ollama_host: Optional[str] - Ollama server URL
        workspace_path: Path to workspace directory

    Raises:
        IOError: If workspace config creation fails (non-fatal - logged as warning)
    """
    from lobster.config.workspace_config import WorkspaceProviderConfig

    try:
        # Create workspace config from structured dict
        workspace_config = WorkspaceProviderConfig.load(workspace_path)
        workspace_config.lang = _init_lang(workspace_config.lang)

        # Set provider (required)
        if "provider" in config_dict:
            workspace_config.global_provider = config_dict["provider"]

        # Set profile (for Anthropic/Bedrock)
        if "profile" in config_dict:
            workspace_config.profile = config_dict["profile"]

        # Set Ollama-specific settings
        if "ollama_model" in config_dict:
            workspace_config.ollama_model = config_dict["ollama_model"]

        if "ollama_host" in config_dict:
            workspace_config.ollama_host = config_dict["ollama_host"]

        # Set model for the selected provider (from wizard model_id field)
        if "model_id" in config_dict and config_dict["model_id"]:
            provider = config_dict.get("provider", "")
            if provider:
                workspace_config.set_model_for_provider(
                    provider, config_dict["model_id"]
                )

        # Ensure workspace directory exists and save config
        workspace_path.mkdir(parents=True, exist_ok=True)
        workspace_config.save(workspace_path)

        logger.info(
            f"Created workspace config at {workspace_path / 'provider_config.json'}"
        )

    except Exception as e:
        if config_dict.get("provider") == "omics-os":
            from lobster.config.cloud_setup import fail_cloud_setup_write

            fail_cloud_setup_write(e)
        # Preserve legacy best-effort behaviour for other providers.
        logger.warning(f"Failed to create workspace config: {e}")
        console.print(
            h(
                "[dim yellow]⚠️  Note: Workspace config creation skipped ({e})[/dim yellow]"
            ).format(e=e)
        )


# Language names are shown in their own language and never translated.
_INIT_LANGUAGES = (("en", "English"), ("zh", "简体中文 (Simplified Chinese)"))


def _ask_language(use_questionary: bool) -> Optional[str]:
    """Ask for the UI language once, unless the user already chose it.

    Returns the chosen code, or None when skipped. Raises KeyboardInterrupt
    when the user cancels the questionary prompt.
    """
    from lobster.i18n import current_lang, language_was_explicit, set_lang

    if language_was_explicit():
        return None
    default = current_lang()
    prompt = h("Choose the language for Lobster AI.")
    if use_questionary:
        import questionary

        choice = questionary.select(
            prompt,
            choices=[
                questionary.Choice(label, value=code) for code, label in _INIT_LANGUAGES
            ],
            default=default,
        ).ask()
        if choice is None:
            raise KeyboardInterrupt
    else:
        for index, (code, label) in enumerate(_INIT_LANGUAGES, 1):
            console.print(f"  {index}. {label}")
        codes = [code for code, _ in _INIT_LANGUAGES]
        picked = Prompt.ask(
            prompt,
            choices=[str(i) for i in range(1, len(codes) + 1)],
            default=str(codes.index(default) + 1 if default in codes else 1),
        )
        choice = codes[int(picked) - 1]
    return set_lang(choice)


def _init_lang(stored: Optional[str]) -> str:
    """Language to save at init: an explicit choice wins over the stored one."""
    from lobster.i18n import current_lang, language_was_explicit

    if language_was_explicit():
        return current_lang()
    return stored or current_lang()


def _create_global_config(config_dict: Dict[str, Any]) -> Path:
    """
    Create global user-level provider configuration.

    Saves to ~/.config/lobster/providers.json for use across all workspaces.

    Args:
        config_dict: Structured configuration dictionary with keys:
            - provider: str - LLM provider ("anthropic", "bedrock", "ollama", "gemini")
            - profile: Optional[str] - Agent profile (for Anthropic/Bedrock)
            - ollama_model: Optional[str] - Ollama model name
            - ollama_host: Optional[str] - Ollama server URL

    Returns:
        Path: Path to the created global config file

    Raises:
        IOError: If global config creation fails
    """
    from lobster.config.global_config import GlobalProviderConfig

    try:
        global_config = GlobalProviderConfig()
        global_config.lang = _init_lang(None)
        # A setting this run did not ask about survives it.
        global_config.lamin_enabled = config_dict.get(
            "lamin_enabled", GlobalProviderConfig.load().lamin_enabled
        )

        # Set provider (required)
        if "provider" in config_dict:
            global_config.default_provider = config_dict["provider"]

        # Set profile (for Anthropic/Bedrock)
        if "profile" in config_dict:
            global_config.default_profile = config_dict["profile"]

        # Set Ollama-specific settings
        if "ollama_model" in config_dict:
            global_config.ollama_default_model = config_dict["ollama_model"]

        if "ollama_host" in config_dict:
            global_config.ollama_default_host = config_dict["ollama_host"]

        # Set model for the selected provider (from wizard model_id field)
        if "model_id" in config_dict and config_dict["model_id"]:
            provider = config_dict.get("provider", "")
            if provider:
                global_config.set_model_for_provider(provider, config_dict["model_id"])

        global_config.save()
        config_path = GlobalProviderConfig.get_config_path()
        logger.info(f"Created global config at {config_path}")
        return config_path

    except Exception as e:
        logger.error(f"Failed to create global config: {e}")
        raise


# =============================================================================
# Init Wizard: Optional Package Installation Helpers
# =============================================================================

# Map provider choice → pip package name
_PROVIDER_PACKAGES = {
    "anthropic": "langchain-anthropic",
    "bedrock": "langchain-aws",
    "ollama": "langchain-ollama",
    "gemini": "langchain-google-genai",
    "azure": "langchain-azure-ai",
    "openai": "langchain-openai",
    "openrouter": "langchain-openai",
    "nebius": "langchain-openai",
    "ollama-cloud": "langchain-openai",
}

# Map provider choice → Python import module name
_PROVIDER_IMPORT_NAMES = {
    "anthropic": "langchain_anthropic",
    "bedrock": "langchain_aws",
    "ollama": "langchain_ollama",
    "gemini": "langchain_google_genai",
    "azure": "langchain_azure_ai",
    "openai": "langchain_openai",
    "openrouter": "langchain_openai",
    "nebius": "langchain_openai",
    "ollama-cloud": "langchain_openai",
}


def _uv_tool_env_handoff(
    provider_name: str | None,
    selected_agents: list[str] | None,
    skip_extras: bool,
    include_vector_search: bool = False,
    include_lamin: bool = False,
    confirm: bool = True,
) -> None:
    """In a uv tool env, build and optionally run the install command.

    ``confirm=False`` (non-interactive init) runs it without asking: the flags
    already chose the packages.

    Called at the end of ``lobster init`` when we detect we're inside a uv tool
    environment.  Instead of calling ``_uv_pip_install()`` (which would be
    wiped on the next ``uv tool install``), we build the correct
    ``uv tool install`` command and let the user run it.

    .. warning::

        If the user confirms execution, this function **terminates the process
        via ``os._exit()``** and never returns.  ``uv tool install`` replaces
        the venv on disk; any Rich/importlib call after that risks
        ``ModuleNotFoundError``.  Callers must not place cleanup logic after
        this call — save config **before** calling.

    If the user declines, this function returns normally.
    """
    if skip_extras:
        return  # No prompt, command construction, or package handoff under --skip-extras.

    import shlex

    from rich.prompt import Confirm

    from lobster.core.uv_tool_env import detect_uv_tool_env, run_tool_install
    from lobster.i18n import t

    info = detect_uv_tool_env()
    if not info:
        console.print(f"  {_bold_command(_next_chat_text())}")
        return  # Not in a tool env — nothing to do

    cmd = _build_uv_tool_init_command(
        None if skip_extras else provider_name,
        selected_agents,
        include_vector_search=include_vector_search,
        include_lamin=include_lamin,
    )
    cmd.insert(3, "--reinstall")
    cmd_str = shlex.join(cmd)

    # Skip if the computed command wouldn't add anything new.
    existing_cmd = _build_uv_tool_init_command(None, None)
    existing_cmd.insert(3, "--reinstall")
    if cmd == existing_cmd:
        console.print(f"  {_bold_command(_next_chat_text())}")
        return

    # The project first: the install ends the process, and Ctrl-C at the
    # prompt must not leave a configured folder without one.
    if _finish_before_tool_install is not None:
        _finish_before_tool_install()
    from lobster.core.uv_tool_env import (
        windows_locks_running_tool,
        write_pending_install,
    )

    if windows_locks_running_tool():
        write_pending_install(["uv", *cmd[1:]])  # the installer runs it after init
        run_tool_install(cmd, _next_chat_text())  # prints the command to run later

    console.print()
    console.print(
        h(
            "[bold white]Package installation needed[/bold white]\n  Your Lobster is installed via [cyan]uv tool[/cyan], so packages must\n  be added through [cyan]uv tool install[/cyan] rather than pip.\n"
        )
    )
    console.print(f"  {cmd_str}\n", markup=False)

    if not confirm or Confirm.ask(h("  Run this now?"), default=True):
        console.print(h("[dim]  Running uv tool install...[/dim]"))
        if console.file:
            console.file.flush()
        run_tool_install(cmd, _next_chat_text())
    else:
        console.print(
            "  " + t("cli.init.next_install", command=cmd_str),
            markup=False,
            soft_wrap=True,
        )
        if _in_other_folder():
            console.print("  " + _next_chat_text(), markup=False, soft_wrap=True)


def _ensure_provider_installed(provider_name: str) -> bool:
    """Check if provider package is installed; install if missing.

    Args:
        provider_name: Provider key (any key of _PROVIDER_PACKAGES)

    Returns:
        True if package is available (already installed or successfully installed)
    """
    import importlib

    pkg = _PROVIDER_PACKAGES.get(provider_name)
    module_name = _PROVIDER_IMPORT_NAMES.get(provider_name)
    if not pkg or not module_name:
        return True  # Unknown provider, skip

    try:
        importlib.import_module(module_name)
        return True  # Already installed
    except ImportError:
        pass

    console.print(h("\n  [dim]Installing {pkg}...[/dim]").format(pkg=pkg))
    from lobster.cli_internal.commands.light.agent_commands import _uv_pip_install

    success, msg = _uv_pip_install(pkg)
    if success:
        console.print(h("  [green]✓[/green] {pkg} installed").format(pkg=pkg))
        return True
    else:
        from lobster.core.component_registry import get_install_command

        console.print(
            h("  [yellow]⚠ Could not auto-install {pkg}[/yellow]").format(pkg=pkg)
        )
        console.print(
            h("  [dim]Run manually: {get_install_command}[/dim]").format(
                get_install_command=get_install_command(pkg)
            )
        )
        return False


def _prompt_docling_install() -> None:
    """Ask user about docling and install if desired."""
    try:
        import docling  # noqa: F401

        return  # Already installed, skip prompt
    except ImportError:
        pass

    console.print(h("\n[bold white]📄 Document Intelligence (Optional)[/bold white]"))
    console.print(
        h("  [dim]Structured extraction from scientific PDFs and web pages.[/dim]")
    )
    console.print(h("  [dim]Requires ~700 MB additional download.[/dim]\n"))
    console.print(h("    [cyan]1[/cyan] - Yes, install Docling"))
    console.print(h("    [cyan]2[/cyan] - No, skip for now"))
    console.print()

    choice = Prompt.ask(
        h("  Will you analyze PDF publications?"),
        choices=["1", "2"],
        default="2",
    )

    if choice == "1":
        console.print(h("  [dim]Installing docling (this may take a moment)...[/dim]"))
        from rich.markup import escape as rich_escape

        from lobster.cli_internal.commands.light.agent_commands import _uv_pip_install
        from lobster.core.component_registry import get_install_command

        s1, _ = _uv_pip_install("docling>=2.60.0")
        s2, _ = _uv_pip_install("docling-core>=2.50.0")
        if s1 and s2:
            console.print(h("  [green]✓[/green] Docling installed"))
        else:
            docling_cmd = rich_escape(get_install_command("docling", is_extra=True))
            console.print(
                h("  [yellow]⚠ Install manually: {docling_cmd}[/yellow]").format(
                    docling_cmd=docling_cmd
                )
            )
    else:
        from rich.markup import escape as rich_escape

        from lobster.core.component_registry import get_install_command

        docling_cmd = rich_escape(get_install_command("docling", is_extra=True))
        console.print(
            h("  [dim]Skipped. Install later: {docling_cmd}[/dim]").format(
                docling_cmd=docling_cmd
            )
        )


def _postprocess_tui_init_result(
    result: Dict[str, Any],
    *,
    workspace_path: Path,
    env_path: Path,
    global_config: bool,
    skip_extras: bool,
    preset: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize, persist, and complete a Go/questionary init result.

    ``preset`` (``--preset``) is recorded when the picked packages are its own.
    """
    from lobster.core.uv_tool_env import is_uv_tool_env
    from lobster.ui.bridge.init_adapter import apply_tui_init_result

    normalized_result = dict(result)

    # The language screen ran first; show everything after it in that language.
    chosen_lang = (normalized_result.get("lang") or "").strip()
    if chosen_lang:
        from lobster.i18n import set_lang

        set_lang(chosen_lang)

    # Handle Omics-OS Cloud authentication.
    provider_name = (normalized_result.get("provider") or "").strip().lower()
    api_key = (normalized_result.get("api_key") or "").strip()
    if provider_name == "omics-os":
        from lobster.config.cloud_setup import (
            OMICS_OS_AUTH_GUIDANCE,
            OMICS_OS_KEY_NOT_VERIFIED,
        )

        if not api_key:
            console.print(h("[red]Omics-OS Cloud API key is required.[/red]"))
            console.print(OMICS_OS_AUTH_GUIDANCE, markup=False)
            raise typer.Exit(1)
        normalized_result["api_key"] = api_key

    normalized_agents = _normalize_selected_agents(
        normalized_result.get("agents") or []
    )
    if normalized_agents:
        if skip_extras:
            _print_deferred_agent_install(normalized_agents)
        else:
            normalized_agents = _check_and_prompt_install_packages(
                normalized_agents, workspace_path
            )
    normalized_result["agents"] = normalized_agents
    normalized_result["preset"] = _kept_preset(preset, normalized_agents)

    try:
        apply_tui_init_result(
            normalized_result,
            workspace_path=workspace_path,
            env_path=env_path,
            global_config=global_config,
        )
    except typer.Exit:
        raise
    except Exception as error:
        if provider_name == "omics-os":
            from lobster.config.cloud_setup import fail_cloud_setup_write

            fail_cloud_setup_write(error)
        raise
    if provider_name == "omics-os":
        console.print(OMICS_OS_KEY_NOT_VERIFIED)
    _print_lamin_notes(
        normalized_result.get("lamin_enabled"),
        global_config=global_config,
        install=not skip_extras,
    )

    if is_uv_tool_env():
        return normalized_result

    if not skip_extras:
        provider_name = (normalized_result.get("provider") or "").strip().lower()
        if provider_name:
            _ensure_provider_installed(provider_name)

        if normalized_result.get("smart_standardization_enabled"):
            _install_smart_standardization_dependencies()

    return normalized_result


def _validate_oauth_credentials() -> None:
    """Validate Go TUI OAuth tokens against the gateway and enrich credentials.

    The Go TUI saves raw tokens (access_token, id_token) to credentials.json
    but cannot call the gateway to get email/tier/user_id. This function
    mirrors what the Python browser login does after receiving tokens.
    """
    try:
        from lobster.config.credentials import (
            _save_active_profile,
            get_api_key,
            get_endpoint,
        )

        token = get_api_key()
        if not token:
            console.print(
                h("[yellow]OAuth tokens found but could not be loaded.[/yellow]")
            )
            return

        endpoint = get_endpoint()

        from lobster.cli_internal.commands.light.cloud_commands import (
            _validate_credentials,
        )

        data = _validate_credentials(  # nosec B106 # token_type is a label
            endpoint, token, token_type="token"
        )
        if data is None:
            console.print(
                h(
                    "[yellow]Token validation failed. Run 'lobster cloud login' to re-authenticate.[/yellow]"
                )
            )
            return

        # Enrich the active profile with gateway data (V2-safe: preserves
        # siblings + version + active_profile; enrich-only, no token fields).
        _save_active_profile(
            {
                "user_id": data.get("user_id", ""),
                "email": data.get("email", ""),
                "tier": data.get("tier", "free"),
            }
        )

        email = data.get("email", "")
        tier = data.get("tier", "free")
        console.print(
            h("[green]Authenticated as {email} (tier: {tier})[/green]").format(
                email=email, tier=tier
            )
        )

    except Exception as exc:
        logger.debug(f"OAuth validation failed: {exc}")
        console.print(
            h(
                "[yellow]Could not validate OAuth tokens. Run 'lobster cloud login' if issues persist.[/yellow]"
            )
        )


def _download_ontology_databases() -> bool:
    """Download pre-built ontology databases from S3.

    Imports ChromaDBBackend and triggers auto-download for each ontology
    collection defined in ONTOLOGY_TARBALLS. Graceful per-collection failure.

    Returns:
        True if at least one collection was downloaded successfully.
    """
    try:
        from lobster.vector.backends.chromadb_backend import (
            ONTOLOGY_TARBALLS,
            ChromaDBBackend,
        )
    except ImportError:
        console.print(
            h(
                "  [yellow]⚠ Vector backend not available. Skipping ontology download.[/yellow]"
            )
        )
        return False

    backend = ChromaDBBackend()
    success_count = 0

    for collection_name in ONTOLOGY_TARBALLS:
        try:
            result = backend._ensure_ontology_data(collection_name)
            if result:
                success_count += 1
                console.print(f"  [green]✓[/green] {collection_name}")
            else:
                console.print(
                    h("  [yellow]⚠[/yellow] {collection_name} (skipped)").format(
                        collection_name=collection_name
                    )
                )
        except Exception as exc:
            console.print(f"  [yellow]⚠[/yellow] {collection_name}: {exc}")

    return success_count > 0


# Agents whose workflows benefit from Smart Standardization
_SMART_STD_AGENTS = {
    "metadata_assistant",
    "transcriptomics_expert",
    "annotation_expert",
    "proteomics_expert",
}


def _prompt_smart_standardization(
    selected_agents: list[str] | None = None,
) -> list[str]:
    """Prompt user to set up Smart Standardization (local-only for now).

    Smart Standardization uses OpenAI embeddings + ChromaDB to match
    biomedical terms to ontology concepts (diseases, tissues, cell types).

    Args:
        selected_agents: List of selected agent names from agent selection step.
            Used to show context-aware messaging about which agents benefit.

    Returns:
        List of env lines to append (e.g., OPENAI_API_KEY=...,
        LOBSTER_EMBEDDING_PROVIDER=openai). Empty list if skipped.
    """
    import os

    if not _is_vector_search_backend_available():
        console.print(h("\n[bold white]Smart Standardization (Optional)[/bold white]"))
        _print_vector_backend_unavailable()
        return []

    benefiting = _get_smart_standardization_beneficiaries(selected_agents)

    console.print(h("\n[bold white]Smart Standardization (Optional)[/bold white]"))
    console.print(
        h(
            "  [dim]Map biomedical terms to ontology concepts (diseases, tissues, cell types).[/dim]"
        )
    )
    console.print(
        h(
            "  [dim]Uses OpenAI text-embedding-3-small (~$0.002 per 1,000 queries).[/dim]"
        )
    )
    if benefiting:
        agents_str = ", ".join(benefiting)
        console.print(
            h("  [dim]Your selected agents that benefit: {agents_str}[/dim]").format(
                agents_str=agents_str
            )
        )
    console.print()
    console.print(
        h(
            "    [cyan]1[/cyan] - Yes, set up locally (OpenAI key + ChromaDB + ontology download)"
        )
    )
    console.print(h("    [cyan]2[/cyan] - Skip for now"))
    console.print()

    choice = Prompt.ask(
        h("  Enable Smart Standardization?"),
        choices=["1", "2"],
        default="2",
    )

    if choice != "1":
        console.print(
            h(
                "  [dim]Skipped. Enable later: {_vector_search_install_command} && lobster init --force[/dim]"
            ).format(_vector_search_install_command=_vector_search_install_command())
        )
        return []

    env_lines: list[str] = []

    # --- OpenAI API key ---
    existing_key = os.environ.get("OPENAI_API_KEY", "")
    if existing_key:
        console.print(
            h(
                "  [green]✓[/green] OPENAI_API_KEY detected (sk-...{existing_key})"
            ).format(existing_key=existing_key[-4:])
        )
        env_lines.append(f"OPENAI_API_KEY={existing_key}")
    else:
        openai_key = Prompt.ask(
            h("  [bold white]Enter your OpenAI API key[/bold white]"),
            password=True,
        )
        openai_key = openai_key.strip()
        if openai_key:
            env_lines.append(f"OPENAI_API_KEY={openai_key}")
            console.print(h("  [green]✓[/green] OpenAI API key configured"))
        else:
            console.print(
                h(
                    "  [yellow]⚠ No API key provided. Smart Standardization requires an OpenAI key.[/yellow]"
                )
            )
            return []

    env_lines.append("LOBSTER_EMBEDDING_PROVIDER=openai")

    _install_smart_standardization_dependencies()

    return env_lines


def _install_extended_data() -> None:
    """Silently install extended data access packages."""
    import importlib

    from lobster.cli_internal.commands.light.agent_commands import _uv_pip_install

    packages = {
        "polars": "polars",
        "pysradb": "pysradb",
        "cloudscraper": "cloudscraper",
        "rispy": "rispy",
        "GEOparse": "GEOparse",
    }
    missing = []
    for import_name, pip_name in packages.items():
        try:
            importlib.import_module(import_name.lower())
        except ImportError:
            missing.append(pip_name)

    if missing:
        console.print(h("\n  [dim]Installing data access packages...[/dim]"))
        for pkg in missing:
            success, _ = _uv_pip_install(pkg)
            if success:
                console.print(f"  [green]✓[/green] {pkg}")
        # Don't warn on failure — these all have graceful fallbacks


def _cancelled_text() -> str:
    from lobster.i18n import t

    return _markup_escape(t("cli.init.cancelled"))


def _lamin_saved(workspace_path: Path) -> Dict[str, str]:
    """The workspace's LaminDB settings before init writes anything."""
    from lobster.cli_internal.commands.light.lamin_commands import saved_settings

    return saved_settings(workspace_path)


def _lamin_default(flag: Optional[bool], global_config: bool) -> Optional[bool]:
    """What the wizard question starts at; None when a flag already answered it."""
    if flag is not None:
        return None
    from lobster.cli_internal.commands.light.lamin_commands import init_default

    return init_default(None, global_config=global_config)


def _ask_lamin(global_config: bool) -> bool:
    """The classic flow's LaminDB question."""
    from lobster.cli_internal.commands.light.lamin_commands import init_question

    title, text = init_question()
    console.print(f"\n[bold white]{_markup_escape(title)}[/bold white]")
    console.print(text, markup=False)
    return Confirm.ask(
        h("Keep a LaminDB catalog?"),
        default=_lamin_default(None, global_config),
    )


def _save_lamin_choice(
    choice: Optional[bool], workspace_path: Path, *, previous: Dict[str, str]
) -> None:
    from lobster.cli_internal.commands.light.lamin_commands import apply_init_choice

    apply_init_choice(choice, workspace_path, previous=previous)


def _print_lamin_notes(
    choice: Optional[bool], *, global_config: bool, install: bool = False
) -> None:
    from lobster.cli_internal.commands.light.lamin_commands import (
        _engine_installed,
        init_notes,
    )
    from lobster.core.uv_tool_env import is_uv_tool_env

    # Under uv tool the handoff adds the extra to the tool instead.
    if install and choice and not is_uv_tool_env() and not _engine_installed():
        import importlib

        from rich.markup import escape

        from lobster.cli_internal.commands.light.agent_commands import (
            _uv_pip_install,
        )

        package = "lobster-ai[lamin]"
        console.print(h("  [dim]Installing {pkg}...[/dim]").format(pkg=escape(package)))
        if _uv_pip_install(package)[0]:
            importlib.invalidate_caches()
    notes = init_notes(
        choice,
        global_config=global_config,
        install_hint=not (install and is_uv_tool_env()),
    )
    for line in notes:
        console.print(line, markup=False)


def _resolve_workspace_or_exit(
    create: bool = True, workspace: Optional[Path] = None
) -> Path:
    """Resolve the init workspace, exiting 1 with a clear message on OS errors."""
    from lobster.core.runtime.workspace import resolve_workspace

    try:
        return resolve_workspace(workspace, create=create)
    except OSError as exc:
        configured = (
            str(workspace)
            if workspace is not None
            else os.environ.get("LOBSTER_WORKSPACE")
            or str(Path.cwd() / ".lobster_workspace")
        )
        console.print(
            h(
                "[red]Error:[/red] cannot create the workspace at {_markup_escape} ({_markup_escape3}). Fix the path or unset LOBSTER_WORKSPACE."
            ).format(
                _markup_escape=_markup_escape(configured),
                _markup_escape3=_markup_escape(exc.strerror or str(exc)),
            ),
            soft_wrap=True,
        )
        raise typer.Exit(1)


def _print_init_summary(
    result: Dict[str, Any],
    workspace_path: Path,
    extra_lines: Optional[List[str]] = None,
    next_step: bool = True,
) -> None:
    """Print the localized final summary of a TUI/questionary init run.

    ``extra_lines`` are already-localized plain-text lines printed after the
    workspace line and before the closing "next step" line. ``next_step=False``
    leaves that line to the uv tool install that follows.
    """
    from lobster.i18n import t

    provider = (result.get("provider") or "").strip()
    model = (result.get("model_id") or result.get("ollama_model") or "").strip()
    provider_text = f"{provider} ({model})" if provider and model else provider
    esc = _markup_escape
    failed = _failed_install_lines()
    mark = "[bold yellow]⚠[/bold yellow]" if failed else "[bold green]✓[/bold green]"
    console.print(f"{mark} {esc(t('cli.init.done'))}")
    console.print(f"  {esc(t('cli.init.done_provider', provider=provider_text))}")
    console.print(f"  {esc(t('cli.init.done_workspace', path=str(workspace_path)))}")
    for line in extra_lines or []:
        console.print(f"  {esc(line)}")
    for line in failed:
        console.print(f"  [yellow]{esc(line)}[/yellow]")
    if not failed and next_step:
        console.print(f"  {_bold_command(_next_chat_text())}")


def _tool_install_follows(skip_extras: bool) -> bool:
    """True when init ends with the uv tool install, which owns the next step."""
    from lobster.core.uv_tool_env import is_uv_tool_env

    return is_uv_tool_env() and not skip_extras


def _failed_install_lines() -> List[str]:
    """Localized lines naming the installs that failed in this run, and the fix."""
    import shutil

    from lobster.cli_internal.commands.light.agent_commands import FAILED_INSTALLS
    from lobster.core.uv_tool_env import command_line
    from lobster.i18n import t

    failed = list(dict.fromkeys(FAILED_INSTALLS))
    if not failed:
        return []
    if shutil.which("uv"):
        command = ["uv", "pip", "install", "--python", sys.executable, *failed]
    else:
        command = [sys.executable, "-m", "pip", "install", *failed]
    return [
        t("cli.init.install_failed", packages=", ".join(failed)),
        t("cli.init.next_install", command=command_line(command)),
    ]


def _bold_command(text: str) -> str:
    """Escape Rich markup and bold the ``lobster chat`` command inside text."""
    from rich.markup import escape

    escaped = escape(text)
    return escaped.replace("lobster chat", "[bold]lobster chat[/bold]")


def _ensure_tui_installed() -> None:
    """Install textual for TUI support if not present."""
    try:
        import textual  # noqa: F401

        return
    except ImportError:
        pass

    from lobster.cli_internal.commands.light.agent_commands import _uv_pip_install

    console.print(h("  [dim]Installing TUI support...[/dim]"))
    success, _ = _uv_pip_install("textual>=6.7.1")
    if success:
        console.print(h("  [green]✓[/green] textual installed"))


def _init_target(folder: Optional[Path]) -> Tuple[Path, Optional[Path]]:
    """The folder ``lobster init`` configures and the workspace path to use for it.

    Init configures the folder it is run in (or the one named), never the project
    that happens to enclose it, so the workspace is pinned to that folder. Only a
    ``LOBSTER_WORKSPACE`` set without a named folder keeps meaning what it always
    did: the workspace is taken from it (second value ``None``).
    """
    from lobster.core.runtime.workspace import WORKSPACE_ENV_VAR

    root = Path(os.path.abspath(Path.cwd() if folder is None else folder))
    if folder is None and os.environ.get(WORKSPACE_ENV_VAR):
        return root, None
    return root, root / ".lobster_workspace"


def _ensure_project(
    folder: Optional[Path],
    project_name: Optional[str],
    env_before: Optional[bytes] = None,
) -> None:
    """Give a folder that init just configured its project identity (kept if present).

    ``env_before`` is what the folder's ``.env`` held before init ran (None: no
    file); a file that now differs is one init wrote, and the project records it.
    """
    from lobster.core.projects import ProjectError, init_project, record_created_file
    from lobster.core.runtime.workspace import (
        WORKSPACE_FOLDER_NAME,
        resolve_workspace,
    )

    _, pinned = _init_target(folder)
    workspace = resolve_workspace(pinned, create=False)
    # Nothing was written (cancelled, or a failed run): no project either. A
    # workspace named otherwise (LOBSTER_WORKSPACE) is not a project folder.
    if (
        workspace.name != WORKSPACE_FOLDER_NAME
        or not (workspace / "provider_config.json").is_file()
    ):
        return
    try:
        project, created = init_project(workspace.parent, name=project_name)
    except ProjectError as exc:
        console.print(
            f"[yellow]⚠ {_markup_escape(exc.message)}[/yellow]", soft_wrap=True
        )
        return
    try:
        env = project.root / ".env"
        if env.is_file() and env.read_bytes() != env_before:
            record_created_file(project, ".env")
    except OSError:
        pass  # the .env note is advice only; init itself went through
    console.print(
        h("✓ Project {name} {state} at {path}").format(
            name=_markup_escape(project.name),
            state=h("created") if created else h("kept"),
            path=_markup_escape(str(project.root)),
        ),
        soft_wrap=True,
    )


def _missing_ancestors(folder: Path) -> List[Path]:
    """The folders above ``folder`` (and itself) that do not exist yet, nearest first."""
    missing = []
    for parent in (folder, *folder.parents):
        if parent.exists() or parent.is_symlink():
            break
        missing.append(parent)
    return missing


def _undo_empty_init(pinned: Path, made: List[Path]) -> None:
    """Remove the workspace and folders this call made when no setup was written."""
    import shutil

    if (
        pinned.is_dir()
        and not pinned.is_symlink()
        and not (pinned / "provider_config.json").exists()
        and not (pinned / "project.json").exists()
    ):
        shutil.rmtree(pinned, ignore_errors=True)
    for folder in made:  # nearest first: a folder goes only once it is empty
        try:
            folder.rmdir()
        except OSError:
            break


class _Handled(typer.Exit):
    """``init`` finished and said everything itself; the project is already made."""


_KEY_FLAG_NAMES = (
    "anthropic_key",
    "bedrock_access_key",
    "bedrock_secret_key",
    "use_ollama",
    "gemini_key",
    "openai_key",
    "openrouter_key",
    "nebius_key",
    "ollama_cloud_key",
    "azure_endpoint",
    "azure_credential",
    "use_omics_os",
    "omics_os_key",
    "cloud_key",
)


# Options the key-free path cannot apply; passing one is refused, not dropped.
_TERMINAL_ONLY_OPTIONS = (
    ("--ncbi-key", "ncbi_key"),
    ("--profile", "profile"),
    ("--ollama-model", "ollama_model"),
    ("--cloud-endpoint", "cloud_endpoint"),
    ("--install-docling", "install_docling"),
    ("--install-vector-search", "install_vector_search"),
    ("--agents-description", "agents_description"),
    ("--ssl-verify", "ssl_verify"),
    ("--ssl-cert-path", "ssl_cert_path"),
    ("--skip-ssl-test", "skip_ssl_test"),
)


_SWITCH_OPTIONS = {"ssl_verify": "--no-ssl-verify"}


def _terminal_only_given(name: str, value: Any) -> bool:
    if name in _SWITCH_OPTIONS:  # Optional[bool]: --no-x is False, unset is None
        return value is not None
    return value is not None and value is not False


def _given(value: Any) -> bool:
    """A flag the user passed (an unset typer option is not a str or True)."""
    return value is True or isinstance(value, str)


def _where(setting: Any) -> str:
    return {
        "global": setting.where or "the global config",
        "shell": "the shell",
        "project": "this project",
        "default": "provider default",
    }[setting.source]


def _init_without_terminal(
    init_root: Path,
    init_ws: Path,
    project_name: Optional[str],
    terminal_only: List[str],
    provider_flag: Optional[str],
    model: Any,
    agents: Any,
    preset: Any,
    auto_agents: Any,
    lamin: Optional[bool] = None,
) -> None:
    """``lobster init <folder>`` with no terminal and no key flag: a host's call.

    The project gets its identity, and runs with the provider and key of the setup
    the user already has: nothing secret is read into, copied or written under the
    project. ``--provider``, ``--model`` and ``--agents``/``--preset`` go into the
    project's own settings; ``--lamin``/``--no-lamin`` writes the catalog switch (no
    secret) into the project's ``.env``. Every check comes before the first write, so
    a refused call changes nothing.
    """
    import shutil
    import unicodedata

    from lobster.cli_internal import provider_hint
    from lobster.config.workspace_agent_config import WorkspaceAgentConfig
    from lobster.config.workspace_config import WorkspaceProviderConfig
    from lobster.core.component_registry import AGENT_TO_PACKAGE
    from lobster.core.projects import (
        Effective,
        ProjectError,
        effective_settings,
        init_project,
        load_project,
        project_settings,
    )

    def say(text: str) -> None:
        console.print(_markup_escape(text), soft_wrap=True)

    def refuse(code: int, *lines: str) -> None:
        for line in lines:
            say(line)
        raise typer.Exit(code)

    folder_text = str(init_root)

    # the caller's own flags (never a key), repeated on the lines a refusal suggests
    caller_flags: List[str] = []
    if isinstance(model, str) and model.strip():
        caller_flags += ["--model", model.strip()]
    if isinstance(preset, str) and preset:
        caller_flags += ["--preset", preset]
    elif isinstance(agents, str) and agents:
        caller_flags += ["--agents", agents]
    if lamin is not None:
        caller_flags.append("--lamin" if lamin else "--no-lamin")

    def cmd(*args: str) -> str:
        return provider_hint.command_line(*args)

    if terminal_only:
        refuse(
            2,
            f"✗ bad_argument: {', '.join(terminal_only)} needs a terminal; nothing was changed.",
            *(
                [
                    "In a terminal the setup asks for the provider and its key; --provider is for a call without one."
                ]
                if isinstance(provider_flag, str) and provider_flag.strip()
                else []
            ),
            f"In a terminal: {cmd('init', folder_text, *caller_flags)}",
        )

    # What exists here: an earlier project of init's own, or something else.
    existing = None
    if init_ws.exists():
        try:
            existing = load_project(init_root)
        except ProjectError:
            refuse(
                1,
                f"✗ bad_argument: {init_ws} exists but is not a project made by lobster init. Nothing was changed.",
                f"In a terminal: {cmd('init', folder_text, *caller_flags)}",
            )

    flag = provider_flag.strip() if isinstance(provider_flag, str) else None
    provider, model_now = effective_settings(init_ws)
    if flag:
        provider = Effective(flag, "flag")
        model_now = None
    elif provider is None:
        refuse(
            1,
            "✗ " + provider_hint.no_setup_message(folder_text, extra=caller_flags),
        )

    new_model = None
    if isinstance(model, str):
        new_model = model.strip()
        if (
            not new_model
            or len(new_model) > 200
            or any(unicodedata.category(c).startswith("C") for c in new_model)
        ):
            refuse(
                2,
                "✗ bad_argument: --model needs a model name (one line, at most 200 characters). Nothing was changed.",
            )

    model_field = (
        provider.value.replace("-", "_") + WorkspaceProviderConfig().model_field_suffix
    )
    if new_model and not hasattr(WorkspaceProviderConfig(), model_field):
        refuse(
            2,
            f"✗ bad_argument: {provider.value} has no model setting; drop --model. Nothing was changed.",
        )

    if auto_agents is True:
        refuse(
            2,
            "✗ bad_argument: --auto-agents needs a terminal; pass --agents or --preset. Nothing was changed.",
        )
    requested: List[str] = []
    preset_name = None
    if isinstance(preset, str) and preset:
        from lobster.config.agent_presets import expand_preset, is_valid_preset

        if not is_valid_preset(preset):
            refuse(
                1,
                f"✗ bad_argument: invalid preset {preset}. Nothing was changed.",
                "Valid presets: scrna-basic, scrna-full, multiomics-full",
            )
        requested = _filter_unpublished_preset_agents(
            expand_preset(preset), console, preset
        )
        preset_name = preset
    elif isinstance(agents, str) and agents:
        requested = _normalize_selected_agents(
            [a.strip() for a in agents.split(",") if a.strip()]
        )
        if not requested:
            refuse(
                1,
                "✗ bad_argument: --agents names no expert. Nothing was changed.",
            )
    installed_names = set(_get_installed_agents())
    written = [a for a in requested if a in installed_names]
    left_out = [a for a in requested if a not in installed_names]

    def left_out_lines() -> List[str]:
        lines = []
        for name in left_out:
            package = AGENT_TO_PACKAGE.get(name)
            lines.append(
                f"{name}: not installed; run {cmd('agents', 'install', package)}"
                if package
                else f"{name}: unknown expert"
            )
        return lines

    if requested and not written:
        refuse(
            1,
            "✗ bad_argument: none of those experts is installed. Nothing was changed.",
            *left_out_lines(),
        )

    pin_provider = bool(flag) or provider.source == "shell" or bool(new_model)
    if existing is not None and not (pin_provider or written or lamin is not None):
        say(
            f"Already a project: {existing.name} at {existing.root}. Nothing was changed."
        )
        raise _Handled(0)

    # ---- the first write ----
    from lobster.core.projects import forget_project

    settings_files = (
        init_ws / "provider_config.json",
        init_ws / "config.toml",
        init_ws / ".env",
    )
    lamin_before = _lamin_saved(init_ws)
    before = {
        path: path.read_bytes() if path.is_file() else None for path in settings_files
    }

    def undo() -> List[Path]:
        # a failed call leaves the folder as it was: a project this call made is removed;
        # in an existing one the settings files get back the bytes they had.
        # Returns what could not be given back.
        failed: List[Path] = []
        if existing is None:
            shutil.rmtree(init_ws, ignore_errors=True)
            forget_project(init_root)
            if init_ws.exists() or init_ws.is_symlink():
                failed.append(init_ws)
            return failed
        for path, old in before.items():
            try:
                if old is None:
                    path.unlink(missing_ok=True)
                elif not path.is_file() or path.read_bytes() != old:
                    path.write_bytes(old)
            except OSError:
                failed.append(path)
        return failed

    def give_back() -> bool:
        """Undo; names what could not be given back. True when all is as it was."""
        failed = undo()
        for path in failed:
            say(f"✗ {path} could not be given back; check it yourself.")
        return not failed

    try:
        init_root.mkdir(parents=True, exist_ok=True)
        if existing is None:
            project, created = init_project(init_root, name=project_name)
        else:
            project, created = existing, False
        if pin_provider:
            _create_workspace_config(
                {"provider": provider.value, "model_id": new_model}, init_ws
            )
            # the writer reports a failure as a note and goes on: read the file back
            on_disk = WorkspaceProviderConfig.load(init_ws)
            if (init_ws / "provider_config.json").is_symlink() or (
                on_disk.global_provider != provider.value
                or (
                    new_model
                    and on_disk.get_model_for_provider(provider.value) != new_model
                )
            ):
                raise OSError(
                    f"{init_ws / 'provider_config.json'} does not hold the provider"
                    + (" and model" if new_model else "")
                    + " that was set"
                )
        if written:
            agent_config = WorkspaceAgentConfig.load(init_ws)
            agent_config.enabled_agents = written
            agent_config.preset = preset_name if not left_out else None
            agent_config.save(init_ws)
            # a kept preset is stored alone; a list is stored without a preset
            saved = WorkspaceAgentConfig.load(init_ws)
            if (saved.preset, saved.enabled_agents) != (
                (agent_config.preset, []) if agent_config.preset else (None, written)
            ):
                raise OSError(f"{init_ws / 'config.toml'} does not hold the experts")
        if lamin is not None:
            _save_lamin_choice(lamin, init_ws, previous=lamin_before)
    except ProjectError as exc:
        give_back()
        refuse(2, f"✗ {exc.code}: {exc.message}")
    except OSError as exc:
        restored = give_back()
        refuse(
            1,
            f"✗ Could not write the project's settings ({exc})."
            + (" Nothing was changed." if restored else ""),
        )
    except BaseException:  # an exit or error from a helper: nothing of this call stays
        give_back()
        raise

    state = h("created") if created else h("kept")
    say(f"✓ Project {project.name} {state} at {project.root}")
    kept_experts = [] if written else project_settings(project).experts
    if pin_provider:
        taken = {
            "flag": "--provider",
            "shell": "the shell",
            "global": provider.where or "the global config",
            "project": "this project",
        }[provider.source]
        say(f"  provider: {provider.value} (set for this project, taken from {taken})")
    elif provider.source == "project":
        say(f"  provider: {provider.value} (set for this project)")
    else:
        taken = provider.where if provider.source == "global" else "the shell"
        say(f"  provider: {provider.value} (not set in this project; from {taken})")
    if new_model:
        say(f"  model: {new_model} (set for this project; not checked)")
        say("  The project is now fixed to that provider and model.")
    elif model_now is not None and not pin_provider:
        if model_now.source == "project":
            say(f"  model: {model_now.value} (set for this project)")
        else:
            src = (
                model_now.where if model_now.source == "global" else "provider default"
            )
            say(f"  model: {model_now.value} (not set in this project; from {src})")
    if written:
        say(
            f"  experts: {', '.join(written)} ({len(written)} of {len(requested)} experts set)"
        )
        for line in left_out_lines():
            say(f"  {line}")
    elif kept_experts:
        say(f"  experts: {', '.join(kept_experts)} (set for this project)")
    else:
        say("  experts: every installed expert (no list set for this project)")
    from lobster.cli_internal.commands.light.lamin_commands import init_notes

    for note in init_notes(lamin, global_config=False):
        say(note)
    if not (pin_provider or written or kept_experts or provider.source == "project"):
        say(
            f"  Run {cmd('init', folder_text)} in a terminal to give this project its own settings."
        )
    raise _Handled(0)


# Set by _creates_project while init runs: the uv tool install ends the
# process, so it finishes the project first.
_finish_before_tool_install: Optional[Callable[[], None]] = None
# The folder init configures, for the next step (None: global setup).
_init_folder: Optional[Path] = None


def _in_other_folder() -> bool:
    """Whether init configures a folder other than the one it runs in."""
    return _init_folder is not None and Path.cwd().resolve() != _init_folder.resolve()


def _next_chat_text() -> str:
    """The next step; from another folder it changes into the project first."""
    import shlex

    from lobster.i18n import t

    if not _in_other_folder():
        return t("cli.init.next_chat")
    # Windows PowerShell 5.1 rejects && between statements.
    separator = "; " if sys.platform == "win32" else " && "
    command = shlex.join(["cd", str(_init_folder)]) + separator + "lobster chat"
    return t("cli.init.next_chat_in", command=command)


def _has_omics_os_account() -> bool:
    """Ask the classic wizard's account question; "No" shows the access request.

    Cancelling the question, like "No", returns to the provider list.
    """
    from lobster.config.cloud_setup import request_access

    console.print(f"\n[bold white]{h('Do you have an Omics-OS account?')}[/bold white]")
    console.print(f"  [cyan] 1[/cyan] - {h('Yes, I have an account')}")
    console.print(f"  [cyan] 2[/cyan] - {h('No, request access')}")
    try:
        answer = Prompt.ask(
            h("[bold white]Selection[/bold white]"), choices=["1", "2"], default="1"
        )
    except (KeyboardInterrupt, EOFError):
        return False
    if answer == "1":
        return True
    console.print()
    request_access(
        # Sentences wrap at word boundaries; the address stays on one line to copy.
        lambda line: console.print(
            line, markup=False, highlight=False, soft_wrap=" " not in line
        ),
        lambda prompt: Prompt.ask(prompt, default="", show_default=False),
    )
    return False


def _creates_project(func):
    """Run ``func`` (``init_impl``); when it ends normally, project the folder.

    A run that leaves no setup behind (cancelled, refused) also leaves no empty
    workspace or new folder it made itself.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from lobster.core.uv_tool_env import clear_pending_install

        clear_pending_install()  # only this run's Windows reinstall may remain
        pinned = None
        made: List[Path] = []
        if not kwargs.get("global_config"):
            _, pinned = _init_target(kwargs.get("folder"))
            if pinned is not None and not (pinned.exists() or pinned.is_symlink()):
                made = _missing_ancestors(pinned.parent)
            else:
                pinned = None  # an existing workspace is never ours to remove

        env_before = None
        if not kwargs.get("global_config"):
            try:
                env_before = (
                    _init_target(kwargs.get("folder"))[0] / ".env"
                ).read_bytes()
            except OSError:
                pass

        finished = []

        def finish():
            # Runs once: before a uv tool install, or when init ends.
            if not kwargs.get("global_config") and not finished:
                finished.append(True)
                _ensure_project(
                    kwargs.get("folder"), kwargs.get("project_name"), env_before
                )

        global _finish_before_tool_install, _init_folder
        _finish_before_tool_install = finish
        if not kwargs.get("global_config"):
            _init_folder = _init_target(kwargs.get("folder"))[0]
        try:
            result = func(*args, **kwargs)
        except typer.Exit as exc:
            if not exc.exit_code and not isinstance(exc, _Handled):
                finish()
            raise
        else:
            finish()
            return result
        finally:
            _finish_before_tool_install = _init_folder = None
            if pinned is not None:
                _undo_empty_init(pinned, made)

    return wrapper


@_creates_project
def init_impl(
    folder: Optional[Path] = None,
    global_config: bool = typer.Option(
        False,
        "--global",
        "-g",
        help=h("Save configuration globally (~/.config/lobster/) for all workspaces"),
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help=h("Overwrite existing configuration")
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help=h("Non-interactive mode for CI/CD (requires API key flags)"),
    ),
    use_omics_os: bool = False,
    omics_os_key: Optional[str] = None,
    anthropic_key: Optional[str] = typer.Option(
        None, "--anthropic-key", help=h("Claude API key (non-interactive mode)")
    ),
    bedrock_access_key: Optional[str] = typer.Option(
        None,
        "--bedrock-access-key",
        help=h("AWS Bedrock access key (non-interactive mode)"),
    ),
    bedrock_secret_key: Optional[str] = typer.Option(
        None,
        "--bedrock-secret-key",
        help=h("AWS Bedrock secret key (non-interactive mode)"),
    ),
    use_ollama: bool = typer.Option(
        False,
        "--use-ollama",
        help=h("Use Ollama (local LLM) provider (non-interactive mode)"),
    ),
    ollama_model: Optional[str] = typer.Option(
        None,
        "--ollama-model",
        help=h(
            "Ollama model name (default: {DEFAULT_OLLAMA_MODEL}, non-interactive mode)"
        ).format(DEFAULT_OLLAMA_MODEL=provider_setup.DEFAULT_OLLAMA_MODEL),
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help=h(
            "Default model ID for the selected provider (e.g., claude-sonnet-4-20250514, gpt-4o). Saved to workspace config."
        ),
    ),
    gemini_key: Optional[str] = typer.Option(
        None, "--gemini-key", help=h("Google API key (non-interactive mode)")
    ),
    openai_key: Optional[str] = typer.Option(
        None, "--openai-key", help=h("OpenAI API key (non-interactive mode)")
    ),
    openrouter_key: Optional[str] = typer.Option(
        None, "--openrouter-key", help=h("OpenRouter API key (non-interactive mode)")
    ),
    nebius_key: Optional[str] = typer.Option(
        None, "--nebius-key", help=h("Nebius AI Studio API key (non-interactive mode)")
    ),
    ollama_cloud_key: Optional[str] = typer.Option(
        None,
        "--ollama-cloud-key",
        help=h("Ollama Cloud API key (non-interactive mode)"),
    ),
    azure_endpoint: Optional[str] = typer.Option(
        None, "--azure-endpoint", help=h("Azure AI endpoint URL (non-interactive mode)")
    ),
    azure_credential: Optional[str] = typer.Option(
        None,
        "--azure-credential",
        help=h("Azure AI API credential (non-interactive mode)"),
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help=h(
            "Agent profile (development, production, performance, max). Only for Anthropic/Bedrock providers."
        ),
    ),
    ncbi_key: Optional[str] = typer.Option(
        None, "--ncbi-key", help=h("NCBI API key (optional, non-interactive mode)")
    ),
    cloud_key: Optional[str] = typer.Option(
        None,
        "--cloud-key",
        help=h("Omics-OS Cloud API key (optional, enables premium tier)"),
    ),
    cloud_endpoint: Optional[str] = typer.Option(
        None,
        "--cloud-endpoint",
        help=h("Custom cloud endpoint URL (optional)"),
    ),
    skip_ssl_test: bool = typer.Option(
        False,
        "--skip-ssl-test",
        help=h("Skip SSL connectivity test during init"),
    ),
    ssl_verify: Optional[bool] = typer.Option(
        None,
        "--ssl-verify/--no-ssl-verify",
        help=h("Enable/disable SSL verification (non-interactive mode)"),
    ),
    ssl_cert_path: Optional[str] = typer.Option(
        None,
        "--ssl-cert-path",
        help=h("Path to custom CA certificate bundle"),
    ),
    # Agent selection parameters (CLI-01, CLI-02, CONF-07, CONF-08)
    agents: Optional[str] = typer.Option(
        None,
        "--agents",
        help=h("Comma-separated list of agents to enable (non-interactive mode)"),
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help=h("Agent preset name (scrna-basic, scrna-full, multiomics-full)"),
    ),
    auto_agents: bool = typer.Option(
        False,
        "--auto-agents",
        help=h(
            "Use LLM to suggest agents based on description (requires --agents-description)"
        ),
    ),
    agents_description: Optional[str] = typer.Option(
        None,
        "--agents-description",
        help=h("Workflow description for LLM agent suggestion (with --auto-agents)"),
    ),
    skip_docling: bool = typer.Option(
        False,
        "--skip-docling",
        help=h("Skip docling installation prompt (non-interactive mode)"),
    ),
    install_docling: bool = typer.Option(
        False,
        "--install-docling",
        help=h("Install docling for PDF intelligence (non-interactive mode)"),
    ),
    install_vector_search: bool = typer.Option(
        False,
        "--install-vector-search",
        help=h(
            "Install Smart Standardization dependencies (the lobster-vector package)"
        ),
    ),
    skip_extras: bool = typer.Option(
        False,
        "--skip-extras",
        help=h("Skip optional package installation (provider, TUI, extended-data)"),
    ),
    ui_mode: str = typer.Option(
        "auto",
        "--ui",
        help=h(
            "UI mode: auto (Go TUI, then questionary, classic), go (Go TUI), classic (Rich prompts)"
        ),
    ),
    project_name: Optional[str] = None,
    provider_flag: Optional[str] = None,
    lamin: Optional[bool] = None,
):
    """
    Initialize Lobster AI configuration.

    By default, creates workspace-specific configuration (.env + provider_config.json)
    and makes the folder a Lobster project. Run it again to change the settings; the
    project keeps its identity. Use --global to set user-wide defaults that apply to
    all workspaces.

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
      lobster init my-study                           # Set up a new project in ./my-study
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
        --use-ollama --ollama-model=qwen3:14b       # CI/CD: Ollama with custom model
      lobster init --non-interactive \\
        --anthropic-key=sk-ant-xxx \\
        --cloud-key=cloud_xxx                        # CI/CD: With cloud access
      lobster init --non-interactive \\
        --openrouter-key=sk-or-xxx                   # CI/CD: OpenRouter (600+ models)
      lobster init --non-interactive \\
        --nebius-key=xxx                             # CI/CD: Nebius AI Studio (open-weight models)
      lobster init --non-interactive \\
        --azure-endpoint=https://xxx.inference.ai.azure.com/ \\
        --azure-credential=xxx                       # CI/CD: Azure AI
    """
    import datetime

    from lobster.config.global_config import GlobalProviderConfig

    flags = locals()  # a snapshot of the options; a generator below has its own locals
    if ui_mode not in ("auto", "go", "classic"):
        console.print(
            h(
                "[red]Error:[/red] Invalid --ui value '{ui_mode}'. Must be one of: auto, go, classic"
            ).format(ui_mode=ui_mode)
        )
        raise typer.Exit(1)

    if isinstance(provider_flag, str):
        from lobster.cli_internal import provider_hint
        from lobster.config.constants import VALID_PROVIDERS

        if provider_flag not in VALID_PROVIDERS:
            console.print(
                h(
                    "[red]Error:[/red] bad_argument: unknown provider {name}. Valid providers: {valid}. Nothing was changed."
                ).format(
                    name=_markup_escape(provider_flag), valid=", ".join(VALID_PROVIDERS)
                ),
                soft_wrap=True,
            )
            raise typer.Exit(2)
        if global_config:
            console.print(
                h(
                    "[red]Error:[/red] --provider names the provider of one project; it does not work with --global."
                )
            )
            raise typer.Exit(2)
        if any(_given(flags.get(name)) for name in _KEY_FLAG_NAMES):
            console.print(
                h(
                    "[red]Error:[/red] the key flag already names the provider; drop --provider."
                )
            )
            raise typer.Exit(2)
        if non_interactive or folder is None or sys.stdin.isatty():
            console.print(
                h(
                    "[red]Error:[/red] --provider is for a project made without a terminal: {line}. In a terminal, pick the provider in the setup."
                ).format(
                    line=_markup_escape(
                        provider_hint.command_line(
                            "init",
                            provider_hint.Placeholder("<folder>"),
                            "--provider",
                            provider_hint.Placeholder("<name>"),
                        )
                    )
                ),
                soft_wrap=True,
            )
            raise typer.Exit(2)

    from lobster.core.runtime.workspace import resolve_workspace

    if global_config and (folder is not None or project_name is not None):
        console.print(
            h(
                "[red]Error:[/red] --global sets user-wide defaults; it takes no folder or --name."
            )
        )
        raise typer.Exit(2)
    if project_name is not None:
        from lobster.core.projects import ProjectError, _validate_name

        try:
            _validate_name(project_name)
        except ProjectError as exc:
            console.print(f"[red]Error:[/red] {_markup_escape(exc.message)}")
            raise typer.Exit(2)
    init_root, init_ws = _init_target(folder)
    if not global_config and init_ws is not None:
        from lobster.core.projects import enclosing_project, is_home_or_above

        if is_home_or_above(init_root):
            console.print(
                h(
                    "[red]Error:[/red] {path} is your home folder or contains it; make a subfolder, e.g. lobster init ~/my-study"
                ).format(path=_markup_escape(str(init_root))),
                soft_wrap=True,
            )
            raise typer.Exit(2)
        if init_ws.is_symlink():
            console.print(
                h(
                    "[red]Error:[/red] {path} is a link; a project needs a real folder. Nothing was written."
                ).format(path=_markup_escape(str(init_ws))),
                soft_wrap=True,
            )
            raise typer.Exit(2)

        outer = (
            None
            if (init_root / ".lobster_workspace" / "project.json").is_file()
            else enclosing_project(init_root)
        )
        if outer is not None:
            console.print(
                h(
                    "[red]Error:[/red] {path} is inside the project {name} at {root}; projects do not nest. Run init in that folder instead. If {root} should not be a project: lobster projects delete {root}"
                ).format(
                    path=_markup_escape(str(init_root)),
                    name=_markup_escape(outer.name),
                    root=_markup_escape(str(outer.root)),
                ),
                soft_wrap=True,
            )
            raise typer.Exit(2)
    if folder is not None and not global_config:
        if init_root.exists() and not init_root.is_dir():
            console.print(
                h("[red]Error:[/red] {path} is not a folder.").format(
                    path=_markup_escape(str(init_root))
                ),
                soft_wrap=True,
            )
            raise typer.Exit(2)

    # A host calls init with no terminal and no key: the project takes the
    # user's existing provider setup (see _init_without_terminal).
    if (
        not global_config
        and not non_interactive
        and folder is not None
        and init_ws is not None
        and not sys.stdin.isatty()
        and not any(_given(flags.get(name)) for name in _KEY_FLAG_NAMES)
    ):
        _init_without_terminal(
            init_root,
            init_ws,
            project_name,
            [
                (
                    _SWITCH_OPTIONS[name]
                    if name in _SWITCH_OPTIONS and flags[name] is False
                    else flag
                )
                for flag, name in _TERMINAL_ONLY_OPTIONS
                if _terminal_only_given(name, flags.get(name))
            ],
            provider_flag,
            model,
            agents,
            preset,
            auto_agents,
            lamin,
        )

    # Determine paths based on global vs workspace mode
    if global_config:
        config_path = GlobalProviderConfig.get_config_path()
        env_path = None  # Global mode doesn't create .env (use env vars for API keys)
    else:
        if folder is not None:
            try:
                init_root.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                console.print(
                    h("[red]Error:[/red] cannot create {path} ({reason}).").format(
                        path=_markup_escape(str(init_root)),
                        reason=_markup_escape(exc.strerror or str(exc)),
                    ),
                    soft_wrap=True,
                )
                raise typer.Exit(1)
        config_path = resolve_workspace(init_ws, create=False) / "provider_config.json"
        env_path = init_root / ".env"
    # The LaminDB catalog answer (--lamin/--no-lamin, else asked when interactive).
    lamin_previous = {} if global_config else _lamin_saved(config_path.parent)

    # Cloud is additive: retain the established priority of explicit providers.
    # Resolve/validate its input before a skipped configuration or force backup
    # can falsely look like a completed change to the gateway provider.
    if not isinstance(ollama_cloud_key, str):
        ollama_cloud_key = None

    cloud_selected = (
        non_interactive
        and (use_omics_os or omics_os_key is not None)
        and not any(
            (
                anthropic_key is not None,
                bedrock_access_key is not None and bedrock_secret_key is not None,
                use_ollama,
                gemini_key is not None,
                openai_key is not None,
                openrouter_key is not None,
                nebius_key is not None,
                ollama_cloud_key is not None,
                azure_endpoint is not None and azure_credential is not None,
            )
        )
    )
    cloud_api_key = None
    if cloud_selected:
        from lobster.config.cloud_setup import OMICS_OS_AUTH_GUIDANCE
        from lobster.config.credentials import get_api_key

        cloud_api_key = omics_os_key if omics_os_key is not None else get_api_key()
        if omics_os_key is None and (not cloud_api_key or not cloud_api_key.strip()):
            from dotenv import dotenv_values

            from lobster.config.global_config import get_global_credentials_path

            saved_paths = [] if global_config else [env_path]
            saved_paths.append(get_global_credentials_path())
            for saved_path in saved_paths:
                if saved_path and saved_path.is_file():
                    saved_key = dotenv_values(saved_path).get("OMICS_OS_API_KEY")
                    if saved_key and saved_key.strip():
                        cloud_api_key = saved_key
                        break
        if not cloud_api_key or not cloud_api_key.strip():
            console.print(
                h(
                    "[red]Omics-OS Cloud API key is required: supply --omics-os-key or OMICS_OS_API_KEY.[/red]"
                )
            )
            console.print(OMICS_OS_AUTH_GUIDANCE, markup=False)
            raise typer.Exit(1)
        cloud_api_key = cloud_api_key.strip()

    # Check if configuration already exists
    workspace_provider_path = (
        resolve_workspace(init_ws, create=False) / "provider_config.json"
    )
    config_exists = (
        config_path.exists() if global_config else (env_path and env_path.exists())
    )
    # The workspace may be resolved outside the cwd (LOBSTER_WORKSPACE), so an
    # earlier init from another directory leaves only provider_config.json here.
    if not global_config:
        from lobster.config.workspace_config import WorkspaceProviderConfig

        workspace_has_provider = bool(
            workspace_provider_path.exists()
            and WorkspaceProviderConfig.load(
                workspace_provider_path.parent
            ).global_provider
        )
        config_exists = config_exists or workspace_has_provider
    if cloud_selected and not global_config:
        config_exists = config_exists or workspace_provider_path.exists()
    if config_exists and not force:
        from lobster.config.global_config import get_global_credentials_path

        _repair_skipped_secret_permissions(
            get_global_credentials_path() if global_config else env_path
        )
        if cloud_selected:
            from dotenv import dotenv_values

            from lobster.config.credentials import get_api_key

            if global_config:
                from lobster.config.global_config import get_global_credentials_path

                existing = GlobalProviderConfig.load()
                existing_provider = existing.default_provider if existing else None
                credentials_path = get_global_credentials_path()
            else:
                from lobster.config.workspace_config import WorkspaceProviderConfig

                existing = WorkspaceProviderConfig.load(
                    resolve_workspace(init_ws, create=False)
                )
                existing_provider = (
                    existing.global_provider
                    if existing
                    else dotenv_values(env_path).get("LOBSTER_LLM_PROVIDER")
                )
                credentials_path = env_path
            stored_key = (
                dotenv_values(credentials_path).get("OMICS_OS_API_KEY")
                if credentials_path.exists()
                else None
            )
            existing_key = get_api_key() or stored_key
            if (
                existing_provider == "omics-os"
                and existing_key
                and existing_key.strip()
            ):
                console.print(
                    h(
                        "already configured for Omics-OS Cloud (account access not checked)."
                    )
                )
                raise typer.Exit(0)
            console.print(
                h(
                    "Existing configuration found at {v1}; rerun with --force to replace it (a backup is kept)."
                ).format(
                    v1=(
                        config_path
                        if global_config
                        else (
                            workspace_provider_path
                            if workspace_provider_path.exists()
                            else env_path
                        )
                    )
                )
            )
            raise typer.Exit(1)
        console.print()
        if global_config:
            console.print(
                Panel.fit(
                    h(
                        "[bold yellow]⚠️  Global Configuration Already Exists[/bold yellow]\n\n"
                        "A global config already exists at:\n[cyan]{config_path}[/cyan]\n\n"
                        "To reconfigure, use:\n"
                        "[bold {PRIMARY_ORANGE}]lobster init --global --force[/bold {PRIMARY_ORANGE}]\n\n"
                        "Or edit the file manually."
                    ).format(
                        config_path=config_path,
                        PRIMARY_ORANGE=LobsterTheme.PRIMARY_ORANGE,
                    ),
                    border_style="yellow",
                    padding=(1, 2),
                )
            )
        else:
            _existing_path = env_path if env_path.exists() else workspace_provider_path
            console.print(
                Panel.fit(
                    h(
                        "[bold yellow]⚠️  Configuration Already Exists[/bold yellow]\n\n"
                        "A configuration file already exists at:\n[cyan]{existing_path}[/cyan]\n\n"
                        "To reconfigure, use:\n"
                        "[bold {PRIMARY_ORANGE}]lobster init --force[/bold {PRIMARY_ORANGE}]\n\n"
                        "Or edit the file manually."
                    ).format(
                        existing_path=_markup_escape(str(_existing_path)),
                        PRIMARY_ORANGE=LobsterTheme.PRIMARY_ORANGE,
                    ),
                    border_style="yellow",
                    padding=(1, 2),
                )
            )
        console.print()
        console.print(
            h("[dim]Configuration file: {_markup_escape}[/dim]").format(
                _markup_escape=_markup_escape(
                    str(config_path if global_config else _existing_path)
                )
            )
        )
        raise typer.Exit(0)

    # If force flag and config exists, create backup and continue.
    # `--force` is already explicit consent; adding a second prompt breaks
    # the Go/questionary wizard path and creates inconsistent semantics.
    backup_path = None
    if config_exists and force:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not global_config:
            backup_sources = [
                p for p in (env_path, workspace_provider_path) if p.exists()
            ]
        else:
            backup_sources = [config_path if global_config else env_path]
        try:
            for backup_source in backup_sources:
                destination = backup_source.with_name(
                    f"{backup_source.name}.backup.{timestamp}"
                )
                if not non_interactive:
                    console.print(
                        h("[yellow]⚠️  Existing config will be backed up to:[/yellow]")
                    )
                    console.print(f"[yellow]   {destination}[/yellow]")
                _backup_init_file(backup_source, destination)
                if backup_path is None:
                    backup_path = destination
        except Exception as e:
            console.print(
                h("[red]❌ Failed to create backup: {str}[/red]").format(str=str(e))
            )
            raise typer.Exit(1)

    # Non-interactive mode: validate and create .env from parameters
    if non_interactive:
        # Detect uv tool env for non-interactive path
        from lobster.core.uv_tool_env import is_uv_tool_env as _is_uv_tool_env_ni

        _in_uv_tool_ni = _is_uv_tool_env_ni()

        # === AGENT SELECTION (BEFORE PROVIDER SETUP) ===
        # Resolve workspace path for agent config
        workspace_path = _resolve_workspace_or_exit(workspace=init_ws)

        # Handle agent selection via flags (CLI-01, CLI-02)
        if agents or preset or auto_agents:
            selected_agents, preset_name = _perform_agent_selection_non_interactive(
                agents_flag=agents,
                preset_flag=preset,
                auto_agents_flag=auto_agents,
                agents_description_flag=agents_description,
                workspace_path=workspace_path,
            )

            # Check and install missing packages without prompting.
            if skip_extras:
                _print_deferred_agent_install(selected_agents)
                installed_agents = selected_agents
            else:
                installed_agents = _check_and_prompt_install_packages(
                    selected_agents,
                    workspace_path,
                    prompt_for_install=False,
                    auto_install_missing=not _in_uv_tool_ni,
                )

            if preset_name and set(installed_agents) != set(selected_agents):
                preset_name = None

            # Save agent config BEFORE provider setup
            _save_agent_config(installed_agents, workspace_path, preset_name)
        # === END AGENT SELECTION ===

        env_lines = []
        env_lines.append("# Lobster AI Configuration")
        env_lines.append("# Generated by lobster init --non-interactive\n")

        # Initialize structured config dict for workspace config
        config_dict = {}

        # Validate provider configuration using provider_setup module
        has_anthropic = anthropic_key is not None
        has_bedrock = bedrock_access_key is not None and bedrock_secret_key is not None
        has_ollama = use_ollama
        has_gemini = gemini_key is not None
        has_openai = openai_key is not None
        has_openrouter = openrouter_key is not None
        if ollama_cloud_key is not None and not ollama_cloud_key.strip():
            console.print(h("[red]Ollama Cloud API key cannot be empty[/red]"))
            raise typer.Exit(1)
        has_ollama_cloud = ollama_cloud_key is not None
        has_nebius = nebius_key is not None
        has_azure = azure_endpoint is not None and azure_credential is not None

        # Validate at least one provider
        valid, error_msg = provider_setup.validate_provider_choice(
            has_anthropic,
            has_bedrock,
            has_ollama,
            has_gemini,
            has_openai,
            has_openrouter=has_openrouter,
            has_nebius=has_nebius,
            has_ollama_cloud=has_ollama_cloud,
            has_azure=has_azure,
        )
        if cloud_selected:
            valid = True
        if not valid:
            console.print(
                h("[red]❌ Error: {error_msg}[/red]").format(error_msg=error_msg)
            )
            console.print()
            console.print(h("You must provide one of:"))
            console.print(h("  • Claude API: --anthropic-key=xxx"))
            console.print(
                h("  • AWS Bedrock: --bedrock-access-key=xxx --bedrock-secret-key=xxx")
            )
            console.print(h("  • Ollama (Local): --use-ollama"))
            console.print(h("  • Google Gemini: --gemini-key=xxx"))
            console.print(h("  • OpenAI: --openai-key=xxx"))
            console.print(h("  • OpenRouter: --openrouter-key=xxx"))
            console.print(h("  • Nebius AI Studio: --nebius-key=xxx"))
            console.print(h("  • Ollama Cloud: --ollama-cloud-key=xxx"))
            console.print(
                h("  • Azure AI: --azure-endpoint=xxx --azure-credential=xxx")
            )
            raise typer.Exit(1)

        # Warn if multiple providers
        priority_warning = provider_setup.get_provider_priority_warning(
            has_anthropic,
            has_bedrock,
            has_ollama,
            has_gemini,
            has_openai,
            has_openrouter=has_openrouter,
            has_nebius=has_nebius,
            has_ollama_cloud=has_ollama_cloud,
            has_azure=has_azure,
        )
        if priority_warning:
            console.print(
                h("[yellow]⚠️  Warning: {priority_warning}[/yellow]").format(
                    priority_warning=priority_warning
                )
            )
        if (use_omics_os or omics_os_key is not None) and not cloud_selected:
            console.print(
                h(
                    "[yellow]Warning: an explicit provider takes priority over Omics-OS Cloud.[/yellow]"
                )
            )

        # Validate profile parameter (only relevant for Anthropic/Bedrock)
        from lobster.config.constants import (
            DEPRECATED_PROFILE_ALIASES,
            VALID_PROFILES,
        )

        valid_profiles = VALID_PROFILES + list(DEPRECATED_PROFILE_ALIASES.keys())
        selected_profile = None

        if profile:
            # Profile provided - validate it
            if profile not in valid_profiles:
                console.print(
                    h("[red]❌ Error: Invalid profile '{profile}'[/red]").format(
                        profile=profile
                    )
                )
                console.print(
                    h("Valid profiles: {join}").format(join=", ".join(VALID_PROFILES))
                )
                raise typer.Exit(1)

            # Resolve deprecated aliases
            if profile in DEPRECATED_PROFILE_ALIASES:
                new_name = DEPRECATED_PROFILE_ALIASES[profile]
                console.print(
                    h(
                        "[yellow]⚠️  Profile '{profile}' is deprecated, using '{new_name}' instead.[/yellow]"
                    ).format(profile=profile, new_name=new_name)
                )
                profile = new_name

            # Warn if profile used with Ollama (it will be ignored)
            if has_ollama and not has_anthropic and not has_bedrock:
                console.print(
                    h(
                        "[yellow]⚠️  Warning: --profile ignored for Ollama (uses local models)[/yellow]"
                    )
                )
            else:
                selected_profile = profile
        else:
            # No profile provided - use default for Anthropic/Bedrock only
            if has_anthropic or has_bedrock:
                selected_profile = "production"  # Default profile

        # Create provider configuration
        if cloud_selected:
            config = provider_setup.create_omics_os_config(cloud_api_key)
            for key, value in config.env_vars.items():
                env_lines.append(f"{key}={value}")
            config_dict["provider"] = "omics-os"
        elif has_anthropic:
            config = provider_setup.create_anthropic_config(anthropic_key)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "anthropic"
        elif has_bedrock:
            config = provider_setup.create_bedrock_config(
                bedrock_access_key, bedrock_secret_key
            )
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "bedrock"
        elif has_ollama:
            config = provider_setup.create_ollama_config(model_name=ollama_model)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "ollama"
                if ollama_model:
                    config_dict["ollama_model"] = ollama_model
                console.print(h("[green]✓ Ollama provider configured[/green]"))
        elif has_gemini:
            config = provider_setup.create_gemini_config(gemini_key)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "gemini"
                console.print(h("[green]✓ Gemini provider configured[/green]"))
        elif has_openai:
            config = provider_setup.create_openai_config(openai_key)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "openai"
                console.print(h("[green]✓ OpenAI provider configured[/green]"))
        elif has_openrouter:
            config = provider_setup.create_openrouter_config(openrouter_key)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "openrouter"
                console.print(h("[green]✓ OpenRouter provider configured[/green]"))
        elif has_nebius:
            config = provider_setup.create_nebius_config(nebius_key)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "nebius"
                console.print(
                    h("[green]✓ Nebius AI Studio provider configured[/green]")
                )
        elif has_azure:
            config = provider_setup.create_azure_config(
                azure_endpoint, azure_credential
            )
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "azure"
                console.print(h("[green]✓ Azure AI provider configured[/green]"))
        elif has_ollama_cloud:
            config = provider_setup.create_ollama_cloud_config(ollama_cloud_key)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "ollama-cloud"
                console.print(h("[green]✓ Ollama Cloud provider configured[/green]"))

        # Write profile configuration (only for Anthropic/Bedrock)
        if selected_profile:
            env_lines.append("\n# Agent Configuration Profile")
            env_lines.append("# Determines which Claude models are used for analysis")
            env_lines.append(f"LOBSTER_PROFILE={selected_profile}")
            config_dict["profile"] = selected_profile
            console.print(
                h("[green]✓ Profile set to: {selected_profile}[/green]").format(
                    selected_profile=selected_profile
                )
            )

        # Save model override (--model flag or --ollama-model for Ollama)
        # Guard: model may be a typer.OptionInfo if init_impl is called directly
        _model_str = model if isinstance(model, str) else None
        _ollama_model_str = ollama_model if isinstance(ollama_model, str) else None
        effective_model = _model_str or (_ollama_model_str if has_ollama else None)
        if effective_model:
            config_dict["model_id"] = effective_model
            console.print(
                h("[green]✓ Default model: {effective_model}[/green]").format(
                    effective_model=effective_model
                )
            )

        if ncbi_key:
            env_lines.append("\n# Optional: Enhanced literature search")
            env_lines.append(f"NCBI_API_KEY={ncbi_key.strip()}")

        if cloud_key:
            env_lines.append("\n# Omics-OS Cloud configuration (enables premium tier)")
            env_lines.append(f"LOBSTER_CLOUD_KEY={cloud_key.strip()}")
            if cloud_endpoint:
                env_lines.append(f"LOBSTER_ENDPOINT={cloud_endpoint.strip()}")
            console.print(
                h("[green]✓ Cloud API key configured (premium tier enabled)[/green]")
            )

        # Handle SSL configuration from flags
        if ssl_verify is not None or ssl_cert_path:
            env_lines.append("\n# SSL/HTTPS configuration")
            if ssl_verify is False:
                env_lines.append("LOBSTER_SSL_VERIFY=false")
                console.print(
                    h(
                        "[yellow]⚠️  SSL verification disabled (--no-ssl-verify)[/yellow]"
                    )
                )
            if ssl_cert_path:
                env_lines.append(f"LOBSTER_SSL_CERT_PATH={ssl_cert_path}")
                console.print(
                    h("[green]✓ SSL cert path: {ssl_cert_path}[/green]").format(
                        ssl_cert_path=ssl_cert_path
                    )
                )

        # Test SSL connectivity (non-interactive)
        if not skip_ssl_test and ssl_verify is not False:
            from lobster.config.ssl_setup import test_ssl_connectivity

            console.print(h("\n[dim]Testing connectivity to NCBI databases...[/dim]"))
            success, is_ssl_error, error_msg = test_ssl_connectivity(timeout=10)

            if is_ssl_error:
                console.print(
                    h("\n[yellow]⚠️  SSL Certificate Issue Detected[/yellow]")
                )
                console.print(
                    h("[dim]Error: {error_msg}...[/dim]\n").format(
                        error_msg=error_msg[:80]
                    )
                )
                console.print(
                    h(
                        "[yellow]To fix this in non-interactive mode, use one of:[/yellow]"
                    )
                )
                console.print(
                    h("  --no-ssl-verify           # Disable SSL (testing only)")
                )
                console.print(
                    h("  --ssl-cert-path=/path/to/ca.pem  # Use corporate CA")
                )
                console.print()
                console.print(
                    h(
                        "[dim]Continuing with setup - you may encounter errors later.[/dim]"
                    )
                )
            elif success:
                console.print(h("[green]✓ NCBI connectivity: OK[/green]"))
            else:
                # Non-SSL network error
                console.print(
                    h("[yellow]⚠️  Network test failed: {error_msg}[/yellow]").format(
                        error_msg=error_msg[:60]
                    )
                )
                console.print(h("[dim]You can still continue (may work later)[/dim]"))

        # === NON-INTERACTIVE: Optional package installation ===
        if not skip_extras and not _in_uv_tool_ni:
            # Standard venv — install directly via pip/uv pip
            # Auto-install provider package
            if has_anthropic:
                _ensure_provider_installed("anthropic")
            elif has_bedrock:
                _ensure_provider_installed("bedrock")
            elif has_ollama:
                _ensure_provider_installed("ollama")
            elif has_gemini:
                _ensure_provider_installed("gemini")
            elif has_openai:
                _ensure_provider_installed("openai")
            elif has_openrouter:
                _ensure_provider_installed("openrouter")
            elif has_nebius:
                _ensure_provider_installed("nebius")
            elif has_ollama_cloud:
                _ensure_provider_installed("ollama-cloud")
            elif has_azure:
                _ensure_provider_installed("azure")

            # Install docling if requested
            if install_docling:
                console.print(h("[dim]Installing docling...[/dim]"))
                from lobster.cli_internal.commands.light.agent_commands import (
                    _uv_pip_install,
                )

                s1, _ = _uv_pip_install("docling>=2.60.0")
                s2, _ = _uv_pip_install("docling-core>=2.50.0")
                if s1 and s2:
                    console.print(h("[green]✓ Docling installed[/green]"))
                else:
                    from rich.markup import escape as rich_escape

                    from lobster.core.component_registry import get_install_command

                    docling_cmd = rich_escape(
                        get_install_command("docling", is_extra=True)
                    )
                    console.print(
                        h(
                            "[yellow]⚠ Docling install failed. Run: {docling_cmd}[/yellow]"
                        ).format(docling_cmd=docling_cmd)
                    )

            # Install vector search if requested (Smart Standardization — OpenAI)
            if install_vector_search:
                if not _is_vector_search_backend_available():
                    _print_vector_backend_unavailable()
                else:
                    console.print(
                        h("[dim]Installing Smart Standardization deps...[/dim]")
                    )
                    from lobster.cli_internal.commands.light.agent_commands import (
                        _uv_pip_install,
                    )

                    s1, _ = _uv_pip_install("chromadb>=1.0.0")
                    s2, _ = _uv_pip_install("openai>=1.0.0")
                    if s1 and s2:
                        console.print(
                            h("[green]✓ Smart Standardization installed[/green]")
                        )
                        env_lines.append("LOBSTER_EMBEDDING_PROVIDER=openai")
                    else:
                        console.print(
                            h(
                                "[yellow]⚠ Install manually: {_vector_search_install_command}[/yellow]"
                            ).format(
                                _vector_search_install_command=_vector_search_install_command()
                            )
                        )

            # Extended data + TUI (silent)
            _install_extended_data()
            _ensure_tui_installed()

        # Save configuration based on mode
        if global_config:
            # Global mode: save provider config AND credentials
            try:
                from lobster.config.global_config import save_global_credentials

                # Extract credentials and settings from env_lines
                credentials = {}
                for line in env_lines:
                    if "=" in line and not line.startswith("#"):
                        key, _, value = line.partition("=")
                        key = key.strip()
                        # Save API keys and SSL settings
                        if any(
                            cred_key in key
                            for cred_key in [
                                "API_KEY",
                                "ACCESS_KEY",
                                "SECRET_KEY",
                                "CLOUD_KEY",
                                "LOBSTER_SSL",  # SSL settings (LOBSTER_SSL_VERIFY, LOBSTER_SSL_CERT_PATH)
                            ]
                        ):
                            credentials[key] = value.strip()

                # Save provider config
                if lamin is not None:
                    config_dict["lamin_enabled"] = lamin
                global_config_path = _create_global_config(config_dict)

                # Save credentials to credentials.env (with secure permissions)
                credentials_path = None
                if credentials:
                    credentials_path = save_global_credentials(credentials)

                console.print(h("[green]✅ Global configuration saved:[/green]"))
                console.print(
                    h("  • Config: {global_config_path}").format(
                        global_config_path=global_config_path
                    )
                )
                if credentials_path:
                    console.print(
                        h(
                            "  • Credentials: {credentials_path} [dim](mode: 0o600)[/dim]"
                        ).format(credentials_path=credentials_path)
                    )
                if backup_path:
                    console.print(
                        h("  • Backup: {backup_path}").format(backup_path=backup_path)
                    )
            except Exception as e:
                if config_dict.get("provider") == "omics-os":
                    from lobster.config.cloud_setup import fail_cloud_setup_write

                    fail_cloud_setup_write(e)
                console.print(
                    h("[red]❌ Failed to write global config: {str}[/red]").format(
                        str=str(e)
                    )
                )
                raise typer.Exit(1)
        else:
            # Workspace mode: write both .env and workspace config
            try:
                from lobster.config.global_config import _open_private_file

                with _open_private_file(env_path) as f:
                    f.write("\n".join(env_lines))
                    f.write("\n")
            except Exception as e:
                if config_dict.get("provider") == "omics-os":
                    from lobster.config.cloud_setup import fail_cloud_setup_write

                    fail_cloud_setup_write(e)
                console.print(
                    h("[red]❌ Failed to write .env file: {str}[/red]").format(
                        str=str(e)
                    )
                )
                raise typer.Exit(1)

            # Create workspace config from structured config_dict
            workspace_path = _resolve_workspace_or_exit(workspace=init_ws)
            _create_workspace_config(config_dict, workspace_path)
            _save_lamin_choice(lamin, workspace_path, previous=lamin_previous)

            # Success message for non-interactive
            from lobster.i18n import t

            summary_extras = [t("cli.init.done_env", path=str(env_path))]
            if backup_path:
                summary_extras.append(t("cli.init.done_backup", path=str(backup_path)))
            _print_init_summary(
                config_dict,
                workspace_path,
                summary_extras,
                next_step=not _tool_install_follows(skip_extras),
            )

        _print_lamin_notes(lamin, global_config=global_config, install=not skip_extras)
        if cloud_selected:
            from lobster.config.cloud_setup import OMICS_OS_KEY_NOT_VERIFIED

            console.print(OMICS_OS_KEY_NOT_VERIFIED)

        # In a uv tool env, run the install now. It ends the process when it
        # runs: the tool's environment is replaced on disk.
        if _in_uv_tool_ni and not skip_extras:
            provider_name_ni = None
            if has_anthropic:
                provider_name_ni = "anthropic"
            elif has_bedrock:
                provider_name_ni = "bedrock"
            elif has_ollama:
                provider_name_ni = "ollama"
            elif has_gemini:
                provider_name_ni = "gemini"
            elif has_openai:
                provider_name_ni = "openai"
            elif has_openrouter:
                provider_name_ni = "openrouter"
            elif has_nebius:
                provider_name_ni = "nebius"
            elif has_ollama_cloud:
                provider_name_ni = "ollama-cloud"
            elif has_azure:
                provider_name_ni = "azure"
            _uv_tool_env_handoff(
                provider_name_ni,
                locals().get("installed_agents"),
                skip_extras,
                include_vector_search=install_vector_search,
                include_lamin=bool(lamin),
                confirm=False,
            )
        raise typer.Exit(0)

    # Interactive mode: run wizard
    # =========================================================================
    # Guard: if stdin is not a TTY (e.g. piped from curl | bash), no
    # interactive UI path can work — Go TUI, questionary, and Rich prompts
    # all need a real terminal.  Check BEFORE dispatching to any UI.
    import sys as _sys

    if not non_interactive and not _sys.stdin.isatty():
        console.print(
            h(
                "[yellow]Cannot run interactive setup — stdin is not a terminal.[/yellow]\nRun [bold]lobster init --global[/bold] manually in your terminal to configure."
            )
        )
        raise typer.Exit(1)

    if preset:
        _check_preset_or_exit(preset, console)

    # === TRY GO TUI / QUESTIONARY FIRST (ui_mode: auto | go | classic) ===
    # The Go binary and the questionary fallback both return the same JSON dict,
    # so a single apply_tui_init_result() call handles both paths.
    # The classic Rich-prompt flow further below is used when neither is
    # available (or when --ui classic is passed explicitly).
    # =========================================================================
    if ui_mode != "classic" and not non_interactive:
        _tui_handled = False
        # Agents are saved in the workspace even with --global.
        _saved_packages = _starting_packages(
            _resolve_workspace_or_exit(create=False, workspace=init_ws), preset
        )

        # ---- Try Go TUI ----
        if not _tui_handled and ui_mode in ("auto", "go"):
            try:
                from lobster.ui.bridge.binary_finder import find_tui_binary

                _binary = find_tui_binary()
                if _binary:
                    try:
                        from lobster.ui.bridge.go_tui_bridge import run_init_wizard

                        try:
                            _tui_result = run_init_wizard(
                                _binary,
                                lamin_default=_lamin_default(lamin, global_config),
                                agent_packages=_saved_packages,
                            )
                        except KeyboardInterrupt:
                            _tui_result = {"cancelled": True}
                        if _tui_result.get("cancelled", False):
                            console.print(f"[yellow]{_cancelled_text()}[/yellow]")
                            raise typer.Exit(0)
                        # An older lobster-tui ignores the question.
                        lamin_skipped = (
                            lamin is None and "lamin_enabled" not in _tui_result
                        )
                        if lamin is not None:
                            _tui_result["lamin_enabled"] = lamin

                        _ws_path = _resolve_workspace_or_exit(workspace=init_ws)
                        _ev_path = init_root / ".env"
                        _tui_result = _postprocess_tui_init_result(
                            _tui_result,
                            workspace_path=_ws_path,
                            env_path=_ev_path,
                            global_config=global_config,
                            skip_extras=skip_extras,
                            preset=preset,
                        )

                        _print_init_summary(
                            _tui_result,
                            _ws_path,
                            next_step=not _tool_install_follows(skip_extras),
                        )
                        if lamin_skipped:
                            console.print(
                                h(
                                    "[dim]This lobster-tui did not ask about the LaminDB catalog; its setting is unchanged. Set it with `lobster init --lamin` or `--no-lamin`.[/dim]"
                                )
                            )
                        if _tool_install_follows(skip_extras):
                            _uv_tool_env_handoff(
                                provider_name=(_tui_result.get("provider") or "")
                                .strip()
                                .lower()
                                or None,
                                selected_agents=_tui_result.get("agents") or [],
                                skip_extras=skip_extras,
                                include_vector_search=bool(
                                    _tui_result.get("smart_standardization_enabled")
                                )
                                and not skip_extras,
                                include_lamin=bool(_tui_result.get("lamin_enabled")),
                            )
                        _tui_handled = True
                    except typer.Exit:
                        raise
                    except Exception as _go_exc:
                        if ui_mode == "go":
                            console.print(
                                h("[red]Go TUI init failed: {_go_exc}[/red]").format(
                                    _go_exc=_go_exc
                                )
                            )
                            raise typer.Exit(1)
                        console.print(
                            h(
                                "[dim]Go TUI unavailable ({_go_exc}), falling back.[/dim]"
                            ).format(_go_exc=_go_exc)
                        )
                elif ui_mode == "go" and os.name != "nt":
                    # --ui go was explicitly requested but binary not found
                    console.print(
                        h(
                            "[red]lobster-tui binary not found. Install it or use --ui classic.[/red]"
                        )
                    )
                    raise typer.Exit(1)
            except ImportError as _ie:
                console.print(
                    h(
                        "[dim]Go TUI bridge unavailable ({_ie}), falling back.[/dim]"
                    ).format(_ie=_ie)
                )

        # ---- Try questionary fallback (auto mode only, when Go TUI was unavailable) ----
        if not _tui_handled and ui_mode == "auto":
            try:
                from lobster.ui.bridge.questionary_fallback import run_questionary_init
                from lobster.ui.wizard.manifest import build_init_manifest

                try:
                    _q_lang = _ask_language(use_questionary=True)
                except KeyboardInterrupt:
                    console.print(f"[yellow]{_cancelled_text()}[/yellow]")
                    raise typer.Exit(0)
                _q_result = run_questionary_init(
                    build_init_manifest(),
                    lamin_default=_lamin_default(lamin, global_config),
                    agent_packages=_saved_packages,
                )
                if lamin is not None:
                    _q_result["lamin_enabled"] = lamin
                if _q_lang:
                    _q_result["lang"] = _q_lang
                if _q_result.get("cancelled", False):
                    console.print(f"[yellow]{_cancelled_text()}[/yellow]")
                    raise typer.Exit(0)

                _ws_path = _resolve_workspace_or_exit(workspace=init_ws)
                _ev_path = init_root / ".env"
                _q_result = _postprocess_tui_init_result(
                    _q_result,
                    workspace_path=_ws_path,
                    env_path=_ev_path,
                    global_config=global_config,
                    skip_extras=skip_extras,
                    preset=preset,
                )

                _print_init_summary(
                    _q_result,
                    _ws_path,
                    next_step=not _tool_install_follows(skip_extras),
                )
                if _tool_install_follows(skip_extras):
                    _uv_tool_env_handoff(
                        provider_name=(_q_result.get("provider") or "").strip().lower()
                        or None,
                        selected_agents=_q_result.get("agents") or [],
                        skip_extras=skip_extras,
                        include_vector_search=bool(
                            _q_result.get("smart_standardization_enabled")
                        )
                        and not skip_extras,
                        include_lamin=bool(_q_result.get("lamin_enabled")),
                    )
                _tui_handled = True
            except typer.Exit:
                raise
            except ImportError:
                # questionary not installed — fall through to classic Rich prompts silently
                pass
            except Exception as _q_exc:
                console.print(
                    h(
                        "[dim]Questionary wizard failed ({_q_exc}), falling back to classic mode.[/dim]"
                    ).format(_q_exc=_q_exc)
                )

        if _tui_handled:
            raise typer.Exit(0)
    # === END GO TUI / QUESTIONARY PATH ===

    _ask_language(use_questionary=False)
    console.print("\n")
    if global_config:
        from lobster.config.global_config import get_global_credentials_path

        creds_path = get_global_credentials_path()
        console.print(
            Panel.fit(
                h(
                    "[bold white]🦞 Welcome to Lobster AI![/bold white]\n\n"
                    "Let's set up your [cyan]global[/cyan] provider defaults.\n"
                    "Config: [cyan]{config_path}[/cyan]\n"
                    "Credentials: [cyan]{creds_path}[/cyan]\n\n"
                    "[dim]These defaults apply to all workspaces without their own config.[/dim]"
                ).format(config_path=config_path, creds_path=creds_path),
                border_style="bright_blue",
                padding=(1, 2),
            )
        )
    else:
        console.print(
            Panel.fit(
                (
                    h(
                        "[bold white]🦞 Welcome to Lobster AI![/bold white]\n\n"
                        "Let's set up your API keys.\n"
                        "This wizard will create a [cyan].env[/cyan] file in your current directory."
                    )
                    if folder is None
                    else h(
                        "[bold white]🦞 Welcome to Lobster AI![/bold white]\n\n"
                        "Let's set up your API keys.\n"
                        "This wizard will create a [cyan].env[/cyan] file in {folder}."
                    ).format(folder=_markup_escape(str(init_root)))
                ),
                border_style="bright_blue",
                padding=(1, 2),
            )
        )
    console.print()

    # Detect uv tool env early so we can adjust messaging
    from lobster.core.uv_tool_env import is_uv_tool_env as _is_uv_tool_env

    _in_uv_tool = _is_uv_tool_env()
    if _in_uv_tool:
        console.print(
            h(
                "[dim]  Detected uv tool installation. API keys will be configured here;\n  any new packages will be installed after setup completes.[/dim]\n"
            )
        )

    workspace_path = _resolve_workspace_or_exit(workspace=init_ws)

    try:
        env_lines = []
        env_lines.append("# Lobster AI Configuration")
        env_lines.append("# Generated by lobster init\n")

        # Initialize structured config dict for workspace config
        config_dict = {}

        # ================================================================== #
        # Step 1 — Provider selection (unified list)                          #
        # ================================================================== #
        provider_menu = [
            h("Omics-OS Cloud          — Omics-OS Cloud account"),
            h("Claude API (Anthropic)  — direct access to Claude models"),
            h("AWS Bedrock             — production, enterprise"),
            h("Ollama (local)          — privacy, zero cost, offline"),
            h("Google Gemini           — Gemini models"),
            h("Azure AI                — enterprise Azure deployments"),
            h("OpenAI                  — GPT models"),
            h("OpenRouter              — 600+ models via one API key"),
            h("Nebius AI Studio        — open-weight models"),
            h("Ollama Cloud            — hosted models, requires API key"),
        ]
        provider_map = {
            "1": "omics-os",
            "2": "anthropic",
            "3": "bedrock",
            "4": "ollama",
            "5": "gemini",
            "6": "azure",
            "7": "openai",
            "8": "openrouter",
            "9": "nebius",
            "10": "ollama-cloud",
        }
        default = {"default": "1"}
        while True:
            console.print(h("[bold white]Select your LLM provider:[/bold white]"))
            for _n, _label in enumerate(provider_menu, start=1):
                console.print(f"  [cyan]{_n:>2}[/cyan] - {_label}")
            provider = Prompt.ask(
                h("[bold white]Choose provider[/bold white]"),
                choices=list(provider_map),
                **default,
            )
            provider_name = provider_map[provider]
            if provider_name != "omics-os" or _has_omics_os_account():
                break
            default = {}  # back to the list with Omics-OS not picked
            console.print()

        # ================================================================== #
        # Step 2 — Credentials (provider-specific)                            #
        # ================================================================== #
        if provider_name == "omics-os":
            from lobster.config.cloud_setup import (
                OMICS_OS_AUTH_GUIDANCE,
                OMICS_OS_KEY_NOT_VERIFIED,
            )

            console.print(OMICS_OS_AUTH_GUIDANCE, markup=False)
            api_key = Prompt.ask(
                h("[bold white]Enter your Omics-OS API key[/bold white]"),
                password=True,
                default="",
            )
            config = provider_setup.create_omics_os_config(api_key)
            if not config.success:
                console.print(f"[red]❌ {config.message}[/red]")
                raise typer.Exit(1)
            for key, value in config.env_vars.items():
                env_lines.append(f"{key}={value}")
            config_dict["provider"] = "omics-os"

        elif provider_name == "anthropic":
            console.print(h("\n[bold white]🔑 Claude API Configuration[/bold white]"))
            console.print(
                h(
                    "Get your API key from: [link]https://console.anthropic.com/[/link]\n"
                )
            )
            api_key = Prompt.ask(
                h("[bold white]Enter your Claude API key[/bold white]"), password=True
            )
            if not api_key.strip():
                console.print(h("[red]❌ API key cannot be empty[/red]"))
                raise typer.Exit(1)
            env_lines.append(f"ANTHROPIC_API_KEY={api_key.strip()}")
            config_dict["provider"] = "anthropic"

        elif provider_name == "bedrock":
            console.print(h("\n[bold white]🔑 AWS Bedrock Configuration[/bold white]"))
            console.print(
                h(
                    "You'll need AWS access key and secret key with Bedrock permissions.\n"
                )
            )
            access_key = Prompt.ask(
                h("[bold white]Enter your AWS access key[/bold white]"), password=True
            )
            secret_key = Prompt.ask(
                h("[bold white]Enter your AWS secret key[/bold white]"), password=True
            )
            if not access_key.strip() or not secret_key.strip():
                console.print(h("[red]❌ AWS credentials cannot be empty[/red]"))
                raise typer.Exit(1)
            env_lines.append(f"AWS_BEDROCK_ACCESS_KEY={access_key.strip()}")
            env_lines.append(f"AWS_BEDROCK_SECRET_ACCESS_KEY={secret_key.strip()}")
            config_dict["provider"] = "bedrock"

        elif provider_name == "ollama":
            console.print(
                h("\n[bold white]🏠 Ollama (Local LLM) Configuration[/bold white]")
            )
            console.print(h("Ollama runs models locally - no API keys needed!\n"))
            ollama_status = provider_setup.get_ollama_status()

            if not ollama_status.installed:
                console.print(
                    h("[yellow]⚠️  Ollama is not installed on this system.[/yellow]")
                )
                install_instructions = provider_setup.get_ollama_install_instructions()
                console.print(
                    h("  • macOS/Linux: {install_instructions}").format(
                        install_instructions=install_instructions["macos_linux"]
                    )
                )
                console.print(
                    h("  • Windows: {install_instructions}").format(
                        install_instructions=h(install_instructions["windows"])
                    )
                )
                console.print()
                install_later = Confirm.ask(
                    h("Configure for Ollama anyway? (you can install it later)"),
                    default=True,
                )
                if not install_later:
                    console.print(
                        h(
                            "[yellow]Please install Ollama first, then run 'lobster init' again[/yellow]"
                        )
                    )
                    raise typer.Exit(0)

            model_name = None
            if ollama_status.running and ollama_status.models:
                console.print(h("[green]✓ Ollama is installed and running[/green]"))
                console.print(h("\n[bold white]Available models:[/bold white]"))
                for m in ollama_status.models[:5]:
                    console.print(f"  • {m}")
                if len(ollama_status.models) > 5:
                    console.print(
                        h("  ... and {v1} more").format(
                            v1=len(ollama_status.models) - 5
                        )
                    )
                console.print()
                use_custom = Confirm.ask(
                    h("Specify a model? (default: {DEFAULT_OLLAMA_MODEL})").format(
                        DEFAULT_OLLAMA_MODEL=provider_setup.DEFAULT_OLLAMA_MODEL
                    ),
                    default=False,
                )
                if use_custom:
                    model_name = Prompt.ask(
                        h("[bold white]Enter model name[/bold white]"),
                        default=provider_setup.DEFAULT_OLLAMA_MODEL,
                    )
            else:
                if ollama_status.installed:
                    console.print(h("[green]✓ Ollama is installed[/green]"))
                    if not ollama_status.running:
                        console.print(
                            h(
                                "[yellow]⚠️  Ollama server is not running. Start with: ollama serve[/yellow]"
                            )
                        )

            config = provider_setup.create_ollama_config(model_name=model_name)
            for key, value in config.env_vars.items():
                env_lines.append(f"{key}={value}")
            config_dict["provider"] = "ollama"
            if model_name:
                config_dict["ollama_model"] = model_name
            console.print(h("[green]✓ Ollama provider configured[/green]"))

        elif provider_name == "gemini":
            console.print(
                h("\n[bold white]🔑 Google Gemini Configuration[/bold white]")
            )
            console.print(
                h(
                    "Get your API key from: [link]https://aistudio.google.com/apikey[/link]\n"
                )
            )
            api_key = Prompt.ask(
                h("[bold white]Enter your Google API key[/bold white]"), password=True
            )
            if not api_key.strip():
                console.print(h("[red]❌ API key cannot be empty[/red]"))
                raise typer.Exit(1)
            config = provider_setup.create_gemini_config(api_key)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "gemini"
                console.print(h("[green]✓ Gemini provider configured[/green]"))

        elif provider_name == "azure":
            console.print(h("\n[bold white]🔑 Azure AI Configuration[/bold white]"))
            console.print(
                h("You'll need Azure AI Foundry endpoint and API credential.\n")
            )
            endpoint = Prompt.ask(
                h("[bold white]Enter your Azure AI endpoint[/bold white]"),
                default="https://your-project.inference.ai.azure.com/",
            )
            credential = Prompt.ask(
                h("[bold white]Enter your Azure API credential[/bold white]"),
                password=True,
            )
            if not endpoint.strip() or not credential.strip():
                console.print(
                    h("[red]❌ Azure endpoint and credential cannot be empty[/red]")
                )
                raise typer.Exit(1)
            config = provider_setup.create_azure_config(endpoint, credential)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "azure"
                console.print(h("[green]✓ Azure AI provider configured[/green]"))
            else:
                console.print(
                    h("[red]❌ Configuration failed: {message}[/red]").format(
                        message=config.message
                    )
                )
                raise typer.Exit(1)

        elif provider_name == "openai":
            console.print(h("\n[bold white]🔑 OpenAI Configuration[/bold white]"))
            console.print(
                h(
                    "Get your API key from: [link]https://platform.openai.com/api-keys[/link]\n"
                )
            )
            api_key = Prompt.ask(
                h("[bold white]Enter your OpenAI API key[/bold white]"), password=True
            )
            if not api_key.strip():
                console.print(h("[red]❌ API key cannot be empty[/red]"))
                raise typer.Exit(1)
            config = provider_setup.create_openai_config(api_key)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "openai"
                console.print(h("[green]✓ OpenAI provider configured[/green]"))

        elif provider_name == "openrouter":
            console.print(h("\n[bold white]🔀 OpenRouter Configuration[/bold white]"))
            console.print(
                h("Get your API key from: [link]https://openrouter.ai/keys[/link]\n")
            )
            api_key = Prompt.ask(
                h("[bold white]Enter your OpenRouter API key[/bold white]"),
                password=True,
            )
            if not api_key.strip():
                console.print(h("[red]❌ API key cannot be empty[/red]"))
                raise typer.Exit(1)
            config = provider_setup.create_openrouter_config(api_key)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "openrouter"
                console.print(h("[green]✓ OpenRouter provider configured[/green]"))
            else:
                console.print(
                    h("[red]❌ Configuration failed: {message}[/red]").format(
                        message=config.message
                    )
                )
                raise typer.Exit(1)

        elif provider_name == "nebius":
            console.print(
                h("\n[bold white]🧬 Nebius AI Studio Configuration[/bold white]")
            )
            console.print(h("Get your API key from: [link]https://nebius.com[/link]\n"))
            api_key = Prompt.ask(
                h("[bold white]Enter your Nebius API key[/bold white]"), password=True
            )
            if not api_key.strip():
                console.print(h("[red]❌ API key cannot be empty[/red]"))
                raise typer.Exit(1)
            config = provider_setup.create_nebius_config(api_key)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "nebius"
                console.print(
                    h("[green]✓ Nebius AI Studio provider configured[/green]")
                )
            else:
                console.print(
                    h("[red]❌ Configuration failed: {message}[/red]").format(
                        message=config.message
                    )
                )
                raise typer.Exit(1)

        elif provider_name == "ollama-cloud":
            console.print(h("\n[bold white]🧬 Ollama Cloud Configuration[/bold white]"))
            console.print(
                h(
                    "Get your API key from: [link]https://ollama.com/settings/keys[/link]\n"
                )
            )
            api_key = Prompt.ask(
                h("[bold white]Enter your Ollama Cloud API key[/bold white]"),
                password=True,
            )
            if not api_key.strip():
                console.print(h("[red]❌ API key cannot be empty[/red]"))
                raise typer.Exit(1)
            config = provider_setup.create_ollama_cloud_config(api_key)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "ollama-cloud"
                console.print(h("[green]✓ Ollama Cloud provider configured[/green]"))
            else:
                console.print(
                    h("[red]❌ Configuration failed: {message}[/red]").format(
                        message=config.message
                    )
                )
                raise typer.Exit(1)

        # Auto-install provider package if missing
        if (
            not skip_extras
            and not _in_uv_tool
            and provider_name not in ("omics-os", "ollama")
        ):
            _ensure_provider_installed(provider_name)

        # ================================================================== #
        # Step 3 — Profile selection (Anthropic / Bedrock only)               #
        # ================================================================== #
        if provider_name in ("anthropic", "bedrock"):
            console.print(
                h("\n[bold white]⚙️  Agent Configuration Profile[/bold white]")
            )
            console.print(h("Choose which Claude models to use for analysis:"))
            console.print()
            console.print(
                h(
                    "  [cyan]1[/cyan] - Development  (Sonnet 4 - fastest, most affordable)"
                )
            )
            console.print(
                h(
                    "  [cyan]2[/cyan] - Production   (Sonnet 4 + Sonnet 4.5 supervisor) [recommended]"
                )
            )
            console.print(
                h("  [cyan]3[/cyan] - Performance  (Sonnet 4.5 - highest quality)")
            )
            console.print(
                h(
                    "  [cyan]4[/cyan] - Max          (Opus 4.5 supervisor - most capable, most expensive)"
                )
            )
            console.print()

            profile_choice = Prompt.ask(
                h("[bold white]Choose profile[/bold white]"),
                choices=["1", "2", "3", "4"],
                default="2",
            )

            profile_map = {
                "1": "development",
                "2": "production",
                "3": "performance",
                "4": "max",
            }
            profile_to_write = profile_map[profile_choice]

            env_lines.append("\n# Agent Configuration Profile")
            env_lines.append("# Determines which Claude models are used for analysis")
            env_lines.append(f"LOBSTER_PROFILE={profile_to_write}")
            config_dict["profile"] = profile_to_write
            console.print(
                h("[green]✓ Profile set to: {profile_to_write}[/green]").format(
                    profile_to_write=profile_to_write
                )
            )

        # ================================================================== #
        # Step 4 — Agent selection                                            #
        # ================================================================== #
        selected_agents, preset_name = _perform_agent_selection_interactive(
            workspace_path, skip_extras=skip_extras, preset=preset
        )
        if selected_agents:
            _save_agent_config(selected_agents, workspace_path, preset_name)
        console.print()

        # ================================================================== #
        # Step 5 — Optional NCBI key (single)                                #
        # ================================================================== #
        add_ncbi = Confirm.ask(
            h("Add an NCBI API key? (enhances literature search, optional)"),
            default=False,
        )

        if add_ncbi:
            ncbi_key_val = Prompt.ask(
                h("[bold white]Enter your NCBI API key[/bold white]"), password=True
            )
            if ncbi_key_val.strip():
                env_lines.append("\n# NCBI API key for literature search")
                env_lines.append(f"NCBI_API_KEY={ncbi_key_val.strip()}")
                console.print(h("[green]✓ NCBI API key configured[/green]"))

        # ================================================================== #
        # Step 6 — Optional Cloud key (skip if Omics-OS)                      #
        # ================================================================== #
        if provider_name != "omics-os":
            add_cloud = Confirm.ask(
                h("Add an Omics-OS Cloud API key? (enables premium tier, optional)"),
                default=False,
            )
            if add_cloud:
                cloud_key_val = Prompt.ask(
                    h("[bold white]Enter your Omics-OS Cloud API key[/bold white]"),
                    password=True,
                )
                if cloud_key_val.strip():
                    env_lines.append("\n# Omics-OS Cloud configuration")
                    env_lines.append(f"LOBSTER_CLOUD_KEY={cloud_key_val.strip()}")
                    console.print(h("[green]✓ Cloud API key configured[/green]"))

        # ================================================================== #
        # Step 7 — Smart Standardization (if not uv tool env)                 #
        # ================================================================== #
        if not skip_extras and not _in_uv_tool:
            smart_std_lines = _prompt_smart_standardization(
                selected_agents=selected_agents,
            )
            env_lines.extend(smart_std_lines)

        # Step 8 — LaminDB catalog (default No)
        if lamin is None:
            lamin = _ask_lamin(global_config)
        if global_config and lamin is not None:
            config_dict["lamin_enabled"] = lamin

        # ================================================================== #
        # Save configuration                                                  #
        # ================================================================== #
        console.print()
        if global_config:
            try:
                from lobster.config.global_config import save_global_credentials

                credentials = {}
                for line in env_lines:
                    if "=" in line and not line.startswith("#"):
                        key, _, value = line.partition("=")
                        key = key.strip()
                        if any(
                            tok in key
                            for tok in [
                                "API_KEY",
                                "ACCESS_KEY",
                                "SECRET_KEY",
                                "CLOUD_KEY",
                            ]
                        ):
                            credentials[key] = value.strip()

                global_config_path = _create_global_config(config_dict)

                credentials_path = None
                if credentials:
                    credentials_path = save_global_credentials(credentials)

                success_message = h(
                    "[bold green]✅ Global configuration saved![/bold green]\n\n"
                )
                success_message += h("Config: [cyan]{path}[/cyan]\n").format(
                    path=global_config_path
                )
                if credentials_path:
                    success_message += h(
                        "Credentials: [cyan]{path}[/cyan] [dim](secure: mode 0o600)[/dim]\n"
                    ).format(path=credentials_path)
                if backup_path:
                    success_message += h("Backup: [cyan]{path}[/cyan]\n").format(
                        path=backup_path
                    )
                success_message += "\n"
            except Exception as e:
                if config_dict.get("provider") == "omics-os":
                    from lobster.config.cloud_setup import fail_cloud_setup_write

                    fail_cloud_setup_write(e)
                console.print(
                    h("[red]❌ Failed to write global config: {str}[/red]").format(
                        str=str(e)
                    )
                )
                raise typer.Exit(1)
        else:
            try:
                from lobster.config.global_config import _open_private_file

                with _open_private_file(env_path) as f:
                    f.write("\n".join(env_lines))
                    f.write("\n")
            except Exception as error:
                if config_dict.get("provider") == "omics-os":
                    from lobster.config.cloud_setup import fail_cloud_setup_write

                    fail_cloud_setup_write(error)
                raise

            _create_workspace_config(config_dict, workspace_path)
            _save_lamin_choice(lamin, workspace_path, previous=lamin_previous)

            success_message = h("[bold green]✅ Configuration saved![/bold green]\n\n")
            success_message += h("Environment: [cyan]{path}[/cyan]\n").format(
                path=env_path
            )
            success_message += h("Workspace:   [cyan]{path}[/cyan]\n").format(
                path=workspace_path / "provider_config.json"
            )
            if backup_path:
                success_message += h("Backup: [cyan]{path}[/cyan]\n\n").format(
                    path=backup_path
                )
            else:
                success_message += "\n"

        if config_dict.get("provider") == "omics-os":
            console.print(OMICS_OS_KEY_NOT_VERIFIED)
        # A failed package install replaces the "run lobster chat" next step.
        failed_installs = _failed_install_lines()
        if failed_installs:
            success_message += "\n".join(
                f"[yellow]{_markup_escape(line)}[/yellow]" for line in failed_installs
            )
        elif _in_uv_tool and not skip_extras:
            pass  # the install says the next step
        elif _in_other_folder():
            success_message += _bold_command(_next_chat_text())
        else:
            success_message += h(
                "[bold white]Next step:[/bold white] Run [bold {PRIMARY_ORANGE}]lobster chat[/bold {PRIMARY_ORANGE}] to start analyzing!"
            ).format(PRIMARY_ORANGE=LobsterTheme.PRIMARY_ORANGE)
        console.print(
            Panel.fit(
                success_message,
                border_style="yellow" if failed_installs else "green",
            )
        )
        _print_lamin_notes(lamin, global_config=global_config, install=not skip_extras)
        console.print()

        # In uv tool env, offer to run `uv tool install` for new packages
        if _in_uv_tool and not skip_extras:
            _want_vector_search = False
            if not skip_extras and _is_vector_search_backend_available():
                try:
                    import chromadb  # noqa: F401
                except ImportError:
                    _vs_choice = Prompt.ask(
                        h(
                            "\n  Include Smart Standardization (ontology matching, OpenAI embeddings)?"
                        ),
                        choices=["y", "n"],
                        default="n",
                    )
                    _want_vector_search = _vs_choice == "y"
            elif not skip_extras:
                _print_vector_backend_unavailable()

            _uv_tool_env_handoff(
                provider_name=provider_name,
                selected_agents=selected_agents,
                skip_extras=skip_extras,
                include_vector_search=_want_vector_search,
                include_lamin=bool(lamin),
            )

    except KeyboardInterrupt:
        console.print(h("\n[yellow]⚠️  Configuration cancelled[/yellow]"))
        raise typer.Exit(1)
    except Exception as e:
        console.print(
            h("\n[red]❌ Configuration failed: {str}[/red]").format(str=str(e))
        )
        raise typer.Exit(1)
