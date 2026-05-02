"""
Init command body and helper functions.

Extracted from cli.py -- the `lobster init` wizard and all supporting
helper functions for provider setup, agent selection, and package installation.
"""

import logging
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import typer
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from lobster.config import provider_setup
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
    (
        "lobster-research",
        "Literature search & data discovery",
        ["research_agent", "data_expert_agent"],
        True,
        False,
    ),
    (
        "lobster-transcriptomics",
        "Single-cell & bulk RNA-seq analysis",
        ["transcriptomics_expert", "annotation_expert", "de_analysis_expert"],
        True,
        False,
    ),
    (
        "lobster-visualization",
        "Data visualization & plotting",
        ["visualization_expert_agent"],
        True,
        False,
    ),
    (
        "lobster-genomics",
        "Genomics/DNA analysis (VCF, GWAS, clinical variants)",
        ["genomics_expert", "variant_analysis_expert"],
        True,
        False,
    ),
    (
        "lobster-proteomics",
        "Mass spec & affinity proteomics analysis",
        [
            "proteomics_expert",
            "proteomics_de_analysis_expert",
            "biomarker_discovery_expert",
        ],
        True,
        False,
    ),
    (
        "lobster-metabolomics",
        "LC-MS, GC-MS, and NMR metabolomics analysis",
        ["metabolomics_expert"],
        True,
        False,
    ),
    (
        "lobster-ml",
        "Machine learning, feature selection, and survival analysis",
        [
            "machine_learning_expert",
            "feature_selection_expert",
            "survival_analysis_expert",
        ],
        True,
        True,
    ),
    (
        "lobster-drug-discovery",
        "Drug discovery, cheminformatics, and translational strategy",
        [
            "drug_discovery_expert",
            "cheminformatics_expert",
            "clinical_dev_expert",
            "pharmacogenomics_expert",
        ],
        True,
        True,
    ),
    (
        "lobster-metadata",
        "Metadata filtering & standardization",
        ["metadata_assistant"],
        True,
        True,
    ),
    (
        "lobster-structural-viz",
        "Protein structure visualization (PyMOL, PDB)",
        ["protein_structure_visualization_expert"],
        True,
        True,
    ),
]

_DEFAULT_TUI_AGENT_PACKAGES = ("lobster-research", "lobster-transcriptomics")


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
        title="Available Agents", box=box.ROUNDED, show_header=True, header_style="bold"
    )
    table.add_column("#", style="cyan", width=4)
    table.add_column("Name", style="white")
    table.add_column("Description", style="dim")
    table.add_column("Tier", style="yellow")

    # Sort agents alphabetically
    sorted_agents = sorted(agents_dict.keys())
    for i, name in enumerate(sorted_agents, 1):
        config = agents_dict[name]
        desc = getattr(config, "description", "")[:50]
        tier = getattr(config, "tier_requirement", "free") or "free"
        table.add_row(str(i), name, desc, tier)

    console.print(table)
    return sorted_agents


def _prompt_manual_agent_selection(workspace_path: Path) -> list[str]:
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
        title="Available Agent Packages",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
    )
    table.add_column("#", style="cyan", width=3)
    table.add_column("Package", style="white")
    table.add_column("Description", style="dim")
    table.add_column("Agents", style="yellow")
    table.add_column("Status", style="green", width=12)

    for i, (pkg_name, desc, agents, published, experimental) in enumerate(
        AVAILABLE_AGENT_PACKAGES, 1
    ):
        # Check if this package's agents are installed
        is_installed = any(a in installed_agent_names for a in agents)
        if is_installed:
            status = "[green]installed[/green]"
        elif not published:
            status = "[dim]coming soon[/dim]"
        elif experimental:
            status = "[yellow]experimental[/yellow]"
        else:
            status = ""
        agents_str = ", ".join(agents[:2])  # Show first 2 agents
        if len(agents) > 2:
            agents_str += f" +{len(agents) - 2}"
        table.add_row(str(i), pkg_name, desc, agents_str, status)

    console.print(table)

    console.print(
        "\n[bold white]Select packages to install (comma-separated):[/bold white]"
    )
    console.print("[dim]Example: 1,2,3 or 1-4 or 'all' for all packages[/dim]")
    console.print("[dim]Already installed packages will be skipped[/dim]")

    selection = Prompt.ask("[bold white]Selection[/bold white]", default="all")

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

    # Install missing packages
    if packages_to_install:
        from lobster.cli_internal.commands.light.agent_commands import _uv_pip_install

        console.print(
            f"\n[dim]Installing {len(packages_to_install)} package(s)...[/dim]"
        )
        for pkg in packages_to_install:
            console.print(f"  [dim]Installing {pkg}...[/dim]")
            success, msg = _uv_pip_install(pkg)
            if success:
                console.print(f"  [green]✓[/green] {pkg}")
            else:
                console.print(f"  [yellow]⚠[/yellow] {pkg} failed: {msg[:50]}")
    elif selected_indices:
        console.print("[dim]All selected packages already installed[/dim]")

    return list(dict.fromkeys(all_agents))  # Remove duplicates preserving order


def _prompt_automatic_agent_selection(workspace_path: Path) -> list[str]:
    """LLM-assisted automatic agent selection. Returns list of selected agent names."""
    from rich.console import Console
    from rich.prompt import Confirm, Prompt

    console = Console()

    console.print("\n[bold white]Describe your workflow and data types:[/bold white]")
    console.print(
        "[dim]Example: I have single-cell RNA-seq data and want to identify cell types and run differential expression[/dim]"
    )

    description = Prompt.ask("[bold white]Workflow description[/bold white]")

    if not description.strip():
        console.print(
            "[yellow]No description provided. Falling back to manual selection.[/yellow]"
        )
        return _prompt_manual_agent_selection(workspace_path)

    # Call Omics-OS endpoint
    console.print("\n[dim]Contacting Omics-OS service for agent suggestions...[/dim]")

    try:
        from lobster.config.agent_config_endpoint import suggest_agents

        result = suggest_agents(description)
    except ImportError:
        result = None

    if result is None:
        console.print(
            "[yellow]Can't reach Omics-OS service. Falling back to manual selection.[/yellow]"
        )
        return _prompt_manual_agent_selection(workspace_path)

    # Show suggestions with reasoning
    suggested_agents = result.get("agents", [])
    reasoning = result.get("reasoning", {})

    if not suggested_agents:
        console.print(
            "[yellow]No agents suggested. Falling back to manual selection.[/yellow]"
        )
        return _prompt_manual_agent_selection(workspace_path)

    console.print("\n[bold green]Suggested agents:[/bold green]")
    for agent in suggested_agents:
        reason = reasoning.get(agent, "")
        console.print(f"  [cyan]v[/cyan] {agent}")
        if reason:
            console.print(f"      [dim]{reason}[/dim]")

    # Allow user to modify
    if not Confirm.ask(
        "\n[bold white]Accept these suggestions?[/bold white]", default=True
    ):
        console.print("[dim]Opening manual selection to modify...[/dim]")
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
            "\n[dim]  Agent packages will be added via uv tool install after setup completes.[/dim]"
        )
        return normalized_agents

    console.print("\n[yellow]These agent packages are not installed:[/yellow]")

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
        console.print("\n[bold white]Install these packages?[/bold white]")
        for pkg in sorted(packages_to_install):
            console.print(f"  [cyan]{pkg}[/cyan]")

        should_install = auto_install_missing
        if prompt_for_install and not auto_install_missing:
            should_install = Confirm.ask(
                "\n[bold white]Install now?[/bold white]", default=True
            )

        if should_install:
            from lobster.cli_internal.commands.light.agent_commands import (
                _uv_pip_install,
            )

            for pkg in sorted(packages_to_install):
                console.print(f"[dim]Installing {pkg}...[/dim]")
                success, msg = _uv_pip_install(pkg)
                if success:
                    console.print(f"[green]v[/green] {pkg} installed")
                    confirmed_names.update(packages_to_install[pkg])
                else:
                    console.print(
                        f"[yellow]Warning: Failed to install {pkg}: {msg}[/yellow]"
                    )
        elif prompt_for_install:
            console.print("[dim]Skipping package installation for now[/dim]")
        else:
            console.print(
                "[dim]Skipping package installation in non-interactive mode[/dim]"
            )

    unresolved = [agent for agent in normalized_agents if agent not in confirmed_names]
    if unresolved:
        console.print(
            "\n[yellow]Skipping uninstalled agents:[/yellow] " + ", ".join(unresolved)
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

    from lobster.config.agent_presets import expand_preset, is_valid_preset

    console = Console()

    # Priority: preset > agents > auto_agents
    if preset_flag:
        if not is_valid_preset(preset_flag):
            console.print(f"[red]Invalid preset: {preset_flag}[/red]")
            console.print(
                "[dim]Valid presets: scrna-basic, scrna-full, multiomics-full[/dim]"
            )
            raise typer.Exit(1)

        expanded = expand_preset(preset_flag)
        expanded = _filter_unpublished_preset_agents(expanded, console, preset_flag)
        console.print(
            f"[green]v[/green] Using preset: {preset_flag} ({len(expanded)} agents)"
        )
        return expanded, preset_flag

    elif agents_flag:
        # Parse comma-separated list
        agents = [a.strip() for a in agents_flag.split(",") if a.strip()]
        if not agents:
            console.print("[red]No agents specified in --agents flag[/red]")
            raise typer.Exit(1)

        console.print(
            f"[green]v[/green] Enabling {len(agents)} agent(s): {', '.join(agents)}"
        )
        return agents, None

    elif auto_agents_flag:
        if not agents_description_flag:
            console.print("[red]--auto-agents requires --agents-description[/red]")
            raise typer.Exit(1)

        console.print("[dim]Contacting Omics-OS service for agent suggestions...[/dim]")

        try:
            from lobster.config.agent_config_endpoint import suggest_agents

            result = suggest_agents(agents_description_flag)
        except ImportError:
            result = None

        if result is None:
            console.print("[yellow]Can't reach Omics-OS service.[/yellow]")
            console.print(
                "[yellow]Use --agents or --preset instead for non-interactive mode.[/yellow]"
            )
            raise typer.Exit(1)

        suggested = result.get("agents", [])
        if not suggested:
            console.print(
                "[yellow]No agents suggested. Use --agents or --preset instead.[/yellow]"
            )
            raise typer.Exit(1)

        console.print(
            f"[green]v[/green] Auto-selected {len(suggested)} agents: {', '.join(suggested)}"
        )
        return suggested, None

    # No agent selection flags - use all installed agents (default behavior)
    agents_dict = _get_installed_agents()
    all_agents = list(agents_dict.keys())
    console.print(
        f"[dim]Using all {len(all_agents)} installed agents (no agent selection flags provided)[/dim]"
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
            "[yellow]⚠️  Preset includes agents not published on public PyPI; "
            f"skipping for this install surface: {', '.join(skipped)}[/yellow]"
        )
        console.print(
            f"[dim]Preset '{preset_name}' will be saved as explicit agents instead.[/dim]"
        )

    return filtered


def _build_uv_tool_init_command(
    provider_name: str | None,
    selected_agents: list[str] | None,
    *,
    include_vector_search: bool = False,
) -> list[str]:
    """Build the uv tool install command needed after init."""
    import importlib

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
                extras.append(provider_name)

    if include_vector_search:
        try:
            importlib.import_module("chromadb")
        except ImportError:
            extras.append("vector-search")

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
        if package_name not in with_packages:
            with_packages.append(package_name)

    return build_tool_install_command(
        extras=extras or None,
        with_packages=with_packages or None,
    )


def _perform_agent_selection_interactive(workspace_path: Path) -> tuple[list[str], str]:
    """Handle interactive agent selection. Returns (selected_agents, preset_name or None)."""
    from rich.console import Console

    console = Console()

    console.print("\n[bold white]Agent Selection[/bold white]")
    console.print("Select which agents to enable for your workspace:")
    console.print()

    selected = _prompt_manual_agent_selection(workspace_path)

    if selected:
        # Check for missing packages
        selected = _check_and_prompt_install_packages(selected, workspace_path)
        console.print(f"\n[green]v[/green] {len(selected)} agent(s) configured")

    return selected, None


_SMART_STD_AGENT_DISPLAY = {
    "metadata_assistant": "Metadata Assistant",
    "transcriptomics_expert": "Transcriptomics Expert",
    "annotation_expert": "Annotation Expert",
    "proteomics_expert": "Proteomics Expert",
}


def _vector_search_install_command() -> str:
    """Return an environment-aware install command for vector dependencies."""
    from lobster.core.component_registry import get_install_command

    return get_install_command("vector-search", is_extra=True)


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
        "  [yellow]⚠ Smart Standardization backend requires lobster-metadata (development package).[/yellow]"
    )
    console.print(
        f"  [dim]Install vector dependencies: {_vector_search_install_command()}[/dim]"
    )
    console.print(
        "  [dim]Backend modules are not currently shipped in public lobster-ai releases.[/dim]"
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

    console.print("\n  [dim]Installing Smart Standardization dependencies...[/dim]")
    from lobster.cli_internal.commands.light.agent_commands import _uv_pip_install

    s1, _ = _uv_pip_install("chromadb>=1.0.0")
    s2, _ = _uv_pip_install("openai>=1.0.0")
    if s1 and s2:
        console.print("  [green]✓[/green] Dependencies installed")
    else:
        console.print(
            f"  [yellow]⚠ Install manually: {_vector_search_install_command()}[/yellow]"
        )
        return

    console.print(
        "\n  [dim]Downloading ontology databases (MONDO, UBERON, Cell Ontology)...[/dim]"
    )
    if _download_ontology_databases():
        console.print("  [green]✓[/green] Ontology databases ready")
    else:
        console.print(
            "  [yellow]⚠ Some databases failed to download. They will be auto-downloaded on first use.[/yellow]"
        )


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
        workspace_config = WorkspaceProviderConfig()

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
        # Non-fatal error - .env is still valid, workspace config is optimization
        logger.warning(f"Failed to create workspace config: {e}")
        console.print(
            f"[dim yellow]⚠️  Note: Workspace config creation skipped ({e})[/dim yellow]"
        )


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
}


def _uv_tool_env_handoff(
    provider_name: str | None,
    selected_agents: list[str] | None,
    skip_extras: bool,
    include_vector_search: bool = False,
) -> None:
    """In a uv tool env, build and optionally run the install command.

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
    import subprocess as _sp

    from rich.prompt import Confirm

    from lobster.core.uv_tool_env import detect_uv_tool_env

    info = detect_uv_tool_env()
    if not info:
        return  # Not in a tool env — nothing to do

    cmd = _build_uv_tool_init_command(
        None if skip_extras else provider_name,
        selected_agents,
        include_vector_search=include_vector_search,
    )
    cmd_str = " ".join(cmd)

    # Skip if the computed command wouldn't add anything new.
    existing_cmd = " ".join(_build_uv_tool_init_command(None, None))
    if cmd_str == existing_cmd:
        return

    console.print()
    console.print(
        "[bold white]Package installation needed[/bold white]\n"
        "  Your Lobster is installed via [cyan]uv tool[/cyan], so packages must\n"
        "  be added through [cyan]uv tool install[/cyan] rather than pip.\n"
    )
    console.print(f"  [bold]{cmd_str}[/bold]\n")

    if Confirm.ask("  Run this now?", default=True):
        console.print("[dim]  Running uv tool install...[/dim]")
        # ── CRITICAL: venv-safe subprocess execution ──────────────────
        # After `uv tool install` completes, the venv on disk is replaced
        # while this process still has OLD modules loaded in memory.
        # Rich lazy-loads _unicode_data modules on demand — any
        # console.print() after the swap crashes with ModuleNotFoundError.
        #
        # Rules after subprocess.run():
        #   1. NO Rich (console.print, Panel, Prompt, etc.)
        #   2. Plain print() only, respecting NO_COLOR
        #   3. os._exit() to skip Python teardown (atexit, __del__, etc.)
        #      Safe here: no DB/network connections, config already saved.
        # ──────────────────────────────────────────────────────────────
        import os as _os

        if console.file:
            console.file.flush()

        # Stream output so the user sees resolver/download progress
        # instead of staring at a frozen terminal.
        result = _sp.run(cmd)

        # === DANGER ZONE: venv on disk has been replaced ===
        _use_color = _os.environ.get("NO_COLOR") is None and sys.stdout.isatty()

        if result.returncode == 0:
            if _use_color:
                print("  \033[32m✓ Packages installed successfully.\033[0m")
            else:
                print("  ✓ Packages installed successfully.")
            print("  Restart lobster to use the new packages.")
            sys.stdout.flush()
            _os._exit(0)
        else:
            if _use_color:
                print("  \033[31m✗ Installation failed.\033[0m")
            else:
                print("  ✗ Installation failed.")
            print(f"  Run manually: {cmd_str}")
            sys.stdout.flush()
            _os._exit(result.returncode)
    else:
        console.print(f"  [dim]Run manually when ready:[/dim]\n  {cmd_str}\n")


def _ensure_provider_installed(provider_name: str) -> bool:
    """Check if provider package is installed; install if missing.

    Args:
        provider_name: Provider key (anthropic, bedrock, ollama, gemini, azure)

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

    console.print(f"\n  [dim]Installing {pkg}...[/dim]")
    from lobster.cli_internal.commands.light.agent_commands import _uv_pip_install

    success, msg = _uv_pip_install(pkg)
    if success:
        console.print(f"  [green]✓[/green] {pkg} installed")
        return True
    else:
        from lobster.core.component_registry import get_install_command

        console.print(f"  [yellow]⚠ Could not auto-install {pkg}[/yellow]")
        console.print(f"  [dim]Run manually: {get_install_command(pkg)}[/dim]")
        return False


def _prompt_docling_install() -> None:
    """Ask user about docling and install if desired."""
    try:
        import docling  # noqa: F401

        return  # Already installed, skip prompt
    except ImportError:
        pass

    console.print("\n[bold white]📄 Document Intelligence (Optional)[/bold white]")
    console.print(
        "  [dim]Structured extraction from scientific PDFs and web pages.[/dim]"
    )
    console.print("  [dim]Requires ~700 MB additional download.[/dim]\n")
    console.print("    [cyan]1[/cyan] - Yes, install Docling")
    console.print("    [cyan]2[/cyan] - No, skip for now")
    console.print()

    choice = Prompt.ask(
        "  Will you analyze PDF publications?",
        choices=["1", "2"],
        default="2",
    )

    if choice == "1":
        console.print("  [dim]Installing docling (this may take a moment)...[/dim]")
        from rich.markup import escape as rich_escape

        from lobster.cli_internal.commands.light.agent_commands import _uv_pip_install
        from lobster.core.component_registry import get_install_command

        s1, _ = _uv_pip_install("docling>=2.60.0")
        s2, _ = _uv_pip_install("docling-core>=2.50.0")
        if s1 and s2:
            console.print("  [green]✓[/green] Docling installed")
        else:
            docling_cmd = rich_escape(get_install_command("docling", is_extra=True))
            console.print(f"  [yellow]⚠ Install manually: {docling_cmd}[/yellow]")
    else:
        from rich.markup import escape as rich_escape

        from lobster.core.component_registry import get_install_command

        docling_cmd = rich_escape(get_install_command("docling", is_extra=True))
        console.print(f"  [dim]Skipped. Install later: {docling_cmd}[/dim]")


def _postprocess_tui_init_result(
    result: Dict[str, Any],
    *,
    workspace_path: Path,
    env_path: Path,
    global_config: bool,
    skip_extras: bool,
) -> Dict[str, Any]:
    """Normalize, persist, and complete a Go/questionary init result."""
    from lobster.core.uv_tool_env import is_uv_tool_env
    from lobster.ui.bridge.init_adapter import apply_tui_init_result

    normalized_result = dict(result)

    # Handle Omics-OS Cloud authentication.
    provider_name = (normalized_result.get("provider") or "").strip().lower()
    api_key = (normalized_result.get("api_key") or "").strip()
    oauth_done = normalized_result.get("oauth_authenticated", False)

    if provider_name == "omics-os" and not api_key:
        if oauth_done:
            # Go TUI saved raw tokens to credentials.json. Validate against
            # the gateway to enrich with email/tier/user_id (matching the
            # Python browser login flow in cloud_commands.py).
            _validate_oauth_credentials()
        else:
            from lobster.cli_internal.commands.light.cloud_commands import (
                attempt_login_for_init,
            )

            success = attempt_login_for_init()
            if not success:
                console.print(
                    "[yellow]Browser login did not complete. "
                    "You can paste an API key instead.[/yellow]"
                )
                api_key = Prompt.ask(
                    "[bold white]Enter your Omics-OS API key (omk_...)[/bold white]",
                    default="",
                )
                if api_key.strip():
                    normalized_result["api_key"] = api_key.strip()
                else:
                    console.print(
                        "[yellow]No credentials provided. Run 'lobster cloud login' later.[/yellow]"
                    )

    normalized_agents = _normalize_selected_agents(
        normalized_result.get("agents") or []
    )
    if normalized_agents:
        normalized_agents = _check_and_prompt_install_packages(
            normalized_agents, workspace_path
        )
    normalized_result["agents"] = normalized_agents

    apply_tui_init_result(
        normalized_result,
        workspace_path=workspace_path,
        env_path=env_path,
        global_config=global_config,
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
            get_api_key,
            get_endpoint,
            load_credentials,
            save_credentials,
        )

        token = get_api_key()
        if not token:
            console.print(
                "[yellow]OAuth tokens found but could not be loaded.[/yellow]"
            )
            return

        endpoint = get_endpoint()

        from lobster.cli_internal.commands.light.cloud_commands import (
            _validate_credentials,
        )

        data = _validate_credentials(endpoint, token, token_type="token")
        if data is None:
            console.print(
                "[yellow]Token validation failed. "
                "Run 'lobster cloud login' to re-authenticate.[/yellow]"
            )
            return

        # Enrich the existing credentials with gateway data.
        creds = load_credentials() or {}
        creds["user_id"] = data.get("user_id", "")
        creds["email"] = data.get("email", "")
        creds["tier"] = data.get("tier", "free")
        save_credentials(creds)

        email = data.get("email", "")
        tier = data.get("tier", "free")
        console.print(f"[green]Authenticated as {email} (tier: {tier})[/green]")

    except Exception as exc:
        logger.debug(f"OAuth validation failed: {exc}")
        console.print(
            "[yellow]Could not validate OAuth tokens. "
            "Run 'lobster cloud login' if issues persist.[/yellow]"
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
            "  [yellow]⚠ Vector backend not available. Skipping ontology download.[/yellow]"
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
                console.print(f"  [yellow]⚠[/yellow] {collection_name} (skipped)")
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
        console.print("\n[bold white]Smart Standardization (Optional)[/bold white]")
        _print_vector_backend_unavailable()
        return []

    benefiting = _get_smart_standardization_beneficiaries(selected_agents)

    console.print("\n[bold white]Smart Standardization (Optional)[/bold white]")
    console.print(
        "  [dim]Map biomedical terms to ontology concepts (diseases, tissues, cell types).[/dim]"
    )
    console.print(
        "  [dim]Uses OpenAI text-embedding-3-small (~$0.002 per 1,000 queries).[/dim]"
    )
    if benefiting:
        agents_str = ", ".join(benefiting)
        console.print(f"  [dim]Your selected agents that benefit: {agents_str}[/dim]")
    console.print()
    console.print(
        "    [cyan]1[/cyan] - Yes, set up locally (OpenAI key + ChromaDB + ontology download)"
    )
    console.print("    [cyan]2[/cyan] - Skip for now")
    console.print()

    choice = Prompt.ask(
        "  Enable Smart Standardization?",
        choices=["1", "2"],
        default="2",
    )

    if choice != "1":
        console.print(
            f"  [dim]Skipped. Enable later: {_vector_search_install_command()} && lobster init --force[/dim]"
        )
        return []

    env_lines: list[str] = []

    # --- OpenAI API key ---
    existing_key = os.environ.get("OPENAI_API_KEY", "")
    if existing_key:
        console.print(
            f"  [green]✓[/green] OPENAI_API_KEY detected (sk-...{existing_key[-4:]})"
        )
        env_lines.append(f"OPENAI_API_KEY={existing_key}")
    else:
        openai_key = Prompt.ask(
            "  [bold white]Enter your OpenAI API key[/bold white]",
            password=True,
        )
        openai_key = openai_key.strip()
        if openai_key:
            env_lines.append(f"OPENAI_API_KEY={openai_key}")
            console.print("  [green]✓[/green] OpenAI API key configured")
        else:
            console.print(
                "  [yellow]⚠ No API key provided. Smart Standardization requires an OpenAI key.[/yellow]"
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
        console.print("\n  [dim]Installing data access packages...[/dim]")
        for pkg in missing:
            success, _ = _uv_pip_install(pkg)
            if success:
                console.print(f"  [green]✓[/green] {pkg}")
        # Don't warn on failure — these all have graceful fallbacks


def _offer_npm_cross_install() -> None:
    """Offer to install @omicsos/lobster npm CLI if not already present."""
    from lobster.cli_internal.npm_launcher import find_npm_binary

    if find_npm_binary():
        console.print("[green]✓[/green] Cloud TUI installed (npm)")
        return

    console.print()
    console.print(
        "[bold]Cloud TUI[/bold] enables interactive cloud sessions "
        "from your terminal."
    )
    console.print(
        "  Install: [bold cyan]npm install -g @omicsos/lobster[/bold cyan]"
    )

    try:
        install = Confirm.ask("  Install now?", default=True, console=console)
    except (KeyboardInterrupt, EOFError):
        return

    if not install:
        console.print("  [dim]Skipped. Run 'npm install -g @omicsos/lobster' later.[/dim]")
        return

    npm = shutil.which("npm")
    if not npm:
        console.print(
            "  [yellow]npm not found.[/yellow] Install Node.js 22+ first, then:\n"
            "  [bold]npm install -g @omicsos/lobster[/bold]"
        )
        return

    import subprocess
    console.print("  [dim]Installing @omicsos/lobster...[/dim]")
    result = subprocess.run(
        [npm, "install", "-g", "@omicsos/lobster"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        console.print("  [green]✓[/green] @omicsos/lobster installed")
    else:
        console.print(
            f"  [yellow]Install failed (exit {result.returncode}).[/yellow]\n"
            "  Run manually: [bold]npm install -g @omicsos/lobster[/bold]"
        )
        if result.stderr:
            console.print(f"  [dim]{result.stderr[:200]}[/dim]")


def _ensure_tui_installed() -> None:
    """Install textual for TUI support if not present."""
    try:
        import textual  # noqa: F401

        return
    except ImportError:
        pass

    from lobster.cli_internal.commands.light.agent_commands import _uv_pip_install

    console.print("  [dim]Installing TUI support...[/dim]")
    success, _ = _uv_pip_install("textual>=6.7.1")
    if success:
        console.print("  [green]✓[/green] textual installed")


def init_impl(
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
        help=f"Ollama model name (default: {provider_setup.DEFAULT_OLLAMA_MODEL}, non-interactive mode)",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Default model ID for the selected provider (e.g., claude-sonnet-4-20250514, gpt-4o). Saved to workspace config.",
    ),
    gemini_key: Optional[str] = typer.Option(
        None, "--gemini-key", help="Google API key (non-interactive mode)"
    ),
    openai_key: Optional[str] = typer.Option(
        None, "--openai-key", help="OpenAI API key (non-interactive mode)"
    ),
    openrouter_key: Optional[str] = typer.Option(
        None, "--openrouter-key", help="OpenRouter API key (non-interactive mode)"
    ),
    azure_endpoint: Optional[str] = typer.Option(
        None, "--azure-endpoint", help="Azure AI endpoint URL (non-interactive mode)"
    ),
    azure_credential: Optional[str] = typer.Option(
        None,
        "--azure-credential",
        help="Azure AI API credential (non-interactive mode)",
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
        --use-ollama --ollama-model=qwen3:14b       # CI/CD: Ollama with custom model
      lobster init --non-interactive \\
        --anthropic-key=sk-ant-xxx \\
        --cloud-key=cloud_xxx                        # CI/CD: With cloud access
      lobster init --non-interactive \\
        --openrouter-key=sk-or-xxx                   # CI/CD: OpenRouter (600+ models)
      lobster init --non-interactive \\
        --azure-endpoint=https://xxx.inference.ai.azure.com/ \\
        --azure-credential=xxx                       # CI/CD: Azure AI
    """
    import datetime

    from lobster.config.global_config import GlobalProviderConfig

    # Determine paths based on global vs workspace mode
    if global_config:
        config_path = GlobalProviderConfig.get_config_path()
        env_path = None  # Global mode doesn't create .env (use env vars for API keys)
    else:
        config_path = Path.cwd() / ".lobster_workspace" / "provider_config.json"
        env_path = Path.cwd() / ".env"

    # Check if configuration already exists
    config_exists = (
        config_path.exists() if global_config else (env_path and env_path.exists())
    )
    if config_exists and not force:
        console.print()
        if global_config:
            console.print(
                Panel.fit(
                    "[bold yellow]⚠️  Global Configuration Already Exists[/bold yellow]\n\n"
                    f"A global config already exists at:\n[cyan]{config_path}[/cyan]\n\n"
                    "To reconfigure, use:\n"
                    f"[bold {LobsterTheme.PRIMARY_ORANGE}]lobster init --global --force[/bold {LobsterTheme.PRIMARY_ORANGE}]\n\n"
                    "Or edit the file manually.",
                    border_style="yellow",
                    padding=(1, 2),
                )
            )
        else:
            console.print(
                Panel.fit(
                    "[bold yellow]⚠️  Configuration Already Exists[/bold yellow]\n\n"
                    f"A .env file already exists at:\n[cyan]{env_path}[/cyan]\n\n"
                    "To reconfigure, use:\n"
                    f"[bold {LobsterTheme.PRIMARY_ORANGE}]lobster init --force[/bold {LobsterTheme.PRIMARY_ORANGE}]\n\n"
                    "Or edit the file manually.",
                    border_style="yellow",
                    padding=(1, 2),
                )
            )
        console.print()
        console.print(
            f"[dim]Configuration file: {config_path if global_config else env_path}[/dim]"
        )
        raise typer.Exit(0)

    # If force flag and config exists, create backup and continue.
    # `--force` is already explicit consent; adding a second prompt breaks
    # the Go/questionary wizard path and creates inconsistent semantics.
    backup_path = None
    if config_exists and force:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if global_config:
            backup_path = config_path.parent / f"providers.json.backup.{timestamp}"
            backup_source = config_path
        else:
            backup_path = Path.cwd() / f".env.backup.{timestamp}"
            backup_source = env_path

        if not non_interactive:
            console.print("[yellow]⚠️  Existing config will be backed up to:[/yellow]")
            console.print(f"[yellow]   {backup_path}[/yellow]")
            console.print()

        # Create backup
        try:
            shutil.copy2(backup_source, backup_path)
        except Exception as e:
            console.print(f"[red]❌ Failed to create backup: {str(e)}[/red]")
            raise typer.Exit(1)

    # Non-interactive mode: validate and create .env from parameters
    if non_interactive:
        # Detect uv tool env for non-interactive path
        from lobster.core.uv_tool_env import is_uv_tool_env as _is_uv_tool_env_ni

        _in_uv_tool_ni = _is_uv_tool_env_ni()

        # === AGENT SELECTION (BEFORE PROVIDER SETUP) ===
        # Resolve workspace path for agent config
        workspace_path = Path.cwd() / ".lobster_workspace"
        workspace_path.mkdir(parents=True, exist_ok=True)

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
            installed_agents = _check_and_prompt_install_packages(
                selected_agents,
                workspace_path,
                prompt_for_install=False,
                auto_install_missing=(not skip_extras and not _in_uv_tool_ni),
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
        has_azure = azure_endpoint is not None and azure_credential is not None

        # Validate at least one provider
        valid, error_msg = provider_setup.validate_provider_choice(
            has_anthropic,
            has_bedrock,
            has_ollama,
            has_gemini,
            has_openai,
            has_openrouter=has_openrouter,
            has_azure=has_azure,
        )
        if not valid:
            console.print(f"[red]❌ Error: {error_msg}[/red]")
            console.print()
            console.print("You must provide one of:")
            console.print("  • Claude API: --anthropic-key=xxx")
            console.print(
                "  • AWS Bedrock: --bedrock-access-key=xxx --bedrock-secret-key=xxx"
            )
            console.print("  • Ollama (Local): --use-ollama")
            console.print("  • Google Gemini: --gemini-key=xxx")
            console.print("  • OpenAI: --openai-key=xxx")
            console.print("  • OpenRouter: --openrouter-key=xxx")
            console.print("  • Azure AI: --azure-endpoint=xxx --azure-credential=xxx")
            raise typer.Exit(1)

        # Warn if multiple providers
        priority_warning = provider_setup.get_provider_priority_warning(
            has_anthropic,
            has_bedrock,
            has_ollama,
            has_gemini,
            has_openai,
            has_openrouter=has_openrouter,
            has_azure=has_azure,
        )
        if priority_warning:
            console.print(f"[yellow]⚠️  Warning: {priority_warning}[/yellow]")

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
                console.print(f"[red]❌ Error: Invalid profile '{profile}'[/red]")
                console.print(f"Valid profiles: {', '.join(VALID_PROFILES)}")
                raise typer.Exit(1)

            # Resolve deprecated aliases
            if profile in DEPRECATED_PROFILE_ALIASES:
                new_name = DEPRECATED_PROFILE_ALIASES[profile]
                console.print(
                    f"[yellow]⚠️  Profile '{profile}' is deprecated, "
                    f"using '{new_name}' instead.[/yellow]"
                )
                profile = new_name

            # Warn if profile used with Ollama (it will be ignored)
            if has_ollama and not has_anthropic and not has_bedrock:
                console.print(
                    "[yellow]⚠️  Warning: --profile ignored for Ollama (uses local models)[/yellow]"
                )
            else:
                selected_profile = profile
        else:
            # No profile provided - use default for Anthropic/Bedrock only
            if has_anthropic or has_bedrock:
                selected_profile = "production"  # Default profile

        # Create provider configuration
        if has_anthropic:
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
                console.print("[green]✓ Ollama provider configured[/green]")
        elif has_gemini:
            config = provider_setup.create_gemini_config(gemini_key)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "gemini"
                console.print("[green]✓ Gemini provider configured[/green]")
        elif has_openai:
            config = provider_setup.create_openai_config(openai_key)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "openai"
                console.print("[green]✓ OpenAI provider configured[/green]")
        elif has_openrouter:
            config = provider_setup.create_openrouter_config(openrouter_key)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "openrouter"
                console.print("[green]✓ OpenRouter provider configured[/green]")
        elif has_azure:
            config = provider_setup.create_azure_config(
                azure_endpoint, azure_credential
            )
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "azure"
                console.print("[green]✓ Azure AI provider configured[/green]")

        # Write profile configuration (only for Anthropic/Bedrock)
        if selected_profile:
            env_lines.append("\n# Agent Configuration Profile")
            env_lines.append("# Determines which Claude models are used for analysis")
            env_lines.append(f"LOBSTER_PROFILE={selected_profile}")
            config_dict["profile"] = selected_profile
            console.print(f"[green]✓ Profile set to: {selected_profile}[/green]")

        # Save model override (--model flag or --ollama-model for Ollama)
        # Guard: model may be a typer.OptionInfo if init_impl is called directly
        _model_str = model if isinstance(model, str) else None
        _ollama_model_str = ollama_model if isinstance(ollama_model, str) else None
        effective_model = _model_str or (_ollama_model_str if has_ollama else None)
        if effective_model:
            config_dict["model_id"] = effective_model
            console.print(f"[green]✓ Default model: {effective_model}[/green]")

        if ncbi_key:
            env_lines.append("\n# Optional: Enhanced literature search")
            env_lines.append(f"NCBI_API_KEY={ncbi_key.strip()}")

        if cloud_key:
            env_lines.append("\n# Lobster Cloud configuration (enables premium tier)")
            env_lines.append(f"LOBSTER_CLOUD_KEY={cloud_key.strip()}")
            if cloud_endpoint:
                env_lines.append(f"LOBSTER_ENDPOINT={cloud_endpoint.strip()}")
            console.print(
                "[green]✓ Cloud API key configured (premium tier enabled)[/green]"
            )

        # Handle SSL configuration from flags
        if ssl_verify is not None or ssl_cert_path:
            env_lines.append("\n# SSL/HTTPS configuration")
            if ssl_verify is False:
                env_lines.append("LOBSTER_SSL_VERIFY=false")
                console.print(
                    "[yellow]⚠️  SSL verification disabled (--no-ssl-verify)[/yellow]"
                )
            if ssl_cert_path:
                env_lines.append(f"LOBSTER_SSL_CERT_PATH={ssl_cert_path}")
                console.print(f"[green]✓ SSL cert path: {ssl_cert_path}[/green]")

        # Test SSL connectivity (non-interactive)
        if not skip_ssl_test and ssl_verify is not False:
            from lobster.config.ssl_setup import test_ssl_connectivity

            console.print("\n[dim]Testing connectivity to NCBI databases...[/dim]")
            success, is_ssl_error, error_msg = test_ssl_connectivity(timeout=10)

            if is_ssl_error:
                console.print("\n[yellow]⚠️  SSL Certificate Issue Detected[/yellow]")
                console.print(f"[dim]Error: {error_msg[:80]}...[/dim]\n")
                console.print(
                    "[yellow]To fix this in non-interactive mode, use one of:[/yellow]"
                )
                console.print(
                    "  --no-ssl-verify           # Disable SSL (testing only)"
                )
                console.print("  --ssl-cert-path=/path/to/ca.pem  # Use corporate CA")
                console.print()
                console.print(
                    "[dim]Continuing with setup - you may encounter errors later.[/dim]"
                )
            elif success:
                console.print("[green]✓ NCBI connectivity: OK[/green]")
            else:
                # Non-SSL network error
                console.print(
                    f"[yellow]⚠️  Network test failed: {error_msg[:60]}[/yellow]"
                )
                console.print("[dim]You can still continue (may work later)[/dim]")

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

            # Install docling if requested
            if install_docling:
                console.print("[dim]Installing docling...[/dim]")
                from lobster.cli_internal.commands.light.agent_commands import (
                    _uv_pip_install,
                )

                s1, _ = _uv_pip_install("docling>=2.60.0")
                s2, _ = _uv_pip_install("docling-core>=2.50.0")
                if s1 and s2:
                    console.print("[green]✓ Docling installed[/green]")
                else:
                    from rich.markup import escape as rich_escape

                    from lobster.core.component_registry import get_install_command

                    docling_cmd = rich_escape(
                        get_install_command("docling", is_extra=True)
                    )
                    console.print(
                        f"[yellow]⚠ Docling install failed. Run: {docling_cmd}[/yellow]"
                    )

            # Install vector search if requested (Smart Standardization — OpenAI)
            if install_vector_search:
                if not _is_vector_search_backend_available():
                    _print_vector_backend_unavailable()
                else:
                    console.print("[dim]Installing Smart Standardization deps...[/dim]")
                    from lobster.cli_internal.commands.light.agent_commands import (
                        _uv_pip_install,
                    )

                    s1, _ = _uv_pip_install("chromadb>=1.0.0")
                    s2, _ = _uv_pip_install("openai>=1.0.0")
                    if s1 and s2:
                        console.print(
                            "[green]✓ Smart Standardization installed[/green]"
                        )
                        env_lines.append("LOBSTER_EMBEDDING_PROVIDER=openai")
                    else:
                        console.print(
                            f"[yellow]⚠ Install manually: {_vector_search_install_command()}[/yellow]"
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
                global_config_path = _create_global_config(config_dict)

                # Save credentials to credentials.env (with secure permissions)
                credentials_path = None
                if credentials:
                    credentials_path = save_global_credentials(credentials)

                console.print("[green]✅ Global configuration saved:[/green]")
                console.print(f"  • Config: {global_config_path}")
                if credentials_path:
                    console.print(
                        f"  • Credentials: {credentials_path} [dim](mode: 0o600)[/dim]"
                    )
                if backup_path:
                    console.print(f"  • Backup: {backup_path}")
            except Exception as e:
                console.print(f"[red]❌ Failed to write global config: {str(e)}[/red]")
                raise typer.Exit(1)
        else:
            # Workspace mode: write both .env and workspace config
            try:
                with open(env_path, "w") as f:
                    f.write("\n".join(env_lines))
                    f.write("\n")
            except Exception as e:
                console.print(f"[red]❌ Failed to write .env file: {str(e)}[/red]")
                raise typer.Exit(1)

            # Create workspace config from structured config_dict
            workspace_path = Path.cwd() / ".lobster_workspace"
            _create_workspace_config(config_dict, workspace_path)

            # Success message for non-interactive
            console.print("[green]✅ Configuration saved:[/green]")
            console.print(f"  • Environment: {env_path}")
            console.print(f"  • Workspace:   {workspace_path / 'provider_config.json'}")
            if backup_path:
                console.print(f"  • Backup:      {backup_path}")

        # In uv tool env (non-interactive), print the command for the caller
        if _in_uv_tool_ni and not skip_extras:
            provider_name_ni = None
            if has_anthropic:
                provider_name_ni = "anthropic"
            elif has_bedrock:
                provider_name_ni = "bedrock"
            elif has_gemini:
                provider_name_ni = "gemini"
            elif has_openai:
                provider_name_ni = "openai"
            cmd = _build_uv_tool_init_command(
                provider_name_ni,
                locals().get("installed_agents"),
                include_vector_search=install_vector_search,
            )
            console.print(f"\n[bold]uv tool command:[/bold] {' '.join(cmd)}")

        raise typer.Exit(0)

    # Interactive mode: run wizard
    # =========================================================================
    # Guard: if stdin is not a TTY (e.g. piped from curl | bash), no
    # interactive UI path can work — Go TUI, questionary, and Rich prompts
    # all need a real terminal.  Check BEFORE dispatching to any UI.
    import sys as _sys

    if not non_interactive and not _sys.stdin.isatty():
        console.print(
            "[yellow]Cannot run interactive setup — stdin is not a terminal.[/yellow]\n"
            "Run [bold]lobster init --global[/bold] manually in your terminal to configure."
        )
        raise typer.Exit(1)

    # === TRY GO TUI / QUESTIONARY FIRST (ui_mode: auto | go | classic) ===
    # The Go binary and the questionary fallback both return the same JSON dict,
    # so a single apply_tui_init_result() call handles both paths.
    # The classic Rich-prompt flow further below is used when neither is
    # available (or when --ui classic is passed explicitly).
    # =========================================================================
    if ui_mode != "classic" and not non_interactive:
        _tui_handled = False

        # ---- Try Ink TUI (new default) ----
        if ui_mode in ("auto", "ink"):
            try:
                from lobster.cli_internal.ink_launcher import run_ink_init_wizard

                _ink_result = run_ink_init_wizard()
                if _ink_result.get("cancelled", False):
                    console.print("[yellow]Setup cancelled.[/yellow]")
                    raise typer.Exit(0)
                _ws_path = Path.cwd() / ".lobster_workspace"
                _ev_path = Path.cwd() / ".env"
                _ink_result = _postprocess_tui_init_result(
                    _ink_result,
                    workspace_path=_ws_path,
                    env_path=_ev_path,
                    global_config=global_config,
                    skip_extras=skip_extras,
                )
                console.print(
                    "[bold green]Configuration saved![/bold green] "
                    "Run [bold]lobster chat[/bold] to start analyzing."
                )
                from lobster.core.uv_tool_env import is_uv_tool_env as _is_uv

                if _is_uv():
                    _uv_tool_env_handoff(
                        provider_name=(_ink_result.get("provider") or "")
                        .strip()
                        .lower()
                        or None,
                        selected_agents=_ink_result.get("agents") or [],
                        skip_extras=skip_extras,
                        include_vector_search=bool(
                            _ink_result.get("smart_standardization_enabled")
                        )
                        and not skip_extras,
                    )
                _tui_handled = True
            except typer.Exit:
                raise
            except ImportError:
                if ui_mode == "ink":
                    console.print("[red]Ink TUI binary not found. Use --ui auto.[/red]")
                    raise typer.Exit(1)
            except Exception as _ink_exc:
                if ui_mode == "ink":
                    console.print(f"[red]Ink TUI init failed: {_ink_exc}[/red]")
                    raise typer.Exit(1)
                console.print(f"[dim]Ink TUI unavailable ({_ink_exc}), falling back.[/dim]")

        # ---- Try Go TUI ----
        if not _tui_handled and ui_mode in ("auto", "go"):
            try:
                from lobster.ui.bridge.binary_finder import find_tui_binary

                _binary = find_tui_binary()
                if _binary:
                    try:
                        from lobster.ui.bridge.go_tui_bridge import run_init_wizard

                        _tui_result = run_init_wizard(_binary)
                        if _tui_result.get("cancelled", False):
                            console.print("[yellow]Setup cancelled.[/yellow]")
                            raise typer.Exit(0)

                        _ws_path = Path.cwd() / ".lobster_workspace"
                        _ev_path = Path.cwd() / ".env"
                        _tui_result = _postprocess_tui_init_result(
                            _tui_result,
                            workspace_path=_ws_path,
                            env_path=_ev_path,
                            global_config=global_config,
                            skip_extras=skip_extras,
                        )

                        console.print(
                            "[bold green]Configuration saved![/bold green] "
                            "Run [bold]lobster chat[/bold] to start analyzing."
                        )
                        from lobster.core.uv_tool_env import is_uv_tool_env as _is_uv

                        if _is_uv():
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
                            )
                        _tui_handled = True
                    except typer.Exit:
                        raise
                    except Exception as _go_exc:
                        if ui_mode == "go":
                            console.print(f"[red]Go TUI init failed: {_go_exc}[/red]")
                            raise typer.Exit(1)
                        console.print(
                            f"[dim]Go TUI unavailable ({_go_exc}), falling back.[/dim]"
                        )
                elif ui_mode == "go":
                    # --ui go was explicitly requested but binary not found
                    console.print(
                        "[red]lobster-tui binary not found. "
                        "Install it or use --ui classic.[/red]"
                    )
                    raise typer.Exit(1)
            except ImportError as _ie:
                console.print(
                    f"[dim]Go TUI bridge unavailable ({_ie}), falling back.[/dim]"
                )

        # ---- Try questionary fallback (auto mode only, when Go TUI was unavailable) ----
        if not _tui_handled and ui_mode == "auto":
            try:
                from lobster.ui.bridge.questionary_fallback import run_questionary_init

                _q_result = run_questionary_init()
                if _q_result.get("cancelled", False):
                    console.print("[yellow]Setup cancelled.[/yellow]")
                    raise typer.Exit(0)

                _ws_path = Path.cwd() / ".lobster_workspace"
                _ev_path = Path.cwd() / ".env"
                _q_result = _postprocess_tui_init_result(
                    _q_result,
                    workspace_path=_ws_path,
                    env_path=_ev_path,
                    global_config=global_config,
                    skip_extras=skip_extras,
                )

                console.print(
                    "[bold green]Configuration saved![/bold green] "
                    "Run [bold]lobster chat[/bold] to start analyzing."
                )
                from lobster.core.uv_tool_env import is_uv_tool_env as _is_uv

                if _is_uv():
                    _uv_tool_env_handoff(
                        provider_name=(_q_result.get("provider") or "").strip().lower()
                        or None,
                        selected_agents=_q_result.get("agents") or [],
                        skip_extras=skip_extras,
                        include_vector_search=bool(
                            _q_result.get("smart_standardization_enabled")
                        )
                        and not skip_extras,
                    )
                _tui_handled = True
            except typer.Exit:
                raise
            except ImportError:
                # questionary not installed — fall through to classic Rich prompts silently
                pass
            except Exception as _q_exc:
                console.print(
                    f"[dim]Questionary wizard failed ({_q_exc}), falling back to classic mode.[/dim]"
                )

        if _tui_handled:
            if not skip_extras:
                _offer_npm_cross_install()
            raise typer.Exit(0)
    # === END GO TUI / QUESTIONARY PATH ===

    console.print("\n")
    if global_config:
        from lobster.config.global_config import get_global_credentials_path

        creds_path = get_global_credentials_path()
        console.print(
            Panel.fit(
                "[bold white]🦞 Welcome to Lobster AI![/bold white]\n\n"
                "Let's set up your [cyan]global[/cyan] provider defaults.\n"
                f"Config: [cyan]{config_path}[/cyan]\n"
                f"Credentials: [cyan]{creds_path}[/cyan]\n\n"
                "[dim]These defaults apply to all workspaces without their own config.[/dim]",
                border_style="bright_blue",
                padding=(1, 2),
            )
        )
    else:
        console.print(
            Panel.fit(
                "[bold white]🦞 Welcome to Lobster AI![/bold white]\n\n"
                "Let's set up your API keys.\n"
                "This wizard will create a [cyan].env[/cyan] file in your current directory.",
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
            "[dim]  Detected uv tool installation. API keys will be configured here;"
            "\n  any new packages will be installed after setup completes.[/dim]\n"
        )

    workspace_path = Path.cwd() / ".lobster_workspace"
    workspace_path.mkdir(parents=True, exist_ok=True)

    try:
        env_lines = []
        env_lines.append("# Lobster AI Configuration")
        env_lines.append("# Generated by lobster init\n")

        # Initialize structured config dict for workspace config
        config_dict = {}

        # ================================================================== #
        # Step 1 — Provider selection (unified list)                          #
        # ================================================================== #
        console.print("[bold white]Select your LLM provider:[/bold white]")
        console.print(
            "  [cyan]1[/cyan] - Omics-OS Cloud          — managed, login via browser"
        )
        console.print(
            "  [cyan]2[/cyan] - Claude API (Anthropic)  — direct access to Claude models"
        )
        console.print(
            "  [cyan]3[/cyan] - AWS Bedrock             — production, enterprise"
        )
        console.print(
            "  [cyan]4[/cyan] - Ollama (local)          — privacy, zero cost, offline"
        )
        console.print(
            "  [cyan]5[/cyan] - Google Gemini           — Gemini models + thinking"
        )
        console.print(
            "  [cyan]6[/cyan] - Azure AI                — enterprise Azure deployments"
        )
        console.print(
            "  [cyan]7[/cyan] - OpenAI                  — GPT-4o, o-series reasoning"
        )
        console.print(
            "  [cyan]8[/cyan] - OpenRouter              — 600+ models via one API key"
        )
        console.print()

        provider = Prompt.ask(
            "[bold white]Choose provider[/bold white]",
            choices=["1", "2", "3", "4", "5", "6", "7", "8"],
            default="1",
        )

        provider_map = {
            "1": "omics-os",
            "2": "anthropic",
            "3": "bedrock",
            "4": "ollama",
            "5": "gemini",
            "6": "azure",
            "7": "openai",
            "8": "openrouter",
        }
        provider_name = provider_map[provider]

        # ================================================================== #
        # Step 2 — Credentials (provider-specific)                            #
        # ================================================================== #
        if provider_name == "omics-os":
            from lobster.cli_internal.commands.light.cloud_commands import (
                attempt_login_for_init,
            )

            success = attempt_login_for_init()
            if success:
                config_dict["provider"] = "omics-os"
                env_lines.append("LOBSTER_LLM_PROVIDER=omics-os")
                console.print("[green]✓ Omics-OS Cloud provider configured[/green]")
            else:
                console.print(
                    "\n[yellow]Browser login did not complete. "
                    "You can paste an API key instead.[/yellow]"
                )
                api_key = Prompt.ask(
                    "[bold white]Enter your Omics-OS API key (omk_...)[/bold white]",
                    default="",
                )
                if api_key.strip():
                    config = provider_setup.create_omics_os_config(api_key)
                    if config.success:
                        for key, value in config.env_vars.items():
                            env_lines.append(f"{key}={value}")
                        config_dict["provider"] = "omics-os"
                        console.print(
                            "[green]✓ Omics-OS Cloud provider configured[/green]"
                        )
                    else:
                        console.print(f"[red]❌ {config.message}[/red]")
                        raise typer.Exit(1)
                else:
                    console.print(
                        "[yellow]No credentials provided. Run 'lobster cloud login' later.[/yellow]"
                    )
                    config_dict["provider"] = "omics-os"
                    env_lines.append("LOBSTER_LLM_PROVIDER=omics-os")

        elif provider_name == "anthropic":
            console.print("\n[bold white]🔑 Claude API Configuration[/bold white]")
            console.print(
                "Get your API key from: [link]https://console.anthropic.com/[/link]\n"
            )
            api_key = Prompt.ask(
                "[bold white]Enter your Claude API key[/bold white]", password=True
            )
            if not api_key.strip():
                console.print("[red]❌ API key cannot be empty[/red]")
                raise typer.Exit(1)
            env_lines.append(f"ANTHROPIC_API_KEY={api_key.strip()}")
            config_dict["provider"] = "anthropic"

        elif provider_name == "bedrock":
            console.print("\n[bold white]🔑 AWS Bedrock Configuration[/bold white]")
            console.print(
                "You'll need AWS access key and secret key with Bedrock permissions.\n"
            )
            access_key = Prompt.ask(
                "[bold white]Enter your AWS access key[/bold white]", password=True
            )
            secret_key = Prompt.ask(
                "[bold white]Enter your AWS secret key[/bold white]", password=True
            )
            if not access_key.strip() or not secret_key.strip():
                console.print("[red]❌ AWS credentials cannot be empty[/red]")
                raise typer.Exit(1)
            env_lines.append(f"AWS_BEDROCK_ACCESS_KEY={access_key.strip()}")
            env_lines.append(f"AWS_BEDROCK_SECRET_ACCESS_KEY={secret_key.strip()}")
            config_dict["provider"] = "bedrock"

        elif provider_name == "ollama":
            console.print(
                "\n[bold white]🏠 Ollama (Local LLM) Configuration[/bold white]"
            )
            console.print("Ollama runs models locally - no API keys needed!\n")
            ollama_status = provider_setup.get_ollama_status()

            if not ollama_status.installed:
                console.print(
                    "[yellow]⚠️  Ollama is not installed on this system.[/yellow]"
                )
                install_instructions = provider_setup.get_ollama_install_instructions()
                console.print(f"  • macOS/Linux: {install_instructions['macos_linux']}")
                console.print(f"  • Windows: {install_instructions['windows']}")
                console.print()
                install_later = Confirm.ask(
                    "Configure for Ollama anyway? (you can install it later)",
                    default=True,
                )
                if not install_later:
                    console.print(
                        "[yellow]Please install Ollama first, then run 'lobster init' again[/yellow]"
                    )
                    raise typer.Exit(0)

            model_name = None
            if ollama_status.running and ollama_status.models:
                console.print("[green]✓ Ollama is installed and running[/green]")
                console.print("\n[bold white]Available models:[/bold white]")
                for m in ollama_status.models[:5]:
                    console.print(f"  • {m}")
                if len(ollama_status.models) > 5:
                    console.print(f"  ... and {len(ollama_status.models) - 5} more")
                console.print()
                use_custom = Confirm.ask(
                    f"Specify a model? (default: {provider_setup.DEFAULT_OLLAMA_MODEL})",
                    default=False,
                )
                if use_custom:
                    model_name = Prompt.ask(
                        "[bold white]Enter model name[/bold white]",
                        default=provider_setup.DEFAULT_OLLAMA_MODEL,
                    )
            else:
                if ollama_status.installed:
                    console.print("[green]✓ Ollama is installed[/green]")
                    if not ollama_status.running:
                        console.print(
                            "[yellow]⚠️  Ollama server is not running. Start with: ollama serve[/yellow]"
                        )

            config = provider_setup.create_ollama_config(model_name=model_name)
            for key, value in config.env_vars.items():
                env_lines.append(f"{key}={value}")
            config_dict["provider"] = "ollama"
            if model_name:
                config_dict["ollama_model"] = model_name
            console.print("[green]✓ Ollama provider configured[/green]")

        elif provider_name == "gemini":
            console.print("\n[bold white]🔑 Google Gemini Configuration[/bold white]")
            console.print(
                "Get your API key from: [link]https://aistudio.google.com/apikey[/link]\n"
            )
            api_key = Prompt.ask(
                "[bold white]Enter your Google API key[/bold white]", password=True
            )
            if not api_key.strip():
                console.print("[red]❌ API key cannot be empty[/red]")
                raise typer.Exit(1)
            config = provider_setup.create_gemini_config(api_key)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "gemini"
                console.print("[green]✓ Gemini provider configured[/green]")

        elif provider_name == "azure":
            console.print("\n[bold white]🔑 Azure AI Configuration[/bold white]")
            console.print("You'll need Azure AI Foundry endpoint and API credential.\n")
            endpoint = Prompt.ask(
                "[bold white]Enter your Azure AI endpoint[/bold white]",
                default="https://your-project.inference.ai.azure.com/",
            )
            credential = Prompt.ask(
                "[bold white]Enter your Azure API credential[/bold white]",
                password=True,
            )
            if not endpoint.strip() or not credential.strip():
                console.print(
                    "[red]❌ Azure endpoint and credential cannot be empty[/red]"
                )
                raise typer.Exit(1)
            config = provider_setup.create_azure_config(endpoint, credential)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "azure"
                console.print("[green]✓ Azure AI provider configured[/green]")
            else:
                console.print(f"[red]❌ Configuration failed: {config.message}[/red]")
                raise typer.Exit(1)

        elif provider_name == "openai":
            console.print("\n[bold white]🔑 OpenAI Configuration[/bold white]")
            console.print(
                "Get your API key from: [link]https://platform.openai.com/api-keys[/link]\n"
            )
            api_key = Prompt.ask(
                "[bold white]Enter your OpenAI API key[/bold white]", password=True
            )
            if not api_key.strip():
                console.print("[red]❌ API key cannot be empty[/red]")
                raise typer.Exit(1)
            config = provider_setup.create_openai_config(api_key)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "openai"
                console.print("[green]✓ OpenAI provider configured[/green]")

        elif provider_name == "openrouter":
            console.print("\n[bold white]🔀 OpenRouter Configuration[/bold white]")
            console.print(
                "Get your API key from: [link]https://openrouter.ai/keys[/link]\n"
            )
            api_key = Prompt.ask(
                "[bold white]Enter your OpenRouter API key[/bold white]", password=True
            )
            if not api_key.strip():
                console.print("[red]❌ API key cannot be empty[/red]")
                raise typer.Exit(1)
            config = provider_setup.create_openrouter_config(api_key)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "openrouter"
                console.print("[green]✓ OpenRouter provider configured[/green]")
            else:
                console.print(f"[red]❌ Configuration failed: {config.message}[/red]")
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
            console.print("\n[bold white]⚙️  Agent Configuration Profile[/bold white]")
            console.print("Choose which Claude models to use for analysis:")
            console.print()
            console.print(
                "  [cyan]1[/cyan] - Development  (Sonnet 4 - fastest, most affordable)"
            )
            console.print(
                "  [cyan]2[/cyan] - Production   (Sonnet 4 + Sonnet 4.5 supervisor) [recommended]"
            )
            console.print(
                "  [cyan]3[/cyan] - Performance  (Sonnet 4.5 - highest quality)"
            )
            console.print(
                "  [cyan]4[/cyan] - Max          (Opus 4.5 supervisor - most capable, most expensive)"
            )
            console.print()

            profile_choice = Prompt.ask(
                "[bold white]Choose profile[/bold white]",
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
            console.print(f"[green]✓ Profile set to: {profile_to_write}[/green]")

        # ================================================================== #
        # Step 4 — Agent selection                                            #
        # ================================================================== #
        selected_agents, preset_name = _perform_agent_selection_interactive(
            workspace_path
        )
        if selected_agents:
            _save_agent_config(selected_agents, workspace_path, preset_name)
        console.print()

        # ================================================================== #
        # Step 5 — Optional NCBI key (single)                                #
        # ================================================================== #
        add_ncbi = Confirm.ask(
            "Add an NCBI API key? (enhances literature search, optional)",
            default=False,
        )

        if add_ncbi:
            ncbi_key_val = Prompt.ask(
                "[bold white]Enter your NCBI API key[/bold white]", password=True
            )
            if ncbi_key_val.strip():
                env_lines.append("\n# NCBI API key for literature search")
                env_lines.append(f"NCBI_API_KEY={ncbi_key_val.strip()}")
                console.print("[green]✓ NCBI API key configured[/green]")

        # ================================================================== #
        # Step 6 — Optional Cloud key (skip if Omics-OS)                      #
        # ================================================================== #
        if provider_name != "omics-os":
            add_cloud = Confirm.ask(
                "Add an Omics-OS Cloud API key? (enables premium tier, optional)",
                default=False,
            )
            if add_cloud:
                cloud_key_val = Prompt.ask(
                    "[bold white]Enter your Omics-OS Cloud API key[/bold white]",
                    password=True,
                )
                if cloud_key_val.strip():
                    env_lines.append("\n# Lobster Cloud configuration")
                    env_lines.append(f"LOBSTER_CLOUD_KEY={cloud_key_val.strip()}")
                    console.print("[green]✓ Cloud API key configured[/green]")

        # ================================================================== #
        # Step 7 — Smart Standardization (if not uv tool env)                 #
        # ================================================================== #
        if not skip_extras and not _in_uv_tool:
            smart_std_lines = _prompt_smart_standardization(
                selected_agents=selected_agents,
            )
            env_lines.extend(smart_std_lines)

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

                success_message = (
                    "[bold green]✅ Global configuration saved![/bold green]\n\n"
                )
                success_message += f"Config: [cyan]{global_config_path}[/cyan]\n"
                if credentials_path:
                    success_message += f"Credentials: [cyan]{credentials_path}[/cyan] [dim](secure: mode 0o600)[/dim]\n"
                if backup_path:
                    success_message += f"Backup: [cyan]{backup_path}[/cyan]\n"
                success_message += f"\n[bold white]Next step:[/bold white] Run [bold {LobsterTheme.PRIMARY_ORANGE}]lobster chat[/bold {LobsterTheme.PRIMARY_ORANGE}] to start analyzing!"
            except Exception as e:
                console.print(f"[red]❌ Failed to write global config: {str(e)}[/red]")
                raise typer.Exit(1)
        else:
            with open(env_path, "w") as f:
                f.write("\n".join(env_lines))
                f.write("\n")

            _create_workspace_config(config_dict, workspace_path)

            success_message = "[bold green]✅ Configuration saved![/bold green]\n\n"
            success_message += f"Environment: [cyan]{env_path}[/cyan]\n"
            success_message += (
                f"Workspace:   [cyan]{workspace_path / 'provider_config.json'}[/cyan]\n"
            )
            if backup_path:
                success_message += f"Backup: [cyan]{backup_path}[/cyan]\n\n"
            else:
                success_message += "\n"
            success_message += f"[bold white]Next step:[/bold white] Run [bold {LobsterTheme.PRIMARY_ORANGE}]lobster chat[/bold {LobsterTheme.PRIMARY_ORANGE}] to start analyzing!"

        console.print(Panel.fit(success_message, border_style="green"))
        console.print()

        # In uv tool env, offer to run `uv tool install` for new packages
        if _in_uv_tool:
            _want_vector_search = False
            if not skip_extras and _is_vector_search_backend_available():
                try:
                    import chromadb  # noqa: F401
                except ImportError:
                    _vs_choice = Prompt.ask(
                        "\n  Include Smart Standardization (ontology matching, OpenAI embeddings)?",
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
            )

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Configuration cancelled[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]❌ Configuration failed: {str(e)}[/red]")
        raise typer.Exit(1)
