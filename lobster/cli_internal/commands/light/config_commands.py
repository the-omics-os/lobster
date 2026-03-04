"""
Shared configuration commands for CLI and Dashboard.

Extracted from cli.py to enable reuse across interfaces.
All commands accept OutputAdapter for UI-agnostic rendering.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from lobster.core.client import AgentClient

from lobster.cli_internal.commands.output_adapter import OutputAdapter


def _build_agent_hierarchy(output: OutputAdapter, current_tier: str) -> None:
    """
    Display ASCII hierarchy of agent relationships.

    Shows supervisor at top, then worker agents with their child agents indented.
    """
    from lobster.config.agent_registry import AGENT_REGISTRY, get_valid_handoffs
    from lobster.config.subscription_tiers import is_agent_available

    valid_handoffs = get_valid_handoffs()
    supervisor_targets = valid_handoffs.get("supervisor", set())

    # Filter to available agents first
    available_agents = [
        name
        for name in sorted(supervisor_targets)
        if is_agent_available(name, current_tier) and AGENT_REGISTRY.get(name)
    ]

    if not available_agents:
        return

    output.print("\n[bold cyan]🔀 Agent Hierarchy[/bold cyan]")
    output.print("[dim]" + "─" * 96 + "[/dim]")  # Match table width (100 - borders)
    output.print("[bold white]supervisor[/bold white] [dim](orchestrator)[/dim]")

    for i, agent_name in enumerate(available_agents):
        config = AGENT_REGISTRY[agent_name]
        is_last = i == len(available_agents) - 1
        branch = "└── " if is_last else "├── "

        output.print(f"  {branch}[yellow]{config.display_name}[/yellow]")

        # Show child agents if any
        if config.child_agents:
            available_children = [
                c
                for c in config.child_agents
                if is_agent_available(c, current_tier) and AGENT_REGISTRY.get(c)
            ]
            child_prefix = "      " if is_last else "  │   "
            for j, child_name in enumerate(available_children):
                child_config = AGENT_REGISTRY[child_name]
                child_is_last = j == len(available_children) - 1
                child_branch = "└── " if child_is_last else "├── "
                output.print(
                    f"{child_prefix}{child_branch}[dim]{child_config.display_name}[/dim]"
                )

    output.print("")


def _build_agent_composition(
    output: OutputAdapter, workspace_path: Path, table_width: int = 100
) -> tuple[str, int, int]:
    """
    Display agent composition from workspace config.toml.

    Shows:
    1. Configuration source (preset, explicit enabled list, or defaults)
    2. Agent table with Name, Status (Installed/Missing), Tier

    Args:
        output: OutputAdapter for rendering
        workspace_path: Path to workspace directory
        table_width: Width of the table

    Returns:
        Tuple of (config_source_description, installed_count, total_count)
    """
    from lobster.config.agent_presets import expand_preset, get_preset_description
    from lobster.config.workspace_agent_config import WorkspaceAgentConfig
    from lobster.core.component_registry import component_registry

    # Load workspace agent config
    config = WorkspaceAgentConfig.load(workspace_path)

    # Get all installed agents from ComponentRegistry (single source of truth)
    installed_agents = component_registry.list_agents()

    # Determine configuration source and get agent list to display
    if config.preset:
        # Using a preset
        preset_agents = expand_preset(config.preset)
        description = get_preset_description(config.preset)
        if preset_agents:
            config_source = f"preset: [bold cyan]{config.preset}[/bold cyan]"
            config_detail = f"[dim]{description}[/dim]" if description else ""
            display_agents = preset_agents
        else:
            # Invalid preset, fall back to defaults
            config_source = f"preset: [red]{config.preset}[/red] [dim](invalid, using defaults)[/dim]"
            config_detail = ""
            display_agents = list(installed_agents.keys())
    elif config.enabled_agents:
        # Using explicit enabled list
        config_source = "[bold cyan]explicit agent list[/bold cyan]"
        config_detail = (
            f"[dim]{len(config.enabled_agents)} agents configured in config.toml[/dim]"
        )
        display_agents = config.enabled_agents
    else:
        # Using defaults (all available agents)
        config_source = "[bold cyan]defaults[/bold cyan]"
        config_detail = (
            "[dim]All available agents enabled (no config.toml or empty)[/dim]"
        )
        display_agents = list(installed_agents.keys())

    # Display configuration source
    output.print("\n[bold cyan]📦 Agent Composition[/bold cyan]")
    output.print("[dim]" + "-" * 96 + "[/dim]")
    output.print(f"  Configuration: {config_source}")
    if config_detail:
        output.print(f"  {config_detail}")
    output.print("")

    # Build agent table
    agent_table_data = {
        "title": None,  # No title, we already have the section header
        "width": table_width,
        "columns": [
            {"name": "Agent", "style": "cyan"},
            {"name": "Status", "style": "white"},
            {"name": "Tier", "style": "yellow"},
        ],
        "rows": [],
    }

    installed_count = 0
    missing_agents = []

    for agent_name in sorted(display_agents):
        agent_config = installed_agents.get(agent_name)

        if agent_config:
            # Agent is installed
            installed_count += 1
            display_name = getattr(agent_config, "display_name", agent_name)
            tier = getattr(agent_config, "tier_requirement", "free")
            tier_display = tier.capitalize() if tier else "Free"

            agent_table_data["rows"].append(
                [
                    f"[cyan]{display_name}[/cyan]",
                    "[green]Installed[/green]",
                    f"[yellow]{tier_display}[/yellow]",
                ]
            )
        else:
            # Agent is missing
            missing_agents.append(agent_name)
            agent_table_data["rows"].append(
                [f"[dim]{agent_name}[/dim]", "[red]Missing[/red]", "[dim]-[/dim]"]
            )

    output.print_table(agent_table_data)

    # Summary
    total_count = len(display_agents)
    if installed_count == total_count:
        output.print(
            f"\n[green]✓ {installed_count} of {total_count} agents installed and ready[/green]"
        )
    else:
        output.print(
            f"\n[yellow]⚠ {installed_count} of {total_count} agents installed[/yellow]"
        )

        # Helpful hint for missing agents
        if missing_agents:
            output.print("\n[dim]To install missing agents:[/dim]")
            output.print("  [white]lobster agents install <package>[/white]")
            output.print(
                "[dim]Run 'lobster agents list' to see available packages[/dim]"
            )

    return config_source, installed_count, total_count


def config_show(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Show current configuration including provider, profile, config files, and agent models.

    Displays three tables:
    1. Current Configuration (provider, profile)
    2. Configuration Files (workspace, global)
    3. Agent Models (per-agent model configuration)

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    from lobster.config.agent_registry import AGENT_REGISTRY
    from lobster.config.global_config import CONFIG_DIR as GLOBAL_CONFIG_DIR
    from lobster.config.providers import get_provider
    from lobster.config.settings import get_settings
    from lobster.config.subscription_tiers import is_agent_available
    from lobster.config.workspace_config import WorkspaceProviderConfig
    from lobster.core.config_resolver import ConfigResolver
    from lobster.core.license_manager import get_current_tier

    # Create resolver
    resolver = ConfigResolver(workspace_path=Path(client.workspace_path))

    # Resolve provider and profile
    provider, p_source = resolver.resolve_provider(
        runtime_override=client.provider_override
    )
    profile, pf_source = resolver.resolve_profile()

    # Check if config files exist
    workspace_config_exists = WorkspaceProviderConfig.exists(
        Path(client.workspace_path)
    )
    global_config_path = GLOBAL_CONFIG_DIR / "providers.json"
    global_config_exists = global_config_path.exists()

    # ========================================================================
    # Table 1: Current Configuration
    # ========================================================================
    # Fixed table width for visual consistency
    TABLE_WIDTH = 100

    config_table_data = {
        "title": "⚙️  Current Configuration",
        "width": TABLE_WIDTH,
        "columns": [
            {"name": "Setting", "style": "cyan"},
            {"name": "Value", "style": "white"},
            {"name": "Source", "style": "yellow"},
        ],
        "rows": [
            ["Provider", f"[bold]{provider}[/bold]", p_source],
            ["Profile", f"[bold]{profile}[/bold]", pf_source],
        ],
    }

    output.print_table(config_table_data)

    # ========================================================================
    # Table 2: Config Files Status
    # ========================================================================
    workspace_status = (
        "[green]✓ Exists[/green]"
        if workspace_config_exists
        else "[grey50]✗ Not found[/grey50]"
    )
    workspace_path_str = str(Path(client.workspace_path) / "provider_config.json")

    global_status = (
        "[green]✓ Exists[/green]"
        if global_config_exists
        else "[grey50]✗ Not found[/grey50]"
    )
    global_path_str = str(global_config_path)

    status_table_data = {
        "title": "📁 Configuration Files",
        "width": TABLE_WIDTH,
        "columns": [
            {"name": "Location", "style": "cyan"},
            {"name": "Status", "style": "white"},
            {"name": "Path", "style": "dim", "overflow": "ellipsis"},
        ],
        "rows": [
            ["Workspace Config", workspace_status, workspace_path_str],
            ["Global Config", global_status, global_path_str],
        ],
    }

    output.print_table(status_table_data)

    # ========================================================================
    # Table 3: Per-Agent Model Configuration
    # ========================================================================
    settings = get_settings()
    current_tier = get_current_tier()
    provider_obj = get_provider(provider)

    agent_table_data = {
        "title": "🤖 Agent Models",
        "width": TABLE_WIDTH,
        "columns": [
            {"name": "Agent", "style": "cyan"},
            {"name": "Model", "style": "yellow"},
            {"name": "Source", "style": "dim"},
        ],
        "rows": [],
    }

    # Show models for available agents
    for agent_name, agent_cfg in AGENT_REGISTRY.items():
        # Filter by license tier
        if not is_agent_available(agent_name, current_tier):
            continue

        try:
            # Get model parameters
            model_params = settings.get_agent_llm_params(agent_name)

            # Resolve model for this agent
            model_id, model_source = resolver.resolve_model(
                agent_name=agent_name, runtime_override=None, provider=provider
            )

            # If no model resolved, get provider's default model
            if not model_id:
                if provider_obj:
                    model_id = provider_obj.get_default_model()
                    model_source = "provider default"
                else:
                    model_id = model_params.get("model_id", "unknown")
                    model_source = "profile config"

            # Add row
            agent_table_data["rows"].append(
                [agent_cfg.display_name, model_id, model_source]
            )
        except Exception:
            # Skip agents with config errors
            continue

    output.print_table(agent_table_data)

    # ========================================================================
    # Agent Hierarchy (ASCII tree)
    # ========================================================================
    _build_agent_hierarchy(output, current_tier)

    # ========================================================================
    # Agent Composition (from config.toml)
    # ========================================================================
    workspace_path = Path(client.workspace_path)
    _build_agent_composition(output, workspace_path, TABLE_WIDTH)

    # ========================================================================
    # Usage hints
    # ========================================================================
    output.print("\n[cyan]💡 Available Config Commands:[/cyan]")
    output.print("\n[yellow]Provider Management:[/yellow]")
    output.print("  • [white]/config provider[/white] - List available providers")
    output.print(
        "  • [white]/config provider <name>[/white] - Switch provider (runtime only)"
    )
    output.print(
        "  • [white]/config provider <name> --save[/white] - Switch and persist to workspace"
    )

    output.print("\n[yellow]Model Management:[/yellow]")
    output.print(
        "  • [white]/config model[/white] - List available models for current provider"
    )
    output.print(
        "  • [white]/config model <name>[/white] - Switch model (runtime only)"
    )
    output.print(
        "  • [white]/config model <name> --save[/white] - Switch model and persist to workspace"
    )

    output.print("\n[yellow]Configuration Display:[/yellow]")
    output.print(
        "  • [white]/config[/white] or [white]/config show[/white] - Show this configuration summary"
    )

    return f"Displayed configuration (provider: {provider}, profile: {profile})"


def config_provider_list(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    List all available LLM providers with their configuration status.

    Displays a table showing:
    - Provider name
    - Configuration status (configured or not)
    - Active indicator (● for currently active provider)

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    from lobster.config.llm_factory import LLMFactory
    from lobster.config.providers import ProviderRegistry

    available_providers = LLMFactory.get_available_providers()
    current_provider = client.provider_override or LLMFactory.get_current_provider()

    # Build provider table
    provider_table_data = {
        "title": "🔌 LLM Providers",
        "columns": [
            {"name": "Provider", "style": "cyan"},
            {"name": "Status", "style": "white"},
            {"name": "Active", "style": "green"},
        ],
        "rows": [],
    }

    # Dynamically fetch all registered providers from ProviderRegistry
    all_providers = ProviderRegistry.get_all()

    for provider_obj in all_providers:
        provider_name = provider_obj.name
        configured = (
            "✓ Configured"
            if provider_name in available_providers
            else "✗ Not configured"
        )
        active = "●" if provider_name == current_provider else ""

        status_style = "green" if provider_name in available_providers else "grey50"
        provider_table_data["rows"].append(
            [
                provider_obj.display_name,
                f"[{status_style}]{configured}[/{status_style}]",
                f"[bold green]{active}[/bold green]" if active else "",
            ]
        )

    output.print_table(provider_table_data)

    output.print("\n[cyan]💡 Usage:[/cyan]")
    output.print(
        "  • [white]/config provider <name>[/white] - Switch to specified provider (runtime)"
    )
    output.print(
        "  • [white]/config provider <name> --save[/white] - Switch and persist to workspace"
    )

    # Dynamically show available providers
    provider_names = ", ".join([p.name for p in all_providers])
    output.print(f"\n[cyan]Available providers:[/cyan] {provider_names}")

    if current_provider:
        output.print(f"\n[green]✓ Current provider: {current_provider}[/green]")

    return f"Listed providers (current: {current_provider})"


def config_provider_switch(
    client: "AgentClient", output: OutputAdapter, provider_name: str, save: bool = False
) -> Optional[str]:
    """
    Switch to a different LLM provider.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        provider_name: Name of provider to switch to (e.g., 'ollama', 'anthropic')
        save: If True, persist to workspace config; if False, runtime only

    Returns:
        Summary string for conversation history, or None
    """
    new_provider = provider_name.lower()

    if save:
        output.print(
            f"[yellow]Switching to {new_provider} provider and saving to workspace...[/yellow]"
        )
    else:
        output.print(
            f"[yellow]Switching to {new_provider} provider (runtime only)...[/yellow]"
        )

    # Switch runtime first
    result = client.switch_provider(new_provider)

    if not result["success"]:
        error_msg = result.get("error", "Unknown error")
        hint = result.get("hint", "")
        output.print(f"[red]✗ {error_msg}[/red]")
        if hint:
            output.print(f"[dim]{hint}[/dim]")
        return None

    # Success message
    output.print(
        f"[green]✓ Successfully switched to {result['provider']} provider[/green]"
    )

    # If not saving, show hint about persisting
    if not save:
        output.print(
            f"[dim]💡 Use [white]/config provider {new_provider} --save[/white] to persist this change[/dim]"
        )
        return f"Switched to {result['provider']} provider (runtime only)"

    # Persist to workspace config
    from lobster.config.workspace_config import WorkspaceProviderConfig

    try:
        workspace_path = Path(client.workspace_path)
        config = WorkspaceProviderConfig.load(workspace_path)
        config.global_provider = new_provider
        config.save(workspace_path)

        output.print(
            f"[green]✓ Saved to workspace config: {workspace_path / 'provider_config.json'}[/green]"
        )
        output.print(
            "[dim]This provider will be used for all future sessions in this workspace.[/dim]"
        )

        return f"Switched to {result['provider']} provider and saved to workspace"

    except Exception as e:
        output.print(f"[red]✗ Failed to save workspace config: {str(e)}[/red]")
        output.print("[dim]Check workspace directory permissions[/dim]")
        return None


def config_model_list(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    List available models for the current LLM provider.

    Displays a table with:
    - Model name
    - Display name
    - Description
    - Indicators for current and default models

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    from lobster.config.providers import get_provider
    from lobster.config.workspace_config import WorkspaceProviderConfig
    from lobster.core.config_resolver import ConfigResolver

    # Get current provider
    workspace_path = Path(client.workspace_path)
    resolver = ConfigResolver(workspace_path=workspace_path)
    current_provider, provider_source = resolver.resolve_provider(
        runtime_override=client.provider_override
    )

    try:
        provider_obj = get_provider(current_provider)
        if not provider_obj:
            output.print(f"[red]✗ Provider '{current_provider}' not registered[/red]")
            return None

        # For Ollama, check if server is available
        if current_provider == "ollama":
            if not provider_obj.is_available():
                output.print("[red]✗ Ollama server not accessible[/red]")
                output.print("[dim]Make sure Ollama is running: 'ollama serve'[/dim]")
                return None

        models = provider_obj.list_models()

        if not models:
            if current_provider == "ollama":
                output.print("[yellow]No Ollama models installed[/yellow]")
                output.print("\n[cyan]💡 Install a model:[/cyan]")
                output.print("  ollama pull llama3:8b-instruct")
            else:
                output.print(
                    f"[yellow]No models available for {current_provider}[/yellow]"
                )
            return None

        # Provider-specific table title
        provider_icons = {
            "anthropic": "🤖",
            "bedrock": "☁️",
            "ollama": "🦙",
            "gemini": "✨",
            "azure": "🔷",
        }
        icon = provider_icons.get(current_provider, "🤖")
        title = f"{icon} Available {current_provider.capitalize()} Models"

        # Get current model from config
        config = WorkspaceProviderConfig.load(workspace_path)
        current_model = (
            config.get_model_for_provider(current_provider)
            if WorkspaceProviderConfig.exists(workspace_path)
            else None
        )

        model_table_data = {
            "title": title,
            "columns": [
                {"name": "Model", "style": "yellow"},
                {"name": "Display Name", "style": "cyan"},
                {"name": "Description", "style": "white", "max_width": 50},
            ],
            "rows": [],
        }

        for model in models:
            is_current = "[green]●[/green]" if model.name == current_model else ""
            is_default = "[dim](default)[/dim]" if model.is_default else ""
            model_table_data["rows"].append(
                [
                    f"[bold]{model.name}[/bold] {is_current}",
                    f"{model.display_name} {is_default}",
                    model.description,
                ]
            )

        output.print_table(model_table_data)
        output.print(
            f"\n[cyan]Current provider:[/cyan] {current_provider} (from {provider_source})"
        )
        output.print("\n[cyan]💡 Usage:[/cyan]")
        output.print("  • [white]/config model <name>[/white] - Switch model (runtime)")
        output.print(
            "  • [white]/config model <name> --save[/white] - Switch + persist"
        )
        output.print(
            "  • [white]/config provider <name>[/white] - Change provider first"
        )

        if current_model:
            output.print(f"\n[green]✓ Current model: {current_model}[/green]")

        return f"Listed models for {current_provider} provider"

    except Exception as e:
        output.print(
            f"[red]✗ Failed to list models for {current_provider}: {str(e)}[/red]"
        )
        output.print("[dim]Check provider configuration[/dim]")
        return None


def config_model_switch(
    client: "AgentClient", output: OutputAdapter, model_name: str, save: bool = False
) -> Optional[str]:
    """
    Switch to a different model for the current provider.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        model_name: Name of model to switch to
        save: If True, persist to workspace config; if False, runtime only

    Returns:
        Summary string for conversation history, or None
    """
    from lobster.config.providers import get_provider
    from lobster.config.workspace_config import WorkspaceProviderConfig
    from lobster.core.config_resolver import ConfigResolver

    # Get current provider
    workspace_path = Path(client.workspace_path)
    resolver = ConfigResolver(workspace_path=workspace_path)
    current_provider, provider_source = resolver.resolve_provider(
        runtime_override=client.provider_override
    )

    try:
        provider_obj = get_provider(current_provider)
        if not provider_obj:
            output.print(f"[red]✗ Provider '{current_provider}' not registered[/red]")
            return None

        # For Ollama, check server availability
        if current_provider == "ollama":
            if not provider_obj.is_available():
                output.print("[red]✗ Ollama server not accessible[/red]")
                output.print("[dim]Make sure Ollama is running: 'ollama serve'[/dim]")
                return None

        # Validate model
        if not provider_obj.validate_model(model_name):
            model_names = provider_obj.get_model_names()
            available = ", ".join(model_names[:5])
            if len(model_names) > 5:
                available += ", ..."
            hint = f"Available models: {available}"
            if current_provider == "ollama":
                hint += f"\nInstall with: ollama pull {model_name}"
            output.print(
                f"[red]✗ Model '{model_name}' not valid for {current_provider}[/red]"
            )
            output.print(f"[dim]{hint}[/dim]")
            return None

        # Store in environment for this session
        env_var_map = {
            "ollama": "OLLAMA_DEFAULT_MODEL",
            "anthropic": "ANTHROPIC_MODEL",
            "bedrock": "BEDROCK_MODEL",
        }
        env_var = env_var_map.get(current_provider)
        if env_var:
            os.environ[env_var] = model_name

        output.print(f"[green]✓ Switched to model: {model_name}[/green]")
        output.print(f"[dim]Provider: {current_provider}[/dim]")

        if not save:
            output.print("[dim]This change is temporary (session only)[/dim]")
            output.print(f"[dim]To persist: /config model {model_name} --save[/dim]")
            return f"Switched to model {model_name} (runtime only)"

        # Persist to workspace config
        try:
            config = WorkspaceProviderConfig.load(workspace_path)
            config.set_model_for_provider(current_provider, model_name)
            config.save(workspace_path)

            output.print(
                f"[green]✓ Saved to workspace config ({current_provider}_model)[/green]"
            )
            output.print(
                f"[dim]Config file: {workspace_path}/provider_config.json[/dim]"
            )
            output.print(
                f"\n[dim]This model will be used for {current_provider} in this workspace[/dim]"
            )

            return f"Switched to model {model_name} and saved to workspace"

        except Exception as e:
            output.print(f"[red]✗ Failed to save configuration: {e}[/red]")
            output.print("[dim]Check file permissions[/dim]")
            return None

    except Exception as e:
        output.print(f"[red]✗ Failed to switch model: {str(e)}[/red]")
        output.print("[dim]Check provider configuration[/dim]")
        return None


# ============================================================================
# Config subcommand impl functions (extracted from cli.py Plan 08-02)
# ============================================================================


def config_test_impl(output_json: bool = False):
    """Test API connectivity and validate configuration (extracted from cli.py)."""
    import json as json_module
    import os
    from pathlib import Path

    from dotenv import load_dotenv
    from rich import box
    from rich.panel import Panel
    from rich.table import Table

    from lobster.ui import LobsterTheme
    from lobster.ui.console_manager import get_console_manager

    import typer

    console_manager = get_console_manager()
    console = console_manager.console

    import json as json_module

    from dotenv import load_dotenv

    # Results structure for JSON output
    test_results = {
        "valid": False,
        "env_file": None,
        "checks": {
            "llm_provider": {
                "status": "fail",
                "provider": None,
                "message": None,
                "hint": None,
            },
            "ncbi_api": {"status": "skip", "has_key": False, "message": None},
            "workspace": {"status": "fail", "path": None, "message": None},
        },
    }

    def log(msg: str, style: str = None):
        """Print message only if not in JSON mode."""
        if not output_json:
            if style:
                console.print(f"[{style}]{msg}[/{style}]")
            else:
                console.print(msg)

    # Check .env file exists
    env_file = Path.cwd() / ".env"
    test_results["env_file"] = str(env_file) if env_file.exists() else None

    if not env_file.exists():
        if output_json:
            test_results["checks"]["llm_provider"]["message"] = "No .env file found"
            print(json_module.dumps(test_results, indent=2))
        else:
            console.print("[red]❌ No .env file found in current directory[/red]")
            console.print("[dim]Run 'lobster init' to create configuration[/dim]")
        raise typer.Exit(1)

    load_dotenv()

    if not output_json:
        console.print()
        console.print(
            Panel.fit(
                f"[bold {LobsterTheme.PRIMARY_ORANGE}]🔍 Configuration Test[/bold {LobsterTheme.PRIMARY_ORANGE}]",
                border_style=LobsterTheme.PRIMARY_ORANGE,
                padding=(0, 2),
            )
        )
        console.print()
        console.print(f"[green]✅ Found .env file:[/green] {env_file}")
        console.print()

    # Test LLM Provider
    log("Testing LLM Provider...", "bold")
    try:
        from lobster.config.llm_factory import LLMFactory

        workspace_path = Path.cwd() / ".lobster_workspace"
        provider = LLMFactory.get_current_provider(workspace_path=workspace_path)
        if provider is None:
            test_results["checks"]["llm_provider"]["message"] = "No API keys found"
            test_results["checks"]["llm_provider"][
                "hint"
            ] = "Set ANTHROPIC_API_KEY, AWS_BEDROCK_ACCESS_KEY, or run Ollama"
            log("❌ No LLM provider configured", "red")
        else:
            # provider is a string, not an enum
            test_results["checks"]["llm_provider"]["provider"] = provider
            log(f"  Detected provider: {provider}", "yellow")

            # Provider-specific pre-checks for Ollama
            if provider == "ollama":
                ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                log(f"  Checking Ollama server at {ollama_url}...", "yellow")
                try:
                    import requests

                    resp = requests.get(f"{ollama_url}/api/tags", timeout=5)
                    if resp.status_code != 200:
                        raise Exception(
                            f"Ollama server returned status {resp.status_code}"
                        )
                    models = resp.json().get("models", [])
                    if not models:
                        test_results["checks"]["llm_provider"][
                            "message"
                        ] = "Ollama running but no models installed"
                        test_results["checks"]["llm_provider"][
                            "hint"
                        ] = "Run: ollama pull llama3.2:3b"
                        log("❌ Ollama: No models installed", "red")
                        provider = None
                    else:
                        log(f"  Ollama server: Running ({len(models)} models)", "green")
                except requests.exceptions.ConnectionError:
                    test_results["checks"]["llm_provider"][
                        "message"
                    ] = "Ollama server not running"
                    test_results["checks"]["llm_provider"][
                        "hint"
                    ] = "Start Ollama: ollama serve"
                    log("❌ Ollama server not accessible", "red")
                    provider = None
                except Exception as e:
                    test_results["checks"]["llm_provider"][
                        "message"
                    ] = f"Ollama error: {str(e)[:60]}"
                    log(f"❌ Ollama check failed: {e}", "red")
                    provider = None

            # Test LLM connectivity
            if provider is not None:
                try:
                    # Use provider's default model for connectivity test
                    from lobster.config.providers import get_provider

                    provider_obj = get_provider(provider)
                    default_model = (
                        provider_obj.get_default_model() if provider_obj else ""
                    )
                    test_config = {
                        "model_id": default_model,
                        "temperature": 1.0,
                        "max_tokens": 50,
                    }

                    test_llm = LLMFactory.create_llm(
                        test_config, "config_test", workspace_path=workspace_path
                    )
                    log("  Testing API connectivity...", "yellow")
                    test_llm.invoke("Reply with just 'ok'")

                    test_results["checks"]["llm_provider"]["status"] = "pass"
                    test_results["checks"]["llm_provider"]["message"] = "Connected"
                    log(f"✅ {provider} API: Connected", "green")
                except Exception as e:
                    error_msg = str(e)
                    if (
                        "invalid_api_key" in error_msg.lower()
                        or "authentication" in error_msg.lower()
                    ):
                        hint = "Check your API key"
                    elif "rate_limit" in error_msg.lower():
                        hint = "Rate limited - wait and retry"
                    elif (
                        "model" in error_msg.lower()
                        and "not found" in error_msg.lower()
                    ):
                        hint = "Model not available in your region/plan"
                    else:
                        hint = None

                    test_results["checks"]["llm_provider"]["message"] = error_msg[:100]
                    test_results["checks"]["llm_provider"]["hint"] = hint
                    log(f"❌ {provider} API: {error_msg[:60]}", "red")
    except Exception as e:
        test_results["checks"]["llm_provider"]["message"] = str(e)[:100]
        log(f"❌ LLM Provider test failed: {e}", "red")

    if not output_json:
        console.print()

    # Test NCBI API
    log("Testing NCBI API...", "bold")
    ncbi_key = os.environ.get("NCBI_API_KEY")
    test_results["checks"]["ncbi_api"]["has_key"] = bool(ncbi_key)

    if ncbi_key or os.environ.get("NCBI_EMAIL"):
        try:
            import urllib.request
            import xml.etree.ElementTree as ET

            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {"db": "pubmed", "term": "test", "retmax": "1", "retmode": "xml"}
            if ncbi_key:
                params["api_key"] = ncbi_key

            url = f"{base_url}?" + "&".join([f"{k}={v}" for k, v in params.items()])
            log("  Testing NCBI E-utilities...", "yellow")

            with urllib.request.urlopen(url, timeout=10) as response:
                root = ET.fromstring(response.read())
                error = root.find(".//ERROR")
                if error is not None:
                    test_results["checks"]["ncbi_api"]["status"] = "fail"
                    test_results["checks"]["ncbi_api"]["message"] = error.text
                    log(f"❌ NCBI API Error: {error.text}", "red")
                else:
                    test_results["checks"]["ncbi_api"]["status"] = "pass"
                    test_results["checks"]["ncbi_api"]["message"] = "Connected"
                    log(
                        f"✅ NCBI API: Connected {'(with API key)' if ncbi_key else ''}",
                        "green",
                    )
        except Exception as e:
            test_results["checks"]["ncbi_api"]["status"] = "fail"
            test_results["checks"]["ncbi_api"]["message"] = str(e)[:60]
            log(f"❌ NCBI API: {str(e)[:60]}", "red")
    else:
        test_results["checks"]["ncbi_api"]["status"] = "skip"
        test_results["checks"]["ncbi_api"]["message"] = "Not configured (optional)"
        log("⊘ NCBI API: Not configured (optional)", "dim")

    if not output_json:
        console.print()

    # Test Workspace
    log("Testing Workspace...", "bold")
    try:
        from lobster.core.workspace import resolve_workspace

        workspace_path = resolve_workspace(explicit_path=None, create=True)
        test_results["checks"]["workspace"]["path"] = str(workspace_path)

        test_file = workspace_path / ".config_test_write"
        try:
            test_file.write_text("test")
            test_file.unlink()
            test_results["checks"]["workspace"]["status"] = "pass"
            test_results["checks"]["workspace"]["message"] = "Writable"
            log("✅ Workspace: Writable", "green")
            log(f"  Path: {workspace_path}", "dim")
        except PermissionError:
            test_results["checks"]["workspace"]["message"] = "Permission denied"
            log("❌ Workspace: Permission denied", "red")
        except Exception as e:
            test_results["checks"]["workspace"]["message"] = str(e)[:60]
            log(f"❌ Workspace: {e}", "red")
    except Exception as e:
        test_results["checks"]["workspace"]["message"] = str(e)[:60]
        log(f"❌ Workspace setup failed: {e}", "red")

    # Determine overall validity
    test_results["valid"] = test_results["checks"]["llm_provider"]["status"] == "pass"

    # Output
    if output_json:
        print(json_module.dumps(test_results, indent=2))
    else:
        console.print()

        # Summary table
        table = Table(title="Configuration Test Summary", box=box.ROUNDED)
        table.add_column("Service", style="cyan", no_wrap=True)
        table.add_column("Status", style="bold", no_wrap=True)
        table.add_column("Details", style="dim")

        for check_name, check_data in test_results["checks"].items():
            status = check_data["status"]
            icon = "✅" if status == "pass" else ("❌" if status == "fail" else "⊘")
            style = (
                "green" if status == "pass" else ("red" if status == "fail" else "dim")
            )
            details = check_data.get("message") or ""
            if check_name == "llm_provider" and check_data.get("provider"):
                details = f"{check_data['provider']}: {details}"
            table.add_row(
                check_name.replace("_", " ").title(),
                f"[{style}]{icon}[/{style}]",
                details,
            )

        console.print(table)
        console.print()

        if test_results["valid"]:
            console.print(
                Panel.fit(
                    "[bold green]✅ Configuration Valid[/bold green]\n\n"
                    "All required services are accessible.\n"
                    f"You can now run: [bold {LobsterTheme.PRIMARY_ORANGE}]lobster chat[/bold {LobsterTheme.PRIMARY_ORANGE}]",
                    border_style="green",
                    padding=(1, 2),
                )
            )
        else:
            console.print(
                Panel.fit(
                    "[bold red]❌ Configuration Issues Detected[/bold red]\n\n"
                    "Please check your API keys in the .env file.\n"
                    f"Run: [bold {LobsterTheme.PRIMARY_ORANGE}]lobster init --force[/bold {LobsterTheme.PRIMARY_ORANGE}] to reconfigure",
                    border_style="red",
                    padding=(1, 2),
                )
            )

    if not test_results["valid"]:
        raise typer.Exit(1)




def list_models_impl():
    """List all available model presets (extracted from cli.py)."""
    from rich import box
    from rich.table import Table

    from lobster.config.agent_config import LobsterAgentConfigurator
    from lobster.ui.console_manager import get_console_manager

    console = get_console_manager().console

    """List all available model presets."""
    configurator = LobsterAgentConfigurator()
    models = configurator.list_available_models()

    console.print("\n[cyan]🤖 Available Model Presets[/cyan]")
    console.print("[cyan]" + "=" * 60 + "[/cyan]")

    table = Table(
        box=box.ROUNDED,
        border_style="cyan",
        title="🤖 Available Model Presets",
        title_style="bold cyan",
    )

    table.add_column("Preset Name", style="bold white")
    table.add_column("Tier", style="cyan")
    table.add_column("Region", style="white")
    table.add_column("Temperature", style="white")
    table.add_column("Description", style="white")

    for name, config in models.items():
        description = (
            config.description[:40] + "..."
            if len(config.description) > 40
            else config.description
        )
        table.add_row(
            name,
            config.tier.value.title(),
            config.region,
            f"{config.temperature}",
            description,
        )

    console.print(table)




def list_profiles_impl():
    """List all available testing profiles (extracted from cli.py)."""
    from lobster.config.agent_config import LobsterAgentConfigurator
    from lobster.ui.console_manager import get_console_manager

    console = get_console_manager().console

    """List all available testing profiles."""
    configurator = LobsterAgentConfigurator()
    profiles = configurator.list_available_profiles()

    console.print("\n[cyan]⚙️  Available Testing Profiles[/cyan]")
    console.print("[cyan]" + "=" * 60 + "[/cyan]")

    for profile_name, config in profiles.items():
        console.print(f"\n[yellow]📋 {profile_name.title()}[/yellow]")
        for agent, model in config.items():
            console.print(f"   {agent}: {model}")




def show_config_impl(workspace=None, show_all: bool = False):
    """Show current runtime configuration (extracted from cli.py)."""
    import os
    from pathlib import Path

    from rich.panel import Panel

    from lobster.core.workspace import resolve_workspace
    from lobster.ui import LobsterTheme
    from lobster.ui.console_manager import get_console_manager

    import typer

    console = get_console_manager().console

    """Show current runtime configuration from ConfigResolver and ProviderRegistry."""
    from lobster.config.agent_registry import AGENT_REGISTRY
    from lobster.config.global_config import GlobalProviderConfig
    from lobster.config.providers import get_provider
    from lobster.config.subscription_tiers import (
        get_tier_display_name,
        is_agent_available,
    )
    from lobster.config.workspace_config import WorkspaceProviderConfig
    from lobster.core.config_resolver import ConfigResolver, ConfigurationError
    from lobster.core.license_manager import get_current_tier

    # Resolve workspace path
    workspace_path = resolve_workspace(workspace, create=False)

    # Get ConfigResolver instance
    resolver = ConfigResolver.get_instance(workspace_path)

    # Get license tier
    current_tier = get_current_tier()

    console.print()
    console.print(
        Panel.fit(
            f"[bold {LobsterTheme.PRIMARY_ORANGE}]🔧 Lobster AI Runtime Configuration[/bold {LobsterTheme.PRIMARY_ORANGE}]",
            border_style=LobsterTheme.PRIMARY_ORANGE,
            padding=(0, 2),
        )
    )
    console.print()

    # Section 1: License & Workspace
    console.print("[bold cyan]📍 Environment[/bold cyan]")
    console.print(f"   Workspace: [yellow]{workspace_path}[/yellow]")
    console.print(
        f"   License Tier: [green]{get_tier_display_name(current_tier)}[/green]"
    )
    console.print()

    # Section 2: Provider Configuration
    console.print("[bold cyan]🔌 Provider Configuration[/bold cyan]")

    try:
        provider_name, provider_source = resolver.resolve_provider()
        provider_obj = get_provider(provider_name)

        console.print(
            f"   Active Provider: [green]{provider_obj.display_name if provider_obj else provider_name}[/green]"
        )
        console.print(f"   Source: [dim]{provider_source}[/dim]")

        if provider_obj:
            # Show provider-specific information
            default_model = provider_obj.get_default_model()
            console.print(f"   Default Model: [yellow]{default_model}[/yellow]")

            # Show available models for this provider
            models = provider_obj.list_models()
            if models:
                console.print(f"   Available Models: [dim]{len(models)} model(s)[/dim]")

    except ConfigurationError as e:
        console.print("   [red]✗ No provider configured[/red]")
        console.print(f"   [dim]Error: {str(e)}[/dim]")
        console.print(
            "   [yellow]💡 Run 'lobster init' to configure a provider[/yellow]"
        )

    console.print()

    # Section 3: Profile Configuration
    console.print("[bold cyan]⚙️  Profile Configuration[/bold cyan]")

    try:
        profile, profile_source = resolver.resolve_profile()
        console.print(f"   Active Profile: [green]{profile}[/green]")
        console.print(f"   Source: [dim]{profile_source}[/dim]")
    except Exception:
        console.print(
            "   Active Profile: [yellow]production[/yellow] [dim](default)[/dim]"
        )

    console.print()

    # Section 4: Configuration Files
    console.print("[bold cyan]📁 Configuration Files[/bold cyan]")

    # Workspace config
    workspace_config_path = workspace_path / "provider_config.json"
    if workspace_config_path.exists():
        console.print(f"   [green]✓[/green] Workspace: {workspace_config_path}")
        try:
            ws_config = WorkspaceProviderConfig.load(workspace_path)
            if ws_config.global_provider:
                console.print(
                    f"      Provider: [yellow]{ws_config.global_provider}[/yellow]"
                )
            if ws_config.profile:
                console.print(f"      Profile: [yellow]{ws_config.profile}[/yellow]")
        except Exception as e:
            console.print(f"      [dim]Could not read: {str(e)}[/dim]")
    else:
        console.print(f"   [dim]○ Workspace: {workspace_config_path} (not found)[/dim]")

    # Global config
    global_config_path = GlobalProviderConfig.get_config_path()
    if global_config_path.exists():
        console.print(f"   [green]✓[/green] Global: {global_config_path}")
        try:
            global_config = GlobalProviderConfig.load()
            if global_config.default_provider:
                console.print(
                    f"      Provider: [yellow]{global_config.default_provider}[/yellow]"
                )
            if global_config.default_profile:
                console.print(
                    f"      Profile: [yellow]{global_config.default_profile}[/yellow]"
                )
        except Exception as e:
            console.print(f"      [dim]Could not read: {str(e)}[/dim]")
    else:
        console.print(f"   [dim]○ Global: {global_config_path} (not found)[/dim]")

    console.print()

    # Section 5: Per-Agent Configuration
    console.print("[bold cyan]🤖 Agent Configuration[/bold cyan]")
    console.print()

    # Get agent configurations from settings (for temperature/thinking)
    from lobster.config.settings import get_settings

    settings = get_settings()

    displayed_count = 0
    filtered_count = 0

    for agent_name, agent_cfg in AGENT_REGISTRY.items():
        # Check if agent is available for current tier
        if not show_all and not is_agent_available(agent_name, current_tier):
            filtered_count += 1
            continue

        displayed_count += 1

        try:
            # Get model parameters from settings (temperature, thinking)
            model_params = settings.get_agent_llm_params(agent_name)

            # Try to resolve model for this agent
            try:
                provider_name, _ = resolver.resolve_provider()
                model_id, model_source = resolver.resolve_model(
                    agent_name=agent_name, runtime_override=None, provider=provider_name
                )

                # If no model resolved, get provider's default model
                if not model_id:
                    provider_obj = get_provider(provider_name)
                    if provider_obj:
                        model_id = provider_obj.get_default_model()
                        model_source = "provider default"
                    else:
                        model_id = model_params.get("model_id", "unknown")
                        model_source = "profile config"

            except Exception:
                # Fallback: try to get from provider's default
                try:
                    provider_name, _ = resolver.resolve_provider()
                    provider_obj = get_provider(provider_name)
                    if provider_obj:
                        model_id = provider_obj.get_default_model()
                        model_source = "provider default"
                    else:
                        model_id = model_params.get("model_id", "unknown")
                        model_source = "profile config"
                except Exception:
                    model_id = model_params.get("model_id", "unknown")
                    model_source = "profile config"

            # Display agent configuration
            console.print(
                f"   [bold white]{agent_cfg.display_name}[/bold white] ([dim]{agent_name}[/dim])"
            )
            console.print(
                f"      Model: [yellow]{model_id}[/yellow] [dim](from {model_source})[/dim]"
            )
            console.print(
                f"      Temperature: [cyan]{model_params.get('temperature', 1.0)}[/cyan]"
            )

            # Show thinking configuration if present
            additional_fields = model_params.get("additional_model_request_fields", {})
            if "thinking" in additional_fields:
                thinking = additional_fields["thinking"]
                budget = thinking.get("budget_tokens", "unknown")
                console.print(
                    f"      [dim]🧠 Thinking: Enabled (Budget: {budget} tokens)[/dim]"
                )

            console.print()

        except Exception as e:
            console.print(
                f"   [bold white]{agent_cfg.display_name}[/bold white]: [red]Error - {str(e)}[/red]"
            )
            console.print()

    # Summary
    if filtered_count > 0 and not show_all:
        console.print(f"[dim]{'─' * 60}[/dim]")
        console.print(
            f"[cyan]📊 Summary:[/cyan] Showing {displayed_count} agents available for [green]{current_tier}[/green] tier"
        )
        console.print(f"   [dim]({filtered_count} premium agents hidden)[/dim]")
        console.print(
            "   [yellow]💡 Use '--show-all' to see all configured agents[/yellow]"
        )

    console.print()




def test_impl(profile=None, agent=None):
    """Test LLM provider connectivity and configuration (extracted from cli.py)."""
    import os
    from pathlib import Path

    from rich.panel import Panel

    from lobster.ui import LobsterTheme
    from lobster.ui.console_manager import get_console_manager

    import typer

    console = get_console_manager().console

    # Load .env file for provider detection
    from dotenv import load_dotenv

    load_dotenv()

    # If no profile specified, test current provider connectivity
    if not profile:
        console.print()
        console.print(
            Panel.fit(
                f"[bold {LobsterTheme.PRIMARY_ORANGE}]🔍 Testing LLM Provider[/bold {LobsterTheme.PRIMARY_ORANGE}]",
                border_style=LobsterTheme.PRIMARY_ORANGE,
                padding=(0, 2),
            )
        )
        console.print()

        from lobster.config.providers import ProviderRegistry

        # Auto-detect current provider (use ProviderRegistry directly)
        try:
            # Get list of configured providers
            configured = ProviderRegistry.get_configured_providers()

            if not configured:
                console.print("[red]❌ No LLM provider configured[/red]")
                console.print(
                    "[yellow]💡 Run 'lobster init' to configure a provider[/yellow]"
                )
                raise typer.Exit(1)

            # Check environment variable first
            import os

            explicit_provider = os.getenv("LOBSTER_LLM_PROVIDER")
            if explicit_provider and explicit_provider in [p.name for p in configured]:
                provider = explicit_provider
            else:
                # Use first configured provider
                provider = configured[0].name

            if provider is None:
                console.print("[red]❌ No LLM provider configured[/red]")
                console.print(
                    "[yellow]💡 Run 'lobster init' to configure a provider[/yellow]"
                )
                raise typer.Exit(1)

            # provider is a string, not an enum
            console.print(f"[cyan]Detected provider:[/cyan] {provider}")
            console.print()

            # Test provider connectivity
            console.print("[yellow]Testing API connectivity...[/yellow]")

            try:
                # Get provider object and create model directly (bypass ConfigResolver)
                from lobster.config.providers import get_provider

                provider_obj = get_provider(provider)

                if not provider_obj:
                    raise Exception(f"Provider '{provider}' not found in registry")

                # Get default model for provider
                default_model = provider_obj.get_default_model()

                # Create test model directly via provider
                test_llm = provider_obj.create_chat_model(
                    model_id=default_model, temperature=1.0, max_tokens=50
                )

                test_llm.invoke("Reply with just 'ok'")

                console.print()
                console.print(
                    Panel.fit(
                        f"[bold green]✅ Provider Test Successful[/bold green]\n\n"
                        f"Provider: [cyan]{provider}[/cyan]\n"
                        f"Status: [green]Connected[/green]\n\n"
                        f"Your {provider} provider is working correctly.\n"
                        f"Run [bold {LobsterTheme.PRIMARY_ORANGE}]lobster chat[/bold {LobsterTheme.PRIMARY_ORANGE}] to start analyzing!",
                        border_style="green",
                        padding=(1, 2),
                    )
                )
                return True

            except Exception as e:
                error_msg = str(e)
                console.print()
                console.print(
                    Panel.fit(
                        f"[bold red]❌ Provider Test Failed[/bold red]\n\n"
                        f"Provider: [cyan]{provider}[/cyan]\n"
                        f"Error: [red]{error_msg[:100]}[/red]\n\n"
                        f"[yellow]💡 Check your API keys in .env file[/yellow]\n"
                        f"Run [bold {LobsterTheme.PRIMARY_ORANGE}]lobster init --force[/bold {LobsterTheme.PRIMARY_ORANGE}] to reconfigure",
                        border_style="red",
                        padding=(1, 2),
                    )
                )
                raise typer.Exit(1)

        except Exception as e:
            console.print(f"[red]❌ Error: {str(e)}[/red]")
            raise typer.Exit(1)

    # Profile-based testing (original functionality)
    try:
        configurator = initialize_configurator(profile=profile)

        if agent:
            # Test specific agent
            try:
                config = configurator.get_agent_model_config(agent)
                configurator.get_llm_params(agent)

                console.print(
                    f"\n[green]✅ Agent '{agent}' configuration is valid[/green]"
                )
                console.print(f"   Model: {config.model_config.model_id}")
                console.print(f"   Tier: {config.model_config.tier.value}")
                console.print(f"   Region: {config.model_config.region}")

            except KeyError:
                console.print(
                    f"\n[red]❌ Agent '{agent}' not found in profile '{profile}'[/red]"
                )
                return False
        else:
            # Test all agents dynamically
            console.print(f"\n[yellow]🧪 Testing Profile: {profile}[/yellow]")
            all_valid = True

            # Get all agents from the configurator's DEFAULT_AGENTS
            available_agents = configurator.DEFAULT_AGENTS

            for agent_name in available_agents:
                try:
                    config = configurator.get_agent_model_config(agent_name)
                    configurator.get_llm_params(agent_name)
                    console.print(
                        f"   [green]✅ {agent_name}: {config.model_config.model_id}[/green]"
                    )
                except Exception as e:
                    console.print(f"   [red]❌ {agent_name}: {str(e)}[/red]")
                    all_valid = False

            if all_valid:
                console.print(
                    f"\n[green]🎉 Profile '{profile}' is fully configured and valid![/green]"
                )
            else:
                console.print(
                    f"\n[yellow]⚠️  Profile '{profile}' has configuration issues[/yellow]"
                )

        return True

    except Exception as e:
        console.print(f"\n[red]❌ Error testing configuration: {str(e)}[/red]")
        return False




def create_custom_impl():
    """Interactive creation of custom configuration (extracted from cli.py)."""
    import json
    from pathlib import Path

    from rich.prompt import Prompt

    from lobster.config.agent_config import LobsterAgentConfigurator
    from lobster.ui.console_manager import get_console_manager

    console = get_console_manager().console

    """Interactive creation of custom configuration."""
    console.print("\n[cyan]🛠️  Create Custom Configuration[/cyan]")
    console.print("[cyan]" + "=" * 50 + "[/cyan]")

    configurator = LobsterAgentConfigurator()
    available_models = configurator.list_available_models()

    # Show available models
    console.print("\n[yellow]Available models:[/yellow]")
    for i, (name, config) in enumerate(available_models.items(), 1):
        console.print(f"{i:2}. {name} ({config.tier.value}, {config.region})")

    config_data = {"profile": "custom", "agents": {}}

    # Use dynamic agent list
    agents = configurator.DEFAULT_AGENTS

    for agent in agents:
        console.print(f"\n[yellow]Configuring {agent}:[/yellow]")
        console.print("Choose a model preset (enter number or name):")

        choice = Prompt.ask(f"Model for {agent}")

        # Handle numeric choice
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(available_models):
                model_name = list(available_models.keys())[idx]
            else:
                console.print(
                    "[yellow]Invalid choice, using default (claude-sonnet)[/yellow]"
                )
                model_name = "claude-sonnet"
        else:
            # Handle name choice
            if choice in available_models:
                model_name = choice
            else:
                console.print(
                    "[yellow]Invalid choice, using default (claude-sonnet)[/yellow]"
                )
                model_name = "claude-sonnet"

        model_config = available_models[model_name]
        config_data["agents"][agent] = {
            "model_config": {
                "provider": model_config.provider.value,
                "model_id": model_config.model_id,
                "tier": model_config.tier.value,
                "temperature": model_config.temperature,
                "region": model_config.region,
                "description": model_config.description,
            },
            "enabled": True,
            "custom_params": {},
        }

        console.print(f"   [green]Selected: {model_name}[/green]")

    # Save configuration
    config_file = "config/custom_agent_config.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)

    console.print(f"\n[green]✅ Custom configuration saved to: {config_file}[/green]")
    console.print("[yellow]To use this configuration, set:[/yellow]")
    console.print(f"   export LOBSTER_CONFIG_FILE={config_file}", style="yellow")




def generate_env_impl():
    """Generate .env template with all available options (extracted from cli.py)."""
    from lobster.ui.console_manager import get_console_manager

    console = get_console_manager().console

    """Generate .env template with all available options."""
    template = """# LOBSTER AI Configuration Template
# Copy this file to .env and configure as needed

# =============================================================================
# API KEYS (Required)
# =============================================================================
AWS_BEDROCK_ACCESS_KEY="your-aws-access-key-here"
AWS_BEDROCK_SECRET_ACCESS_KEY="your-aws-secret-key-here"
NCBI_API_KEY="your-ncbi-api-key-here"

# =============================================================================
# AGENT CONFIGURATION (Professional System)
# =============================================================================

# Profile-based configuration (recommended)
# Available profiles: development, production, performance, max
LOBSTER_PROFILE=production

# OR use custom configuration file
# LOBSTER_CONFIG_FILE=config/custom_agent_config.json

# Per-agent model overrides (optional)
# Available models: claude-haiku, claude-sonnet, claude-sonnet-eu, claude-opus, claude-opus-eu, claude-3-7-sonnet, claude-3-7-sonnet-eu
# LOBSTER_SUPERVISOR_MODEL=claude-haiku
# LOBSTER_TRANSCRIPTOMICS_EXPERT_MODEL=claude-opus
# LOBSTER_METHOD_AGENT_MODEL=claude-sonnet
# LOBSTER_GENERAL_CONVERSATION_MODEL=claude-haiku

# Global model override (overrides all agents)
# LOBSTER_GLOBAL_MODEL=claude-sonnet

# Per-agent temperature overrides
# LOBSTER_SUPERVISOR_TEMPERATURE=0.5
# LOBSTER_TRANSCRIPTOMICS_EXPERT_TEMPERATURE=0.7
# LOBSTER_METHOD_AGENT_TEMPERATURE=0.3

# =============================================================================
# APPLICATION SETTINGS
# =============================================================================

# Server configuration
PORT=8501
HOST=0.0.0.0
DEBUG=False

# Data processing
LOBSTER_MAX_FILE_SIZE_MB=500
LOBSTER_CLUSTER_RESOLUTION=0.5
LOBSTER_CACHE_DIR=data/cache

# =============================================================================
# EXAMPLE CONFIGURATIONS
# =============================================================================

# Example 1: Development setup (Claude Sonnet 4 - fastest, most affordable)
# LOBSTER_PROFILE=development

# Example 2: Production setup (Claude Sonnet 4 + Sonnet 4.5 supervisor)
# LOBSTER_PROFILE=production

# Example 3: Performance setup (Claude Sonnet 4.5 - highest quality)
# LOBSTER_PROFILE=performance

# Example 4: Max setup (Claude Opus 4.5 supervisor - most capable, most expensive)
# LOBSTER_PROFILE=max
"""

    with open(".env.template", "w") as f:
        f.write(template)

    console.print("[green]✅ Environment template saved to: .env.template[/green]")
    console.print("[yellow]Copy this file to .env and configure your API keys[/yellow]")

