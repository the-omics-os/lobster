"""
Shared configuration commands for CLI and Dashboard.

Extracted from cli.py to enable reuse across interfaces.
All commands accept OutputAdapter for UI-agnostic rendering.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from lobster.core.client import AgentClient

from lobster.cli_internal.commands.output_adapter import (
    OutputAdapter,
    OutputBlock,
    alert_block,
    hint_block,
    kv_block,
    list_block,
    section_block,
    table_block,
)
from lobster.i18n import h, t

logger = logging.getLogger(__name__)

_NCBI_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


def _check_ncbi_eutils(api_key: Optional[str], timeout: float = 10.0) -> tuple:
    """Run one NCBI E-utilities search; return (status, message) for config test.

    The response is requested as JSON, so no XML parser touches network data.
    The API key travels as a query parameter and is masked in any message.
    """
    import httpx

    params = {"db": "pubmed", "term": "test", "retmax": "1", "retmode": "json"}
    if api_key:
        params["api_key"] = api_key

    def masked(text: str) -> str:
        return text.replace(api_key, "***") if api_key else text

    try:
        response = httpx.get(_NCBI_ESEARCH_URL, params=params, timeout=timeout)
        payload = response.json()
    except Exception as exc:
        # Optional probe: any client or transport failure is a failed check.
        return "fail", masked(str(exc) or type(exc).__name__)[:100]

    if not isinstance(payload, dict):
        return "fail", f"HTTP {response.status_code}: unexpected response"
    result = payload.get("esearchresult")
    error = payload.get("error") or (
        result.get("ERROR") if isinstance(result, dict) else None
    )
    if error:
        return "fail", masked(str(error))[:100]
    if response.status_code != 200 or not isinstance(result, dict):
        return "fail", f"HTTP {response.status_code}: unexpected response"
    return "pass", "Connected"


def _render_blocks(output: OutputAdapter, blocks: list[OutputBlock]) -> None:
    output.render_blocks(blocks)


def _is_protocol_output(output: OutputAdapter) -> bool:
    """Treat protocol adapters by interface/name so tests and shims use the compact path."""
    return output.__class__.__name__ == "ProtocolOutputAdapter"


def _print_table_or_empty(
    output: OutputAdapter, table_data: dict, empty_message: str
) -> bool:
    """Render a table when rows exist, otherwise print a compact empty-state note."""
    if table_data.get("rows"):
        output.print_table(table_data)
        return True

    title = table_data.get("title")
    if title:
        output.print(str(title))
    output.print(f"[dim]{empty_message}[/dim]")
    return False


def _print_config_command_reference(
    output: OutputAdapter, table_width: int = 100
) -> None:
    """Render a structured config command reference that works in both Rich and Go mode."""
    output.print_table(
        {
            "title": h("💡 Config Commands"),
            "width": table_width,
            "columns": [
                {"name": h("Area"), "style": "cyan", "width": 12},
                {
                    "name": h("Command"),
                    "style": "white",
                    "width": 34,
                    "overflow": "fold",
                },
                {
                    "name": h("Purpose"),
                    "style": "dim",
                    "width": 46,
                    "overflow": "fold",
                },
            ],
            "rows": [
                [
                    h("Provider"),
                    "/config provider",
                    h("List providers and show the active runtime provider."),
                ],
                [
                    h("Provider"),
                    "/config provider <name>",
                    h("Switch provider for the current session only."),
                ],
                [
                    h("Provider"),
                    "/config provider <name> --save",
                    h("Switch provider and persist it to this workspace."),
                ],
                [
                    h("Model"),
                    "/config model",
                    h("List models available for the active provider."),
                ],
                [
                    h("Model"),
                    "/config model <name>",
                    h("Switch model for the current session only."),
                ],
                [
                    h("Model"),
                    "/config model <name> --save",
                    h("Switch model and persist it to this workspace."),
                ],
                [
                    h("Display"),
                    "/config or /config show",
                    h("Show the full configuration summary."),
                ],
            ],
        }
    )


def _print_config_model_command_reference(
    output: OutputAdapter, table_width: int = 100
) -> None:
    """Render focused command help for `/config model` using structured output."""
    output.print_table(
        {
            "title": h("💡 Model Commands"),
            "width": table_width,
            "columns": [
                {
                    "name": h("Command"),
                    "style": "white",
                    "width": 34,
                    "overflow": "fold",
                },
                {
                    "name": h("Purpose"),
                    "style": "dim",
                    "width": 62,
                    "overflow": "fold",
                },
            ],
            "rows": [
                [
                    "/config model <name>",
                    h("Switch the active model for the current session only."),
                ],
                [
                    "/config model <name> --save",
                    h("Switch the active model and persist it to this workspace."),
                ],
                [
                    "/config provider",
                    h("List providers and confirm which model catalog is active."),
                ],
                [
                    "/config provider <name>",
                    h(
                        "Change providers first when you need a different model catalog."
                    ),
                ],
            ],
        }
    )


def _source_label(source: str) -> str:
    """Display text for a config source. The raw English value stays in JSON."""
    prefixes = (
        ("runtime flag", h("runtime flag")),
        ("workspace config", h("workspace config")),
        ("global user config", h("global user config")),
        ("global provider", h("global provider")),
        ("provider default", h("provider default")),
        ("profile config", h("profile config")),
        ("Explicit agent list", h("Explicit agent list")),
        ("Defaults", h("Defaults")),
        ("Preset:", h("Preset:")),
    )
    for english, label in prefixes:
        if source.startswith(english):
            return label + source[len(english) :]
    return source


def _truncate_middle(text: str, max_length: int) -> str:
    """Shorten a long string while preserving both ends."""
    if max_length <= 0 or len(text) <= max_length:
        return text
    if max_length <= 3:
        return text[:max_length]
    keep = max_length - 3
    left = keep // 2
    right = keep - left
    return f"{text[:left]}...{text[-right:]}"


def _summarize_names(names: list[str], *, limit: int = 3, max_length: int = 42) -> str:
    """Return a compact preview of display names for table cells."""
    if not names:
        return "-"

    shown = names[:limit]
    summary = ", ".join(shown)
    remaining = len(names) - len(shown)
    if remaining > 0:
        summary += f" (+{remaining} more)"
    return _truncate_middle(summary, max_length)


def _resolve_agent_composition_summary(
    workspace_path: Path,
) -> tuple[str, Optional[str], list[str], int, int]:
    """Return compact agent-composition facts without rendering the full table."""
    from lobster.config.agent_presets import expand_preset, get_preset_description
    from lobster.config.workspace_agent_config import WorkspaceAgentConfig
    from lobster.core.component_registry import component_registry

    config = WorkspaceAgentConfig.load(workspace_path)
    installed_agents = component_registry.list_agents()

    if config.preset:
        preset_agents = expand_preset(config.preset)
        description = get_preset_description(config.preset)
        if preset_agents:
            config_source = f"Preset: {config.preset}"
            config_detail = description or None
            display_agents = preset_agents
        else:
            config_source = f"Preset: {config.preset} (invalid, using defaults)"
            config_detail = None
            display_agents = list(installed_agents.keys())
    elif config.enabled_agents:
        config_source = "Explicit agent list"
        config_detail = f"{len(config.enabled_agents)} agents configured in config.toml"
        display_agents = config.enabled_agents
    else:
        config_source = "Defaults"
        config_detail = "All available agents enabled"
        display_agents = list(installed_agents.keys())

    installed_count = sum(
        1 for agent_name in display_agents if agent_name in installed_agents
    )
    total_count = len(display_agents)
    return config_source, config_detail, display_agents, installed_count, total_count


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

    output.print(h("\n[bold cyan]🔀 Agent Hierarchy[/bold cyan]"))
    output.print("[dim]" + "─" * 96 + "[/dim]")  # Match table width (100 - borders)
    output.print(h("[bold white]supervisor[/bold white] [dim](orchestrator)[/dim]"))

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
    from lobster.core.component_registry import component_registry

    # Get all installed agents from ComponentRegistry (single source of truth)
    installed_agents = component_registry.list_agents()
    config_source, config_detail, display_agents, _installed, _total = (
        _resolve_agent_composition_summary(workspace_path)
    )

    # Display configuration source
    output.print(h("\n[bold cyan]📦 Agent Composition[/bold cyan]"))
    output.print("[dim]" + "-" * 96 + "[/dim]")
    output.print(
        h("  Configuration: [bold cyan]{config_source}[/bold cyan]").format(
            config_source=_source_label(config_source)
        )
    )
    if config_detail:
        output.print(f"  [dim]{config_detail}[/dim]")
    output.print("")

    # Build agent table
    agent_table_data = {
        "title": None,  # No title, we already have the section header
        "width": table_width,
        "columns": [
            {"name": h("Agent"), "style": "cyan"},
            {"name": h("Status"), "style": "white"},
            {"name": h("Tier"), "style": "yellow"},
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
                    f"[green]{h('Installed')}[/green]",
                    f"[yellow]{tier_display}[/yellow]",
                ]
            )
        else:
            # Agent is missing
            missing_agents.append(agent_name)
            agent_table_data["rows"].append(
                [
                    f"[dim]{agent_name}[/dim]",
                    f"[red]{h('Missing')}[/red]",
                    "[dim]-[/dim]",
                ]
            )

    if not display_agents:
        output.print(
            h("[dim]No agents are configured or discoverable in this workspace.[/dim]")
        )
        return config_source, 0, 0

    output.print_table(agent_table_data)

    # Summary
    total_count = len(display_agents)
    if installed_count == total_count:
        output.print(
            h(
                "\n[green]✓ {installed_count} of {total_count} agents installed and ready[/green]"
            ).format(installed_count=installed_count, total_count=total_count)
        )
    else:
        output.print(
            h(
                "\n[yellow]⚠ {installed_count} of {total_count} agents installed[/yellow]"
            ).format(installed_count=installed_count, total_count=total_count)
        )

        # Helpful hint for missing agents
        if missing_agents:
            output.print(h("\n[dim]To install missing agents:[/dim]"))
            output.print("  [white]lobster agents install <package>[/white]")
            output.print(
                h("[dim]Run 'lobster agents list' to see available packages[/dim]")
            )

    return config_source, installed_count, total_count


def config_show(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Show current configuration including provider, profile, config files, and agent models.

    Displays a compact configuration summary that fits the CLI TUI:
    1. Current Configuration (provider, profile)
    2. Configuration Files (workspace, global)
    3. Agent model assignments grouped by resolved model/source
    4. Agent configuration summary and focused follow-up commands

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
    protocol_mode = _is_protocol_output(output)

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
        "title": h("⚙️  Current Configuration"),
        "width": TABLE_WIDTH,
        "columns": [
            {"name": h("Setting"), "style": "cyan"},
            {"name": h("Value"), "style": "white"},
            {"name": h("Source"), "style": "yellow"},
        ],
        "rows": [
            [h("Provider"), f"[bold]{provider}[/bold]", _source_label(p_source)],
            [h("Profile"), f"[bold]{profile}[/bold]", _source_label(pf_source)],
        ],
    }

    output.print_table(config_table_data)

    # ========================================================================
    # Table 2: Config Files Status
    # ========================================================================
    workspace_status = (
        h("[green]✓ Exists[/green]")
        if workspace_config_exists
        else h("[grey50]✗ Not found[/grey50]")
    )
    workspace_path_str = str(Path(client.workspace_path) / "provider_config.json")

    global_status = (
        h("[green]✓ Exists[/green]")
        if global_config_exists
        else h("[grey50]✗ Not found[/grey50]")
    )
    global_path_str = str(global_config_path)

    status_table_data = {
        "title": h("📁 Configuration Files"),
        "width": TABLE_WIDTH,
        "columns": [
            {"name": h("Location"), "style": "cyan"},
            {"name": h("Status"), "style": "white"},
            {"name": h("Path"), "style": "dim", "overflow": "ellipsis"},
        ],
        "rows": [
            [
                h("Workspace Config"),
                workspace_status,
                _truncate_middle(workspace_path_str, 56),
            ],
            [h("Global Config"), global_status, _truncate_middle(global_path_str, 56)],
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
        "title": h("🤖 Agent Model Assignments"),
        "width": TABLE_WIDTH,
        "columns": [
            {"name": h("Model"), "style": "yellow", "width": 28},
            {"name": h("Source"), "style": "dim", "width": 26},
            {"name": "#", "style": "white", "width": 4},
            {"name": h("Examples"), "style": "cyan", "overflow": "fold"},
        ],
        "rows": [],
    }

    model_groups: dict[tuple[str, str], list[str]] = defaultdict(list)

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

            model_groups[(model_id, model_source)].append(agent_cfg.display_name)
        except Exception as exc:
            # Skip agents with config errors
            logger.debug("Skipping model row for %s: %s", agent_name, exc)

    for (model_id, model_source), display_names in sorted(
        model_groups.items(),
        key=lambda item: (-len(item[1]), item[0][0], item[0][1]),
    ):
        sorted_names = sorted(display_names)
        agent_table_data["rows"].append(
            [
                _truncate_middle(model_id, 28),
                _truncate_middle(_source_label(model_source), 26),
                str(len(sorted_names)),
                _summarize_names(sorted_names),
            ]
        )

    if protocol_mode:
        agent_preview_rows = agent_table_data["rows"][:4]
        if agent_preview_rows:
            preview_table = dict(agent_table_data)
            preview_table["rows"] = agent_preview_rows
            output.print_table(preview_table)
            if len(agent_table_data["rows"]) > len(agent_preview_rows):
                _render_blocks(
                    output,
                    [
                        hint_block(
                            h("... and {count} more model groups").format(
                                count=len(agent_table_data["rows"])
                                - len(agent_preview_rows)
                            )
                        )
                    ],
                )
        else:
            _print_table_or_empty(
                output,
                agent_table_data,
                h(
                    "No agent-specific model resolution data is available for the current configuration."
                ),
            )
    else:
        _print_table_or_empty(
            output,
            agent_table_data,
            h(
                "No agent-specific model resolution data is available for the current configuration."
            ),
        )

    workspace_path = Path(client.workspace_path)
    (
        composition_source,
        composition_detail,
        _display_agents,
        installed_count,
        total_count,
    ) = _resolve_agent_composition_summary(workspace_path)

    summary_rows = [
        [h("Tier"), current_tier.capitalize()],
        [h("Config source"), composition_source],
        [h("Configured agents"), str(total_count)],
        [h("Installed agents"), f"{installed_count}/{total_count}"],
        [h("Distinct model configs"), str(len(model_groups))],
    ]
    if composition_detail:
        summary_rows.append([h("Config detail"), composition_detail])

    _render_blocks(
        output,
        [
            kv_block(summary_rows, title=h("Agent Configuration Summary")),
            hint_block(
                h(
                    "Use /status for the full agent roster and /config model for provider model catalogs."
                )
            ),
        ],
    )

    if protocol_mode:
        _render_blocks(
            output,
            [
                hint_block(
                    h(
                        "Use /config provider and /config model for focused follow-up commands."
                    )
                )
            ],
        )
        return f"Displayed configuration (provider: {provider}, profile: {profile})"

    # ========================================================================
    # Usage hints
    # ========================================================================
    _print_config_command_reference(output, TABLE_WIDTH)

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

    rows = []

    # Dynamically fetch all registered providers from ProviderRegistry
    all_providers = ProviderRegistry.get_all()

    for provider_obj in all_providers:
        provider_name = provider_obj.name
        configured = (
            h("Configured")
            if provider_name in available_providers
            else h("Not configured")
        )
        active = "●" if provider_name == current_provider else ""
        rows.append([provider_obj.display_name, configured, active])

    provider_names = ", ".join([p.name for p in all_providers])

    blocks: list[OutputBlock] = [
        table_block(
            title=h("LLM Providers"),
            columns=[
                {"name": h("Provider")},
                {"name": h("Status")},
                {"name": h("Active")},
            ],
            rows=rows,
        ),
        list_block(
            [
                h("/config provider <name> - Switch to specified provider (runtime)"),
                h("/config provider <name> --save - Switch and persist to workspace"),
            ],
            title=h("Usage"),
        ),
        hint_block(h("Available providers: {names}").format(names=provider_names)),
    ]
    if current_provider:
        blocks.append(
            hint_block(h("Current provider: {name}").format(name=current_provider))
        )
    _render_blocks(output, blocks)

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
            h(
                "[yellow]Switching to {name} provider and saving to workspace...[/yellow]"
            ).format(name=new_provider)
        )
    else:
        output.print(
            h("[yellow]Switching to {name} provider (runtime only)...[/yellow]").format(
                name=new_provider
            )
        )

    # Switch runtime first
    result = client.switch_provider(new_provider)

    if not result["success"]:
        error_msg = result.get("error", h("Unknown error"))
        hint = result.get("hint", "")
        output.print(f"[red]✗ {error_msg}[/red]")
        if hint:
            output.print(f"[dim]{hint}[/dim]")
        return None

    # Success message
    output.print(
        h("[green]✓ Successfully switched to {name} provider[/green]").format(
            name=result["provider"]
        )
    )

    # If not saving, show hint about persisting
    if not save:
        output.print(
            h(
                "[dim]💡 Use [white]/config provider {name} --save[/white] to persist this change[/dim]"
            ).format(name=new_provider)
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
            h("[green]✓ Saved to workspace config: {path}[/green]").format(
                path=workspace_path / "provider_config.json"
            )
        )
        output.print(
            h(
                "[dim]This provider will be used for all future sessions in this workspace.[/dim]"
            )
        )

        return f"Switched to {result['provider']} provider and saved to workspace"

    except Exception as e:
        output.print(
            h("[red]✗ Failed to save workspace config: {error}[/red]").format(
                error=str(e)
            )
        )
        output.print(h("[dim]Check workspace directory permissions[/dim]"))
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
    from lobster.core.config_resolver import ConfigResolver

    workspace_path = Path(client.workspace_path)
    resolver = ConfigResolver(workspace_path=workspace_path)
    table_width = 100

    try:
        current_provider, provider_source = resolver.resolve_provider(
            runtime_override=client.provider_override
        )
        provider_obj = get_provider(current_provider)
        if not provider_obj:
            _render_blocks(
                output,
                [
                    alert_block(
                        h("Provider '{name}' not registered").format(
                            name=current_provider
                        ),
                        level="error",
                    )
                ],
            )
            return None

        provider_display_name = getattr(
            provider_obj, "display_name", current_provider.capitalize()
        )
        current_model, model_source = resolver.resolve_model(
            runtime_override=getattr(client, "model_override", None),
            provider=current_provider,
        )
        if not current_model:
            current_model = provider_obj.get_default_model()
            model_source = f"provider default ({current_provider})"

        blocks: list[OutputBlock] = [
            table_block(
                title=h("Active Model Selection"),
                width=table_width,
                columns=[
                    {"name": h("Setting"), "style": "cyan", "width": 16},
                    {"name": h("Value"), "style": "white", "width": 36},
                    {"name": h("Source"), "style": "yellow", "overflow": "fold"},
                ],
                rows=[
                    [h("Provider"), current_provider, provider_source],
                    [
                        h("Current model"),
                        current_model or h("Provider default"),
                        model_source,
                    ],
                ],
            )
        ]

        # For Ollama, check if server is available
        if current_provider == "ollama":
            if not provider_obj.is_available():
                blocks.extend(
                    [
                        alert_block(h("Ollama server not accessible"), level="error"),
                        hint_block(h("Make sure Ollama is running: 'ollama serve'")),
                    ]
                )
                _render_blocks(output, blocks)
                return None

        models = provider_obj.list_models()

        from lobster.config.constants import PROVIDER_ICONS

        icon = PROVIDER_ICONS.get(current_provider, "🤖")
        title = h("{icon} Available {name} Models").format(
            icon=icon, name=provider_display_name
        )

        model_rows = []

        if not models:
            empty_message = h("No models are currently available for {name}.").format(
                name=provider_display_name
            )
            if current_provider == "ollama":
                empty_message = h(
                    "No Ollama models are installed for the active provider."
                )
            blocks.extend(
                [
                    section_block(title=title),
                    hint_block(empty_message),
                ]
            )
            if current_provider == "ollama":
                blocks.append(
                    hint_block(
                        h("Install one first, for example: ollama pull llama3.2")
                    )
                )
            blocks.append(
                table_block(
                    title=h("Model Commands"),
                    width=table_width,
                    columns=[
                        {
                            "name": h("Command"),
                            "style": "white",
                            "width": 34,
                            "overflow": "fold",
                        },
                        {
                            "name": h("Purpose"),
                            "style": "dim",
                            "width": 62,
                            "overflow": "fold",
                        },
                    ],
                    rows=[
                        [
                            "/config model <name>",
                            h("Switch the active model for the current session only."),
                        ],
                        [
                            "/config model <name> --save",
                            h(
                                "Switch the active model and persist it to this workspace."
                            ),
                        ],
                        [
                            "/config provider",
                            h(
                                "List providers and confirm which model catalog is active."
                            ),
                        ],
                        [
                            "/config provider <name>",
                            h(
                                "Change providers first when you need a different model catalog."
                            ),
                        ],
                    ],
                )
            )
            _render_blocks(output, blocks)
            return None

        for model in models:
            status_parts = []
            if model.name == current_model:
                status_parts.append(h("Current"))
            if model.is_default:
                status_parts.append(h("Default"))
            model_rows.append(
                [
                    model.name,
                    model.display_name,
                    ", ".join(status_parts) if status_parts else "-",
                    model.description,
                ]
            )

        blocks.extend(
            [
                table_block(
                    title=title,
                    width=table_width,
                    columns=[
                        {"name": h("Model"), "style": "yellow", "width": 24},
                        {"name": h("Display Name"), "style": "cyan", "width": 26},
                        {"name": h("Status"), "style": "green", "width": 18},
                        {
                            "name": h("Description"),
                            "style": "white",
                            "overflow": "fold",
                        },
                    ],
                    rows=model_rows,
                ),
                table_block(
                    title=h("Model Commands"),
                    width=table_width,
                    columns=[
                        {
                            "name": h("Command"),
                            "style": "white",
                            "width": 34,
                            "overflow": "fold",
                        },
                        {
                            "name": h("Purpose"),
                            "style": "dim",
                            "width": 62,
                            "overflow": "fold",
                        },
                    ],
                    rows=[
                        [
                            "/config model <name>",
                            h("Switch the active model for the current session only."),
                        ],
                        [
                            "/config model <name> --save",
                            h(
                                "Switch the active model and persist it to this workspace."
                            ),
                        ],
                        [
                            "/config provider",
                            h(
                                "List providers and confirm which model catalog is active."
                            ),
                        ],
                        [
                            "/config provider <name>",
                            h(
                                "Change providers first when you need a different model catalog."
                            ),
                        ],
                    ],
                ),
            ]
        )
        _render_blocks(output, blocks)

        return f"Listed models for {current_provider} provider"

    except Exception as e:
        _render_blocks(
            output,
            [
                alert_block(
                    h("Failed to list models: {error}").format(error=str(e)),
                    level="error",
                ),
                hint_block(
                    h("Use /config provider to confirm the active backend first.")
                ),
            ],
        )
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
            output.print(
                h("[red]✗ Provider '{name}' not registered[/red]").format(
                    name=current_provider
                )
            )
            return None

        # For Ollama, check server availability
        if current_provider == "ollama":
            if not provider_obj.is_available():
                output.print(h("[red]✗ Ollama server not accessible[/red]"))
                output.print(
                    h("[dim]Make sure Ollama is running: 'ollama serve'[/dim]")
                )
                return None

        # Validate model
        if not provider_obj.validate_model(model_name):
            model_names = provider_obj.get_model_names()
            available = ", ".join(model_names[:5])
            if len(model_names) > 5:
                available += ", ..."
            hint = h("Available models: {models}").format(models=available)
            if current_provider == "ollama":
                hint += h("\nInstall with: ollama pull {model}").format(
                    model=model_name
                )
            output.print(
                h("[red]✗ Model '{model}' not valid for {name}[/red]").format(
                    model=model_name, name=current_provider
                )
            )
            output.print(f"[dim]{hint}[/dim]")
            return None

        # Always persist to workspace config — env vars are not read by
        # ConfigResolver and would be lost on next session
        try:
            config = WorkspaceProviderConfig.load(workspace_path)
            config.set_model_for_provider(current_provider, model_name)
            config.save(workspace_path)
        except Exception as e:
            output.print(
                h("[red]✗ Failed to save model config: {error}[/red]").format(error=e)
            )
            return None

        # Update client's model override so the live TUI footer reflects the change
        if hasattr(client, "model_override"):
            client.model_override = model_name

        output.print(
            h("[green]✓ Switched to model: {model}[/green]").format(model=model_name)
        )
        output.print(h("[dim]Provider: {name}[/dim]").format(name=current_provider))

        if not save:
            output.print(
                h("[dim]Saved to workspace config ({name}_model)[/dim]").format(
                    name=current_provider
                )
            )
            return f"Switched to model {model_name}"

        # --save flag: already persisted above, just confirm
        output.print(
            h("[green]✓ Saved to workspace config ({name}_model)[/green]").format(
                name=current_provider
            )
        )
        output.print(
            h("[dim]Config file: {path}/provider_config.json[/dim]").format(
                path=workspace_path
            )
        )
        output.print(
            h(
                "\n[dim]This model will be used for {name} in this workspace[/dim]"
            ).format(name=current_provider)
        )

        return f"Switched to model {model_name} and saved to workspace"

    except Exception as e:
        output.print(
            h("[red]✗ Failed to switch model: {error}[/red]").format(error=str(e))
        )
        output.print(h("[dim]Check provider configuration[/dim]"))
        return None


# ============================================================================
# Config subcommand impl functions (extracted from cli.py Plan 08-02)
# ============================================================================


def config_test_impl(output_json: bool = False):
    """Test API connectivity and validate configuration (extracted from cli.py)."""
    import json as json_module
    import os
    from pathlib import Path

    import typer
    from dotenv import load_dotenv
    from rich import box
    from rich.panel import Panel
    from rich.table import Table

    from lobster.ui import LobsterTheme
    from lobster.ui.console_manager import get_console_manager

    console_manager = get_console_manager()
    console = console_manager.console

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

    # Load credentials from multiple sources (local .env, global config, env vars)
    from lobster.config.global_config import (
        get_global_credentials_path,
        global_credentials_exist,
    )

    env_file = Path.cwd() / ".env"
    credential_source = None

    if env_file.exists():
        load_dotenv(env_file)
        credential_source = "local .env"
        test_results["env_file"] = str(env_file)
    elif global_credentials_exist():
        global_creds = get_global_credentials_path()
        load_dotenv(global_creds)
        credential_source = f"global credentials ({global_creds})"
        test_results["env_file"] = str(global_creds)
    else:
        credential_source = "environment variables"
        test_results["env_file"] = None
        # Don't exit — env vars or workspace config may provide credentials

    test_results["credential_source"] = credential_source

    if not output_json:
        console.print()
        console.print(
            Panel.fit(
                f"[bold {LobsterTheme.PRIMARY_ORANGE}]{h('🔍 Configuration Test')}[/bold {LobsterTheme.PRIMARY_ORANGE}]",
                border_style=LobsterTheme.PRIMARY_ORANGE,
                padding=(0, 2),
            )
        )
        console.print()
        if credential_source == "environment variables":
            console.print(
                h(
                    "[yellow]No .env or global credentials found — testing environment variables[/yellow]"
                )
            )
        else:
            console.print(
                h("[green]✅ Credentials source:[/green] {source}").format(
                    source=credential_source
                )
            )
        console.print()

    # Test LLM Provider
    log(h("Testing LLM Provider..."), "bold")
    try:
        from lobster.config.llm_factory import LLMFactory

        workspace_path = Path.cwd() / ".lobster_workspace"
        provider = LLMFactory.get_current_provider(workspace_path=workspace_path)
        if provider is None:
            test_results["checks"]["llm_provider"]["message"] = "No API keys found"
            test_results["checks"]["llm_provider"][
                "hint"
            ] = "Set ANTHROPIC_API_KEY, AWS_BEDROCK_ACCESS_KEY, or run Ollama"
            log(h("❌ No LLM provider configured"), "red")
        else:
            # provider is a string, not an enum
            test_results["checks"]["llm_provider"]["provider"] = provider
            log(h("  Detected provider: {name}").format(name=provider), "yellow")

            # Provider-specific pre-checks for Ollama
            if provider == "ollama":
                ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                log(
                    h("  Checking Ollama server at {url}...").format(url=ollama_url),
                    "yellow",
                )
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
                        log(h("❌ Ollama: No models installed"), "red")
                        provider = None
                    else:
                        log(
                            h("  Ollama server: Running ({count} models)").format(
                                count=len(models)
                            ),
                            "green",
                        )
                except requests.exceptions.ConnectionError:
                    test_results["checks"]["llm_provider"][
                        "message"
                    ] = "Ollama server not running"
                    test_results["checks"]["llm_provider"][
                        "hint"
                    ] = "Start Ollama: ollama serve"
                    log(h("❌ Ollama server not accessible"), "red")
                    provider = None
                except Exception as e:
                    test_results["checks"]["llm_provider"][
                        "message"
                    ] = f"Ollama error: {str(e)[:60]}"
                    log(h("❌ Ollama check failed: {error}").format(error=e), "red")
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
                    log(h("  Testing API connectivity..."), "yellow")
                    test_llm.invoke("Reply with just 'ok'")

                    test_results["checks"]["llm_provider"]["status"] = "pass"
                    test_results["checks"]["llm_provider"]["message"] = "Connected"
                    log(h("✅ {name} API: Connected").format(name=provider), "green")
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
                    log(
                        h("❌ {name} API: {error}").format(
                            name=provider, error=error_msg[:60]
                        ),
                        "red",
                    )
    except Exception as e:
        test_results["checks"]["llm_provider"]["message"] = str(e)[:100]
        log(h("❌ LLM Provider test failed: {error}").format(error=e), "red")

    if not output_json:
        console.print()

    # Test NCBI API
    log(h("Testing NCBI API..."), "bold")
    ncbi_key = os.environ.get("NCBI_API_KEY")
    test_results["checks"]["ncbi_api"]["has_key"] = bool(ncbi_key)

    if ncbi_key or os.environ.get("NCBI_EMAIL"):
        log(h("  Testing NCBI E-utilities..."), "yellow")
        status, message = _check_ncbi_eutils(ncbi_key)
        test_results["checks"]["ncbi_api"]["status"] = status
        test_results["checks"]["ncbi_api"]["message"] = message
        if status == "pass":
            log(
                h("✅ NCBI API: Connected {suffix}").format(
                    suffix=h("(with API key)") if ncbi_key else ""
                ),
                "green",
            )
        else:
            log(h("❌ NCBI API: {message}").format(message=message), "red")
    else:
        test_results["checks"]["ncbi_api"]["status"] = "skip"
        test_results["checks"]["ncbi_api"]["message"] = "Not configured (optional)"
        log(h("⊘ NCBI API: Not configured (optional)"), "dim")

    if not output_json:
        console.print()

    # Test Workspace
    log(h("Testing Workspace..."), "bold")
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
            log(h("✅ Workspace: Writable"), "green")
            log(h("  Path: {path}").format(path=workspace_path), "dim")
        except PermissionError:
            test_results["checks"]["workspace"]["message"] = "Permission denied"
            log(h("❌ Workspace: Permission denied"), "red")
        except Exception as e:
            test_results["checks"]["workspace"]["message"] = str(e)[:60]
            log(h("❌ Workspace: {error}").format(error=e), "red")
    except Exception as e:
        test_results["checks"]["workspace"]["message"] = str(e)[:60]
        log(h("❌ Workspace setup failed: {error}").format(error=e), "red")

    # Determine overall validity
    test_results["valid"] = test_results["checks"]["llm_provider"]["status"] == "pass"

    # Output
    if output_json:
        print(json_module.dumps(test_results, indent=2))
    else:
        console.print()

        # Summary table
        table = Table(title=h("Configuration Test Summary"), box=box.ROUNDED)
        table.add_column(h("Service"), style="cyan", no_wrap=True)
        table.add_column(h("Status"), style="bold", no_wrap=True)
        table.add_column(h("Details"), style="dim")

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
                    h("[bold green]✅ Configuration Valid[/bold green]\n\n")
                    + h("All required services are accessible.\n")
                    + h(
                        "You can now run: [bold {color}]lobster chat[/bold {color}]"
                    ).format(color=LobsterTheme.PRIMARY_ORANGE),
                    border_style="green",
                    padding=(1, 2),
                )
            )
        else:
            console.print(
                Panel.fit(
                    h("[bold red]❌ Configuration Issues Detected[/bold red]\n\n")
                    + h("Please check your API keys.\n")
                    + h(
                        "Run: [bold {color}]lobster init[/bold {color}] to reconfigure"
                    ).format(color=LobsterTheme.PRIMARY_ORANGE),
                    border_style="red",
                    padding=(1, 2),
                )
            )

    if not test_results["valid"]:
        raise typer.Exit(1)


def list_models_impl():
    """List all available models from registered providers."""
    from rich import box
    from rich.table import Table

    from lobster.config.providers.registry import ProviderRegistry
    from lobster.ui.console_manager import get_console_manager

    console = get_console_manager().console

    console.print(h("\n[cyan]Available Models[/cyan]"))

    table = Table(
        box=box.ROUNDED,
        border_style="cyan",
        title=h("Available Models (by provider)"),
        title_style="bold cyan",
    )

    table.add_column(h("Provider"), style="bold white")
    table.add_column(h("Model ID"), style="yellow")
    table.add_column(h("Display Name"), style="white")
    table.add_column(h("Input $/M"), style="cyan", justify="right")
    table.add_column(h("Output $/M"), style="cyan", justify="right")

    all_models = ProviderRegistry.get_all_models_with_pricing()
    cloud_provider = ProviderRegistry.get("ollama-cloud")
    if cloud_provider:
        for model in cloud_provider.list_models():
            all_models[model.name] = {
                "provider": "ollama-cloud",
                "display_name": model.display_name,
                "input_per_million": model.input_cost_per_million,
                "output_per_million": model.output_cost_per_million,
            }
    for model_id, info in sorted(all_models.items()):
        table.add_row(
            info.get("provider", ""),
            model_id,
            info.get("display_name", ""),
            (
                f"${info['input_per_million']:.2f}"
                if info.get("input_per_million") is not None
                else h("unknown")
            ),
            (
                f"${info['output_per_million']:.2f}"
                if info.get("output_per_million") is not None
                else h("unknown")
            ),
        )

    console.print(table)


def list_profiles_impl():
    """List available configuration profiles."""
    from lobster.config.agent_defaults import get_current_profile
    from lobster.ui.console_manager import get_console_manager

    console = get_console_manager().console

    profiles = ["development", "production", "performance", "max"]
    current = get_current_profile()

    console.print(h("\n[cyan]Available Profiles[/cyan]"))
    for name in profiles:
        marker = f" [green]{h('(active)')}[/green]" if name == current else ""
        console.print(f"   {name}{marker}")
    console.print()
    console.print(
        h("[dim]Switch profile: lobster config provider --profile <name> --save[/dim]")
    )


def show_config_impl(workspace=None, show_all: bool = False):
    """Show current runtime configuration (extracted from cli.py)."""

    from rich.panel import Panel

    from lobster.core.workspace import resolve_workspace
    from lobster.ui import LobsterTheme
    from lobster.ui.console_manager import get_console_manager

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
    console.print(h("[bold cyan]📍 Environment[/bold cyan]"))
    console.print(
        h("   Workspace: [yellow]{path}[/yellow]").format(path=workspace_path)
    )
    console.print(
        h("   License Tier: [green]{tier}[/green]").format(
            tier=get_tier_display_name(current_tier)
        )
    )
    console.print()

    # Section 2: Provider Configuration
    console.print(h("[bold cyan]🔌 Provider Configuration[/bold cyan]"))

    try:
        provider_name, provider_source = resolver.resolve_provider()
        provider_obj = get_provider(provider_name)

        console.print(
            h("   Active Provider: [green]{name}[/green]").format(
                name=provider_obj.display_name if provider_obj else provider_name
            )
        )
        console.print(
            h("   Source: [dim]{source}[/dim]").format(source=provider_source)
        )

        if provider_obj:
            # Show provider-specific information
            default_model = provider_obj.get_default_model()
            console.print(
                h("   Default Model: [yellow]{model}[/yellow]").format(
                    model=default_model
                )
            )

            # Show available models for this provider
            models = provider_obj.list_models()
            if models:
                console.print(
                    h("   Available Models: [dim]{count} model(s)[/dim]").format(
                        count=len(models)
                    )
                )

    except ConfigurationError as e:
        console.print(h("   [red]✗ No provider configured[/red]"))
        console.print(h("   [dim]Error: {error}[/dim]").format(error=str(e)))
        console.print(
            h("   [yellow]💡 Run 'lobster init' to configure a provider[/yellow]")
        )

    console.print()

    # Section 3: Profile Configuration
    console.print(h("[bold cyan]⚙️  Profile Configuration[/bold cyan]"))

    try:
        profile, profile_source = resolver.resolve_profile()
        console.print(
            h("   Active Profile: [green]{name}[/green]").format(name=profile)
        )
        console.print(h("   Source: [dim]{source}[/dim]").format(source=profile_source))
    except Exception:
        console.print(
            h("   Active Profile: [yellow]production[/yellow] [dim](default)[/dim]")
        )

    console.print()

    # Section 4: Configuration Files
    console.print(h("[bold cyan]📁 Configuration Files[/bold cyan]"))

    # Workspace config
    workspace_config_path = workspace_path / "provider_config.json"
    if workspace_config_path.exists():
        console.print(
            h("   [green]✓[/green] Workspace: {path}").format(
                path=workspace_config_path
            )
        )
        try:
            ws_config = WorkspaceProviderConfig.load(workspace_path)
            if ws_config.global_provider:
                console.print(
                    h("      Provider: [yellow]{name}[/yellow]").format(
                        name=ws_config.global_provider
                    )
                )
            if ws_config.profile:
                console.print(
                    h("      Profile: [yellow]{name}[/yellow]").format(
                        name=ws_config.profile
                    )
                )
        except Exception as e:
            console.print(
                h("      [dim]Could not read: {error}[/dim]").format(error=str(e))
            )
    else:
        console.print(
            h("   [dim]○ Workspace: {path} (not found)[/dim]").format(
                path=workspace_config_path
            )
        )

    # Global config
    global_config_path = GlobalProviderConfig.get_config_path()
    if global_config_path.exists():
        console.print(
            h("   [green]✓[/green] Global: {path}").format(path=global_config_path)
        )
        try:
            global_config = GlobalProviderConfig.load()
            if global_config.default_provider:
                console.print(
                    h("      Provider: [yellow]{name}[/yellow]").format(
                        name=global_config.default_provider
                    )
                )
            if global_config.default_profile:
                console.print(
                    h("      Profile: [yellow]{name}[/yellow]").format(
                        name=global_config.default_profile
                    )
                )
        except Exception as e:
            console.print(
                h("      [dim]Could not read: {error}[/dim]").format(error=str(e))
            )
    else:
        console.print(
            h("   [dim]○ Global: {path} (not found)[/dim]").format(
                path=global_config_path
            )
        )

    console.print()

    # Section 5: Per-Agent Configuration
    console.print(h("[bold cyan]🤖 Agent Configuration[/bold cyan]"))
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
                h(
                    "      Model: [yellow]{model}[/yellow] [dim](from {source})[/dim]"
                ).format(model=model_id, source=model_source)
            )
            console.print(
                h("      Temperature: [cyan]{value}[/cyan]").format(
                    value=model_params.get("temperature", 1.0)
                )
            )

            # Show thinking configuration if present
            additional_fields = model_params.get("additional_model_request_fields", {})
            if "thinking" in additional_fields:
                thinking = additional_fields["thinking"]
                budget = thinking.get("budget_tokens", "unknown")
                console.print(
                    h(
                        "      [dim]🧠 Thinking: Enabled (Budget: {budget} tokens)[/dim]"
                    ).format(budget=budget)
                )

            console.print()

        except Exception as e:
            console.print(
                h(
                    "   [bold white]{name}[/bold white]: [red]Error - {error}[/red]"
                ).format(name=agent_cfg.display_name, error=str(e))
            )
            console.print()

    # Summary
    if filtered_count > 0 and not show_all:
        console.print(f"[dim]{'─' * 60}[/dim]")
        console.print(
            h(
                "[cyan]📊 Summary:[/cyan] Showing {shown} agents available for [green]{tier}[/green] tier"
            ).format(shown=displayed_count, tier=current_tier)
        )
        console.print(
            h("   [dim]({count} premium agents hidden)[/dim]").format(
                count=filtered_count
            )
        )
        console.print(
            h("   [yellow]💡 Use '--show-all' to see all configured agents[/yellow]")
        )

    console.print()


def test_impl(profile=None, agent=None):
    """Test LLM provider connectivity and configuration (extracted from cli.py)."""
    import os

    import typer
    from rich.panel import Panel

    from lobster.ui import LobsterTheme
    from lobster.ui.console_manager import get_console_manager

    console = get_console_manager().console

    # Load .env file for provider detection
    from dotenv import load_dotenv

    load_dotenv()

    # If no profile specified, test current provider connectivity
    if not profile:
        console.print()
        console.print(
            Panel.fit(
                f"[bold {LobsterTheme.PRIMARY_ORANGE}]{h('🔍 Testing LLM Provider')}[/bold {LobsterTheme.PRIMARY_ORANGE}]",
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
                console.print(f"[red]❌ {t('light.config.no_provider')}[/red]")
                console.print(f"[yellow]💡 {t('light.config.run_init')}[/yellow]")
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
                console.print(f"[red]❌ {t('light.config.no_provider')}[/red]")
                console.print(f"[yellow]💡 {t('light.config.run_init')}[/yellow]")
                raise typer.Exit(1)

            # provider is a string, not an enum
            console.print(
                h("[cyan]Detected provider:[/cyan] {name}").format(name=provider)
            )
            console.print()

            # Test provider connectivity
            console.print(h("[yellow]Testing API connectivity...[/yellow]"))

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
                        h("[bold green]✅ Provider Test Successful[/bold green]\n\n")
                        + h("Provider: [cyan]{name}[/cyan]\n").format(name=provider)
                        + h("Status: [green]Connected[/green]\n\n")
                        + h("Your {name} provider is working correctly.\n").format(
                            name=provider
                        )
                        + h(
                            "Run [bold {color}]lobster chat[/bold {color}] to start analyzing!"
                        ).format(color=LobsterTheme.PRIMARY_ORANGE),
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
                        h("[bold red]❌ Provider Test Failed[/bold red]\n\n")
                        + h("Provider: [cyan]{name}[/cyan]\n").format(name=provider)
                        + h("Error: [red]{error}[/red]\n\n").format(
                            error=error_msg[:100]
                        )
                        + h("[yellow]💡 Check your API keys in .env file[/yellow]\n")
                        + h(
                            "Run [bold {color}]lobster init --force[/bold {color}] to reconfigure"
                        ).format(color=LobsterTheme.PRIMARY_ORANGE),
                        border_style="red",
                        padding=(1, 2),
                    )
                )
                raise typer.Exit(1)

        except Exception as e:
            console.print(h("[red]❌ Error: {error}[/red]").format(error=str(e)))
            raise typer.Exit(1)

    # Profile-based testing: test agent LLM params resolution
    try:
        from lobster.config.agent_defaults import get_agent_params
        from lobster.core.component_registry import component_registry

        component_registry.load_components()
        agents = component_registry.list_agents()

        if agent:
            if agent in agents:
                params = get_agent_params(agent)
                console.print(
                    h("\n[green]Agent '{name}' configuration valid[/green]").format(
                        name=agent
                    )
                )
                console.print(
                    h("   Temperature: {value}").format(
                        value=params.get("temperature", 1.0)
                    )
                )
            else:
                console.print(
                    h("\n[red]Agent '{name}' not installed[/red]").format(name=agent)
                )
                return False
        else:
            console.print(
                h("\n[yellow]Testing {count} installed agents...[/yellow]").format(
                    count=len(agents)
                )
            )
            all_valid = True

            for agent_name in sorted(agents.keys()):
                try:
                    params = get_agent_params(agent_name)
                    console.print(
                        h("   [green]{name}: temp={value}[/green]").format(
                            name=agent_name, value=params.get("temperature", 1.0)
                        )
                    )
                except Exception as e:
                    console.print(
                        h("   [red]{name}: {error}[/red]").format(
                            name=agent_name, error=str(e)
                        )
                    )
                    all_valid = False

            if all_valid:
                console.print(
                    h(
                        "\n[green]All {count} agents configured correctly[/green]"
                    ).format(count=len(agents))
                )
            else:
                console.print(
                    h("\n[yellow]Some agents have configuration issues[/yellow]")
                )

        return True

    except Exception as e:
        console.print(
            h("\n[red]Error testing configuration: {error}[/red]").format(error=str(e))
        )
        return False


def create_custom_impl():
    """Interactive per-agent model configuration. Use 'lobster config models' instead."""
    from lobster.ui.console_manager import get_console_manager

    console = get_console_manager().console
    console.print(
        h(
            "\n[yellow]This command has been replaced by 'lobster config models'.[/yellow]"
        )
    )
    console.print(h("[dim]Run: lobster config models[/dim]"))


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

# Model selection: use --model, or the workspace/global provider config
# (lobster config). Per-agent models are set in the workspace config.
# For AWS Bedrock, this sets the default model when nothing else chooses one:
# LOBSTER_BEDROCK_MODEL_ID=global.anthropic.claude-sonnet-5-5

# Per-agent temperature overrides
# LOBSTER_SUPERVISOR_TEMPERATURE=0.5
# LOBSTER_TRANSCRIPTOMICS_EXPERT_TEMPERATURE=0.7
# LOBSTER_METHOD_AGENT_TEMPERATURE=0.3

# =============================================================================
# APPLICATION SETTINGS
# =============================================================================

# Server configuration
PORT=8501
# Localhost only; use 0.0.0.0 inside a container
HOST=127.0.0.1
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

    console.print(h("[green]✅ Environment template saved to: .env.template[/green]"))
    console.print(
        h("[yellow]Copy this file to .env and configure your API keys[/yellow]")
    )


def config_models_impl(workspace=None):
    """Interactive per-agent model configuration using prompt_toolkit dialogs."""

    from prompt_toolkit.shortcuts import radiolist_dialog

    from lobster.config.providers import get_provider
    from lobster.config.workspace_config import WorkspaceProviderConfig
    from lobster.core.component_registry import component_registry
    from lobster.core.config_resolver import ConfigResolver, ConfigurationError
    from lobster.core.workspace import resolve_workspace
    from lobster.ui.console_manager import get_console_manager

    console = get_console_manager().console

    # Resolve workspace
    workspace_path = resolve_workspace(workspace, create=False)

    # Load current config
    ws_config = WorkspaceProviderConfig.load(workspace_path)

    # Resolve current provider
    try:
        resolver = ConfigResolver.get_instance(workspace_path)
        provider_name, _ = resolver.resolve_provider()
        provider_obj = get_provider(provider_name)
    except (ConfigurationError, Exception):
        console.print(f"[red]{t('light.config.no_provider_first')}[/red]")
        return

    if not provider_obj:
        console.print(
            f"[red]{t('light.config.provider_not_found', name=provider_name)}[/red]"
        )
        return

    # Get available models for current provider
    available_models = provider_obj.list_models()
    if not available_models:
        console.print(
            h("[yellow]No models available for provider '{name}'.[/yellow]").format(
                name=provider_name
            )
        )
        return

    # Discover all installed agents
    component_registry.load_components()
    agents = component_registry.list_agents()

    # Build agent choices with current model info
    agent_choices = []
    for agent_name in sorted(agents.keys()):
        current_override = ws_config.per_agent_models.get(agent_name)
        label = agent_name
        if current_override:
            label += f"  [{current_override}]"
        else:
            label += f"  {h('(default)')}"
        agent_choices.append((agent_name, label))

    if not agent_choices:
        console.print(h("[yellow]No agents installed.[/yellow]"))
        return

    # Step 1: Select agent
    selected_agent = radiolist_dialog(
        title=h("Lobster AI — Per-Agent Model Configuration"),
        text=h("Select an agent to configure (arrow keys + Enter):"),
        values=agent_choices,
    ).run()

    if selected_agent is None:
        console.print(h("[dim]Cancelled.[/dim]"))
        return

    # Step 2: Select model
    model_choices = []
    current_override = ws_config.per_agent_models.get(selected_agent, "")

    # Add "use default" option first
    default_model = provider_obj.get_default_model()
    default_label = h("(use provider default: {model})").format(model=default_model)
    model_choices.append(("__default__", default_label))

    for model in available_models:
        cost_info = ""
        if model.input_cost_per_million and model.output_cost_per_million:
            cost_info = h("  (${input}/${output} per M)").format(
                input=f"{model.input_cost_per_million:.1f}",
                output=f"{model.output_cost_per_million:.1f}",
            )
        marker = " *" if model.name == current_override else ""
        label = f"{model.display_name}{cost_info}{marker}"
        model_choices.append((model.name, label))

    selected_model = radiolist_dialog(
        title=h("Select model for {name}").format(name=selected_agent),
        text=h("Current: {current}\nProvider: {provider}").format(
            current=current_override or h("(default)"), provider=provider_name
        ),
        values=model_choices,
    ).run()

    if selected_model is None:
        console.print(h("[dim]Cancelled.[/dim]"))
        return

    # Step 3: Apply and save
    if selected_model == "__default__":
        # Remove override
        if selected_agent in ws_config.per_agent_models:
            del ws_config.per_agent_models[selected_agent]
            ws_config.save(workspace_path)
            console.print(
                h(
                    "[green]Removed model override for {name} (now uses default: {model})[/green]"
                ).format(name=selected_agent, model=default_model)
            )
        else:
            console.print(
                h("[dim]{name} already uses the default model.[/dim]").format(
                    name=selected_agent
                )
            )
    else:
        ws_config.per_agent_models[selected_agent] = selected_model
        ws_config.save(workspace_path)
        console.print(
            h("[green]Set {name} → {model}[/green]").format(
                name=selected_agent, model=selected_model
            )
        )

    console.print(
        h("[dim]Saved to: {path}[/dim]").format(
            path=workspace_path / "provider_config.json"
        )
    )
