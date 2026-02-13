"""
Agent configuration resolver with 3-layer priority resolution.

This module provides resolution of enabled agents from multiple sources
with proper priority ordering: runtime CLI > workspace TOML > defaults.

Priority Resolution:
    1. Runtime overrides (CLI flags like --agents or --preset)
    2. Workspace TOML configuration (.lobster_workspace/config.toml)
    3. Default (all available agents from ComponentRegistry)

Usage:
    from lobster.config.agent_config_resolver import AgentConfigResolver

    # Create resolver for a workspace
    resolver = AgentConfigResolver(Path(".lobster_workspace"))

    # Resolve enabled agents with runtime override
    agents, source = resolver.resolve_enabled_agents(
        runtime_agents=["research_agent", "data_expert_agent"]
    )

    # Or use preset from CLI
    agents, source = resolver.resolve_enabled_agents(runtime_preset="scrna-basic")

    # Get agent-specific settings
    settings = resolver.get_agent_settings("transcriptomics_expert")

    # Get provider settings from TOML [provider] section (CONF-04)
    provider = resolver.get_provider_settings()
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class AgentConfigResolver:
    """
    Resolve agent configuration with 3-layer priority.

    This class implements the configuration resolution hierarchy:
    1. Runtime CLI overrides (highest priority)
    2. Workspace TOML configuration
    3. Default all-agents fallback (lowest priority)

    Attributes:
        workspace_path: Path to workspace directory
        _config: Lazily loaded WorkspaceAgentConfig
    """

    def __init__(self, workspace_path: Path):
        """Initialize resolver with workspace path.

        Args:
            workspace_path: Path to workspace directory containing config.toml
        """
        self.workspace_path = workspace_path
        self._config = None
        self._loaded = False

    def _load_config(self) -> None:
        """Lazy-load workspace configuration."""
        if self._loaded:
            return

        from lobster.config.workspace_agent_config import WorkspaceAgentConfig

        self._config = WorkspaceAgentConfig.load(self.workspace_path)
        self._loaded = True

    @property
    def config(self):
        """Get loaded workspace configuration."""
        self._load_config()
        return self._config

    def resolve_enabled_agents(
        self,
        runtime_agents: Optional[List[str]] = None,
        runtime_preset: Optional[str] = None,
    ) -> Tuple[List[str], str]:
        """
        Resolve enabled agents using 3-layer priority.

        Priority order:
        1. runtime_agents (CLI --agents flag) - highest
        2. runtime_preset (CLI --preset flag)
        3. workspace config enabled list
        4. workspace config preset
        5. default all-agents fallback - lowest

        Args:
            runtime_agents: Explicit agent list from CLI (e.g., --agents)
            runtime_preset: Preset name from CLI (e.g., --preset scrna-basic)

        Returns:
            Tuple of (agent_names, decision_source) where:
            - agent_names: List of resolved agent names
            - decision_source: Human-readable description of where config came from

        Example:
            >>> resolver = AgentConfigResolver(Path(".lobster_workspace"))
            >>> agents, source = resolver.resolve_enabled_agents(
            ...     runtime_agents=["research_agent"]
            ... )
            >>> print(f"Using {len(agents)} agents from {source}")
        """
        # Priority 1: Runtime agent list (highest priority)
        if runtime_agents:
            logger.debug(f"Using runtime agent override: {runtime_agents}")
            return runtime_agents, "runtime --agents flag"

        # Priority 2: Runtime preset
        if runtime_preset:
            from lobster.config.agent_presets import expand_preset

            expanded = expand_preset(runtime_preset)
            if expanded is not None:
                logger.debug(f"Using runtime preset '{runtime_preset}': {expanded}")
                return expanded, f"runtime --preset {runtime_preset}"
            else:
                logger.warning(
                    f"Unknown preset '{runtime_preset}', falling back to defaults"
                )

        # Load workspace config (lazy)
        self._load_config()

        # Priority 3: Workspace config enabled list
        if self._config and self._config.enabled_agents:
            # Validate and filter to installed agents only
            valid_agents, missing = self._config.validate_enabled_agents()
            if valid_agents:
                logger.debug(f"Using workspace config enabled agents: {valid_agents}")
                return valid_agents, "workspace config.toml [enabled]"

        # Priority 4: Workspace config preset
        if self._config and self._config.preset:
            from lobster.config.agent_presets import expand_preset

            expanded = expand_preset(self._config.preset)
            if expanded is not None:
                logger.debug(
                    f"Using workspace preset '{self._config.preset}': {expanded}"
                )
                return expanded, f"workspace config.toml [preset={self._config.preset}]"
            else:
                logger.warning(
                    f"Unknown workspace preset '{self._config.preset}', "
                    "falling back to defaults"
                )

        # Priority 5: Default - all available agents
        return self._get_default_agents()

    def _get_default_agents(self) -> Tuple[List[str], str]:
        """Get default agent list from ComponentRegistry.

        Returns:
            Tuple of (all_agent_names, "default (all available agents)")
        """
        from lobster.core.component_registry import component_registry

        all_agents = list(component_registry.list_agents().keys())
        logger.debug(f"Using default all available agents: {all_agents}")
        return all_agents, "default (all available agents)"

    def get_agent_settings(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        Get per-agent settings from workspace configuration.

        Args:
            agent_name: Name of the agent

        Returns:
            Dict of settings if configured, None otherwise.
            Settings may include: model, thinking_preset, temperature

        Example:
            >>> resolver = AgentConfigResolver(workspace)
            >>> settings = resolver.get_agent_settings("transcriptomics_expert")
            >>> if settings and settings.get("model"):
            ...     print(f"Using model: {settings['model']}")
        """
        self._load_config()

        if not self._config or not self._config.agent_settings:
            return None

        agent_settings = self._config.agent_settings.get(agent_name)
        if agent_settings is None:
            return None

        # Convert Pydantic model to dict, filtering None values
        return {k: v for k, v in agent_settings.model_dump().items() if v is not None}

    def get_provider_settings(self) -> Optional[Dict[str, Any]]:
        """
        Get provider settings from workspace TOML [provider] section.

        This exposes the CONF-04 provider settings, allowing workspace-level
        provider configuration as an alternative to environment variables.

        Returns:
            Dict of provider settings if [provider] section exists, None otherwise.
            May include: default, ollama_host, models

        Example:
            >>> resolver = AgentConfigResolver(workspace)
            >>> provider = resolver.get_provider_settings()
            >>> if provider and provider.get("default"):
            ...     print(f"Using provider: {provider['default']}")
        """
        self._load_config()

        if not self._config or not self._config.has_provider_settings():
            return None

        # Convert Pydantic model to dict, filtering None/empty values
        return {
            k: v
            for k, v in self._config.provider_settings.model_dump().items()
            if v is not None and v != {}
        }

    def has_toml_provider_settings(self) -> bool:
        """
        Check if workspace TOML has [provider] section configured.

        Returns:
            True if [provider] section exists in config.toml

        Example:
            >>> resolver = AgentConfigResolver(workspace)
            >>> if resolver.has_toml_provider_settings():
            ...     # Use TOML provider config instead of env vars
            ...     provider = resolver.get_provider_settings()
        """
        self._load_config()
        return self._config.has_provider_settings() if self._config else False


def resolve_agents_for_graph(
    workspace_path: Path,
    runtime_agents: Optional[List[str]] = None,
    runtime_preset: Optional[str] = None,
) -> Tuple[List[str], str]:
    """
    Convenience function for resolving agents for graph creation.

    This is the primary entry point for graph.py to resolve which agents
    to include in the LangGraph bioinformatics graph.

    Args:
        workspace_path: Path to workspace directory
        runtime_agents: Explicit agent list from CLI
        runtime_preset: Preset name from CLI

    Returns:
        Tuple of (agent_names, decision_source)

    Example:
        >>> agents, source = resolve_agents_for_graph(
        ...     Path(".lobster_workspace"),
        ...     runtime_agents=["research_agent"]
        ... )
        >>> print(f"Creating graph with agents: {agents} (from {source})")
    """
    resolver = AgentConfigResolver(workspace_path)
    return resolver.resolve_enabled_agents(runtime_agents, runtime_preset)
