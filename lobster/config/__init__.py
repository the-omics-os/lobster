"""
Configuration module.

This module handles application configuration settings, providing
centralized configuration management for the application.
"""

from lobster.config.workspace_agent_config import (
    AgentSettings,
    ProviderSettings,
    WorkspaceAgentConfig,
)
from lobster.config.agent_presets import (
    AGENT_PRESETS,
    expand_preset,
    get_preset_description,
    list_presets,
)
from lobster.config.agent_config_resolver import (
    AgentConfigResolver,
    resolve_agents_for_graph,
)

__all__ = [
    # Workspace agent configuration
    "AgentSettings",
    "ProviderSettings",
    "WorkspaceAgentConfig",
    # Agent presets
    "AGENT_PRESETS",
    "expand_preset",
    "get_preset_description",
    "list_presets",
    # Agent config resolution
    "AgentConfigResolver",
    "resolve_agents_for_graph",
]
