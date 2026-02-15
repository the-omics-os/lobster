"""
Workspace-scoped agent composition configuration with Pydantic validation.

This module provides configuration management for workspace-specific agent
composition and settings, loaded from a human-readable TOML file.

Storage Location: .lobster_workspace/config.toml

Example config.toml:
    config_version = "1.0"
    enabled = ["research_agent", "data_expert", "transcriptomics_expert"]
    preset = "production"

    [agent_settings.transcriptomics_expert]
    model = "claude-sonnet-4-20250514"
    thinking_preset = "high"
    temperature = 0.7

    # Optional provider settings (CONF-04)
    [provider]
    default = "anthropic"
    ollama_host = "http://localhost:11434"
    models = { anthropic = "claude-sonnet-4-20250514", ollama = "llama3:70b-instruct" }

Example:
    >>> from lobster.config.workspace_agent_config import WorkspaceAgentConfig
    >>> from pathlib import Path
    >>>
    >>> # Load config from workspace
    >>> config = WorkspaceAgentConfig.load(Path(".lobster_workspace"))
    >>>
    >>> # Check enabled agents
    >>> if "transcriptomics_expert" in config.enabled_agents:
    ...     print("Transcriptomics agent enabled")
    >>>
    >>> # Validate and save configuration
    >>> valid, missing = config.validate_enabled_agents()
    >>> config.save(Path(".lobster_workspace"))
"""

import logging
import tomllib
from pathlib import Path
from typing import List, Optional, Tuple

import tomli_w
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Configuration file name
CONFIG_FILE_NAME = "config.toml"


class AgentSettings(BaseModel):
    """
    Per-agent settings configurable in TOML.

    These settings allow fine-tuning individual agent behavior
    without modifying code.

    Attributes:
        model: Model override for this agent (e.g., "claude-sonnet-4-20250514")
        thinking_preset: Thinking preset override (e.g., "high", "medium", "low")
        temperature: Temperature override for model responses

    Example:
        >>> settings = AgentSettings(
        ...     model="claude-sonnet-4-20250514",
        ...     thinking_preset="high",
        ...     temperature=0.7
        ... )
    """

    model: Optional[str] = Field(
        None, description="Model override (e.g., 'claude-sonnet-4-20250514')"
    )

    thinking_preset: Optional[str] = Field(
        None, description="Thinking preset override (e.g., 'high', 'medium', 'low')"
    )

    temperature: Optional[float] = Field(
        None, ge=0.0, le=2.0, description="Temperature override (0.0-2.0)"
    )


class ProviderSettings(BaseModel):
    """
    Provider settings configurable in TOML [provider] section (CONF-04).

    These settings allow workspace-level provider configuration
    as an alternative to environment variables.

    Attributes:
        default: Default LLM provider (e.g., "anthropic", "ollama", "bedrock")
        ollama_host: Ollama server URL
        models: Per-provider model mappings

    Example:
        >>> settings = ProviderSettings(
        ...     default="anthropic",
        ...     ollama_host="http://localhost:11434",
        ...     models={"anthropic": "claude-sonnet-4-20250514"}
        ... )
    """

    default: Optional[str] = Field(
        None,
        description="Default LLM provider (anthropic | ollama | bedrock | gemini | azure)",
    )

    ollama_host: Optional[str] = Field(
        None, description="Ollama server URL (e.g., 'http://localhost:11434')"
    )

    models: dict[str, str] = Field(
        default_factory=dict,
        description="Per-provider model mappings (e.g., {'anthropic': 'claude-sonnet-4-20250514'})",
    )


class WorkspaceAgentConfig(BaseModel):
    """
    Workspace-scoped agent composition configuration.

    This configuration enables users to specify which agents are enabled,
    configure per-agent settings, and optionally specify provider settings
    via a human-readable TOML config file.

    Attributes:
        config_version: Schema version for future migrations
        enabled_agents: List of explicitly enabled agent names
        preset: Preset name for agent configuration
        agent_settings: Per-agent setting overrides
        provider_settings: Optional provider configuration (CONF-04)

    Example:
        >>> config = WorkspaceAgentConfig(
        ...     enabled_agents=["research_agent", "data_expert"],
        ...     preset="production",
        ...     agent_settings={"data_expert": AgentSettings(model="claude-sonnet-4-20250514")}
        ... )
    """

    config_version: str = Field(
        "1.0", description="Schema version for future migrations"
    )

    enabled_agents: list[str] = Field(
        default_factory=list,
        alias="enabled",
        description="List of explicitly enabled agent names",
    )

    preset: Optional[str] = Field(
        None, description="Preset name for agent configuration"
    )

    agent_settings: dict[str, AgentSettings] = Field(
        default_factory=dict, description="Per-agent setting overrides"
    )

    provider_settings: Optional[ProviderSettings] = Field(
        None, description="Optional provider configuration (CONF-04)"
    )

    model_config = {"populate_by_name": True}

    @classmethod
    def load(cls, workspace_path: Path) -> "WorkspaceAgentConfig":
        """
        Load configuration from workspace TOML file with graceful error handling.

        Handles:
        - Missing file: Returns default configuration
        - Corrupted TOML: Logs warning, returns defaults
        - Invalid schema: Logs validation errors, returns defaults

        Args:
            workspace_path: Path to workspace directory

        Returns:
            WorkspaceAgentConfig: Loaded or default configuration

        Example:
            >>> config = WorkspaceAgentConfig.load(Path(".lobster_workspace"))
            >>> if config.enabled_agents:
            ...     print(f"Enabled agents: {config.enabled_agents}")
        """
        config_path = workspace_path / CONFIG_FILE_NAME

        # File doesn't exist - return defaults
        if not config_path.exists():
            logger.debug(f"No agent config found at {config_path}, using defaults")
            return cls()

        try:
            # Parse TOML
            with open(config_path, "rb") as f:
                data = tomllib.load(f)

            # Extract provider settings if present
            provider_data = data.pop("provider", None)
            if provider_data:
                data["provider_settings"] = ProviderSettings(**provider_data)

            # Parse agent_settings into AgentSettings objects
            if "agent_settings" in data:
                data["agent_settings"] = {
                    name: AgentSettings(**settings)
                    for name, settings in data["agent_settings"].items()
                }

            config = cls(**data)
            logger.info(f"Loaded agent config from {config_path}")
            return config

        except tomllib.TOMLDecodeError as e:
            logger.warning(
                f"Corrupted TOML config at {config_path}: {e}. Using defaults."
            )
            return cls()

        except Exception as e:
            # Catch Pydantic validation errors and other exceptions
            logger.warning(
                f"Invalid config schema at {config_path}: {e}. Using defaults."
            )
            return cls()

    @classmethod
    def exists(cls, workspace_path: Path) -> bool:
        """
        Check if workspace configuration file exists.

        Args:
            workspace_path: Path to workspace directory

        Returns:
            bool: True if configuration file exists

        Example:
            >>> if WorkspaceAgentConfig.exists(Path(".lobster_workspace")):
            ...     config = WorkspaceAgentConfig.load(Path(".lobster_workspace"))
        """
        config_path = workspace_path / CONFIG_FILE_NAME
        return config_path.exists()

    def has_provider_settings(self) -> bool:
        """
        Check if provider settings are configured.

        Returns:
            bool: True if provider_settings is not None

        Example:
            >>> config = WorkspaceAgentConfig.load(workspace)
            >>> if config.has_provider_settings():
            ...     provider = config.provider_settings.default
        """
        return self.provider_settings is not None

    def validate_enabled_agents(self) -> Tuple[List[str], List[str]]:
        """
        Validate enabled agents against installed packages.

        Uses ComponentRegistry as the single source of truth for installed agents.
        Logs helpful warnings for uninstalled agents with installation instructions.

        Returns:
            Tuple of (valid_agents, missing_agents):
            - valid_agents: List of agent names that are installed
            - missing_agents: List of agent names that are NOT installed

        Example:
            >>> config = WorkspaceAgentConfig(enabled=["research_agent", "fake_agent"])
            >>> valid, missing = config.validate_enabled_agents()
            >>> print(f"Valid: {valid}, Missing: {missing}")
            Valid: ['research_agent'], Missing: ['fake_agent']
        """
        # Lazy import to avoid circular dependencies
        from lobster.core.component_registry import component_registry

        # Get all installed agents from ComponentRegistry (single source of truth)
        installed_agents = set(component_registry.list_agents().keys())

        valid_agents = []
        missing_agents = []

        for agent_name in self.enabled_agents:
            if agent_name in installed_agents:
                valid_agents.append(agent_name)
            else:
                missing_agents.append(agent_name)

        # Log helpful warnings for missing agents
        if missing_agents:
            for agent_name in missing_agents:
                # Provide helpful install suggestion
                package_hint = self._suggest_package_for_agent(agent_name)
                logger.warning(f"Agent '{agent_name}' not found. {package_hint}")

        return valid_agents, missing_agents

    def _suggest_package_for_agent(self, agent_name: str) -> str:
        """Generate installation suggestion for missing agent.

        Args:
            agent_name: Name of the missing agent

        Returns:
            Human-readable installation suggestion
        """
        # Known agent-to-package mappings for helpful suggestions
        known_packages = {
            "metadata_assistant": "lobster-premium",
            "proteomics_expert": "pip install lobster-ai[full] or lobster-proteomics",
            "machine_learning_expert": "pip install lobster-ai[ml]",
            "genomics_expert": "pip install lobster-ai[genomics] or lobster-genomics",
        }

        if agent_name in known_packages:
            return f"Install with: {known_packages[agent_name]}"

        # Check if it looks like a custom package agent
        if agent_name.startswith("custom_") or "_custom_" in agent_name:
            return "Install the corresponding lobster-custom-* package."

        return "Check available agents with: lobster agents list"

    def save(self, workspace_path: Path) -> Path:
        """
        Save configuration to workspace TOML file.

        Writes a valid TOML file that can be loaded back with load().
        Handles preset vs enabled list (mutually exclusive - preset takes priority).

        Args:
            workspace_path: Path to workspace directory

        Returns:
            Path: Path to saved config file

        Example:
            >>> config = WorkspaceAgentConfig(enabled=["research_agent"])
            >>> config_path = config.save(Path(".lobster_workspace"))
            >>> print(f"Config saved to {config_path}")
        """
        config_path = workspace_path / CONFIG_FILE_NAME

        # Build TOML-serializable dictionary
        data: dict = {
            "config_version": self.config_version,
        }

        # Add preset OR enabled list (mutually exclusive, preset takes priority)
        if self.preset:
            data["preset"] = self.preset
        elif self.enabled_agents:
            data["enabled"] = self.enabled_agents

        # Add agent_settings if present
        if self.agent_settings:
            data["agent_settings"] = {
                name: {k: v for k, v in settings.model_dump().items() if v is not None}
                for name, settings in self.agent_settings.items()
            }
            # Remove empty agent_settings entries
            data["agent_settings"] = {
                k: v for k, v in data["agent_settings"].items() if v
            }
            if not data["agent_settings"]:
                del data["agent_settings"]

        # Add provider settings if present (CONF-04)
        if self.provider_settings:
            provider_data = {
                k: v
                for k, v in self.provider_settings.model_dump().items()
                if v is not None and v != {}
            }
            if provider_data:
                data["provider"] = provider_data

        # Ensure workspace directory exists
        workspace_path.mkdir(parents=True, exist_ok=True)

        # Write TOML file
        with open(config_path, "wb") as f:
            tomli_w.dump(data, f)

        logger.info(f"Saved agent config to {config_path}")
        return config_path
