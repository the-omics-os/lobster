"""Prompt registry for discovering and managing agent prompt configs."""

import logging
from typing import Dict, List, Optional

from lobster.prompts.config import PromptConfig

logger = logging.getLogger(__name__)


class PromptRegistry:
    """Singleton that discovers YAML configs from installed packages.

    The registry maintains a mapping of agent names to their prompt configs.
    Configs can be:
    - Registered programmatically via register_config()
    - Auto-discovered from installed packages via entry points (future)

    Usage:
        >>> registry = get_prompt_registry()
        >>> config = registry.get_prompt_config("transcriptomics_expert")
    """

    _instance: Optional["PromptRegistry"] = None

    def __new__(cls) -> "PromptRegistry":
        """Singleton pattern - return existing instance if available."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._configs: Dict[str, PromptConfig] = {}
            cls._instance._loaded = False
        return cls._instance

    def get_prompt_config(self, agent_name: str) -> Optional[PromptConfig]:
        """Get the prompt config for an agent.

        Args:
            agent_name: Agent identifier (e.g., "transcriptomics_expert")

        Returns:
            PromptConfig if found, None otherwise
        """
        if not self._loaded:
            self._discover_prompts()
        return self._configs.get(agent_name)

    def list_agents(self) -> List[str]:
        """List all agents with registered prompt configs.

        Returns:
            List of agent names with configs
        """
        if not self._loaded:
            self._discover_prompts()
        return list(self._configs.keys())

    def has_config(self, agent_name: str) -> bool:
        """Check if an agent has a registered prompt config.

        Args:
            agent_name: Agent identifier

        Returns:
            True if config exists
        """
        if not self._loaded:
            self._discover_prompts()
        return agent_name in self._configs

    def _discover_prompts(self) -> None:
        """Discover prompt configs from installed packages.

        Currently loads core prompts. Future: entry point discovery.
        """
        self._load_core_prompts()
        self._loaded = True
        logger.debug(f"PromptRegistry loaded {len(self._configs)} prompt configs")

    def _load_core_prompts(self) -> None:
        """Load prompt configs from core package.

        Override point for loading YAML configs from lobster.prompts.
        """
        # Core prompts will be loaded from YAML files in future iterations
        # For now, configs are registered programmatically
        pass

    def register_config(self, config: PromptConfig) -> None:
        """Register a prompt config for an agent.

        Args:
            config: PromptConfig to register

        Note:
            This overwrites any existing config for the same agent_name.
        """
        self._configs[config.agent_name] = config
        logger.debug(f"Registered prompt config for agent: {config.agent_name}")

    def unregister_config(self, agent_name: str) -> bool:
        """Remove a prompt config from the registry.

        Args:
            agent_name: Agent identifier to remove

        Returns:
            True if config was removed, False if not found
        """
        if agent_name in self._configs:
            del self._configs[agent_name]
            logger.debug(f"Unregistered prompt config for agent: {agent_name}")
            return True
        return False

    def clear(self) -> None:
        """Clear all registered configs and reset loaded state.

        Use this for testing or hot-reload scenarios.
        """
        self._configs.clear()
        self._loaded = False
        logger.debug("PromptRegistry cleared")

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton instance (for testing only)."""
        cls._instance = None


# Module-level singleton accessor
_registry: Optional[PromptRegistry] = None


def get_prompt_registry() -> PromptRegistry:
    """Get the global PromptRegistry singleton.

    Returns:
        The shared PromptRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry
