"""
Unified component registry for premium services and agents via entry points.

This module discovers and loads components from:
- lobster-ai: Core agents (data_expert, research_agent, transcriptomics_expert, etc.)
- lobster-premium: Shared premium features
- lobster-custom-*: Customer-specific features

Components are advertised via entry points in pyproject.toml:

    [project.entry-points."lobster.services"]
    publication_processing = "package.module:ServiceClass"

    [project.entry-points."lobster.agents"]
    metadata_assistant = "package.module:AGENT_CONFIG"

    [project.entry-points."lobster.agent_configs"]
    metadata_assistant = "package.module:CUSTOM_AGENT_CONFIG"

Usage:
    from lobster.core.component_registry import component_registry

    # Services
    ServiceClass = component_registry.get_service('publication_processing')

    # Agents (ALL agents discovered via entry points)
    agent_config = component_registry.get_agent('research_agent')
    all_agents = component_registry.list_agents()  # Single source of truth

    # Agent LLM Configs
    llm_config = component_registry.get_agent_config('metadata_assistant')
    all_configs = component_registry.list_agent_configs()
"""

import importlib.metadata
import logging
import sys
from typing import Any, Dict, Optional, Type

logger = logging.getLogger(__name__)


def check_plugin_compatibility(package_name: str) -> tuple[bool, str]:
    """Check if a plugin package is compatible with current lobster-ai.

    Validates that the plugin's declared minimum lobster-ai version requirement
    is satisfied by the currently installed version. Uses semantic versioning
    for comparison (e.g., 0.4.10 > 0.4.3).

    Args:
        package_name: Name of the plugin package (e.g., "lobster-transcriptomics")

    Returns:
        Tuple of (is_compatible, message) where:
        - is_compatible: True if plugin is compatible or compatibility cannot be determined
        - message: Human-readable description of compatibility status
    """
    try:
        from packaging import version

        from lobster.version import __version__ as current_version

        # Get package dependencies
        requires = importlib.metadata.requires(package_name) or []

        for req in requires:
            if req.startswith("lobster-ai"):
                # Parse version constraint (supports >=, ==, ~=)
                if ">=" in req:
                    min_version = req.split(">=")[1].split(",")[0].split(";")[0].strip()
                    if version.parse(current_version) < version.parse(min_version):
                        return (
                            False,
                            f"Requires lobster-ai>={min_version}, found {current_version}",
                        )

        return True, "Compatible"

    except importlib.metadata.PackageNotFoundError:
        return False, f"Package {package_name} not installed"
    except Exception as e:
        logger.warning(f"Could not verify compatibility for {package_name}: {e}")
        return True, "Could not verify (assuming compatible)"


class ComponentConflictError(Exception):
    """Raised when a custom component conflicts with a core component."""

    pass


class ComponentRegistry:
    """
    Unified registry for dynamically loaded services and agents.

    Services: Premium service classes loaded via entry points
    Agents: AgentRegistryConfig instances discovered via entry points
           (both core lobster-ai and custom lobster-custom-* packages)
    """

    def __init__(self):
        self._services: Dict[str, Type[Any]] = {}
        self._agents: Dict[str, Any] = {}  # ALL AgentRegistryConfig instances
        self._custom_agent_configs: Dict[str, Any] = {}  # CustomAgentConfig instances
        self._loaded = False

    def load_components(self) -> None:
        """
        Discover and load all components from entry points.
        Idempotent - safe to call multiple times.
        """
        if self._loaded:
            return

        logger.debug("Discovering components via entry points...")

        # Load services from 'lobster.services' entry point
        self._load_entry_point_group("lobster.services", self._services)

        # Load ALL agents from 'lobster.agents' entry point
        # This includes both core (lobster-ai) and custom (lobster-custom-*) agents
        self._load_entry_point_group("lobster.agents", self._agents)

        # Load custom agent LLM configs from 'lobster.agent_configs' entry point
        self._load_entry_point_group(
            "lobster.agent_configs", self._custom_agent_configs
        )

        self._loaded = True
        logger.debug(
            f"Component discovery complete. "
            f"Services: {len(self._services)}, "
            f"Agents: {len(self._agents)}, "
            f"Custom agent configs: {len(self._custom_agent_configs)}"
        )

    def _load_entry_point_group(self, group: str, target_dict: Dict[str, Any]) -> None:
        """Load all entry points from a specific group into target dict.

        Performs version compatibility checking for external plugin packages
        before loading. Incompatible plugins are skipped with a warning.
        """
        # Handle Python 3.10+ vs 3.9 API differences
        if sys.version_info >= (3, 10):
            from importlib.metadata import entry_points

            discovered = entry_points(group=group)
        else:
            from importlib.metadata import entry_points

            eps = entry_points()
            discovered = eps.get(group, [])

        for entry in discovered:
            try:
                # Check version compatibility for external plugins
                dist_name = None
                try:
                    dist_name = entry.dist.name if hasattr(entry, "dist") else None
                except Exception:
                    pass  # Some entry points may not have dist info

                # Only check compatibility for external packages (not lobster-ai itself)
                if dist_name and dist_name != "lobster-ai":
                    is_compatible, msg = check_plugin_compatibility(dist_name)
                    if not is_compatible:
                        logger.warning(
                            f"Skipping incompatible plugin {entry.name} from {dist_name}: {msg}"
                        )
                        continue

                loaded = entry.load()
                target_dict[entry.name] = loaded
                logger.info(
                    f"Loaded {group.split('.')[-1]} '{entry.name}' from {entry.value}"
                )
            except Exception as e:
                logger.warning(f"Failed to load {group} '{entry.name}': {e}")

    # =========================================================================
    # SERVICE API
    # =========================================================================

    def get_service(self, name: str, required: bool = False) -> Optional[Type[Any]]:
        """
        Get a premium service class by name.

        Args:
            name: Service name (e.g., 'publication_processing')
            required: If True, raise error when service not found

        Returns:
            Service class if found, None otherwise

        Raises:
            ValueError: If required=True and service not found
        """
        if not self._loaded:
            self.load_components()

        service = self._services.get(name)

        if service is None and required:
            raise ValueError(
                f"Required service '{name}' not found. "
                f"Available services: {list(self._services.keys())}"
            )

        return service

    def has_service(self, name: str) -> bool:
        """Check if a service is available."""
        if not self._loaded:
            self.load_components()
        return name in self._services

    def list_services(self) -> Dict[str, str]:
        """List all available services with their module paths."""
        if not self._loaded:
            self.load_components()
        return {
            name: f"{cls.__module__}.{cls.__name__}"
            for name, cls in self._services.items()
        }

    # =========================================================================
    # AGENT API
    # =========================================================================

    def get_agent(self, name: str, required: bool = False) -> Optional[Any]:
        """
        Get an agent config by name.

        Args:
            name: Agent name (e.g., 'research_agent', 'data_expert_agent')
            required: If True, raise error when agent not found

        Returns:
            AgentRegistryConfig if found, None otherwise

        Raises:
            ValueError: If required=True and agent not found
        """
        if not self._loaded:
            self.load_components()

        agent = self._agents.get(name)

        if agent is None and required:
            raise ValueError(
                f"Required agent '{name}' not found. "
                f"Available agents: {list(self._agents.keys())}"
            )

        return agent

    def has_agent(self, name: str) -> bool:
        """Check if an agent is available."""
        if not self._loaded:
            self.load_components()
        return name in self._agents

    def list_custom_agents(self) -> Dict[str, Any]:
        """List custom agents only (DEPRECATED - use list_agents() instead).

        Note: This method is deprecated. With dynamic discovery, all agents
        (core + custom) come from entry points. Use list_agents() instead.

        For backward compatibility, this filters to non-core agents
        (agents from packages other than lobster-ai).
        """
        if not self._loaded:
            self.load_components()
        # For backward compatibility, filter to non-core agents
        # Core agents have package_name=None (from lobster-ai)
        return {
            name: config
            for name, config in self._agents.items()
            if getattr(config, "package_name", None) is not None
        }

    def list_agents(self) -> Dict[str, Any]:
        """
        List ALL agents discovered via entry points.

        This is the single source of truth for agent discovery. All agents
        (core lobster-ai + custom lobster-custom-*) are discovered via
        the 'lobster.agents' entry point group.

        Returns:
            Dict[str, AgentRegistryConfig] - All available agents
        """
        if not self._loaded:
            self.load_components()
        return dict(self._agents)

    # =========================================================================
    # AGENT CONFIG API
    # =========================================================================

    def get_agent_config(self, name: str, required: bool = False) -> Optional[Any]:
        """
        Get a custom agent LLM config by name.

        Args:
            name: Agent name (e.g., 'metadata_assistant')
            required: If True, raise error when config not found

        Returns:
            CustomAgentConfig if found, None otherwise

        Raises:
            ValueError: If required=True and config not found
        """
        if not self._loaded:
            self.load_components()

        config = self._custom_agent_configs.get(name)

        if config is None and required:
            raise ValueError(
                f"Required agent config '{name}' not found. "
                f"Available configs: {list(self._custom_agent_configs.keys())}"
            )

        return config

    def has_agent_config(self, name: str) -> bool:
        """Check if a custom agent config is available."""
        if not self._loaded:
            self.load_components()
        return name in self._custom_agent_configs

    def list_agent_configs(self) -> Dict[str, Any]:
        """
        List all custom agent LLM configs (from entry points).

        Returns:
            Dict[str, CustomAgentConfig] - All available agent configs
        """
        if not self._loaded:
            self.load_components()
        return dict(self._custom_agent_configs)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive registry info for diagnostics."""
        if not self._loaded:
            self.load_components()

        return {
            "services": {
                "count": len(self._services),
                "names": list(self._services.keys()),
            },
            "agents": {
                "count": len(self._agents),
                "names": list(self._agents.keys()),
            },
            "custom_agent_configs": {
                "count": len(self._custom_agent_configs),
                "names": list(self._custom_agent_configs.keys()),
            },
        }

    def reset(self) -> None:
        """Reset the registry state (for testing)."""
        self._services.clear()
        self._agents.clear()
        self._custom_agent_configs.clear()
        self._loaded = False


# Singleton instance
component_registry = ComponentRegistry()
