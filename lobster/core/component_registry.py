"""
Unified component registry for services, agents, and omics plugins via entry points.

This module discovers and loads components from:
- lobster-ai: Core agents (data_expert, research_agent, transcriptomics_expert, etc.)
- lobster-premium: Shared premium features
- lobster-custom-*: Customer-specific features
- Any package advertising lobster.* entry points

Components are advertised via entry points in pyproject.toml:

    [project.entry-points."lobster.services"]
    publication_processing = "package.module:ServiceClass"

    [project.entry-points."lobster.agents"]
    metadata_assistant = "package.module:AGENT_CONFIG"

    [project.entry-points."lobster.adapters"]
    metabolomics_lc_ms = "package.module:create_lc_ms_adapter"

    [project.entry-points."lobster.providers"]
    metabolights = "package.module:MetaboLightsProvider"

    [project.entry-points."lobster.download_services"]
    metabolights = "package.module:MetaboLightsDownloadService"

    [project.entry-points."lobster.queue_preparers"]
    metabolights = "package.module:MetaboLightsQueuePreparer"

Usage:
    from lobster.core.component_registry import component_registry

    # Services
    ServiceClass = component_registry.get_service('publication_processing')

    # Agents (ALL agents discovered via entry points)
    agent_config = component_registry.get_agent('research_agent')
    all_agents = component_registry.list_agents()  # Single source of truth

    # Adapters (factory callables returning configured instances)
    adapter_factory = component_registry.get_adapter('metabolomics_lc_ms')
    adapter = adapter_factory()  # Returns configured adapter instance

    # Providers, Download Services, Queue Preparers
    provider_cls = component_registry.get_provider('metabolights')
    dl_service_cls = component_registry.get_download_service('metabolights')
    preparer_cls = component_registry.get_queue_preparer('metabolights')

Note on factory contract:
    - lobster.adapters entry points MUST be callables (factory functions) that
      return configured adapter instances. Adapters often need constructor args
      (e.g., data_type="lc_ms") that raw classes can't express.
    - lobster.providers, lobster.download_services, lobster.queue_preparers
      entry points should be classes that can be instantiated with no args,
      or factory callables returning instances.
"""

import importlib.metadata
import logging
import sys
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

logger = logging.getLogger(__name__)

_ENTRY_POINT_GROUPS: Tuple[str, ...] = (
    "lobster.services",
    "lobster.agents",
    "lobster.adapters",
    "lobster.providers",
    "lobster.download_services",
    "lobster.queue_preparers",
)


# =============================================================================
# KNOWN PACKAGE MAPPINGS (single source of truth for install suggestions)
# =============================================================================
# Maps agent entry-point names to their PyPI package.
# Used when a user references an agent that isn't installed — the registry
# can't discover uninstalled packages (entry points don't exist), so this
# static mapping is the ONLY way to suggest the correct install command.
#
# Authoritative source: packages/*/pyproject.toml [project.entry-points."lobster.agents"]
# Update this when adding new first-party agent packages.

AGENT_TO_PACKAGE: Dict[str, str] = {
    # lobster-research
    "research_agent": "lobster-research",
    "data_expert_agent": "lobster-research",
    # lobster-transcriptomics
    "transcriptomics_expert": "lobster-transcriptomics",
    "annotation_expert": "lobster-transcriptomics",
    "de_analysis_expert": "lobster-transcriptomics",
    # lobster-visualization
    "visualization_expert_agent": "lobster-visualization",
    # lobster-metadata
    "metadata_assistant": "lobster-metadata",
    # lobster-structural-viz
    "protein_structure_visualization_expert": "lobster-structural-viz",
    # lobster-genomics
    "genomics_expert": "lobster-genomics",
    "variant_analysis_expert": "lobster-genomics",
    # lobster-proteomics
    "proteomics_expert": "lobster-proteomics",
    "proteomics_de_analysis_expert": "lobster-proteomics",
    "biomarker_discovery_expert": "lobster-proteomics",
    # lobster-metabolomics
    "metabolomics_expert": "lobster-metabolomics",
    # lobster-ml
    "machine_learning_expert": "lobster-ml",
    "feature_selection_expert": "lobster-ml",
    "survival_analysis_expert": "lobster-ml",
    # lobster-drug-discovery
    "drug_discovery_expert": "lobster-drug-discovery",
    "cheminformatics_expert": "lobster-drug-discovery",
    "clinical_dev_expert": "lobster-drug-discovery",
    "pharmacogenomics_expert": "lobster-drug-discovery",
}

# Maps LLM provider names to their lobster-ai extra and the actual PyPI package.
# Format: provider_name -> (lobster_extra, pypi_package)
LLM_PROVIDER_PACKAGES: Dict[str, Tuple[str, str]] = {
    "anthropic": ("anthropic", "langchain-anthropic"),
    "bedrock": ("bedrock", "langchain-aws"),
    "ollama": ("ollama", "langchain-ollama"),
    "gemini": ("gemini", "langchain-google-genai"),
    "azure": ("azure", "langchain-openai"),
    "openai": ("openai", "langchain-openai"),
}


def get_install_command(package: str, *, is_extra: bool = False) -> str:
    """Build the correct install command for the user's environment.

    Detects whether the user is in a uv tool environment (installed via
    ``uv tool install``) and returns the appropriate command.

    Args:
        package: PyPI package name (e.g., "lobster-transcriptomics") or
            lobster-ai extra name (e.g., "anthropic") when *is_extra* is True.
        is_extra: If True, *package* is a lobster-ai extra (e.g., "anthropic"),
            and the command should use ``lobster-ai[extra]`` syntax.

    Returns:
        Human-readable install command string.
    """
    try:
        from lobster.core.uv_tool_env import is_uv_tool_env
    except ImportError:
        # Fail-open: if uv_tool_env can't be imported, assume regular env
        is_uv_tool_env = lambda: False  # noqa: E731

    if is_extra:
        specifier = f"lobster-ai[{package}]"
    else:
        specifier = package

    if is_uv_tool_env():
        if is_extra:
            from lobster.core.uv_tool_env import build_tool_install_command

            return " ".join(build_tool_install_command(extras=[package]))
        else:
            return f"uv tool install lobster-ai --with {specifier}"
    else:
        return f"uv pip install '{specifier}'"


def get_provider_install_command(provider_name: str) -> Optional[str]:
    """Get the install command for a missing LLM provider package.

    Args:
        provider_name: Provider name (e.g., "anthropic", "bedrock").

    Returns:
        Install command string, or None if provider is unknown.
    """
    info = LLM_PROVIDER_PACKAGES.get(provider_name)
    if info is None:
        return None
    extra_name, _ = info
    return get_install_command(extra_name, is_extra=True)


def diagnose_missing_agent(agent_name: str) -> str:
    """Produce an actionable message for a missing agent.

    Distinguishes three cases:
    1. Known agent in a known package that isn't installed.
    2. Known agent whose entry point was discovered but failed to load
       (package installed but broken).
    3. Unknown agent — not in any known package.

    Args:
        agent_name: The agent name that was not found.

    Returns:
        Human-readable diagnostic message.
    """
    # Case 1: Known agent in an uninstalled package
    package = AGENT_TO_PACKAGE.get(agent_name)
    if package is not None:
        cmd = get_install_command(package)
        return (
            f"Agent '{agent_name}' requires package '{package}' which is not installed.\n"
            f"Install with: {cmd}"
        )

    # Case 2: Check if it was a failed entry point (package installed but broken)
    failed = component_registry.get_failed_entry_points()
    for (group, name), error in failed.items():
        if name == agent_name and "agents" in group:
            return (
                f"Agent '{agent_name}' is installed but failed to load: {error}\n"
                f"This is likely a broken dependency — check the error above."
            )

    # Case 3: Truly unknown
    return (
        f"Agent '{agent_name}' is not available. "
        f"It may not be supported yet, or you may need to install a plugin package.\n"
        f"See available agents with: lobster agents list"
    )


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
        # Omics plugin groups
        self._adapters: Dict[str, Union[Callable, Type[Any]]] = {}
        self._providers: Dict[str, Union[Callable, Type[Any]]] = {}
        self._download_services: Dict[str, Union[Callable, Type[Any]]] = {}
        self._queue_preparers: Dict[str, Union[Callable, Type[Any]]] = {}
        # Track entry points that were discovered but failed to load.
        # Keyed by (group, name) to avoid cross-group collisions
        # (e.g., "metabolights" exists in both providers and download_services).
        self._failed_entries: Dict[Tuple[str, str], str] = {}
        self._loaded_groups: set[str] = set()
        self._agent_contract_checked = False
        self._loaded = False

    def load_components(self) -> None:
        """
        Discover and load all components from entry points.
        Idempotent - safe to call multiple times.
        """
        if len(self._loaded_groups) == len(_ENTRY_POINT_GROUPS):
            return

        logger.debug("Discovering components via entry points...")

        for group in _ENTRY_POINT_GROUPS:
            self._ensure_group_loaded(group)

        logger.debug(
            f"Component discovery complete. "
            f"Services: {len(self._services)}, "
            f"Agents: {len(self._agents)}, "
            f"Adapters: {len(self._adapters)}, "
            f"Providers: {len(self._providers)}, "
            f"Download services: {len(self._download_services)}, "
            f"Queue preparers: {len(self._queue_preparers)}"
        )

    def _target_dict_for_group(self, group: str) -> Dict[str, Any]:
        """Return the storage dictionary for an entry-point group."""
        group_targets: Dict[str, Dict[str, Any]] = {
            "lobster.services": self._services,
            "lobster.agents": self._agents,
            "lobster.adapters": self._adapters,
            "lobster.providers": self._providers,
            "lobster.download_services": self._download_services,
            "lobster.queue_preparers": self._queue_preparers,
        }
        try:
            return group_targets[group]
        except KeyError as exc:
            raise ValueError(f"Unknown component group: {group}") from exc

    def _ensure_group_loaded(self, group: str) -> None:
        """Load a single entry-point group on demand."""
        if group in self._loaded_groups:
            return

        self._load_entry_point_group(group, self._target_dict_for_group(group))
        self._loaded_groups.add(group)
        self._loaded = True

        if group == "lobster.agents" and not self._agent_contract_checked:
            self._check_agent_contract_compliance()
            self._agent_contract_checked = True

    def _check_agent_contract_compliance(self) -> None:
        """Soft enforcement: warn about agents missing plugin contract fields.

        Checks loaded AGENT_CONFIG objects for required fields (tier_requirement,
        package_name). This is a detection mechanism, not a hard gate — existing
        pre-AQUADIF agents won't be blocked.

        Tool-level AQUADIF metadata is enforced by contract test mixin at test
        time, not at discovery time (tools require factory instantiation).
        """
        for name, config in self._agents.items():
            if not hasattr(config, "tier_requirement"):
                logger.warning(
                    f"Agent '{name}' missing tier_requirement field. "
                    f"Add tier_requirement to AGENT_CONFIG for plugin contract compliance."
                )
            if not hasattr(config, "name") or not config.name:
                logger.warning(
                    f"Agent '{name}' has empty or missing name in AGENT_CONFIG."
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
                if group == "lobster.agents":
                    self._annotate_agent_package(entry.name, loaded, dist_name)
                target_dict[entry.name] = loaded
                logger.info(
                    f"Loaded {group.split('.')[-1]} '{entry.name}' from {entry.value}"
                )
            except Exception as e:
                self._failed_entries[(group, entry.name)] = str(e)
                logger.warning(f"Failed to load {group} '{entry.name}': {e}")

    def _annotate_agent_package(
        self, entry_name: str, config: Any, dist_name: Optional[str]
    ) -> None:
        """Populate package metadata from the entry point distribution.

        Agent packages historically omitted ``package_name`` from AGENT_CONFIG and
        relied on entry point ownership instead. Runtime commands and integration
        tests need the owning package to be available from the config object.
        """
        if getattr(config, "package_name", None):
            return

        package_name = dist_name or AGENT_TO_PACKAGE.get(entry_name)
        if package_name and package_name != "lobster-ai":
            try:
                config.package_name = package_name
            except Exception:
                logger.debug(
                    "Could not annotate package_name for agent '%s'", entry_name
                )

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
        self._ensure_group_loaded("lobster.services")

        service = self._services.get(name)

        if service is None and required:
            raise ValueError(
                f"Required service '{name}' not found. "
                f"Available services: {list(self._services.keys())}"
            )

        return service

    def has_service(self, name: str) -> bool:
        """Check if a service is available."""
        self._ensure_group_loaded("lobster.services")
        return name in self._services

    def list_services(self) -> Dict[str, str]:
        """List all available services with their module paths."""
        self._ensure_group_loaded("lobster.services")
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
        self._ensure_group_loaded("lobster.agents")

        agent = self._agents.get(name)

        if agent is None and required:
            message = diagnose_missing_agent(name)
            if self._agents:
                message += f"\nAvailable agents: {list(self._agents.keys())}"
            raise ValueError(message)

        return agent

    def has_agent(self, name: str) -> bool:
        """Check if an agent is available."""
        self._ensure_group_loaded("lobster.agents")
        return name in self._agents

    def list_custom_agents(self) -> Dict[str, Any]:
        """List custom agents only (DEPRECATED - use list_agents() instead).

        Note: This method is deprecated. With dynamic discovery, all agents
        (core + custom) come from entry points. Use list_agents() instead.

        For backward compatibility, this filters to non-core agents
        (agents from packages other than lobster-ai).
        """
        self._ensure_group_loaded("lobster.agents")
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
        self._ensure_group_loaded("lobster.agents")
        self._check_agent_name_collisions()
        return dict(self._agents)

    def _check_agent_name_collisions(self) -> None:
        """Reject custom agents that shadow known first-party agent names."""
        for name, config in self._agents.items():
            package_name = getattr(config, "package_name", None)
            first_party_package = AGENT_TO_PACKAGE.get(name)
            if package_name and first_party_package and package_name != first_party_package:
                raise ComponentConflictError(
                    f"Agent name collision: '{name}' is provided by custom package "
                    f"'{package_name}' but is reserved for '{first_party_package}'."
                )

    # =========================================================================
    # ADAPTER API (factory callables returning configured instances)
    # =========================================================================

    def get_adapter(
        self, name: str, required: bool = False
    ) -> Optional[Union[Callable, Type[Any]]]:
        """Get an adapter factory by name.

        Adapters are registered as factory callables that return configured
        instances (e.g., ``create_lc_ms_adapter() -> MetabolomicsAdapter``).

        Args:
            name: Adapter name (e.g., 'metabolomics_lc_ms')
            required: If True, raise error when not found

        Returns:
            Callable/class if found, None otherwise
        """
        self._ensure_group_loaded("lobster.adapters")
        adapter = self._adapters.get(name)
        if adapter is None and required:
            raise ValueError(
                f"Required adapter '{name}' not found. "
                f"Available adapters: {list(self._adapters.keys())}"
            )
        return adapter

    def has_adapter(self, name: str) -> bool:
        """Check if an adapter is available."""
        self._ensure_group_loaded("lobster.adapters")
        return name in self._adapters

    def list_adapters(self) -> Dict[str, Union[Callable, Type[Any]]]:
        """List all available adapters."""
        self._ensure_group_loaded("lobster.adapters")
        return dict(self._adapters)

    # =========================================================================
    # PROVIDER API
    # =========================================================================

    def get_provider(
        self, name: str, required: bool = False
    ) -> Optional[Union[Callable, Type[Any]]]:
        """Get a provider class/factory by name.

        Args:
            name: Provider name (e.g., 'metabolights')
            required: If True, raise error when not found

        Returns:
            Class/callable if found, None otherwise
        """
        self._ensure_group_loaded("lobster.providers")
        provider = self._providers.get(name)
        if provider is None and required:
            raise ValueError(
                f"Required provider '{name}' not found. "
                f"Available providers: {list(self._providers.keys())}"
            )
        return provider

    def has_provider(self, name: str) -> bool:
        """Check if a provider is available."""
        self._ensure_group_loaded("lobster.providers")
        return name in self._providers

    def list_providers(self) -> Dict[str, Union[Callable, Type[Any]]]:
        """List all available providers."""
        self._ensure_group_loaded("lobster.providers")
        return dict(self._providers)

    # =========================================================================
    # DOWNLOAD SERVICE API
    # =========================================================================

    def get_download_service(
        self, name: str, required: bool = False
    ) -> Optional[Union[Callable, Type[Any]]]:
        """Get a download service class/factory by name.

        Args:
            name: Download service name (e.g., 'metabolights')
            required: If True, raise error when not found

        Returns:
            Class/callable if found, None otherwise
        """
        self._ensure_group_loaded("lobster.download_services")
        svc = self._download_services.get(name)
        if svc is None and required:
            raise ValueError(
                f"Required download service '{name}' not found. "
                f"Available: {list(self._download_services.keys())}"
            )
        return svc

    def has_download_service(self, name: str) -> bool:
        """Check if a download service is available."""
        self._ensure_group_loaded("lobster.download_services")
        return name in self._download_services

    def list_download_services(self) -> Dict[str, Union[Callable, Type[Any]]]:
        """List all available download services."""
        self._ensure_group_loaded("lobster.download_services")
        return dict(self._download_services)

    # =========================================================================
    # QUEUE PREPARER API
    # =========================================================================

    def get_queue_preparer(
        self, name: str, required: bool = False
    ) -> Optional[Union[Callable, Type[Any]]]:
        """Get a queue preparer class/factory by name.

        Args:
            name: Queue preparer name (e.g., 'metabolights')
            required: If True, raise error when not found

        Returns:
            Class/callable if found, None otherwise
        """
        self._ensure_group_loaded("lobster.queue_preparers")
        prep = self._queue_preparers.get(name)
        if prep is None and required:
            raise ValueError(
                f"Required queue preparer '{name}' not found. "
                f"Available: {list(self._queue_preparers.keys())}"
            )
        return prep

    def has_queue_preparer(self, name: str) -> bool:
        """Check if a queue preparer is available."""
        self._ensure_group_loaded("lobster.queue_preparers")
        return name in self._queue_preparers

    def list_queue_preparers(self) -> Dict[str, Union[Callable, Type[Any]]]:
        """List all available queue preparers."""
        self._ensure_group_loaded("lobster.queue_preparers")
        return dict(self._queue_preparers)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive registry info for diagnostics."""
        self.load_components()
        custom_agents = self.list_custom_agents()

        return {
            "services": {
                "count": len(self._services),
                "names": list(self._services.keys()),
            },
            "agents": {
                "count": len(self._agents),
                "names": list(self._agents.keys()),
            },
            "custom_agents": {
                "count": len(custom_agents),
                "names": list(custom_agents.keys()),
            },
            "total_agents": len(self._agents),
            "adapters": {
                "count": len(self._adapters),
                "names": list(self._adapters.keys()),
            },
            "providers": {
                "count": len(self._providers),
                "names": list(self._providers.keys()),
            },
            "download_services": {
                "count": len(self._download_services),
                "names": list(self._download_services.keys()),
            },
            "queue_preparers": {
                "count": len(self._queue_preparers),
                "names": list(self._queue_preparers.keys()),
            },
            "failed_entries": {
                "count": len(self._failed_entries),
                "entries": {
                    f"{group}:{name}": error
                    for (group, name), error in self._failed_entries.items()
                },
            },
        }

    def get_failed_entry_points(self) -> Dict[Tuple[str, str], str]:
        """Get entry points that were discovered but failed to load.

        Returns:
            Dict mapping (group, name) tuples to error messages.
            Example: {("lobster.agents", "transcriptomics_expert"): "No module named 'scipy'"}
        """
        return dict(self._failed_entries)

    def reset(self) -> None:
        """Reset the registry state (for testing)."""
        self._services.clear()
        self._agents.clear()
        self._adapters.clear()
        self._providers.clear()
        self._download_services.clear()
        self._queue_preparers.clear()
        self._failed_entries.clear()
        self._loaded_groups.clear()
        self._agent_contract_checked = False
        self._loaded = False


# Singleton instance
component_registry = ComponentRegistry()
