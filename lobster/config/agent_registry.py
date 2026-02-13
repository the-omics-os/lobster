"""
Centralized Agent Registry for the Lobster system.

This module provides backward-compatible facade functions for agent discovery.
All agent discovery now flows through ComponentRegistry which discovers agents
via entry points (lobster.agents).

The hardcoded AGENT_REGISTRY dict has been eliminated in favor of dynamic
discovery. Core agents are registered as entry points in pyproject.toml.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class AgentRegistryConfig:
    """Configuration for an agent in the system.

    Attributes:
        name: Unique identifier for the agent.
        display_name: Human-readable name for UI display.
        description: Agent's purpose and capabilities.
        factory_function: Module path to the factory function (e.g., 'lobster.agents.x.x').
        handoff_tool_name: Name of the handoff tool (None = not directly accessible).
        handoff_tool_description: Description for when to use this agent.
        child_agents: List of agent names this agent can delegate to.
        supervisor_accessible: Controls whether supervisor can directly handoff to this agent.
            - None (default): Inferred from child_agents relationships. If this agent
              appears in ANY parent's child_agents list, it's NOT supervisor-accessible.
            - True: Explicitly allow supervisor access (override inference).
            - False: Explicitly deny supervisor access (override inference).
        tier_requirement: Subscription tier required to use this agent ("free", "premium",
            "enterprise"). Controls runtime access based on user's subscription level.
        package_name: PyPI package that provides this agent (None = core lobster-ai).
            Used for tracing which package registered an agent.
        service_dependencies: List of service names this agent requires to function.
            Used for validation that required services are available at runtime.
    """

    name: str
    display_name: str
    description: str
    factory_function: str  # Module path to the factory function
    handoff_tool_name: Optional[str] = None
    handoff_tool_description: Optional[str] = None
    child_agents: Optional[List[str]] = (
        None  # List of agent names this agent can delegate to
    )
    supervisor_accessible: Optional[bool] = None  # None=infer, True/False=override
    # === Phase 2: Agent Package Contract fields ===
    tier_requirement: str = "free"  # "free", "premium", "enterprise"
    package_name: Optional[str] = None  # PyPI package name (None = core lobster-ai)
    service_dependencies: Optional[List[str]] = None  # Services required by this agent


# =============================================================================
# FACADE FUNCTIONS - Delegate to ComponentRegistry
# =============================================================================
# These functions maintain backward compatibility for all code that imports
# from agent_registry.py. Internally, they delegate to ComponentRegistry
# which discovers agents via entry points.


def get_all_agent_names() -> list[str]:
    """Get all agent names including system agents."""
    from lobster.core.component_registry import component_registry

    return list(component_registry.list_agents().keys())


def get_worker_agents() -> Dict[str, AgentRegistryConfig]:
    """Get only the worker agents (excluding system agents).

    Note: With dynamic discovery, this returns all agents discovered
    via entry points. The "worker" vs "system" distinction is historical.
    """
    from lobster.core.component_registry import component_registry

    return component_registry.list_agents()


def get_agent_registry_config(agent_name: str) -> Optional[AgentRegistryConfig]:
    """Get registry configuration for a specific agent."""
    from lobster.core.component_registry import component_registry

    return component_registry.get_agent(agent_name)


def get_valid_handoffs() -> Dict[str, set]:
    """
    Build a map of valid agent handoffs from the registry.

    Returns a dict mapping agent_name -> set of agents it can hand off to.
    Used to validate/correct handoff display during parallel tool calls.

    Example:
        {
            "supervisor": {"data_expert_agent", "research_agent", ...},
            "data_expert_agent": {"metadata_assistant"},
            "research_agent": {"metadata_assistant"},
            "transcriptomics_expert": {"annotation_expert", "de_analysis_expert"},
        }
    """
    from lobster.core.component_registry import component_registry

    all_agents = component_registry.list_agents()

    valid_handoffs: Dict[str, set] = {}

    # Build supervisor's valid targets (all supervisor-accessible agents)
    supervisor_targets = set()
    for name, config in all_agents.items():
        # Check explicit supervisor_accessible flag first
        if config.supervisor_accessible is True:
            supervisor_targets.add(name)
        elif config.supervisor_accessible is False:
            continue  # Explicitly not accessible
        else:
            # Infer: accessible if NOT a child of any other agent
            is_child = False
            for other_config in all_agents.values():
                if other_config.child_agents and name in other_config.child_agents:
                    is_child = True
                    break
            if not is_child:
                supervisor_targets.add(name)

    valid_handoffs["supervisor"] = supervisor_targets

    # Build each agent's valid targets from child_agents
    for name, config in all_agents.items():
        if config.child_agents:
            valid_handoffs[name] = set(config.child_agents)
        else:
            valid_handoffs[name] = set()

    return valid_handoffs


def is_valid_handoff(from_agent: str, to_agent: str) -> bool:
    """
    Check if a handoff from one agent to another is valid.

    Args:
        from_agent: The agent initiating the handoff
        to_agent: The agent being handed off to

    Returns:
        True if the handoff is valid according to the agent hierarchy
    """
    valid_handoffs = get_valid_handoffs()
    return to_agent in valid_handoffs.get(from_agent, set())


def import_agent_factory(factory_path: str) -> Callable:
    """Dynamically import an agent factory function."""
    module_path, function_name = factory_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[function_name])
    return getattr(module, function_name)


# =============================================================================
# BACKWARD COMPATIBILITY SHIMS
# =============================================================================
# These maintain compatibility with code that still imports AGENT_REGISTRY
# directly. New code should use the facade functions above instead.


def _ensure_plugins_loaded() -> None:
    """Ensure plugins are discovered. No-op - ComponentRegistry loads on access."""
    pass


class _AgentRegistryProxy:
    """Lazy proxy that returns ComponentRegistry agents when accessed as dict."""

    def __getitem__(self, key: str) -> AgentRegistryConfig:
        config = get_agent_registry_config(key)
        if config is None:
            raise KeyError(key)
        return config

    def __contains__(self, key: str) -> bool:
        return get_agent_registry_config(key) is not None

    def __iter__(self):
        return iter(get_all_agent_names())

    def keys(self):
        return get_all_agent_names()

    def values(self):
        return list(get_worker_agents().values())

    def items(self):
        return get_worker_agents().items()

    def get(self, key: str, default=None):
        config = get_agent_registry_config(key)
        return config if config is not None else default

    def __len__(self):
        return len(get_all_agent_names())


# Backward-compatible AGENT_REGISTRY - acts like a dict but delegates to ComponentRegistry
AGENT_REGISTRY = _AgentRegistryProxy()
