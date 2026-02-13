"""Generic handoff tool builder from AGENT_CONFIG declarations.

This module provides build_handoff_tools() which creates delegation tools
based on child_agents declared in AgentRegistryConfig. This replaces manual
tool creation in individual agent factories.

Pattern:
1. AGENT_CONFIG declares child_agents: ["annotation_expert", "de_analysis_expert"]
2. build_handoff_tools() creates lazy delegation tools for each child
3. Missing/uninstalled children are silently excluded (graceful degradation)
"""

from typing import Any, Callable, Dict, List, Optional, Set
import logging

from langchain_core.tools import tool

from lobster.config.agent_registry import AgentRegistryConfig

logger = logging.getLogger(__name__)


def build_handoff_tools(
    agent_config: AgentRegistryConfig,
    agents_dict: Dict[str, Any],
    installed_agents: Optional[Set[str]] = None,
) -> List[Callable]:
    """Build handoff tools from AGENT_CONFIG child_agents declarations.

    Creates lazy delegation tools for each child agent declared in the config.
    Tools resolve agents at invocation time from agents_dict, enabling
    single-pass graph creation.

    Args:
        agent_config: Config declaring handoffs via child_agents field
        agents_dict: Shared dict for lazy agent resolution (captured by reference)
        installed_agents: Optional set of installed agent names. If None,
            uses ComponentRegistry to determine availability.

    Returns:
        List of handoff tool functions. Missing/uninstalled agents are
        silently excluded (no error, no tool created).

    Example:
        config = AgentRegistryConfig(
            name="transcriptomics_expert",
            child_agents=["annotation_expert", "de_analysis_expert"],
            ...
        )
        tools = build_handoff_tools(config, agents_dict)
        # Returns tools for available children only
    """
    tools = []

    if not agent_config.child_agents:
        return tools

    # Determine installed agents if not provided
    if installed_agents is None:
        # Lazy import to avoid circular dependency
        from lobster.core.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        installed_agents = set(registry.list_agents())

    for child_name in agent_config.child_agents:
        # Silent exclusion of uninstalled/unavailable agents
        if child_name not in installed_agents:
            logger.debug(f"Excluding handoff to '{child_name}' - not installed")
            continue

        # Get child config for description
        from lobster.core.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        child_config = registry.get_agent(child_name)

        if child_config is None:
            logger.debug(f"Excluding handoff to '{child_name}' - config not found")
            continue

        # Get description from child config
        description = (
            child_config.handoff_tool_description or child_config.description or ""
        )

        # Create lazy delegation tool using Phase 5 pattern
        handoff_tool = _create_lazy_handoff_tool(
            agent_name=child_name,
            agents_dict=agents_dict,
            description=description,
        )
        tools.append(handoff_tool)

    logger.debug(
        f"Built {len(tools)} handoff tools for {agent_config.name} "
        f"(declared {len(agent_config.child_agents)}, installed {len(installed_agents)})"
    )

    return tools


def _create_lazy_handoff_tool(
    agent_name: str,
    agents_dict: Dict[str, Any],
    description: str,
) -> Callable:
    """Create a delegation tool with lazy agent resolution.

    The tool captures agents_dict by reference. When invoked, it looks up
    the agent by name from the dict. This enables single-pass agent creation
    where children may not exist when parent is created.

    This is the same pattern as _create_lazy_delegation_tool in graph.py,
    extracted for reuse.

    Args:
        agent_name: Name of the child agent to delegate to
        agents_dict: Shared dict that will contain created agents
        description: Description for the tool

    Returns:
        Tool function with lazy agent resolution
    """
    # Capture in closure (avoid Python closure variable capture issues)
    _name = agent_name
    _dict = agents_dict
    _desc = description

    @tool(f"handoff_to_{_name}", description=f"Delegate task to {_name}. {_desc}")
    def invoke_agent_lazy(task_description: str) -> str:
        """Invoke a sub-agent with a task description (lazy resolution).

        Args:
            task_description: Detailed description of what the agent should do,
                including all relevant context. Should be in task format starting
                with 'Your task is to ...'

        Returns:
            Agent's response as string
        """
        # Lazy resolution: look up agent at invocation time
        agent = _dict.get(_name)

        if agent is None:
            logger.warning(f"Agent '{_name}' not found in agents dict")
            return (
                f"Agent '{_name}' is not available. "
                "It may be excluded by configuration or subscription tier."
            )

        logger.debug(
            f"[lazy handoff] Invoking {_name} with task: {task_description[:100]}..."
        )

        # Pass explicit agent name in config for proper callback attribution
        config = {
            "run_name": _name,
            "tags": [_name],
        }

        # Invoke the sub-agent with the task as a user message
        result = agent.invoke(
            {"messages": [{"role": "user", "content": task_description}]}, config=config
        )

        # Extract the final message content
        final_msg = result.get("messages", [])[-1] if result.get("messages") else None
        if final_msg is None:
            return f"Agent {_name} returned no response."

        content = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
        logger.debug(
            f"[lazy handoff] Agent {_name} completed. Response length: {len(content)}"
        )

        return content

    return invoke_agent_lazy


def get_unavailable_agents(
    agent_config: AgentRegistryConfig,
    installed_agents: Optional[Set[str]] = None,
) -> List[str]:
    """Get list of declared but unavailable child agents.

    Useful for prompt generation - inform agent about missing capabilities.

    Args:
        agent_config: Config declaring child_agents
        installed_agents: Optional set of installed agents

    Returns:
        List of agent names that are declared but not installed
    """
    if not agent_config.child_agents:
        return []

    if installed_agents is None:
        from lobster.core.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        installed_agents = set(registry.list_agents())

    return [name for name in agent_config.child_agents if name not in installed_agents]
