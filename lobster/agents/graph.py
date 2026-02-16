"""
LangGraph multi-agent graph for bioinformatics analysis.

Implementation using Tool Calling pattern: supervisor invokes sub-agents as tools.
This is simpler and more appropriate for centralized orchestration where users
only interact with the supervisor.

See: https://docs.langchain.com/oss/langchain/multi-agent#tool-calling
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from langchain_core.tools import tool

if TYPE_CHECKING:
    from lobster.config.workspace_agent_config import WorkspaceAgentConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

from lobster.agents.state import OverallState
from lobster.agents.supervisor import create_supervisor_prompt
from lobster.config.agent_registry import import_agent_factory
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.config.supervisor_config import SupervisorConfig
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.custom_code_tool import create_execute_custom_code_tool
from lobster.tools.todo_tools import create_todo_tools
from lobster.tools.workspace_tool import (
    create_delete_from_workspace_tool,
    create_get_content_from_workspace_tool,
    create_list_modalities_tool,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Graph Metadata Types (for API exposure)
# =============================================================================


@dataclass
class AgentInfo:
    """Serializable agent info for frontend consumption.

    This provides a clean, frontend-friendly representation of agent configuration
    that can be easily serialized to JSON for API responses.
    """

    name: str
    display_name: str
    description: str
    is_supervisor_accessible: bool
    parent_agent: Optional[str] = None  # For sub-agents like annotation_expert
    child_agents: Optional[List[str]] = None
    handoff_tool_name: Optional[str] = None

    def to_dict(self) -> Dict:
        """Serialize to dictionary for JSON responses."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "is_supervisor_accessible": self.is_supervisor_accessible,
            "parent_agent": self.parent_agent,
            "child_agents": self.child_agents,
            "handoff_tool_name": self.handoff_tool_name,
        }


@dataclass
class GraphMetadata:
    """Metadata about the created graph for API exposure.

    This captures the state of the agent graph at creation time, including:
    - Which agents are available (after tier filtering)
    - Agent hierarchy (parent/child relationships)
    - Which agents the supervisor can directly access
    - Which agents were excluded due to subscription tier

    Use this to expose agent information to frontends via API.
    """

    subscription_tier: str
    available_agents: List[AgentInfo] = field(default_factory=list)
    supervisor_accessible_agents: List[str] = field(default_factory=list)
    filtered_out_agents: List[str] = field(default_factory=list)

    @property
    def agent_count(self) -> int:
        """Number of available agents."""
        return len(self.available_agents)

    @property
    def supervisor_accessible_count(self) -> int:
        """Number of supervisor-accessible agents."""
        return len(self.supervisor_accessible_agents)

    def to_dict(self) -> Dict:
        """Serialize for JSON API response."""
        return {
            "subscription_tier": self.subscription_tier,
            "available_agents": [a.to_dict() for a in self.available_agents],
            "supervisor_accessible_agents": self.supervisor_accessible_agents,
            "filtered_out_agents": self.filtered_out_agents,
            "agent_count": len(self.available_agents),
            "supervisor_accessible_count": len(self.supervisor_accessible_agents),
        }

    def get_agent_by_name(self, name: str) -> Optional[AgentInfo]:
        """Get agent info by name."""
        for agent in self.available_agents:
            if agent.name == name:
                return agent
        return None


def _get_parent_agent(agent_name: str, worker_agents: Dict) -> Optional[str]:
    """Find the parent agent for a given agent name.

    Args:
        agent_name: Name of the agent to find parent for
        worker_agents: Dictionary of agent configs

    Returns:
        Parent agent name if found, None otherwise
    """
    for parent_name, parent_config in worker_agents.items():
        if parent_config.child_agents and agent_name in parent_config.child_agents:
            return parent_name
    return None


def _create_agent_tool(agent_name: str, agent, tool_name: str, description: str):
    """Create a tool that invokes a sub-agent (Tool Calling pattern).

    This follows the LangChain Tool Calling pattern where sub-agents are
    invoked as tools and return their results directly. The supervisor
    maintains centralized control - sub-agents never interact with users.

    See: https://docs.langchain.com/oss/langchain/multi-agent#tool-calling

    Args:
        agent_name: Internal name of the agent (for logging)
        agent: The compiled agent (Pregel) to invoke
        tool_name: Name for the tool (e.g., "handoff_to_research_agent")
        description: Description of when to use this tool
    """

    @tool(tool_name, description=description)
    def invoke_agent(task_description: str) -> str:
        """Invoke a sub-agent with a task description.

        Args:
            task_description: Detailed description of what the agent should do,
                including all relevant context. Should be in task format starting
                with 'Your task is to ...'
        """
        logger.debug(f"Invoking {agent_name} with task: {task_description[:100]}...")

        # Pass explicit agent name in config for proper callback attribution.
        # metadata propagates to all sub-calls and is passed to handle*Start
        # callbacks (per RunnableConfig docs). This is the most reliable
        # mechanism for agent attribution in nested LangGraph runs.
        config = {
            "run_name": agent_name,
            "tags": [agent_name],
            "metadata": {"agent_name": agent_name},
        }

        # Invoke the sub-agent with the task as a user message
        result = agent.invoke(
            {"messages": [{"role": "user", "content": task_description}]}, config=config
        )

        # Extract the final message content
        final_msg = result.get("messages", [])[-1] if result.get("messages") else None
        if final_msg is None:
            return f"Agent {agent_name} returned no response."

        content = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
        logger.debug(f"Agent {agent_name} completed. Response length: {len(content)}")

        return content

    return invoke_agent


def _create_delegation_tool(agent_name: str, agent, description: str):
    """Create a delegation tool for parent-child agent relationships.

    This is a simplified version for hierarchical delegation within agent families
    (e.g., transcriptomics_expert -> de_analysis_expert).
    """
    return _create_agent_tool(
        agent_name=agent_name,
        agent=agent,
        tool_name=f"handoff_to_{agent_name}",
        description=f"Delegate task to {agent_name}. {description}",
    )


def _create_lazy_delegation_tool(
    agent_name: str,
    agents_dict: Dict[str, Any],
    description: str,
):
    """Create delegation tool with lazy agent resolution.

    The tool captures `agents_dict` by reference. When invoked, it looks up
    the agent by name from the dict. This enables single-pass agent creation
    where children may not exist when parent is created.

    This pattern eliminates the two-pass agent creation where parent agents
    were created twice (once without delegation tools, then again with them).
    Now all agents are created exactly once, and delegation tools resolve
    agents at invocation time.

    Args:
        agent_name: Name of the child agent to delegate to
        agents_dict: Shared dict that will contain created agents
        description: Description for the tool

    Returns:
        Tool function with lazy agent resolution
    """
    # Capture agent_name and agents_dict in closure
    # Use a helper function to avoid Python closure variable capture issues
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
        """
        # Lazy resolution: look up agent at invocation time
        agent = _dict.get(_name)

        if agent is None:
            logger.warning(
                f"Agent '{_name}' not found in agents dict (config may have excluded it)"
            )
            return f"Agent '{_name}' is not available. It may be excluded by configuration or subscription tier."

        logger.debug(
            f"[lazy delegation] Invoking {_name} with task: {task_description[:100]}..."
        )

        # Pass explicit agent name in config for proper callback attribution.
        # metadata propagates to all sub-calls and is passed to handle*Start
        # callbacks (per RunnableConfig docs). This is the most reliable
        # mechanism for agent attribution in nested LangGraph runs.
        config = {
            "run_name": _name,
            "tags": [_name],
            "metadata": {"agent_name": _name},
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
            f"[lazy delegation] Agent {_name} completed. Response length: {len(content)}"
        )

        return content

    return invoke_agent_lazy


def create_bioinformatics_graph(
    data_manager: DataManagerV2,
    checkpointer: InMemorySaver = None,
    store: InMemoryStore = None,
    callback_handler=None,
    manual_model_params: dict = None,
    supervisor_config: Optional[SupervisorConfig] = None,
    subscription_tier: str = None,
    agent_filter: callable = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    workspace_path: Optional[Path] = None,
    config: Optional["WorkspaceAgentConfig"] = None,
    enabled_agents: Optional[List[str]] = None,
) -> Tuple:
    """Create the bioinformatics multi-agent graph using langgraph_supervisor.

    Args:
        data_manager: DataManagerV2 instance for data operations
        checkpointer: Optional memory saver for conversation persistence
        store: Optional in-memory store for shared state
        callback_handler: Optional callback for streaming responses
        manual_model_params: Optional override for supervisor model parameters
        supervisor_config: Optional supervisor configuration
        subscription_tier: Subscription tier for feature gating (free/premium/enterprise).
            If None, will be auto-detected from license. Controls which agents are
            available and which handoffs are allowed.
        agent_filter: Optional callable(agent_name, agent_config) -> bool to filter
            which agents are included in the graph. Used for tier-based restrictions.
        provider_override: Optional explicit provider name (e.g., "bedrock", "anthropic", "ollama")
        model_override: Optional explicit model name (e.g., "llama3:70b-instruct", "claude-4-sonnet")
        workspace_path: Optional path to workspace directory
        config: Optional WorkspaceAgentConfig for agent filtering via TOML config.
            If provided with enabled_agents, this determines which agents to load.
        enabled_agents: Optional list of agent names to enable. Takes precedence
            over config.enabled_agents if both are provided.

    Returns:
        Tuple of (compiled_graph, GraphMetadata):
            - compiled_graph: The compiled LangGraph ready for invocation
            - GraphMetadata: Metadata about available agents, tier, hierarchy

    Note: When invoking this graph, set the recursion_limit in the config to prevent
    hitting the default limit of 25. Example:
        config = {"recursion_limit": 100, ...}
        graph.invoke(input, config)
    """
    # Auto-detect subscription tier if not provided
    if subscription_tier is None:
        try:
            from lobster.core.license_manager import get_current_tier

            subscription_tier = get_current_tier()
        except ImportError:
            subscription_tier = "free"
    logger.debug(f"Creating graph with subscription tier: {subscription_tier}")

    # Create tier-based agent filter if not provided
    if agent_filter is None:
        from lobster.config.subscription_tiers import is_agent_available

        def agent_filter(name, config):
            return is_agent_available(name, subscription_tier)
    logger.debug("Creating bioinformatics multi-agent graph")

    # Get model configuration for the supervisor
    settings = get_settings()

    # ensure this for later
    if manual_model_params:
        # Use provided manual model parameters if available
        model_params = manual_model_params
    else:
        model_params = settings.get_agent_llm_params("supervisor")

    supervisor_model = create_llm(
        "supervisor",
        model_params,
        provider_override=provider_override,
        model_override=model_override,
        workspace_path=workspace_path,
    )

    # Normalize callbacks to a flat list (fix double-nesting bug)
    # callback_handler may be a single callback, a list of callbacks, or None
    if callback_handler and hasattr(supervisor_model, "with_config"):
        callbacks = (
            callback_handler
            if isinstance(callback_handler, list)
            else [callback_handler]
        )
        supervisor_model = supervisor_model.with_config(callbacks=callbacks)

    # ==========================================================================
    # Agent Discovery: Use ComponentRegistry as single source of truth (GRAPH-05)
    # ==========================================================================
    from lobster.core.component_registry import component_registry

    created_agents: Dict[str, Any] = {}
    agent_tools = []  # Tools for supervisor to invoke sub-agents

    # Get ALL agents from ComponentRegistry (single source of truth)
    all_agents = component_registry.list_agents()

    # Resolve enabled agents: enabled_agents param > config.enabled_agents > all available
    if enabled_agents:
        enabled_set = set(enabled_agents)
        logger.debug(f"Using enabled_agents param: {enabled_set}")
    elif config and config.enabled_agents:
        enabled_set = set(config.enabled_agents)
        logger.debug(f"Using config.enabled_agents: {enabled_set}")
    else:
        enabled_set = None  # All agents enabled
        logger.debug("No enabled_agents filter - using all available agents")

    # Filter to enabled agents
    if enabled_set:
        worker_agents = {n: c for n, c in all_agents.items() if n in enabled_set}
        # Log agents requested but not available
        skipped = enabled_set - set(worker_agents.keys())
        if skipped:
            logger.warning(f"Config requested agents not available: {skipped}")
    else:
        worker_agents = all_agents

    # Apply agent filter for tier-based restrictions
    filtered_worker_agents = {}
    filtered_out_agents = []
    for agent_name, agent_config in worker_agents.items():
        if agent_filter(agent_name, agent_config):
            filtered_worker_agents[agent_name] = agent_config
        else:
            filtered_out_agents.append(agent_name)
    if filtered_out_agents:
        logger.info(
            f"Tier '{subscription_tier}' excludes agents: {filtered_out_agents}"
        )
    worker_agents = filtered_worker_agents

    # Pre-compute child agents for supervisor_accessible inference
    # Agents that appear in ANY parent's child_agents are NOT supervisor-accessible by default
    child_agent_names = set()
    for agent_config in worker_agents.values():
        if agent_config.child_agents:
            child_agent_names.update(agent_config.child_agents)
    if child_agent_names:
        logger.debug(
            f"Child agents (not supervisor-accessible by default): {child_agent_names}"
        )

    # ==========================================================================
    # SINGLE PASS: Create all agents with lazy delegation tools (GRAPH-03)
    # ==========================================================================
    # This eliminates the two-pass pattern where parent agents were created twice
    # (once without delegation tools, then again with them). Lazy delegation tools
    # capture the created_agents dict by reference and resolve agents at invocation time.

    for agent_name, agent_config in worker_agents.items():
        factory_function = import_agent_factory(agent_config.factory_function)

        # Create delegation tools LAZILY (reference dict, not agent instance)
        delegation_tools = None
        if agent_config.child_agents:
            delegation_tools = []
            for child_name in agent_config.child_agents:
                if child_name in worker_agents:  # Only if child is enabled
                    child_config = worker_agents[child_name]
                    delegation_tools.append(
                        _create_lazy_delegation_tool(
                            child_name,
                            created_agents,  # Dict reference - resolved at invocation
                            child_config.description,
                        )
                    )
                else:
                    logger.warning(
                        f"Child '{child_name}' not enabled, skipping delegation tool"
                    )

        # Build kwargs for agent factory (standardized signature)
        factory_kwargs = {
            "data_manager": data_manager,
            "callback_handler": callback_handler,
            "agent_name": agent_config.name,
        }

        # Add optional parameters based on factory signature
        sig = inspect.signature(factory_function)
        if "delegation_tools" in sig.parameters:
            factory_kwargs["delegation_tools"] = delegation_tools
        if "subscription_tier" in sig.parameters:
            factory_kwargs["subscription_tier"] = subscription_tier
        if "provider_override" in sig.parameters:
            factory_kwargs["provider_override"] = provider_override
        if "model_override" in sig.parameters:
            factory_kwargs["model_override"] = model_override
        if "workspace_path" in sig.parameters:
            factory_kwargs["workspace_path"] = workspace_path

        # Create agent ONCE (single pass)
        created_agents[agent_name] = factory_function(**factory_kwargs)
        logger.debug(
            f"Created agent: {agent_config.display_name} ({agent_name}) [single pass]"
        )

    # ==========================================================================
    # Phase 3: Create agent tools for supervisor (Tool Calling pattern)
    # ==========================================================================
    # Only create tools for supervisor-accessible agents (not child agents)
    supervisor_accessible_names = []

    for agent_name, agent_config in worker_agents.items():
        if (
            not agent_config.handoff_tool_name
            or not agent_config.handoff_tool_description
        ):
            continue

        # Determine supervisor accessibility
        if agent_config.supervisor_accessible is None:
            is_supervisor_accessible = agent_name not in child_agent_names
        else:
            is_supervisor_accessible = agent_config.supervisor_accessible

        if is_supervisor_accessible:
            agent_tool = _create_agent_tool(
                agent_name=agent_config.name,
                agent=created_agents[agent_name],
                tool_name=agent_config.handoff_tool_name,
                description=agent_config.handoff_tool_description,
            )
            agent_tools.append(agent_tool)
            supervisor_accessible_names.append(agent_config.name)
            logger.debug(f"Created supervisor tool: {agent_config.handoff_tool_name}")

    # ==========================================================================
    # Phase 4: Create shared tools and supervisor
    # ==========================================================================
    # Create shared tools with data_manager access
    list_available_modalities = create_list_modalities_tool(data_manager)
    get_content_from_workspace = create_get_content_from_workspace_tool(data_manager)
    delete_from_workspace = create_delete_from_workspace_tool(data_manager)

    # Create execute_custom_code tool for supervisor fallback
    # This gives the supervisor a code execution escape hatch for tasks that
    # no domain agent handles (e.g., cross-modal regression, reading adata.uns,
    # loading non-h5ad formats like parquet/CSV directly)
    from lobster.services.execution.custom_code_execution_service import (
        CustomCodeExecutionService,
    )

    supervisor_code_service = CustomCodeExecutionService(data_manager)
    execute_custom_code = create_execute_custom_code_tool(
        data_manager=data_manager,
        custom_code_service=supervisor_code_service,
        agent_name="supervisor",
        post_processor=None,
    )

    logger.debug(f"Supervisor-accessible agents: {supervisor_accessible_names}")
    logger.debug(
        f"Total agents created: {len(created_agents)}, "
        f"Supervisor tools: {len(agent_tools)}"
    )

    # Create supervisor prompt with active agents list
    system_prompt = create_supervisor_prompt(
        data_manager=data_manager,
        config=supervisor_config,
        active_agents=supervisor_accessible_names,
    )

    # Create todo tools for planning
    write_todos, read_todos = create_todo_tools()

    # Combine all tools for the supervisor
    all_supervisor_tools = agent_tools + [  # Tools to invoke sub-agents
        list_available_modalities,
        get_content_from_workspace,
        delete_from_workspace,
        write_todos,  # Planning tools
        read_todos,
        execute_custom_code,  # Fallback code execution
    ]

    # ==========================================================================
    # Create supervisor using simple Tool Calling pattern
    # ==========================================================================
    # This is much simpler than the Handoffs pattern:
    # - Supervisor is a ReAct agent with tools that invoke sub-agents
    # - No graph-based routing, no Command/Send complexity
    # - Sub-agents are invoked directly and return results
    # - User only ever interacts with supervisor
    #
    # See: https://docs.langchain.com/oss/langchain/multi-agent#tool-calling

    # Create the supervisor as a ReAct agent
    supervisor_agent = create_react_agent(
        model=supervisor_model,
        tools=all_supervisor_tools,
        prompt=system_prompt,
        state_schema=OverallState,
    )

    # Wrap in a StateGraph with explicit "supervisor" node name
    # This ensures events are keyed by "supervisor" (backward compatible with client)
    workflow = StateGraph(OverallState)
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("supervisor", END)

    # Compile with checkpointer and store
    graph = workflow.compile(checkpointer=checkpointer, store=store)

    # ==========================================================================
    # Phase 5: Build metadata for API exposure
    # ==========================================================================
    available_agent_infos = []
    for agent_name, agent_config in worker_agents.items():
        agent_info = AgentInfo(
            name=agent_config.name,
            display_name=agent_config.display_name,
            description=agent_config.description,
            is_supervisor_accessible=agent_config.name in supervisor_accessible_names,
            parent_agent=_get_parent_agent(agent_config.name, worker_agents),
            child_agents=agent_config.child_agents,
            handoff_tool_name=agent_config.handoff_tool_name,
        )
        available_agent_infos.append(agent_info)

    metadata = GraphMetadata(
        subscription_tier=subscription_tier,
        available_agents=available_agent_infos,
        supervisor_accessible_agents=supervisor_accessible_names,
        filtered_out_agents=filtered_out_agents,
    )

    logger.debug(
        "Bioinformatics multi-agent graph created successfully (Tool Calling pattern)"
    )
    logger.debug(
        f"Graph metadata: {len(available_agent_infos)} agents, "
        f"{len(supervisor_accessible_names)} supervisor-accessible"
    )
    return graph, metadata
