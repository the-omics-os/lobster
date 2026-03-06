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


def _invoke_and_store(agent, agent_name: str, task_description: str, store) -> str:
    """Shared invoke pipeline for all delegation tools.

    Constructs config for callback attribution, invokes the agent, extracts
    the final message, and optionally dual-writes the result to store.

    Args:
        agent: The compiled agent (Pregel) to invoke
        agent_name: Agent name for logging and callback attribution
        task_description: Task to send to the agent
        store: Optional InMemoryStore for dual-write result storage

    Returns:
        Agent response content, with [store_key=...] appended if stored
    """
    # Config for proper callback attribution in nested LangGraph runs.
    # metadata propagates to all sub-calls via RunnableConfig.
    config = {
        "run_name": agent_name,
        "tags": [agent_name],
        "metadata": {"agent_name": agent_name},
    }

    result = agent.invoke(
        {"messages": [{"role": "user", "content": task_description}]}, config=config
    )

    # Extract the final message content
    final_msg = result.get("messages", [])[-1] if result.get("messages") else None
    if final_msg is None:
        return f"Agent {agent_name} returned no response."

    content = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
    logger.debug(f"Agent {agent_name} completed. Response length: {len(content)}")

    # Dual-write: store full result for later retrieval
    if store is not None:
        from lobster.tools.store_tools import store_delegation_result

        store_key = store_delegation_result(store, agent_name, content)
        if store_key:
            content = f"{content}\n\n[store_key={store_key}]"

    return content


def _create_agent_tool(
    agent_name: str, agent, tool_name: str, description: str, store=None
):
    """Create a tool that invokes a sub-agent (Tool Calling pattern).

    Used for supervisor → agent handoffs where the agent is already created.

    Args:
        agent_name: Internal name of the agent (for logging)
        agent: The compiled agent (Pregel) to invoke
        tool_name: Name for the tool (e.g., "handoff_to_research_agent")
        description: Description of when to use this tool
        store: Optional InMemoryStore for dual-write result storage
    """
    _store = store

    @tool(tool_name, description=description)
    def invoke_agent(task_description: str) -> str:
        """Invoke a sub-agent with a task description.

        Args:
            task_description: Detailed description of what the agent should do,
                including all relevant context. Should be in task format starting
                with 'Your task is to ...'
        """
        logger.info(
            f"=== HANDOFF TO {agent_name} ===\n{task_description[:500]}\n=== END HANDOFF ==="
        )
        return _invoke_and_store(agent, agent_name, task_description, _store)

    invoke_agent.metadata = {"categories": ["DELEGATE"], "provenance": False}
    invoke_agent.tags = ["DELEGATE"]

    return invoke_agent


def _create_lazy_delegation_tool(
    agent_name: str,
    agents_dict: Dict[str, Any],
    description: str,
    store=None,
):
    """Create delegation tool with lazy agent resolution.

    Used for parent → child handoffs where the child may not exist yet at
    tool creation time. Captures `agents_dict` by reference and resolves
    the agent at invocation time.

    Args:
        agent_name: Name of the child agent to delegate to
        agents_dict: Shared dict that will contain created agents
        description: Description for the tool
        store: Optional InMemoryStore for dual-write result storage

    Returns:
        Tool function with lazy agent resolution
    """
    _name = agent_name
    _dict = agents_dict
    _desc = description
    _store = store

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
            from lobster.core.component_registry import (
                AGENT_TO_PACKAGE,
                get_install_command,
            )

            package = AGENT_TO_PACKAGE.get(_name)
            if package:
                cmd = get_install_command(package)
                logger.warning(
                    f"Agent '{_name}' not available — package '{package}' may not be installed"
                )
                return (
                    f"Agent '{_name}' is not available. "
                    f"It requires the '{package}' package.\n"
                    f"Install with: {cmd}"
                )
            else:
                logger.warning(
                    f"Agent '{_name}' not found in agents dict (config may have excluded it)"
                )
                return (
                    f"Agent '{_name}' is not available. "
                    f"It may be excluded by configuration or not yet supported."
                )

        logger.info(
            f"=== CHILD DELEGATION TO {_name} ===\n{task_description[:500]}\n=== END CHILD DELEGATION ==="
        )
        return _invoke_and_store(agent, _name, task_description, _store)

    invoke_agent_lazy.metadata = {"categories": ["DELEGATE"], "provenance": False}
    invoke_agent_lazy.tags = ["DELEGATE"]

    return invoke_agent_lazy


def _resolve_worker_agents(
    enabled_agents: Optional[List[str]],
    config: Optional["WorkspaceAgentConfig"],
    agent_filter: callable,
    subscription_tier: str,
) -> Tuple[Dict, List[str], set]:
    """Resolve which agents to include in the graph.

    Applies three filters in order:
    1. enabled_agents param > config.enabled_agents > all discovered agents
    2. Auto-include child agents for enabled parents
    3. Tier-based agent_filter

    Args:
        enabled_agents: Explicit agent list (None = defer, [] = zero agents)
        config: Optional WorkspaceAgentConfig
        agent_filter: Callable(name, config) -> bool for tier gating
        subscription_tier: Current tier (for logging only)

    Returns:
        (worker_agents, filtered_out_agents, child_agent_names):
            - worker_agents: Dict of agent configs that passed all filters
            - filtered_out_agents: Names excluded by tier filter
            - child_agent_names: Set of names that are children (not supervisor-accessible)
    """
    from lobster.core.component_registry import component_registry

    all_agents = component_registry.list_agents()

    # Resolve enabled agents: param > config > all
    # `is not None` (not truthiness) so enabled_agents=[] means "zero agents"
    # config.enabled_agents uses truthiness because [] is its default (= no preference)
    if enabled_agents is not None:
        enabled_set = set(enabled_agents)
        logger.debug(f"Using enabled_agents param: {enabled_set}")
    elif config and config.enabled_agents:
        enabled_set = set(config.enabled_agents)
        logger.debug(f"Using config.enabled_agents: {enabled_set}")
    else:
        enabled_set = None  # All agents enabled
        logger.debug("No enabled_agents filter - using all available agents")

    # Filter to enabled agents
    if enabled_set is not None:
        worker_agents = {n: c for n, c in all_agents.items() if n in enabled_set}
        # Auto-include child agents for enabled parents
        child_additions = {}
        for agent_name, agent_config in list(worker_agents.items()):
            if agent_config.child_agents:
                for child_name in agent_config.child_agents:
                    if child_name not in worker_agents and child_name in all_agents:
                        child_additions[child_name] = all_agents[child_name]
        if child_additions:
            worker_agents.update(child_additions)
            logger.info(
                f"Auto-included child agents for enabled parents: {list(child_additions.keys())}"
            )
        skipped = enabled_set - set(worker_agents.keys())
        if skipped:
            logger.debug(f"Config requested agents not available: {skipped}")
    else:
        worker_agents = all_agents

    # Apply tier-based restrictions
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
    child_agent_names = set()
    for agent_config in worker_agents.values():
        if agent_config.child_agents:
            child_agent_names.update(agent_config.child_agents)
    if child_agent_names:
        logger.debug(
            f"Child agents (not supervisor-accessible by default): {child_agent_names}"
        )

    return worker_agents, filtered_out_agents, child_agent_names


def _create_agents_single_pass(
    worker_agents: Dict,
    created_agents: Dict[str, Any],
    data_manager: DataManagerV2,
    callback_handler,
    store,
    subscription_tier: str,
    provider_override: Optional[str],
    model_override: Optional[str],
    workspace_path: Optional[Path],
) -> Dict:
    """Create all agents in a single pass with lazy delegation tools.

    IMPORTANT: `created_agents` is mutated in-place. Lazy delegation tools
    capture this dict by reference and resolve agents at invocation time.
    Do NOT replace the dict object — only add to it.

    Args:
        worker_agents: Dict of agent configs to create (from _resolve_worker_agents)
        created_agents: Empty dict to populate with compiled agents (mutated in-place)
        data_manager: DataManagerV2 for agent factories
        callback_handler: Callback for streaming
        store: Optional InMemoryStore for dual-write
        subscription_tier: For factories that need it
        provider_override: For factories that need it
        model_override: For factories that need it
        workspace_path: For factories that need it

    Returns:
        Pruned worker_agents dict (failed agents removed)
    """
    failed_agents = set()

    for agent_name, agent_config in worker_agents.items():
        try:
            factory_function = import_agent_factory(agent_config.factory_function)
        except (ImportError, ModuleNotFoundError, AttributeError, SyntaxError) as e:
            from lobster.core.component_registry import AGENT_TO_PACKAGE

            package = AGENT_TO_PACKAGE.get(agent_name)
            if package:
                logger.warning(
                    f"Skipping agent '{agent_name}' (package '{package}'): {e}"
                )
            else:
                logger.warning(
                    f"Skipping agent '{agent_name}': factory import failed: {e}"
                )
            failed_agents.add(agent_name)
            continue

        # Create delegation tools LAZILY (reference dict, not agent instance)
        delegation_tools = None
        if agent_config.child_agents:
            delegation_tools = []
            for child_name in agent_config.child_agents:
                if child_name in worker_agents and child_name not in failed_agents:
                    child_config = worker_agents[child_name]
                    delegation_tools.append(
                        _create_lazy_delegation_tool(
                            child_name,
                            created_agents,  # Dict reference — resolved at invocation
                            child_config.description,
                            store=store,
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
        if "store" in sig.parameters:
            factory_kwargs["store"] = store

        try:
            created_agents[agent_name] = factory_function(**factory_kwargs).with_config(
                {"recursion_limit": 50}
            )
        except Exception as e:
            logger.warning(
                f"Skipping agent '{agent_name}': factory execution failed: {e}"
            )
            failed_agents.add(agent_name)
            continue

        logger.debug(
            f"Created agent: {agent_config.display_name} ({agent_name}) [single pass]"
        )

    # Remove failed agents so downstream phases are consistent
    if failed_agents:
        worker_agents = {
            k: v for k, v in worker_agents.items() if k not in failed_agents
        }
        from lobster.core.component_registry import (
            AGENT_TO_PACKAGE,
            get_install_command,
        )

        missing_packages = {}
        for agent in sorted(failed_agents):
            pkg = AGENT_TO_PACKAGE.get(agent)
            if pkg:
                missing_packages.setdefault(pkg, []).append(agent)

        if missing_packages:
            install_hints = []
            for pkg, agents in sorted(missing_packages.items()):
                cmd = get_install_command(pkg)
                install_hints.append(f"  {pkg} ({', '.join(agents)}): {cmd}")
            hints_str = "\n".join(install_hints)
            logger.warning(
                f"Agents failed to load ({len(failed_agents)}): {sorted(failed_agents)}.\n"
                f"Install missing packages:\n{hints_str}\n"
                f"Graph continues with {len(worker_agents)} agents."
            )
        else:
            logger.warning(
                f"Agents failed to load ({len(failed_agents)}): {sorted(failed_agents)}. "
                f"Graph continues with {len(worker_agents)} agents."
            )

    return worker_agents


def _build_supervisor_tools(
    worker_agents: Dict,
    created_agents: Dict[str, Any],
    child_agent_names: set,
    data_manager: DataManagerV2,
    store,
) -> Tuple[List, List[str]]:
    """Build all supervisor tools: handoff, workspace, code execution, and todo.

    Args:
        worker_agents: Dict of agent configs (after filtering)
        created_agents: Dict of compiled agents (for handoff tool creation)
        child_agent_names: Set of agent names that are children (not supervisor-accessible)
        data_manager: DataManagerV2 for workspace/code tools
        store: Optional InMemoryStore for dual-write

    Returns:
        (all_supervisor_tools, supervisor_accessible_names):
            - all_supervisor_tools: Full list of tools for the supervisor
            - supervisor_accessible_names: Names of agents the supervisor can handoff to
    """
    agent_tools = []
    supervisor_accessible_names = []

    for agent_name, agent_config in worker_agents.items():
        if not agent_config.handoff_tool_name:
            continue

        if agent_config.supervisor_accessible is None:
            is_supervisor_accessible = agent_name not in child_agent_names
        else:
            is_supervisor_accessible = agent_config.supervisor_accessible

        if is_supervisor_accessible:
            # Use agent's own handoff_tool_description for routing signal.
            # Small models (8B-20B) rely on tool descriptions for selection —
            # they cannot cross-reference a separate Agent Directory block.
            # Rich descriptions with domain keywords enable pattern matching.
            desc = agent_config.handoff_tool_description or (
                f"Delegate task to {agent_config.display_name}."
            )
            agent_tool = _create_agent_tool(
                agent_name=agent_config.name,
                agent=created_agents[agent_name],
                tool_name=agent_config.handoff_tool_name,
                description=desc,
                store=store,
            )
            agent_tools.append(agent_tool)
            supervisor_accessible_names.append(agent_config.name)
            logger.debug(f"Created supervisor tool: {agent_config.handoff_tool_name}")

    # Shared workspace tools
    list_available_modalities = create_list_modalities_tool(data_manager)
    get_content_from_workspace = create_get_content_from_workspace_tool(data_manager)
    delete_from_workspace = create_delete_from_workspace_tool(data_manager)

    # Code execution fallback
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

    # Todo tools for planning
    write_todos, read_todos = create_todo_tools()

    from lobster.tools.user_interaction import ask_user

    all_supervisor_tools = agent_tools + [
        list_available_modalities,
        get_content_from_workspace,
        delete_from_workspace,
        write_todos,
        read_todos,
        execute_custom_code,
        ask_user,
    ]

    logger.debug(f"Supervisor-accessible agents: {supervisor_accessible_names}")
    logger.debug(
        f"Total agents created: {len(created_agents)}, "
        f"Supervisor tools: {len(agent_tools)}"
    )

    return all_supervisor_tools, supervisor_accessible_names


def _build_graph_metadata(
    worker_agents: Dict,
    supervisor_accessible_names: List[str],
    filtered_out_agents: List[str],
    subscription_tier: str,
) -> GraphMetadata:
    """Build GraphMetadata for API exposure.

    Pure function — reads from worker_agents and supervisor_accessible_names,
    produces a serializable GraphMetadata dataclass.
    """
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
        f"Graph metadata: {len(available_agent_infos)} agents, "
        f"{len(supervisor_accessible_names)} supervisor-accessible"
    )

    return metadata


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
    aquadif_monitor=None,
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

    # Agent discovery, filtering, and tier gating
    worker_agents, filtered_out_agents, child_agent_names = _resolve_worker_agents(
        enabled_agents=enabled_agents,
        config=config,
        agent_filter=agent_filter,
        subscription_tier=subscription_tier,
    )

    # Create all agents in a single pass with lazy delegation tools
    created_agents: Dict[str, Any] = {}
    worker_agents = _create_agents_single_pass(
        worker_agents=worker_agents,
        created_agents=created_agents,  # Mutated in-place (lazy tools capture by reference)
        data_manager=data_manager,
        callback_handler=callback_handler,
        store=store,
        subscription_tier=subscription_tier,
        provider_override=provider_override,
        model_override=model_override,
        workspace_path=workspace_path,
    )

    # Assemble all supervisor tools (handoff + workspace + code + todo)
    all_supervisor_tools, supervisor_accessible_names = _build_supervisor_tools(
        worker_agents=worker_agents,
        created_agents=created_agents,
        child_agent_names=child_agent_names,
        data_manager=data_manager,
        store=store,
    )

    # Create supervisor prompt with active agents list
    system_prompt = create_supervisor_prompt(
        data_manager=data_manager,
        config=supervisor_config,
        active_agents=supervisor_accessible_names,
    )

    # ==========================================================================
    # Context Management: pre_model_hook + retrieve_agent_result
    # ==========================================================================
    pre_model_hook = None
    if store is not None:
        from lobster.agents.context_management import (
            create_supervisor_pre_model_hook,
            resolve_context_budget,
            resolve_context_window,
        )
        from lobster.tools.store_tools import create_retrieve_agent_result_tool

        # Add retrieval tool to supervisor
        retrieve_tool = create_retrieve_agent_result_tool(store)
        all_supervisor_tools.append(retrieve_tool)

        # Resolve context budget from provider config
        context_window = resolve_context_window(
            provider_override=provider_override,
            model_override=model_override,
            workspace_path=workspace_path,
        )
        budget = resolve_context_budget(
            context_window=context_window,
            tools=all_supervisor_tools,
        )
        pre_model_hook = create_supervisor_pre_model_hook(max_tokens=budget)

        logger.info(
            f"Context management enabled: budget={budget} tokens"
            + (f" (window={context_window})" if context_window else " (default window)")
        )

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
        pre_model_hook=pre_model_hook,
        store=store,
    )

    # Wrap in a StateGraph with explicit "supervisor" node name
    # This ensures events are keyed by "supervisor" (backward compatible with client)
    workflow = StateGraph(OverallState)
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("supervisor", END)

    # Compile with checkpointer and store
    graph = workflow.compile(checkpointer=checkpointer, store=store)

    # Build metadata for API exposure
    metadata = _build_graph_metadata(
        worker_agents=worker_agents,
        supervisor_accessible_names=supervisor_accessible_names,
        filtered_out_agents=filtered_out_agents,
        subscription_tier=subscription_tier,
    )

    # ==========================================================================
    # AQUADIF Monitoring: build tool_metadata_map and wire monitor
    # ==========================================================================
    if aquadif_monitor is not None:
        tool_metadata_map: Dict[str, Any] = {}

        # Collect from ALL supervisor tools (single source of truth)
        # This covers agent handoff tools, workspace tools, todo tools,
        # retrieve_agent_result, and execute_custom_code.
        for t in all_supervisor_tools:
            if hasattr(t, "metadata"):
                tool_metadata_map[t.name] = t.metadata

        # Collect from all child agent tools (PregelNode traversal)
        for agent_name, agent in created_agents.items():
            try:
                tools_node = agent.nodes.get("tools")
                if tools_node and hasattr(tools_node, "runnable"):
                    for t in tools_node.runnable.tools or []:
                        if hasattr(t, "metadata"):
                            tool_metadata_map[t.name] = t.metadata
            except Exception:
                pass  # Fail-open: skip agents whose tools can't be extracted

        # Populate monitor's tool map (monitor already attached to callback by reference)
        aquadif_monitor._tool_metadata_map = tool_metadata_map

        # Wire monitor to DataManagerV2 (setattr avoids changing DM constructor)
        data_manager._aquadif_monitor = aquadif_monitor

        logger.debug(f"AQUADIF monitor: {len(tool_metadata_map)} tools mapped")

    return graph, metadata
