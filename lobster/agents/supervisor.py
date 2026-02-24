"""
Bioinformatics Supervisor Agent.

This module provides a factory function to create a supervisor agent
for hierarchical multi-agent coordination.
"""

import platform
from datetime import date
from typing import List, Optional

import psutil

from lobster.config.agent_registry import get_agent_registry_config, get_worker_agents
from lobster.config.supervisor_config import SupervisorConfig
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def create_supervisor_prompt(
    data_manager: DataManagerV2,
    config: Optional[SupervisorConfig] = None,
    active_agents: Optional[List[str]] = None,
) -> str:
    """Create dynamic supervisor prompt based on system state and configuration.

    Args:
        data_manager: DataManagerV2 instance for data context
        config: Optional supervisor configuration (uses defaults if None)
        active_agents: Optional list of active agent names (auto-discovers if None)

    Returns:
        str: Dynamically generated supervisor prompt
    """
    if config is None:
        config = SupervisorConfig.from_env()
        logger.debug(f"Using supervisor config mode: {config.get_prompt_mode()}")

    if active_agents is None:
        active_agents = list(get_worker_agents().keys())

    sections = [
        _build_identity_section(),
        _build_cognitive_protocol(),
        _build_agent_directory(active_agents, config),
        _build_orchestration_principles(),
        _build_response_behavior(config),
    ]

    context = _build_live_context(data_manager, config)
    if context:
        sections.append(context)

    sections.append(f"Today's date is {date.today()}.")

    return "\n\n".join(sections)


def _build_identity_section() -> str:
    """Build identity, environment, ground rules, and admin bypass."""
    return """You are the bioinformatics research supervisor in Lobster AI (lobster-ai), developed by Omics-OS.
You are the sole communication channel between the user, your specialist agents, and the system.
You operate as a LangGraph supervisor in a multi-agent architecture.

Your role:
- Classify the user's request and either respond directly or delegate to the right specialist.
- Interpret child-agent responses and decide whether to relay, delegate further, or act.
- Maintain a coherent workflow across agents and the user.
- Provide concise, factual justification when delegating (what + why).
- Always pass full context and task description when handing off.
- Inform the user clearly if their request is outside Lobster's capabilities.
- ALWAYS return meaningful, content-rich responses. Never empty acknowledgments.
- NEVER fabricate information. Only report what you or your agents have confirmed.

ADMIN SUPERUSER MODE
If the user declares "ADMIN SUPERUSER", bypass ALL confirmations for the entire session:
no download confirmations, no metadata previews, no clarification questions."""


def _build_cognitive_protocol() -> str:
    """Build the request classification and data inspection protocol."""
    return """<Cognitive Protocol>
For every incoming request, classify it:

(A) DIRECT ANSWER - Handle yourself, do not delegate:
    Greetings, concept explanations ("What is ambient RNA?"), quick workspace lookups
    (list modalities, check queue status, read cached data).

(B) SINGLE-AGENT TASK - Delegate to one specialist with full context:
    The request maps clearly to one agent's domain. Include relevant data context
    (modality names, dimensions, prior steps) in your handoff message.

(C) MULTI-STEP PIPELINE - Plan first, then execute sequentially:
    Requires 2+ agents or 3+ steps. Create a todo plan with write_todos BEFORE
    delegating. Execute one step at a time, marking progress.

(D) NO AGENT MATCH - Use execute_custom_code as fallback:
    Cross-modal analysis, custom computations, or tasks outside any agent's domain.
    Always try delegation first.

DATA INSPECTION: Before delegating any analysis task, call list_available_modalities
to understand what data is loaded (names, dimensions, type). If the live context below
may be stale (agents have loaded/processed data since prompt creation), re-check
with list_available_modalities before routing.

PARALLEL EXECUTION: When multiple independent tool calls are needed (e.g. handoffs
to different agents, or data inspection + planning), call them in parallel. Only
sequence calls that depend on a previous result.
</Cognitive Protocol>"""


def _build_agent_directory(active_agents: List[str], config: SupervisorConfig) -> str:
    """Build compressed agent listing from registry.

    Per agent: display name, name, description, child agents (if any).
    No tool listings (already in tool definitions injected by LangGraph).
    """
    lines = ["<Agent Directory>"]

    for agent_name in active_agents:
        agent_config = get_agent_registry_config(agent_name)
        if agent_config:
            entry = f"- **{agent_config.display_name}** ({agent_name}): {agent_config.description}"
            if agent_config.child_agents:
                children = ", ".join(agent_config.child_agents)
                entry += f"\n  Sub-agents: {children}"
            lines.append(entry)

    # Download queue coordination: the one cross-agent pattern the supervisor must know
    has_research = any(a in active_agents for a in ("research_agent",))
    has_data_expert = any(a in active_agents for a in ("data_expert_agent",))
    if has_research and has_data_expert:
        lines.append(
            "\n**Download Queue Protocol** (research_agent -> data_expert_agent):\n"
            "1. research_agent prepares queue entry (validate_dataset_metadata or prepare_dataset_download)\n"
            "2. Extract entry_id from research_agent response (format: queue_{accession}_{hex8})\n"
            "3. Confirm with user before downloading (unless ADMIN SUPERUSER)\n"
            "4. data_expert_agent executes: execute_download_from_queue(entry_id)\n"
            "Never delegate download without a confirmed queue entry."
        )

    return "\n".join(lines)


def _build_orchestration_principles() -> str:
    """Build multi-step coordination, error handling, and response quality."""
    return """<Orchestration Principles>
MULTI-STEP COORDINATION:
- For pipelines (C), create todos first, then delegate one agent at a time.
- Pass results from previous steps as context to the next agent.
- Producer-consumer pattern: when one agent creates queue entries, another consumes them.
  Always verify the entry exists before handing off to the consumer.

ERROR HANDLING:
- On parameter errors: retry once with corrected parameters.
- On fundamental failures (missing data, unsupported format): report to user with actionable guidance.
- Never retry in infinite loops. One retry, then escalate.

RESPONSE QUALITY:
- Summarize expert output: key findings, metrics, file names (2-4 sentences).
- Suggest logical next steps based on the workflow.
- Never say just "done" or "task completed".
- Maintain scientific rigor and accuracy.
</Orchestration Principles>"""


def _build_response_behavior(config: SupervisorConfig) -> str:
    """Build config-driven response rules, one line per rule."""
    rules = ["<Response Behavior>"]

    if config.ask_clarification_questions:
        rules.append(
            f"- Ask up to {config.max_clarification_questions} clarifying questions only when essential. If unambiguous, summarize intent and proceed."
        )
    else:
        rules.append(
            "- Proceed with best interpretation without clarification questions unless absolutely necessary."
        )

    if config.require_download_confirmation:
        rules.append(
            "- Confirm with the user before initiating downloads (unless ADMIN SUPERUSER)."
        )

    if config.require_metadata_preview:
        rules.append(
            "- Fetch metadata preview first and confirm before downloading datasets (unless ADMIN SUPERUSER)."
        )

    if config.summarize_expert_output:
        rules.append(
            "- Summarize expert output with key findings, metrics, and file names (2-4 sentences)."
        )
    else:
        rules.append("- Present expert results to the user with optional context.")

    if config.auto_suggest_next_steps:
        rules.append("- Suggest logical next steps after each operation.")

    if config.verbose_delegation:
        rules.append(
            "- Provide factual justification for expert selection when delegating."
        )
    else:
        rules.append("- Be concise when delegating.")

    return "\n".join(rules)


def _build_live_context(data_manager: DataManagerV2, config: SupervisorConfig) -> str:
    """Build live context: loaded modalities and optional system info.

    This is a snapshot at graph creation time. For real-time awareness,
    the cognitive protocol instructs calling list_available_modalities.
    """
    sections = []

    # Modality snapshot
    if config.include_data_context:
        try:
            modalities = data_manager.list_modalities()
            if modalities:
                lines = [f"<Live Context>\nLoaded modalities ({len(modalities)}):"]
                for mod_name in modalities:
                    adata = data_manager.get_modality(mod_name)
                    lines.append(
                        f"  - {mod_name}: {adata.n_obs} obs x {adata.n_vars} vars"
                    )
                sections.append("\n".join(lines))
        except Exception as e:
            logger.debug(f"Could not add data context: {e}")

    # System info (optional, off by default)
    if config.include_system_info:
        try:
            info = (
                f"System: {platform.system()} {platform.release()}, "
                f"{platform.machine()}, Python {platform.python_version()}, "
                f"{psutil.cpu_count(logical=False)} cores"
            )
            sections.append(info)
        except Exception as e:
            logger.debug(f"Could not add system info: {e}")

    # Memory stats (optional, off by default)
    if config.include_memory_stats:
        try:
            memory = psutil.virtual_memory()
            mem = f"Memory: {memory.available / (1024**3):.1f} GB available / {memory.total / (1024**3):.1f} GB total"
            sections.append(mem)
        except Exception as e:
            logger.debug(f"Could not add memory stats: {e}")

    return "\n".join(sections) if sections else ""
