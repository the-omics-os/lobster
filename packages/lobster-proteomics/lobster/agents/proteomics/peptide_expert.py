"""
Peptide Expert — child agent of Proteomics Expert.

Specialist in peptide sequence analysis, property calculation,
activity prediction (AMP/CPP/toxicity), enzymatic digestion, and SAR.
"""

# Agent configuration for entry point discovery (must be at top, before heavy imports)
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="peptide_expert",
    display_name="Peptide Expert",
    description="Peptide sequence analysis: property calculation, activity prediction, enzymatic digestion, variant generation",
    factory_function="lobster.agents.proteomics.peptide_expert.peptide_expert",
    handoff_tool_name="handoff_to_peptide_expert",
    handoff_tool_description="Assign peptide analysis: property calculation (MW, pI, charge, GRAVY), activity prediction (antimicrobial/CPP/toxicity), enzymatic digestion, variant generation, and bioactive peptide annotation",
    supervisor_accessible=False,
    tier_requirement="free",
)

# === Heavy imports below ===
from pathlib import Path
from typing import Optional

from langgraph.prebuilt import create_react_agent

from lobster.agents.proteomics.peptide_prompts import create_peptide_expert_prompt
from lobster.agents.proteomics.peptide_tools import create_peptide_tools
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def peptide_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "peptide_expert",
    delegation_tools: list = None,
    workspace_path: Optional[Path] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
):
    """
    Factory function for peptide expert child agent.

    Args:
        data_manager: DataManagerV2 instance for modality management
        callback_handler: Optional callback handler for LLM interactions
        agent_name: Name identifier for the agent instance
        delegation_tools: List of delegation tools from parent agent
        workspace_path: Optional workspace path for file operations
        provider_override: Optional LLM provider override
        model_override: Optional model override

    Returns:
        Configured ReAct agent with peptide analysis capabilities
    """
    settings = get_settings()
    model_params = settings.get_agent_llm_params("peptide_expert")
    llm = create_llm(
        "peptide_expert",
        model_params,
        provider_override=provider_override,
        model_override=model_override,
        workspace_path=workspace_path,
    )

    # Normalize callbacks
    if callback_handler and hasattr(llm, "with_config"):
        callbacks = (
            callback_handler
            if isinstance(callback_handler, list)
            else [callback_handler]
        )
        llm = llm.with_config(callbacks=callbacks)

    # Create domain tools (8 tools)
    domain_tools = create_peptide_tools(
        data_manager=data_manager,
        workspace_path=workspace_path,
    )

    # Add delegation tools from parent if provided
    tools = domain_tools
    if delegation_tools:
        tools = tools + delegation_tools

    # System prompt
    system_prompt = create_peptide_expert_prompt()

    # Import state
    from lobster.agents.proteomics.state import PeptideExpertState

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=PeptideExpertState,
    )
