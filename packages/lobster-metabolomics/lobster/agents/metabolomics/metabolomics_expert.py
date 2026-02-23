"""
Metabolomics Expert Agent for untargeted metabolomics analysis (LC-MS, GC-MS, NMR).

This agent handles quality assessment, preprocessing, multivariate statistics,
metabolite annotation, lipid classification, and pathway enrichment for
untargeted metabolomics data.

The agent auto-detects platform type and applies appropriate defaults.
"""

# Agent configuration for entry point discovery (must be at top, before heavy imports)
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="metabolomics_expert",
    display_name="Metabolomics Expert",
    description="Metabolomics analysis: LC-MS/GC-MS/NMR QC, preprocessing, multivariate statistics, metabolite annotation",
    factory_function="lobster.agents.metabolomics.metabolomics_expert.metabolomics_expert",
    handoff_tool_name="handoff_to_metabolomics_expert",
    handoff_tool_description="Assign metabolomics analysis tasks: LC-MS/GC-MS/NMR quality assessment, normalization (PQN/TIC), univariate/multivariate statistics (PCA/PLS-DA/OPLS-DA), metabolite annotation, lipid class analysis, pathway enrichment",
    child_agents=None,  # No children in v1 (structured for future extraction)
    supervisor_accessible=True,
    tier_requirement="free",
)

# === Heavy imports below ===
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.metabolomics.shared_tools import create_shared_tools
from lobster.agents.metabolomics.state import MetabolomicsExpertState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.analysis.metabolomics_analysis_service import (
    MetabolomicsAnalysisService,
)
from lobster.services.annotation.metabolomics_annotation_service import (
    MetabolomicsAnnotationService,
)
from lobster.services.quality.metabolomics_preprocessing_service import (
    MetabolomicsPreprocessingService,
)
from lobster.services.quality.metabolomics_quality_service import (
    MetabolomicsQualityService,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class MetabolomicsAgentError(Exception):
    """Base exception for metabolomics agent operations."""

    pass


class ModalityNotFoundError(MetabolomicsAgentError):
    """Raised when requested modality doesn't exist."""

    pass


def metabolomics_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "metabolomics_expert",
    delegation_tools: list = None,
    force_platform_type: Optional[str] = None,
    workspace_path: Optional[Path] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
):
    """
    Factory function for metabolomics expert agent.

    This agent handles LC-MS, GC-MS, and NMR untargeted metabolomics analysis.
    It auto-detects platform type from data characteristics and applies
    appropriate defaults for preprocessing, normalization, and analysis.

    Args:
        data_manager: DataManagerV2 instance for modality management
        callback_handler: Optional callback handler for LLM interactions
        agent_name: Name identifier for the agent instance
        delegation_tools: List of delegation tools (for future sub-agent support)
        force_platform_type: Override auto-detection ("lc_ms", "gc_ms", or "nmr")
        workspace_path: Optional workspace path for LLM operations
        provider_override: Optional LLM provider override
        model_override: Optional model override

    Returns:
        Configured ReAct agent with metabolomics analysis capabilities
    """
    # Lazy prompt import (D17 pattern - allows AGENT_CONFIG discovery before prompt exists)
    from lobster.agents.metabolomics.prompts import create_metabolomics_expert_prompt

    settings = get_settings()
    model_params = settings.get_agent_llm_params("metabolomics_expert")
    llm = create_llm(
        "metabolomics_expert",
        model_params,
        provider_override=provider_override,
        model_override=model_override,
        workspace_path=workspace_path,
    )

    # Normalize callbacks to a flat list (fix double-nesting bug)
    if callback_handler and hasattr(llm, "with_config"):
        callbacks = (
            callback_handler
            if isinstance(callback_handler, list)
            else [callback_handler]
        )
        llm = llm.with_config(callbacks=callbacks)

    # Validate data manager type
    if not isinstance(data_manager, DataManagerV2):
        raise ValueError("MetabolomicsExpert requires DataManagerV2 for modular analysis")

    # Initialize stateless services
    quality_service = MetabolomicsQualityService()
    preprocessing_service = MetabolomicsPreprocessingService()
    analysis_service = MetabolomicsAnalysisService()
    annotation_service = MetabolomicsAnnotationService()

    # =========================================================================
    # GET SHARED TOOLS (10 metabolomics tools)
    # =========================================================================

    shared_tools = create_shared_tools(
        data_manager,
        quality_service,
        preprocessing_service,
        analysis_service,
        annotation_service,
        force_platform_type=force_platform_type,
    )

    # =========================================================================
    # COLLECT ALL TOOLS
    # =========================================================================

    # Combine: shared tools + delegation tools (if any)
    tools = shared_tools
    if delegation_tools:
        tools = tools + delegation_tools

    # Create system prompt
    system_prompt = create_metabolomics_expert_prompt()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=MetabolomicsExpertState,
    )
