"""
Clinical Development Expert child agent for clinical translation and synergy.

This child agent handles all clinical development tasks delegated by the
drug_discovery_expert parent agent:
- Target-disease evidence retrieval from Open Targets
- Drug combination synergy scoring (Bliss Independence, Loewe Additivity, HSA)
- Full dose-response combination matrix analysis
- Drug safety profiling and adverse event assessment
- Target tractability assessment (small molecule, antibody, PROTAC)
- Clinical trial phase data lookup from ChEMBL
- Indication mapping from Open Targets
- Side-by-side drug candidate comparison

The agent is NOT directly accessible by the supervisor -- it receives tasks
only from the drug_discovery_expert parent agent via delegation tools.
"""

# Agent configuration for entry point discovery (must be at top, before heavy imports)
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="clinical_dev_expert",
    display_name="Clinical Development Expert",
    description=(
        "Clinical development: target-disease evidence, drug synergy scoring "
        "(Bliss/Loewe/HSA), safety profiling, tractability assessment, "
        "indication mapping"
    ),
    factory_function="lobster.agents.drug_discovery.clinical_dev_expert.clinical_dev_expert",
    handoff_tool_name=None,
    handoff_tool_description=None,
    supervisor_accessible=False,
    tier_requirement="free",
)

# === Heavy imports below ===
from pathlib import Path
from typing import Optional

from langgraph.prebuilt import create_react_agent

from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def clinical_dev_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "clinical_dev_expert",
    delegation_tools: list = None,
    workspace_path: Optional[Path] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
):
    """
    Factory function for the clinical development expert child agent.

    This agent handles clinical translation tasks delegated by the
    drug_discovery_expert parent agent, including target-disease evidence,
    drug synergy scoring, safety profiling, and tractability assessment.

    Args:
        data_manager: DataManagerV2 instance for modality management.
        callback_handler: Optional callback handler for LLM interactions.
        agent_name: Name identifier for the agent instance.
        delegation_tools: Not used (child agent has no grandchildren).
        workspace_path: Optional workspace path for LLM operations.
        provider_override: Optional LLM provider override.
        model_override: Optional model override.

    Returns:
        Configured ReAct agent with clinical development capabilities.
    """
    # Lazy prompt import (D17 pattern)
    from lobster.agents.drug_discovery.prompts import (
        create_clinical_dev_expert_prompt,
    )
    from lobster.agents.drug_discovery.state import ClinicalDevExpertState

    # LLM creation
    settings = get_settings()
    model_params = settings.get_agent_llm_params("clinical_dev_expert")
    llm = create_llm(
        "clinical_dev_expert",
        model_params,
        provider_override=provider_override,
        model_override=model_override,
        workspace_path=workspace_path,
    )

    # Normalize callbacks to a flat list
    if callback_handler and hasattr(llm, "with_config"):
        callbacks = (
            callback_handler
            if isinstance(callback_handler, list)
            else [callback_handler]
        )
        llm = llm.with_config(callbacks=callbacks)

    # Validate data manager
    if not isinstance(data_manager, DataManagerV2):
        raise ValueError(
            "clinical_dev_expert requires DataManagerV2 for modular analysis"
        )

    # Initialize stateless services
    from lobster.services.drug_discovery.chembl_service import ChEMBLService
    from lobster.services.drug_discovery.opentargets_service import (
        OpenTargetsService,
    )
    from lobster.services.drug_discovery.synergy_scoring_service import (
        SynergyScoringService,
    )
    from lobster.services.drug_discovery.target_scoring_service import (
        TargetScoringService,
    )

    opentargets_service = OpenTargetsService()
    synergy_service = SynergyScoringService()
    chembl_service = ChEMBLService()
    target_scoring_service = TargetScoringService()

    # Create tools via tool factory
    from lobster.agents.drug_discovery.clinical_tools import (
        create_clinical_tools,
    )

    tools = create_clinical_tools(
        data_manager=data_manager,
        opentargets_service=opentargets_service,
        synergy_service=synergy_service,
        chembl_service=chembl_service,
        target_scoring_service=target_scoring_service,
    )

    # Build prompt and create agent
    system_prompt = create_clinical_dev_expert_prompt()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=ClinicalDevExpertState,
    )
