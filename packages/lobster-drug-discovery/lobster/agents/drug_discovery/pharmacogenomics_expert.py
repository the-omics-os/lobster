"""
Pharmacogenomics Expert child agent for drug-gene-variant analysis.

This child agent handles all pharmacogenomics tasks delegated by the
drug_discovery_expert parent agent:
- Protein mutation effect prediction using ESM2 fill-mask scoring
- Protein embedding extraction from ESM2
- Wild-type vs mutant sequence comparison
- Drug-variant interaction lookup from Open Targets
- Pharmacogenomic evidence retrieval from ChEMBL
- Variant impact scoring with combined clinical and drug context
- Expression-drug sensitivity correlation analysis
- Mutation frequency and co-occurrence pattern analysis

PLM tools require the [plm] extra (transformers + torch) and degrade
gracefully when dependencies are unavailable.

The agent is NOT directly accessible by the supervisor -- it receives tasks
only from the drug_discovery_expert parent agent via delegation tools.
"""

# Agent configuration for entry point discovery (must be at top, before heavy imports)
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="pharmacogenomics_expert",
    display_name="Pharmacogenomics Expert",
    description=(
        "Pharmacogenomics: protein language model mutation prediction (ESM2), "
        "variant-drug interactions, pharmacogenomic evidence, variant impact scoring"
    ),
    factory_function="lobster.agents.drug_discovery.pharmacogenomics_expert.pharmacogenomics_expert",
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


def pharmacogenomics_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "pharmacogenomics_expert",
    delegation_tools: list = None,
    workspace_path: Optional[Path] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
):
    """
    Factory function for the pharmacogenomics expert child agent.

    This agent handles drug-gene-variant interaction analysis delegated
    by the drug_discovery_expert parent agent, including ESM2 mutation
    prediction, protein embedding extraction, variant impact scoring,
    and pharmacogenomic evidence retrieval.

    Args:
        data_manager: DataManagerV2 instance for modality management.
        callback_handler: Optional callback handler for LLM interactions.
        agent_name: Name identifier for the agent instance.
        delegation_tools: Not used (child agent has no grandchildren).
        workspace_path: Optional workspace path for LLM operations.
        provider_override: Optional LLM provider override.
        model_override: Optional model override.

    Returns:
        Configured ReAct agent with pharmacogenomics capabilities.
    """
    # Lazy prompt import (D17 pattern)
    from lobster.agents.drug_discovery.prompts import (
        create_pharmacogenomics_expert_prompt,
    )
    from lobster.agents.drug_discovery.state import PharmacogenomicsExpertState

    # LLM creation
    settings = get_settings()
    model_params = settings.get_agent_llm_params("pharmacogenomics_expert")
    llm = create_llm(
        "pharmacogenomics_expert",
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
            "pharmacogenomics_expert requires DataManagerV2 for modular analysis"
        )

    # Initialize stateless services
    from lobster.services.drug_discovery.chembl_service import ChEMBLService
    from lobster.services.drug_discovery.opentargets_service import (
        OpenTargetsService,
    )

    opentargets_service = OpenTargetsService()
    chembl_service = ChEMBLService()

    # Create tools via tool factory
    from lobster.agents.drug_discovery.pharmacogenomics_tools import (
        create_pharmacogenomics_tools,
    )

    tools = create_pharmacogenomics_tools(
        data_manager=data_manager,
        opentargets_service=opentargets_service,
        chembl_service=chembl_service,
    )

    # Build prompt and create agent
    system_prompt = create_pharmacogenomics_expert_prompt()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=PharmacogenomicsExpertState,
    )
