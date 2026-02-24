"""
Drug Discovery Expert Parent Agent for target identification, compound profiling,
and drug development orchestration.

This agent serves as the main orchestrator for drug discovery analysis, with:
- 10 shared tools (from shared_tools.py) for target search/scoring and compound lookup
- Delegation to cheminformatics_expert for molecular analysis (RDKit, ADMET, fingerprints)
- Delegation to clinical_dev_expert for clinical translation (synergy, safety, tractability)
- Delegation to pharmacogenomics_expert for drug-gene-variant interactions (ESM2, PGx)

The agent queries ChEMBL, Open Targets, and PubChem APIs and delegates
specialized computational chemistry, clinical, and pharmacogenomics workflows
to its child agents.
"""

# Agent configuration for entry point discovery (must be at top, before heavy imports)
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="drug_discovery_expert",
    display_name="Drug Discovery Expert",
    description="Drug target identification, compound profiling, and drug development orchestration",
    factory_function="lobster.agents.drug_discovery.drug_discovery_expert.drug_discovery_expert",
    handoff_tool_name="handoff_to_drug_discovery_expert",
    handoff_tool_description=(
        "Assign drug discovery tasks: target identification and scoring, "
        "compound search and bioactivity lookup (ChEMBL/PubChem), drug indication "
        "mapping, target-disease evidence (Open Targets), drug combination synergy "
        "scoring (Bliss/Loewe/HSA models), molecular descriptor calculation, "
        "Lipinski Rule of 5 assessment, ADMET prediction, and fingerprint "
        "similarity analysis. ALWAYS delegate here for any drug synergy computation, "
        "molecular property calculation, or target druggability scoring — even if "
        "the math seems simple — because these tools log provenance."
    ),
    child_agents=["cheminformatics_expert", "clinical_dev_expert", "pharmacogenomics_expert"],
    supervisor_accessible=True,
    tier_requirement="free",
)

# === Heavy imports below ===
from pathlib import Path
from typing import Optional

from langgraph.prebuilt import create_react_agent

from lobster.agents.drug_discovery.shared_tools import create_shared_tools
from lobster.agents.drug_discovery.state import DrugDiscoveryExpertState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.drug_discovery.chembl_service import ChEMBLService
from lobster.services.drug_discovery.opentargets_service import OpenTargetsService
from lobster.services.drug_discovery.pubchem_service import PubChemService
from lobster.services.drug_discovery.target_scoring_service import TargetScoringService
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class DrugDiscoveryAgentError(Exception):
    """Base exception for drug discovery agent operations."""

    pass


def drug_discovery_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "drug_discovery_expert",
    delegation_tools: list = None,
    workspace_path: Optional[Path] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
):
    """
    Factory function for drug discovery expert parent agent.

    This agent handles target identification, compound profiling, and drug
    development orchestration. It delegates specialized analysis to three
    child agents: cheminformatics_expert, clinical_dev_expert, and
    pharmacogenomics_expert.

    Args:
        data_manager: DataManagerV2 instance for modality management.
        callback_handler: Optional callback handler for LLM interactions.
        agent_name: Name identifier for the agent instance.
        delegation_tools: List of delegation tools for child agents
            (cheminformatics_expert, clinical_dev_expert, pharmacogenomics_expert).
        workspace_path: Optional workspace path for LLM operations.
        provider_override: Optional LLM provider override.
        model_override: Optional model override.

    Returns:
        Configured ReAct agent with drug discovery capabilities.
    """
    # Lazy import of prompts (D17 pattern -- allows AGENT_CONFIG discovery
    # before prompts.py exists during incremental development)
    from lobster.agents.drug_discovery.prompts import (
        create_drug_discovery_expert_prompt,
    )

    # Create LLM
    settings = get_settings()
    model_params = settings.get_agent_llm_params("drug_discovery_expert")
    llm = create_llm(
        "drug_discovery_expert",
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
        raise ValueError(
            "DrugDiscoveryExpert requires DataManagerV2 for modular analysis"
        )

    # Initialize stateless services
    chembl_service = ChEMBLService()
    opentargets_service = OpenTargetsService()
    pubchem_service = PubChemService()
    target_scoring_service = TargetScoringService()

    # =========================================================================
    # GET SHARED TOOLS (10 tools: target search/score/rank, compound search,
    # bioactivity, target compounds, properties, indications, status, databases)
    # =========================================================================

    shared_tools = create_shared_tools(
        data_manager,
        chembl_service,
        opentargets_service,
        pubchem_service,
        target_scoring_service,
    )

    # =========================================================================
    # COLLECT ALL TOOLS
    # =========================================================================

    # Combine: shared tools (10 direct tools for parent)
    all_tools = shared_tools

    # Add delegation tools if provided (cheminformatics, clinical_dev, pharmacogenomics)
    if delegation_tools:
        all_tools = all_tools + delegation_tools

    # Create system prompt
    system_prompt = create_drug_discovery_expert_prompt()

    return create_react_agent(
        model=llm,
        tools=all_tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=DrugDiscoveryExpertState,
    )
