"""
Cheminformatics Expert child agent for molecular analysis.

This child agent handles all molecular-level computations delegated by the
drug_discovery_expert parent agent:
- Molecular descriptor calculation (MW, LogP, TPSA, HBD, HBA, etc.)
- Lipinski Rule of Five compliance checking
- Fingerprint similarity computation (Morgan/ECFP)
- ADMET property prediction (absorption, distribution, metabolism, excretion, toxicity)
- 3D structure preparation (ETKDG + MMFF94)
- CAS-to-SMILES conversion via PubChem
- PubChem structure similarity search
- Binding site identification from PDB content
- Side-by-side molecule comparison

The agent is NOT directly accessible by the supervisor -- it receives tasks
only from the drug_discovery_expert parent agent via delegation tools.
"""

# Agent configuration for entry point discovery (must be at top, before heavy imports)
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="cheminformatics_expert",
    display_name="Cheminformatics Expert",
    description=(
        "Molecular analysis: descriptors, Lipinski, fingerprint similarity, "
        "ADMET prediction, 3D structure preparation, binding site identification"
    ),
    factory_function="lobster.agents.drug_discovery.cheminformatics_expert.cheminformatics_expert",
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


def cheminformatics_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "cheminformatics_expert",
    delegation_tools: list = None,
    workspace_path: Optional[Path] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
):
    """
    Factory function for the cheminformatics expert child agent.

    This agent handles molecular-level analysis tasks delegated by the
    drug_discovery_expert parent agent, including descriptor calculation,
    drug-likeness checks, similarity analysis, ADMET prediction, and
    3D structure preparation.

    Args:
        data_manager: DataManagerV2 instance for modality management.
        callback_handler: Optional callback handler for LLM interactions.
        agent_name: Name identifier for the agent instance.
        delegation_tools: Not used (child agent has no grandchildren).
        workspace_path: Optional workspace path for LLM operations.
        provider_override: Optional LLM provider override.
        model_override: Optional model override.

    Returns:
        Configured ReAct agent with cheminformatics capabilities.
    """
    # Lazy prompt import (D17 pattern)
    from lobster.agents.drug_discovery.prompts import (
        create_cheminformatics_expert_prompt,
    )
    from lobster.agents.drug_discovery.state import CheminformaticsExpertState

    # LLM creation
    settings = get_settings()
    model_params = settings.get_agent_llm_params("cheminformatics_expert")
    llm = create_llm(
        "cheminformatics_expert",
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
            "cheminformatics_expert requires DataManagerV2 for modular analysis"
        )

    # Initialize stateless services
    from lobster.services.drug_discovery.admet_prediction_service import (
        ADMETPredictionService,
    )
    from lobster.services.drug_discovery.compound_preparation_service import (
        CompoundPreparationService,
    )
    from lobster.services.drug_discovery.molecular_analysis_service import (
        MolecularAnalysisService,
    )
    from lobster.services.drug_discovery.pubchem_service import PubChemService

    molecular_analysis_service = MolecularAnalysisService()
    admet_service = ADMETPredictionService()
    pubchem_service = PubChemService()
    compound_prep_service = CompoundPreparationService()

    # Create tools via tool factory
    from lobster.agents.drug_discovery.cheminformatics_tools import (
        create_cheminformatics_tools,
    )

    tools = create_cheminformatics_tools(
        data_manager=data_manager,
        molecular_analysis_service=molecular_analysis_service,
        admet_service=admet_service,
        pubchem_service=pubchem_service,
        compound_prep_service=compound_prep_service,
    )

    # Build prompt and create agent
    system_prompt = create_cheminformatics_expert_prompt()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=CheminformaticsExpertState,
    )
