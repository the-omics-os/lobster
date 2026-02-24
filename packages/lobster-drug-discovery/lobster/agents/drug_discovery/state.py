"""
State definitions for drug discovery agents (parent + 3 child agents).

This module defines state classes for the drug discovery expert parent agent
and its child agents (cheminformatics, clinical development, pharmacogenomics).

Following the LangGraph 0.2.x multi-agent template pattern, with fields
tailored to drug discovery workflows.
"""

from typing import Any, Dict, List

from langgraph.prebuilt.chat_agent_executor import AgentState

__all__ = [
    "DrugDiscoveryExpertState",
    "CheminformaticsExpertState",
    "ClinicalDevExpertState",
    "PharmacogenomicsExpertState",
]


class DrugDiscoveryExpertState(AgentState):
    """
    State for the drug discovery expert parent agent.

    Handles target identification, compound search, bioactivity lookup,
    and orchestration of child agents for specialized analysis.
    """

    next: str = ""

    # Task context
    task_description: str = ""
    workflow_type: str = ""  # e.g., "target_identification", "compound_profiling"

    # Target scoring state
    target_scores: Dict[str, Any] = {}
    ranked_targets: List[str] = []

    # Compound library state
    compound_library: Dict[str, Any] = {}
    bioactivity_data: Dict[str, Any] = {}

    # Workflow tracking
    analysis_complete: bool = False

    # Cross-cutting
    file_paths: List[str] = []
    intermediate_outputs: Dict[str, Any] = {}


class CheminformaticsExpertState(AgentState):
    """
    State for the cheminformatics expert child agent.

    Handles molecular analysis, descriptors, fingerprints, ADMET prediction,
    and 3D structure preparation.
    """

    next: str = ""

    # Task context
    task_description: str = ""

    # Molecular analysis state
    descriptor_results: Dict[str, Any] = {}
    similarity_matrix: Dict[str, Any] = {}
    admet_predictions: Dict[str, Any] = {}

    # Structure state
    prepared_structures: List[str] = []
    binding_sites: Dict[str, Any] = {}

    # Cross-cutting
    intermediate_outputs: Dict[str, Any] = {}


class ClinicalDevExpertState(AgentState):
    """
    State for the clinical development expert child agent.

    Handles target-disease evidence, drug synergy scoring,
    safety profiling, and clinical tractability assessment.
    """

    next: str = ""

    # Task context
    task_description: str = ""

    # Clinical evidence state
    disease_evidence: Dict[str, Any] = {}
    safety_profile: Dict[str, Any] = {}
    tractability_assessment: Dict[str, Any] = {}

    # Synergy scoring state
    synergy_results: Dict[str, Any] = {}
    combination_data: Dict[str, Any] = {}

    # Cross-cutting
    intermediate_outputs: Dict[str, Any] = {}


class PharmacogenomicsExpertState(AgentState):
    """
    State for the pharmacogenomics expert child agent.

    Handles drug-gene-variant interactions, ESM2 mutation predictions,
    variant impact scoring, and expression-drug sensitivity analysis.
    """

    next: str = ""

    # Task context
    task_description: str = ""

    # Variant analysis state
    variant_drug_interactions: Dict[str, Any] = {}
    mutation_predictions: Dict[str, Any] = {}
    variant_impact_scores: Dict[str, Any] = {}

    # Expression state
    expression_sensitivity: Dict[str, Any] = {}
    mutation_frequencies: Dict[str, Any] = {}

    # Cross-cutting
    intermediate_outputs: Dict[str, Any] = {}
