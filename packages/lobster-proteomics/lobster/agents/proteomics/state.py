"""
State definitions for proteomics agents (parent + sub-agents).

This module defines state classes for the proteomics expert parent agent
and its sub-agents (DE analysis, biomarker discovery).

Following the LangGraph 0.2.x multi-agent template pattern, with fields
tailored to proteomics analysis workflows.
"""

from typing import Any, Dict, List

from langgraph.prebuilt.chat_agent_executor import AgentState

__all__ = [
    "ProteomicsExpertState",
    "DEAnalysisExpertState",
    "BiomarkerDiscoveryExpertState",
]


class ProteomicsExpertState(AgentState):
    """
    State for the unified proteomics expert parent agent.

    This agent handles both mass spectrometry (DDA/DIA) and affinity-based
    (Olink, SomaScan) proteomics analysis, auto-detecting platform type
    and applying appropriate defaults.
    """

    next: str = ""

    # Task context
    task_description: str = ""
    platform_type: str = ""  # "mass_spec" or "affinity" (auto-detected from data)

    # QC state
    quality_metrics: Dict[str, Any] = {}
    preprocessing_applied: bool = False

    # Platform-specific state
    platform_defaults: Dict[str, Any] = {}  # defaults applied based on detection
    missing_value_info: Dict[str, Any] = {}  # MNAR vs MAR patterns
    normalization_info: Dict[str, Any] = {}

    # Analysis state
    differential_results: Dict[str, Any] = {}
    clustering_results: Dict[str, Any] = {}

    # Cross-cutting
    file_paths: List[str] = []
    intermediate_outputs: Dict[str, Any] = {}


class DEAnalysisExpertState(AgentState):
    """
    State for the proteomics DE analysis sub-agent.

    Handles differential expression, time course analysis, and
    correlation analysis for proteomics data.
    """

    next: str = ""

    # Task context
    task_description: str = ""
    platform_type: str = ""

    # DE-specific state
    comparison_groups: Dict[str, Any] = {}
    de_results: Dict[str, Any] = {}
    time_course_results: Dict[str, Any] = {}
    correlation_results: Dict[str, Any] = {}

    # Cross-cutting
    intermediate_outputs: Dict[str, Any] = {}


class BiomarkerDiscoveryExpertState(AgentState):
    """
    State for the proteomics biomarker discovery sub-agent.

    Handles WGCNA network analysis, module-trait correlations,
    Cox survival analysis, and Kaplan-Meier biomarker identification.
    """

    next: str = ""

    # Task context
    task_description: str = ""

    # Network analysis state
    network_modules: Dict[str, Any] = {}
    module_trait_correlations: Dict[str, Any] = {}

    # Survival analysis state
    survival_results: Dict[str, Any] = {}
    biomarker_candidates: List[str] = []

    # Cross-cutting
    intermediate_outputs: Dict[str, Any] = {}
