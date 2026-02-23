"""
State definitions for the metabolomics expert agent.

This module defines the state class for the metabolomics expert agent,
following the LangGraph 0.2.x multi-agent template pattern with fields
tailored to metabolomics analysis workflows (LC-MS, GC-MS, NMR).
"""

from typing import Any, Dict, List

from langgraph.prebuilt.chat_agent_executor import AgentState

__all__ = [
    "MetabolomicsExpertState",
]


class MetabolomicsExpertState(AgentState):
    """
    State for the metabolomics expert agent.

    This agent handles LC-MS, GC-MS, and NMR untargeted metabolomics analysis,
    auto-detecting platform type and applying appropriate defaults for QC,
    preprocessing, multivariate statistics, and metabolite annotation.
    """

    next: str = ""

    # Task context
    task_description: str = ""
    platform_type: str = ""  # "lc_ms", "gc_ms", or "nmr" (auto-detected from data)

    # QC state
    quality_metrics: Dict[str, Any] = {}
    preprocessing_applied: bool = False

    # Platform-specific state
    platform_defaults: Dict[str, Any] = {}  # defaults applied based on detection
    missing_value_info: Dict[str, Any] = {}  # imputation method and stats
    normalization_info: Dict[str, Any] = {}

    # Analysis state
    statistical_results: Dict[str, Any] = {}
    multivariate_results: Dict[str, Any] = {}
    annotation_results: Dict[str, Any] = {}

    # Cross-cutting
    file_paths: List[str] = []
    intermediate_outputs: Dict[str, Any] = {}
