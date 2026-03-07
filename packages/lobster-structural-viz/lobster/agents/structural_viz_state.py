"""
State definitions for the protein structure visualization expert agent.

This module defines the state class for the structural viz expert agent
which handles PDB structure fetching, PyMOL visualization, and structural analysis.
"""

from typing import Any, Dict, List

from langgraph.prebuilt.chat_agent_executor import AgentState

__all__ = ["ProteinStructureVisualizationExpertState"]


class ProteinStructureVisualizationExpertState(AgentState):
    """
    State for the protein structure visualization expert agent.

    This agent specializes in fetching protein structures from PDB,
    creating PyMOL visualizations, performing structural analysis,
    and linking structures to omics data.
    """

    next: str = ""

    # Protein structure specific context
    task_description: str = ""  # Description of the current task
    structure_data: Dict[str, Any] = {}  # Current protein structure data
    pdb_ids: List[str] = []  # List of PDB IDs being worked with
    visualization_settings: Dict[str, Any] = {}  # PyMOL visualization parameters
    analysis_results: Dict[
        str, Any
    ] = {}  # Structure analysis results (RMSD, secondary structure, geometry)
    comparison_results: Dict[
        str, Any
    ] = {}  # RMSD comparison results between structures
    metadata: Dict[
        str, Any
    ] = {}  # PDB metadata (organism, resolution, experiment method)
    file_paths: List[str] = []  # Paths to structure files and visualizations
    methodology_parameters: Dict[str, Any] = {}  # Analysis parameters and settings
    data_context: str = ""  # Structural biology context
    intermediate_outputs: Dict[str, Any] = {}  # For partial structure analysis work
