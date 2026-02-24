"""
State definitions for the visualization expert agent.

This module defines the state class for the visualization expert agent
which handles data visualization and plotting operations.
"""

from typing import Any, Dict, List

from langgraph.prebuilt.chat_agent_executor import AgentState

__all__ = ["VisualizationExpertState"]


class VisualizationExpertState(AgentState):
    """
    State for the visualization expert agent.

    This agent creates visualizations for omics data including UMAPs,
    heatmaps, volcano plots, and other scientific visualizations.
    """

    next: str = ""

    # Visualization specific context
    task_description: str = ""  # Description of the current task
    current_request: Dict[str, Any] = {}  # Current visualization request details
    last_plot_id: str = ""  # Last created plot ID for tracking
    visualization_history: List[Dict[str, Any]] = (
        []
    )  # History of created visualizations
    plot_metadata: Dict[str, Any] = {}  # Metadata for current plot session
    active_modalities: List[str] = []  # Modalities currently being visualized
    visualization_parameters: Dict[str, Any] = {}  # Current visualization parameters
    plot_queue: List[Dict[str, Any]] = []  # Queue of pending visualization tasks
    file_paths: List[str] = []  # Paths to saved visualization files
    methodology_parameters: Dict[str, Any] = {}  # Visualization method parameters
    data_context: str = ""  # Visualization data context
    intermediate_outputs: Dict[str, Any] = {}  # For partial visualization work
