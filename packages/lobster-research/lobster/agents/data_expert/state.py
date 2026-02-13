"""State schema for Data Expert agent."""

from typing import Any, Dict, List

from langgraph.prebuilt.chat_agent_executor import AgentState

__all__ = ["DataExpertState"]


class DataExpertState(AgentState):
    """
    State for the data expert agent.

    Handles local data acquisition, modality management, and workspace I/O.
    """

    next: str

    # Data acquisition and workspace context
    task_description: str
    modalities: List[str]
    dataset_registry: Dict[str, Any]
    download_queue_entries: List[str]

    # Standard expert fields (for consistency)
    file_paths: List[str]
    methodology_parameters: Dict[str, Any]
    data_context: str
    intermediate_outputs: Dict[str, Any]
