"""State definitions for the metadata assistant agent."""

from typing import Any, Dict, List

from langgraph.prebuilt.chat_agent_executor import AgentState

__all__ = ["MetadataAssistantState"]


class MetadataAssistantState(AgentState):
    """State for metadata harmonization and sample-mapping workflows."""

    next: str

    # Metadata-specific context
    task_description: str
    metadata_sources: List[str]
    candidate_mappings: Dict[str, Any]
    standardization_results: Dict[str, Any]
    validation_results: Dict[str, Any]

    # Standard expert fields
    file_paths: List[str]
    methodology_parameters: Dict[str, Any]
    data_context: str
    intermediate_outputs: Dict[str, Any]
