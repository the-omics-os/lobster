"""
State definitions for the bioinformatics multi-agent system.

Following the LangGraph 0.2.x multi-agent template pattern, but expanded to
capture routing metadata, agent-specific working memory, and intermediate outputs.

This module also provides get_all_state_classes() for dynamic state discovery
from installed packages via entry points.
"""

from importlib.metadata import entry_points
from typing import Annotated, Any, Dict, List, Optional, Type
from typing_extensions import TypedDict
import logging
import sys

from langgraph.prebuilt.chat_agent_executor import AgentState

logger = logging.getLogger(__name__)


# =============================================================================
# Dynamic State Discovery
# =============================================================================


def get_all_state_classes() -> Dict[str, Type]:
    """Discover state classes from installed packages via entry points.

    Returns a dict mapping state class names to their types. Core states
    are always included. Package states are discovered from the
    'lobster.states' entry point group.

    Returns:
        Dict mapping state names to state classes

    Example:
        states = get_all_state_classes()
        # {'OverallState': <class>, 'TodoItem': <class>, 'SingleCellExpertState': <class>}
    """
    states: Dict[str, Type] = {}

    # Core states always available (defined below in this module)
    # These are forward-referenced and will be resolved after class definitions
    current_module = sys.modules[__name__]

    # Core base states
    overall_state = getattr(current_module, "OverallState", None)
    todo_item = getattr(current_module, "TodoItem", None)
    if overall_state is not None:
        states["OverallState"] = overall_state
    if todo_item is not None:
        states["TodoItem"] = todo_item

    # Core states still defined in this module (not yet migrated to packages)
    # Domain-specific states are discovered via lobster.states entry points
    core_state_classes = [
        "SingleCellExpertState",  # Legacy compatibility
        "BulkRNASeqExpertState",  # Legacy compatibility
        "MethodState",  # Core methodology state
        "CustomFeatureAgentState",  # custom_feature_agent.py still in core
    ]

    for state_name in core_state_classes:
        state_class = getattr(current_module, state_name, None)
        if state_class is not None:
            states[state_name] = state_class

    # Discover package states via entry points
    try:
        eps = entry_points(group="lobster.states")
        for ep in eps:
            try:
                state_class = ep.load()
                states[ep.name] = state_class
                logger.debug(f"Discovered state class: {ep.name} from {ep.value}")
            except Exception as e:
                logger.warning(f"Failed to load state {ep.name}: {e}")
    except Exception as e:
        logger.debug(f"No lobster.states entry points found: {e}")

    logger.debug(f"get_all_state_classes() found {len(states)} state classes")
    return states


# =============================================================================
# Todo List State (v3.5+)
# =============================================================================


class TodoItem(TypedDict):
    """Individual todo item for planning multi-step tasks.

    Used by supervisor to decompose complex requests into trackable subtasks.
    Follows the DeepAgents/LangChain TodoListMiddleware pattern.

    Attributes:
        content: Task description in imperative form (e.g., "Download GSE12345")
        status: Current state - "pending", "in_progress", or "completed"
        activeForm: Present continuous form for UI display (e.g., "Downloading GSE12345")
    """

    content: str
    status: str  # "pending" | "in_progress" | "completed"
    activeForm: str


def _todo_reducer(
    left: Optional[List[TodoItem]], right: Optional[List[TodoItem]]
) -> List[TodoItem]:
    """Reducer for todos field - replaces entire list on update.

    This is a replace reducer (not append) because:
    1. Todos represent the CURRENT plan state, not history
    2. Agent sends complete updated list on each write_todos call
    3. Matches DeepAgents/LangChain TodoListMiddleware behavior

    Args:
        left: Previous todos list (or None)
        right: New todos list from tool update (or None)

    Returns:
        The new todos list (right), or previous (left), or empty list
    """
    if right is not None:
        return right
    return left if left is not None else []


# =============================================================================
# Core Agent States
# =============================================================================


class OverallState(AgentState):
    """
    Base state for the Lobster multi-agent system.

    Agent packages extend this class to add domain-specific fields:

        class TranscriptomicsState(OverallState):
            clustering_results: Dict[str, Any] = {}
            marker_genes: List[str] = []

    The supervisor uses OverallState directly. Specialist agents
    use extended state schemas via create_react_agent(state_schema=...).

    Attributes:
        last_active_agent: Name of the agent that last handled conversation
        conversation_id: Unique session identifier
        current_task: Current task description for handoffs
        task_context: Additional context passed between agents
        todos: Task list managed via write_todos tool

    Note:
        This is the extensible base class for modular agent packages.
        Individual agents are subgraphs with their own state schemas
        that extend this base.
    """

    # Meta routing information
    last_active_agent: str = ""
    conversation_id: str = ""

    # Optional: Task context for handoffs
    current_task: str = ""
    task_context: Dict[str, Any] = {}

    # Todo list for planning multi-step tasks (v3.5+)
    # Updated via write_todos tool using Command pattern
    todos: Annotated[List[TodoItem], _todo_reducer] = []


# class SupervisorState(AgentState):
#     """
#     State for the supervisor agent.
#     """
#     next: str  # The next node to route to (agent name or END)
#     last_active_agent: str

#     # Bioinformatics-specific context the supervisor might maintain
#     analysis_results: Dict[str, Any]     # Combined results from multiple experts
#     methodology_parameters: Dict[str, Any]
#     data_context: str                    # High-level description of the data in use

#     # Control information
#     delegation_history: List[str]        # Track which agents have been called
#     pending_tasks: List[str]


# class TranscriptomicsExpertState(AgentState):
#     """
#     State for the transcriptomics expert agent.
#     """
#     next: str

#     # Transcriptomics-specific context
#     analysis_results: Dict[str, Any]     # Gene expression, DEG lists, etc.
#     file_paths: List[str]            # Paths to input/output files
#     methodology_parameters: Dict[str, Any]
#     data_context: str                    # Type/source of transcriptomics data (RNA-seq, microarray)
#     quality_control_metrics: Dict[str, Any]
#     intermediate_outputs: Dict[str, Any] # For partial computations before returning to supervisor


class SingleCellExpertState(AgentState):
    """
    State for the single-cell RNA-seq expert agent.
    """

    next: str

    # Single-cell specific context
    task_description: str  # Description of the current task
    analysis_results: Dict[str, Any]  # Single-cell analysis results, clustering, etc.
    clustering_parameters: Dict[
        str, Any
    ]  # Leiden resolution, batch correction settings
    cell_type_annotations: Dict[str, Any]  # Cell type assignment results
    quality_control_metrics: Dict[str, Any]  # QC metrics specific to single-cell
    doublet_detection_results: Dict[str, Any]  # Doublet detection outcomes
    marker_genes: Dict[str, Any]  # Marker genes per cluster
    file_paths: List[str]  # Paths to input/output files
    methodology_parameters: Dict[str, Any]
    data_context: str  # Single-cell data context
    intermediate_outputs: Dict[
        str, Any
    ]  # For partial computations before returning to supervisor


class BulkRNASeqExpertState(AgentState):
    """
    State for the bulk RNA-seq expert agent.
    """

    next: str

    # Bulk RNA-seq specific context
    task_description: str  # Description of the current task
    analysis_results: Dict[str, Any]  # Bulk RNA-seq analysis results, DE genes, etc.
    differential_expression_results: Dict[str, Any]  # DE analysis outcomes
    pathway_enrichment_results: Dict[str, Any]  # Pathway analysis results
    experimental_design: Dict[str, Any]  # Sample grouping and experimental setup
    quality_control_metrics: Dict[str, Any]  # QC metrics specific to bulk RNA-seq
    statistical_parameters: Dict[str, Any]  # Statistical method parameters
    file_paths: List[str]  # Paths to input/output files
    methodology_parameters: Dict[str, Any]
    data_context: str  # Bulk RNA-seq data context
    intermediate_outputs: Dict[
        str, Any
    ]  # For partial computations before returning to supervisor


# Note: DataExpertState is now in lobster-research package
# Note: ResearchAgentState is now in lobster-research package
# Both discovered via lobster.states entry points


class MethodState(AgentState):
    """
    State for the method expert agent.
    """

    next: str

    # Methodology-specific context
    task_description: str  # Description of the current task
    methods_information: Dict[
        str, Any
    ]  # Details about computational/experimental methods
    data_context: str
    evaluation_metrics: Dict[str, Any]  # Accuracy, runtime, reproducibility metrics
    recommendations: List[str]  # Suggested methods or pipelines
    references: List[str]

    # Standard expert fields (for consistency)
    file_paths: List[str]  # Paths to method documentation, protocols, exports
    intermediate_outputs: Dict[
        str, Any
    ]  # Partial method analysis before handoff to supervisor


# Note: MachineLearningExpertState is now in lobster-ml package
# Note: VisualizationExpertState is now in lobster-visualization package
# Both discovered via lobster.states entry points


class CustomFeatureAgentState(AgentState):
    """
    State for the custom feature creation agent.

    This agent uses Claude Code SDK to generate new agents, services,
    providers, tools, tests, and documentation following Lobster patterns.
    """

    next: str

    # Feature creation specific context
    task_description: str  # Description of the feature creation task
    feature_name: str  # Name of the feature being created
    feature_type: str  # Type: agent, service, provider, agent_with_service
    requirements: str  # Detailed feature requirements
    research_findings: Dict[str, Any]  # Internet research results (Linkup)
    created_files: List[str]  # List of files created during feature generation
    validation_errors: List[str]  # Validation errors encountered
    sdk_output: str  # Output from Claude Code SDK
    integration_instructions: str  # Instructions for manual integration steps
    test_results: Dict[str, Any]  # Results from automated testing
    file_paths: List[str]  # Paths to created files
    methodology_parameters: Dict[str, Any]  # Feature creation parameters
    data_context: str  # Context about the feature being created
    intermediate_outputs: Dict[
        str, Any
    ]  # For partial feature creation work before returning to supervisor


# Note: ProteinStructureVisualizationExpertState is now in lobster-structural-viz package
# Discovered via lobster.states entry points


class ProteinStructureVisualizationExpertState(AgentState):
    """
    State for the protein structure visualization expert agent.

    Tracks structure loading, visualization outputs, and annotation context.
    """

    next: str

    # Visualization-specific context
    task_description: str
    structures_loaded: List[str]
    visualization_outputs: Dict[str, Any]
    annotations: Dict[str, Any]

    # Standard expert fields (for consistency)
    file_paths: List[str]
    methodology_parameters: Dict[str, Any]
    data_context: str
    intermediate_outputs: Dict[str, Any]


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Dynamic state discovery
    "get_all_state_classes",
    # Core base state for agent extension
    "OverallState",
    "TodoItem",
    # Core states (not yet migrated to packages)
    "SingleCellExpertState",  # Legacy compatibility
    "BulkRNASeqExpertState",  # Legacy compatibility
    "MethodState",
    "CustomFeatureAgentState",
    "ProteinStructureVisualizationExpertState",
    # Package states are discovered via lobster.states entry points:
    # - TranscriptomicsExpertState, AnnotationExpertState, DEAnalysisExpertState (lobster-transcriptomics)
    # - ResearchAgentState, DataExpertState (lobster-research)
    # - MachineLearningExpertState (lobster-ml)
    # - ProteomicsExpertState (lobster-proteomics)
    # - ProteinStructureVisualizationExpertState (lobster-structural-viz)
]
