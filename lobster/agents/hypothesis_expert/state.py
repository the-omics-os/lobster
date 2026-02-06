"""
State definitions for the HypothesisExpert agent.

Following the LangGraph 0.2.x multi-agent template pattern.
"""

from typing import Any, Dict, List, Optional

from langgraph.prebuilt.chat_agent_executor import AgentState

__all__ = ["HypothesisExpertState"]


class HypothesisExpertState(AgentState):
    """
    State for the hypothesis expert agent.

    Tracks hypothesis generation state including the current hypothesis,
    evidence sources, and research context.
    """

    next: str = ""

    # Task context
    task_description: str = ""
    research_objective: str = ""

    # Hypothesis state
    current_hypothesis: Optional[str] = None  # Current hypothesis text
    hypothesis_iteration: int = 0  # Number of hypothesis iterations
    evidence_sources: List[Dict[str, Any]] = []  # Evidence used for hypothesis

    # Research context
    key_insights: List[str] = []  # Accumulated research insights
    methodology: Optional[str] = None  # Current methodology description

    # Cross-cutting
    intermediate_outputs: Dict[str, Any] = {}
