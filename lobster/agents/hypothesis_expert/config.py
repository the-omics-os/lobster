"""
Configuration for the HypothesisExpert agent.

Defines the AgentRegistryConfig for integration with Lobster's agent registry.
"""

from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="hypothesis_expert",
    display_name="Hypothesis Expert",
    description=(
        "Synthesizes research findings into novel, evidence-linked scientific hypotheses. "
        "Uses literature, analysis results, and dataset metadata to generate testable hypotheses "
        "with rationale, novelty statements, experimental designs, and follow-up recommendations."
    ),
    factory_function="lobster.agents.hypothesis_expert.hypothesis_expert.hypothesis_expert",
    handoff_tool_name="handoff_to_hypothesis_expert",
    handoff_tool_description=(
        "Delegate hypothesis generation tasks. Use when: (1) synthesizing research findings "
        "into a formal hypothesis, (2) generating novel research directions from literature review, "
        "(3) creating testable hypotheses after exploratory analysis, (4) updating/refining "
        "existing hypotheses with new evidence. Pass evidence sources as workspace keys."
    ),
    child_agents=None,  # No sub-agents
    supervisor_accessible=True,  # Accessible from supervisor
)
