"""
Agent configuration for machine learning experts.

Defines AGENT_CONFIG for each ML agent for entry point discovery.
"""

from lobster.config.agent_registry import AgentRegistryConfig

__all__ = [
    "ML_EXPERT_CONFIG",
    "FEATURE_SELECTION_EXPERT_CONFIG",
    "SURVIVAL_ANALYSIS_EXPERT_CONFIG",
]

# Main ML expert agent configuration
ML_EXPERT_CONFIG = AgentRegistryConfig(
    name="machine_learning_expert",
    display_name="Machine Learning Expert",
    description="ML data preparation: feature engineering, data splitting, framework export, scVI embeddings",
    factory_function="lobster.agents.machine_learning.machine_learning_expert.machine_learning_expert",
    handoff_tool_name="handoff_to_machine_learning_expert",
    handoff_tool_description="Assign ML preparation tasks: feature selection, data splitting, PyTorch/TensorFlow export",
    child_agents=["feature_selection_expert", "survival_analysis_expert"],
    supervisor_accessible=True,
    tier_requirement="free",  # All agents free — commercial value in Omics-OS Cloud
)

# Feature selection sub-agent configuration
FEATURE_SELECTION_EXPERT_CONFIG = AgentRegistryConfig(
    name="feature_selection_expert",
    display_name="Feature Selection Expert",
    description="Biomarker discovery: stability-based selection, LASSO/Elastic Net, importance ranking",
    factory_function="lobster.agents.machine_learning.feature_selection_expert.feature_selection_expert",
    handoff_tool_name="handoff_to_feature_selection_expert",
    handoff_tool_description="Assign biomarker discovery and feature selection tasks for high-dimensional omics data",
    supervisor_accessible=False,  # Sub-agent, accessed through ML expert
    tier_requirement="free",  # All agents free — commercial value in Omics-OS Cloud
)

# Survival analysis sub-agent configuration
SURVIVAL_ANALYSIS_EXPERT_CONFIG = AgentRegistryConfig(
    name="survival_analysis_expert",
    display_name="Survival Analysis Expert",
    description="Time-to-event analysis: Cox PH models, Kaplan-Meier, risk stratification",
    factory_function="lobster.agents.machine_learning.survival_analysis_expert.survival_analysis_expert",
    handoff_tool_name="handoff_to_survival_analysis_expert",
    handoff_tool_description="Assign survival analysis tasks: Cox regression, hazard ratios, risk scoring",
    supervisor_accessible=False,  # Sub-agent, accessed through ML expert
    tier_requirement="free",  # All agents free — commercial value in Omics-OS Cloud
)
