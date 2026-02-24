"""
Premium agent LLM configurations for custom packages.

These configurations are loaded via entry points (lobster.agent_configs)
and merged into the LobsterAgentConfigurator at runtime.

This file is synced to custom packages (e.g., lobster-custom-{customer})
via scripts/sync_to_custom.py.

Custom packages should register these configs in their pyproject.toml:

    [project.entry-points."lobster.agent_configs"]
    metadata_assistant = "lobster_custom_{customer}.config.agent_configs:METADATA_ASSISTANT_CONFIG"
"""

from lobster.config.agent_config import CustomAgentConfig

# =============================================================================
# METADATA ASSISTANT CONFIG
# =============================================================================
# Metadata operations: publication queue processing, cross-dataset sample mapping,
# microbiome filtering, disease standardization, Pydantic schema validation.
#
# Uses dynamic model reference to follow research_agent's configuration.
# This ensures metadata_assistant adapts to the customer's provider settings:
#   - development profile → claude-4-sonnet
#   - production profile → claude-4-sonnet (supervisor: claude-4-5-sonnet)
#   - performance profile → claude-4-5-sonnet
#   - max profile → claude-4-5-opus (supervisor)

METADATA_ASSISTANT_CONFIG = CustomAgentConfig(
    name="metadata_assistant",
    model_reference="research_agent",  # Dynamic: uses same model as research_agent
    thinking_preset="standard",  # 2000 token budget for complex operations
    custom_params={
        "description": (
            "Metadata operations for publication queue filtering, "
            "cross-dataset sample mapping, and Pydantic schema validation"
        ),
    },
)

# =============================================================================
# FUTURE PREMIUM AGENTS
# =============================================================================
# Add additional premium agent configs here as needed.
# Each config should be registered via entry points in the custom package.

# Example:
# PROTEOMICS_EXPERT_CONFIG = CustomAgentConfig(
#     name="proteomics_expert",
#     model_preset="claude-4-sonnet",
#     thinking_preset="standard",
# )
