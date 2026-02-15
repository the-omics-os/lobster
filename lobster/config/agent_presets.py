"""
Agent group presets for common workflows.

This module provides preset configurations for agent groups, allowing users
to specify a preset name (like "scrna-basic") instead of listing individual
agents. Presets simplify configuration for standard use cases.

Usage:
    from lobster.config.agent_presets import expand_preset, list_presets

    # Get agents for a preset
    agents = expand_preset("scrna-basic")

    # List all available presets
    presets = list_presets()
"""

from typing import Dict, List, Optional

# =============================================================================
# AGENT PRESETS DICTIONARY
# =============================================================================
# Each preset defines a list of agent names for common workflow configurations.
# Agent names must match those registered in the agent registry.

AGENT_PRESETS: Dict[str, Dict[str, any]] = {
    "scrna-basic": {
        "agents": [
            "research_agent",
            "data_expert_agent",
            "transcriptomics_expert",
            "visualization_expert_agent",
        ],
        "description": "Basic single-cell RNA-seq workflow with literature search, data loading, analysis, and visualization",
    },
    "scrna-full": {
        "agents": [
            "research_agent",
            "data_expert_agent",
            "transcriptomics_expert",
            "visualization_expert_agent",
            "annotation_expert",
            "de_analysis_expert",
            "metadata_assistant",
        ],
        "description": "Full single-cell RNA-seq workflow including cell annotation, differential expression, and metadata management (includes premium metadata_assistant)",
    },
    "multiomics-full": {
        "agents": [
            "research_agent",
            "data_expert_agent",
            "transcriptomics_expert",
            "visualization_expert_agent",
            "annotation_expert",
            "de_analysis_expert",
            "metadata_assistant",
            "proteomics_expert",
            "genomics_expert",
            "machine_learning_expert_agent",
        ],
        "description": "Complete multi-omics workflow with all available agents including proteomics, genomics, and machine learning capabilities",
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def expand_preset(preset_name: str) -> Optional[List[str]]:
    """Expand a preset name to a list of agent names.

    Args:
        preset_name: Name of the preset (e.g., "scrna-basic")

    Returns:
        List of agent names if preset exists, None otherwise

    Examples:
        >>> expand_preset("scrna-basic")
        ['research_agent', 'data_expert_agent', 'transcriptomics_expert', 'visualization_expert_agent']
        >>> expand_preset("invalid-preset")
        None
    """
    preset = AGENT_PRESETS.get(preset_name)
    if preset is None:
        return None
    return preset["agents"].copy()


def list_presets() -> Dict[str, int]:
    """List all available presets with their agent counts.

    Returns:
        Dictionary mapping preset name to number of agents

    Examples:
        >>> list_presets()
        {'scrna-basic': 4, 'scrna-full': 7, 'multiomics-full': 10}
    """
    return {name: len(preset["agents"]) for name, preset in AGENT_PRESETS.items()}


def get_preset_description(preset_name: str) -> Optional[str]:
    """Get a human-readable description of a preset.

    Args:
        preset_name: Name of the preset

    Returns:
        Description string if preset exists, None otherwise

    Examples:
        >>> get_preset_description("scrna-basic")
        'Basic single-cell RNA-seq workflow with literature search, data loading, analysis, and visualization'
    """
    preset = AGENT_PRESETS.get(preset_name)
    if preset is None:
        return None
    return preset["description"]


def get_preset_agents(preset_name: str) -> Optional[List[str]]:
    """Alias for expand_preset() for semantic clarity.

    Args:
        preset_name: Name of the preset

    Returns:
        List of agent names if preset exists, None otherwise
    """
    return expand_preset(preset_name)


def is_valid_preset(preset_name: str) -> bool:
    """Check if a preset name is valid.

    Args:
        preset_name: Name to check

    Returns:
        True if the preset exists, False otherwise
    """
    return preset_name in AGENT_PRESETS


def get_all_preset_info() -> Dict[str, Dict[str, any]]:
    """Get full information about all presets.

    Returns:
        Dictionary with preset name as key and dict with agents, description, and count

    Examples:
        >>> info = get_all_preset_info()
        >>> info["scrna-basic"]["count"]
        4
    """
    return {
        name: {
            "agents": preset["agents"].copy(),
            "description": preset["description"],
            "count": len(preset["agents"]),
        }
        for name, preset in AGENT_PRESETS.items()
    }
