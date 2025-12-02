# Proteomics Agent Module
# Unified agent for mass spectrometry and affinity proteomics analysis

from lobster.agents.proteomics.proteomics_expert import proteomics_expert
from lobster.agents.proteomics.platform_config import (
    PlatformConfig,
    PLATFORM_CONFIGS,
    detect_platform_type,
    get_platform_config,
)
from lobster.agents.proteomics.state import ProteomicsExpertState
from lobster.agents.proteomics.deprecated import ms_proteomics_alias, affinity_proteomics_alias

__all__ = [
    # Main agent
    "proteomics_expert",
    # Platform configuration
    "PlatformConfig",
    "PLATFORM_CONFIGS",
    "detect_platform_type",
    "get_platform_config",
    # State class
    "ProteomicsExpertState",
    # Deprecated aliases
    "ms_proteomics_alias",
    "affinity_proteomics_alias",
]
