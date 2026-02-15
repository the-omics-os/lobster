# Proteomics Agent Module
# Parent agent for mass spectrometry and affinity proteomics analysis,
# with DE analysis and biomarker discovery sub-agents.
#
# Note: The proteomics_expert agent and config are PREMIUM features.
# This module uses graceful imports to avoid crashes in the FREE tier.

# State classes are always available (FREE tier)
from lobster.agents.proteomics.state import (
    BiomarkerDiscoveryExpertState,
    DEAnalysisExpertState,
    ProteomicsExpertState,
)

# Try to import PREMIUM components, gracefully degrade if not available
try:
    from lobster.agents.proteomics.config import (
        PLATFORM_CONFIGS,
        PlatformConfig,
        detect_platform_type,
        get_platform_config,
    )
    from lobster.agents.proteomics.prompts import (
        create_biomarker_discovery_expert_prompt,
        create_de_analysis_expert_prompt,
        create_proteomics_expert_prompt,
    )
    from lobster.agents.proteomics.proteomics_expert import proteomics_expert

    PROTEOMICS_EXPERT_AVAILABLE = True
except ImportError:
    # PREMIUM components not available in FREE tier
    PROTEOMICS_EXPERT_AVAILABLE = False
    proteomics_expert = None
    create_proteomics_expert_prompt = None
    create_de_analysis_expert_prompt = None
    create_biomarker_discovery_expert_prompt = None
    PLATFORM_CONFIGS = {}
    PlatformConfig = None
    detect_platform_type = None
    get_platform_config = None

__all__ = [
    # Availability flag
    "PROTEOMICS_EXPERT_AVAILABLE",
    # Main agent (PREMIUM - may be None in FREE tier)
    "proteomics_expert",
    # Platform configuration (PREMIUM - may be None in FREE tier)
    "PlatformConfig",
    "PLATFORM_CONFIGS",
    "detect_platform_type",
    "get_platform_config",
    # Prompts (PREMIUM - may be None in FREE tier)
    "create_proteomics_expert_prompt",
    "create_de_analysis_expert_prompt",
    "create_biomarker_discovery_expert_prompt",
    # State classes (FREE - always available)
    "ProteomicsExpertState",
    "DEAnalysisExpertState",
    "BiomarkerDiscoveryExpertState",
]
