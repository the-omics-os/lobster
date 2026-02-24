# Metabolomics Agent Module
# Agent for LC-MS, GC-MS, and NMR untargeted metabolomics analysis
# with quality assessment, preprocessing, multivariate statistics,
# metabolite annotation, and pathway enrichment.
#
# Note: The metabolomics_expert agent and config are PREMIUM features.
# This module uses graceful imports to avoid crashes in the FREE tier.

# State classes are always available (FREE tier)
from lobster.agents.metabolomics.state import MetabolomicsExpertState

# Try to import PREMIUM components, gracefully degrade if not available
try:
    from lobster.agents.metabolomics.config import (
        PLATFORM_CONFIGS,
        MetabPlatformConfig,
        detect_platform_type,
        get_platform_config,
    )
    from lobster.agents.metabolomics.metabolomics_expert import metabolomics_expert
    from lobster.agents.metabolomics.prompts import (
        create_metabolomics_expert_prompt,
    )

    METABOLOMICS_EXPERT_AVAILABLE = True
except ImportError:
    # PREMIUM components not available in FREE tier
    METABOLOMICS_EXPERT_AVAILABLE = False
    metabolomics_expert = None
    create_metabolomics_expert_prompt = None
    PLATFORM_CONFIGS = {}
    MetabPlatformConfig = None
    detect_platform_type = None
    get_platform_config = None

__all__ = [
    # Availability flag
    "METABOLOMICS_EXPERT_AVAILABLE",
    # Main agent (PREMIUM - may be None in FREE tier)
    "metabolomics_expert",
    # Platform configuration (PREMIUM - may be None in FREE tier)
    "MetabPlatformConfig",
    "PLATFORM_CONFIGS",
    "detect_platform_type",
    "get_platform_config",
    # Prompt (PREMIUM - may be None in FREE tier)
    "create_metabolomics_expert_prompt",
    # State classes (FREE - always available)
    "MetabolomicsExpertState",
]
