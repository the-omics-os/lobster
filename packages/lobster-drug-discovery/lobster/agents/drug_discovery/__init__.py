# Drug Discovery Agent Module
# Parent agent for drug target identification, compound profiling, and
# clinical translation, with cheminformatics, clinical development, and
# pharmacogenomics child agents.
#
# Note: All drug discovery agents are FREE tier.
# This module uses graceful imports to avoid crashes if dependencies are missing.

# State classes are always available (FREE tier)
from lobster.agents.drug_discovery.state import (
    CheminformaticsExpertState,
    ClinicalDevExpertState,
    DrugDiscoveryExpertState,
    PharmacogenomicsExpertState,
)

# Try to import components, gracefully degrade if not available
try:
    from lobster.agents.drug_discovery.drug_discovery_expert import (
        drug_discovery_expert,
    )
    from lobster.agents.drug_discovery.prompts import (
        create_cheminformatics_expert_prompt,
        create_clinical_dev_expert_prompt,
        create_drug_discovery_expert_prompt,
        create_pharmacogenomics_expert_prompt,
    )

    DRUG_DISCOVERY_EXPERT_AVAILABLE = True
except ImportError:
    DRUG_DISCOVERY_EXPERT_AVAILABLE = False
    drug_discovery_expert = None
    create_drug_discovery_expert_prompt = None
    create_cheminformatics_expert_prompt = None
    create_clinical_dev_expert_prompt = None
    create_pharmacogenomics_expert_prompt = None

__all__ = [
    # Availability flag
    "DRUG_DISCOVERY_EXPERT_AVAILABLE",
    # Main agent (may be None if deps missing)
    "drug_discovery_expert",
    # Prompts (may be None if deps missing)
    "create_drug_discovery_expert_prompt",
    "create_cheminformatics_expert_prompt",
    "create_clinical_dev_expert_prompt",
    "create_pharmacogenomics_expert_prompt",
    # State classes (always available)
    "DrugDiscoveryExpertState",
    "CheminformaticsExpertState",
    "ClinicalDevExpertState",
    "PharmacogenomicsExpertState",
]
