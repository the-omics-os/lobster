"""
Configuration constants for drug discovery agents.

Defines API endpoints, scoring weights, thresholds, and default parameters
used across all drug discovery agents and services.
"""

__all__ = [
    "CHEMBL_API_BASE",
    "OPENTARGETS_GRAPHQL",
    "PUBCHEM_API_BASE",
    "CLINICALTRIALS_API_BASE",
    "TARGET_EVIDENCE_WEIGHTS",
    "SYNERGY_THRESHOLDS",
    "LIPINSKI_RULES",
    "ADMET_THRESHOLDS",
    "DEFAULT_HTTP_TIMEOUT",
    "DEFAULT_SEARCH_LIMIT",
]

# =============================================================================
# API Endpoints
# =============================================================================

CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"
OPENTARGETS_GRAPHQL = "https://api.platform.opentargets.org/api/v4/graphql"
PUBCHEM_API_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CLINICALTRIALS_API_BASE = "https://clinicaltrials.gov/api/v2"

# =============================================================================
# HTTP Client Defaults
# =============================================================================

DEFAULT_HTTP_TIMEOUT = 45.0  # seconds (ChEMBL can be slow for broad target queries)
DEFAULT_SEARCH_LIMIT = 20

# =============================================================================
# Target Scoring Weights
# =============================================================================

TARGET_EVIDENCE_WEIGHTS = {
    "genetic_association": 0.30,
    "known_drug": 0.25,
    "expression_specificity": 0.20,
    "pathogenicity": 0.15,
    "literature": 0.10,
}

# =============================================================================
# Synergy Model Thresholds
# =============================================================================

SYNERGY_THRESHOLDS = {
    "synergistic": 0.1,     # Bliss excess > 0.1
    "additive_upper": 0.1,  # -0.1 < Bliss excess < 0.1
    "additive_lower": -0.1,
    "antagonistic": -0.1,   # Bliss excess < -0.1
}

# =============================================================================
# Lipinski Rule of Five
# =============================================================================

LIPINSKI_RULES = {
    "mw_max": 500.0,
    "logp_max": 5.0,
    "hbd_max": 5,
    "hba_max": 10,
}

# =============================================================================
# ADMET Prediction Thresholds
# =============================================================================

ADMET_THRESHOLDS = {
    "tpsa_max_oral": 140.0,      # Topological polar surface area for oral absorption
    "rotatable_bonds_max": 10,
    "logp_range": (-0.4, 5.6),   # Ghose filter range
    "mw_range": (160, 480),      # Ghose filter range
}
