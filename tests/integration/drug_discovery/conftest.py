"""
Shared fixtures for drug discovery integration tests.

Provides module-scoped service instances and RDKit availability detection.
All services are stateless singletons — module scope avoids repeated
instantiation across test classes.
"""

import pytest

_drug_discovery_available = None


def _check_drug_discovery():
    global _drug_discovery_available
    if _drug_discovery_available is None:
        try:
            import lobster.services.drug_discovery.chembl_service  # noqa: F401

            _drug_discovery_available = True
        except ImportError:
            _drug_discovery_available = False
    return _drug_discovery_available


try:
    from lobster.services.drug_discovery.molecular_analysis_service import (
        RDKIT_AVAILABLE,
        MolecularAnalysisService,
    )
except ImportError:
    RDKIT_AVAILABLE = False
    MolecularAnalysisService = None

try:
    from lobster.services.drug_discovery.admet_prediction_service import (
        ADMETPredictionService,
    )

    ADMET_AVAILABLE = True
except ImportError:
    ADMET_AVAILABLE = False
    ADMETPredictionService = None


# ---------------------------------------------------------------------------
# Service fixtures (module-scoped: one instance per test module)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pubchem():
    if not _check_drug_discovery():
        pytest.skip("lobster-drug-discovery not installed")
    from lobster.services.drug_discovery.pubchem_service import PubChemService

    return PubChemService()


@pytest.fixture(scope="module")
def chembl():
    if not _check_drug_discovery():
        pytest.skip("lobster-drug-discovery not installed")
    from lobster.services.drug_discovery.chembl_service import ChEMBLService

    return ChEMBLService()


@pytest.fixture(scope="module")
def ot():
    if not _check_drug_discovery():
        pytest.skip("lobster-drug-discovery not installed")
    from lobster.services.drug_discovery.opentargets_service import OpenTargetsService

    return OpenTargetsService()


@pytest.fixture(scope="module")
def scorer():
    if not _check_drug_discovery():
        pytest.skip("lobster-drug-discovery not installed")
    from lobster.services.drug_discovery.target_scoring_service import (
        TargetScoringService,
    )

    return TargetScoringService()


@pytest.fixture(scope="module")
def syn():
    if not _check_drug_discovery():
        pytest.skip("lobster-drug-discovery not installed")
    from lobster.services.drug_discovery.synergy_scoring_service import (
        SynergyScoringService,
    )

    return SynergyScoringService()


@pytest.fixture(scope="module")
def mol_svc():
    if not RDKIT_AVAILABLE:
        pytest.skip("RDKit not installed")
    return MolecularAnalysisService()


@pytest.fixture(scope="module")
def admet_svc():
    if not ADMET_AVAILABLE:
        pytest.skip("ADMET service not available")
    return ADMETPredictionService()


# ---------------------------------------------------------------------------
# Known-answer reference data
# ---------------------------------------------------------------------------

# PubChem published reference values (source: PubChem Compound pages)
KNOWN_DRUGS = {
    "aspirin": {
        "mw": 180.16,
        "mw_tol": 0.5,
        "hbd": 1,
        "hba": 4,
        "lipinski_compliant": True,
    },
    "imatinib": {
        "mw": 493.6,
        "mw_tol": 1.0,
        "hbd": 2,
        "hba": 7,
        "lipinski_compliant": True,
    },
    "metformin": {
        "mw": 129.16,
        "mw_tol": 0.5,
        "lipinski_compliant": True,
    },
    "cyclosporine": {
        "mw_range": (1190, 1210),
        "lipinski_compliant": False,
    },
}

# SMILES for known molecules (from PubChem canonical SMILES)
KNOWN_SMILES = {
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "imatinib": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
    "metformin": "CN(C)C(=N)NC(=N)N",
    "erlotinib": "C=CC(=O)NC1=CC2=C(C=C1)N=CN=C2NC3=CC(=C(C=C3)OC)OCCCOC",
    "gefitinib": "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
}

# Ensembl IDs for known drug targets
KNOWN_TARGETS = {
    "EGFR": "ENSG00000146648",
    "BRAF": "ENSG00000157764",
    "ABL1": "ENSG00000097007",
    "TP53": "ENSG00000141510",
    "BRCA1": "ENSG00000012048",
    "ALK": "ENSG00000171094",
    "KRAS": "ENSG00000133703",
}
