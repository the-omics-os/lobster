"""Unit tests for ADMETPredictionService."""

import pytest

from lobster.services.drug_discovery.admet_prediction_service import (
    RDKIT_AVAILABLE,
    ADMETPredictionService,
)

if RDKIT_AVAILABLE:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski


def _expected_delaney_log_s(smiles: str) -> float:
    """Delaney (2004) ESOL equation: JCICS 44(3):1000-1005."""
    mol = Chem.MolFromSmiles(smiles)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)
    n_aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    aromatic_proportion = n_aromatic_atoms / max(mol.GetNumHeavyAtoms(), 1)
    return round(
        0.16
        - 0.63 * logp
        - 0.0062 * mw
        + 0.066 * rot_bonds
        - 0.74 * aromatic_proportion,
        2,
    )


@pytest.mark.unit
@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
class TestADMETDelaneyESOL:
    """Verify Delaney ESOL solubility uses aromatic atom proportion."""

    @pytest.mark.parametrize(
        ("smiles", "expected_class"),
        [
            ("c1ccccc1", "soluble"),  # benzene: 6/6 aromatic atoms -> AP = 1.0
            ("c1ccc2ccccc2c1", "moderately_soluble"),  # naphthalene: 10/10 -> AP = 1.0
            ("CC(=O)Oc1ccccc1C(=O)O", "soluble"),  # aspirin: 6/13 aromatic atoms
            ("CC(C)Cc1ccc(C(C)C(=O)O)cc1", "moderately_soluble"),  # ibuprofen: 6/15
            ("CCO", "highly_soluble"),  # ethanol: 0/3 aromatic atoms
        ],
    )
    def test_estimated_log_solubility_matches_delaney_esol(
        self, smiles: str, expected_class: str
    ):
        service = ADMETPredictionService()
        _, result, _ = service.predict_admet(smiles)
        absorption = result["predictions"]["absorption"]

        assert absorption["estimated_log_solubility"] == pytest.approx(
            _expected_delaney_log_s(smiles), abs=1e-6
        )
        assert absorption["solubility_class"] == expected_class
