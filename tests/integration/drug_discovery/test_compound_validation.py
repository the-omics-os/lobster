"""
Step 1: Known-Answer Compound Validation

Verify PubChem + RDKit molecular analysis services return correct properties
for well-characterized drugs with known published values.

Ground truth sourced from PubChem Compound pages and DrugBank.
"""

import pytest

from lobster.core.analysis_ir import AnalysisStep

from .conftest import KNOWN_DRUGS, KNOWN_SMILES, RDKIT_AVAILABLE

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# PubChem Known-Answer Tests (real API)
# ---------------------------------------------------------------------------


@pytest.mark.real_api
class TestPubChemKnownAnswerCompounds:
    """Validate PubChem returns correct properties for well-characterized drugs."""

    def test_aspirin_molecular_weight(self, pubchem):
        """Aspirin MW must match PubChem value within ±0.5 Da."""
        _, stats, ir = pubchem.get_compound_properties("aspirin", id_type="name")
        if "error" in stats:
            pytest.skip(f"PubChem unavailable: {stats['error'][:60]}")

        mw = float(stats["molecular_weight"])
        expected = KNOWN_DRUGS["aspirin"]["mw"]
        assert abs(mw - expected) < 0.5, (
            f"Aspirin MW={mw}, expected ~{expected} (±0.5 Da)"
        )

    def test_aspirin_hbond_counts(self, pubchem):
        """Aspirin HBD=1 (carboxylic OH), HBA=4 (all oxygens)."""
        _, stats, _ = pubchem.get_compound_properties("aspirin", id_type="name")
        if "error" in stats:
            pytest.skip(f"PubChem unavailable: {stats['error'][:60]}")

        assert stats["hbond_donor_count"] == KNOWN_DRUGS["aspirin"]["hbd"]
        assert stats["hbond_acceptor_count"] == KNOWN_DRUGS["aspirin"]["hba"]

    def test_aspirin_lipinski_compliant(self, pubchem):
        """Aspirin (MW=180, small molecule) must pass Lipinski Ro5."""
        _, stats, _ = pubchem.get_compound_properties("aspirin", id_type="name")
        if "error" in stats:
            pytest.skip(f"PubChem unavailable: {stats['error'][:60]}")

        assert stats["lipinski"]["compliant"] is True, (
            f"Aspirin should be Lipinski compliant, got violations={stats['lipinski']['n_violations']}"
        )

    def test_imatinib_molecular_weight(self, pubchem):
        """Imatinib MW must match PubChem value within ±1 Da."""
        _, stats, _ = pubchem.get_compound_properties("imatinib", id_type="name")
        if "error" in stats:
            pytest.skip(f"PubChem unavailable: {stats['error'][:60]}")

        mw = float(stats["molecular_weight"])
        expected = KNOWN_DRUGS["imatinib"]["mw"]
        assert abs(mw - expected) < 1.0, (
            f"Imatinib MW={mw}, expected ~{expected} (±1.0 Da)"
        )

    def test_imatinib_hbond_counts(self, pubchem):
        """Imatinib HBD=2 (two NH groups), HBA=7."""
        _, stats, _ = pubchem.get_compound_properties("imatinib", id_type="name")
        if "error" in stats:
            pytest.skip(f"PubChem unavailable: {stats['error'][:60]}")

        assert stats["hbond_donor_count"] == KNOWN_DRUGS["imatinib"]["hbd"]
        assert stats["hbond_acceptor_count"] == KNOWN_DRUGS["imatinib"]["hba"]

    def test_imatinib_lipinski_compliant(self, pubchem):
        """Imatinib (MW~493, borderline) should still pass Lipinski."""
        _, stats, _ = pubchem.get_compound_properties("imatinib", id_type="name")
        if "error" in stats:
            pytest.skip(f"PubChem unavailable: {stats['error'][:60]}")

        assert stats["lipinski"]["compliant"] is True

    def test_cyclosporine_lipinski_fails(self, pubchem):
        """Cyclosporine (MW=1202, HBA=23) MUST fail Lipinski — it's a macrocycle."""
        _, stats, _ = pubchem.get_compound_properties("cyclosporine", id_type="name")
        if "error" in stats:
            pytest.skip(f"PubChem unavailable: {stats['error'][:60]}")

        mw = float(stats["molecular_weight"])
        assert mw > 1000, f"Cyclosporine MW should be >1000 Da, got {mw}"
        assert stats["lipinski"]["compliant"] is False, (
            "Cyclosporine MUST fail Lipinski (MW=1202, HBA=23)"
        )
        assert stats["lipinski"]["n_violations"] >= 2, (
            f"Cyclosporine should have ≥2 Lipinski violations, got {stats['lipinski']['n_violations']}"
        )

    def test_metformin_molecular_weight(self, pubchem):
        """Metformin MW must match PubChem value within ±0.5 Da."""
        _, stats, _ = pubchem.get_compound_properties("metformin", id_type="name")
        if "error" in stats:
            pytest.skip(f"PubChem unavailable: {stats['error'][:60]}")

        mw = float(stats["molecular_weight"])
        expected = KNOWN_DRUGS["metformin"]["mw"]
        assert abs(mw - expected) < 0.5, (
            f"Metformin MW={mw}, expected ~{expected}"
        )

    def test_pubchem_returns_valid_ir(self, pubchem):
        """PubChem service must produce valid AnalysisStep provenance."""
        _, stats, ir = pubchem.get_compound_properties("aspirin", id_type="name")
        if "error" in stats:
            pytest.skip(f"PubChem unavailable: {stats['error'][:60]}")

        assert isinstance(ir, AnalysisStep)
        assert ir.operation == "pubchem.compound.properties"
        assert ir.tool_name == "get_compound_properties"
        assert ir.code_template is not None
        assert len(ir.code_template) > 0


# ---------------------------------------------------------------------------
# RDKit Cross-Validation Tests (offline)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
class TestRDKitDescriptorCrossValidation:
    """Validate RDKit descriptors for known molecules."""

    def test_aspirin_descriptors(self, mol_svc):
        """Aspirin: MW~180, 1 aromatic ring, 0 stereocenters."""
        _, desc, _ = mol_svc.calculate_descriptors(KNOWN_SMILES["aspirin"])
        assert abs(desc["molecular_weight"] - 180.16) < 1.0
        assert desc["aromatic_rings"] == 1
        assert desc["stereocenters"] == 0

    def test_imatinib_descriptors(self, mol_svc):
        """Imatinib: MW~493, ≥3 aromatic rings, 0 stereocenters."""
        _, desc, _ = mol_svc.calculate_descriptors(KNOWN_SMILES["imatinib"])
        assert abs(desc["molecular_weight"] - 493.6) < 2.0
        assert desc["aromatic_rings"] >= 3
        assert desc["stereocenters"] == 0
        assert desc["hbd"] == 2

    def test_metformin_descriptors(self, mol_svc):
        """Metformin: MW~129, 0 aromatic rings, very small molecule."""
        _, desc, _ = mol_svc.calculate_descriptors(KNOWN_SMILES["metformin"])
        assert abs(desc["molecular_weight"] - 129.16) < 1.0
        assert desc["aromatic_rings"] == 0

    def test_imatinib_lipinski_rdkit(self, mol_svc):
        """Imatinib passes Lipinski via RDKit."""
        _, lip, _ = mol_svc.lipinski_check(KNOWN_SMILES["imatinib"])
        assert lip["overall_pass"] is True
        assert lip["classification"] == "drug-like"


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
@pytest.mark.real_api
class TestPubChemVsRDKitCrossValidation:
    """Cross-validate PubChem MW vs RDKit MW — must agree within ±1 Da."""

    def test_aspirin_mw_agreement(self, pubchem, mol_svc):
        """PubChem and RDKit molecular weights must agree for aspirin."""
        _, pub_stats, _ = pubchem.get_compound_properties("aspirin", id_type="name")
        if "error" in pub_stats:
            pytest.skip(f"PubChem unavailable: {pub_stats['error'][:60]}")

        _, rdkit_desc, _ = mol_svc.calculate_descriptors(KNOWN_SMILES["aspirin"])

        mw_pubchem = float(pub_stats["molecular_weight"])
        mw_rdkit = rdkit_desc["molecular_weight"]
        assert abs(mw_pubchem - mw_rdkit) < 1.0, (
            f"PubChem MW ({mw_pubchem}) vs RDKit MW ({mw_rdkit}) differ by >1 Da"
        )

    def test_imatinib_mw_agreement(self, pubchem, mol_svc):
        """PubChem and RDKit molecular weights must agree for imatinib."""
        _, pub_stats, _ = pubchem.get_compound_properties("imatinib", id_type="name")
        if "error" in pub_stats:
            pytest.skip(f"PubChem unavailable: {pub_stats['error'][:60]}")

        _, rdkit_desc, _ = mol_svc.calculate_descriptors(KNOWN_SMILES["imatinib"])

        mw_pubchem = float(pub_stats["molecular_weight"])
        mw_rdkit = rdkit_desc["molecular_weight"]
        assert abs(mw_pubchem - mw_rdkit) < 1.0, (
            f"PubChem MW ({mw_pubchem}) vs RDKit MW ({mw_rdkit}) differ by >1 Da"
        )
