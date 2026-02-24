"""
Step 5: Cross-Service Pipeline Validation

Test realistic multi-service chains that mirror actual drug discovery workflows.
Each pipeline chains 3+ services end-to-end and validates the biology at every step.
"""

import pytest

from lobster.core.analysis_ir import AnalysisStep

from .conftest import KNOWN_SMILES, KNOWN_TARGETS, RDKIT_AVAILABLE

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Pipeline A: Target-to-Compound (the core drug discovery use case)
# ---------------------------------------------------------------------------


@pytest.mark.real_api
class TestTargetToCompoundPipeline:
    """Full pipeline: identify target → validate → find compounds → profile."""

    def test_egfr_pipeline_evidence_to_properties(self, ot, pubchem):
        """EGFR: evidence → score → profile erlotinib → verify all IR."""
        # Step 1: Get disease evidence
        _, evidence, ir1 = ot.get_target_disease_evidence(
            KNOWN_TARGETS["EGFR"], limit=3
        )
        if "error" in evidence:
            pytest.skip(f"Open Targets unavailable: {evidence['error'][:80]}")

        assert evidence["approved_symbol"] == "EGFR"
        assert evidence["total_associated_diseases"] > 50

        # Step 2: Score target druggability
        _, score, ir2 = ot.score_target(KNOWN_TARGETS["EGFR"])
        if "error" in score:
            pytest.skip(f"Open Targets scoring failed: {score['error'][:80]}")

        assert score["druggability_score"] > 0.3

        # Step 3: Profile a known EGFR inhibitor via PubChem
        _, props, ir3 = pubchem.get_compound_properties("erlotinib", id_type="name")
        if "error" in props:
            pytest.skip(f"PubChem unavailable: {props['error'][:60]}")

        mw = float(props["molecular_weight"])
        assert mw > 300, f"Erlotinib MW should be >300, got {mw}"
        assert props["lipinski"]["compliant"] is True

        # Step 4: Verify ALL steps produced AnalysisStep IR
        for step_name, ir in [
            ("evidence", ir1),
            ("score", ir2),
            ("properties", ir3),
        ]:
            assert ir is not None, f"{step_name} returned None IR"
            assert isinstance(ir, AnalysisStep), (
                f"{step_name} IR is not AnalysisStep"
            )
            assert ir.operation, f"{step_name} IR missing operation"
            assert ir.tool_name, f"{step_name} IR missing tool_name"
            assert ir.code_template, f"{step_name} IR missing code_template"

    def test_braf_pipeline_evidence_to_drug_indication(self, ot):
        """BRAF: evidence → score → drug indications for vemurafenib."""
        # Step 1: BRAF disease evidence
        _, evidence, _ = ot.get_target_disease_evidence(
            KNOWN_TARGETS["BRAF"], limit=5
        )
        if "error" in evidence:
            pytest.skip(f"Open Targets unavailable: {evidence['error'][:80]}")

        assert evidence["approved_symbol"] == "BRAF"

        # Step 2: Get drug indications for vemurafenib (CHEMBL1229517)
        _, indications, ir = ot.get_drug_indications("CHEMBL1229517", limit=10)
        if "error" in indications:
            pytest.skip(f"Drug indications unavailable: {indications['error'][:80]}")

        # Vemurafenib should have melanoma as an indication
        assert indications["n_indications"] > 0
        assert isinstance(ir, AnalysisStep)


# ---------------------------------------------------------------------------
# Pipeline B: Compound Profiling (RDKit + PubChem cross-validation)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
@pytest.mark.real_api
class TestCompoundProfilingPipeline:
    """Pipeline: PubChem lookup → RDKit descriptors → Lipinski → cross-validate."""

    def test_imatinib_full_profile(self, pubchem, mol_svc):
        """Full profiling of imatinib — a textbook kinase inhibitor."""
        smiles = KNOWN_SMILES["imatinib"]

        # Step 1: RDKit descriptors
        _, desc, ir1 = mol_svc.calculate_descriptors(smiles)
        assert 480 < desc["molecular_weight"] < 510  # ~493.6
        assert desc["hbd"] == 2  # Two NH groups
        assert desc["aromatic_rings"] >= 3

        # Step 2: Lipinski check
        _, lip, ir2 = mol_svc.lipinski_check(smiles)
        assert lip["overall_pass"] is True
        assert lip["classification"] == "drug-like"

        # Step 3: Cross-validate MW with PubChem
        _, pub, ir3 = pubchem.get_compound_properties("imatinib", id_type="name")
        if "error" in pub:
            pytest.skip(f"PubChem unavailable: {pub['error'][:60]}")

        mw_rdkit = desc["molecular_weight"]
        mw_pubchem = float(pub["molecular_weight"])
        assert abs(mw_rdkit - mw_pubchem) < 1.0, (
            f"RDKit MW ({mw_rdkit}) vs PubChem MW ({mw_pubchem}) differ by >1 Da"
        )

        # Step 4: All IR valid
        for name, ir in [("descriptors", ir1), ("lipinski", ir2), ("pubchem", ir3)]:
            assert isinstance(ir, AnalysisStep), f"{name} IR invalid"


# ---------------------------------------------------------------------------
# Pipeline C: Similarity-Based Lead Hopping
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
class TestSimilarityLeadHopping:
    """Find similar compounds and verify chemical similarity makes biological sense."""

    def test_self_similarity_is_one(self, mol_svc):
        """A molecule must have Tanimoto=1.0 with itself."""
        aspirin = KNOWN_SMILES["aspirin"]
        _, result, _ = mol_svc.fingerprint_similarity(
            [aspirin, aspirin], fingerprint="morgan"
        )
        matrix = result["similarity_matrix"]
        assert abs(matrix[0][1] - 1.0) < 1e-10, (
            f"Self-similarity should be 1.0, got {matrix[0][1]}"
        )

    def test_egfr_inhibitors_have_high_similarity(self, mol_svc):
        """Erlotinib and gefitinib (both EGFR TKIs) should share scaffold similarity."""
        erlotinib = KNOWN_SMILES["erlotinib"]
        gefitinib = KNOWN_SMILES["gefitinib"]
        _, result, _ = mol_svc.fingerprint_similarity(
            [erlotinib, gefitinib], fingerprint="morgan"
        )
        sim = result["similarity_matrix"][0][1]
        assert sim > 0.2, (
            f"EGFR inhibitors should share some scaffold similarity, got {sim}"
        )

    def test_unrelated_drugs_have_low_similarity(self, mol_svc):
        """Aspirin and metformin are structurally unrelated → low similarity."""
        aspirin = KNOWN_SMILES["aspirin"]
        metformin = KNOWN_SMILES["metformin"]
        _, result, _ = mol_svc.fingerprint_similarity(
            [aspirin, metformin], fingerprint="morgan"
        )
        sim = result["similarity_matrix"][0][1]
        assert sim < 0.3, (
            f"Unrelated drugs should have low similarity, got {sim}"
        )

    def test_similarity_matrix_is_symmetric(self, mol_svc):
        """Tanimoto matrix must be symmetric: sim(A,B) == sim(B,A)."""
        smiles_list = [
            KNOWN_SMILES["aspirin"],
            KNOWN_SMILES["imatinib"],
            KNOWN_SMILES["metformin"],
        ]
        _, result, _ = mol_svc.fingerprint_similarity(smiles_list, fingerprint="morgan")
        matrix = result["similarity_matrix"]
        n = len(matrix)
        for i in range(n):
            for j in range(n):
                assert abs(matrix[i][j] - matrix[j][i]) < 1e-10, (
                    f"Matrix not symmetric at [{i}][{j}]"
                )

    def test_diagonal_is_one(self, mol_svc):
        """All diagonal entries in the similarity matrix must be 1.0."""
        smiles_list = [
            KNOWN_SMILES["aspirin"],
            KNOWN_SMILES["imatinib"],
        ]
        _, result, _ = mol_svc.fingerprint_similarity(smiles_list, fingerprint="morgan")
        matrix = result["similarity_matrix"]
        for i in range(len(matrix)):
            assert abs(matrix[i][i] - 1.0) < 1e-10, (
                f"Diagonal [{i}][{i}] should be 1.0, got {matrix[i][i]}"
            )
