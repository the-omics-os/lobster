"""
Step 7: ChEMBL Bioactivity Validation

Verify ChEMBL returns correct bioactivity data for known drug-target pairs.
These are well-established pharmacological facts.
"""

import pytest

from lobster.core.analysis_ir import AnalysisStep

pytestmark = [pytest.mark.integration, pytest.mark.real_api]


class TestChEMBLCompoundSearch:
    """Validate ChEMBL compound search returns real compounds."""

    def test_search_vemurafenib_returns_results(self, chembl):
        """Vemurafenib is a well-known drug — must return at least 1 result."""
        _, stats, ir = chembl.search_compounds("vemurafenib", limit=5)
        if "error" in stats:
            pytest.skip(f"ChEMBL API unavailable: {stats['error'][:60]}")

        compounds = stats.get("compounds", [])
        assert len(compounds) > 0, "Vemurafenib should return at least 1 compound"

        # All returned compounds should have valid ChEMBL IDs
        for c in compounds:
            assert c["chembl_id"].startswith("CHEMBL"), (
                f"Invalid ChEMBL ID: {c['chembl_id']}"
            )

        assert isinstance(ir, AnalysisStep)

    def test_search_imatinib_returns_results(self, chembl):
        """Imatinib (Gleevec) — a blockbuster drug — must be findable."""
        _, stats, _ = chembl.search_compounds("imatinib", limit=5)
        if "error" in stats:
            pytest.skip(f"ChEMBL API unavailable: {stats['error'][:60]}")

        compounds = stats.get("compounds", [])
        assert len(compounds) > 0, "Imatinib should return at least 1 compound"

    def test_search_returns_ir(self, chembl):
        """ChEMBL search must produce valid AnalysisStep."""
        _, stats, ir = chembl.search_compounds("aspirin", limit=3)
        assert isinstance(ir, AnalysisStep)
        assert ir.operation == "chembl.molecule.search"


class TestChEMBLBioactivity:
    """Validate bioactivity data for known drug-target pairs."""

    def test_imatinib_has_bioactivity(self, chembl):
        """Imatinib (CHEMBL941) must have bioactivity records."""
        _, stats, _ = chembl.get_bioactivity("CHEMBL941")
        if "error" in stats:
            pytest.skip(f"ChEMBL API unavailable: {stats['error'][:60]}")

        activities = stats.get("activities", [])
        assert len(activities) > 0, (
            "Imatinib should have bioactivity records in ChEMBL"
        )

    def test_imatinib_targets_include_kinase(self, chembl):
        """Imatinib targets ABL1 kinase — target names should include kinase-related terms."""
        _, stats, _ = chembl.get_bioactivity("CHEMBL941")
        if "error" in stats:
            pytest.skip(f"ChEMBL API unavailable: {stats['error'][:60]}")

        activities = stats.get("activities", [])
        if not activities:
            pytest.skip("No activities returned for imatinib")

        target_names = [
            a.get("target_pref_name", "").lower() for a in activities
        ]
        # ABL1, BCR-ABL, c-Kit, PDGFR are all imatinib targets
        has_kinase_target = any(
            any(term in t for term in ["abl", "kit", "pdgf", "kinase"])
            for t in target_names
            if t
        )
        assert has_kinase_target, (
            f"Imatinib should target kinases (ABL/Kit/PDGFR), got: {target_names[:5]}"
        )

    def test_bioactivity_returns_valid_ir(self, chembl):
        """Bioactivity queries must produce valid AnalysisStep."""
        _, stats, ir = chembl.get_bioactivity("CHEMBL941")
        assert isinstance(ir, AnalysisStep)
        assert ir.operation == "chembl.activity.get"
        assert ir.tool_name == "get_bioactivity"


class TestChEMBLTargetCompounds:
    """Validate target-compound association queries."""

    def test_egfr_has_active_compounds(self, chembl):
        """EGFR (CHEMBL203) should have many active compounds — it's a major drug target."""
        _, stats, _ = chembl.get_target_compounds(
            "CHEMBL203", activity_type="IC50", limit=10
        )
        if "error" in stats:
            pytest.skip(f"ChEMBL API unavailable: {stats['error'][:60]}")

        compounds = stats.get("compounds", [])
        assert len(compounds) > 0, (
            "EGFR should have active compounds in ChEMBL"
        )

    def test_target_compounds_returns_valid_ir(self, chembl):
        """Target-compound queries must produce valid AnalysisStep."""
        _, stats, ir = chembl.get_target_compounds(
            "CHEMBL203", activity_type="IC50", limit=5
        )
        assert isinstance(ir, AnalysisStep)
        assert ir.tool_name == "get_target_compounds"
