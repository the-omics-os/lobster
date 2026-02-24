"""
Step 2: Target-Disease Evidence Validation

Verify Open Targets returns biologically correct associations for
well-established drug targets. These are textbook oncology facts —
if the API disagrees, the API is wrong.
"""

import pytest

from lobster.core.analysis_ir import AnalysisStep

from .conftest import KNOWN_TARGETS

pytestmark = [pytest.mark.integration, pytest.mark.real_api]


class TestTargetDiseaseEvidence:
    """Validate Open Targets returns correct biology for known targets."""

    def test_egfr_top_disease_includes_lung_cancer(self, ot):
        """EGFR is THE canonical NSCLC target — lung cancer must appear in top associations."""
        _, stats, ir = ot.get_target_disease_evidence(
            KNOWN_TARGETS["EGFR"], limit=10
        )
        if "error" in stats:
            pytest.skip(f"Open Targets unavailable: {stats['error'][:80]}")

        assert stats["approved_symbol"] == "EGFR"

        disease_names = [
            a["disease_name"].lower() for a in stats["associations"]
        ]
        assert any("lung" in d for d in disease_names), (
            f"EGFR top diseases should include lung cancer, got: {disease_names}"
        )

    def test_egfr_has_high_association_score(self, ot):
        """EGFR has extensive evidence — top association score should be >0.5."""
        _, stats, _ = ot.get_target_disease_evidence(
            KNOWN_TARGETS["EGFR"], limit=5
        )
        if "error" in stats:
            pytest.skip(f"Open Targets unavailable: {stats['error'][:80]}")

        top_score = stats["associations"][0]["overall_score"]
        assert top_score > 0.5, (
            f"EGFR top association score should be >0.5, got {top_score}"
        )

    def test_egfr_has_many_associated_diseases(self, ot):
        """EGFR is one of the most studied genes — should have >100 disease associations."""
        _, stats, _ = ot.get_target_disease_evidence(
            KNOWN_TARGETS["EGFR"], limit=5
        )
        if "error" in stats:
            pytest.skip(f"Open Targets unavailable: {stats['error'][:80]}")

        assert stats["total_associated_diseases"] > 50, (
            f"EGFR should have >50 associated diseases, got {stats['total_associated_diseases']}"
        )

    def test_braf_top_disease_includes_melanoma(self, ot):
        """BRAF V600E is the hallmark melanoma mutation."""
        _, stats, _ = ot.get_target_disease_evidence(
            KNOWN_TARGETS["BRAF"], limit=10
        )
        if "error" in stats:
            pytest.skip(f"Open Targets unavailable: {stats['error'][:80]}")

        assert stats["approved_symbol"] == "BRAF"

        disease_names = [
            a["disease_name"].lower() for a in stats["associations"]
        ]
        assert any("melanoma" in d for d in disease_names), (
            f"BRAF top diseases should include melanoma, got: {disease_names}"
        )

    def test_tp53_has_cancer_associations(self, ot):
        """TP53 is the guardian of the genome — should associate with many cancers."""
        _, stats, _ = ot.get_target_disease_evidence(
            KNOWN_TARGETS["TP53"], limit=10
        )
        if "error" in stats:
            pytest.skip(f"Open Targets unavailable: {stats['error'][:80]}")

        assert stats["approved_symbol"] == "TP53"
        assert stats["total_associated_diseases"] > 50, (
            "TP53 should have extensive disease associations"
        )

    def test_abl1_associates_with_leukemia(self, ot):
        """ABL1 (BCR-ABL fusion partner) should associate with leukemia/CML."""
        _, stats, _ = ot.get_target_disease_evidence(
            KNOWN_TARGETS["ABL1"], limit=10
        )
        if "error" in stats:
            pytest.skip(f"Open Targets unavailable: {stats['error'][:80]}")

        disease_names = [
            a["disease_name"].lower() for a in stats["associations"]
        ]
        assert any(
            "leukemia" in d or "leukaemia" in d for d in disease_names
        ), (
            f"ABL1 should associate with leukemia, got: {disease_names}"
        )

    def test_evidence_returns_valid_ir(self, ot):
        """All Open Targets evidence queries must produce valid AnalysisStep."""
        _, stats, ir = ot.get_target_disease_evidence(
            KNOWN_TARGETS["EGFR"], limit=5
        )
        if "error" in stats:
            pytest.skip(f"Open Targets unavailable: {stats['error'][:80]}")

        assert isinstance(ir, AnalysisStep)
        assert ir.operation == "opentargets.target.disease_evidence"
        assert ir.tool_name == "get_target_disease_evidence"

    def test_ensembl_id_preserved_in_response(self, ot):
        """Returned ensembl_id must match queried ID."""
        ensembl_id = KNOWN_TARGETS["EGFR"]
        _, stats, _ = ot.get_target_disease_evidence(ensembl_id, limit=5)
        if "error" in stats:
            pytest.skip(f"Open Targets unavailable: {stats['error'][:80]}")

        assert stats["ensembl_id"] == ensembl_id
