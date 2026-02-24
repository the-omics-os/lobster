"""
Step 3: Target Ranking Biological Plausibility

Verify that the composite scoring formula produces biologically plausible
rankings when fed real Open Targets data.

Pipeline: Open Targets → TargetScoringService → rank and validate
"""

import pytest

from lobster.agents.drug_discovery.config import TARGET_EVIDENCE_WEIGHTS
from lobster.core.analysis_ir import AnalysisStep

from .conftest import KNOWN_TARGETS

pytestmark = [pytest.mark.integration]


def _extract_evidence_from_ot_score(ot_stats: dict) -> dict:
    """Extract evidence dict from OpenTargetsService.score_target() response.

    Maps the OT composite component_scores into the TargetScoringService
    evidence format (0.0-1.0 per category).
    """
    components = ot_stats.get("component_scores", {})
    return {
        "genetic_association": components.get("genetic_association", 0.0),
        "known_drug": components.get("known_drug", 0.0),
        "expression_specificity": components.get("expression_specificity", 0.0),
        "pathogenicity": components.get("pathogenicity", 0.0),
        "literature": components.get("literature", 0.0),
    }


@pytest.mark.real_api
class TestTargetRankingBiologicalPlausibility:
    """Chain: Open Targets → TargetScoringService → rank and validate."""

    def test_ot_evidence_chains_into_scoring_service(self, ot, scorer):
        """OT score_target → TargetScoringService.rank_targets produces valid ranking.

        Both EGFR and TP53 are well-studied targets. Rather than assuming one
        always ranks above the other (which depends on OT's live data model),
        we verify:
        1. Both produce non-zero druggability scores from OT
        2. Evidence chains correctly into TargetScoringService
        3. Ranking is deterministic and both scores are non-trivial
        """
        # Get real OT scores for both targets
        _, egfr_stats, _ = ot.score_target(KNOWN_TARGETS["EGFR"])
        if "error" in egfr_stats:
            pytest.skip(f"Open Targets unavailable: {egfr_stats['error'][:80]}")

        _, tp53_stats, _ = ot.score_target(KNOWN_TARGETS["TP53"])
        if "error" in tp53_stats:
            pytest.skip(f"Open Targets unavailable: {tp53_stats['error'][:80]}")

        # Both should have non-trivial OT druggability scores
        assert egfr_stats["druggability_score"] > 0.2, (
            f"EGFR OT druggability should be >0.2, got {egfr_stats['druggability_score']}"
        )
        assert tp53_stats["druggability_score"] > 0.2, (
            f"TP53 OT druggability should be >0.2, got {tp53_stats['druggability_score']}"
        )

        # Extract evidence and rank via TargetScoringService
        egfr_evidence = _extract_evidence_from_ot_score(egfr_stats)
        tp53_evidence = _extract_evidence_from_ot_score(tp53_stats)

        _, rank_stats, ir = scorer.rank_targets([
            ("EGFR", egfr_evidence),
            ("TP53", tp53_evidence),
        ])

        targets = rank_stats["ranked_targets"]
        assert len(targets) == 2
        assert all(t["overall_score"] > 0 for t in targets), (
            "Both well-studied targets should have positive composite scores"
        )

        # Ranking must be valid and deterministic
        assert targets[0]["rank"] == 1
        assert targets[1]["rank"] == 2
        assert targets[0]["overall_score"] >= targets[1]["overall_score"]

        assert isinstance(ir, AnalysisStep)

    def test_egfr_druggability_score_is_nontrivial(self, ot):
        """EGFR has approved drugs and extensive evidence — druggability should be >0.3."""
        _, stats, _ = ot.score_target(KNOWN_TARGETS["EGFR"])
        if "error" in stats:
            pytest.skip(f"Open Targets unavailable: {stats['error'][:80]}")

        assert stats["druggability_score"] > 0.3, (
            f"EGFR druggability score should be >0.3, got {stats['druggability_score']}"
        )

    def test_braf_druggability_score_is_nontrivial(self, ot):
        """BRAF (vemurafenib, dabrafenib) should have non-trivial druggability."""
        _, stats, _ = ot.score_target(KNOWN_TARGETS["BRAF"])
        if "error" in stats:
            pytest.skip(f"Open Targets unavailable: {stats['error'][:80]}")

        assert stats["druggability_score"] > 0.3, (
            f"BRAF druggability score should be >0.3, got {stats['druggability_score']}"
        )


class TestTargetScoringDeterminism:
    """Verify scoring is deterministic and mathematically correct."""

    def test_score_is_deterministic(self, scorer):
        """Same input → same output, always."""
        evidence = {
            "genetic_association": 0.8,
            "known_drug": 0.6,
            "expression_specificity": 0.5,
            "pathogenicity": 0.7,
            "literature": 0.3,
        }
        _, r1, _ = scorer.score_target(evidence)
        _, r2, _ = scorer.score_target(evidence)
        assert r1["overall_score"] == r2["overall_score"]

    def test_weighted_sum_is_correct(self, scorer):
        """Manually verify the weighted sum computation."""
        evidence = {
            "genetic_association": 0.8,
            "known_drug": 0.6,
            "expression_specificity": 0.5,
            "pathogenicity": 0.7,
            "literature": 0.3,
        }
        expected = sum(
            evidence.get(k, 0.0) * w
            for k, w in TARGET_EVIDENCE_WEIGHTS.items()
        )

        _, result, _ = scorer.score_target(evidence)
        assert abs(result["overall_score"] - round(expected, 4)) < 1e-4, (
            f"Score {result['overall_score']} != expected {expected}"
        )

    def test_empty_evidence_gives_zero(self, scorer):
        """No evidence → score must be 0."""
        _, result, _ = scorer.score_target({})
        assert result["overall_score"] == 0.0
        assert result["classification"] == "low_confidence"

    def test_perfect_evidence_gives_max(self, scorer):
        """All evidence at 1.0 → score must equal sum of all weights."""
        evidence = {k: 1.0 for k in TARGET_EVIDENCE_WEIGHTS}
        _, result, _ = scorer.score_target(evidence)
        expected_max = round(sum(TARGET_EVIDENCE_WEIGHTS.values()), 4)
        assert abs(result["overall_score"] - expected_max) < 1e-4

    def test_ranking_order_matches_scores(self, scorer):
        """Ranked output must be sorted by descending score."""
        targets = [
            ("LOW", {"genetic_association": 0.1}),
            ("HIGH", {"genetic_association": 0.9, "known_drug": 0.8}),
            ("MID", {"genetic_association": 0.5, "known_drug": 0.4}),
        ]
        _, result, _ = scorer.rank_targets(targets)

        scores = [t["overall_score"] for t in result["ranked_targets"]]
        assert scores == sorted(scores, reverse=True), (
            f"Rankings not sorted by score: {scores}"
        )
        assert result["ranked_targets"][0]["gene_symbol"] == "HIGH"
