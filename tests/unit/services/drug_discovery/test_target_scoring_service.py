"""
Unit tests for TargetScoringService.

Tests the target druggability scoring system with real computation:
- Weighted evidence scoring with known inputs
- Partial evidence handling (missing categories default to 0.0)
- Boundary value validation
- Multi-target ranking with correct sort order
- Confidence classification at thresholds
- 3-tuple return contract (None, Dict, AnalysisStep)
"""

import pytest

from lobster.agents.drug_discovery.config import TARGET_EVIDENCE_WEIGHTS
from lobster.core.analysis_ir import AnalysisStep
from lobster.services.drug_discovery.target_scoring_service import (
    TargetScoringError,
    TargetScoringService,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def service():
    """Create a fresh TargetScoringService instance."""
    return TargetScoringService()


# =============================================================================
# SCORE_TARGET TESTS
# =============================================================================


class TestScoreTarget:
    """Tests for TargetScoringService.score_target()."""

    def test_full_evidence_weighted_sum(self, service):
        """score_target with all 5 evidence categories returns correct weighted sum.

        Evidence:
            genetic_association=0.8 * 0.30 = 0.240
            known_drug=0.6 * 0.25 = 0.150
            expression_specificity=0.5 * 0.20 = 0.100
            pathogenicity=0.3 * 0.15 = 0.045
            literature=0.7 * 0.10 = 0.070
            Total = 0.605
        """
        evidence = {
            "genetic_association": 0.8,
            "known_drug": 0.6,
            "expression_specificity": 0.5,
            "pathogenicity": 0.3,
            "literature": 0.7,
        }
        adata, result, ir = service.score_target(evidence)

        # First element must be None (no AnnData)
        assert adata is None

        # Verify overall score
        expected_score = (
            0.8 * 0.30 + 0.6 * 0.25 + 0.5 * 0.20 + 0.3 * 0.15 + 0.7 * 0.10
        )
        assert abs(result["overall_score"] - expected_score) < 1e-4, (
            f"Expected {expected_score}, got {result['overall_score']}"
        )

        # Verify classification
        assert result["classification"] == "medium_confidence"

        # Verify component breakdown exists
        assert "component_scores" in result
        for cat in TARGET_EVIDENCE_WEIGHTS:
            assert cat in result["component_scores"]
            comp = result["component_scores"][cat]
            assert "raw_score" in comp
            assert "weight" in comp
            assert "weighted_score" in comp

        # Verify strongest/weakest evidence
        assert result["strongest_evidence"] == "genetic_association"
        assert result["weakest_evidence"] == "pathogenicity"

        # Verify no missing categories
        assert result["evidence_missing"] == []

    def test_partial_evidence_missing_categories_default_zero(self, service):
        """score_target with partial evidence treats missing categories as 0.0.

        Only genetic_association=0.9 provided.
        Score = 0.9 * 0.30 = 0.270
        """
        evidence = {"genetic_association": 0.9}
        _, result, ir = service.score_target(evidence)

        expected_score = 0.9 * 0.30
        assert abs(result["overall_score"] - expected_score) < 1e-4

        # Should report missing categories
        missing = result["evidence_missing"]
        assert "known_drug" in missing
        assert "expression_specificity" in missing
        assert "pathogenicity" in missing
        assert "literature" in missing
        assert len(missing) == 4

    def test_all_zeros_returns_zero_score(self, service):
        """score_target with all zeros returns 0.0 overall score."""
        evidence = {
            "genetic_association": 0.0,
            "known_drug": 0.0,
            "expression_specificity": 0.0,
            "pathogenicity": 0.0,
            "literature": 0.0,
        }
        _, result, _ = service.score_target(evidence)
        assert result["overall_score"] == 0.0
        assert result["classification"] == "low_confidence"

    def test_all_ones_returns_max_score(self, service):
        """score_target with all 1.0 values returns maximum possible score.

        Maximum = sum of all weights = 1.0
        """
        evidence = {
            "genetic_association": 1.0,
            "known_drug": 1.0,
            "expression_specificity": 1.0,
            "pathogenicity": 1.0,
            "literature": 1.0,
        }
        _, result, _ = service.score_target(evidence)

        max_possible = sum(TARGET_EVIDENCE_WEIGHTS.values())
        assert abs(result["overall_score"] - max_possible) < 1e-4
        assert result["classification"] == "high_confidence"

    def test_empty_evidence_returns_zero(self, service):
        """score_target with empty dict returns 0.0 (all categories missing)."""
        _, result, _ = service.score_target({})
        assert result["overall_score"] == 0.0
        assert result["classification"] == "low_confidence"
        assert len(result["evidence_missing"]) == 5

    def test_validates_value_above_one(self, service):
        """score_target raises TargetScoringError for values > 1.0."""
        evidence = {"genetic_association": 1.5}
        with pytest.raises(TargetScoringError, match="must be in.*0.0.*1.0"):
            service.score_target(evidence)

    def test_validates_value_below_zero(self, service):
        """score_target raises TargetScoringError for values < 0.0."""
        evidence = {"known_drug": -0.1}
        with pytest.raises(TargetScoringError, match="must be in.*0.0.*1.0"):
            service.score_target(evidence)

    def test_validates_non_numeric_value(self, service):
        """score_target raises TargetScoringError for non-numeric values."""
        evidence = {"genetic_association": "high"}
        with pytest.raises(TargetScoringError, match="must be numeric"):
            service.score_target(evidence)

    def test_ignores_unknown_keys(self, service):
        """score_target ignores unknown evidence categories (with warning)."""
        evidence = {
            "genetic_association": 0.8,
            "unknown_category": 0.5,
        }
        # Should not raise, but should warn
        _, result, _ = service.score_target(evidence)

        # Score should only include genetic_association
        expected_score = 0.8 * 0.30
        assert abs(result["overall_score"] - expected_score) < 1e-4

    def test_returns_analysis_step_ir(self, service):
        """score_target third element must be an AnalysisStep."""
        evidence = {"genetic_association": 0.5}
        _, _, ir = service.score_target(evidence)

        assert isinstance(ir, AnalysisStep)
        assert ir.tool_name == "score_target"
        assert "druggability" in ir.description.lower() or "score" in ir.description.lower()

    def test_integer_values_accepted(self, service):
        """score_target should accept integer values (0 and 1)."""
        evidence = {"genetic_association": 1, "known_drug": 0}
        _, result, _ = service.score_target(evidence)

        expected_score = 1.0 * 0.30 + 0.0 * 0.25
        assert abs(result["overall_score"] - expected_score) < 1e-4

    def test_boundary_value_zero(self, service):
        """score_target with exactly 0.0 is valid."""
        evidence = {"genetic_association": 0.0}
        _, result, _ = service.score_target(evidence)
        assert result["overall_score"] == 0.0

    def test_boundary_value_one(self, service):
        """score_target with exactly 1.0 is valid."""
        evidence = {"genetic_association": 1.0}
        _, result, _ = service.score_target(evidence)
        expected = 1.0 * 0.30
        assert abs(result["overall_score"] - expected) < 1e-4

    def test_weights_sum_to_one(self, service):
        """TARGET_EVIDENCE_WEIGHTS should sum to 1.0 for proper normalization."""
        total = sum(TARGET_EVIDENCE_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-10, (
            f"TARGET_EVIDENCE_WEIGHTS sum to {total}, expected 1.0"
        )


# =============================================================================
# RANK_TARGETS TESTS
# =============================================================================


class TestRankTargets:
    """Tests for TargetScoringService.rank_targets()."""

    def test_sorts_correctly_descending(self, service):
        """rank_targets sorts targets by descending overall score."""
        targets = [
            ("GENE_LOW", {"genetic_association": 0.1}),
            ("GENE_HIGH", {"genetic_association": 0.9}),
            ("GENE_MED", {"genetic_association": 0.5}),
        ]
        _, result, ir = service.rank_targets(targets)

        ranked = result["ranked_targets"]
        assert len(ranked) == 3
        assert ranked[0]["gene_symbol"] == "GENE_HIGH"
        assert ranked[1]["gene_symbol"] == "GENE_MED"
        assert ranked[2]["gene_symbol"] == "GENE_LOW"

        # Verify rank numbers
        assert ranked[0]["rank"] == 1
        assert ranked[1]["rank"] == 2
        assert ranked[2]["rank"] == 3

    def test_empty_list_raises_error(self, service):
        """rank_targets with empty list raises TargetScoringError."""
        with pytest.raises(TargetScoringError, match="Empty targets list"):
            service.rank_targets([])

    def test_single_target(self, service):
        """rank_targets with a single target works correctly."""
        targets = [("BRAF", {"genetic_association": 0.8, "known_drug": 0.7})]
        _, result, _ = service.rank_targets(targets)

        assert result["n_targets"] == 1
        ranked = result["ranked_targets"]
        assert len(ranked) == 1
        assert ranked[0]["gene_symbol"] == "BRAF"
        assert ranked[0]["rank"] == 1

    def test_mean_score_calculation(self, service):
        """rank_targets computes correct mean score."""
        targets = [
            ("A", {"genetic_association": 0.6}),  # 0.6 * 0.3 = 0.18
            ("B", {"genetic_association": 0.4}),  # 0.4 * 0.3 = 0.12
        ]
        _, result, _ = service.rank_targets(targets)

        expected_mean = (0.18 + 0.12) / 2
        assert abs(result["mean_score"] - expected_mean) < 1e-4

    def test_summary_classification_counts(self, service):
        """rank_targets summary correctly counts confidence tiers."""
        targets = [
            # high_confidence: score > 0.7 requires high evidence
            ("HIGH", {
                "genetic_association": 1.0,
                "known_drug": 1.0,
                "expression_specificity": 1.0,
                "pathogenicity": 1.0,
                "literature": 1.0,
            }),
            # medium_confidence: score > 0.4
            ("MED", {
                "genetic_association": 0.8,
                "known_drug": 0.6,
                "expression_specificity": 0.5,
            }),
            # low_confidence: score <= 0.4
            ("LOW", {"literature": 0.1}),
        ]
        _, result, _ = service.rank_targets(targets)

        summary = result["summary"]
        assert summary["high_confidence"] == 1
        assert summary["medium_confidence"] == 1
        assert summary["low_confidence"] == 1

    def test_best_target_identified(self, service):
        """rank_targets correctly identifies best_target and best_score."""
        targets = [
            ("EGFR", {"genetic_association": 0.5}),
            ("BRAF", {"genetic_association": 0.9}),
        ]
        _, result, _ = service.rank_targets(targets)

        assert result["best_target"] == "BRAF"
        expected_best = 0.9 * 0.30
        assert abs(result["best_score"] - expected_best) < 1e-4

    def test_returns_none_dict_analysistep(self, service):
        """rank_targets returns (None, Dict, AnalysisStep) tuple."""
        targets = [("A", {"genetic_association": 0.5})]
        adata, result, ir = service.rank_targets(targets)

        assert adata is None
        assert isinstance(result, dict)
        assert isinstance(ir, AnalysisStep)
        assert ir.tool_name == "rank_targets"

    def test_tie_breaking_alphabetical(self, service):
        """rank_targets breaks ties alphabetically by gene symbol."""
        targets = [
            ("ZETA", {"genetic_association": 0.5}),
            ("ALPHA", {"genetic_association": 0.5}),
        ]
        _, result, _ = service.rank_targets(targets)

        ranked = result["ranked_targets"]
        # Same score, so alphabetical: ALPHA first
        assert ranked[0]["gene_symbol"] == "ALPHA"
        assert ranked[1]["gene_symbol"] == "ZETA"

    def test_validates_evidence_per_target(self, service):
        """rank_targets validates evidence for each target."""
        targets = [
            ("GOOD", {"genetic_association": 0.5}),
            ("BAD", {"genetic_association": 2.0}),  # out of range
        ]
        with pytest.raises(TargetScoringError, match="must be in.*0.0.*1.0"):
            service.rank_targets(targets)


# =============================================================================
# CLASSIFY_TARGET TESTS
# =============================================================================


class TestClassifyTarget:
    """Tests for TargetScoringService.classify_target()."""

    def test_high_confidence_above_0_7(self, service):
        """Scores > 0.7 are classified as high_confidence."""
        assert service.classify_target(0.71) == "high_confidence"
        assert service.classify_target(0.9) == "high_confidence"
        assert service.classify_target(1.0) == "high_confidence"

    def test_medium_confidence_between_0_4_and_0_7(self, service):
        """Scores > 0.4 and <= 0.7 are classified as medium_confidence."""
        assert service.classify_target(0.41) == "medium_confidence"
        assert service.classify_target(0.5) == "medium_confidence"
        assert service.classify_target(0.7) == "medium_confidence"

    def test_low_confidence_at_or_below_0_4(self, service):
        """Scores <= 0.4 are classified as low_confidence."""
        assert service.classify_target(0.4) == "low_confidence"
        assert service.classify_target(0.2) == "low_confidence"
        assert service.classify_target(0.0) == "low_confidence"

    def test_boundary_0_7_is_medium(self, service):
        """Score of exactly 0.7 is medium_confidence (boundary: > 0.7 for high)."""
        assert service.classify_target(0.7) == "medium_confidence"

    def test_boundary_0_4_is_low(self, service):
        """Score of exactly 0.4 is low_confidence (boundary: > 0.4 for medium)."""
        assert service.classify_target(0.4) == "low_confidence"

    def test_just_above_0_7(self, service):
        """Score just above 0.7 is high_confidence."""
        assert service.classify_target(0.7001) == "high_confidence"

    def test_just_above_0_4(self, service):
        """Score just above 0.4 is medium_confidence."""
        assert service.classify_target(0.4001) == "medium_confidence"
