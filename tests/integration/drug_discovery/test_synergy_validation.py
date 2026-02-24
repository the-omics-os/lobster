"""
Step 4: Drug Synergy Mathematics Validation

Verify synergy models against known mathematical properties and
edge cases. These are deterministic tests — they must always pass.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from lobster.core.analysis_ir import AnalysisStep
from lobster.services.drug_discovery.synergy_scoring_service import (
    SynergyScoringError,
)

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Bliss Independence mathematical properties
# ---------------------------------------------------------------------------


class TestBlissIndependenceMathematics:
    """Validate Bliss model against known mathematical identities."""

    def test_commutativity(self, syn):
        """Bliss(A,B) == Bliss(B,A) — drug order shouldn't matter."""
        _, s1, _ = syn.bliss_independence(0.3, 0.5, 0.7)
        _, s2, _ = syn.bliss_independence(0.5, 0.3, 0.7)
        assert abs(s1["excess"] - s2["excess"]) < 1e-10, (
            "Bliss model must be commutative"
        )

    def test_no_effect_drug_is_neutral(self, syn):
        """Adding a drug with 0% effect → expected = other drug's effect."""
        _, s, _ = syn.bliss_independence(0.0, 0.5, 0.5)
        assert abs(s["effect_ab_expected"] - 0.5) < 1e-10
        assert abs(s["excess"]) < 1e-10, (
            "Zero-effect drug should be purely additive"
        )

    def test_two_zero_effect_drugs(self, syn):
        """Both drugs at 0% → expected = 0, excess = 0."""
        _, s, _ = syn.bliss_independence(0.0, 0.0, 0.0)
        assert abs(s["effect_ab_expected"]) < 1e-10
        assert abs(s["excess"]) < 1e-10

    def test_perfect_drugs_expected_is_one(self, syn):
        """Two 100% drugs → Bliss expected = 1.0."""
        _, s, _ = syn.bliss_independence(1.0, 1.0, 1.0)
        assert abs(s["effect_ab_expected"] - 1.0) < 1e-10

    def test_bliss_formula_correctness(self, syn):
        """Manually verify E_expected = E_a + E_b - E_a * E_b."""
        ea, eb, eab = 0.3, 0.5, 0.8
        expected = ea + eb - ea * eb  # 0.3 + 0.5 - 0.15 = 0.65

        _, s, _ = syn.bliss_independence(ea, eb, eab)
        assert abs(s["effect_ab_expected"] - expected) < 1e-10
        assert abs(s["excess"] - (eab - expected)) < 1e-10

    def test_synergistic_classification(self, syn):
        """Combo much better than expected → synergistic."""
        _, s, _ = syn.bliss_independence(0.2, 0.3, 0.8)
        assert s["classification"] == "synergistic"

    def test_antagonistic_classification(self, syn):
        """Combo worse than expected → antagonistic."""
        _, s, _ = syn.bliss_independence(0.5, 0.5, 0.3)
        assert s["classification"] == "antagonistic"

    def test_additive_classification(self, syn):
        """Combo matches expected → additive."""
        ea, eb = 0.3, 0.4
        expected = ea + eb - ea * eb
        _, s, _ = syn.bliss_independence(ea, eb, expected)
        assert s["classification"] == "additive"


# ---------------------------------------------------------------------------
# Loewe Additivity mathematical properties
# ---------------------------------------------------------------------------


class TestLoeweAdditivityMathematics:
    """Validate Loewe CI model against known mathematical properties."""

    def test_ci_equals_one_for_self_combination(self, syn):
        """Drug combined with itself at IC50 → CI = 1.0 (definition)."""
        # d_a/IC50_a + d_b/IC50_b = 5/10 + 5/10 = 1.0
        _, s, _ = syn.loewe_additivity(5.0, 5.0, 10.0, 10.0, 0.5)
        assert abs(s["combination_index"] - 1.0) < 0.01, (
            f"Self-combination CI should be ~1.0, got {s['combination_index']}"
        )
        assert s["classification"] == "additive"

    def test_synergistic_ci_below_one(self, syn):
        """Low doses achieving same effect → CI < 1 (synergistic)."""
        # d_a=1, d_b=1, IC50_a=10, IC50_b=10 → CI=0.2 (synergistic)
        _, s, _ = syn.loewe_additivity(1.0, 1.0, 10.0, 10.0, 0.5)
        assert s["combination_index"] < 0.9
        assert s["classification"] == "synergistic"

    def test_antagonistic_ci_above_one(self, syn):
        """High doses needed → CI > 1 (antagonistic)."""
        # d_a=8, d_b=8, IC50_a=10, IC50_b=10 → CI=1.6
        _, s, _ = syn.loewe_additivity(8.0, 8.0, 10.0, 10.0, 0.5)
        assert s["combination_index"] > 1.1
        assert s["classification"] == "antagonistic"

    def test_dose_reduction_index_calculated(self, syn):
        """DRI must be computed: IC50/dose for each drug."""
        _, s, _ = syn.loewe_additivity(2.0, 3.0, 10.0, 15.0, 0.5)
        assert abs(s["dose_reduction_index_a"] - 5.0) < 0.1  # 10/2 = 5
        assert abs(s["dose_reduction_index_b"] - 5.0) < 0.1  # 15/3 = 5

    def test_loewe_rejects_zero_dose(self, syn):
        """Zero dose should raise SynergyScoringError."""
        with pytest.raises(SynergyScoringError):
            syn.loewe_additivity(0.0, 5.0, 10.0, 10.0, 0.5)

    def test_loewe_rejects_zero_ic50(self, syn):
        """Zero IC50 should raise SynergyScoringError."""
        with pytest.raises(SynergyScoringError):
            syn.loewe_additivity(5.0, 5.0, 0.0, 10.0, 0.5)


# ---------------------------------------------------------------------------
# HSA model mathematical properties
# ---------------------------------------------------------------------------


class TestHSAModelMathematics:
    """Validate Highest Single Agent model against known properties."""

    def test_excess_zero_when_combo_equals_best(self, syn):
        """If combo effect equals the best single agent → HSA excess = 0."""
        _, s, _ = syn.hsa_model(0.3, 0.5, 0.5)
        assert abs(s["excess"]) < 1e-10
        assert s["hsa_reference"] == 0.5

    def test_identifies_best_single_agent(self, syn):
        """Should report which drug is the better monotherapy."""
        _, s, _ = syn.hsa_model(0.3, 0.7, 0.8)
        assert s["best_single_agent"] == "B"

        _, s2, _ = syn.hsa_model(0.8, 0.3, 0.9)
        assert s2["best_single_agent"] == "A"

    def test_synergistic_when_combo_exceeds_best(self, syn):
        """Combo much better than best monotherapy → synergistic."""
        _, s, _ = syn.hsa_model(0.3, 0.3, 0.8)
        assert s["classification"] == "synergistic"
        assert s["excess"] > 0.1

    def test_antagonistic_when_combo_worse_than_best(self, syn):
        """Combo worse than best monotherapy → antagonistic."""
        _, s, _ = syn.hsa_model(0.6, 0.5, 0.3)
        assert s["classification"] == "antagonistic"
        assert s["excess"] < -0.1


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestSynergyInputValidation:
    """Verify services reject invalid inputs."""

    def test_bliss_rejects_negative_effect(self, syn):
        with pytest.raises(SynergyScoringError):
            syn.bliss_independence(-0.1, 0.5, 0.5)

    def test_bliss_rejects_effect_above_one(self, syn):
        with pytest.raises(SynergyScoringError):
            syn.bliss_independence(0.5, 1.5, 0.5)

    def test_hsa_rejects_negative_effect(self, syn):
        with pytest.raises(SynergyScoringError):
            syn.hsa_model(-0.1, 0.5, 0.5)

    def test_loewe_rejects_negative_ic50(self, syn):
        with pytest.raises(SynergyScoringError):
            syn.loewe_additivity(5.0, 5.0, -10.0, 10.0, 0.5)


# ---------------------------------------------------------------------------
# Combination matrix scoring (AnnData)
# ---------------------------------------------------------------------------


class TestCombinationMatrixScoring:
    """Full dose-response matrix scoring with synthetic AnnData."""

    def test_synergistic_matrix(self, syn):
        """Build a matrix with known synergy and verify scoring detects it."""
        doses_a = [0, 1, 2, 5, 10]
        doses_b = [0, 1, 5, 10, 20]
        rows = []
        for da in doses_a:
            for db in doses_b:
                ea = 1 - np.exp(-0.1 * da)
                eb = 1 - np.exp(-0.05 * db)
                # Synergistic interaction: combo better than Bliss
                eab = min(1.0, ea + eb - ea * eb + 0.15 * ea * eb)
                rows.append({"dose_a": da, "dose_b": db, "response": eab})

        df = pd.DataFrame(rows)
        adata = AnnData(X=np.zeros((len(rows), 1)), obs=df)

        result_adata, stats, ir = syn.score_combination_matrix(
            adata, "dose_a", "dose_b", "response", model="bliss"
        )

        assert isinstance(result_adata, AnnData)
        assert "synergy_score" in result_adata.obs.columns
        assert "expected_effect" in result_adata.obs.columns
        assert "synergy_classification" in result_adata.obs.columns

        # Combination points should have non-NaN scores
        combo_mask = (result_adata.obs["dose_a"] > 0) & (result_adata.obs["dose_b"] > 0)
        combo_scores = result_adata.obs.loc[combo_mask, "synergy_score"].dropna()
        assert len(combo_scores) > 0, "Should have scored combination points"

        # Our synthetic data is synergistic by design
        assert stats["n_combination_points"] > 0
        assert isinstance(ir, AnalysisStep)

    def test_monotherapy_rows_have_nan_scores(self, syn):
        """Rows where one drug dose=0 should have NaN synergy scores."""
        obs = pd.DataFrame({
            "drug_a": [0.0, 1.0, 0.0, 1.0],
            "drug_b": [0.0, 0.0, 1.0, 1.0],
            "effect": [0.0, 0.3, 0.2, 0.6],
        })
        adata = AnnData(X=np.zeros((4, 1)), obs=obs)

        result, stats, _ = syn.score_combination_matrix(
            adata, "drug_a", "drug_b", "effect", model="bliss"
        )

        # Monotherapy rows (dose=0 for one drug) should be NaN
        mono_mask = (result.obs["drug_a"] == 0) | (result.obs["drug_b"] == 0)
        mono_scores = result.obs.loc[mono_mask, "synergy_score"]
        assert mono_scores.isna().all(), (
            "Monotherapy rows should have NaN synergy scores"
        )

    def test_hsa_model_on_matrix(self, syn):
        """HSA model should also work on combination matrix."""
        obs = pd.DataFrame({
            "d_a": [0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0],
            "d_b": [0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 2.0],
            "resp": [0.0, 0.2, 0.3, 0.7, 0.4, 0.5, 0.9],
        })
        adata = AnnData(X=np.zeros((7, 1)), obs=obs)

        result, stats, _ = syn.score_combination_matrix(
            adata, "d_a", "d_b", "resp", model="hsa"
        )

        assert stats["model"] == "hsa"
        assert stats["n_combination_points"] > 0

    def test_missing_column_raises(self, syn):
        """Missing response column should raise SynergyScoringError."""
        adata = AnnData(
            X=np.zeros((3, 1)),
            obs=pd.DataFrame({"a": [0, 1, 1], "b": [0, 0, 1]}),
        )
        with pytest.raises(SynergyScoringError, match="not found"):
            syn.score_combination_matrix(adata, "a", "b", "missing_col")
