"""
Unit tests for SynergyScoringService.

Tests the drug combination synergy scoring system with known mathematical results:
- Bliss Independence: E_expected = E_a + E_b - E_a * E_b
- Loewe Additivity: CI = d_a / IC50_a + d_b / IC50_b
- HSA: excess = E_ab - max(E_a, E_b)
- Full dose-response combination matrix scoring with AnnData

All models are tested with deterministic inputs and expected outputs.
Classification thresholds are imported from config.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from lobster.agents.drug_discovery.config import SYNERGY_THRESHOLDS
from lobster.core.analysis_ir import AnalysisStep
from lobster.services.drug_discovery.synergy_scoring_service import (
    SynergyScoringError,
    SynergyScoringService,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def service():
    """Create a fresh SynergyScoringService instance."""
    return SynergyScoringService()


# =============================================================================
# BLISS INDEPENDENCE TESTS
# =============================================================================


class TestBlissIndependence:
    """Tests for SynergyScoringService.bliss_independence()."""

    def test_synergistic_combination(self, service):
        """Bliss with observed > expected is synergistic.

        E_a=0.3, E_b=0.5
        E_expected = 0.3 + 0.5 - 0.3*0.5 = 0.65
        E_ab=0.8
        excess = 0.8 - 0.65 = 0.15 (> 0.1 threshold -> synergistic)
        """
        adata, result, ir = service.bliss_independence(
            effect_a=0.3, effect_b=0.5, effect_ab=0.8
        )

        assert adata is None
        assert abs(result["effect_ab_expected"] - 0.65) < 1e-6
        assert abs(result["excess"] - 0.15) < 1e-6
        assert result["classification"] == "synergistic"
        assert result["model"] == "bliss_independence"

    def test_antagonistic_combination(self, service):
        """Bliss with observed << expected is antagonistic.

        E_a=0.3, E_b=0.5
        E_expected = 0.65
        E_ab=0.4
        excess = 0.4 - 0.65 = -0.25 (< -0.1 threshold -> antagonistic)
        """
        _, result, _ = service.bliss_independence(
            effect_a=0.3, effect_b=0.5, effect_ab=0.4
        )

        assert abs(result["effect_ab_expected"] - 0.65) < 1e-6
        assert abs(result["excess"] - (-0.25)) < 1e-6
        assert result["classification"] == "antagonistic"

    def test_additive_combination(self, service):
        """Bliss with observed ~ expected is additive.

        E_a=0.3, E_b=0.5
        E_expected = 0.65
        E_ab=0.68 (excess = 0.03, within +/- 0.1 -> additive)
        """
        _, result, _ = service.bliss_independence(
            effect_a=0.3, effect_b=0.5, effect_ab=0.68
        )

        assert abs(result["excess"] - 0.03) < 1e-6
        assert result["classification"] == "additive"

    def test_exact_expected_is_additive(self, service):
        """When E_ab equals E_expected exactly, excess=0 -> additive."""
        _, result, _ = service.bliss_independence(
            effect_a=0.3, effect_b=0.5, effect_ab=0.65
        )

        assert abs(result["excess"]) < 1e-6
        assert result["classification"] == "additive"

    def test_no_effect_drugs(self, service):
        """Both drugs with zero effect: expected=0, combo=0 -> additive."""
        _, result, _ = service.bliss_independence(
            effect_a=0.0, effect_b=0.0, effect_ab=0.0
        )

        assert abs(result["effect_ab_expected"]) < 1e-6
        assert abs(result["excess"]) < 1e-6
        assert result["classification"] == "additive"

    def test_full_effect_drugs(self, service):
        """Both drugs with full effect: expected=1, combo=1 -> additive."""
        _, result, _ = service.bliss_independence(
            effect_a=1.0, effect_b=1.0, effect_ab=1.0
        )

        # E_expected = 1 + 1 - 1*1 = 1.0
        assert abs(result["effect_ab_expected"] - 1.0) < 1e-6
        assert abs(result["excess"]) < 1e-6
        assert result["classification"] == "additive"

    def test_single_drug_zero_other(self, service):
        """One drug effective, other zero.

        E_a=0.5, E_b=0.0
        E_expected = 0.5 + 0 - 0 = 0.5
        """
        _, result, _ = service.bliss_independence(
            effect_a=0.5, effect_b=0.0, effect_ab=0.5
        )

        assert abs(result["effect_ab_expected"] - 0.5) < 1e-6
        assert abs(result["excess"]) < 1e-6

    def test_returns_none_dict_analysistep(self, service):
        """bliss_independence returns (None, Dict, AnalysisStep) tuple."""
        adata, result, ir = service.bliss_independence(0.3, 0.5, 0.65)

        assert adata is None
        assert isinstance(result, dict)
        assert isinstance(ir, AnalysisStep)
        assert ir.tool_name == "bliss_independence"

    def test_result_contains_thresholds(self, service):
        """Result dict should contain the thresholds used for classification."""
        _, result, _ = service.bliss_independence(0.3, 0.5, 0.65)

        assert "thresholds" in result
        assert result["thresholds"] == dict(SYNERGY_THRESHOLDS)

    def test_result_records_inputs(self, service):
        """Result dict should record the input effect values."""
        _, result, _ = service.bliss_independence(0.3, 0.5, 0.65)

        assert result["effect_a"] == 0.3
        assert result["effect_b"] == 0.5
        assert result["effect_ab_observed"] == 0.65

    def test_validates_effect_a_above_one(self, service):
        """Bliss raises SynergyScoringError if effect_a > 1.0."""
        with pytest.raises(SynergyScoringError, match="effect_a"):
            service.bliss_independence(1.1, 0.5, 0.8)

    def test_validates_effect_b_below_zero(self, service):
        """Bliss raises SynergyScoringError if effect_b < 0.0."""
        with pytest.raises(SynergyScoringError, match="effect_b"):
            service.bliss_independence(0.3, -0.1, 0.3)

    def test_validates_effect_ab_non_numeric(self, service):
        """Bliss raises SynergyScoringError for non-numeric effect_ab."""
        with pytest.raises(SynergyScoringError, match="effect_ab"):
            service.bliss_independence(0.3, 0.5, "high")

    def test_threshold_boundary_synergistic(self, service):
        """Bliss excess just below 0.1 should be additive (threshold is >0.1 for synergistic).

        E_a=0.2, E_b=0.3
        E_expected = 0.2 + 0.3 - 0.06 = 0.44
        E_ab = 0.539 -> excess = 0.099 (< 0.1 threshold -> additive)

        Note: floating-point arithmetic means exact 0.10 (e.g., 0.54 - 0.44)
        can round to 0.10000000000000003 which exceeds the threshold.
        """
        _, result, _ = service.bliss_independence(0.2, 0.3, 0.539)

        assert result["excess"] < SYNERGY_THRESHOLDS["synergistic"]
        assert result["classification"] == "additive"

    def test_threshold_boundary_antagonistic(self, service):
        """Bliss excess of exactly -0.1 should be additive (threshold is <-0.1 for antagonistic).

        E_a=0.2, E_b=0.3
        E_expected = 0.44
        E_ab = 0.34 -> excess = -0.10
        """
        _, result, _ = service.bliss_independence(0.2, 0.3, 0.34)

        assert abs(result["excess"] - (-0.10)) < 1e-6
        # Threshold: excess < -0.1 for antagonistic, so exactly -0.1 is additive
        assert result["classification"] == "additive"


# =============================================================================
# LOEWE ADDITIVITY TESTS
# =============================================================================


class TestLoeweAdditivity:
    """Tests for SynergyScoringService.loewe_additivity()."""

    def test_additive_combination(self, service):
        """Loewe CI = 1.0 when dose equals IC50.

        dose_a=10, dose_b=20, ic50_a=10, ic50_b=20
        CI = 10/10 + 20/20 = 1.0 + 1.0 = 2.0 (antagonistic: CI > 1.1)

        For CI=1.0 exactly:
        dose_a=5, dose_b=10, ic50_a=10, ic50_b=20
        CI = 5/10 + 10/20 = 0.5 + 0.5 = 1.0 (additive: 0.9 <= CI <= 1.1)
        """
        _, result, _ = service.loewe_additivity(
            dose_a=5.0, dose_b=10.0, ic50_a=10.0, ic50_b=20.0, effect_ab=0.5
        )

        assert abs(result["combination_index"] - 1.0) < 1e-6
        assert result["classification"] == "additive"
        assert result["model"] == "loewe_additivity"

    def test_synergistic_combination(self, service):
        """Loewe CI < 0.9 is synergistic.

        dose_a=2, dose_b=5, ic50_a=10, ic50_b=20
        CI = 2/10 + 5/20 = 0.2 + 0.25 = 0.45 (synergistic)
        """
        _, result, _ = service.loewe_additivity(
            dose_a=2.0, dose_b=5.0, ic50_a=10.0, ic50_b=20.0, effect_ab=0.5
        )

        assert abs(result["combination_index"] - 0.45) < 1e-6
        assert result["classification"] == "synergistic"

    def test_antagonistic_combination(self, service):
        """Loewe CI > 1.1 is antagonistic.

        dose_a=10, dose_b=20, ic50_a=10, ic50_b=20
        CI = 10/10 + 20/20 = 1.0 + 1.0 = 2.0 (antagonistic)
        """
        _, result, _ = service.loewe_additivity(
            dose_a=10.0, dose_b=20.0, ic50_a=10.0, ic50_b=20.0, effect_ab=0.8
        )

        assert abs(result["combination_index"] - 2.0) < 1e-6
        assert result["classification"] == "antagonistic"

    def test_dose_reduction_index(self, service):
        """Loewe should compute DRI (dose reduction index) = IC50/dose.

        dose_a=5, ic50_a=10 -> DRI_a = 10/5 = 2.0
        dose_b=10, ic50_b=20 -> DRI_b = 20/10 = 2.0
        """
        _, result, _ = service.loewe_additivity(
            dose_a=5.0, dose_b=10.0, ic50_a=10.0, ic50_b=20.0, effect_ab=0.5
        )

        assert abs(result["dose_reduction_index_a"] - 2.0) < 1e-4
        assert abs(result["dose_reduction_index_b"] - 2.0) < 1e-4

    def test_returns_none_dict_analysistep(self, service):
        """loewe_additivity returns (None, Dict, AnalysisStep) tuple."""
        adata, result, ir = service.loewe_additivity(5, 10, 10, 20, 0.5)

        assert adata is None
        assert isinstance(result, dict)
        assert isinstance(ir, AnalysisStep)
        assert ir.tool_name == "loewe_additivity"

    def test_records_input_values(self, service):
        """Result dict records all input values."""
        _, result, _ = service.loewe_additivity(5, 10, 10, 20, 0.5)

        assert result["dose_a"] == 5
        assert result["dose_b"] == 10
        assert result["ic50_a"] == 10
        assert result["ic50_b"] == 20
        assert result["effect_ab_observed"] == 0.5

    def test_validates_dose_positive(self, service):
        """Loewe raises SynergyScoringError for non-positive dose."""
        with pytest.raises(SynergyScoringError, match="dose_a.*positive"):
            service.loewe_additivity(0, 10, 10, 20, 0.5)

    def test_validates_ic50_positive(self, service):
        """Loewe raises SynergyScoringError for non-positive IC50."""
        with pytest.raises(SynergyScoringError, match="ic50_a.*positive"):
            service.loewe_additivity(5, 10, 0, 20, 0.5)

    def test_validates_negative_dose(self, service):
        """Loewe raises SynergyScoringError for negative dose."""
        with pytest.raises(SynergyScoringError, match="dose_b.*positive"):
            service.loewe_additivity(5, -1, 10, 20, 0.5)

    def test_validates_effect_range(self, service):
        """Loewe validates effect_ab is in [0, 1]."""
        with pytest.raises(SynergyScoringError, match="effect_ab"):
            service.loewe_additivity(5, 10, 10, 20, 1.5)

    def test_ci_boundary_synergistic(self, service):
        """CI of exactly 0.9 is additive (< 0.9 for synergistic).

        dose_a=4.5, dose_b=9, ic50_a=10, ic50_b=20
        CI = 4.5/10 + 9/20 = 0.45 + 0.45 = 0.90 (additive)
        """
        _, result, _ = service.loewe_additivity(
            dose_a=4.5, dose_b=9.0, ic50_a=10.0, ic50_b=20.0, effect_ab=0.5
        )

        assert abs(result["combination_index"] - 0.9) < 1e-6
        assert result["classification"] == "additive"

    def test_ci_boundary_antagonistic(self, service):
        """CI of exactly 1.1 is additive (> 1.1 for antagonistic).

        dose_a=5.5, dose_b=11, ic50_a=10, ic50_b=20
        CI = 5.5/10 + 11/20 = 0.55 + 0.55 = 1.10 (additive)
        """
        _, result, _ = service.loewe_additivity(
            dose_a=5.5, dose_b=11.0, ic50_a=10.0, ic50_b=20.0, effect_ab=0.5
        )

        assert abs(result["combination_index"] - 1.1) < 1e-6
        assert result["classification"] == "additive"

    def test_has_interpretation_string(self, service):
        """Result should contain a human-readable interpretation."""
        _, result, _ = service.loewe_additivity(5, 10, 10, 20, 0.5)
        assert "interpretation" in result
        assert isinstance(result["interpretation"], str)
        assert "CI=" in result["interpretation"]


# =============================================================================
# HSA MODEL TESTS
# =============================================================================


class TestHSAModel:
    """Tests for SynergyScoringService.hsa_model()."""

    def test_synergistic_above_best_monotherapy(self, service):
        """HSA with combination > max(single agents) is synergistic.

        E_a=0.3, E_b=0.5
        hsa_ref = max(0.3, 0.5) = 0.5
        E_ab=0.7
        excess = 0.7 - 0.5 = 0.2 (> 0.1 -> synergistic)
        """
        adata, result, ir = service.hsa_model(
            effect_a=0.3, effect_b=0.5, effect_ab=0.7
        )

        assert adata is None
        assert abs(result["hsa_reference"] - 0.5) < 1e-6
        assert abs(result["excess"] - 0.2) < 1e-6
        assert result["classification"] == "synergistic"
        assert result["model"] == "hsa"
        assert result["best_single_agent"] == "B"

    def test_antagonistic_below_best_monotherapy(self, service):
        """HSA with combination < max(single agents) is antagonistic.

        E_a=0.5, E_b=0.3
        hsa_ref = 0.5
        E_ab=0.3
        excess = 0.3 - 0.5 = -0.2 (< -0.1 -> antagonistic)
        """
        _, result, _ = service.hsa_model(
            effect_a=0.5, effect_b=0.3, effect_ab=0.3
        )

        assert abs(result["hsa_reference"] - 0.5) < 1e-6
        assert abs(result["excess"] - (-0.2)) < 1e-6
        assert result["classification"] == "antagonistic"
        assert result["best_single_agent"] == "A"

    def test_additive_near_best_monotherapy(self, service):
        """HSA with combination ~ max(single agents) is additive.

        E_a=0.5, E_b=0.3
        hsa_ref = 0.5
        E_ab=0.55
        excess = 0.55 - 0.5 = 0.05 (within +/- 0.1 -> additive)
        """
        _, result, _ = service.hsa_model(
            effect_a=0.5, effect_b=0.3, effect_ab=0.55
        )

        assert abs(result["excess"] - 0.05) < 1e-6
        assert result["classification"] == "additive"

    def test_hsa_formula_correctness(self, service):
        """Verify HSA formula: excess = E_ab - max(E_a, E_b).

        Various input combinations to verify the formula.
        """
        test_cases = [
            (0.2, 0.8, 0.9, 0.8, 0.1),   # excess = 0.9 - 0.8 = 0.1
            (0.6, 0.4, 0.6, 0.6, 0.0),   # excess = 0.6 - 0.6 = 0.0
            (0.0, 0.0, 0.0, 0.0, 0.0),   # excess = 0.0 - 0.0 = 0.0
            (1.0, 1.0, 1.0, 1.0, 0.0),   # excess = 1.0 - 1.0 = 0.0
        ]

        for e_a, e_b, e_ab, expected_ref, expected_excess in test_cases:
            _, result, _ = service.hsa_model(e_a, e_b, e_ab)
            assert abs(result["hsa_reference"] - expected_ref) < 1e-6, (
                f"HSA ref failed for ({e_a}, {e_b}, {e_ab})"
            )
            assert abs(result["excess"] - expected_excess) < 1e-6, (
                f"HSA excess failed for ({e_a}, {e_b}, {e_ab})"
            )

    def test_best_single_agent_a(self, service):
        """HSA correctly identifies drug A when E_a >= E_b."""
        _, result, _ = service.hsa_model(0.6, 0.4, 0.7)
        assert result["best_single_agent"] == "A"

    def test_best_single_agent_b(self, service):
        """HSA correctly identifies drug B when E_b > E_a."""
        _, result, _ = service.hsa_model(0.3, 0.7, 0.8)
        assert result["best_single_agent"] == "B"

    def test_equal_effects_best_is_a(self, service):
        """When E_a == E_b, best_single_agent should be A (>= comparison)."""
        _, result, _ = service.hsa_model(0.5, 0.5, 0.6)
        assert result["best_single_agent"] == "A"

    def test_returns_none_dict_analysistep(self, service):
        """hsa_model returns (None, Dict, AnalysisStep) tuple."""
        adata, result, ir = service.hsa_model(0.3, 0.5, 0.6)

        assert adata is None
        assert isinstance(result, dict)
        assert isinstance(ir, AnalysisStep)
        assert ir.tool_name == "hsa_model"

    def test_result_contains_thresholds(self, service):
        """Result dict should contain the synergy thresholds."""
        _, result, _ = service.hsa_model(0.3, 0.5, 0.6)
        assert "thresholds" in result

    def test_validates_effects(self, service):
        """HSA validates all three effect values."""
        with pytest.raises(SynergyScoringError, match="effect_a"):
            service.hsa_model(1.5, 0.5, 0.8)

        with pytest.raises(SynergyScoringError, match="effect_b"):
            service.hsa_model(0.3, -0.1, 0.3)

        with pytest.raises(SynergyScoringError, match="effect_ab"):
            service.hsa_model(0.3, 0.5, 2.0)


# =============================================================================
# SCORE_COMBINATION_MATRIX TESTS
# =============================================================================


class TestScoreCombinationMatrix:
    """Tests for SynergyScoringService.score_combination_matrix()."""

    def _make_combination_adata(self):
        """Create a minimal dose-response combination matrix AnnData.

        Layout:
        - 3 monotherapy A points (drug_b=0): dose_a=[0.5, 1.0, 2.0], response=[0.1, 0.2, 0.4]
        - 3 monotherapy B points (drug_a=0): dose_b=[0.5, 1.0, 2.0], response=[0.15, 0.3, 0.5]
        - 4 combination points: various doses, various responses
        """
        n_obs = 10
        obs = pd.DataFrame({
            "drug_a": [0.5, 1.0, 2.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 2.0],
            "drug_b": [0.0, 0.0, 0.0, 0.5, 1.0, 2.0, 0.5, 1.0, 2.0, 2.0],
            "response": [0.1, 0.2, 0.4, 0.15, 0.3, 0.5, 0.35, 0.6, 0.85, 0.95],
        }, index=[f"well_{i}" for i in range(n_obs)])

        X = np.zeros((n_obs, 2))
        adata = AnnData(X=X, obs=obs)
        return adata

    def test_bliss_matrix_scoring(self, service):
        """Score combination matrix with Bliss model."""
        adata = self._make_combination_adata()
        result_adata, stats, ir = service.score_combination_matrix(
            adata, drug_a_col="drug_a", drug_b_col="drug_b",
            response_col="response", model="bliss"
        )

        # Verify result is AnnData (not None)
        assert isinstance(result_adata, AnnData)

        # Verify new columns were added
        assert "synergy_score" in result_adata.obs.columns
        assert "expected_effect" in result_adata.obs.columns
        assert "synergy_classification" in result_adata.obs.columns
        assert "synergy_model" in result_adata.obs.columns

        # All model entries should be "bliss"
        model_values = result_adata.obs["synergy_model"].unique()
        assert "bliss" in model_values

        # Verify stats
        assert stats["model"] == "bliss"
        assert stats["n_observations"] == 10
        assert stats["n_combination_points"] == 4
        assert stats["n_monotherapy_a"] == 3
        assert stats["n_monotherapy_b"] == 3

        # Classification counts should sum to n_combination_points
        n_total = stats["n_synergistic"] + stats["n_additive"] + stats["n_antagonistic"]
        assert n_total == stats["n_combination_points"]

        # Verify columns_added list
        assert "synergy_score" in stats["columns_added"]
        assert "expected_effect" in stats["columns_added"]

    def test_hsa_matrix_scoring(self, service):
        """Score combination matrix with HSA model."""
        adata = self._make_combination_adata()
        result_adata, stats, ir = service.score_combination_matrix(
            adata, drug_a_col="drug_a", drug_b_col="drug_b",
            response_col="response", model="hsa"
        )

        assert isinstance(result_adata, AnnData)
        assert stats["model"] == "hsa"
        assert stats["n_combination_points"] == 4

    def test_monotherapy_rows_have_nan_scores(self, service):
        """Monotherapy rows should have NaN synergy scores."""
        adata = self._make_combination_adata()
        result_adata, _, _ = service.score_combination_matrix(
            adata, drug_a_col="drug_a", drug_b_col="drug_b",
            response_col="response", model="bliss"
        )

        # Monotherapy A: rows 0, 1, 2 (drug_b=0)
        for idx in [0, 1, 2]:
            assert np.isnan(result_adata.obs["synergy_score"].iloc[idx]), (
                f"Monotherapy A row {idx} should have NaN synergy score"
            )

        # Monotherapy B: rows 3, 4, 5 (drug_a=0)
        for idx in [3, 4, 5]:
            assert np.isnan(result_adata.obs["synergy_score"].iloc[idx]), (
                f"Monotherapy B row {idx} should have NaN synergy score"
            )

    def test_combination_rows_have_numeric_scores(self, service):
        """Combination rows should have non-NaN synergy scores."""
        adata = self._make_combination_adata()
        result_adata, _, _ = service.score_combination_matrix(
            adata, drug_a_col="drug_a", drug_b_col="drug_b",
            response_col="response", model="bliss"
        )

        # Combination rows: 6, 7, 8, 9 (both drug_a > 0 and drug_b > 0)
        for idx in [6, 7, 8, 9]:
            assert not np.isnan(result_adata.obs["synergy_score"].iloc[idx]), (
                f"Combination row {idx} should have a numeric synergy score"
            )

    def test_does_not_modify_original(self, service):
        """score_combination_matrix should not modify the original AnnData."""
        adata = self._make_combination_adata()
        original_cols = set(adata.obs.columns)

        _ = service.score_combination_matrix(
            adata, drug_a_col="drug_a", drug_b_col="drug_b",
            response_col="response"
        )

        # Original should be unchanged
        assert set(adata.obs.columns) == original_cols

    def test_missing_column_raises_error(self, service):
        """score_combination_matrix raises SynergyScoringError for missing column."""
        adata = self._make_combination_adata()
        with pytest.raises(SynergyScoringError, match="not found in adata.obs"):
            service.score_combination_matrix(
                adata, drug_a_col="nonexistent",
                drug_b_col="drug_b", response_col="response"
            )

    def test_unsupported_model_raises_error(self, service):
        """score_combination_matrix raises SynergyScoringError for unsupported model."""
        adata = self._make_combination_adata()
        with pytest.raises(SynergyScoringError, match="Unsupported synergy model"):
            service.score_combination_matrix(
                adata, drug_a_col="drug_a", drug_b_col="drug_b",
                response_col="response", model="invalid_model"
            )

    def test_returns_adata_dict_analysistep(self, service):
        """score_combination_matrix returns (AnnData, Dict, AnalysisStep) tuple."""
        adata = self._make_combination_adata()
        result_adata, result_dict, ir = service.score_combination_matrix(
            adata, drug_a_col="drug_a", drug_b_col="drug_b",
            response_col="response"
        )

        assert isinstance(result_adata, AnnData)
        assert isinstance(result_dict, dict)
        assert isinstance(ir, AnalysisStep)
        assert ir.tool_name == "score_combination_matrix"

    def test_mean_synergy_score_calculated(self, service):
        """Stats should include mean_synergy_score when combinations exist."""
        adata = self._make_combination_adata()
        _, stats, _ = service.score_combination_matrix(
            adata, drug_a_col="drug_a", drug_b_col="drug_b",
            response_col="response"
        )

        assert stats["mean_synergy_score"] is not None
        assert isinstance(stats["mean_synergy_score"], float)

    def test_max_min_synergy_scores(self, service):
        """Stats should include max and min synergy scores."""
        adata = self._make_combination_adata()
        _, stats, _ = service.score_combination_matrix(
            adata, drug_a_col="drug_a", drug_b_col="drug_b",
            response_col="response"
        )

        assert stats["max_synergy_score"] is not None
        assert stats["min_synergy_score"] is not None
        assert stats["max_synergy_score"] >= stats["min_synergy_score"]
