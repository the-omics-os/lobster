"""
Synergy scoring service for combination therapy assessment.

Implements three classical drug combination models -- Bliss Independence,
Loewe Additivity, and Highest Single Agent (HSA) -- plus a full dose-response
matrix scorer that integrates with AnnData.

Non-AnnData methods return (None, Dict, AnalysisStep).
The matrix scorer returns (AnnData, Dict, AnalysisStep) since it annotates data.

All methods use SYNERGY_THRESHOLDS from config for classification cutoffs.
"""

import math
from typing import Any, Dict, Optional, Tuple

import anndata
import numpy as np

from lobster.agents.drug_discovery.config import SYNERGY_THRESHOLDS
from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class SynergyScoringError(Exception):
    """Base exception for synergy scoring operations."""

    pass


class SynergyScoringService:
    """
    Stateless synergy scoring service for combination therapy evaluation.

    Provides three classical interaction models:
    - **Bliss Independence**: E_expected = E_a + E_b - E_a * E_b
    - **Loewe Additivity**: CI = d_a/D_a + d_b/D_b (combination index)
    - **HSA (Highest Single Agent)**: excess = E_ab - max(E_a, E_b)

    Synergy classification thresholds are imported from config:
    - synergistic: excess > 0.1
    - additive: -0.1 <= excess <= 0.1
    - antagonistic: excess < -0.1
    """

    def __init__(self):
        """Initialize the synergy scoring service (stateless)."""
        logger.debug("Initializing SynergyScoringService")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _classify_synergy(self, excess: float) -> str:
        """Classify interaction based on excess effect and thresholds.

        Args:
            excess: Difference between observed and expected effect.

        Returns:
            Classification string: 'synergistic', 'additive', or 'antagonistic'.
        """
        if excess > SYNERGY_THRESHOLDS["synergistic"]:
            return "synergistic"
        elif excess < SYNERGY_THRESHOLDS["antagonistic"]:
            return "antagonistic"
        else:
            return "additive"

    def _validate_effect(self, value: float, name: str) -> None:
        """Validate that an effect value is in [0.0, 1.0].

        Args:
            value: Effect value to validate.
            name: Parameter name for error messages.

        Raises:
            SynergyScoringError: If value is out of range.
        """
        if not isinstance(value, (int, float)):
            raise SynergyScoringError(
                f"{name} must be numeric, got {type(value).__name__}"
            )
        if not (0.0 <= value <= 1.0):
            raise SynergyScoringError(
                f"{name} must be in [0.0, 1.0], got {value}. "
                "Effect values represent fractional inhibition (0 = no effect, 1 = full effect)."
            )

    def _validate_dose(self, value: float, name: str) -> None:
        """Validate that a dose value is positive.

        Args:
            value: Dose value to validate.
            name: Parameter name for error messages.

        Raises:
            SynergyScoringError: If value is not positive.
        """
        if not isinstance(value, (int, float)):
            raise SynergyScoringError(
                f"{name} must be numeric, got {type(value).__name__}"
            )
        if value <= 0:
            raise SynergyScoringError(
                f"{name} must be positive, got {value}"
            )

    # -------------------------------------------------------------------------
    # IR builders
    # -------------------------------------------------------------------------

    def _create_ir_bliss(
        self, effect_a: float, effect_b: float, effect_ab: float
    ) -> AnalysisStep:
        return AnalysisStep(
            operation="lobster.services.drug_discovery.synergy.bliss",
            tool_name="bliss_independence",
            description=(
                "Bliss Independence model: E_expected = E_a + E_b - E_a*E_b. "
                "Excess > 0 indicates synergy."
            ),
            library="lobster",
            code_template="""# Bliss Independence Model
E_a = {{ effect_a }}
E_b = {{ effect_b }}
E_ab = {{ effect_ab }}
E_expected = E_a + E_b - E_a * E_b
excess = E_ab - E_expected
print(f"Expected: {E_expected:.4f}, Observed: {E_ab:.4f}, Excess: {excess:.4f}")""",
            imports=[],
            parameters={
                "effect_a": effect_a,
                "effect_b": effect_b,
                "effect_ab": effect_ab,
            },
            parameter_schema={
                "effect_a": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.0,
                    required=True,
                    validation_rule="0 <= effect_a <= 1",
                    description="Fractional effect of drug A alone",
                ),
                "effect_b": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.0,
                    required=True,
                    validation_rule="0 <= effect_b <= 1",
                    description="Fractional effect of drug B alone",
                ),
                "effect_ab": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.0,
                    required=True,
                    validation_rule="0 <= effect_ab <= 1",
                    description="Fractional effect of combination A+B",
                ),
            },
            input_entities=["effect_a", "effect_b", "effect_ab"],
            output_entities=["bliss_result"],
        )

    def _create_ir_loewe(
        self,
        dose_a: float,
        dose_b: float,
        ic50_a: float,
        ic50_b: float,
        effect_ab: float,
    ) -> AnalysisStep:
        return AnalysisStep(
            operation="lobster.services.drug_discovery.synergy.loewe",
            tool_name="loewe_additivity",
            description=(
                "Loewe Additivity model: CI = d_a/D_a + d_b/D_b. "
                "CI < 1 = synergy, CI = 1 = additive, CI > 1 = antagonism."
            ),
            library="lobster",
            code_template="""# Loewe Additivity Model
d_a, d_b = {{ dose_a }}, {{ dose_b }}
IC50_a, IC50_b = {{ ic50_a }}, {{ ic50_b }}
# CI = d_a/D_a + d_b/D_b where D_x is dose of x alone for same effect
CI = d_a / IC50_a + d_b / IC50_b
print(f"Combination Index: {CI:.4f}")""",
            imports=[],
            parameters={
                "dose_a": dose_a,
                "dose_b": dose_b,
                "ic50_a": ic50_a,
                "ic50_b": ic50_b,
                "effect_ab": effect_ab,
            },
            parameter_schema={
                "dose_a": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.0,
                    required=True,
                    validation_rule="dose_a > 0",
                    description="Dose of drug A in combination",
                ),
                "dose_b": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.0,
                    required=True,
                    validation_rule="dose_b > 0",
                    description="Dose of drug B in combination",
                ),
                "ic50_a": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.0,
                    required=True,
                    validation_rule="ic50_a > 0",
                    description="IC50 of drug A alone",
                ),
                "ic50_b": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.0,
                    required=True,
                    validation_rule="ic50_b > 0",
                    description="IC50 of drug B alone",
                ),
                "effect_ab": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.0,
                    required=True,
                    validation_rule="0 <= effect_ab <= 1",
                    description="Observed fractional effect of the combination",
                ),
            },
            input_entities=["dose_a", "dose_b", "ic50_a", "ic50_b", "effect_ab"],
            output_entities=["loewe_result"],
        )

    def _create_ir_hsa(
        self, effect_a: float, effect_b: float, effect_ab: float
    ) -> AnalysisStep:
        return AnalysisStep(
            operation="lobster.services.drug_discovery.synergy.hsa",
            tool_name="hsa_model",
            description=(
                "Highest Single Agent model: excess = E_ab - max(E_a, E_b). "
                "Positive excess indicates the combination exceeds the best monotherapy."
            ),
            library="lobster",
            code_template="""# Highest Single Agent (HSA) Model
E_a = {{ effect_a }}
E_b = {{ effect_b }}
E_ab = {{ effect_ab }}
hsa = max(E_a, E_b)
excess = E_ab - hsa
print(f"HSA: {hsa:.4f}, Observed: {E_ab:.4f}, Excess: {excess:.4f}")""",
            imports=[],
            parameters={
                "effect_a": effect_a,
                "effect_b": effect_b,
                "effect_ab": effect_ab,
            },
            parameter_schema={
                "effect_a": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.0,
                    required=True,
                    validation_rule="0 <= effect_a <= 1",
                    description="Fractional effect of drug A alone",
                ),
                "effect_b": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.0,
                    required=True,
                    validation_rule="0 <= effect_b <= 1",
                    description="Fractional effect of drug B alone",
                ),
                "effect_ab": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.0,
                    required=True,
                    validation_rule="0 <= effect_ab <= 1",
                    description="Fractional effect of combination A+B",
                ),
            },
            input_entities=["effect_a", "effect_b", "effect_ab"],
            output_entities=["hsa_result"],
        )

    def _create_ir_score_matrix(
        self,
        drug_a_col: str,
        drug_b_col: str,
        response_col: str,
        model: str,
    ) -> AnalysisStep:
        return AnalysisStep(
            operation="lobster.services.drug_discovery.synergy.score_matrix",
            tool_name="score_combination_matrix",
            description=(
                f"Score full dose-response combination matrix using the {model} model. "
                "Adds synergy scores to adata.obs."
            ),
            library="lobster",
            code_template="""from lobster.services.drug_discovery.synergy_scoring_service import SynergyScoringService

service = SynergyScoringService()
adata_scored, stats, _ = service.score_combination_matrix(
    adata,
    drug_a_col={{ drug_a_col | tojson }},
    drug_b_col={{ drug_b_col | tojson }},
    response_col={{ response_col | tojson }},
    model={{ model | tojson }},
)
print(f"Synergy: {stats['n_synergistic']}, Additive: {stats['n_additive']}, Antagonistic: {stats['n_antagonistic']}")""",
            imports=[
                "from lobster.services.drug_discovery.synergy_scoring_service import SynergyScoringService",
            ],
            parameters={
                "drug_a_col": drug_a_col,
                "drug_b_col": drug_b_col,
                "response_col": response_col,
                "model": model,
            },
            parameter_schema={
                "drug_a_col": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="",
                    required=True,
                    description="Column name for drug A dose/concentration in adata.obs",
                ),
                "drug_b_col": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="",
                    required=True,
                    description="Column name for drug B dose/concentration in adata.obs",
                ),
                "response_col": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="",
                    required=True,
                    description="Column name for response/effect (fractional inhibition, 0-1) in adata.obs",
                ),
                "model": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="bliss",
                    required=False,
                    description="Synergy model: 'bliss', 'hsa'",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_scored"],
        )

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------

    def bliss_independence(
        self, effect_a: float, effect_b: float, effect_ab: float
    ) -> Tuple[None, Dict[str, Any], AnalysisStep]:
        """
        Evaluate drug interaction using the Bliss Independence model.

        Bliss assumes drugs act independently on different targets. The expected
        combination effect is:

            E_expected = E_a + E_b - E_a * E_b

        The excess (E_observed - E_expected) indicates synergy (>0) or
        antagonism (<0).

        Args:
            effect_a: Fractional inhibition of drug A alone (0.0-1.0).
            effect_b: Fractional inhibition of drug B alone (0.0-1.0).
            effect_ab: Observed fractional inhibition of A+B (0.0-1.0).

        Returns:
            Tuple of (None, result dict with expected/observed/excess/classification,
            AnalysisStep).
        """
        logger.info(
            "Bliss Independence: E_a=%.3f, E_b=%.3f, E_ab=%.3f",
            effect_a,
            effect_b,
            effect_ab,
        )

        try:
            self._validate_effect(effect_a, "effect_a")
            self._validate_effect(effect_b, "effect_b")
            self._validate_effect(effect_ab, "effect_ab")

            expected = effect_a + effect_b - effect_a * effect_b
            excess = effect_ab - expected
            classification = self._classify_synergy(excess)

            result = {
                "model": "bliss_independence",
                "effect_a": effect_a,
                "effect_b": effect_b,
                "effect_ab_observed": effect_ab,
                "effect_ab_expected": round(expected, 6),
                "excess": round(excess, 6),
                "classification": classification,
                "thresholds": dict(SYNERGY_THRESHOLDS),
            }

            ir = self._create_ir_bliss(effect_a, effect_b, effect_ab)
            logger.info(
                "Bliss result: expected=%.4f, excess=%.4f, class=%s",
                expected,
                excess,
                classification,
            )
            return None, result, ir

        except SynergyScoringError:
            raise
        except Exception as e:
            raise SynergyScoringError(
                f"Bliss Independence calculation failed: {e}"
            ) from e

    def loewe_additivity(
        self,
        dose_a: float,
        dose_b: float,
        ic50_a: float,
        ic50_b: float,
        effect_ab: float,
    ) -> Tuple[None, Dict[str, Any], AnalysisStep]:
        """
        Evaluate drug interaction using the Loewe Additivity model.

        The Combination Index (CI) is computed as:

            CI = d_a / D_a + d_b / D_b

        where d_a, d_b are combination doses and D_a, D_b are doses of each
        drug alone that produce the same effect as the combination. Here we
        approximate D_a, D_b using IC50 values (i.e., the dose needed for
        50% effect alone, scaled to the observed combination effect).

        For the simplified model at effect_ab ~ 0.5:
            CI = d_a / IC50_a + d_b / IC50_b

        CI < 1: synergistic, CI = 1: additive, CI > 1: antagonistic.

        Args:
            dose_a: Dose of drug A in the combination.
            dose_b: Dose of drug B in the combination.
            ic50_a: IC50 of drug A alone.
            ic50_b: IC50 of drug B alone.
            effect_ab: Observed fractional effect of the combination (0.0-1.0).

        Returns:
            Tuple of (None, result dict with CI and classification, AnalysisStep).
        """
        logger.info(
            "Loewe Additivity: d_a=%.3f, d_b=%.3f, IC50_a=%.3f, IC50_b=%.3f",
            dose_a,
            dose_b,
            ic50_a,
            ic50_b,
        )

        try:
            self._validate_dose(dose_a, "dose_a")
            self._validate_dose(dose_b, "dose_b")
            self._validate_dose(ic50_a, "ic50_a")
            self._validate_dose(ic50_b, "ic50_b")
            self._validate_effect(effect_ab, "effect_ab")

            # Standard Loewe CI
            ci = dose_a / ic50_a + dose_b / ic50_b

            # Classify using CI thresholds
            if ci < 0.9:
                classification = "synergistic"
            elif ci > 1.1:
                classification = "antagonistic"
            else:
                classification = "additive"

            # Dose reduction index: how much less of each drug is needed
            dri_a = ic50_a / dose_a if dose_a > 0 else float("inf")
            dri_b = ic50_b / dose_b if dose_b > 0 else float("inf")

            result = {
                "model": "loewe_additivity",
                "dose_a": dose_a,
                "dose_b": dose_b,
                "ic50_a": ic50_a,
                "ic50_b": ic50_b,
                "effect_ab_observed": effect_ab,
                "combination_index": round(ci, 6),
                "classification": classification,
                "dose_reduction_index_a": round(dri_a, 4),
                "dose_reduction_index_b": round(dri_b, 4),
                "interpretation": (
                    f"CI={ci:.3f}: {'Synergistic' if ci < 0.9 else 'Antagonistic' if ci > 1.1 else 'Additive'}. "
                    f"Drug A dose reduced {dri_a:.1f}x, Drug B dose reduced {dri_b:.1f}x."
                ),
            }

            ir = self._create_ir_loewe(dose_a, dose_b, ic50_a, ic50_b, effect_ab)
            logger.info(
                "Loewe result: CI=%.4f, class=%s", ci, classification
            )
            return None, result, ir

        except SynergyScoringError:
            raise
        except Exception as e:
            raise SynergyScoringError(
                f"Loewe Additivity calculation failed: {e}"
            ) from e

    def hsa_model(
        self, effect_a: float, effect_b: float, effect_ab: float
    ) -> Tuple[None, Dict[str, Any], AnalysisStep]:
        """
        Evaluate drug interaction using the Highest Single Agent (HSA) model.

        The HSA reference is the maximum effect of either drug alone:

            excess = E_ab - max(E_a, E_b)

        Positive excess means the combination provides benefit beyond the
        best monotherapy.

        Args:
            effect_a: Fractional inhibition of drug A alone (0.0-1.0).
            effect_b: Fractional inhibition of drug B alone (0.0-1.0).
            effect_ab: Observed fractional inhibition of A+B (0.0-1.0).

        Returns:
            Tuple of (None, result dict with HSA reference/excess/classification,
            AnalysisStep).
        """
        logger.info(
            "HSA model: E_a=%.3f, E_b=%.3f, E_ab=%.3f",
            effect_a,
            effect_b,
            effect_ab,
        )

        try:
            self._validate_effect(effect_a, "effect_a")
            self._validate_effect(effect_b, "effect_b")
            self._validate_effect(effect_ab, "effect_ab")

            hsa_reference = max(effect_a, effect_b)
            excess = effect_ab - hsa_reference
            classification = self._classify_synergy(excess)

            # Which drug is the better single agent?
            best_single_agent = "A" if effect_a >= effect_b else "B"

            result = {
                "model": "hsa",
                "effect_a": effect_a,
                "effect_b": effect_b,
                "effect_ab_observed": effect_ab,
                "hsa_reference": round(hsa_reference, 6),
                "excess": round(excess, 6),
                "classification": classification,
                "best_single_agent": best_single_agent,
                "thresholds": dict(SYNERGY_THRESHOLDS),
            }

            ir = self._create_ir_hsa(effect_a, effect_b, effect_ab)
            logger.info(
                "HSA result: ref=%.4f, excess=%.4f, class=%s",
                hsa_reference,
                excess,
                classification,
            )
            return None, result, ir

        except SynergyScoringError:
            raise
        except Exception as e:
            raise SynergyScoringError(
                f"HSA model calculation failed: {e}"
            ) from e

    def score_combination_matrix(
        self,
        adata: anndata.AnnData,
        drug_a_col: str,
        drug_b_col: str,
        response_col: str,
        model: str = "bliss",
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Score a full dose-response combination matrix from AnnData.

        Each observation (row) in adata represents a single well/measurement
        with columns for drug A dose, drug B dose, and observed response.
        Monotherapy controls are identified as rows where one drug dose is 0.

        For the Bliss model, the method:
        1. Extracts monotherapy dose-response curves for each drug.
        2. Interpolates single-agent effects at each combination dose.
        3. Computes Bliss expected effect and excess for each combination point.
        4. Adds synergy scores and classifications to adata.obs.

        For the HSA model, the method uses max single-agent effect instead of
        Bliss expected.

        Args:
            adata: AnnData with combination screen data. Must have drug_a_col,
                drug_b_col, and response_col in adata.obs.
            drug_a_col: Column name for drug A dose in adata.obs.
            drug_b_col: Column name for drug B dose in adata.obs.
            response_col: Column name for response (fractional inhibition, 0-1).
            model: Synergy model to apply ('bliss' or 'hsa').

        Returns:
            Tuple of (annotated AnnData, summary stats dict, AnalysisStep).

        Raises:
            SynergyScoringError: If required columns are missing or model is
            unsupported.
        """
        if model not in ("bliss", "hsa"):
            raise SynergyScoringError(
                f"Unsupported synergy model: '{model}'. Use 'bliss' or 'hsa'."
            )

        for col in [drug_a_col, drug_b_col, response_col]:
            if col not in adata.obs.columns:
                raise SynergyScoringError(
                    f"Column '{col}' not found in adata.obs. "
                    f"Available: {list(adata.obs.columns)}"
                )

        logger.info(
            "Scoring combination matrix (%d observations) with %s model",
            adata.n_obs,
            model,
        )

        try:
            adata_result = adata.copy()
            obs = adata_result.obs

            doses_a = obs[drug_a_col].values.astype(float)
            doses_b = obs[drug_b_col].values.astype(float)
            responses = obs[response_col].values.astype(float)

            # Build monotherapy lookup: dose -> mean response
            # Drug A alone: drug_b_col == 0
            mono_a_mask = doses_b == 0
            mono_a_doses = doses_a[mono_a_mask]
            mono_a_effects = responses[mono_a_mask]

            # Drug B alone: drug_a_col == 0
            mono_b_mask = doses_a == 0
            mono_b_doses = doses_b[mono_b_mask]
            mono_b_effects = responses[mono_b_mask]

            # Build dose -> mean effect maps for interpolation
            def _build_dose_map(doses, effects):
                dose_map = {}
                for d, e in zip(doses, effects):
                    dose_map.setdefault(d, []).append(e)
                return {d: np.mean(es) for d, es in dose_map.items()}

            map_a = _build_dose_map(mono_a_doses, mono_a_effects)
            map_b = _build_dose_map(mono_b_doses, mono_b_effects)

            # Sort for interpolation
            sorted_a = sorted(map_a.items())
            sorted_b = sorted(map_b.items())

            def _interpolate_effect(dose, sorted_map):
                """Linear interpolation of effect at a given dose."""
                if not sorted_map:
                    return 0.0
                if dose <= sorted_map[0][0]:
                    return sorted_map[0][1]
                if dose >= sorted_map[-1][0]:
                    return sorted_map[-1][1]
                for i in range(len(sorted_map) - 1):
                    d_low, e_low = sorted_map[i]
                    d_high, e_high = sorted_map[i + 1]
                    if d_low <= dose <= d_high:
                        if d_high == d_low:
                            return e_low
                        frac = (dose - d_low) / (d_high - d_low)
                        return e_low + frac * (e_high - e_low)
                return sorted_map[-1][1]

            # Score each combination point
            synergy_scores = np.full(adata.n_obs, np.nan)
            expected_effects = np.full(adata.n_obs, np.nan)
            classifications = np.full(adata.n_obs, "", dtype=object)

            # Identify combination points (both doses > 0)
            combo_mask = (doses_a > 0) & (doses_b > 0)

            for idx in np.where(combo_mask)[0]:
                da = doses_a[idx]
                db = doses_b[idx]
                e_ab = responses[idx]

                e_a = _interpolate_effect(da, sorted_a)
                e_b = _interpolate_effect(db, sorted_b)

                # Clamp to [0, 1]
                e_a = max(0.0, min(1.0, e_a))
                e_b = max(0.0, min(1.0, e_b))

                if model == "bliss":
                    expected = e_a + e_b - e_a * e_b
                else:  # hsa
                    expected = max(e_a, e_b)

                excess = e_ab - expected

                synergy_scores[idx] = round(excess, 6)
                expected_effects[idx] = round(expected, 6)
                classifications[idx] = self._classify_synergy(excess)

            # Add to adata.obs
            adata_result.obs["synergy_score"] = synergy_scores
            adata_result.obs["expected_effect"] = expected_effects
            adata_result.obs["synergy_classification"] = classifications
            adata_result.obs["synergy_model"] = model

            # Summary statistics
            combo_scores = synergy_scores[combo_mask]
            valid_scores = combo_scores[~np.isnan(combo_scores)]
            combo_classes = classifications[combo_mask]

            n_synergistic = int(np.sum(combo_classes == "synergistic"))
            n_additive = int(np.sum(combo_classes == "additive"))
            n_antagonistic = int(np.sum(combo_classes == "antagonistic"))

            stats = {
                "model": model,
                "n_observations": int(adata.n_obs),
                "n_combination_points": int(combo_mask.sum()),
                "n_monotherapy_a": int(mono_a_mask.sum()),
                "n_monotherapy_b": int(mono_b_mask.sum()),
                "n_synergistic": n_synergistic,
                "n_additive": n_additive,
                "n_antagonistic": n_antagonistic,
                "mean_synergy_score": (
                    round(float(np.nanmean(valid_scores)), 4)
                    if len(valid_scores) > 0
                    else None
                ),
                "max_synergy_score": (
                    round(float(np.nanmax(valid_scores)), 4)
                    if len(valid_scores) > 0
                    else None
                ),
                "min_synergy_score": (
                    round(float(np.nanmin(valid_scores)), 4)
                    if len(valid_scores) > 0
                    else None
                ),
                "columns_added": [
                    "synergy_score",
                    "expected_effect",
                    "synergy_classification",
                    "synergy_model",
                ],
            }

            ir = self._create_ir_score_matrix(
                drug_a_col, drug_b_col, response_col, model
            )
            logger.info(
                "Matrix scoring complete: %d synergistic, %d additive, %d antagonistic (of %d combos)",
                n_synergistic,
                n_additive,
                n_antagonistic,
                int(combo_mask.sum()),
            )
            return adata_result, stats, ir

        except SynergyScoringError:
            raise
        except Exception as e:
            raise SynergyScoringError(
                f"Combination matrix scoring failed: {e}"
            ) from e
