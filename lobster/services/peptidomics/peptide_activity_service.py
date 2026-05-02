"""
Peptide activity prediction service — heuristic scoring.

Feature-based heuristics for antimicrobial (AMP), cell-penetrating (CPP),
and toxicity prediction. No ML models — deterministic and explainable.
"""

import logging
from typing import Any, Dict, List, Tuple

import anndata as ad
import numpy as np

logger = logging.getLogger(__name__)

HYDROPHOBIC_AA = set("AILMFWV")


def _get_peptide_calc(sequence: str):
    """Get peptides.Peptide instance, or None if package unavailable."""
    try:
        from peptides import Peptide

        return Peptide(sequence)
    except (ImportError, ValueError):
        return None


class PeptideActivityService:
    """Heuristic activity prediction for peptides."""

    def predict_activity(
        self,
        adata: ad.AnnData,
        activity_type: str = "antimicrobial",
        sequence_col: str = "sequence",
    ) -> Tuple[ad.AnnData, Dict[str, Any], Dict[str, Any]]:
        """
        Predict peptide activity via feature-based heuristics.

        Args:
            adata: AnnData with sequences in obs[sequence_col]
            activity_type: "antimicrobial", "cell_penetrating", or "toxic"
            sequence_col: Column name containing peptide sequences

        Returns:
            (AnnData, stats_dict, analysis_step_ir)
        """
        if sequence_col not in adata.obs.columns:
            raise ValueError(
                f"Column '{sequence_col}' not found. Available: {list(adata.obs.columns)}"
            )

        predictors = {
            "antimicrobial": self._predict_amp,
            "cell_penetrating": self._predict_cpp,
            "toxic": self._predict_toxicity,
        }

        if activity_type not in predictors:
            raise ValueError(
                f"Unknown activity_type '{activity_type}'. "
                f"Use: {', '.join(predictors.keys())}"
            )

        predict_fn = predictors[activity_type]
        sequences = adata.obs[sequence_col].astype(str).tolist()

        scores = []
        details: List[Dict[str, Any]] = []
        for seq in sequences:
            clean_seq = "".join(c for c in seq.upper() if c.isalpha())
            if not clean_seq or len(clean_seq) < 2:
                scores.append(np.nan)
                details.append({})
                continue
            result = predict_fn(clean_seq)
            scores.append(result["score"])
            details.append(result)

        adata_out = adata.copy()
        col_name = f"peptide_{activity_type}_score"
        adata_out.obs[col_name] = scores
        adata_out.obs[f"peptide_{activity_type}_label"] = [
            "positive" if s > 0.5 else "negative" if not np.isnan(s) else "unknown"
            for s in scores
        ]

        valid_scores = [s for s in scores if not np.isnan(s)]
        n_positive = sum(1 for s in valid_scores if s > 0.5)

        stats = {
            "activity_type": activity_type,
            "n_peptides": len(sequences),
            "n_scored": len(valid_scores),
            "n_positive": n_positive,
            "n_negative": len(valid_scores) - n_positive,
            "mean_score": float(np.mean(valid_scores)) if valid_scores else 0.0,
            "method": "heuristic",
            "confidence": "heuristic — not a validated ML model",
        }

        from lobster.core.provenance.analysis_ir import AnalysisStep, ParameterSpec

        ir = AnalysisStep(
            operation=f"peptidomics.predict_{activity_type}",
            tool_name="predict_peptide_activity",
            description=f"Heuristic {activity_type} prediction: {n_positive}/{len(valid_scores)} positive",
            library="peptides",
            imports=["from peptides import Peptide"],
            code_template=(
                "# Heuristic {{ activity_type }} prediction\n"
                "# See PeptideActivityService for scoring rules"
            ),
            parameters={
                "activity_type": activity_type,
                "sequence_col": sequence_col,
            },
            parameter_schema={
                "activity_type": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="antimicrobial",
                    required=True,
                    description="Activity type: antimicrobial, cell_penetrating, or toxic",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "method": "heuristic",
                "n_positive": n_positive,
                "n_scored": len(valid_scores),
            },
        )

        return adata_out, stats, ir

    def _predict_amp(self, sequence: str) -> Dict[str, Any]:
        """
        AMP scoring based on Wimley-White and Boman indices.

        Rules:
        1. Net charge +2 to +9 (most AMPs are cationic)
        2. Hydrophobic ratio 40-60%
        3. Boman index < 2.48 (membrane binding potential)
        4. Length 10-50 (typical AMP range)
        5. Amphipathicity (hydrophobic moment > 0.5)
        """
        score = 0.0
        reasons = []

        p = _get_peptide_calc(sequence)

        # Rule 1: charge
        if p:
            charge = p.charge(pH=7.4)
        else:
            charge = float(
                sum(1 for aa in sequence if aa in "RK")
                + sum(0.1 for aa in sequence if aa == "H")
                - sum(1 for aa in sequence if aa in "DE")
            )
        if 2 <= charge <= 9:
            score += 0.3
            reasons.append(f"cationic charge ({charge:+.1f})")

        # Rule 2: hydrophobic ratio
        hydro_ratio = sum(1 for aa in sequence if aa in HYDROPHOBIC_AA) / len(sequence)
        if 0.4 <= hydro_ratio <= 0.6:
            score += 0.25
            reasons.append(f"hydrophobic ratio {hydro_ratio:.2f}")

        # Rule 3: Boman index
        if p:
            try:
                boman = p.boman()
                if boman < 2.48:
                    score += 0.2
                    reasons.append(f"Boman {boman:.2f} < 2.48")
            except Exception:
                pass

        # Rule 4: length
        if 10 <= len(sequence) <= 50:
            score += 0.15
            reasons.append(f"length {len(sequence)} in AMP range")

        # Rule 5: amphipathicity
        if p:
            try:
                hm = p.hydrophobic_moment()
                if hm > 0.5:
                    score += 0.1
                    reasons.append(f"HM {hm:.2f} > 0.5")
            except Exception:
                pass

        return {
            "score": min(score, 1.0),
            "method": "Wimley-White/Boman heuristic",
            "reasons": reasons,
        }

    def _predict_cpp(self, sequence: str) -> Dict[str, Any]:
        """
        CPP (cell-penetrating peptide) scoring.

        Rules: arginine-rich (>20%), cationic (charge >3), amphipathic (GRAVY < 0).
        """
        score = 0.0
        reasons = []

        p = _get_peptide_calc(sequence)

        # Rule 1: arginine content >20%
        arg_ratio = sequence.count("R") / len(sequence)
        if arg_ratio > 0.2:
            score += 0.35
            reasons.append(f"Arg content {arg_ratio:.0%}")

        # Rule 2: cationic charge >3
        if p:
            charge = p.charge(pH=7.4)
        else:
            charge = float(
                sum(1 for aa in sequence if aa in "RK")
                - sum(1 for aa in sequence if aa in "DE")
            )
        if charge > 3:
            score += 0.25
            reasons.append(f"charge {charge:+.1f}")

        # Rule 3: GRAVY < 0 (not too hydrophobic)
        if p:
            try:
                gravy = p.hydrophobicity("KyteDoolittle")
                if gravy < 0:
                    score += 0.2
                    reasons.append(f"GRAVY {gravy:.2f}")
            except Exception:
                pass

        # Rule 4: length 5-30
        if 5 <= len(sequence) <= 30:
            score += 0.2
            reasons.append(f"length {len(sequence)}")

        return {
            "score": min(score, 1.0),
            "method": "arginine-rich/cationic heuristic",
            "reasons": reasons,
        }

    def _predict_toxicity(self, sequence: str) -> Dict[str, Any]:
        """
        Toxicity scoring heuristic.

        Rules: high hydrophobicity, short length, high cationic charge.
        """
        score = 0.0
        reasons = []

        p = _get_peptide_calc(sequence)

        # Rule 1: hydrophobic ratio >0.6
        hydro_ratio = sum(1 for aa in sequence if aa in HYDROPHOBIC_AA) / len(sequence)
        if hydro_ratio > 0.6:
            score += 0.3
            reasons.append(f"high hydrophobicity {hydro_ratio:.2f}")

        # Rule 2: short length <30
        if len(sequence) < 30:
            score += 0.2
            reasons.append(f"short ({len(sequence)} aa)")

        # Rule 3: high charge >5
        if p:
            charge = p.charge(pH=7.4)
        else:
            charge = float(
                sum(1 for aa in sequence if aa in "RK")
                - sum(1 for aa in sequence if aa in "DE")
            )
        if charge > 5:
            score += 0.25
            reasons.append(f"high charge {charge:+.1f}")

        # Rule 4: tryptophan-rich (membrane-disruptive)
        trp_ratio = sequence.count("W") / len(sequence)
        if trp_ratio > 0.1:
            score += 0.25
            reasons.append(f"Trp-rich {trp_ratio:.0%}")

        return {
            "score": min(score, 1.0),
            "method": "hydrophobicity/charge heuristic",
            "reasons": reasons,
        }
