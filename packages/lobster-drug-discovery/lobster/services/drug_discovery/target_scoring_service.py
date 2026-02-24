"""
Target scoring service for druggability assessment.

Provides weighted evidence scoring, multi-target ranking, and confidence
classification for drug target prioritization. Uses TARGET_EVIDENCE_WEIGHTS
from config for reproducible, tunable scoring.

All methods return 3-tuples (None, Dict, AnalysisStep) for provenance tracking and
reproducible notebook export via /pipeline export.
"""

from typing import Any, Dict, List, Tuple

from lobster.agents.drug_discovery.config import TARGET_EVIDENCE_WEIGHTS
from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class TargetScoringError(Exception):
    """Base exception for target scoring operations."""

    pass


class TargetScoringService:
    """
    Stateless target druggability scoring service.

    Computes composite druggability scores from multi-dimensional evidence
    vectors using configurable weights. Each evidence category is scored
    on a 0.0-1.0 scale and combined via weighted summation.

    Evidence categories (from TARGET_EVIDENCE_WEIGHTS):
        - genetic_association (0.30): GWAS/Mendelian genetics support
        - known_drug (0.25): Existing drugs targeting this gene/protein
        - expression_specificity (0.20): Tissue-specific expression
        - pathogenicity (0.15): Pathogenic variant association
        - literature (0.10): Publication evidence density
    """

    def __init__(self):
        """Initialize the target scoring service (stateless)."""
        logger.debug("Initializing TargetScoringService")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _validate_evidence(
        self, evidence_dict: Dict[str, float], context: str = ""
    ) -> None:
        """Validate evidence dict values are in [0.0, 1.0].

        Args:
            evidence_dict: Evidence category -> score mapping.
            context: Optional context string for error messages.

        Raises:
            TargetScoringError: If values are out of range or keys are invalid.
        """
        valid_keys = set(TARGET_EVIDENCE_WEIGHTS.keys())

        for key, value in evidence_dict.items():
            if key not in valid_keys:
                logger.warning(
                    "Unknown evidence category '%s' %s; it will be ignored. "
                    "Valid categories: %s",
                    key,
                    f"for {context}" if context else "",
                    sorted(valid_keys),
                )
            if not isinstance(value, (int, float)):
                raise TargetScoringError(
                    f"Evidence score for '{key}' must be numeric, "
                    f"got {type(value).__name__}: {value!r}"
                )
            if not (0.0 <= value <= 1.0):
                raise TargetScoringError(
                    f"Evidence score for '{key}' must be in [0.0, 1.0], "
                    f"got {value}"
                )

    def _compute_score(
        self, evidence_dict: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """Compute weighted composite score.

        Args:
            evidence_dict: Evidence category -> score mapping.

        Returns:
            Tuple of (overall_score, component_scores dict).
        """
        component_scores = {}
        overall = 0.0

        for category, weight in TARGET_EVIDENCE_WEIGHTS.items():
            raw = evidence_dict.get(category, 0.0)
            weighted = raw * weight
            component_scores[category] = {
                "raw_score": round(raw, 4),
                "weight": weight,
                "weighted_score": round(weighted, 4),
            }
            overall += weighted

        return round(overall, 4), component_scores

    # -------------------------------------------------------------------------
    # IR builders
    # -------------------------------------------------------------------------

    def _create_ir_score_target(
        self, evidence_dict: Dict[str, float]
    ) -> AnalysisStep:
        return AnalysisStep(
            operation="lobster.services.drug_discovery.target_scoring.score",
            tool_name="score_target",
            description=(
                "Compute composite druggability score from multi-dimensional "
                "evidence vector using weighted summation"
            ),
            library="lobster",
            code_template="""from lobster.services.drug_discovery.target_scoring_service import TargetScoringService

service = TargetScoringService()
_, result, _ = service.score_target({{ evidence_dict | tojson }})
print(f"Target score: {result['overall_score']:.3f} ({result['classification']})")
for cat, detail in result['component_scores'].items():
    print(f"  {cat}: {detail['raw_score']:.2f} x {detail['weight']:.2f} = {detail['weighted_score']:.3f}")""",
            imports=[
                "from lobster.services.drug_discovery.target_scoring_service import TargetScoringService",
            ],
            parameters={"evidence_dict": evidence_dict},
            parameter_schema={
                "evidence_dict": ParameterSpec(
                    param_type="Dict[str, float]",
                    papermill_injectable=True,
                    default_value={},
                    required=True,
                    description=(
                        "Evidence scores per category (0.0-1.0). Keys: "
                        "genetic_association, known_drug, expression_specificity, "
                        "pathogenicity, literature"
                    ),
                ),
            },
            input_entities=["evidence_dict"],
            output_entities=["score_result"],
        )

    def _create_ir_rank_targets(
        self, n_targets: int
    ) -> AnalysisStep:
        return AnalysisStep(
            operation="lobster.services.drug_discovery.target_scoring.rank",
            tool_name="rank_targets",
            description=f"Rank {n_targets} targets by composite druggability score",
            library="lobster",
            code_template="""from lobster.services.drug_discovery.target_scoring_service import TargetScoringService

service = TargetScoringService()
targets_evidence = {{ targets_evidence | tojson }}
_, result, _ = service.rank_targets(targets_evidence)
for entry in result['ranked_targets']:
    print(f"  {entry['rank']}. {entry['gene_symbol']}: {entry['overall_score']:.3f} ({entry['classification']})")""",
            imports=[
                "from lobster.services.drug_discovery.target_scoring_service import TargetScoringService",
            ],
            parameters={"n_targets": n_targets},
            parameter_schema={
                "n_targets": ParameterSpec(
                    param_type="int",
                    papermill_injectable=False,
                    default_value=0,
                    required=True,
                    description="Number of targets being ranked",
                ),
            },
            input_entities=["targets_evidence"],
            output_entities=["ranked_targets"],
        )

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------

    def classify_target(self, score: float) -> str:
        """
        Classify a target into a confidence tier based on its composite score.

        Args:
            score: Composite druggability score (0.0-1.0).

        Returns:
            Classification string: 'high_confidence', 'medium_confidence',
            or 'low_confidence'.
        """
        if score > 0.7:
            return "high_confidence"
        elif score > 0.4:
            return "medium_confidence"
        else:
            return "low_confidence"

    def score_target(
        self, evidence_dict: Dict[str, float]
    ) -> Tuple[None, Dict[str, Any], AnalysisStep]:
        """
        Compute a composite target druggability score from evidence.

        The score is a weighted sum of evidence categories defined in
        TARGET_EVIDENCE_WEIGHTS. Each evidence value must be in [0.0, 1.0].
        Missing categories default to 0.0 (no evidence).

        Args:
            evidence_dict: Mapping of evidence category to score (0.0-1.0).
                Valid keys: genetic_association, known_drug,
                expression_specificity, pathogenicity, literature.

        Returns:
            Tuple of (None, result dict with overall_score + component_scores
            + classification, AnalysisStep).

        Raises:
            TargetScoringError: If evidence values are out of range.
        """
        logger.info(
            "Scoring target with %d evidence categories", len(evidence_dict)
        )

        try:
            self._validate_evidence(evidence_dict)

            overall_score, component_scores = self._compute_score(evidence_dict)
            classification = self.classify_target(overall_score)

            # Identify strongest and weakest evidence
            present_cats = {
                k: v
                for k, v in evidence_dict.items()
                if k in TARGET_EVIDENCE_WEIGHTS
            }
            strongest = (
                max(present_cats, key=present_cats.get)
                if present_cats
                else None
            )
            weakest = (
                min(present_cats, key=present_cats.get)
                if present_cats
                else None
            )

            # Missing evidence categories
            missing = [
                k
                for k in TARGET_EVIDENCE_WEIGHTS
                if k not in evidence_dict
            ]

            result = {
                "overall_score": overall_score,
                "classification": classification,
                "component_scores": component_scores,
                "evidence_provided": list(present_cats.keys()),
                "evidence_missing": missing,
                "strongest_evidence": strongest,
                "weakest_evidence": weakest,
                "weights_used": dict(TARGET_EVIDENCE_WEIGHTS),
                "max_possible_score": round(
                    sum(TARGET_EVIDENCE_WEIGHTS.values()), 4
                ),
            }

            ir = self._create_ir_score_target(evidence_dict)
            logger.info(
                "Target scored: %.3f (%s)", overall_score, classification
            )
            return None, result, ir

        except TargetScoringError:
            raise
        except Exception as e:
            raise TargetScoringError(
                f"Target scoring failed: {e}"
            ) from e

    def rank_targets(
        self,
        targets_evidence: List[Tuple[str, Dict[str, float]]],
    ) -> Tuple[None, Dict[str, Any], AnalysisStep]:
        """
        Rank multiple targets by composite druggability score.

        Args:
            targets_evidence: List of (gene_symbol, evidence_dict) tuples.
                Each evidence_dict maps evidence categories to scores (0.0-1.0).

        Returns:
            Tuple of (None, dict with ranked_targets list sorted by score
            descending, AnalysisStep).

        Raises:
            TargetScoringError: If any evidence values are invalid.
        """
        if not targets_evidence:
            raise TargetScoringError(
                "Empty targets list. Provide at least one (gene_symbol, evidence_dict) tuple."
            )

        logger.info("Ranking %d targets", len(targets_evidence))

        try:
            scored = []
            for gene_symbol, evidence_dict in targets_evidence:
                self._validate_evidence(evidence_dict, context=gene_symbol)
                overall_score, component_scores = self._compute_score(
                    evidence_dict
                )
                classification = self.classify_target(overall_score)

                scored.append(
                    {
                        "gene_symbol": gene_symbol,
                        "overall_score": overall_score,
                        "classification": classification,
                        "component_scores": component_scores,
                        "evidence_provided": {
                            k: v
                            for k, v in evidence_dict.items()
                            if k in TARGET_EVIDENCE_WEIGHTS
                        },
                    }
                )

            # Sort descending by overall score, then alphabetical for ties
            scored.sort(
                key=lambda x: (-x["overall_score"], x["gene_symbol"])
            )

            # Add rank position
            for idx, entry in enumerate(scored, start=1):
                entry["rank"] = idx

            # Summary statistics
            scores = [s["overall_score"] for s in scored]
            mean_score = sum(scores) / len(scores)
            high_conf = sum(
                1 for s in scored if s["classification"] == "high_confidence"
            )
            med_conf = sum(
                1 for s in scored if s["classification"] == "medium_confidence"
            )
            low_conf = sum(
                1 for s in scored if s["classification"] == "low_confidence"
            )

            result = {
                "ranked_targets": scored,
                "n_targets": len(scored),
                "mean_score": round(mean_score, 4),
                "best_target": scored[0]["gene_symbol"] if scored else None,
                "best_score": scored[0]["overall_score"] if scored else None,
                "summary": {
                    "high_confidence": high_conf,
                    "medium_confidence": med_conf,
                    "low_confidence": low_conf,
                },
            }

            ir = self._create_ir_rank_targets(len(targets_evidence))
            logger.info(
                "Ranking complete: best=%s (%.3f), mean=%.3f",
                result["best_target"],
                result["best_score"],
                mean_score,
            )
            return None, result, ir

        except TargetScoringError:
            raise
        except Exception as e:
            raise TargetScoringError(
                f"Target ranking failed: {e}"
            ) from e
