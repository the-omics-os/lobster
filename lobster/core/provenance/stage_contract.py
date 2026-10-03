"""Declared-stage preconditions for analysis tools.

Each tool declares the input stages it accepts. This helper compares the declaration with
the modality's recorded lineage without inferring stage from matrix values. The policy is
per-tool because different analyses can require different input stages.

The check fails open: missing or unreadable lineage, absent declarations, and unknown
vocabularies do not prevent an analysis from running. An undeclared tool therefore retains
its existing behavior.

``CANONICAL_STEPS`` is a transcriptomics vocabulary. Other domains should define their own
steps before enabling declarations, because similarly named stages may have different
meanings across domains. The ``vocabulary`` parameter provides that extension point.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Iterable, Optional

from lobster.utils.logger import get_logger

if TYPE_CHECKING:
    from anndata import AnnData

logger = get_logger(__name__)

#: Rough pipeline order, cheapest-first. Used ONLY to pick which remedy to suggest when a
#: tool accepts several stages — NOT to infer whether a stage is "later" than another, which
#: would be a domain assumption of the kind this module refuses to make.
#:
#: ``BioinformaticsFileNaming.get_processing_step_order()`` (``utils/file_naming.py:305``)
#: serves the file-naming vocabulary rather than lineage stages, so it is not used here.
#: A dedicated order keeps remedy selection deterministic.
_STAGE_ORDER: tuple[str, ...] = (
    "raw",
    "quality_assessed",
    "filtered",
    "normalized",
    "filtered_normalized",
    "feature_selected",
    "batch_corrected",
    "reduced",
    "embedded",
    "clustered",
)

#: Remedies keyed by the stage a tool needs but did not get. Names the *action*, because a
#: generic "wrong stage" warning is the prose equivalent of the post-failure hints already in
#: this repo, which demonstrably do not work. Recovery belongs with the child agent, so the
#: text must be actionable by it.
_REMEDY: dict[str, str] = {
    "filtered_normalized": "run filter_and_normalize() first (QC + normalization)",
    "quality_assessed": "run assess_data_quality() first",
    "feature_selected": "run select_variable_features() first",
    "reduced": "run run_pca() first",
    "embedded": "run compute_neighbors_and_embed() first",
    "clustered": "run cluster_cells() first",
    # Deliberately tool-agnostic. A generic recovery suggestion avoids directing a
    # caller to a tool for a different analysis domain.
    "raw": (
        "pass the original unprocessed modality instead — this tool's output is only "
        "valid on raw counts"
    ),
}


#: Env flag. **Enabled by default**, like ``LOBSTER_HANDOFF_MANIFEST``. Unlike
#: ``LOBSTER_ROUTING_TELEMETRY`` / ``LOBSTER_CLAIM_VERIFICATION``, which are opt-in
#: measurement/observability instruments — off unless someone is looking. This is a
#: *product* behaviour: the warning is the fix, and a wrong-stage PCA is a scientific
#: correctness problem, so defaulting it off would ship the defect and the cure together.
#: The off switch supports controlled validation of the enabled and disabled behavior.
STAGE_CONTRACT_ENV_VAR = "LOBSTER_STAGE_CONTRACT"


def stage_contract_enabled() -> bool:
    """True unless explicitly disabled. Any of 0/false/no/off turns it off."""
    raw = os.environ.get(STAGE_CONTRACT_ENV_VAR)
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def describe_stage(adata: "AnnData") -> Optional[str]:
    """The input's recorded processing step, or ``None`` if unknown.

    ``None`` means "cannot tell" and must be treated as such by callers — never as "raw".
    Conflating the two makes lineage ambiguous: processed data could be reported as raw
    defaulted to ``"raw"``, so correctly-processed data reported raw and the wrong-stage
    artifact validated.
    """
    try:
        from lobster.core.provenance.lineage import get_lineage_dict

        lineage = get_lineage_dict(adata)
        if not lineage:
            return None
        step = lineage.get("processing_step")
        return str(step) if step else None
    except Exception as exc:  # noqa: BLE001 - never break a tool over metadata
        logger.debug(f"Stage contract: could not read lineage: {exc}")
        return None


def check_input_stage(
    adata: "AnnData",
    accepted: Iterable[str],
    tool_name: str,
    vocabulary: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """Check an input's stage against a tool's declaration. Warning text, or ``None``.

    Args:
        adata: the input modality.
        accepted: stages this tool accepts. Empty/None ⇒ undeclared ⇒ no check.
        tool_name: named in the warning so the supervisor knows which call to fix.
        vocabulary: valid step names for this domain. Defaults to the transcriptomics
            ``CANONICAL_STEPS``. The seam for per-domain vocabularies (see module docstring).

    Returns:
        An actionable warning naming the observed stage, the accepted stages and the
        remedy — or ``None`` when the input is acceptable, or when the check cannot be
        made safely.

    Never raises. A metadata check that can break an analysis is a worse defect than the
    one it reports.
    """
    try:
        if not stage_contract_enabled():
            return None  # off arm of a matched-pair measurement

        accepted_set = {str(s) for s in (accepted or ())}
        if not accepted_set:
            return None  # undeclared: today's behaviour, deliberately

        if vocabulary is None:
            from lobster.core.provenance.lineage import CANONICAL_STEPS

            vocabulary = CANONICAL_STEPS
        vocab = {str(s) for s in vocabulary}

        # A declaration outside the vocabulary is a *declaration bug*, not an input
        # problem. Skipping keeps a wrong-vocabulary declaration from producing confident
        # nonsense; the contract test is what surfaces it at CI time.
        unknown = accepted_set - vocab
        if unknown:
            logger.debug(
                f"Stage contract: {tool_name} declares step(s) outside the vocabulary "
                f"{sorted(unknown)}; skipping check"
            )
            return None

        observed = describe_stage(adata)
        if observed is None:
            return None  # cannot tell -> silent, never assume "raw"
        if observed in accepted_set:
            return None

        wanted = ", ".join(sorted(accepted_set))
        # Pick the remedy for the CHEAPEST acceptable stage, not the alphabetically first.
        # Keyed on alphabetical order this told a caller with raw input to
        # "produce a 'batch_corrected' modality" — technically an accepted stage, but three
        # steps past what it actually needs, and batch correction is not even applicable to
        # a single-batch dataset. A remedy that names the wrong action is worse than none:
        # it sends the child agent down a path it should not take.
        remedy = next(
            (_REMEDY[s] for s in _STAGE_ORDER if s in accepted_set and s in _REMEDY),
            f"produce one of {{{wanted}}} first",
        )
        return (
            f"STAGE WARNING: input is at stage '{observed}'; {tool_name} expects one of "
            f"{{{wanted}}} — {remedy}. Proceeding anyway, but the result may be "
            f"scientifically invalid and will compete with a correctly-staged artifact."
        )
    except Exception as exc:  # noqa: BLE001 - fail open, always
        logger.debug(f"Stage contract: check failed for {tool_name}: {exc}")
        return None


def prepend_stage_warning(response: str, warning: Optional[str]) -> str:
    """Put the warning where the supervisor reads it — first, not buried in a footer."""
    if not warning:
        return response
    return f"⚠️  {warning}\n\n{response}"


def resolve_input_stage(data_manager: Any, modality_name: str) -> Optional[str]:
    """Convenience for tools holding a name rather than an AnnData. ``None`` if unknown."""
    try:
        return describe_stage(data_manager.get_modality(modality_name))
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Stage contract: could not resolve '{modality_name}': {exc}")
        return None
