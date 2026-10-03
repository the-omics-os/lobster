"""Tests for declared-stage preconditions on analysis inputs.

The helper must fail open when it cannot assess an input, avoid treating unknown lineage
as raw, never raise, and provide an actionable warning when a declared stage is unmet.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from lobster.core.provenance.lineage import LineageMetadata, attach_lineage
from lobster.core.provenance.stage_contract import (
    check_input_stage,
    describe_stage,
    prepend_stage_warning,
    resolve_input_stage,
)


def _adata(step: str | None = None):
    a = ad.AnnData(np.ones((4, 3), dtype=float))
    if step is not None:
        attach_lineage(
            a,
            LineageMetadata(
                base_name="x",
                version=1,
                processing_step=step,
                parent_modality=None,
                step_summary=None,
                created_at="2026-08-10T00:00:00",
            ),
        )
    return a


class TestFailOpen:
    """Absent must remain silent. This is what makes the helper safe to add gradually."""

    def test_no_declaration_means_no_check(self):
        assert check_input_stage(_adata("raw"), accepted=(), tool_name="t") is None

    def test_no_lineage_means_no_warning(self):
        assert (
            check_input_stage(
                _adata(None), accepted={"filtered_normalized"}, tool_name="t"
            )
            is None
        )

    def test_unknown_stage_is_not_assumed_to_be_raw(self):
        """Unknown lineage is not assumed to be raw.

        If absence were read as 'raw', every modality without lineage would trip a
        not-raw precondition -- turning a silent gap into a flood of false warnings.
        """
        assert describe_stage(_adata(None)) is None

    def test_declaration_outside_vocabulary_skips_the_check(self):
        """The helper skips declarations outside the active vocabulary.

        This is the seam for per-domain vocabularies: a proteomics tool declaring a step
        that no registered vocabulary contains gets NO check, rather than a wrong one. The
        contract test is what surfaces it at CI time.
        """
        warning = check_input_stage(
            _adata("raw"), accepted={"proteomics_imputed"}, tool_name="t"
        )
        assert warning is None

    def test_never_raises_on_a_broken_object(self):
        class _Exploding:
            @property
            def uns(self):
                raise RuntimeError("boom")

        assert describe_stage(_Exploding()) is None
        assert (
            check_input_stage(_Exploding(), accepted={"reduced"}, tool_name="t") is None
        )

    def test_resolve_input_stage_survives_a_missing_modality(self):
        class _DM:
            def get_modality(self, name):
                raise KeyError(name)

        assert resolve_input_stage(_DM(), "nope") is None


class TestWarningIsActionable:
    """A generic warning is the prose equivalent of hints we know don't work."""

    def test_names_observed_stage_accepted_stages_and_remedy(self):
        warning = check_input_stage(
            _adata("raw"), accepted={"filtered_normalized"}, tool_name="run_pca"
        )
        assert warning is not None
        assert "'raw'" in warning
        assert "filtered_normalized" in warning
        assert "run_pca" in warning
        assert (
            "filter_and_normalize" in warning
        ), "must name the remedy, not just the fault"

    def test_says_it_is_proceeding(self):
        """Warn-and-proceed: the reader must not think the call was refused."""
        warning = check_input_stage(
            _adata("raw"), accepted={"filtered_normalized"}, tool_name="run_pca"
        )
        assert "Proceeding" in warning

    def test_accepted_stage_produces_no_warning(self):
        assert (
            check_input_stage(
                _adata("filtered_normalized"),
                accepted={"filtered_normalized", "feature_selected"},
                tool_name="run_pca",
            )
            is None
        )

    def test_prepend_puts_the_warning_first(self):
        out = prepend_stage_warning("PCA complete", "STAGE WARNING: x")
        assert out.startswith("⚠️")
        assert out.index("STAGE WARNING") < out.index("PCA complete")

    def test_prepend_is_a_noop_without_a_warning(self):
        assert prepend_stage_warning("PCA complete", None) == "PCA complete"


class TestTheConfirmedCase:
    """The behavior that requires stage validation."""

    def test_pca_on_raw_warns(self):
        from lobster.agents.transcriptomics.shared_tools import PCA_ACCEPTED_STAGES

        warning = check_input_stage(
            _adata("raw"), accepted=PCA_ACCEPTED_STAGES, tool_name="run_pca"
        )
        assert warning is not None

    @pytest.mark.parametrize(
        "step", ["filtered_normalized", "feature_selected", "batch_corrected"]
    )
    def test_pca_on_properly_staged_input_is_silent(self, step):
        """The failure mode that matters: rejecting valid work looks like strictness."""
        from lobster.agents.transcriptomics.shared_tools import PCA_ACCEPTED_STAGES

        assert (
            check_input_stage(
                _adata(step), accepted=PCA_ACCEPTED_STAGES, tool_name="run_pca"
            )
            is None
        )


class TestDeRegressionGuard:
    """A DE tool must still accept raw counts.

    A blanket "reject unnormalized input" rule would break DESeq2, whose requirement is
    written in prose at `de_analysis_expert.py:138` ("CRITICAL: DESeq2 requires raw integer
    counts"). Without this guard, the obvious implementation of the stage check silently
    breaks DE analysis, and the breakage looks like correct strictness.
    """

    def test_de_tools_declare_no_stage_precondition(self):
        """DE reads `adata.raw.X`, which survives normalization, so it is stage-agnostic.

        Verified rather than assumed: `_extract_raw_counts` prefers `adata.raw.X` and only
        falls back to `adata.X` with a warning (`de_analysis_expert.py:149-161`). A stage
        declaration would therefore be a *false* precondition here -- the tool works at any
        stage -- which is why DE is deliberately left undeclared.
        """
        from pathlib import Path

        src = (
            Path(__file__).resolve().parents[3]
            / "packages/lobster-transcriptomics/lobster/agents/transcriptomics"
            / "de_analysis_expert.py"
        )
        text = src.read_text()
        assert "adata.raw.X" in text, "DE's raw-count source moved; re-check this guard"
        assert "check_input_stage" not in text, (
            "A DE tool must NOT declare a stage precondition: it reads adata.raw.X and "
            "works at any lineage stage. Declaring one would reject valid input while "
            "looking like correct strictness."
        )

    def test_helper_would_permit_raw_if_a_de_tool_ever_declared_it(self):
        """The mechanism can express "requires raw" -- it is per-tool policy, not a global rule."""
        assert (
            check_input_stage(
                _adata("raw"), accepted={"raw"}, tool_name="create_pseudobulk_matrix"
            )
            is None
        )

    def test_a_de_tool_declaring_raw_would_warn_on_normalized_input(self):
        """The inverse precondition works, which is the point of per-tool policy."""
        warning = check_input_stage(
            _adata("filtered_normalized"),
            accepted={"raw"},
            tool_name="create_pseudobulk_matrix",
        )
        assert warning is not None
        # Remedy is tool-agnostic on purpose -- see the run-4 note in the QC test above.
        assert "raw counts" in warning


class TestRemedyPicksTheCheapestPath:
    """The remedy must name the action the caller should actually take.

    Keyed on alphabetical order, a raw input to `run_pca` was told to "produce a
    'batch_corrected' modality" -- an accepted stage, but three steps further than needed,
    and batch correction is not even applicable to a single-batch dataset. A remedy naming
    the wrong action is worse than none: it sends the child agent down a path it should not
    take, which is how the existing prose hints fail.
    """

    def test_raw_input_is_told_to_normalize_not_to_batch_correct(self):
        from lobster.agents.transcriptomics.shared_tools import PCA_ACCEPTED_STAGES

        warning = check_input_stage(
            _adata("raw"), accepted=PCA_ACCEPTED_STAGES, tool_name="run_pca"
        )
        assert "filter_and_normalize" in warning
        assert "batch_correct" not in warning.split("—")[1]

    def test_unrecognised_accepted_set_falls_back_without_crashing(self):
        """A vocabulary-valid step with no remedy entry must still produce usable text."""
        warning = check_input_stage(
            _adata("raw"), accepted={"pseudobulk"}, tool_name="t"
        )
        assert warning is not None and "pseudobulk" in warning


class TestNoHarmonizationWithFileNamingVocabulary:
    """Pin that file-naming and lineage stage vocabularies remain independent.

    `BioinformaticsFileNaming.get_processing_step_order()` uses a file-naming vocabulary.
    If its coverage changes, revisit whether the two vocabularies should be reconciled.
    """

    def test_file_naming_order_does_not_cover_lineage_vocabulary(self):
        from lobster.core.provenance.lineage import CANONICAL_STEPS
        from lobster.utils.file_naming import BioinformaticsFileNaming

        f = BioinformaticsFileNaming()
        unknown = {s for s in CANONICAL_STEPS if f.get_processing_step_order(s) == 999}
        assert {
            "raw",
            "filtered_normalized",
            "feature_selected",
            "reduced",
        } <= unknown, (
            "file_naming's order now covers lineage steps it previously did not. "
            "Re-evaluate whether the two vocabularies should be reconciled."
        )

    def test_stage_order_covers_the_steps_we_declare_against(self):
        """Our own ordering must cover every stage a declaration can reference."""
        from lobster.agents.transcriptomics.shared_tools import PCA_ACCEPTED_STAGES
        from lobster.core.provenance.stage_contract import _STAGE_ORDER

        assert set(PCA_ACCEPTED_STAGES) <= set(_STAGE_ORDER)


class TestInverseDirectionDeclaration:
    """`assess_data_quality` needs raw counts, unlike PCA.

    Per-tool declarations allow different analysis operations to require different stages.
    """

    def test_qc_on_normalized_input_warns(self):
        from lobster.agents.transcriptomics.shared_tools import QC_ACCEPTED_STAGES

        warning = check_input_stage(
            _adata("filtered_normalized"),
            accepted=QC_ACCEPTED_STAGES,
            tool_name="assess_data_quality",
        )
        assert warning is not None
        assert "raw counts" in warning, "remedy must name the raw requirement"
        assert "DE " not in warning and "DESeq" not in warning, (
            "the remedy must not name a different tool: it should describe the input stage "
            "needed for this analysis"
        )

    @pytest.mark.parametrize("step", ["raw", "filtered", "quality_assessed"])
    def test_qc_on_acceptable_input_is_silent(self, step):
        from lobster.agents.transcriptomics.shared_tools import QC_ACCEPTED_STAGES

        assert (
            check_input_stage(
                _adata(step),
                accepted=QC_ACCEPTED_STAGES,
                tool_name="assess_data_quality",
            )
            is None
        )

    def test_the_two_declarations_are_genuinely_opposed(self):
        """If a global rule could satisfy both, the per-tool design would be unnecessary."""
        from lobster.agents.transcriptomics.shared_tools import (
            PCA_ACCEPTED_STAGES,
            QC_ACCEPTED_STAGES,
        )

        assert not (PCA_ACCEPTED_STAGES & QC_ACCEPTED_STAGES), (
            "the two declarations must be disjoint: one requires processed input, the "
            "other requires raw. A globally-uniform precondition would break one."
        )
        assert "raw" in QC_ACCEPTED_STAGES and "raw" not in PCA_ACCEPTED_STAGES


class TestFlagGating:
    """Disabling the stage contract suppresses its warnings."""

    def test_disabled_suppresses_every_warning(self):
        import os

        os.environ["LOBSTER_STAGE_CONTRACT"] = "0"
        try:
            assert (
                check_input_stage(
                    _adata("raw"), accepted={"filtered_normalized"}, tool_name="run_pca"
                )
                is None
            )
        finally:
            os.environ.pop("LOBSTER_STAGE_CONTRACT", None)

    def test_enabled_by_default(self):
        """Product behaviour, not an instrument: a wrong-stage PCA is a correctness bug."""
        import os

        from lobster.core.provenance.stage_contract import stage_contract_enabled

        os.environ.pop("LOBSTER_STAGE_CONTRACT", None)
        assert stage_contract_enabled() is True
