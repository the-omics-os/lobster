"""Tests for expression file scoring heuristic in geo_service.

Verifies that _score_expression_file correctly distinguishes expression data
files from metadata/feature list files, handling the "gene" keyword ambiguity.
"""

import pytest

from lobster.services.data_access.geo_service import _score_expression_file


class TestExpressionSignals:
    """Tests that expression-related keywords produce positive scores."""

    def test_gene_expression_matrix_scores_positive(self):
        score = _score_expression_file("gene_expression_matrix.txt.gz")
        assert score > 0, f"Expected positive score, got {score}"

    def test_counts_file_scores_positive(self):
        score = _score_expression_file("GSE12345_counts.txt.gz")
        assert score > 0, f"Expected positive score, got {score}"

    def test_tpm_file_scores_positive(self):
        score = _score_expression_file("GSE12345_tpm_values.tsv.gz")
        assert score > 0, f"Expected positive score, got {score}"

    def test_fpkm_file_scores_positive(self):
        score = _score_expression_file("GSE12345_fpkm.csv.gz")
        assert score > 0, f"Expected positive score, got {score}"

    def test_rpkm_file_scores_positive(self):
        score = _score_expression_file("rpkm_matrix.txt.gz")
        assert score > 0, f"Expected positive score, got {score}"

    def test_normalized_expression_scores_positive(self):
        score = _score_expression_file("normalized_expression.h5ad")
        assert score > 0, f"Expected positive score, got {score}"


class TestMetadataSignals:
    """Tests that metadata-related keywords produce negative scores."""

    def test_barcodes_scores_negative(self):
        score = _score_expression_file("barcodes.tsv.gz")
        assert score < 0, f"Expected negative score, got {score}"

    def test_annotation_scores_negative(self):
        score = _score_expression_file("gene_annotation.csv")
        assert score < 0, f"Expected negative score, got {score}"

    def test_metadata_scores_negative(self):
        score = _score_expression_file("metadata.csv.gz")
        assert score < 0, f"Expected negative score, got {score}"

    def test_sample_info_scores_negative(self):
        score = _score_expression_file("sample_info.txt.gz")
        assert score < 0, f"Expected negative score, got {score}"

    def test_clinical_scores_negative(self):
        score = _score_expression_file("clinical_data.csv")
        assert score < 0, f"Expected negative score, got {score}"


class TestGeneAmbiguity:
    """Tests that the ambiguous 'gene' keyword is handled contextually."""

    def test_genes_tsv_scores_negative(self):
        """genes.tsv.gz is a 10X feature list, not expression data."""
        score = _score_expression_file("genes.tsv.gz")
        assert score < 0, f"Expected negative score for feature list, got {score}"

    def test_gene_expression_scores_positive(self):
        """gene_expression_matrix.txt.gz IS expression data."""
        score = _score_expression_file("gene_expression_matrix.txt.gz")
        assert score > 0, f"Expected positive score for expression file, got {score}"

    def test_expression_beats_genes_tsv(self):
        """gene_expression_matrix should rank higher than genes.tsv.gz."""
        expr_score = _score_expression_file("gene_expression_matrix.txt.gz")
        genes_score = _score_expression_file("genes.tsv.gz")
        assert expr_score > genes_score, (
            f"Expression file ({expr_score}) should score higher than feature list ({genes_score})"
        )


class TestFormatBonuses:
    """Tests that structured formats receive scoring bonuses."""

    def test_h5ad_gets_format_bonus(self):
        h5ad_score = _score_expression_file("normalized_expression.h5ad")
        txt_score = _score_expression_file("normalized_expression.txt.gz")
        assert h5ad_score > txt_score, (
            f"h5ad ({h5ad_score}) should score higher than txt ({txt_score})"
        )

    def test_h5_gets_format_bonus(self):
        h5_score = _score_expression_file("expression_data.h5")
        txt_score = _score_expression_file("expression_data.txt.gz")
        assert h5_score > txt_score, (
            f"h5 ({h5_score}) should score higher than txt ({txt_score})"
        )

    def test_h5ad_higher_than_h5(self):
        h5ad_score = _score_expression_file("expression.h5ad")
        h5_score = _score_expression_file("expression.h5")
        assert h5ad_score > h5_score, (
            f"h5ad ({h5ad_score}) should score higher than h5 ({h5_score})"
        )


class TestScoringIntegration:
    """Integration tests for file selection using scoring."""

    def test_expression_matrix_selected_over_genes_tsv(self):
        """Simulates _process_supplementary_files file selection logic."""
        files = [
            "gene_expression_matrix.txt.gz",
            "genes.tsv.gz",
            "barcodes.tsv.gz",
        ]
        scored = [(f, _score_expression_file(f)) for f in files]
        selected = [f for f, score in sorted(scored, key=lambda x: -x[1]) if score > 0]
        assert len(selected) >= 1
        assert selected[0] == "gene_expression_matrix.txt.gz"

    def test_h5ad_preferred_over_text(self):
        """h5ad should rank above equivalent text files."""
        files = [
            "expression_data.txt.gz",
            "expression_data.h5ad",
        ]
        scored = [(f, _score_expression_file(f)) for f in files]
        selected = sorted(scored, key=lambda x: -x[1])
        assert selected[0][0] == "expression_data.h5ad"

    def test_metadata_files_excluded(self):
        """Metadata files should not appear in positive-score results."""
        files = [
            "metadata.csv.gz",
            "barcodes.tsv.gz",
            "sample_info.txt.gz",
        ]
        scored = [(f, _score_expression_file(f)) for f in files]
        positive = [f for f, score in scored if score > 0]
        assert len(positive) == 0, f"Metadata files should all score negative: {scored}"
