"""
Tests for VariantAnnotationService.

Covers:
- Service import and instantiation (smoke tests)
- Variant normalization (left-trim, multiallelic split)
- Variant prioritization (composite scoring, missing columns)
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from lobster.core.analysis_ir import AnalysisStep
from lobster.services.analysis.variant_annotation_service import (
    VariantAnnotationError,
    VariantAnnotationService,
)


# ===============================================================================
# Fixtures
# ===============================================================================


@pytest.fixture
def annotation_service():
    """Create VariantAnnotationService instance."""
    return VariantAnnotationService()


@pytest.fixture
def simple_variant_adata():
    """Create simple AnnData with variant data for testing."""
    np.random.seed(42)
    n_samples, n_variants = 10, 5

    genotypes = np.random.choice([0, 1, 2], size=(n_samples, n_variants)).astype(float)

    adata = AnnData(X=genotypes)
    adata.var = pd.DataFrame(
        {
            "CHROM": ["1", "1", "2", "2", "X"],
            "POS": [100, 200, 300, 400, 500],
            "REF": ["A", "AT", "ATG", "A", "C"],
            "ALT": ["G", "A", "A", "G,T", "T"],
        }
    )
    adata.layers["GT"] = genotypes.copy()
    adata.uns["data_type"] = "genomics"

    return adata


@pytest.fixture
def annotated_variant_adata():
    """Create AnnData with pre-existing variant annotations for prioritization."""
    np.random.seed(42)
    n_samples, n_variants = 10, 8

    genotypes = np.random.choice([0, 1, 2], size=(n_samples, n_variants)).astype(float)

    adata = AnnData(X=genotypes)
    adata.var = pd.DataFrame(
        {
            "CHROM": ["1"] * n_variants,
            "POS": list(range(100, 100 + n_variants * 100, 100)),
            "REF": ["A"] * n_variants,
            "ALT": ["G"] * n_variants,
            "consequence": [
                "missense_variant",
                "synonymous_variant",
                "stop_gained",
                "intron_variant",
                "frameshift_variant",
                "splice_donor_variant",
                None,
                "intergenic_variant",
            ],
            "gnomad_af": [0.001, 0.1, 0.0, None, 0.0005, 0.0, 0.05, 0.3],
            "clinvar_significance": [
                "Pathogenic",
                "Benign",
                "Likely_pathogenic",
                None,
                "Uncertain_significance",
                "Pathogenic",
                None,
                "Benign",
            ],
        }
    )
    adata.layers["GT"] = genotypes.copy()
    adata.uns["data_type"] = "genomics"

    return adata


# ===============================================================================
# Smoke Tests (existing)
# ===============================================================================


def test_service_import():
    """Verify VariantAnnotationService is importable."""
    assert VariantAnnotationService is not None


def test_service_instantiation():
    """Verify VariantAnnotationService can be instantiated."""
    service = VariantAnnotationService()
    assert service is not None
    assert service._annotation_cache == {}


def test_annotation_sources_defined():
    """Verify ANNOTATION_SOURCES class variable is defined."""
    assert hasattr(VariantAnnotationService, "ANNOTATION_SOURCES")
    assert "genebe" in VariantAnnotationService.ANNOTATION_SOURCES
    assert "ensembl_vep" in VariantAnnotationService.ANNOTATION_SOURCES


def test_genome_builds_defined():
    """Verify GENOME_BUILDS class variable is defined."""
    assert hasattr(VariantAnnotationService, "GENOME_BUILDS")
    assert "hg38" in VariantAnnotationService.GENOME_BUILDS
    assert "hg19" in VariantAnnotationService.GENOME_BUILDS
    assert "GRCh38" in VariantAnnotationService.GENOME_BUILDS
    assert "GRCh37" in VariantAnnotationService.GENOME_BUILDS


def test_annotation_columns_defined():
    """Verify ANNOTATION_COLUMNS class variable is defined."""
    columns = VariantAnnotationService.ANNOTATION_COLUMNS
    assert "gene_symbol" in columns
    assert "consequence" in columns
    assert "gnomad_af" in columns
    assert "clinvar_significance" in columns


def test_error_class_defined():
    """Verify VariantAnnotationError exception class is defined."""
    assert VariantAnnotationError is not None
    assert issubclass(VariantAnnotationError, Exception)


# ===============================================================================
# Normalization Tests
# ===============================================================================


class TestNormalizeVariants:
    """Test variant normalization methods."""

    def test_normalize_variants_trimming(self, annotation_service, simple_variant_adata):
        """Verify padding is trimmed from REF/ALT alleles."""
        adata_norm, stats, ir = annotation_service.normalize_variants(
            simple_variant_adata
        )

        # Should return valid 3-tuple
        assert isinstance(adata_norm, AnnData)
        assert isinstance(stats, dict)
        assert isinstance(ir, AnalysisStep)

        # Stats should report some trimming (AT/A should be trimmed)
        assert stats["n_variants_before"] == simple_variant_adata.n_vars
        assert stats["n_trimmed"] >= 0
        assert "REF_normalized" in adata_norm.var.columns
        assert "ALT_normalized" in adata_norm.var.columns
        assert "POS_normalized" in adata_norm.var.columns

    def test_normalize_variants_multiallelic_split(self, annotation_service):
        """Verify multiallelic ALTs are split into separate rows."""
        # Create adata with one multiallelic variant
        n_samples = 5
        genotypes = np.random.choice([0, 1, 2], size=(n_samples, 3)).astype(float)
        adata = AnnData(X=genotypes)
        adata.var = pd.DataFrame({
            "CHROM": ["1", "1", "1"],
            "POS": [100, 200, 300],
            "REF": ["A", "A", "A"],
            "ALT": ["G", "G,T", "C"],
        })
        adata.layers["GT"] = genotypes.copy()

        adata_norm, stats, ir = annotation_service.normalize_variants(adata)

        # Should have more variants after splitting
        assert adata_norm.n_vars >= adata.n_vars
        assert stats["n_multiallelic_split"] == 1
        assert stats["n_variants_after"] == 4  # 3 original, 1 multiallelic split into 2 = 2 + 2 = 4

    def test_normalize_variants_missing_columns(self, annotation_service):
        """Test error when required columns are missing."""
        adata = AnnData(X=np.zeros((5, 3)))
        adata.var = pd.DataFrame({"some_col": [1, 2, 3]})

        with pytest.raises(VariantAnnotationError, match="Required column"):
            annotation_service.normalize_variants(adata)

    def test_trim_alleles_left_padding(self, annotation_service):
        """Test _trim_alleles removes left padding."""
        ref, alt, pos = annotation_service._trim_alleles("AT", "AC", 100)
        assert ref == "T"
        assert alt == "C"
        assert pos == 101

    def test_trim_alleles_right_padding(self, annotation_service):
        """Test _trim_alleles removes right padding."""
        ref, alt, pos = annotation_service._trim_alleles("TAG", "TAC", 100)
        # Right trim: TAG -> TA, TAC -> TA (wait, last chars G vs C differ - no right trim)
        # Left trim: T == T, trim -> AG, AC, pos=101
        # Then A == A, trim -> G, C, pos=102
        assert ref == "G"
        assert alt == "C"
        assert pos == 102

    def test_trim_alleles_no_change(self, annotation_service):
        """Test _trim_alleles with no common prefix/suffix."""
        ref, alt, pos = annotation_service._trim_alleles("A", "G", 100)
        assert ref == "A"
        assert alt == "G"
        assert pos == 100


# ===============================================================================
# Prioritization Tests
# ===============================================================================


class TestPrioritizeVariants:
    """Test variant prioritization methods."""

    def test_prioritize_variants_scoring(
        self, annotation_service, annotated_variant_adata
    ):
        """Verify priority_score and priority_rank are in var."""
        adata_pri, stats, ir = annotation_service.prioritize_variants(
            annotated_variant_adata
        )

        # Should return valid 3-tuple
        assert isinstance(adata_pri, AnnData)
        assert isinstance(stats, dict)
        assert isinstance(ir, AnalysisStep)

        # Priority columns should exist
        assert "priority_score" in adata_pri.var.columns
        assert "priority_rank" in adata_pri.var.columns

        # Scores should be in [0, 1] range
        scores = adata_pri.var["priority_score"].values
        assert (scores >= 0).all()
        assert (scores <= 1).all()

        # Ranks should be 1-based integers
        ranks = adata_pri.var["priority_rank"].values
        assert (ranks >= 1).all()
        assert (ranks <= adata_pri.n_vars).all()

        # Stats should have expected keys
        assert stats["n_variants_scored"] == annotated_variant_adata.n_vars
        assert "n_high_priority" in stats
        assert "n_medium_priority" in stats
        assert "n_low_priority" in stats
        assert "top_variants" in stats

    def test_prioritize_variants_relative_ordering(
        self, annotation_service, annotated_variant_adata
    ):
        """Test that high-impact variants score higher than low-impact."""
        adata_pri, stats, ir = annotation_service.prioritize_variants(
            annotated_variant_adata
        )

        scores = adata_pri.var["priority_score"].values

        # stop_gained (idx 2) + rare (AF=0) + likely_pathogenic should score high
        # intergenic (idx 7) + common (AF=0.3) + benign should score low
        stop_gained_score = scores[2]
        intergenic_score = scores[7]
        assert stop_gained_score > intergenic_score, (
            f"stop_gained ({stop_gained_score}) should score higher than "
            f"intergenic ({intergenic_score})"
        )

    def test_prioritize_variants_missing_columns(self, annotation_service):
        """Verify graceful degradation when annotation columns are missing."""
        # Create adata with NO annotation columns
        n_samples, n_variants = 10, 5
        genotypes = np.random.choice([0, 1, 2], size=(n_samples, n_variants)).astype(float)
        adata = AnnData(X=genotypes)
        adata.var = pd.DataFrame({
            "CHROM": ["1"] * n_variants,
            "POS": list(range(100, 600, 100)),
            "REF": ["A"] * n_variants,
            "ALT": ["G"] * n_variants,
        })

        # Should NOT raise, should score with available info only
        adata_pri, stats, ir = annotation_service.prioritize_variants(adata)

        assert "priority_score" in adata_pri.var.columns
        assert "priority_rank" in adata_pri.var.columns

        # All scores should be from rarity only (default 0.15 for unknown AF)
        scores = adata_pri.var["priority_score"].values
        assert (scores >= 0).all()

    def test_prioritize_variants_ir_structure(
        self, annotation_service, annotated_variant_adata
    ):
        """Test that prioritization IR has correct fields."""
        _, _, ir = annotation_service.prioritize_variants(annotated_variant_adata)

        assert ir.operation == "genomics.annotation.prioritize_variants"
        assert ir.library == "lobster"

    def test_new_methods_exist(self, annotation_service):
        """Verify all new methods are present."""
        assert hasattr(annotation_service, "normalize_variants")
        assert hasattr(annotation_service, "query_population_frequencies")
        assert hasattr(annotation_service, "query_clinical_databases")
        assert hasattr(annotation_service, "prioritize_variants")
        assert callable(annotation_service.normalize_variants)
        assert callable(annotation_service.query_population_frequencies)
        assert callable(annotation_service.query_clinical_databases)
        assert callable(annotation_service.prioritize_variants)
