"""
Smoke tests for VariantAnnotationService.

These tests verify that the service can be imported and instantiated.
Full integration tests require genebe or network access to Ensembl VEP.
"""



def test_service_import():
    """Verify VariantAnnotationService is importable."""
    from lobster.services.analysis.variant_annotation_service import (
        VariantAnnotationService,
    )

    assert VariantAnnotationService is not None


def test_service_instantiation():
    """Verify VariantAnnotationService can be instantiated."""
    from lobster.services.analysis.variant_annotation_service import (
        VariantAnnotationService,
    )

    service = VariantAnnotationService()
    assert service is not None
    assert service._annotation_cache == {}


def test_annotation_sources_defined():
    """Verify ANNOTATION_SOURCES class variable is defined."""
    from lobster.services.analysis.variant_annotation_service import (
        VariantAnnotationService,
    )

    assert hasattr(VariantAnnotationService, "ANNOTATION_SOURCES")
    assert "genebe" in VariantAnnotationService.ANNOTATION_SOURCES
    assert "ensembl_vep" in VariantAnnotationService.ANNOTATION_SOURCES


def test_genome_builds_defined():
    """Verify GENOME_BUILDS class variable is defined."""
    from lobster.services.analysis.variant_annotation_service import (
        VariantAnnotationService,
    )

    assert hasattr(VariantAnnotationService, "GENOME_BUILDS")
    assert "hg38" in VariantAnnotationService.GENOME_BUILDS
    assert "hg19" in VariantAnnotationService.GENOME_BUILDS
    assert "GRCh38" in VariantAnnotationService.GENOME_BUILDS
    assert "GRCh37" in VariantAnnotationService.GENOME_BUILDS


def test_annotation_columns_defined():
    """Verify ANNOTATION_COLUMNS class variable is defined."""
    from lobster.services.analysis.variant_annotation_service import (
        VariantAnnotationService,
    )

    assert hasattr(VariantAnnotationService, "ANNOTATION_COLUMNS")
    columns = VariantAnnotationService.ANNOTATION_COLUMNS
    assert "gene_symbol" in columns
    assert "consequence" in columns
    assert "gnomad_af" in columns
    assert "clinvar_significance" in columns


def test_error_class_defined():
    """Verify VariantAnnotationError exception class is defined."""
    from lobster.services.analysis.variant_annotation_service import (
        VariantAnnotationError,
    )

    assert VariantAnnotationError is not None
    # Verify it's an Exception subclass
    assert issubclass(VariantAnnotationError, Exception)
