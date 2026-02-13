"""
Smoke tests for annotation templates module.
"""

import pytest


def test_annotation_templates_import():
    """Verify annotation templates module is importable."""
    from lobster.services.templates.annotation_templates import (
        AnnotationTemplateService,
    )

    assert AnnotationTemplateService is not None


def test_annotation_template_service_instantiation():
    """Verify AnnotationTemplateService can be instantiated."""
    from lobster.services.templates.annotation_templates import (
        AnnotationTemplateService,
    )

    service = AnnotationTemplateService()
    assert service is not None

    # Should have tissue types
    tissue_types = service.get_all_tissue_types()
    assert len(tissue_types) > 0


def test_tissue_type_enum():
    """Verify TissueType enum is available."""
    from lobster.services.templates.annotation_templates import TissueType

    assert hasattr(TissueType, "PBMC")
    assert hasattr(TissueType, "BRAIN")
    assert hasattr(TissueType, "LUNG")
