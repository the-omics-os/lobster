#!/usr/bin/env python3
"""Update imports after service reorganization."""

import re
from pathlib import Path

# Mapping: old_import → new_import
IMPORT_MAPPINGS = {
    # Analysis services
    "from lobster.tools.clustering_service": "from lobster.services.analysis.clustering_service",
    "from lobster.tools.enhanced_singlecell_service": "from lobster.services.analysis.enhanced_singlecell_service",
    "from lobster.tools.bulk_rnaseq_service": "from lobster.services.analysis.bulk_rnaseq_service",
    "from lobster.tools.differential_formula_service": "from lobster.services.analysis.differential_formula_service",
    "from lobster.tools.pseudobulk_service": "from lobster.services.analysis.pseudobulk_service",
    "from lobster.tools.scvi_embedding_service": "from lobster.services.analysis.scvi_embedding_service",
    "from lobster.tools.proteomics_analysis_service": "from lobster.services.analysis.proteomics_analysis_service",
    "from lobster.tools.proteomics_differential_service": "from lobster.services.analysis.proteomics_differential_service",
    "from lobster.tools.structure_analysis_service": "from lobster.services.analysis.structure_analysis_service",

    # Quality services
    "from lobster.tools.quality_service": "from lobster.services.quality.quality_service",
    "from lobster.tools.preprocessing_service": "from lobster.services.quality.preprocessing_service",
    "from lobster.tools.proteomics_quality_service": "from lobster.services.quality.proteomics_quality_service",
    "from lobster.tools.proteomics_preprocessing_service": "from lobster.services.quality.proteomics_preprocessing_service",

    # Visualization services
    "from lobster.tools.visualization_service": "from lobster.services.visualization.visualization_service",
    "from lobster.tools.bulk_visualization_service": "from lobster.services.visualization.bulk_visualization_service",
    "from lobster.tools.proteomics_visualization_service": "from lobster.services.visualization.proteomics_visualization_service",
    "from lobster.tools.pymol_visualization_service": "from lobster.services.visualization.pymol_visualization_service",
    "from lobster.tools.chimerax_visualization_service_ALPHA": "from lobster.services.visualization.chimerax_visualization_service_ALPHA",

    # Data access services
    "from lobster.tools.geo_service": "from lobster.services.data_access.geo_service",
    "from lobster.tools.geo_download_service": "from lobster.services.data_access.geo_download_service",
    "from lobster.tools.geo_fallback_service": "from lobster.services.data_access.geo_fallback_service",
    "from lobster.tools.content_access_service": "from lobster.services.data_access.content_access_service",
    "from lobster.tools.workspace_content_service": "from lobster.services.data_access.workspace_content_service",
    "from lobster.tools.protein_structure_fetch_service": "from lobster.services.data_access.protein_structure_fetch_service",
    "from lobster.tools.docling_service": "from lobster.services.data_access.docling_service",

    # Data management services
    "from lobster.tools.modality_management_service": "from lobster.services.data_management.modality_management_service",
    "from lobster.tools.concatenation_service": "from lobster.services.data_management.concatenation_service",

    # Metadata services
    "from lobster.tools.metadata_standardization_service": "from lobster.services.metadata.metadata_standardization_service",
    "from lobster.tools.metadata_validation_service": "from lobster.services.metadata.metadata_validation_service",
    "from lobster.tools.disease_standardization_service": "from lobster.services.metadata.disease_standardization_service",
    "from lobster.tools.sample_mapping_service": "from lobster.services.metadata.sample_mapping_service",
    "from lobster.tools.microbiome_filtering_service": "from lobster.services.metadata.microbiome_filtering_service",
    "from lobster.tools.manual_annotation_service": "from lobster.services.metadata.manual_annotation_service",

    # ML services
    "from lobster.tools.ml_transcriptomics_service_ALPHA": "from lobster.services.ml.ml_transcriptomics_service_ALPHA",
    "from lobster.tools.ml_proteomics_service_ALPHA": "from lobster.services.ml.ml_proteomics_service_ALPHA",

    # Templates
    "from lobster.tools.annotation_templates": "from lobster.services.templates.annotation_templates",
}

def update_file(filepath: Path):
    """Update imports in a single file."""
    try:
        content = filepath.read_text()
        original = content

        for old, new in IMPORT_MAPPINGS.items():
            content = content.replace(old, new)

        if content != original:
            filepath.write_text(content)
            print(f"Updated: {filepath}")
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Update all Python files."""
    root = Path(__file__).parent.parent / "lobster"
    test_root = Path(__file__).parent.parent / "tests"

    updated_count = 0

    # Update lobster package files
    for py_file in root.rglob("*.py"):
        if update_file(py_file):
            updated_count += 1

    # Update test files
    for py_file in test_root.rglob("*.py"):
        if update_file(py_file):
            updated_count += 1

    print(f"\nTotal files updated: {updated_count}")

if __name__ == "__main__":
    main()
