"""
Integration tests for multi-modal GEO dataset support.

Tests the ability to load CITE-seq and other multi-modal datasets
by automatically detecting sample types and filtering to RNA samples.
"""

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.data_access.geo_service import GEOService


@pytest.fixture
def geo_service():
    """Create a GEOService instance with test data manager."""
    data_manager = DataManagerV2()
    return GEOService(data_manager=data_manager)


def test_detect_sample_types_cite_seq(geo_service):
    """Test sample type detection for CITE-seq dataset."""
    # Mock metadata from a CITE-seq dataset (similar to GSE157007)
    metadata = {
        "geo_accession": "GSE999999",  # Mock ID
        "title": "CITE-seq study of T cells",
        "samples": {
            "GSM1": {
                "characteristics_ch1": ["cell type: CD4 T cell", "assay: RNA"],
                "title": "Sample1_RNA",
                "library_strategy": "RNA-Seq",
            },
            "GSM2": {
                "characteristics_ch1": ["cell type: CD4 T cell", "assay: RNA"],
                "title": "Sample2_RNA",
                "library_strategy": "RNA-Seq",
            },
            "GSM3": {
                "characteristics_ch1": ["cell type: CD4 T cell", "assay: protein"],
                "title": "Sample1_protein",
                "library_strategy": "",
            },
            "GSM4": {
                "characteristics_ch1": ["cell type: CD4 T cell", "assay: VDJ"],
                "title": "Sample1_VDJ",
                "library_strategy": "",
            },
        },
    }

    # Call sample type detection
    sample_types = geo_service._detect_sample_types(metadata)

    # Assertions
    assert "rna" in sample_types, "Should detect RNA samples"
    assert "protein" in sample_types, "Should detect protein samples"
    assert "vdj" in sample_types, "Should detect VDJ samples"

    assert len(sample_types["rna"]) == 2, "Should find 2 RNA samples"
    assert len(sample_types["protein"]) == 1, "Should find 1 protein sample"
    assert len(sample_types["vdj"]) == 1, "Should find 1 VDJ sample"

    assert "GSM1" in sample_types["rna"]
    assert "GSM2" in sample_types["rna"]
    assert "GSM3" in sample_types["protein"]
    assert "GSM4" in sample_types["vdj"]


def test_detect_sample_types_rna_only(geo_service):
    """Test sample type detection for RNA-only dataset."""
    metadata = {
        "geo_accession": "GSE888888",
        "title": "Single-cell RNA-seq",
        "samples": {
            "GSM1": {
                "characteristics_ch1": [
                    "cell type: CD4 T cell",
                    "library type: gene expression",
                ],
                "title": "Sample1",
                "library_strategy": "RNA-Seq",
            },
            "GSM2": {
                "characteristics_ch1": [
                    "cell type: CD8 T cell",
                    "library type: gene expression",
                ],
                "title": "Sample2",
                "library_strategy": "RNA-Seq",
            },
        },
    }

    sample_types = geo_service._detect_sample_types(metadata)

    assert "rna" in sample_types
    assert len(sample_types["rna"]) == 2
    assert "protein" not in sample_types
    assert "vdj" not in sample_types


def test_detect_sample_types_with_library_strategy(geo_service):
    """Test that library_strategy field is prioritized."""
    metadata = {
        "geo_accession": "GSE777777",
        "samples": {
            "GSM1": {
                "characteristics_ch1": [],  # Empty characteristics
                "title": "Sample1",
                "library_strategy": "RNA-Seq",  # Should use this
            },
            "GSM2": {
                "characteristics_ch1": [],
                "title": "Sample2",
                "library_strategy": "ATAC-seq",  # Should detect ATAC
            },
        },
    }

    sample_types = geo_service._detect_sample_types(metadata)

    assert "rna" in sample_types
    assert "atac" in sample_types
    assert len(sample_types["rna"]) == 1
    assert len(sample_types["atac"]) == 1


def test_detect_sample_types_title_fallback(geo_service):
    """Test fallback to sample title for type detection."""
    metadata = {
        "geo_accession": "GSE666666",
        "samples": {
            "GSM1": {
                "characteristics_ch1": [],
                "title": "CD4_RNA_sample",
                "library_strategy": "",
            },
            "GSM2": {
                "characteristics_ch1": [],
                "title": "CD4_protein_sample",
                "library_strategy": "",
            },
            "GSM3": {
                "characteristics_ch1": [],
                "title": "CD4_vdj_sample",
                "library_strategy": "",
            },
        },
    }

    sample_types = geo_service._detect_sample_types(metadata)

    assert "rna" in sample_types
    assert "protein" in sample_types
    assert "vdj" in sample_types
    assert len(sample_types["rna"]) == 1
    assert len(sample_types["protein"]) == 1
    assert len(sample_types["vdj"]) == 1


def test_detect_sample_types_unknown_defaults_to_rna(geo_service):
    """Test that unknown samples default to RNA for backward compatibility."""
    metadata = {
        "geo_accession": "GSE555555",
        "samples": {
            "GSM1": {
                "characteristics_ch1": ["cell type: unknown"],
                "title": "Mystery_sample",
                "library_strategy": "",
            },
        },
    }

    sample_types = geo_service._detect_sample_types(metadata)

    # Unknown samples should default to RNA
    assert "rna" in sample_types
    assert len(sample_types["rna"]) == 1
    assert "GSM1" in sample_types["rna"]


@pytest.mark.skip(reason="Requires actual GEO download - slow test")
def test_load_cite_seq_dataset_gse157007():
    """
    Integration test with real GSE157007 CITE-seq dataset.

    This test is skipped by default because it requires downloading from GEO.
    Run with: pytest -v -k test_load_cite_seq_dataset_gse157007 --run-slow
    """
    data_manager = DataManagerV2()
    geo_service = GEOService(data_manager=data_manager)

    # This should now work (previously would raise FeatureNotImplementedError)
    result = geo_service.download_dataset("GSE157007")

    # Verify success
    assert "Successfully downloaded" in result
    assert "Multi-Modal Dataset Detected" in result
    assert "RNA" in result
    assert "Skipped" in result

    # Verify data was loaded
    modalities = data_manager.list_modalities()
    assert any("gse157007" in m for m in modalities)

    # Verify provenance tracking
    tool_history = data_manager.provenance.get_tool_history()
    assert any("is_multimodal" in entry.get("parameters", {}) for entry in tool_history)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
