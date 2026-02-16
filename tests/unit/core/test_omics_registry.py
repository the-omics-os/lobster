"""
Tests for the Modular Omics Plugin Architecture.

Tests cover:
1. OmicsTypeRegistry — built-in types, registration, collision handling
2. DataTypeDetector — metadata detection, data detection, backward compat
3. ComponentRegistry — new entry point groups (adapters, providers, etc.)
4. MetabolomicsAdapter — instantiation, factory functions
"""

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# =============================================================================
# OmicsTypeRegistry Tests
# =============================================================================


class TestOmicsTypeRegistry:
    """Tests for OMICS_TYPE_REGISTRY and registration logic."""

    def test_registry_has_builtin_types(self):
        """All 5 built-in omics types must be registered at module load."""
        from lobster.core.omics_registry import OMICS_TYPE_REGISTRY

        assert "transcriptomics" in OMICS_TYPE_REGISTRY
        assert "proteomics" in OMICS_TYPE_REGISTRY
        assert "genomics" in OMICS_TYPE_REGISTRY
        assert "metabolomics" in OMICS_TYPE_REGISTRY
        assert "metagenomics" in OMICS_TYPE_REGISTRY

    def test_registry_type_configs_have_required_fields(self):
        """Each registered type must have name, display_name, and detection config."""
        from lobster.core.omics_registry import OMICS_TYPE_REGISTRY

        for name, config in OMICS_TYPE_REGISTRY.items():
            assert config.name == name, f"Config name mismatch for {name}"
            assert config.display_name, f"Missing display_name for {name}"
            assert config.detection is not None, f"Missing detection config for {name}"
            assert len(config.detection.keywords) > 0, f"No keywords for {name}"
            assert len(config.preferred_databases) > 0, f"No preferred_databases for {name}"

    def test_proteomics_has_higher_weight_than_transcriptomics(self):
        """Proteomics should have higher detection weight (more specific keywords)."""
        from lobster.core.omics_registry import OMICS_TYPE_REGISTRY

        proteomics_weight = OMICS_TYPE_REGISTRY["proteomics"].detection.weight
        transcriptomics_weight = OMICS_TYPE_REGISTRY["transcriptomics"].detection.weight
        assert proteomics_weight > transcriptomics_weight

    def test_register_duplicate_logs_warning(self):
        """Registering a duplicate type should log warning and be skipped."""
        from lobster.core.omics_registry import (
            OMICS_TYPE_REGISTRY,
            OmicsTypeConfig,
            register_omics_type,
        )

        original_count = len(OMICS_TYPE_REGISTRY)
        # Try to re-register transcriptomics
        with warnings.catch_warnings(record=True):
            register_omics_type(OmicsTypeConfig(
                name="transcriptomics",
                display_name="Duplicate",
            ))
        # Count should not change
        assert len(OMICS_TYPE_REGISTRY) == original_count
        # Original display_name preserved
        assert OMICS_TYPE_REGISTRY["transcriptomics"].display_name == "Transcriptomics"

    def test_metabolomics_preferred_databases(self):
        """Metabolomics should prefer MetaboLights and Metabolomics Workbench."""
        from lobster.core.omics_registry import OMICS_TYPE_REGISTRY

        metab = OMICS_TYPE_REGISTRY["metabolomics"]
        assert "metabolights" in metab.preferred_databases
        assert "metabolomics_workbench" in metab.preferred_databases

    def test_all_types_have_feature_count_ranges(self):
        """Each type must have a non-degenerate feature count range."""
        from lobster.core.omics_registry import OMICS_TYPE_REGISTRY

        for name, config in OMICS_TYPE_REGISTRY.items():
            min_f, max_f = config.detection.feature_count_range
            assert max_f > min_f, f"Degenerate range for {name}: ({min_f}, {max_f})"


# =============================================================================
# DataTypeDetector Tests
# =============================================================================


class TestDataTypeDetector:
    """Tests for unified DataTypeDetector."""

    def test_detect_proteomics_from_metadata(self):
        """Proteomics metadata should rank proteomics as top result."""
        from lobster.core.omics_registry import DataTypeDetector

        detector = DataTypeDetector()
        metadata = {
            "title": "TMT-based proteomics of glioblastoma",
            "summary": "LC-MS/MS DIA analysis of tumor samples",
        }
        results = detector.detect_from_metadata(metadata)
        assert results[0][0] == "proteomics"
        assert results[0][1] > 0.01

    def test_detect_transcriptomics_from_metadata(self):
        """Single-cell metadata should rank transcriptomics as top result."""
        from lobster.core.omics_registry import DataTypeDetector

        detector = DataTypeDetector()
        metadata = {
            "title": "scRNA-seq of human brain development",
            "summary": "10x Chromium single cell RNA sequencing",
        }
        results = detector.detect_from_metadata(metadata)
        assert results[0][0] == "transcriptomics"

    def test_detect_metabolomics_from_metadata(self):
        """Metabolomics metadata should rank metabolomics as top result."""
        from lobster.core.omics_registry import DataTypeDetector

        detector = DataTypeDetector()
        metadata = {
            "title": "Untargeted metabolomics of human plasma",
            "summary": "LC-MS lipidomics profiling",
        }
        results = detector.detect_from_metadata(metadata)
        assert results[0][0] == "metabolomics"

    def test_detect_genomics_from_metadata(self):
        """Genomics/GWAS metadata should rank genomics as top result."""
        from lobster.core.omics_registry import DataTypeDetector

        detector = DataTypeDetector()
        metadata = {
            "title": "Whole genome sequencing of breast cancer",
            "summary": "GWAS variant calling with VCF output",
        }
        results = detector.detect_from_metadata(metadata)
        assert results[0][0] == "genomics"

    def test_detect_metagenomics_from_metadata(self):
        """16S amplicon metadata should rank metagenomics as top result."""
        from lobster.core.omics_registry import DataTypeDetector

        detector = DataTypeDetector()
        metadata = {
            "title": "16S rRNA gut microbiome profiling",
            "summary": "Amplicon sequencing with QIIME2 DADA2 pipeline",
        }
        results = detector.detect_from_metadata(metadata)
        assert results[0][0] == "metagenomics"

    def test_detect_from_data_high_feature_count(self):
        """AnnData with 25,000 features should suggest transcriptomics."""
        from lobster.core.omics_registry import DataTypeDetector

        detector = DataTypeDetector()
        adata = MagicMock()
        adata.n_vars = 25000
        adata.obs = MagicMock()
        adata.obs.keys.return_value = []
        adata.var = MagicMock()
        adata.var.keys.return_value = []

        results = detector.detect_from_data(adata)
        # Transcriptomics has range 5000-60000, should score well
        assert any(r[0] == "transcriptomics" for r in results[:2])

    def test_detect_from_data_low_feature_count(self):
        """AnnData with 500 features should suggest proteomics."""
        from lobster.core.omics_registry import DataTypeDetector

        detector = DataTypeDetector()
        adata = MagicMock()
        adata.n_vars = 500
        adata.obs = MagicMock()
        adata.obs.keys.return_value = []
        adata.var = MagicMock()
        adata.var.keys.return_value = []

        results = detector.detect_from_data(adata)
        # Proteomics has range 100-12000, metabolomics 50-5000 — both should score
        top_types = [r[0] for r in results[:3]]
        assert "proteomics" in top_types or "metabolomics" in top_types

    def test_detect_combined_metadata_and_data(self):
        """Combined detection should merge metadata and data signals."""
        from lobster.core.omics_registry import DataTypeDetector

        detector = DataTypeDetector()
        metadata = {"title": "Mass spectrometry proteomics"}
        adata = MagicMock()
        adata.n_vars = 800
        adata.obs = MagicMock()
        adata.obs.keys.return_value = []
        adata.var = MagicMock()
        adata.var.keys.return_value = []

        results = detector.detect(metadata=metadata, adata=adata)
        assert results[0][0] == "proteomics"

    def test_detect_empty_input_returns_empty(self):
        """No input should return empty list."""
        from lobster.core.omics_registry import DataTypeDetector

        detector = DataTypeDetector()
        assert detector.detect() == []

    def test_results_are_sorted_by_confidence(self):
        """Results must be sorted descending by confidence score."""
        from lobster.core.omics_registry import DataTypeDetector

        detector = DataTypeDetector()
        results = detector.detect_from_metadata({"title": "proteomics mass spec"})
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


class TestBackwardCompatibility:
    """Tests that old detection functions still work via thin wrappers."""

    def test_is_single_cell_returns_true(self):
        """_is_single_cell_dataset() must still return True for SC metadata."""
        from lobster.services.data_access.geo_queue_preparer import (
            _is_single_cell_dataset,
        )

        metadata = {
            "title": "scRNA-seq of human brain",
            "summary": "10x Chromium single cell profiling",
        }
        assert _is_single_cell_dataset(metadata) is True

    def test_is_single_cell_returns_false(self):
        """_is_single_cell_dataset() must return False for bulk RNA-seq."""
        from lobster.services.data_access.geo_queue_preparer import (
            _is_single_cell_dataset,
        )

        metadata = {
            "title": "Bulk RNA-seq of liver tissue",
            "summary": "Total RNA extraction from tissue samples",
        }
        assert _is_single_cell_dataset(metadata) is False

    def test_is_proteomics_returns_true(self):
        """_is_proteomics_dataset() must return True for proteomics metadata."""
        from lobster.services.data_access.geo_queue_preparer import (
            _is_proteomics_dataset,
        )

        metadata = {
            "title": "TMT proteomics of cancer",
            "summary": "Orbitrap mass spectrometry analysis",
        }
        assert _is_proteomics_dataset(metadata) is True

    def test_is_proteomics_returns_false(self):
        """_is_proteomics_dataset() must return False for RNA-seq."""
        from lobster.services.data_access.geo_queue_preparer import (
            _is_proteomics_dataset,
        )

        metadata = {
            "title": "RNA-seq of mouse brain",
            "summary": "Illumina HiSeq paired-end sequencing",
        }
        assert _is_proteomics_dataset(metadata) is False


# =============================================================================
# ComponentRegistry New Groups Tests
# =============================================================================


class TestComponentRegistryNewGroups:
    """Tests that ComponentRegistry has the new entry point group storage."""

    def test_registry_has_adapter_dict(self):
        """ComponentRegistry must have _adapters dict."""
        from lobster.core.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        assert hasattr(registry, "_adapters")
        assert isinstance(registry._adapters, dict)

    def test_registry_has_provider_dict(self):
        """ComponentRegistry must have _providers dict."""
        from lobster.core.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        assert hasattr(registry, "_providers")
        assert isinstance(registry._providers, dict)

    def test_registry_has_download_services_dict(self):
        """ComponentRegistry must have _download_services dict."""
        from lobster.core.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        assert hasattr(registry, "_download_services")
        assert isinstance(registry._download_services, dict)

    def test_registry_has_queue_preparers_dict(self):
        """ComponentRegistry must have _queue_preparers dict."""
        from lobster.core.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        assert hasattr(registry, "_queue_preparers")
        assert isinstance(registry._queue_preparers, dict)

    def test_registry_get_adapter_returns_none_for_missing(self):
        """get_adapter() should return None for missing adapters."""
        from lobster.core.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        registry._loaded = True  # Skip loading
        assert registry.get_adapter("nonexistent") is None

    def test_registry_get_adapter_raises_when_required(self):
        """get_adapter(required=True) should raise for missing adapters."""
        from lobster.core.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        registry._loaded = True
        with pytest.raises(ValueError, match="Required adapter"):
            registry.get_adapter("nonexistent", required=True)

    def test_registry_reset_clears_new_groups(self):
        """reset() must clear all new group dicts."""
        from lobster.core.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        registry._adapters["test"] = "value"
        registry._providers["test"] = "value"
        registry._download_services["test"] = "value"
        registry._queue_preparers["test"] = "value"
        registry.reset()
        assert len(registry._adapters) == 0
        assert len(registry._providers) == 0
        assert len(registry._download_services) == 0
        assert len(registry._queue_preparers) == 0

    def test_registry_get_info_includes_new_groups(self):
        """get_info() must include counts for new groups."""
        from lobster.core.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        registry._loaded = True
        info = registry.get_info()
        assert "adapters" in info
        assert "providers" in info
        assert "download_services" in info
        assert "queue_preparers" in info


# =============================================================================
# MetabolomicsAdapter Tests
# =============================================================================


class TestMetabolomicsAdapter:
    """Tests for the MetabolomicsAdapter and factory functions."""

    def test_adapter_instantiation_lc_ms(self):
        """MetabolomicsAdapter should instantiate with lc_ms data type."""
        from lobster.core.adapters.metabolomics_adapter import MetabolomicsAdapter

        adapter = MetabolomicsAdapter(data_type="lc_ms")
        assert adapter.data_type == "lc_ms"
        assert adapter.name == "MetabolomicsAdapter"

    def test_adapter_instantiation_gc_ms(self):
        """MetabolomicsAdapter should instantiate with gc_ms data type."""
        from lobster.core.adapters.metabolomics_adapter import MetabolomicsAdapter

        adapter = MetabolomicsAdapter(data_type="gc_ms")
        assert adapter.data_type == "gc_ms"

    def test_adapter_instantiation_nmr(self):
        """MetabolomicsAdapter should instantiate with nmr data type."""
        from lobster.core.adapters.metabolomics_adapter import MetabolomicsAdapter

        adapter = MetabolomicsAdapter(data_type="nmr")
        assert adapter.data_type == "nmr"

    def test_adapter_invalid_data_type(self):
        """MetabolomicsAdapter should raise ValueError for invalid data types."""
        from lobster.core.adapters.metabolomics_adapter import MetabolomicsAdapter

        with pytest.raises(ValueError, match="Unknown data_type"):
            MetabolomicsAdapter(data_type="invalid")

    def test_factory_create_lc_ms_adapter(self):
        """Factory function should return configured LC-MS adapter."""
        from lobster.core.adapters.metabolomics_adapter import create_lc_ms_adapter

        adapter = create_lc_ms_adapter()
        assert adapter.data_type == "lc_ms"

    def test_factory_create_gc_ms_adapter(self):
        """Factory function should return configured GC-MS adapter."""
        from lobster.core.adapters.metabolomics_adapter import create_gc_ms_adapter

        adapter = create_gc_ms_adapter()
        assert adapter.data_type == "gc_ms"

    def test_factory_create_nmr_adapter(self):
        """Factory function should return configured NMR adapter."""
        from lobster.core.adapters.metabolomics_adapter import create_nmr_adapter

        adapter = create_nmr_adapter()
        assert adapter.data_type == "nmr"

    def test_adapter_supported_formats(self):
        """MetabolomicsAdapter should support expected formats."""
        from lobster.core.adapters.metabolomics_adapter import MetabolomicsAdapter

        adapter = MetabolomicsAdapter(data_type="lc_ms")
        formats = adapter.get_supported_formats()
        assert "csv" in formats
        assert "tsv" in formats
        assert "h5ad" in formats

    def test_adapter_get_schema(self):
        """get_schema() should return a dict with modality key."""
        from lobster.core.adapters.metabolomics_adapter import MetabolomicsAdapter

        adapter = MetabolomicsAdapter(data_type="lc_ms")
        schema = adapter.get_schema()
        assert isinstance(schema, dict)
        assert "modality" in schema
        assert schema["modality"] == "metabolomics"
