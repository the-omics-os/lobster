"""
Unit tests for multi-modal sample detection functionality.

Tests comprehensive edge cases for the _detect_sample_types method
and verifies platform compatibility handling for multi-modal datasets.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from lobster.tools.geo_service import GEOService
from lobster.core.data_manager_v2 import DataManagerV2


@pytest.fixture
def geo_service():
    """Create a GEOService instance with mock data manager."""
    data_manager = DataManagerV2()
    return GEOService(data_manager=data_manager)


class TestSampleTypeDetection:
    """Test suite for _detect_sample_types method."""

    def test_library_strategy_takes_precedence(self, geo_service):
        """Verify library_strategy field is checked first."""
        metadata = {
            "samples": {
                "GSM1": {
                    "characteristics_ch1": ["assay: protein"],  # Misleading
                    "title": "Sample_protein",  # Misleading
                    "library_strategy": "RNA-Seq",  # Should win
                },
            }
        }

        result = geo_service._detect_sample_types(metadata)

        assert "rna" in result
        assert len(result["rna"]) == 1
        assert "GSM1" in result["rna"]
        # Should not detect as protein despite characteristics/title
        assert "protein" not in result

    def test_mixed_case_sensitivity(self, geo_service):
        """Test detection works with various case formats."""
        metadata = {
            "samples": {
                "GSM1": {
                    "characteristics_ch1": ["Assay: RNA", "Library Type: Gene Expression"],
                    "title": "Sample1",
                    "library_strategy": "",
                },
                "GSM2": {
                    "characteristics_ch1": ["ASSAY: PROTEIN"],
                    "title": "Sample2",
                    "library_strategy": "",
                },
                "GSM3": {
                    "characteristics_ch1": ["assay: vdj"],
                    "title": "Sample3",
                    "library_strategy": "",
                },
            }
        }

        result = geo_service._detect_sample_types(metadata)

        assert len(result["rna"]) == 1
        assert len(result["protein"]) == 1
        assert len(result["vdj"]) == 1

    def test_characteristics_as_string_not_list(self, geo_service):
        """Handle characteristics_ch1 as string instead of list."""
        metadata = {
            "samples": {
                "GSM1": {
                    "characteristics_ch1": "assay: RNA, cell type: T cell",  # String
                    "title": "Sample1",
                    "library_strategy": "",
                },
            }
        }

        result = geo_service._detect_sample_types(metadata)

        assert "rna" in result
        assert "GSM1" in result["rna"]

    def test_empty_characteristics(self, geo_service):
        """Handle samples with empty characteristics."""
        metadata = {
            "samples": {
                "GSM1": {
                    "characteristics_ch1": [],
                    "title": "Sample_RNA",
                    "library_strategy": "",
                },
                "GSM2": {
                    "characteristics_ch1": None,
                    "title": "Sample_protein",
                    "library_strategy": "",
                },
            }
        }

        result = geo_service._detect_sample_types(metadata)

        # Should fall back to title matching
        assert "rna" in result
        assert "protein" in result

    def test_multiple_patterns_same_sample(self, geo_service):
        """Ensure first match wins when multiple patterns match."""
        metadata = {
            "samples": {
                "GSM1": {
                    "characteristics_ch1": [
                        "assay: RNA",
                        "library type: gene expression",
                        "data type: rna-seq",
                    ],
                    "title": "Sample_GEX",
                    "library_strategy": "",
                },
            }
        }

        result = geo_service._detect_sample_types(metadata)

        # Should detect as RNA (first match)
        assert "rna" in result
        assert len(result["rna"]) == 1

    def test_atac_detection_patterns(self, geo_service):
        """Test all ATAC detection strategies."""
        metadata = {
            "samples": {
                "GSM1": {
                    "characteristics_ch1": ["assay: ATAC"],
                    "title": "Sample1",
                    "library_strategy": "",
                },
                "GSM2": {
                    "characteristics_ch1": ["chromatin accessibility"],
                    "title": "Sample2",
                    "library_strategy": "",
                },
                "GSM3": {
                    "characteristics_ch1": [],
                    "title": "Sample_ATAC",
                    "library_strategy": "",
                },
                "GSM4": {
                    "characteristics_ch1": [],
                    "title": "Sample4",
                    "library_strategy": "ATAC-seq",
                },
            }
        }

        result = geo_service._detect_sample_types(metadata)

        assert "atac" in result
        assert len(result["atac"]) == 4

    def test_vdj_detection_comprehensive(self, geo_service):
        """Test all VDJ/TCR/BCR detection patterns."""
        metadata = {
            "samples": {
                "GSM1": {"characteristics_ch1": ["assay: VDJ"], "title": "", "library_strategy": ""},
                "GSM2": {"characteristics_ch1": ["tcr-seq"], "title": "", "library_strategy": ""},
                "GSM3": {"characteristics_ch1": ["bcr seq"], "title": "", "library_strategy": ""},
                "GSM4": {"characteristics_ch1": ["immune repertoire"], "title": "", "library_strategy": ""},
                "GSM5": {"characteristics_ch1": [], "title": "Sample_TCR", "library_strategy": ""},
                "GSM6": {"characteristics_ch1": [], "title": "Sample_BCR", "library_strategy": ""},
            }
        }

        result = geo_service._detect_sample_types(metadata)

        assert "vdj" in result
        assert len(result["vdj"]) == 6

    def test_protein_detection_cite_seq_variants(self, geo_service):
        """Test CITE-seq protein detection variants."""
        metadata = {
            "samples": {
                "GSM1": {"characteristics_ch1": ["assay: protein"], "title": "", "library_strategy": ""},
                "GSM2": {"characteristics_ch1": ["library type: antibody capture"], "title": "", "library_strategy": ""},
                "GSM3": {"characteristics_ch1": ["antibody-derived tag"], "title": "", "library_strategy": ""},
                "GSM4": {"characteristics_ch1": ["adt"], "title": "", "library_strategy": ""},
                "GSM5": {"characteristics_ch1": ["cite-seq protein"], "title": "", "library_strategy": ""},
                "GSM6": {"characteristics_ch1": [], "title": "Sample_ADT", "library_strategy": ""},
            }
        }

        result = geo_service._detect_sample_types(metadata)

        assert "protein" in result
        assert len(result["protein"]) == 6

    def test_no_samples_in_metadata(self, geo_service):
        """Handle metadata without samples key."""
        metadata = {"geo_accession": "GSE12345"}

        result = geo_service._detect_sample_types(metadata)

        assert result == {}

    def test_empty_samples_dict(self, geo_service):
        """Handle empty samples dictionary."""
        metadata = {"samples": {}}

        result = geo_service._detect_sample_types(metadata)

        assert result == {}

    def test_malformed_sample_metadata(self, geo_service):
        """Handle samples with missing required fields."""
        metadata = {
            "samples": {
                "GSM1": {},  # No fields
                "GSM2": {"title": "Sample2"},  # Missing characteristics
                "GSM3": {"characteristics_ch1": []},  # Missing title
            }
        }

        result = geo_service._detect_sample_types(metadata)

        # Should default all to RNA
        assert "rna" in result
        assert len(result["rna"]) == 3

    def test_rna_detection_gex_patterns(self, geo_service):
        """Test gene expression (GEX) specific patterns."""
        metadata = {
            "samples": {
                "GSM1": {"characteristics_ch1": ["library type: gex"], "title": "", "library_strategy": ""},
                "GSM2": {"characteristics_ch1": ["library type: gene expression"], "title": "", "library_strategy": ""},
                "GSM3": {"characteristics_ch1": [], "title": "Sample_GEX", "library_strategy": ""},
                "GSM4": {"characteristics_ch1": [], "title": "Sample_gene_expression", "library_strategy": ""},
            }
        }

        result = geo_service._detect_sample_types(metadata)

        assert "rna" in result
        assert len(result["rna"]) == 4


class TestPlatformCompatibilityMultiModal:
    """Test platform compatibility handling for multi-modal datasets."""

    def test_cite_seq_with_rna_samples_allows_loading(self, geo_service):
        """Multi-modal CITE-seq with RNA should allow loading."""
        metadata = {
            "title": "CITE-seq study",
            "summary": "Single-cell RNA and protein profiling",
            "platform_id": ["GPL24676"],  # 10X Chromium (series level - list)
            "samples": {
                "GSM1": {"characteristics_ch1": ["assay: RNA"], "title": "S1_RNA", "library_strategy": "RNA-Seq", "platform_id": "GPL24676"},
                "GSM2": {"characteristics_ch1": ["assay: protein"], "title": "S1_protein", "library_strategy": "", "platform_id": "GPL24676"},
            },
        }

        # Mock the data expert assistant (use create=True for lazy initialization)
        with patch.object(geo_service, "_data_expert_assistant", create=True) as mock_assistant:
            mock_result = Mock()
            mock_result.modality = "cite_seq"
            mock_result.is_supported = False
            mock_result.confidence = 0.95
            mock_result.detected_signals = ["CITE-seq detected"]
            mock_result.compatibility_reason = "Multi-omics dataset"
            mock_result.suggestions = ["Wait for v2.6"]
            mock_assistant.detect_modality.return_value = mock_result

            # Should NOT raise exception
            is_compatible, message = geo_service._check_platform_compatibility("GSE12345", metadata)

            assert is_compatible is True
            assert "Multi-modal dataset" in message or "loading RNA" in message.lower()

    def test_cite_seq_without_rna_raises_error(self, geo_service):
        """CITE-seq with only protein/VDJ should raise error."""
        from lobster.core.exceptions import FeatureNotImplementedError

        metadata = {
            "title": "CITE-seq study",
            "platform_id": ["GPL24676"],  # 10X Chromium (series level - list)
            "samples": {
                "GSM1": {"characteristics_ch1": ["assay: protein"], "title": "S1_protein", "library_strategy": "", "platform_id": "GPL24676"},
                "GSM2": {"characteristics_ch1": ["assay: VDJ"], "title": "S1_VDJ", "library_strategy": "", "platform_id": "GPL24676"},
            },
        }

        with patch.object(geo_service, "_data_expert_assistant", create=True) as mock_assistant:
            mock_result = Mock()
            mock_result.modality = "cite_seq"
            mock_result.is_supported = False
            mock_result.confidence = 0.95
            mock_result.detected_signals = ["CITE-seq detected"]
            mock_result.compatibility_reason = "Multi-omics dataset"
            mock_result.suggestions = ["Wait for v2.6"]
            mock_assistant.detect_modality.return_value = mock_result

            with pytest.raises(FeatureNotImplementedError) as exc_info:
                geo_service._check_platform_compatibility("GSE12345", metadata)

            assert "cite_seq" in str(exc_info.value).lower()

    def test_multimodal_info_stored_in_metadata(self, geo_service):
        """Verify multimodal info is stored in metadata store."""
        metadata = {
            "title": "Multi-omics study",
            "platform_id": ["GPL24676"],  # 10X Chromium (series level - list)
            "samples": {
                "GSM1": {"characteristics_ch1": ["assay: RNA"], "title": "", "library_strategy": "RNA-Seq", "platform_id": "GPL24676"},
                "GSM2": {"characteristics_ch1": ["assay: protein"], "title": "", "library_strategy": "", "platform_id": "GPL24676"},
            },
        }

        geo_id = "GSE12345"

        # Store metadata first
        geo_service.data_manager._store_geo_metadata(
            geo_id=geo_id,
            metadata=metadata,
            stored_by="test"
        )

        with patch.object(geo_service, "_data_expert_assistant", create=True) as mock_assistant:
            mock_result = Mock()
            mock_result.modality = "cite_seq"
            mock_result.is_supported = False
            mock_result.confidence = 0.95
            mock_result.detected_signals = ["Multi-omics"]
            mock_result.compatibility_reason = "Multi-modal dataset"
            mock_result.suggestions = []
            mock_assistant.detect_modality.return_value = mock_result

            geo_service._check_platform_compatibility(geo_id, metadata)

            # Check that multimodal_info was stored
            stored_entry = geo_service.data_manager._get_geo_metadata(geo_id)
            assert stored_entry is not None
            assert "multimodal_info" in stored_entry
            assert stored_entry["multimodal_info"]["is_multimodal"] is True
            assert "rna" in stored_entry["multimodal_info"]["supported_types"]


class TestSampleFilteringIntegration:
    """Test that sample filtering works in the download pipeline."""

    def test_get_sample_info_filters_multimodal(self, geo_service):
        """_get_sample_info should filter samples for multi-modal datasets."""
        # Create mock GSE object
        mock_gse = Mock()
        mock_gse.metadata = {"geo_accession": ["GSE12345"]}

        # Create mock GSM objects
        mock_gsm1 = Mock()
        mock_gsm1.metadata = {"title": ["Sample1_RNA"], "platform_id": ["GPL123"]}

        mock_gsm2 = Mock()
        mock_gsm2.metadata = {"title": ["Sample1_protein"], "platform_id": ["GPL123"]}

        mock_gse.gsms = {
            "GSM1": mock_gsm1,
            "GSM2": mock_gsm2,
        }

        # Store multimodal info in metadata
        geo_id = "GSE12345"
        multimodal_info = {
            "is_multimodal": True,
            "sample_types": {
                "rna": ["GSM1"],
                "protein": ["GSM2"],
            },
            "supported_types": ["rna"],
            "unsupported_types": ["protein"],
        }

        geo_service.data_manager._store_geo_metadata(
            geo_id=geo_id,
            metadata={"test": "data"},
            stored_by="test"
        )

        stored_entry = geo_service.data_manager._get_geo_metadata(geo_id)
        stored_entry["multimodal_info"] = multimodal_info
        geo_service.data_manager.metadata_store[geo_id] = stored_entry

        # Call _get_sample_info
        result = geo_service._get_sample_info(mock_gse)

        # Should only include RNA sample
        assert len(result) == 1
        assert "GSM1" in result
        assert "GSM2" not in result

    def test_get_sample_info_no_filter_for_single_modal(self, geo_service):
        """_get_sample_info should include all samples for single-modal datasets."""
        mock_gse = Mock()
        mock_gse.metadata = {"geo_accession": ["GSE99999"]}

        mock_gsm1 = Mock()
        mock_gsm1.metadata = {"title": ["Sample1"], "platform_id": ["GPL123"]}

        mock_gsm2 = Mock()
        mock_gsm2.metadata = {"title": ["Sample2"], "platform_id": ["GPL123"]}

        mock_gse.gsms = {
            "GSM1": mock_gsm1,
            "GSM2": mock_gsm2,
        }

        # No multimodal info stored (single-modal dataset)
        result = geo_service._get_sample_info(mock_gse)

        # Should include all samples
        assert len(result) == 2
        assert "GSM1" in result
        assert "GSM2" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
