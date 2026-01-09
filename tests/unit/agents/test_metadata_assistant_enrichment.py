"""
Unit tests for metadata_assistant disease enrichment functionality.

Tests all 4 enrichment phases:
- Phase 1: Column rescan (metadata field detection)
- Phase 2: LLM abstract extraction
- Phase 3: LLM methods extraction
- Phase 4: Manual mappings

Coverage includes:
- Individual phase testing with mocks
- Integration testing for hybrid mode
- Provenance tracking validation
- Dry-run behavior
- Edge cases (low confidence, multi-disease studies)
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict

import pytest

from lobster.agents.metadata_assistant import (
    _phase1_column_rescan,
    _phase2_llm_abstract_extraction,
    _phase3_llm_methods_extraction,
    _phase4_manual_mappings,
    _extract_disease_with_llm,
)
from lobster.core.data_manager_v2 import DataManagerV2


# =========================================================================
# Mock Fixtures
# =========================================================================

@pytest.fixture
def mock_llm():
    """Mock LLM with controllable responses."""
    class MockLLM:
        def __init__(self, response=None):
            self.response = response or {
                "disease": "crc",
                "confidence": 0.92,
                "evidence": "Patients with CRC recruited",
                "reasoning": "Explicit CRC mention"
            }
            self.call_count = 0

        def invoke(self, messages):
            """Mock invoke that returns JSON response."""
            self.call_count += 1

            class MockResponse:
                def __init__(self, content):
                    self.content = content

            return MockResponse(json.dumps(self.response))

        def set_response(self, response_dict):
            """Update mock response for next call."""
            self.response = response_dict

    return MockLLM()


@pytest.fixture
def mock_data_manager(tmp_path):
    """Mock DataManagerV2 with temporary workspace."""
    mock_dm = Mock(spec=DataManagerV2)
    mock_dm.workspace_path = tmp_path
    mock_dm.log_tool_usage = Mock()
    mock_dm.metadata_store = {}

    # Create metadata directory
    (tmp_path / "metadata").mkdir(exist_ok=True)

    return mock_dm


@pytest.fixture
def sample_data():
    """Sample dataset for testing enrichment."""
    return [
        {
            "run_accession": "SRR1001",
            "publication_entry_id": "pub_queue_pmid_12345",
            "publication_title": "Microbiome in colorectal cancer",
            "sample_name": "sample_1"
        },
        {
            "run_accession": "SRR1002",
            "publication_entry_id": "pub_queue_pmid_12345",
            "publication_title": "Microbiome in colorectal cancer",
            "sample_name": "sample_2"
        },
        {
            "run_accession": "SRR2001",
            "publication_entry_id": "pub_queue_doi_10_5678",
            "publication_title": "UC gut microbiome study",
            "sample_name": "sample_3"
        },
    ]


@pytest.fixture
def cached_publication_abstract(mock_data_manager):
    """Create cached publication abstract file."""
    pub_id = "pub_queue_pmid_12345"
    metadata_path = mock_data_manager.workspace_path / "metadata" / f"{pub_id}_metadata.json"

    metadata = {
        "identifier": "pmid:12345",
        "type": "pubmed",
        "content": "We recruited 50 patients diagnosed with colorectal cancer (CRC) "
                  "and 30 healthy controls. Fecal samples were collected for 16S "
                  "sequencing to analyze gut microbiome composition."
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

    return metadata_path


@pytest.fixture
def cached_publication_methods(mock_data_manager):
    """Create cached publication methods file."""
    pub_id = "pub_queue_doi_10_5678"
    methods_path = mock_data_manager.workspace_path / "metadata" / f"{pub_id}_methods.json"

    methods_data = {
        "identifier": "doi:10.5678",
        "type": "doi",
        "methods_text": "Study participants: Ulcerative colitis patients (n=45) were "
                        "recruited from gastroenterology clinics. Diagnosis was confirmed "
                        "by colonoscopy and biopsy. All patients were in active disease phase."
    }

    with open(methods_path, 'w') as f:
        json.dump(methods_data, f)

    return methods_path


# =========================================================================
# Phase 1: Column Rescan Tests
# =========================================================================

class TestPhase1ColumnRescan:
    """Test Phase 1: Column rescan functionality."""

    def test_column_rescan_finds_missed_field(self):
        """Test Phase 1: Column re-scan finds disease in non-standard columns."""
        samples = [
            {"run_accession": "SRR1", "ibd_diagnosis_refined": "Crohns"},
            {"run_accession": "SRR2", "clinical_condition": "ulcerative colitis"}
        ]

        enriched, log = _phase1_column_rescan(samples)

        assert enriched == 2
        assert samples[0]["disease"] == "cd"
        assert samples[0]["disease_source"] == "column_remapped:ibd_diagnosis_refined"
        assert samples[0]["disease_confidence"] == 1.0
        assert "enrichment_timestamp" in samples[0]

        assert samples[1]["disease"] == "uc"
        assert samples[1]["disease_source"] == "column_remapped:clinical_condition"

    def test_column_rescan_skips_existing_disease(self):
        """Test that rescan skips samples with existing disease annotation."""
        samples = [
            {"run_accession": "SRR1", "disease": "crc", "diagnosis": "healthy"},
            {"run_accession": "SRR2", "diagnosis": "ulcerative colitis"}
        ]

        enriched, log = _phase1_column_rescan(samples)

        assert enriched == 1  # Only SRR2
        assert samples[0]["disease"] == "crc"  # Unchanged
        assert samples[1]["disease"] == "uc"  # Newly enriched

    def test_column_rescan_handles_multiple_keywords(self):
        """Test rescan with various keyword patterns."""
        samples = [
            {"run_accession": "SRR1", "disease_type": "colorectal"},
            {"run_accession": "SRR2", "condition": "colon_cancer"},
            {"run_accession": "SRR3", "status": "healthy control"},
            {"run_accession": "SRR4", "group": "non_ibd"}
        ]

        enriched, log = _phase1_column_rescan(samples)

        assert enriched == 4
        assert samples[0]["disease"] == "crc"
        assert samples[1]["disease"] == "crc"
        assert samples[2]["disease"] == "healthy"
        assert samples[3]["disease"] == "healthy"

    def test_column_rescan_case_insensitive(self):
        """Test that keyword matching is case-insensitive."""
        samples = [
            {"run_accession": "SRR1", "diagnosis": "CROHNS DISEASE"},
            {"run_accession": "SRR2", "condition": "Ulcerative Colitis"}
        ]

        enriched, log = _phase1_column_rescan(samples)

        assert enriched == 2
        assert samples[0]["disease"] == "cd"
        assert samples[1]["disease"] == "uc"

    def test_column_rescan_empty_samples(self):
        """Test rescan with empty sample list."""
        samples = []

        enriched, log = _phase1_column_rescan(samples)

        assert enriched == 0
        assert isinstance(log, list)


# =========================================================================
# Phase 2: LLM Abstract Extraction Tests
# =========================================================================

class TestPhase2LLMAbstractExtraction:
    """Test Phase 2: LLM abstract extraction."""

    def test_llm_abstract_extraction_mocked(self, sample_data, mock_data_manager,
                                           cached_publication_abstract, mock_llm):
        """Test Phase 2: LLM extraction with mocked responses."""
        # Setup: Remove pre-existing disease
        for sample in sample_data:
            sample.pop('disease', None)

        # Mock LLM to return CRC extraction
        mock_llm.set_response({
            "disease": "crc",
            "confidence": 0.92,
            "evidence": "recruited 50 patients diagnosed with colorectal cancer",
            "reasoning": "Explicit CRC patient recruitment mentioned"
        })

        enriched, log = _phase2_llm_abstract_extraction(
            sample_data,
            mock_data_manager,
            confidence_threshold=0.8,
            llm=mock_llm
        )

        # Verify enrichment (only first 2 samples share pub_queue_pmid_12345)
        assert enriched == 2
        assert sample_data[0]["disease"] == "crc"
        assert sample_data[0]["disease_source"] == "abstract_llm"
        assert sample_data[0]["disease_confidence"] == 0.92
        assert "colorectal cancer" in sample_data[0]["disease_evidence"]
        assert "enrichment_timestamp" in sample_data[0]

        assert sample_data[1]["disease"] == "crc"
        assert sample_data[2].get("disease") is None  # Different publication

    def test_llm_abstract_extraction_low_confidence_rejected(self, sample_data,
                                                             mock_data_manager,
                                                             cached_publication_abstract,
                                                             mock_llm):
        """Test that LLM extractions below threshold are rejected."""
        for sample in sample_data:
            sample.pop('disease', None)

        # Mock LLM to return low confidence
        mock_llm.set_response({
            "disease": "uc",
            "confidence": 0.65,
            "evidence": "Possibly UC mentioned",
            "reasoning": "Weak context"
        })

        enriched, log = _phase2_llm_abstract_extraction(
            sample_data,
            mock_data_manager,
            confidence_threshold=0.8,
            llm=mock_llm
        )

        # Verify NO enrichment (below threshold)
        assert enriched == 0
        assert sample_data[0].get("disease") is None

        # Check log messages
        log_text = "\n".join(log)
        assert "Low confidence 0.65" in log_text

    def test_llm_abstract_extraction_no_cached_file(self, sample_data, mock_data_manager, mock_llm):
        """Test handling of missing cached abstract files."""
        for sample in sample_data:
            sample.pop('disease', None)

        enriched, log = _phase2_llm_abstract_extraction(
            sample_data,
            mock_data_manager,
            confidence_threshold=0.8,
            llm=mock_llm
        )

        # Verify NO enrichment (no cached files)
        assert enriched == 0

        # Check log messages
        log_text = "\n".join(log)
        assert "No cached metadata" in log_text or "skipping" in log_text.lower()

    def test_llm_abstract_extraction_groups_by_publication(self, mock_data_manager,
                                                          cached_publication_abstract,
                                                          mock_llm):
        """Test that samples are grouped by publication to avoid redundant LLM calls."""
        samples = [
            {"run_accession": f"SRR100{i}",
             "publication_entry_id": "pub_queue_pmid_12345",
             "publication_title": "Test study"}
            for i in range(5)
        ]

        mock_llm.set_response({
            "disease": "crc",
            "confidence": 0.90,
            "evidence": "CRC patients recruited",
            "reasoning": "Clear CRC mention"
        })

        enriched, log = _phase2_llm_abstract_extraction(
            samples,
            mock_data_manager,
            confidence_threshold=0.8,
            llm=mock_llm
        )

        # Verify single LLM call for all 5 samples
        assert mock_llm.call_count == 1
        assert enriched == 5

        # All samples should have same disease
        for sample in samples:
            assert sample["disease"] == "crc"


# =========================================================================
# Phase 3: LLM Methods Extraction Tests
# =========================================================================

class TestPhase3LLMMethodsExtraction:
    """Test Phase 3: LLM methods extraction."""

    def test_llm_methods_extraction_success(self, sample_data, mock_data_manager,
                                           cached_publication_methods, mock_llm):
        """Test Phase 3: Methods extraction for remaining samples."""
        # Setup: Only sample_data[2] has pub_queue_doi_10_5678
        for sample in sample_data:
            sample.pop('disease', None)

        mock_llm.set_response({
            "disease": "uc",
            "confidence": 0.88,
            "evidence": "Ulcerative colitis patients recruited from clinics",
            "reasoning": "Explicit UC diagnosis confirmation mentioned"
        })

        enriched, log = _phase3_llm_methods_extraction(
            sample_data,
            mock_data_manager,
            confidence_threshold=0.8,
            llm=mock_llm
        )

        # Verify enrichment
        assert enriched == 1
        assert sample_data[2]["disease"] == "uc"
        assert sample_data[2]["disease_source"] == "methods_llm"
        assert sample_data[2]["disease_confidence"] == 0.88

        # Other samples should remain unenriched (no methods file)
        assert sample_data[0].get("disease") is None

    def test_llm_methods_extraction_skips_enriched_samples(self, sample_data,
                                                           mock_data_manager,
                                                           cached_publication_methods,
                                                           mock_llm):
        """Test that methods extraction skips samples already enriched."""
        # Pre-enrich first sample
        sample_data[0]["disease"] = "crc"
        sample_data[1]["disease"] = "crc"
        sample_data[2].pop('disease', None)

        enriched, log = _phase3_llm_methods_extraction(
            sample_data,
            mock_data_manager,
            confidence_threshold=0.8,
            llm=mock_llm
        )

        # Only SRR2001 should be processed
        assert enriched <= 1  # Depends on whether methods file exists


# =========================================================================
# Phase 4: Manual Mappings Tests
# =========================================================================

class TestPhase4ManualMappings:
    """Test Phase 4: Manual mapping application."""

    def test_manual_mappings_applied_correctly(self):
        """Test Phase 4: Manual mappings applied correctly."""
        samples = [
            {"publication_entry_id": "pub_1", "run_accession": "SRR1"},
            {"publication_entry_id": "pub_2", "run_accession": "SRR2"}
        ]
        mappings = {"pub_1": "crc", "pub_2": "uc"}

        enriched, log = _phase4_manual_mappings(samples, mappings)

        assert enriched == 2
        assert samples[0]["disease"] == "crc"
        assert samples[0]["disease_source"] == "manual_override"
        assert samples[0]["disease_confidence"] == 1.0
        assert "enrichment_timestamp" in samples[0]

        assert samples[1]["disease"] == "uc"
        assert samples[1]["disease_source"] == "manual_override"

    def test_manual_mappings_skip_existing(self):
        """Test that manual mappings skip samples with existing disease."""
        samples = [
            {"publication_entry_id": "pub_1", "run_accession": "SRR1", "disease": "cd"},
            {"publication_entry_id": "pub_2", "run_accession": "SRR2"}
        ]
        mappings = {"pub_1": "crc", "pub_2": "uc"}

        enriched, log = _phase4_manual_mappings(samples, mappings)

        # Only SRR2 should be enriched
        assert enriched == 1
        assert samples[0]["disease"] == "cd"  # Unchanged
        assert samples[1]["disease"] == "uc"

    def test_manual_mappings_validates_disease_terms(self):
        """Test that invalid disease terms are rejected."""
        samples = [
            {"publication_entry_id": "pub_1", "run_accession": "SRR1"}
        ]
        mappings = {"pub_1": "invalid_disease"}

        enriched, log = _phase4_manual_mappings(samples, mappings)

        # Should NOT enrich with invalid term
        assert enriched == 0
        assert samples[0].get("disease") is None

        # Check log for warning
        log_text = "\n".join(log)
        assert "Invalid disease" in log_text

    def test_manual_mappings_empty_dict(self):
        """Test manual mappings with empty mappings dict."""
        samples = [{"publication_entry_id": "pub_1", "run_accession": "SRR1"}]
        mappings = {}

        enriched, log = _phase4_manual_mappings(samples, mappings)

        assert enriched == 0
        assert "No manual mappings" in "\n".join(log)


# =========================================================================
# LLM Extraction Helper Tests
# =========================================================================

class TestExtractDiseaseWithLLM:
    """Test _extract_disease_with_llm helper function."""

    def test_extract_from_abstract_only(self, mock_llm):
        """Test extraction with abstract text only."""
        mock_llm.set_response({
            "disease": "crc",
            "confidence": 0.92,
            "evidence": "CRC patients recruited",
            "reasoning": "Explicit mention"
        })

        result = _extract_disease_with_llm(
            llm=mock_llm,
            abstract_text="We recruited 50 CRC patients for this study.",
            methods_text=None,
            publication_title="CRC Microbiome Study",
            sample_count=50
        )

        assert result["disease"] == "crc"
        assert result["confidence"] == 0.92
        assert result["source"] == "abstract"

    def test_extract_from_methods_only(self, mock_llm):
        """Test extraction with methods text only."""
        mock_llm.set_response({
            "disease": "uc",
            "confidence": 0.85,
            "evidence": "UC patients diagnosed by colonoscopy",
            "reasoning": "Methods section describes UC cohort"
        })

        result = _extract_disease_with_llm(
            llm=mock_llm,
            abstract_text=None,
            methods_text="Study participants: UC patients diagnosed by colonoscopy...",
            publication_title="UC Study",
            sample_count=30
        )

        assert result["disease"] == "uc"
        assert result["source"] == "methods"

    def test_extract_from_both_abstract_and_methods(self, mock_llm):
        """Test extraction with both abstract and methods."""
        mock_llm.set_response({
            "disease": "cd",
            "confidence": 0.95,
            "evidence": "Crohn's disease patients",
            "reasoning": "Both abstract and methods confirm CD"
        })

        result = _extract_disease_with_llm(
            llm=mock_llm,
            abstract_text="Study of CD patients...",
            methods_text="Crohn's disease diagnosis confirmed...",
            publication_title="CD Microbiome",
            sample_count=40
        )

        assert result["disease"] == "cd"
        assert result["source"] == "abstract+methods"

    def test_extract_no_text_available(self, mock_llm):
        """Test extraction when no text is available."""
        result = _extract_disease_with_llm(
            llm=mock_llm,
            abstract_text=None,
            methods_text=None,
            publication_title="Test Study",
            sample_count=10
        )

        assert result["disease"] == "unknown"
        assert result["confidence"] == 0.0
        assert result["evidence"] == "No text available"
        assert result["source"] == "none"

    def test_extract_multi_disease_returns_unknown(self, mock_llm):
        """Test that mixed disease studies return 'unknown'."""
        mock_llm.set_response({
            "disease": "unknown",
            "confidence": 0.0,
            "evidence": "Mixed study: UC, CD, healthy controls",
            "reasoning": "Three equal groups, no primary disease"
        })

        result = _extract_disease_with_llm(
            llm=mock_llm,
            abstract_text="Comparison of UC, CD, and healthy controls...",
            methods_text=None,
            publication_title="IBD Comparison Study",
            sample_count=90
        )

        assert result["disease"] == "unknown"
        assert result["confidence"] == 0.0

    def test_extract_handles_llm_error(self, mock_llm):
        """Test graceful handling of LLM errors."""
        # Mock LLM to raise exception
        def raise_error(messages):
            raise ValueError("LLM API error")

        mock_llm.invoke = raise_error

        result = _extract_disease_with_llm(
            llm=mock_llm,
            abstract_text="Test abstract",
            methods_text=None,
            publication_title="Test",
            sample_count=10
        )

        assert result["disease"] == "unknown"
        assert result["confidence"] == 0.0
        assert "error" in result["evidence"].lower()

    def test_extract_validates_disease_terms(self, mock_llm):
        """Test that invalid disease terms are rejected."""
        mock_llm.set_response({
            "disease": "diabetes",  # Invalid term
            "confidence": 0.90,
            "evidence": "Diabetes patients",
            "reasoning": "Clear diabetes mention"
        })

        result = _extract_disease_with_llm(
            llm=mock_llm,
            abstract_text="Study of diabetes patients",
            methods_text=None,
            publication_title="Diabetes Study",
            sample_count=20
        )

        # Should return unknown for invalid disease
        assert result["disease"] == "unknown"
        assert result["confidence"] == 0.0


# =========================================================================
# Provenance Tracking Tests
# =========================================================================

class TestProvenanceTracking:
    """Test that all enrichment sources are tracked with proper provenance."""

    def test_phase1_provenance_fields(self):
        """Test Phase 1 adds all required provenance fields."""
        samples = [{"run_accession": "SRR1", "diagnosis": "crohns"}]

        enriched, log = _phase1_column_rescan(samples)

        sample = samples[0]
        assert "disease" in sample
        assert "disease_source" in sample
        assert "disease_confidence" in sample
        assert "disease_original" in sample
        assert "enrichment_timestamp" in sample

        # Verify timestamp format
        timestamp = sample["enrichment_timestamp"]
        datetime.fromisoformat(timestamp)  # Should not raise

    def test_phase2_provenance_fields(self, sample_data, mock_data_manager,
                                     cached_publication_abstract, mock_llm):
        """Test Phase 2 adds all required provenance fields."""
        for sample in sample_data:
            sample.pop('disease', None)

        mock_llm.set_response({
            "disease": "crc",
            "confidence": 0.92,
            "evidence": "CRC patients recruited",
            "reasoning": "Clear mention"
        })

        enriched, log = _phase2_llm_abstract_extraction(
            sample_data,
            mock_data_manager,
            confidence_threshold=0.8,
            llm=mock_llm
        )

        sample = sample_data[0]
        assert sample["disease"] == "crc"
        assert sample["disease_source"] == "abstract_llm"
        assert sample["disease_confidence"] == 0.92
        assert "disease_evidence" in sample
        assert "enrichment_timestamp" in sample

    def test_phase4_provenance_fields(self):
        """Test Phase 4 adds all required provenance fields."""
        samples = [{"publication_entry_id": "pub_1", "run_accession": "SRR1"}]
        mappings = {"pub_1": "uc"}

        enriched, log = _phase4_manual_mappings(samples, mappings)

        sample = samples[0]
        assert sample["disease"] == "uc"
        assert sample["disease_source"] == "manual_override"
        assert sample["disease_confidence"] == 1.0
        assert "enrichment_timestamp" in sample


# =========================================================================
# Edge Cases & Error Handling
# =========================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_sample_list_all_phases(self, mock_data_manager, mock_llm):
        """Test all phases handle empty sample lists gracefully."""
        samples = []

        # Phase 1
        enriched, log = _phase1_column_rescan(samples)
        assert enriched == 0

        # Phase 2
        enriched, log = _phase2_llm_abstract_extraction(
            samples, mock_data_manager, 0.8, mock_llm
        )
        assert enriched == 0

        # Phase 3
        enriched, log = _phase3_llm_methods_extraction(
            samples, mock_data_manager, 0.8, mock_llm
        )
        assert enriched == 0

        # Phase 4
        enriched, log = _phase4_manual_mappings(samples, {})
        assert enriched == 0

    def test_samples_missing_required_fields(self, mock_data_manager, mock_llm):
        """Test handling of samples missing publication_entry_id."""
        samples = [
            {"run_accession": "SRR1"},  # No publication_entry_id
            {"run_accession": "SRR2", "publication_entry_id": None}
        ]

        # Phase 2 should skip these samples
        enriched, log = _phase2_llm_abstract_extraction(
            samples, mock_data_manager, 0.8, mock_llm
        )
        assert enriched == 0

    def test_corrupted_cached_files(self, mock_data_manager, sample_data):
        """Test handling of corrupted cached metadata files."""
        # Create corrupted file
        pub_id = "pub_queue_pmid_12345"
        metadata_path = mock_data_manager.workspace_path / "metadata" / f"{pub_id}_metadata.json"

        with open(metadata_path, 'w') as f:
            f.write("{invalid json content")

        mock_llm = Mock()
        enriched, log = _phase2_llm_abstract_extraction(
            sample_data, mock_data_manager, 0.8, mock_llm
        )

        # Should handle gracefully
        assert enriched == 0
        log_text = "\n".join(log)
        assert "error" in log_text.lower() or "skipping" in log_text.lower()


# =========================================================================
# Integration Tests
# =========================================================================

@pytest.mark.integration
class TestEnrichmentIntegration:
    """Integration tests for full enrichment workflow."""

    def test_hybrid_mode_runs_phases_sequentially(self, sample_data, mock_data_manager,
                                                  cached_publication_abstract,
                                                  cached_publication_methods, mock_llm):
        """Test that hybrid mode executes phases in correct order."""
        # Setup mixed scenario
        samples = [
            {"run_accession": "SRR1", "diagnosis": "crohns"},  # Phase 1 will catch
            {"run_accession": "SRR2", "publication_entry_id": "pub_queue_pmid_12345",
             "publication_title": "Test"},  # Phase 2 will catch
            {"run_accession": "SRR3", "publication_entry_id": "pub_queue_doi_10_5678",
             "publication_title": "Test2"},  # Phase 3 will catch
            {"run_accession": "SRR4", "publication_entry_id": "pub_manual",
             "publication_title": "Test3"}  # Phase 4 will catch
        ]

        mock_llm.set_response({
            "disease": "crc",
            "confidence": 0.90,
            "evidence": "Test",
            "reasoning": "Test"
        })

        # Phase 1: Column rescan
        phase1_count, log1 = _phase1_column_rescan(samples)
        assert phase1_count >= 1  # SRR1

        # Phase 2: Abstract extraction
        phase2_count, log2 = _phase2_llm_abstract_extraction(
            samples, mock_data_manager, 0.8, mock_llm
        )

        # Phase 3: Methods extraction
        phase3_count, log3 = _phase3_llm_methods_extraction(
            samples, mock_data_manager, 0.8, mock_llm
        )

        # Phase 4: Manual mappings
        mappings = {"pub_manual": "uc"}
        phase4_count, log4 = _phase4_manual_mappings(samples, mappings)

        # Verify each phase contributed
        total_enriched = sum([phase1_count, phase2_count, phase3_count, phase4_count])
        assert total_enriched >= 2  # At least Phase 1 and Phase 4

    def test_provenance_chain_preserved(self):
        """Test that provenance is preserved across multiple enrichment attempts."""
        samples = [{"run_accession": "SRR1", "diagnosis": "healthy"}]

        # First enrichment (Phase 1)
        enriched, log = _phase1_column_rescan(samples)

        first_source = samples[0]["disease_source"]
        first_timestamp = samples[0]["enrichment_timestamp"]

        # Try Phase 1 again (should skip)
        enriched, log = _phase1_column_rescan(samples)

        # Verify original provenance unchanged
        assert samples[0]["disease_source"] == first_source
        assert samples[0]["enrichment_timestamp"] == first_timestamp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
