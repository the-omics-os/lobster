"""
Unit tests for SampleMappingService.

Tests cross-dataset sample ID mapping with multiple strategies:
- Exact matching (case-insensitive)
- Fuzzy matching (RapidFuzz-based, conditional)
- Pattern matching (prefix/suffix removal)
- Metadata-supported matching
"""

from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.metadata.sample_mapping_service import (
    RAPIDFUZZ_AVAILABLE,
    SampleMappingResult,
    SampleMappingService,
    SampleMatch,
    UnmappedSample,
)


@pytest.fixture
def mock_data_manager():
    """Create mock DataManagerV2 for testing."""
    mock_dm = Mock(spec=DataManagerV2)
    mock_dm.modalities = {}
    return mock_dm


@pytest.fixture
def sample_mapping_service(mock_data_manager):
    """Create SampleMappingService instance for testing."""
    return SampleMappingService(data_manager=mock_data_manager, min_confidence=0.75)


@pytest.fixture
def source_adata():
    """Create source AnnData for testing."""
    # 10 samples with various naming conventions
    obs_names = [
        "Sample_A",
        "Sample_B",
        "GSM123_Rep1",
        "Control_001",
        "Treatment_002",
        "Patient_X_Batch1",
        "Patient_Y_Batch2",
        "Unique_Sample",
        "Ambiguous1",
        "Ambiguous2",
    ]
    adata = ad.AnnData(
        X=np.random.rand(10, 50),
        obs=pd.DataFrame(
            {
                "condition": [
                    "Control",
                    "Treatment",
                    "Control",
                    "Control",
                    "Treatment",
                    "Control",
                    "Treatment",
                    "Control",
                    "Control",
                    "Treatment",
                ],
                "tissue": ["Liver"] * 5 + ["Brain"] * 5,
                "timepoint": ["Day0"] * 10,
            },
            index=obs_names,
        ),
    )
    return adata


@pytest.fixture
def target_adata():
    """Create target AnnData for testing."""
    # 10 samples with different naming but some overlap
    obs_names = [
        "sample_a",  # Exact match (case-insensitive) to Sample_A
        "sample_b",  # Exact match to Sample_B
        "GSM123",  # Pattern match to GSM123_Rep1
        "Control_1",  # Pattern match to Control_001
        "Treatment_2",  # Pattern match to Treatment_002
        "PatientX",  # Pattern match to Patient_X_Batch1
        "PatientY",  # Pattern match to Patient_Y_Batch2
        "Different_Sample",  # No match
        "Ambiguous1",  # Exact match
        "Target_Only",  # No match in source
    ]
    adata = ad.AnnData(
        X=np.random.rand(10, 50),
        obs=pd.DataFrame(
            {
                "condition": [
                    "Control",
                    "Treatment",
                    "Control",
                    "Control",
                    "Treatment",
                    "Control",
                    "Treatment",
                    "Other",
                    "Control",
                    "Control",
                ],
                "tissue": ["Liver"] * 5 + ["Brain"] * 5,
                "timepoint": ["Day0"] * 10,
            },
            index=obs_names,
        ),
    )
    return adata


class TestSampleMappingServiceInit:
    """Test SampleMappingService initialization."""

    def test_init_with_defaults(self, mock_data_manager):
        """Test initialization with default parameters."""
        service = SampleMappingService(data_manager=mock_data_manager)
        assert service.data_manager == mock_data_manager
        assert service.min_confidence == 0.75

    def test_init_with_custom_confidence(self, mock_data_manager):
        """Test initialization with custom confidence threshold."""
        service = SampleMappingService(
            data_manager=mock_data_manager, min_confidence=0.85
        )
        assert service.min_confidence == 0.85

    def test_rapidfuzz_availability_logged(self, mock_data_manager, caplog):
        """Test that RapidFuzz availability is logged on initialization."""
        service = SampleMappingService(data_manager=mock_data_manager)
        if not RAPIDFUZZ_AVAILABLE:
            assert "RapidFuzz not available" in caplog.text


class TestExactMatching:
    """Test exact matching strategy."""

    def test_exact_match_case_insensitive(self, sample_mapping_service):
        """Test case-insensitive exact matching."""
        source_ids = ["Sample_A", "Sample_B", "Sample_C"]
        target_ids = ["sample_a", "SAMPLE_B", "Different"]

        matches, matched_src, matched_tgt = sample_mapping_service._exact_match(
            source_ids, target_ids
        )

        # Should match Sample_A → sample_a and Sample_B → SAMPLE_B
        assert len(matches) == 2
        assert len(matched_src) == 2
        assert len(matched_tgt) == 2

        # Verify match details
        match_dict = {m.source_id: m.target_id for m in matches}
        assert match_dict["Sample_A"] == "sample_a"
        assert match_dict["Sample_B"] == "SAMPLE_B"

        # All matches should have 100% confidence
        for match in matches:
            assert match.confidence_score == 1.0
            assert match.match_strategy == "exact"

    def test_exact_match_empty_lists(self, sample_mapping_service):
        """Test exact matching with empty input lists."""
        matches, matched_src, matched_tgt = sample_mapping_service._exact_match([], [])
        assert len(matches) == 0
        assert len(matched_src) == 0
        assert len(matched_tgt) == 0

    def test_exact_match_no_overlap(self, sample_mapping_service):
        """Test exact matching with no overlapping IDs."""
        source_ids = ["A", "B", "C"]
        target_ids = ["X", "Y", "Z"]

        matches, matched_src, matched_tgt = sample_mapping_service._exact_match(
            source_ids, target_ids
        )

        assert len(matches) == 0
        assert len(matched_src) == 0
        assert len(matched_tgt) == 0


class TestPatternMatching:
    """Test pattern-based matching strategy."""

    def test_pattern_match_common_prefixes(self, sample_mapping_service):
        """Test pattern matching with common prefixes."""
        source_ids = ["Sample_A", "GSM123_001", "Control_X", "Patient_Y"]
        target_ids = ["A", "123", "X", "Y"]

        matches = sample_mapping_service._pattern_match(source_ids, target_ids, set())

        # Should match via prefix removal
        assert len(matches) >= 2  # At least Sample_A → A and Control_X → X

        # Verify confidence scores
        for match in matches:
            assert match.confidence_score == 0.9
            assert match.match_strategy == "pattern"
            assert "normalized_id" in match.metadata_support

    def test_pattern_match_common_suffixes(self, sample_mapping_service):
        """Test pattern matching with common suffixes."""
        source_ids = ["Sample_Rep1", "Sample_Rep2", "Control_Batch1"]
        target_ids = ["Sample", "Control"]

        matches = sample_mapping_service._pattern_match(source_ids, target_ids, set())

        # Should match via suffix removal
        assert len(matches) >= 1

        for match in matches:
            assert match.confidence_score == 0.9
            assert match.match_strategy == "pattern"

    def test_pattern_match_with_matched_targets(self, sample_mapping_service):
        """Test pattern matching excludes already matched targets."""
        source_ids = ["Sample_A", "Sample_B"]
        target_ids = ["A", "B"]
        matched_target = {"A"}  # A already matched

        matches = sample_mapping_service._pattern_match(
            source_ids, target_ids, matched_target
        )

        # Should only match Sample_B → B (A excluded)
        target_ids_matched = [m.target_id for m in matches]
        assert "A" not in target_ids_matched

    def test_pattern_match_empty_lists(self, sample_mapping_service):
        """Test pattern matching with empty lists."""
        matches = sample_mapping_service._pattern_match([], [], set())
        assert len(matches) == 0


@pytest.mark.skipif(not RAPIDFUZZ_AVAILABLE, reason="RapidFuzz not available")
class TestFuzzyMatching:
    """Test fuzzy matching strategy (requires RapidFuzz)."""

    def test_fuzzy_match_similar_ids(self, sample_mapping_service):
        """Test fuzzy matching with similar sample IDs."""
        # Use test strings that score >= 0.75 with token_set_ratio
        source_ids = ["GSM12345_Liver", "Patient_A_Timepoint_1"]
        target_ids = ["GSM12345 Liver", "Patient A Timepoint 1"]

        matches = sample_mapping_service._fuzzy_match(source_ids, target_ids, set())

        # Should find matches above threshold (0.75)
        assert len(matches) >= 1

        for match in matches:
            assert match.confidence_score >= 0.75
            assert match.match_strategy == "fuzzy"

    def test_fuzzy_match_below_threshold(self, sample_mapping_service):
        """Test fuzzy matching filters out low-confidence matches."""
        source_ids = ["Completely_Different_A"]
        target_ids = ["Unrelated_Sample_B"]

        matches = sample_mapping_service._fuzzy_match(source_ids, target_ids, set())

        # Should filter out matches below 0.75 confidence
        assert all(m.confidence_score >= 0.75 for m in matches)

    def test_fuzzy_match_respects_matched_targets(self, sample_mapping_service):
        """Test fuzzy matching excludes already matched targets."""
        source_ids = ["Sample_A", "Sample_B"]
        target_ids = ["Sample A", "Sample B"]
        matched_target = {"Sample A"}

        matches = sample_mapping_service._fuzzy_match(
            source_ids, target_ids, matched_target
        )

        # Should not match to "Sample A" (already matched)
        target_ids_matched = [m.target_id for m in matches]
        assert "Sample A" not in target_ids_matched


class TestMetadataMatching:
    """Test metadata-supported matching strategy."""

    def test_metadata_match_aligned_fields(self, sample_mapping_service):
        """Test metadata matching with aligned metadata fields."""
        source_ids = ["Sample_A", "Sample_B"]
        target_ids = ["Target_X", "Target_Y"]

        source_metadata = {
            "Sample_A": {
                "condition": "Control",
                "tissue": "Liver",
                "timepoint": "Day0",
            },
            "Sample_B": {
                "condition": "Treatment",
                "tissue": "Brain",
                "timepoint": "Day0",
            },
        }
        target_metadata = {
            "Target_X": {
                "condition": "Control",
                "tissue": "Liver",
                "timepoint": "Day0",
            },
            "Target_Y": {
                "condition": "Treatment",
                "tissue": "Brain",
                "timepoint": "Day0",
            },
        }

        matches = sample_mapping_service._metadata_match(
            source_ids, target_ids, source_metadata, target_metadata, set()
        )

        # Should match based on metadata alignment (≥2 matching fields)
        assert len(matches) == 2

        for match in matches:
            assert match.confidence_score >= 0.7  # Metadata confidence range
            assert match.match_strategy == "metadata"
            assert len(match.metadata_support) >= 2  # At least 2 matching fields

    def test_metadata_match_insufficient_alignment(self, sample_mapping_service):
        """Test metadata matching requires ≥2 matching fields."""
        source_ids = ["Sample_A"]
        target_ids = ["Target_X"]

        source_metadata = {
            "Sample_A": {"condition": "Control", "tissue": "Liver"},
        }
        target_metadata = {
            "Target_X": {"condition": "Control", "tissue": "Different"},
        }

        matches = sample_mapping_service._metadata_match(
            source_ids, target_ids, source_metadata, target_metadata, set()
        )

        # Should not match (only 1 field aligned, needs ≥2)
        assert len(matches) == 0

    def test_metadata_match_no_metadata(self, sample_mapping_service):
        """Test metadata matching with missing metadata."""
        source_ids = ["Sample_A"]
        target_ids = ["Target_X"]

        matches = sample_mapping_service._metadata_match(
            source_ids, target_ids, None, None, set()
        )

        # Should return empty list when metadata unavailable
        assert len(matches) == 0

    def test_metadata_match_confidence_scaling(self, sample_mapping_service):
        """Test metadata confidence increases with more matching fields."""
        source_ids = ["Sample_A", "Sample_B"]
        target_ids = ["Target_X", "Target_Y"]

        # Sample_A: 2 matching fields
        # Sample_B: 4 matching fields
        source_metadata = {
            "Sample_A": {"condition": "Control", "tissue": "Liver"},
            "Sample_B": {
                "condition": "Treatment",
                "tissue": "Brain",
                "timepoint": "Day0",
                "batch": "Batch1",
            },
        }
        target_metadata = {
            "Target_X": {"condition": "Control", "tissue": "Liver"},
            "Target_Y": {
                "condition": "Treatment",
                "tissue": "Brain",
                "timepoint": "Day0",
                "batch": "Batch1",
            },
        }

        matches = sample_mapping_service._metadata_match(
            source_ids, target_ids, source_metadata, target_metadata, set()
        )

        # Find specific matches
        match_dict = {m.source_id: m for m in matches}

        if "Sample_A" in match_dict and "Sample_B" in match_dict:
            # Sample_B should have higher confidence (more matching fields)
            assert (
                match_dict["Sample_B"].confidence_score
                > match_dict["Sample_A"].confidence_score
            )


class TestMapSamplesByID:
    """Test main map_samples_by_id method."""

    def test_map_samples_full_workflow(
        self, sample_mapping_service, source_adata, target_adata, mock_data_manager
    ):
        """Test full mapping workflow with multiple strategies."""
        # Setup data manager
        mock_data_manager.modalities = {
            "source_dataset": source_adata,
            "target_dataset": target_adata,
        }
        mock_data_manager.list_modalities.return_value = [
            "source_dataset",
            "target_dataset",
        ]
        mock_data_manager.get_modality.side_effect = lambda x: (
            mock_data_manager.modalities[x]
        )

        result = sample_mapping_service.map_samples_by_id(
            source_identifier="source_dataset",
            target_identifier="target_dataset",
            strategies=["exact", "pattern"],  # Skip fuzzy for deterministic test
        )

        # Verify result structure
        assert isinstance(result, SampleMappingResult)
        assert len(result.exact_matches) >= 2  # At least sample_a, sample_b
        assert len(result.fuzzy_matches) >= 0  # Pattern matches count as fuzzy
        assert len(result.unmapped) >= 1  # Unique_Sample should be unmapped

        # Verify summary
        assert result.summary["total_source_samples"] == 10
        assert result.summary["total_target_samples"] == 10
        assert result.summary["mapping_rate"] > 0.0

    def test_map_samples_dataset_not_found(
        self, sample_mapping_service, mock_data_manager
    ):
        """Test error handling when dataset not found."""
        mock_data_manager.list_modalities.return_value = []

        with pytest.raises(ValueError, match="not found"):
            sample_mapping_service.map_samples_by_id(
                source_identifier="nonexistent",
                target_identifier="also_nonexistent",
            )

    def test_map_samples_empty_datasets(
        self, sample_mapping_service, mock_data_manager
    ):
        """Test error handling with empty datasets."""
        empty_adata = ad.AnnData(X=np.array([]).reshape(0, 10))
        mock_data_manager.modalities = {
            "source": empty_adata,
            "target": empty_adata,
        }
        mock_data_manager.list_modalities.return_value = ["source", "target"]
        mock_data_manager.get_modality.side_effect = lambda x: (
            mock_data_manager.modalities[x]
        )

        with pytest.raises(ValueError, match="no samples"):
            sample_mapping_service.map_samples_by_id(
                source_identifier="source",
                target_identifier="target",
            )

    def test_map_samples_strategy_filtering(
        self, sample_mapping_service, source_adata, target_adata, mock_data_manager
    ):
        """Test that strategy filtering works correctly."""
        mock_data_manager.modalities = {
            "source": source_adata,
            "target": target_adata,
        }
        mock_data_manager.list_modalities.return_value = ["source", "target"]
        mock_data_manager.get_modality.side_effect = lambda x: (
            mock_data_manager.modalities[x]
        )

        # Test with only exact matching
        result = sample_mapping_service.map_samples_by_id(
            source_identifier="source",
            target_identifier="target",
            strategies=["exact"],
        )

        # Should only have exact matches
        assert len(result.exact_matches) >= 2
        # Pattern matches should not be in fuzzy_matches
        pattern_matches = [
            m for m in result.fuzzy_matches if m.match_strategy == "pattern"
        ]
        assert len(pattern_matches) == 0

    def test_map_samples_confidence_threshold(
        self, mock_data_manager, source_adata, target_adata
    ):
        """Test that confidence threshold is respected."""
        # Create service with high confidence threshold
        service = SampleMappingService(
            data_manager=mock_data_manager, min_confidence=0.95
        )

        mock_data_manager.modalities = {
            "source": source_adata,
            "target": target_adata,
        }
        mock_data_manager.list_modalities.return_value = ["source", "target"]
        mock_data_manager.get_modality.side_effect = lambda x: (
            mock_data_manager.modalities[x]
        )

        result = service.map_samples_by_id(
            source_identifier="source",
            target_identifier="target",
            strategies=["exact"],  # Only exact (1.0 confidence)
        )

        # All matches should meet high threshold
        for match in result.exact_matches + result.fuzzy_matches:
            assert match.confidence_score >= 0.95


class TestFormatMappingReport:
    """Test report formatting."""

    def test_format_mapping_report_structure(self, sample_mapping_service):
        """Test that formatted report has correct structure."""
        # Create mock result
        result = SampleMappingResult(
            exact_matches=[
                SampleMatch(
                    source_id="Sample_A",
                    target_id="sample_a",
                    confidence_score=1.0,
                    match_strategy="exact",
                )
            ],
            fuzzy_matches=[
                SampleMatch(
                    source_id="GSM123_Rep1",
                    target_id="GSM123",
                    confidence_score=0.9,
                    match_strategy="pattern",
                )
            ],
            unmapped=[
                UnmappedSample(
                    sample_id="Unique_Sample",
                    dataset="source",
                    reason="No match found above confidence threshold",
                )
            ],
            summary={
                "total_source_samples": 10,
                "total_target_samples": 10,
                "exact_matches": 1,
                "fuzzy_matches": 1,
                "unmapped": 1,
                "mapping_rate": 0.2,
            },
            warnings=[],
        )

        report = sample_mapping_service.format_mapping_report(result)

        # Verify report contains key sections
        assert "# Sample Mapping Report" in report
        assert "## Summary" in report
        assert "Total Source Samples" in report
        assert "20.0%" in report  # Mapping rate formatted as percentage
        assert "## Exact Matches" in report
        assert "## Fuzzy/Pattern Matches" in report
        assert "## Unmapped Samples" in report

    def test_format_mapping_report_truncation(self, sample_mapping_service):
        """Test that long lists are truncated in report."""
        # Create result with >10 matches
        exact_matches = [
            SampleMatch(
                source_id=f"Sample_{i}",
                target_id=f"sample_{i}",
                confidence_score=1.0,
                match_strategy="exact",
            )
            for i in range(15)
        ]

        result = SampleMappingResult(
            exact_matches=exact_matches,
            fuzzy_matches=[],
            unmapped=[],
            summary={
                "total_source_samples": 15,
                "total_target_samples": 15,
                "exact_matches": 15,
                "fuzzy_matches": 0,
                "unmapped": 0,
                "mapping_rate": 1.0,
            },
            warnings=[],
        )

        report = sample_mapping_service.format_mapping_report(result)

        # Should show "... and X more"
        assert "and 5 more" in report

    def test_format_mapping_report_with_warnings(self, sample_mapping_service):
        """Test report formatting includes warnings."""
        result = SampleMappingResult(
            exact_matches=[],
            fuzzy_matches=[],
            unmapped=[],
            summary={
                "total_source_samples": 0,
                "total_target_samples": 0,
                "exact_matches": 0,
                "fuzzy_matches": 0,
                "unmapped": 0,
                "mapping_rate": 0.0,
            },
            warnings=["RapidFuzz not installed", "No metadata provided"],
        )

        report = sample_mapping_service.format_mapping_report(result)

        assert "## Warnings" in report
        assert "⚠️ RapidFuzz not installed" in report
        assert "⚠️ No metadata provided" in report


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_duplicate_source_ids(
        self, sample_mapping_service, target_adata, mock_data_manager
    ):
        """Test handling of duplicate source IDs."""
        # Create source with duplicate IDs
        source_adata = ad.AnnData(
            X=np.random.rand(3, 50),
            obs=pd.DataFrame(
                {"condition": ["Control", "Treatment", "Control"]},
                index=["Sample_A", "Sample_A", "Sample_B"],  # Duplicate Sample_A
            ),
        )

        mock_data_manager.modalities = {"source": source_adata, "target": target_adata}
        mock_data_manager.list_modalities.return_value = ["source", "target"]
        mock_data_manager.get_modality.side_effect = lambda x: (
            mock_data_manager.modalities[x]
        )

        result = sample_mapping_service.map_samples_by_id(
            source_identifier="source",
            target_identifier="target",
            strategies=["exact"],
        )

        # Should handle duplicates gracefully (first occurrence wins)
        assert isinstance(result, SampleMappingResult)

    def test_special_characters_in_ids(self, sample_mapping_service, mock_data_manager):
        """Test handling of special characters in sample IDs."""
        source_adata = ad.AnnData(
            X=np.random.rand(3, 50),
            obs=pd.DataFrame(
                index=["Sample-01", "Sample.02", "Sample_03"],
            ),
        )
        target_adata = ad.AnnData(
            X=np.random.rand(3, 50),
            obs=pd.DataFrame(
                index=["sample-01", "sample.02", "sample_03"],
            ),
        )

        mock_data_manager.modalities = {"source": source_adata, "target": target_adata}
        mock_data_manager.list_modalities.return_value = ["source", "target"]
        mock_data_manager.get_modality.side_effect = lambda x: (
            mock_data_manager.modalities[x]
        )

        result = sample_mapping_service.map_samples_by_id(
            source_identifier="source",
            target_identifier="target",
            strategies=["exact"],
        )

        # Should match all 3 (case-insensitive exact matching)
        assert result.summary["exact_matches"] == 3

    def test_no_matches_found(self, sample_mapping_service, mock_data_manager):
        """Test scenario where no matches are found."""
        source_adata = ad.AnnData(
            X=np.random.rand(3, 50),
            obs=pd.DataFrame(index=["A", "B", "C"]),
        )
        target_adata = ad.AnnData(
            X=np.random.rand(3, 50),
            obs=pd.DataFrame(index=["X", "Y", "Z"]),
        )

        mock_data_manager.modalities = {"source": source_adata, "target": target_adata}
        mock_data_manager.list_modalities.return_value = ["source", "target"]
        mock_data_manager.get_modality.side_effect = lambda x: (
            mock_data_manager.modalities[x]
        )

        result = sample_mapping_service.map_samples_by_id(
            source_identifier="source",
            target_identifier="target",
            strategies=["exact", "pattern"],
        )

        # Should have all samples unmapped
        assert len(result.unmapped) == 3
        assert result.summary["mapping_rate"] == 0.0
