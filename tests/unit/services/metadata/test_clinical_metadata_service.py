"""
Unit tests for ClinicalMetadataService.

Tests cover:
- process_sample_metadata() with validation and column mapping
- create_responder_groups() for responder/non-responder classification
- get_timepoint_samples() for timepoint filtering
- filter_by_response_and_timepoint() for combined filtering
- 3-tuple return pattern compliance
- AnalysisStep IR generation
- Edge cases and error handling
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lobster.core.analysis_ir import AnalysisStep
from lobster.services.metadata.clinical_metadata_service import ClinicalMetadataService


@pytest.fixture
def mock_data_manager():
    """Create a mock DataManagerV2 instance."""
    return MagicMock()


@pytest.fixture
def service(mock_data_manager):
    """Create ClinicalMetadataService instance with mock DataManager."""
    return ClinicalMetadataService(mock_data_manager, cycle_length_days=21)


@pytest.fixture
def sample_metadata_df():
    """Create sample clinical metadata DataFrame for testing."""
    return pd.DataFrame(
        {
            "sample_id": ["S001", "S002", "S003", "S004", "S005"],
            "patient_id": ["P001", "P001", "P002", "P002", "P003"],
            "response_status": [
                "Complete Response",
                "PR",
                "Stable Disease",
                "PD",
                "Not Evaluable",
            ],
            "timepoint": ["C1D1", "C2D1", "C1D1", "C2D1", "C1D1"],
            "pfs_days": [180.0, 180.0, 90.0, 90.0, None],
            "pfs_event": [0, 0, 1, 1, None],
            "age": [55, 55, 62, 62, 48],
            "sex": ["M", "M", "F", "F", "M"],
        }
    )


class TestClinicalMetadataServiceInit:
    """Test ClinicalMetadataService initialization."""

    def test_init_default_cycle_length(self, mock_data_manager):
        """Default cycle length should be 21 days."""
        service = ClinicalMetadataService(mock_data_manager)
        assert service.cycle_length_days == 21

    def test_init_custom_cycle_length(self, mock_data_manager):
        """Custom cycle length should be stored."""
        service = ClinicalMetadataService(mock_data_manager, cycle_length_days=28)
        assert service.cycle_length_days == 28


class TestProcessSampleMetadata:
    """Test process_sample_metadata() method."""

    def test_returns_3_tuple(self, service, sample_metadata_df):
        """Method should return 3-tuple (DataFrame, dict, AnalysisStep)."""
        result = service.process_sample_metadata(sample_metadata_df)

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], dict)
        assert isinstance(result[2], AnalysisStep)

    def test_response_normalization(self, service, sample_metadata_df):
        """Response values should be normalized to canonical RECIST codes."""
        processed_df, stats, ir = service.process_sample_metadata(sample_metadata_df)

        # Check normalized column was created
        assert "response_status_normalized" in processed_df.columns

        # Check normalization correctness
        normalized = processed_df["response_status_normalized"].tolist()
        assert normalized[0] == "CR"  # "Complete Response" → "CR"
        assert normalized[1] == "PR"  # "PR" → "PR"
        assert normalized[2] == "SD"  # "Stable Disease" → "SD"
        assert normalized[3] == "PD"  # "PD" → "PD"
        assert normalized[4] == "NE"  # "Not Evaluable" → "NE"

    def test_response_group_derivation(self, service, sample_metadata_df):
        """Response group should be derived from normalized status."""
        processed_df, stats, ir = service.process_sample_metadata(sample_metadata_df)

        # Check response_group column was created
        assert "response_group" in processed_df.columns

        # Check group classification
        groups = processed_df["response_group"].tolist()
        assert groups[0] == "responder"  # CR
        assert groups[1] == "responder"  # PR
        assert groups[2] == "non_responder"  # SD
        assert groups[3] == "non_responder"  # PD
        assert groups[4] is None  # NE → no group

    def test_timepoint_parsing(self, service, sample_metadata_df):
        """Timepoint should be parsed to cycle and day."""
        processed_df, stats, ir = service.process_sample_metadata(sample_metadata_df)

        # Check parsed columns were created
        assert "cycle" in processed_df.columns
        assert "day" in processed_df.columns
        assert "absolute_day" in processed_df.columns

        # Check parsing for C1D1
        assert processed_df.loc[0, "cycle"] == 1
        assert processed_df.loc[0, "day"] == 1
        assert processed_df.loc[0, "absolute_day"] == 1

        # Check parsing for C2D1
        assert processed_df.loc[1, "cycle"] == 2
        assert processed_df.loc[1, "day"] == 1
        assert processed_df.loc[1, "absolute_day"] == 22  # (2-1)*21 + 1

    def test_column_mapping(self, service):
        """Column mapping should rename columns before processing."""
        df = pd.DataFrame(
            {
                "Sample": ["S001", "S002"],
                "BOR": ["CR", "SD"],  # Best Overall Response
                "TP": ["C1D1", "C2D1"],  # Timepoint abbreviation
            }
        )

        processed_df, stats, ir = service.process_sample_metadata(
            df,
            column_mapping={
                "Sample": "sample_id",
                "BOR": "response_status",
                "TP": "timepoint",
            },
        )

        # Check columns were mapped
        assert "sample_id" in processed_df.columns
        assert "response_status" in processed_df.columns
        assert "timepoint" in processed_df.columns

        # Check processing worked
        assert processed_df.loc[0, "response_status_normalized"] == "CR"

    def test_validation_disabled(self, service, sample_metadata_df):
        """With validate=False, no validation errors should be tracked."""
        processed_df, stats, ir = service.process_sample_metadata(
            sample_metadata_df, validate=False
        )

        assert stats["invalid_samples"] == 0
        assert stats["validation_errors"] == {}

    def test_stats_contain_expected_keys(self, service, sample_metadata_df):
        """Stats dictionary should contain expected keys."""
        processed_df, stats, ir = service.process_sample_metadata(sample_metadata_df)

        expected_keys = [
            "total_samples",
            "valid_samples",
            "invalid_samples",
            "validation_errors",
            "columns_mapped",
            "response_distribution",
            "response_group_distribution",
            "timepoints_parsed",
        ]

        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

    def test_ir_is_not_exportable(self, service, sample_metadata_df):
        """AnalysisStep should have exportable=False."""
        processed_df, stats, ir = service.process_sample_metadata(sample_metadata_df)

        assert ir.exportable is False
        assert ir.operation == "clinical_metadata.process_sample_metadata"


class TestCreateResponderGroups:
    """Test create_responder_groups() method."""

    def test_returns_3_tuple(self, service, sample_metadata_df):
        """Method should return 3-tuple (dict, dict, AnalysisStep)."""
        result = service.create_responder_groups(sample_metadata_df)

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], dict)
        assert isinstance(result[1], dict)
        assert isinstance(result[2], AnalysisStep)

    def test_groups_have_correct_keys(self, service, sample_metadata_df):
        """Groups dict should have responder, non_responder, unknown keys."""
        groups, stats, ir = service.create_responder_groups(sample_metadata_df)

        assert "responder" in groups
        assert "non_responder" in groups
        assert "unknown" in groups

    def test_correct_group_classification(self, service, sample_metadata_df):
        """Samples should be correctly classified into groups."""
        groups, stats, ir = service.create_responder_groups(sample_metadata_df)

        # CR (S001) and PR (S002) should be responders
        assert "S001" in groups["responder"]
        assert "S002" in groups["responder"]

        # SD (S003) and PD (S004) should be non-responders
        assert "S003" in groups["non_responder"]
        assert "S004" in groups["non_responder"]

        # NE (S005) should be unknown
        assert "S005" in groups["unknown"]

    def test_stats_contain_counts(self, service, sample_metadata_df):
        """Stats should contain group counts and percentages."""
        groups, stats, ir = service.create_responder_groups(sample_metadata_df)

        assert stats["responder_count"] == 2
        assert stats["non_responder_count"] == 2
        assert stats["unknown_count"] == 1
        assert stats["total_samples"] == 5
        assert stats["responder_percentage"] == 40.0

    def test_custom_response_column(self, service):
        """Should work with custom response column name."""
        df = pd.DataFrame(
            {
                "sample_id": ["S001", "S002"],
                "best_response": ["CR", "PD"],
            }
        )

        groups, stats, ir = service.create_responder_groups(
            df, response_column="best_response"
        )

        assert "S001" in groups["responder"]
        assert "S002" in groups["non_responder"]

    def test_missing_response_column_raises(self, service):
        """Missing response column should raise ValueError."""
        df = pd.DataFrame({"sample_id": ["S001", "S002"]})

        with pytest.raises(ValueError) as exc_info:
            service.create_responder_groups(df)

        assert "not found" in str(exc_info.value)

    def test_ir_is_not_exportable(self, service, sample_metadata_df):
        """AnalysisStep should have exportable=False."""
        groups, stats, ir = service.create_responder_groups(sample_metadata_df)

        assert ir.exportable is False
        assert ir.operation == "clinical_metadata.create_responder_groups"

    def test_dcr_grouping_strategy(self, service, sample_metadata_df):
        """DCR strategy should group SD with CR/PR as disease control."""
        groups, stats, ir = service.create_responder_groups(
            sample_metadata_df, grouping_strategy="dcr"
        )

        # With DCR: CR/PR/SD = disease_control, PD = progressive
        assert "disease_control" in groups
        assert "progressive" in groups
        assert "unknown" in groups

        # From fixture: CR, PR, SD, PD samples
        assert len(groups["disease_control"]) == 3  # CR + PR + SD
        assert len(groups["progressive"]) == 1  # PD only

        # Check stats
        assert stats["grouping_strategy"] == "dcr"
        assert stats["disease_control_count"] == 3
        assert stats["progressive_count"] == 1

    def test_orr_vs_dcr_different_results(self, service):
        """ORR and DCR strategies should produce different groupings for SD samples."""
        df = pd.DataFrame({
            "sample_id": ["S001", "S002", "S003", "S004"],
            "response_status": ["CR", "PR", "SD", "PD"],
        })

        # ORR grouping
        orr_groups, orr_stats, _ = service.create_responder_groups(df, grouping_strategy="orr")
        assert orr_stats["responder_count"] == 2  # CR + PR
        assert orr_stats["non_responder_count"] == 2  # SD + PD

        # DCR grouping
        dcr_groups, dcr_stats, _ = service.create_responder_groups(df, grouping_strategy="dcr")
        assert dcr_stats["disease_control_count"] == 3  # CR + PR + SD
        assert dcr_stats["progressive_count"] == 1  # PD only

    def test_invalid_grouping_strategy_raises(self, service, sample_metadata_df):
        """Invalid grouping strategy should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            service.create_responder_groups(sample_metadata_df, grouping_strategy="invalid")

        assert "grouping_strategy" in str(exc_info.value).lower()


class TestGetTimepointSamples:
    """Test get_timepoint_samples() method."""

    def test_returns_3_tuple(self, service, sample_metadata_df):
        """Method should return 3-tuple (list, dict, AnalysisStep)."""
        result = service.get_timepoint_samples(sample_metadata_df, "C1D1")

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], list)
        assert isinstance(result[1], dict)
        assert isinstance(result[2], AnalysisStep)

    def test_filters_by_exact_timepoint(self, service, sample_metadata_df):
        """Should return samples matching exact timepoint."""
        samples, stats, ir = service.get_timepoint_samples(sample_metadata_df, "C1D1")

        # S001, S003, S005 have C1D1
        assert len(samples) == 3
        assert "S001" in samples
        assert "S003" in samples
        assert "S005" in samples

    def test_filters_c2d1(self, service, sample_metadata_df):
        """Should filter for C2D1 timepoint."""
        samples, stats, ir = service.get_timepoint_samples(sample_metadata_df, "C2D1")

        # S002, S004 have C2D1
        assert len(samples) == 2
        assert "S002" in samples
        assert "S004" in samples

    def test_case_insensitive_matching(self, service, sample_metadata_df):
        """Timepoint matching should be case-insensitive."""
        samples_upper, _, _ = service.get_timepoint_samples(sample_metadata_df, "C1D1")
        samples_lower, _, _ = service.get_timepoint_samples(sample_metadata_df, "c1d1")

        assert len(samples_upper) == len(samples_lower)
        assert set(samples_upper) == set(samples_lower)

    def test_stats_contain_expected_keys(self, service, sample_metadata_df):
        """Stats should contain matching statistics."""
        samples, stats, ir = service.get_timepoint_samples(sample_metadata_df, "C1D1")

        assert stats["target_timepoint"] == "C1D1"
        assert stats["parsed_cycle"] == 1
        assert stats["parsed_day"] == 1
        assert stats["total_samples"] == 5
        assert stats["matching_samples"] == 3
        assert stats["match_percentage"] == 60.0

    def test_missing_timepoint_column_raises(self, service):
        """Missing timepoint column should raise ValueError."""
        df = pd.DataFrame({"sample_id": ["S001", "S002"]})

        with pytest.raises(ValueError) as exc_info:
            service.get_timepoint_samples(df, "C1D1")

        assert "not found" in str(exc_info.value)

    def test_ir_is_not_exportable(self, service, sample_metadata_df):
        """AnalysisStep should have exportable=False."""
        samples, stats, ir = service.get_timepoint_samples(sample_metadata_df, "C1D1")

        assert ir.exportable is False
        assert ir.operation == "clinical_metadata.get_timepoint_samples"


class TestFilterByResponseAndTimepoint:
    """Test filter_by_response_and_timepoint() method."""

    def test_returns_3_tuple(self, service, sample_metadata_df):
        """Method should return 3-tuple (list, dict, AnalysisStep)."""
        result = service.filter_by_response_and_timepoint(
            sample_metadata_df, response_group="responder", timepoint="C1D1"
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], list)
        assert isinstance(result[1], dict)
        assert isinstance(result[2], AnalysisStep)

    def test_filter_responders_at_c1d1(self, service, sample_metadata_df):
        """Should filter responders at C1D1."""
        samples, stats, ir = service.filter_by_response_and_timepoint(
            sample_metadata_df, response_group="responder", timepoint="C1D1"
        )

        # S001 is responder (CR) at C1D1
        assert len(samples) == 1
        assert "S001" in samples

    def test_filter_non_responders_at_c2d1(self, service, sample_metadata_df):
        """Should filter non-responders at C2D1."""
        samples, stats, ir = service.filter_by_response_and_timepoint(
            sample_metadata_df, response_group="non_responder", timepoint="C2D1"
        )

        # S004 is non-responder (PD) at C2D1
        assert len(samples) == 1
        assert "S004" in samples

    def test_filter_only_by_response(self, service, sample_metadata_df):
        """Should filter only by response group if timepoint is None."""
        samples, stats, ir = service.filter_by_response_and_timepoint(
            sample_metadata_df, response_group="responder", timepoint=None
        )

        # S001 (CR) and S002 (PR) are responders
        assert len(samples) == 2
        assert "S001" in samples
        assert "S002" in samples

    def test_filter_only_by_timepoint(self, service, sample_metadata_df):
        """Should filter only by timepoint if response_group is None."""
        samples, stats, ir = service.filter_by_response_and_timepoint(
            sample_metadata_df, response_group=None, timepoint="C2D1"
        )

        # S002 and S004 are at C2D1
        assert len(samples) == 2
        assert "S002" in samples
        assert "S004" in samples

    def test_no_filters_returns_all(self, service, sample_metadata_df):
        """With no filters, should return all samples."""
        samples, stats, ir = service.filter_by_response_and_timepoint(
            sample_metadata_df, response_group=None, timepoint=None
        )

        assert len(samples) == 5

    def test_invalid_response_group_raises(self, service, sample_metadata_df):
        """Invalid response group should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            service.filter_by_response_and_timepoint(
                sample_metadata_df, response_group="invalid_group"
            )

        assert "Invalid response_group" in str(exc_info.value)

    def test_stats_contain_filter_info(self, service, sample_metadata_df):
        """Stats should contain filter information."""
        samples, stats, ir = service.filter_by_response_and_timepoint(
            sample_metadata_df, response_group="responder", timepoint="C1D1"
        )

        assert stats["response_group_filter"] == "responder"
        assert stats["timepoint_filter"] == "C1D1"
        assert stats["initial_samples"] == 5
        assert stats["final_samples"] == 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self, service):
        """Should handle empty DataFrame gracefully."""
        empty_df = pd.DataFrame(
            columns=["sample_id", "response_status", "timepoint"]
        )

        # process_sample_metadata
        processed, stats, ir = service.process_sample_metadata(empty_df)
        assert len(processed) == 0
        assert stats["total_samples"] == 0

        # create_responder_groups
        groups, stats, ir = service.create_responder_groups(empty_df)
        assert len(groups["responder"]) == 0
        assert len(groups["non_responder"]) == 0

        # get_timepoint_samples
        samples, stats, ir = service.get_timepoint_samples(empty_df, "C1D1")
        assert len(samples) == 0

    def test_missing_values_handled(self, service):
        """Should handle missing values gracefully."""
        df = pd.DataFrame(
            {
                "sample_id": ["S001", "S002", "S003"],
                "response_status": ["CR", None, ""],
                "timepoint": ["C1D1", None, "C2D1"],
            }
        )

        groups, stats, ir = service.create_responder_groups(df)

        # Only S001 should be classified
        assert "S001" in groups["responder"]
        assert "S002" in groups["unknown"]
        assert "S003" in groups["unknown"]

    def test_index_as_sample_id(self, service):
        """Should use DataFrame index as sample_id if column missing."""
        df = pd.DataFrame(
            {
                "response_status": ["CR", "PR", "SD"],
                "timepoint": ["C1D1", "C1D1", "C2D1"],
            },
            index=["Sample_A", "Sample_B", "Sample_C"],
        )

        groups, stats, ir = service.create_responder_groups(df)

        assert "Sample_A" in groups["responder"]
        assert "Sample_B" in groups["responder"]
        assert "Sample_C" in groups["non_responder"]

    def test_custom_cycle_length_affects_absolute_day(self, mock_data_manager):
        """Custom cycle length should affect absolute day calculation."""
        service_21 = ClinicalMetadataService(mock_data_manager, cycle_length_days=21)
        service_28 = ClinicalMetadataService(mock_data_manager, cycle_length_days=28)

        df = pd.DataFrame(
            {
                "sample_id": ["S001"],
                "timepoint": ["C2D1"],
            }
        )

        processed_21, _, _ = service_21.process_sample_metadata(df)
        processed_28, _, _ = service_28.process_sample_metadata(df)

        # C2D1 with 21-day cycle: (2-1)*21 + 1 = 22
        assert processed_21.loc[0, "absolute_day"] == 22

        # C2D1 with 28-day cycle: (2-1)*28 + 1 = 29
        assert processed_28.loc[0, "absolute_day"] == 29


class TestIRGeneration:
    """Test AnalysisStep IR generation."""

    def test_ir_has_required_fields(self, service, sample_metadata_df):
        """IR should have all required AnalysisStep fields."""
        _, _, ir = service.process_sample_metadata(sample_metadata_df)

        assert ir.operation is not None
        assert ir.tool_name is not None
        assert ir.description is not None
        assert ir.library == "lobster"
        assert ir.parameters is not None
        assert ir.parameter_schema is not None

    def test_ir_execution_context(self, service, sample_metadata_df):
        """IR execution context should contain service info and stats."""
        _, _, ir = service.process_sample_metadata(sample_metadata_df)

        assert "timestamp" in ir.execution_context
        assert ir.execution_context["service"] == "ClinicalMetadataService"
        assert "cycle_length_days" in ir.execution_context
        assert "statistics" in ir.execution_context

    def test_ir_can_be_serialized(self, service, sample_metadata_df):
        """IR should be serializable to dict and back."""
        _, _, ir = service.process_sample_metadata(sample_metadata_df)

        # Test serialization
        ir_dict = ir.to_dict()
        assert isinstance(ir_dict, dict)

        # Test deserialization
        restored_ir = AnalysisStep.from_dict(ir_dict)
        assert restored_ir.operation == ir.operation
        assert restored_ir.exportable == ir.exportable
