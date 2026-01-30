"""
Unit tests for clinical trial metadata schema definitions.

Tests cover:
- RECIST 1.1 response normalization
- Response group classification (responder/non-responder)
- Timepoint parsing (C1D1, C2D8, Baseline, etc.)
- ClinicalSample Pydantic model validation
- Edge cases and error handling
"""

import math

import pandas as pd
import pytest

from lobster.core.schemas.clinical_schema import (
    NON_RESPONDER_GROUP,
    RECIST_RESPONSES,
    RESPONDER_GROUP,
    RESPONSE_SYNONYMS,
    SPECIAL_TIMEPOINTS,
    TIMEPOINT_PATTERNS,
    ClinicalSample,
    classify_response_group,
    is_non_responder,
    is_responder,
    normalize_response,
    parse_timepoint,
    timepoint_to_absolute_day,
)


class TestRECISTConstants:
    """Test RECIST 1.1 constant definitions."""

    def test_recist_responses_contains_all_codes(self):
        """Verify all canonical RECIST codes are defined."""
        expected_codes = {"CR", "PR", "SD", "PD", "NE"}
        assert set(RECIST_RESPONSES.keys()) == expected_codes

    def test_responder_group_contains_cr_pr(self):
        """Responder group should include CR and PR only."""
        assert RESPONDER_GROUP == {"CR", "PR"}

    def test_non_responder_group_contains_sd_pd(self):
        """Non-responder group should include SD and PD only."""
        assert NON_RESPONDER_GROUP == {"SD", "PD"}

    def test_ne_not_in_either_group(self):
        """NE (Not Evaluable) should not be in responder or non-responder groups."""
        assert "NE" not in RESPONDER_GROUP
        assert "NE" not in NON_RESPONDER_GROUP

    def test_response_synonyms_lowercase_keys(self):
        """All synonym keys should be lowercase for case-insensitive lookup."""
        for key in RESPONSE_SYNONYMS.keys():
            assert key == key.lower(), f"Synonym key '{key}' should be lowercase"


class TestNormalizeResponse:
    """Test response value normalization to canonical RECIST codes."""

    @pytest.mark.parametrize(
        "input_value,expected",
        [
            # Canonical codes (case variations)
            ("CR", "CR"),
            ("cr", "CR"),
            ("Cr", "CR"),
            ("PR", "PR"),
            ("pr", "PR"),
            ("SD", "SD"),
            ("sd", "SD"),
            ("PD", "PD"),
            ("pd", "PD"),
            ("NE", "NE"),
            ("ne", "NE"),
            # Full names
            ("Complete Response", "CR"),
            ("complete response", "CR"),
            ("COMPLETE RESPONSE", "CR"),
            ("Partial Response", "PR"),
            ("partial response", "PR"),
            ("Stable Disease", "SD"),
            ("stable disease", "SD"),
            ("Progressive Disease", "PD"),
            ("progressive disease", "PD"),
            ("Not Evaluable", "NE"),
            ("not evaluable", "NE"),
            ("Not Evaluated", "NE"),
            # Common variations
            ("complete", "CR"),
            ("partial", "PR"),
            ("stable", "SD"),
            ("progressive", "PD"),
            ("progression", "PD"),
            # Clinical shorthand (unambiguous only)
            ("prog", "PD"),
            ("stab", "SD"),
            # iRECIST (immune RECIST for immunotherapy trials)
            ("iCR", "CR"),
            ("iPR", "PR"),
            ("iSD", "SD"),
            ("iPD", "PD"),
            ("iCPD", "PD"),
            ("iUPD", "PD"),
            ("immune complete response", "CR"),
            ("immune partial response", "PR"),
            ("immune stable disease", "SD"),
            ("immune progressive disease", "PD"),
        ],
    )
    def test_normalize_valid_responses(self, input_value, expected):
        """Test normalization of valid response values."""
        assert normalize_response(input_value) == expected

    @pytest.mark.parametrize(
        "input_value",
        [
            None,
            "",
            "   ",
            "invalid",
            "unknown",
            "xyz",
            123,
        ],
    )
    def test_normalize_invalid_returns_none(self, input_value):
        """Invalid values should return None without raising."""
        assert normalize_response(input_value) is None

    def test_normalize_nan_returns_none(self):
        """pandas NaN values should return None."""
        assert normalize_response(float("nan")) is None
        assert normalize_response(pd.NA) is None

    def test_normalize_responder_returns_none_with_guidance(self, caplog):
        """'responder' is a GROUP, not a code - should return None with debug log."""
        import logging

        caplog.set_level(logging.DEBUG)
        result = normalize_response("responder")
        assert result is None
        # Check debug log was emitted
        assert any("group classification" in record.message.lower() for record in caplog.records)

    def test_normalize_resp_returns_none(self):
        """'resp' is ambiguous (could mean responder GROUP or response) - should return None."""
        # 'resp' was intentionally removed from synonyms due to ambiguity
        result = normalize_response("resp")
        assert result is None

    def test_normalize_non_responder_returns_none(self):
        """'non-responder' variations should return None (they're groups, not codes)."""
        assert normalize_response("non-responder") is None
        assert normalize_response("nonresponder") is None
        assert normalize_response("non_responder") is None

    def test_normalize_whitespace_handling(self):
        """Whitespace should be stripped before normalization."""
        assert normalize_response("  CR  ") == "CR"
        assert normalize_response("\tPR\n") == "PR"
        assert normalize_response(" complete response ") == "CR"


class TestClassifyResponseGroup:
    """Test response code to group classification."""

    @pytest.mark.parametrize(
        "code,expected",
        [
            ("CR", "responder"),
            ("PR", "responder"),
            ("SD", "non_responder"),
            ("PD", "non_responder"),
        ],
    )
    def test_classify_valid_codes(self, code, expected):
        """Valid codes should classify to correct groups."""
        assert classify_response_group(code) == expected

    def test_classify_ne_returns_none(self):
        """NE (Not Evaluable) should not classify to any group."""
        assert classify_response_group("NE") is None

    def test_classify_invalid_returns_none(self):
        """Invalid codes should return None."""
        assert classify_response_group("") is None
        assert classify_response_group("INVALID") is None
        assert classify_response_group(None) is None

    def test_classify_case_insensitive(self):
        """Classification should be case-insensitive."""
        assert classify_response_group("cr") == "responder"
        assert classify_response_group("Cr") == "responder"
        assert classify_response_group("sd") == "non_responder"

    def test_classify_with_whitespace(self):
        """Classification should handle whitespace."""
        assert classify_response_group("  CR  ") == "responder"
        assert classify_response_group("\tPD\n") == "non_responder"


class TestIsResponderNonResponder:
    """Test convenience boolean functions."""

    def test_is_responder_true_for_cr_pr(self):
        """CR and PR should return True for is_responder."""
        assert is_responder("CR") is True
        assert is_responder("PR") is True

    def test_is_responder_false_for_others(self):
        """SD, PD, NE should return False for is_responder."""
        assert is_responder("SD") is False
        assert is_responder("PD") is False
        assert is_responder("NE") is False

    def test_is_non_responder_true_for_sd_pd(self):
        """SD and PD should return True for is_non_responder."""
        assert is_non_responder("SD") is True
        assert is_non_responder("PD") is True

    def test_is_non_responder_false_for_others(self):
        """CR, PR, NE should return False for is_non_responder."""
        assert is_non_responder("CR") is False
        assert is_non_responder("PR") is False
        assert is_non_responder("NE") is False


class TestParseTimepoint:
    """Test clinical trial timepoint parsing."""

    @pytest.mark.parametrize(
        "timepoint,expected_cycle,expected_day",
        [
            # C#D# format (most common)
            ("C1D1", 1, 1),
            ("C2D1", 2, 1),
            ("C2D8", 2, 8),
            ("C3D15", 3, 15),
            ("c1d1", 1, 1),  # Case insensitive
            ("C10D21", 10, 21),
            # Cycle # Day # format (verbose)
            ("Cycle 1 Day 1", 1, 1),
            ("Cycle 2 Day 8", 2, 8),
            ("cycle 1 day 1", 1, 1),
            ("Cycle1Day1", 1, 1),
            # W#D# format (weekly)
            ("W1D1", 1, 1),
            ("W2D3", 2, 3),
            ("w1d1", 1, 1),
            # Week # Day # format (verbose weekly)
            ("Week 1 Day 1", 1, 1),
            ("Week 2 Day 7", 2, 7),
        ],
    )
    def test_parse_standard_timepoints(self, timepoint, expected_cycle, expected_day):
        """Test parsing of standard timepoint formats."""
        cycle, day = parse_timepoint(timepoint)
        assert cycle == expected_cycle
        assert day == expected_day

    @pytest.mark.parametrize(
        "timepoint",
        [
            "baseline",
            "Baseline",
            "BASELINE",
            "screening",
            "Screening",
            "pre-treatment",
            "pretreatment",
            "pre_treatment",
            "day0",
            "d0",
        ],
    )
    def test_parse_baseline_timepoints(self, timepoint):
        """Baseline/screening timepoints should return (0, 0)."""
        cycle, day = parse_timepoint(timepoint)
        assert cycle == 0
        assert day == 0

    @pytest.mark.parametrize(
        "timepoint",
        [
            "eot",
            "EOT",
            "end of treatment",
            "End Of Treatment",
            "follow-up",
            "followup",
            "follow_up",
        ],
    )
    def test_parse_special_timepoints_no_day(self, timepoint):
        """EOT and follow-up have no cycle/day representation."""
        cycle, day = parse_timepoint(timepoint)
        assert cycle is None
        assert day is None

    @pytest.mark.parametrize(
        "timepoint",
        [
            None,
            "",
            "   ",
            "invalid",
            "Day 1",  # Missing cycle
            "C1",  # Missing day
            "random text",
        ],
    )
    def test_parse_invalid_returns_none_none(self, timepoint):
        """Invalid timepoints should return (None, None)."""
        cycle, day = parse_timepoint(timepoint)
        assert cycle is None
        assert day is None

    def test_parse_whitespace_handling(self):
        """Timepoint parsing should handle whitespace."""
        cycle, day = parse_timepoint("  C1D1  ")
        assert cycle == 1
        assert day == 1


class TestTimepointToAbsoluteDay:
    """Test conversion of timepoints to absolute day numbers."""

    @pytest.mark.parametrize(
        "timepoint,cycle_length,expected_day",
        [
            # Standard 21-day cycles
            ("C1D1", 21, 1),
            ("C1D8", 21, 8),
            ("C1D15", 21, 15),
            ("C2D1", 21, 22),  # Day 1 of cycle 2 = 21 + 1
            ("C2D8", 21, 29),  # Day 8 of cycle 2 = 21 + 8
            ("C3D1", 21, 43),  # Day 1 of cycle 3 = 42 + 1
            # 28-day cycles
            ("C1D1", 28, 1),
            ("C2D1", 28, 29),  # Day 1 of cycle 2 = 28 + 1
            # 14-day cycles (biweekly)
            ("C1D1", 14, 1),
            ("C2D1", 14, 15),  # Day 1 of cycle 2 = 14 + 1
        ],
    )
    def test_absolute_day_calculation(self, timepoint, cycle_length, expected_day):
        """Test absolute day calculation with various cycle lengths."""
        result = timepoint_to_absolute_day(timepoint, cycle_length_days=cycle_length)
        assert result == expected_day

    def test_baseline_returns_zero(self):
        """Baseline should return absolute day 0."""
        assert timepoint_to_absolute_day("baseline") == 0
        assert timepoint_to_absolute_day("screening") == 0

    def test_eot_returns_none(self):
        """EOT has no absolute day representation."""
        assert timepoint_to_absolute_day("EOT") is None
        assert timepoint_to_absolute_day("end of treatment") is None

    def test_invalid_returns_none(self):
        """Invalid timepoints should return None."""
        assert timepoint_to_absolute_day("invalid") is None
        assert timepoint_to_absolute_day("") is None


class TestClinicalSampleModel:
    """Test ClinicalSample Pydantic model validation."""

    def test_minimal_valid_sample(self):
        """Minimal sample with only required field should be valid."""
        sample = ClinicalSample(sample_id="S001")
        assert sample.sample_id == "S001"
        assert sample.patient_id is None
        assert sample.response_status is None

    def test_full_sample_creation(self):
        """Test creation with all fields populated."""
        sample = ClinicalSample(
            sample_id="S001",
            patient_id="P001",
            response_status="Complete Response",
            pfs_days=180.5,
            pfs_event=1,
            os_days=365.0,
            os_event=0,
            timepoint="C2D1",
            age=55,
            sex="M",
        )
        assert sample.sample_id == "S001"
        assert sample.patient_id == "P001"
        assert sample.response_status == "CR"  # Normalized
        assert sample.response_group == "responder"  # Derived
        assert sample.pfs_days == 180.5
        assert sample.pfs_event == 1
        assert sample.os_days == 365.0
        assert sample.os_event == 0
        assert sample.timepoint == "C2D1"
        assert sample.cycle == 2  # Parsed
        assert sample.day == 1  # Parsed
        assert sample.age == 55
        assert sample.sex == "M"

    def test_response_status_normalization(self):
        """Response status should be normalized to canonical code."""
        sample = ClinicalSample(sample_id="S001", response_status="partial response")
        assert sample.response_status == "PR"

    def test_response_group_derivation(self):
        """Response group should be derived from response status."""
        cr_sample = ClinicalSample(sample_id="S001", response_status="CR")
        assert cr_sample.response_group == "responder"

        sd_sample = ClinicalSample(sample_id="S002", response_status="SD")
        assert sd_sample.response_group == "non_responder"

        ne_sample = ClinicalSample(sample_id="S003", response_status="NE")
        assert ne_sample.response_group is None

    def test_timepoint_parsing_in_model(self):
        """Timepoint should be parsed to cycle and day."""
        sample = ClinicalSample(sample_id="S001", timepoint="C3D8")
        assert sample.cycle == 3
        assert sample.day == 8

    def test_sex_normalization(self):
        """Sex should be normalized to M/F from explicit string labels only."""
        male_variations = ["M", "m", "Male", "MALE", "Man", "man"]
        for sex in male_variations:
            sample = ClinicalSample(sample_id="S001", sex=sex)
            assert sample.sex == "M", f"Failed for sex='{sex}'"

        female_variations = ["F", "f", "Female", "FEMALE", "Woman", "woman"]
        for sex in female_variations:
            sample = ClinicalSample(sample_id="S001", sex=sex)
            assert sample.sex == "F", f"Failed for sex='{sex}'"

    def test_invalid_sex_returns_none(self):
        """Invalid sex values should be normalized to None."""
        sample = ClinicalSample(sample_id="S001", sex="Unknown")
        assert sample.sex is None

    def test_numeric_sex_values_return_none(self, caplog):
        """Numeric sex values should return None due to conflicting conventions."""
        import logging

        caplog.set_level(logging.WARNING)

        # Numeric values should return None
        for numeric_value in ["0", "1", "2"]:
            sample = ClinicalSample(sample_id="S001", sex=numeric_value)
            assert sample.sex is None, f"Numeric sex value '{numeric_value}' should return None"

        # Check that a warning was logged
        assert any("conflicting conventions" in record.message.lower() for record in caplog.records)

    def test_survival_field_validation(self):
        """Survival fields should enforce constraints."""
        # Valid values
        sample = ClinicalSample(
            sample_id="S001", pfs_days=0, pfs_event=0, os_days=100, os_event=1
        )
        assert sample.pfs_days == 0
        assert sample.pfs_event == 0
        assert sample.os_event == 1

        # Negative days should fail
        with pytest.raises(ValueError):
            ClinicalSample(sample_id="S001", pfs_days=-1)

        # Event > 1 should fail
        with pytest.raises(ValueError):
            ClinicalSample(sample_id="S001", pfs_event=2)

    def test_age_validation(self):
        """Age should be within valid range."""
        valid_sample = ClinicalSample(sample_id="S001", age=65)
        assert valid_sample.age == 65

        # Age > 120 should fail
        with pytest.raises(ValueError):
            ClinicalSample(sample_id="S001", age=150)

        # Negative age should fail
        with pytest.raises(ValueError):
            ClinicalSample(sample_id="S001", age=-1)

    def test_additional_metadata_storage(self):
        """Additional metadata should be stored in additional_metadata field."""
        sample = ClinicalSample.from_dict(
            {
                "sample_id": "S001",
                "response_status": "PR",
                "custom_field_1": "value1",
                "custom_field_2": 42,
            }
        )
        assert sample.sample_id == "S001"
        assert sample.response_status == "PR"
        assert sample.additional_metadata["custom_field_1"] == "value1"
        assert sample.additional_metadata["custom_field_2"] == 42

    def test_to_dict_includes_additional_metadata(self):
        """to_dict should include additional metadata fields."""
        sample = ClinicalSample.from_dict(
            {
                "sample_id": "S001",
                "response_status": "CR",
                "treatment_arm": "A",
            }
        )
        result = sample.to_dict()
        assert result["sample_id"] == "S001"
        assert result["response_status"] == "CR"
        assert result["response_group"] == "responder"
        assert result["treatment_arm"] == "A"

    def test_compute_absolute_day(self):
        """Test compute_absolute_day method."""
        sample = ClinicalSample(sample_id="S001", timepoint="C2D8")

        # With default 21-day cycles
        assert sample.compute_absolute_day() == 29  # (2-1)*21 + 8

        # With custom cycle length
        assert sample.compute_absolute_day(cycle_length_days=28) == 36  # (2-1)*28 + 8

    def test_whitespace_stripping(self):
        """Whitespace should be stripped from string fields."""
        sample = ClinicalSample(
            sample_id="  S001  ",
            patient_id="  P001  ",
        )
        assert sample.sample_id == "S001"
        assert sample.patient_id == "P001"


class TestClinicalSampleEdgeCases:
    """Test edge cases and error handling for ClinicalSample."""

    def test_nan_values_handled(self):
        """NaN values from pandas should be handled gracefully."""
        sample = ClinicalSample(
            sample_id="S001",
            response_status=float("nan"),
            sex=float("nan"),
        )
        assert sample.response_status is None
        assert sample.sex is None

    def test_empty_string_response(self):
        """Empty string response should normalize to None."""
        sample = ClinicalSample(sample_id="S001", response_status="")
        assert sample.response_status is None
        assert sample.response_group is None

    def test_invalid_response_handled(self):
        """Invalid response values should be normalized to None."""
        sample = ClinicalSample(sample_id="S001", response_status="INVALID_CODE")
        assert sample.response_status is None
        assert sample.response_group is None

    def test_baseline_timepoint_parsing(self):
        """Baseline timepoint should parse to cycle=0, day=0."""
        sample = ClinicalSample(sample_id="S001", timepoint="baseline")
        assert sample.cycle == 0
        assert sample.day == 0

    def test_unparseable_timepoint(self):
        """Unparseable timepoint should leave cycle/day as None."""
        sample = ClinicalSample(sample_id="S001", timepoint="random text")
        assert sample.cycle is None
        assert sample.day is None

    def test_explicit_cycle_day_not_overwritten(self):
        """If cycle/day are explicitly set, timepoint parsing shouldn't overwrite."""
        sample = ClinicalSample(
            sample_id="S001",
            timepoint="C1D1",
            cycle=5,  # Explicitly set
            day=10,  # Explicitly set
        )
        # The model_validator only sets cycle/day if they're None
        assert sample.cycle == 5
        assert sample.day == 10

    def test_from_dict_with_none_values(self):
        """from_dict should handle None values gracefully."""
        sample = ClinicalSample.from_dict(
            {
                "sample_id": "S001",
                "patient_id": None,
                "response_status": None,
                "age": None,
            }
        )
        assert sample.sample_id == "S001"
        assert sample.patient_id is None
        assert sample.response_status is None
        assert sample.age is None
