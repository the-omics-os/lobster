"""
Tests for null value handling across the GEO strategy engine.

Covers:
- _sanitize_null_values producing "" not "NA" (GSTR-01)
- _is_null_value helper (GSTR-02)
- PipelineContext.has_file with null values (GSTR-02)
- Strategy rules rejecting null values (GSTR-02)
- data_availability with all-null configs (GSTR-02)
- _derive_analysis with "NA" values (GSTR-02)
- geo_service.py null guards (GSTR-02)
"""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# TestNullSanitization -- _sanitize_null_values producing "" not "NA"
# ---------------------------------------------------------------------------


class TestNullSanitization:
    """Tests for DataExpertAssistant._sanitize_null_values (GSTR-01)."""

    def _make_assistant(self):
        """Create a DataExpertAssistant without LLM initialization."""
        with patch(
            "lobster.agents.data_expert.assistant.get_settings"
        ) as mock_settings:
            mock_settings.return_value = MagicMock()
            from lobster.agents.data_expert.assistant import DataExpertAssistant

            return DataExpertAssistant()

    def test_none_becomes_empty_string(self):
        assistant = self._make_assistant()
        result = assistant._sanitize_null_values({"key": None})
        assert result["key"] == "", f"Expected '' but got {result['key']!r}"

    def test_null_string_becomes_empty_string(self):
        assistant = self._make_assistant()
        result = assistant._sanitize_null_values({"key": "null"})
        assert result["key"] == "", f"Expected '' but got {result['key']!r}"

    def test_na_string_becomes_empty_string(self):
        assistant = self._make_assistant()
        result = assistant._sanitize_null_values({"key": "NA"})
        assert result["key"] == "", f"Expected '' but got {result['key']!r}"

    def test_none_string_becomes_empty_string(self):
        assistant = self._make_assistant()
        result = assistant._sanitize_null_values({"key": "None"})
        assert result["key"] == "", f"Expected '' but got {result['key']!r}"

    def test_n_a_string_becomes_empty_string(self):
        assistant = self._make_assistant()
        result = assistant._sanitize_null_values({"key": "n/a"})
        assert result["key"] == "", f"Expected '' but got {result['key']!r}"

    def test_empty_string_preserved(self):
        assistant = self._make_assistant()
        result = assistant._sanitize_null_values({"key": ""})
        assert result["key"] == "", f"Expected '' but got {result['key']!r}"

    def test_raw_data_available_none_becomes_false(self):
        assistant = self._make_assistant()
        result = assistant._sanitize_null_values({"raw_data_available": None})
        assert result["raw_data_available"] is False

    def test_real_value_preserved(self):
        assistant = self._make_assistant()
        result = assistant._sanitize_null_values({"key": "real_value"})
        assert result["key"] == "real_value"


# ---------------------------------------------------------------------------
# TestIsNullValue -- _is_null_value helper
# ---------------------------------------------------------------------------


class TestIsNullValue:
    """Tests for _is_null_value() helper (GSTR-02)."""

    def test_none_is_null(self):
        from lobster.services.data_access.geo.strategy import _is_null_value

        assert _is_null_value(None) is True

    def test_empty_string_is_null(self):
        from lobster.services.data_access.geo.strategy import _is_null_value

        assert _is_null_value("") is True

    def test_na_is_null(self):
        from lobster.services.data_access.geo.strategy import _is_null_value

        assert _is_null_value("NA") is True

    def test_null_string_is_null(self):
        from lobster.services.data_access.geo.strategy import _is_null_value

        assert _is_null_value("null") is True

    def test_none_string_is_null(self):
        from lobster.services.data_access.geo.strategy import _is_null_value

        assert _is_null_value("None") is True

    def test_n_a_is_null(self):
        from lobster.services.data_access.geo.strategy import _is_null_value

        assert _is_null_value("n/a") is True

    def test_whitespace_na_is_null(self):
        from lobster.services.data_access.geo.strategy import _is_null_value

        assert _is_null_value("  NA  ") is True

    def test_real_file_is_not_null(self):
        from lobster.services.data_access.geo.strategy import _is_null_value

        assert _is_null_value("real_file.txt") is False

    def test_bool_false_is_not_null(self):
        from lobster.services.data_access.geo.strategy import _is_null_value

        assert _is_null_value(False) is False

    def test_numeric_zero_is_not_null(self):
        from lobster.services.data_access.geo.strategy import _is_null_value

        assert _is_null_value(0) is False


# ---------------------------------------------------------------------------
# TestPipelineContext -- has_file and get_file_info with null values
# ---------------------------------------------------------------------------


class TestPipelineContext:
    """Tests for PipelineContext.has_file with null values (GSTR-02)."""

    def test_has_file_false_for_na_name(self):
        from lobster.services.data_access.geo.strategy import PipelineContext

        ctx = PipelineContext(
            geo_id="GSE000",
            strategy_config={"processed_matrix_name": "NA"},
            metadata={},
        )
        assert ctx.has_file("processed_matrix") is False

    def test_has_file_false_for_empty_name(self):
        from lobster.services.data_access.geo.strategy import PipelineContext

        ctx = PipelineContext(
            geo_id="GSE000",
            strategy_config={"processed_matrix_name": ""},
            metadata={},
        )
        assert ctx.has_file("processed_matrix") is False

    def test_has_file_false_for_none_name(self):
        from lobster.services.data_access.geo.strategy import PipelineContext

        ctx = PipelineContext(
            geo_id="GSE000",
            strategy_config={"processed_matrix_name": None},
            metadata={},
        )
        assert ctx.has_file("processed_matrix") is False

    def test_has_file_true_for_real_name(self):
        from lobster.services.data_access.geo.strategy import PipelineContext

        ctx = PipelineContext(
            geo_id="GSE000",
            strategy_config={"processed_matrix_name": "GSE131907_matrix"},
            metadata={},
        )
        assert ctx.has_file("processed_matrix") is True

    def test_get_file_info_empty_for_null_name(self):
        from lobster.services.data_access.geo.strategy import PipelineContext

        ctx = PipelineContext(
            geo_id="GSE000",
            strategy_config={
                "processed_matrix_name": "NA",
                "processed_matrix_filetype": "txt",
            },
            metadata={},
        )
        name, filetype = ctx.get_file_info("processed_matrix")
        assert name == "", f"Expected '' but got {name!r}"
        assert filetype == "", f"Expected '' but got {filetype!r}"


# ---------------------------------------------------------------------------
# TestRulesWithNulls -- Rules rejecting null values + FILETYPES constants
# ---------------------------------------------------------------------------


class TestRulesWithNulls:
    """Tests for strategy rules rejecting null values and using constants (GSTR-02)."""

    def test_processed_matrix_rule_rejects_na_name(self):
        from lobster.services.data_access.geo.strategy import (
            PipelineContext,
            ProcessedMatrixRule,
        )

        ctx = PipelineContext(
            geo_id="GSE000",
            strategy_config={
                "processed_matrix_name": "NA",
                "processed_matrix_filetype": "txt",
            },
            metadata={},
        )
        rule = ProcessedMatrixRule()
        assert rule.evaluate(ctx) is None

    def test_raw_matrix_rule_rejects_na_name(self):
        from lobster.services.data_access.geo.strategy import (
            PipelineContext,
            RawMatrixRule,
        )

        ctx = PipelineContext(
            geo_id="GSE000",
            strategy_config={
                "raw_UMI_like_matrix_name": "NA",
                "raw_UMI_like_matrix_filetype": "csv",
            },
            metadata={},
        )
        rule = RawMatrixRule()
        assert rule.evaluate(ctx) is None

    def test_matrix_filetypes_constant_exists(self):
        from lobster.services.data_access.geo.strategy import MATRIX_FILETYPES

        assert isinstance(MATRIX_FILETYPES, frozenset)
        assert "txt" in MATRIX_FILETYPES
        assert "csv" in MATRIX_FILETYPES
        assert "tsv" in MATRIX_FILETYPES
        assert "h5" in MATRIX_FILETYPES
        assert "h5ad" in MATRIX_FILETYPES

    def test_raw_matrix_filetypes_constant_exists(self):
        from lobster.services.data_access.geo.strategy import RAW_MATRIX_FILETYPES

        assert isinstance(RAW_MATRIX_FILETYPES, frozenset)
        assert "txt" in RAW_MATRIX_FILETYPES
        assert "csv" in RAW_MATRIX_FILETYPES
        assert "tsv" in RAW_MATRIX_FILETYPES
        assert "mtx" in RAW_MATRIX_FILETYPES
        assert "h5" in RAW_MATRIX_FILETYPES

    def test_h5_filetypes_constant_exists(self):
        from lobster.services.data_access.geo.strategy import H5_FILETYPES

        assert isinstance(H5_FILETYPES, frozenset)
        assert "h5" in H5_FILETYPES
        assert "h5ad" in H5_FILETYPES

    def test_h5_format_rule_uses_constant(self):
        """H5FormatRule should use H5_FILETYPES constant."""
        from lobster.services.data_access.geo.strategy import (
            H5_FILETYPES,
            PipelineContext,
            PipelineType,
        )

        # Valid H5 filetype should match
        ctx = PipelineContext(
            geo_id="GSE000",
            strategy_config={
                "processed_matrix_name": "matrix",
                "processed_matrix_filetype": "h5ad",
            },
            metadata={},
        )
        from lobster.services.data_access.geo.strategy import H5FormatRule

        rule = H5FormatRule()
        result = rule.evaluate(ctx)
        assert result == PipelineType.H5_FIRST


# ---------------------------------------------------------------------------
# TestDataAvailability -- data_availability returning NONE with all-null
# ---------------------------------------------------------------------------


class TestDataAvailability:
    """Tests for data_availability returning NONE with all-null configs (GSTR-02)."""

    def test_all_na_returns_none(self):
        from lobster.services.data_access.geo.strategy import (
            DataAvailability,
            PipelineContext,
        )

        ctx = PipelineContext(
            geo_id="GSE000",
            strategy_config={
                "processed_matrix_name": "NA",
                "raw_UMI_like_matrix_name": "NA",
                "summary_file_name": "NA",
                "cell_annotation_name": "NA",
                "raw_data_available": False,
            },
            metadata={},
        )
        assert ctx.data_availability == DataAvailability.NONE

    def test_partial_with_real_processed(self):
        from lobster.services.data_access.geo.strategy import (
            DataAvailability,
            PipelineContext,
        )

        ctx = PipelineContext(
            geo_id="GSE000",
            strategy_config={
                "processed_matrix_name": "GSE131907_matrix",
                "raw_UMI_like_matrix_name": "NA",
                "summary_file_name": "NA",
                "cell_annotation_name": "NA",
                "raw_data_available": False,
            },
            metadata={},
        )
        assert ctx.data_availability == DataAvailability.PARTIAL

    def test_raw_data_na_string_treated_as_false(self):
        """raw_data_available set to string 'NA' should be treated as falsy."""
        from lobster.services.data_access.geo.strategy import (
            DataAvailability,
            PipelineContext,
        )

        ctx = PipelineContext(
            geo_id="GSE000",
            strategy_config={
                "processed_matrix_name": "NA",
                "raw_UMI_like_matrix_name": "NA",
                "summary_file_name": "NA",
                "cell_annotation_name": "NA",
                "raw_data_available": "NA",
            },
            metadata={},
        )
        assert ctx.data_availability == DataAvailability.NONE


# ---------------------------------------------------------------------------
# TestDeriveAnalysis -- _derive_analysis with "NA" values
# ---------------------------------------------------------------------------


class TestDeriveAnalysis:
    """Tests for GEOQueuePreparer._derive_analysis with 'NA' values (GSTR-02)."""

    def _make_url_data(self):
        """Create a minimal mock DownloadUrlResult."""
        url_data = MagicMock()
        url_data.h5_url = None
        url_data.primary_files = []
        return url_data

    def test_na_processed_matrix_is_false(self):
        from lobster.services.data_access.geo_queue_preparer import GEOQueuePreparer

        url_data = self._make_url_data()
        result = GEOQueuePreparer._derive_analysis(
            {"processed_matrix_name": "NA", "raw_UMI_like_matrix_name": "", "raw_data_available": False},
            url_data,
        )
        assert result["has_processed_matrix"] is False

    def test_na_raw_matrix_is_false(self):
        from lobster.services.data_access.geo_queue_preparer import GEOQueuePreparer

        url_data = self._make_url_data()
        result = GEOQueuePreparer._derive_analysis(
            {"processed_matrix_name": "", "raw_UMI_like_matrix_name": "NA", "raw_data_available": False},
            url_data,
        )
        assert result["has_raw_matrix"] is False

    def test_na_raw_data_available_is_false(self):
        from lobster.services.data_access.geo_queue_preparer import GEOQueuePreparer

        url_data = self._make_url_data()
        result = GEOQueuePreparer._derive_analysis(
            {"processed_matrix_name": "", "raw_UMI_like_matrix_name": "", "raw_data_available": "NA"},
            url_data,
        )
        assert result["raw_data_available"] is False


# ---------------------------------------------------------------------------
# TestPipelineStepNullGuards -- geo_service null guards (mocked)
# ---------------------------------------------------------------------------


class TestPipelineStepNullGuards:
    """Tests for geo_service.py null guards rejecting 'NA' matrix names (GSTR-02)."""

    def _make_geo_service(self):
        """Create a GEOService with mocked data_manager."""
        with patch("lobster.services.data_access.geo_service.GEOService.__init__", return_value=None):
            from lobster.services.data_access.geo_service import GEOService

            svc = GEOService.__new__(GEOService)
            svc.data_manager = MagicMock()
            return svc

    def test_try_processed_rejects_na_matrix_name(self):
        svc = self._make_geo_service()
        svc.data_manager.metadata_store = {
            "GSE000": {
                "strategy_config": {
                    "processed_matrix_name": "NA",
                    "processed_matrix_filetype": "txt",
                }
            }
        }
        result = svc._try_processed_matrix_first("GSE000", {})
        assert result.success is False

    def test_try_raw_rejects_na_matrix_name(self):
        svc = self._make_geo_service()
        svc.data_manager.metadata_store = {
            "GSE000": {
                "strategy_config": {
                    "raw_UMI_like_matrix_name": "NA",
                    "raw_UMI_like_matrix_filetype": "csv",
                }
            }
        }
        result = svc._try_raw_matrix_first("GSE000", {})
        assert result.success is False
