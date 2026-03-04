"""
Tests for typed RetryResult replacing string sentinel values in GEO service.

Verifies that _retry_with_backoff returns RetryResult objects with proper
RetryOutcome enum values, and that no string comparisons remain.
"""

import ast
import inspect
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from lobster.services.data_access.geo_service import (
    GEOService,
    RetryOutcome,
    RetryResult,
)


# ---------------------------------------------------------------------------
# RetryOutcome enum tests
# ---------------------------------------------------------------------------


class TestRetryOutcome:
    """Test RetryOutcome enum has required values."""

    def test_has_success(self):
        assert hasattr(RetryOutcome, "SUCCESS")

    def test_has_exhausted(self):
        assert hasattr(RetryOutcome, "EXHAUSTED")

    def test_has_soft_file_missing(self):
        assert hasattr(RetryOutcome, "SOFT_FILE_MISSING")

    def test_enum_values_are_strings(self):
        """Enum values should be lowercase string identifiers."""
        assert RetryOutcome.SUCCESS.value == "success"
        assert RetryOutcome.EXHAUSTED.value == "exhausted"
        assert RetryOutcome.SOFT_FILE_MISSING.value == "soft_file_missing"


# ---------------------------------------------------------------------------
# RetryResult property tests
# ---------------------------------------------------------------------------


class TestRetryResult:
    """Test RetryResult dataclass properties."""

    def test_succeeded_true_for_success(self):
        r = RetryResult(outcome=RetryOutcome.SUCCESS, value="some_data")
        assert r.succeeded is True

    def test_succeeded_false_for_exhausted(self):
        r = RetryResult(outcome=RetryOutcome.EXHAUSTED)
        assert r.succeeded is False

    def test_succeeded_false_for_soft_file_missing(self):
        r = RetryResult(outcome=RetryOutcome.SOFT_FILE_MISSING)
        assert r.succeeded is False

    def test_needs_fallback_true_for_soft_file_missing(self):
        r = RetryResult(outcome=RetryOutcome.SOFT_FILE_MISSING)
        assert r.needs_fallback is True

    def test_needs_fallback_false_for_success(self):
        r = RetryResult(outcome=RetryOutcome.SUCCESS, value="data")
        assert r.needs_fallback is False

    def test_needs_fallback_false_for_exhausted(self):
        r = RetryResult(outcome=RetryOutcome.EXHAUSTED)
        assert r.needs_fallback is False

    def test_success_carries_value(self):
        r = RetryResult(outcome=RetryOutcome.SUCCESS, value={"key": "val"})
        assert r.value == {"key": "val"}

    def test_exhausted_has_none_value(self):
        r = RetryResult(outcome=RetryOutcome.EXHAUSTED)
        assert r.value is None

    def test_retries_used_tracking(self):
        r = RetryResult(outcome=RetryOutcome.EXHAUSTED, retries_used=3)
        assert r.retries_used == 3


# ---------------------------------------------------------------------------
# _retry_with_backoff integration tests (with mocked operations)
# ---------------------------------------------------------------------------


class TestRetryWithBackoffReturnTypes:
    """Test that _retry_with_backoff returns RetryResult objects."""

    @pytest.fixture
    def geo_service(self):
        """Create a minimal GEOService with mocked data_manager."""
        dm = MagicMock()
        dm.metadata_store = {}
        dm.workspace_dir = "/tmp/test_geo"
        svc = GEOService(data_manager=dm)
        return svc

    def test_success_returns_retry_result(self, geo_service):
        """Successful operation returns RetryResult(SUCCESS, value=result)."""
        result = geo_service._retry_with_backoff(
            operation=lambda: "success_data",
            operation_name="test_op",
            max_retries=1,
        )
        assert isinstance(result, RetryResult)
        assert result.outcome == RetryOutcome.SUCCESS
        assert result.succeeded is True
        assert result.value == "success_data"

    def test_oserror_missing_file_returns_soft_file_missing(self, geo_service):
        """OSError with 'No such file' returns RetryResult(SOFT_FILE_MISSING)."""

        def raise_os_error():
            raise OSError("Download failed: No such file or directory")

        result = geo_service._retry_with_backoff(
            operation=raise_os_error,
            operation_name="test_op",
            max_retries=3,
            base_delay=0.01,
        )
        assert isinstance(result, RetryResult)
        assert result.outcome == RetryOutcome.SOFT_FILE_MISSING
        assert result.needs_fallback is True

    @patch("time.sleep")
    def test_exhausted_returns_retry_result(self, mock_sleep, geo_service):
        """Repeated failures return RetryResult(EXHAUSTED)."""
        call_count = 0

        def always_fail():
            nonlocal call_count
            call_count += 1
            raise Exception("transient error")

        result = geo_service._retry_with_backoff(
            operation=always_fail,
            operation_name="test_op",
            max_retries=2,
            base_delay=0.01,
        )
        assert isinstance(result, RetryResult)
        assert result.outcome == RetryOutcome.EXHAUSTED
        assert result.succeeded is False
        assert result.needs_fallback is False

    def test_ftp_550_returns_soft_file_missing(self, geo_service):
        """FTP error 550 (file not found) returns RetryResult(SOFT_FILE_MISSING)."""
        import ftplib

        def raise_ftp_550():
            raise ftplib.error_perm("550 File not found")

        result = geo_service._retry_with_backoff(
            operation=raise_ftp_550,
            operation_name="test_op",
            max_retries=3,
            base_delay=0.01,
            is_ftp=True,
        )
        assert isinstance(result, RetryResult)
        assert result.outcome == RetryOutcome.SOFT_FILE_MISSING
        assert result.needs_fallback is True

    @patch("time.sleep")
    def test_http_error_exhausted_returns_retry_result(self, mock_sleep, geo_service):
        """HTTP 500 errors that exhaust retries return RetryResult(EXHAUSTED)."""
        import requests

        def raise_http_500():
            resp = MagicMock()
            resp.status_code = 500
            raise requests.exceptions.HTTPError(response=resp)

        result = geo_service._retry_with_backoff(
            operation=raise_http_500,
            operation_name="test_op",
            max_retries=2,
            base_delay=0.01,
        )
        assert isinstance(result, RetryResult)
        assert result.outcome == RetryOutcome.EXHAUSTED


# ---------------------------------------------------------------------------
# String sentinel elimination test
# ---------------------------------------------------------------------------


class TestNoStringSentinels:
    """Verify no string sentinel comparisons remain in geo_service.py."""

    def test_no_soft_file_missing_string_comparisons(self):
        """The string literal 'SOFT_FILE_MISSING' must not appear as a comparison
        or return value in geo_service.py. Only the enum definition is allowed."""
        geo_service_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "lobster",
            "services",
            "data_access",
            "geo_service.py",
        )
        geo_service_path = os.path.normpath(geo_service_path)

        with open(geo_service_path) as f:
            source = f.read()

        # Parse AST to find string literal usages
        tree = ast.parse(source)

        string_sentinel_uses = []
        for node in ast.walk(tree):
            # Check for string literal "SOFT_FILE_MISSING" used in comparisons or returns
            if isinstance(node, ast.Constant) and node.value == "SOFT_FILE_MISSING":
                # This is a raw string usage -- the only allowed one is in the enum value
                string_sentinel_uses.append(node.lineno)
            # Also check for string comparisons: x == "SOFT_FILE_MISSING"
            if isinstance(node, ast.Compare):
                for comparator in node.comparators:
                    if (
                        isinstance(comparator, ast.Constant)
                        and comparator.value == "SOFT_FILE_MISSING"
                    ):
                        string_sentinel_uses.append(comparator.lineno)

        # Filter out the enum definition line (which legitimately uses the string
        # as the enum value: SOFT_FILE_MISSING = "soft_file_missing")
        # The enum value is "soft_file_missing" (lowercase), so "SOFT_FILE_MISSING"
        # as a string constant should appear ZERO times.
        assert len(string_sentinel_uses) == 0, (
            f"Found 'SOFT_FILE_MISSING' string literal at lines: {string_sentinel_uses}. "
            f"All uses should be through RetryOutcome enum, not string comparisons."
        )

    def test_no_return_string_soft_file_missing(self):
        """Ensure _retry_with_backoff never returns a plain string."""
        geo_service_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "lobster",
            "services",
            "data_access",
            "geo_service.py",
        )
        geo_service_path = os.path.normpath(geo_service_path)

        with open(geo_service_path) as f:
            source = f.read()

        # Ensure no return "SOFT_FILE_MISSING" statements
        assert 'return "SOFT_FILE_MISSING"' not in source, (
            "Found 'return \"SOFT_FILE_MISSING\"' in geo_service.py. "
            "Must use RetryResult(RetryOutcome.SOFT_FILE_MISSING) instead."
        )
