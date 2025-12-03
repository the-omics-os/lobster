"""Unit tests for DoclingService helpers."""

import json
from pathlib import Path

from lobster.services.data_access.docling_service import DoclingService


class DummyDoc:
    def __init__(self, path: Path):
        self._path = path

    def model_dump(self):
        return {"path": self._path}


def test_cache_document_serializes_path(tmp_path):
    """_cache_document should serialize Path objects safely."""

    service = object.__new__(DoclingService)
    service.converter = object()  # Pretend Docling is available
    service.cache_dir = tmp_path
    service.data_manager = None

    doc = DummyDoc(Path("foo/bar"))

    service._cache_document("identifier", doc)

    cache_file = service._cache_path_for("identifier")
    assert cache_file.exists()

    with open(cache_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    assert data["path"] == "foo/bar"


# ============================================================================
# Negative Caching Tests (Bug Fix: Cache HTTP failures)
# ============================================================================


def test_cache_failure_writes_correct_structure(tmp_path):
    """Verify _cache_failure writes properly structured failure cache."""
    service = object.__new__(DoclingService)
    service.converter = object()  # Pretend Docling is available
    service.cache_dir = tmp_path
    service.data_manager = None

    # Cache a failure
    service._cache_failure(
        source="https://paywalled.com/paper.pdf",
        error_type="HTTP_403",
        error_msg="403 Client Error: Forbidden for url: https://paywalled.com/paper.pdf",
        ttl_hours=24,
    )

    # Verify cache file exists
    cache_file = service._cache_path_for("https://paywalled.com/paper.pdf")
    assert cache_file.exists()

    # Verify structure
    with open(cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["cache_type"] == "failure"
    assert data["error"] == "HTTP_403"
    assert "403" in data["message"]
    assert "cached_at" in data
    assert data["ttl_hours"] == 24


def test_get_cached_document_raises_on_failure_cache(tmp_path):
    """Verify _get_cached_document raises exception when failure cache is found."""
    from lobster.services.data_access.docling_service import PDFExtractionError

    service = object.__new__(DoclingService)
    service.converter = object()
    service.cache_dir = tmp_path
    service.data_manager = None

    # Manually create failure cache
    cache_file = service._cache_path_for("https://paywalled.com/paper.pdf")
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    from datetime import datetime

    failure_data = {
        "cache_type": "failure",
        "error": "HTTP_403",
        "message": "403 Forbidden",
        "cached_at": datetime.now().isoformat(),
        "ttl_hours": 24,
    }

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(failure_data, f)

    # Try to get cached document - should raise with failure message
    import pytest

    with pytest.raises(PDFExtractionError, match="Cached failure"):
        service._get_cached_document("https://paywalled.com/paper.pdf")


def test_get_cached_document_retries_on_expired_failure_cache(tmp_path):
    """Verify expired failure cache is purged and returns None for retry."""
    service = object.__new__(DoclingService)
    service.converter = object()
    service.cache_dir = tmp_path
    service.data_manager = None

    # Create expired failure cache (25 hours old, TTL is 24 hours)
    cache_file = service._cache_path_for("https://paywalled.com/paper.pdf")
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    from datetime import datetime, timedelta

    old_timestamp = (datetime.now() - timedelta(hours=25)).isoformat()
    failure_data = {
        "cache_type": "failure",
        "error": "HTTP_403",
        "message": "403 Forbidden",
        "cached_at": old_timestamp,
        "ttl_hours": 24,
    }

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(failure_data, f)

    # Try to get cached document - should return None (cache expired, retry allowed)
    result = service._get_cached_document("https://paywalled.com/paper.pdf")
    assert result is None

    # Verify cache file was purged
    assert not cache_file.exists()


def test_cache_failure_truncates_long_error_messages(tmp_path):
    """Verify long error messages are truncated to 500 chars."""
    service = object.__new__(DoclingService)
    service.converter = object()
    service.cache_dir = tmp_path
    service.data_manager = None

    # Create long error message (>500 chars)
    long_error = "A" * 1000

    service._cache_failure(
        source="https://example.com/paper.pdf",
        error_type="UnicodeDecodeError",
        error_msg=long_error,
        ttl_hours=168,
    )

    cache_file = service._cache_path_for("https://example.com/paper.pdf")
    with open(cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Message should be truncated to 500 chars
    assert len(data["message"]) == 500
    assert data["message"] == "A" * 500
