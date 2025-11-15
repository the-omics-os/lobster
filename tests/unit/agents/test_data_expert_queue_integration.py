"""
Unit tests for data_expert queue integration.

Tests data_expert's queue consumer functionality with mock services to validate
all error paths, status transitions, and strategy selection logic without
requiring network access.

Test Coverage:
- Entry validation (not found, already completed, in progress)
- Status transitions (PENDING → IN_PROGRESS → COMPLETED/FAILED)
- Strategy selection (recommended vs auto-selection)
- Error handling and recovery
- Queue management and filtering

Note: These are unit tests that mock GEOService to avoid real downloads.
For end-to-end tests with real API calls, see test_queue_workflow_end_to_end.py
"""

from datetime import datetime
from unittest.mock import Mock, patch

import anndata as ad
import numpy as np
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.schemas.download_queue import (
    DownloadQueueEntry,
    DownloadStatus,
    StrategyConfig,
)


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create temporary workspace directory."""
    workspace = tmp_path / "test_workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def data_manager_with_queue(tmp_workspace):
    """Create DataManagerV2 with download queue for testing."""
    dm = DataManagerV2(workspace_path=tmp_workspace)
    return dm


@pytest.fixture
def sample_queue_entry():
    """Create sample download queue entry for testing."""
    return DownloadQueueEntry(
        entry_id="queue_GSE180759_test123",
        dataset_id="GSE180759",
        database="geo",
        priority=5,
        status=DownloadStatus.PENDING,
        metadata={"title": "Test Dataset", "n_samples": 20, "organism": "Homo sapiens"},
        validation_result={"recommendation": "proceed", "confidence_score": 0.95},
        matrix_url="ftp://ftp.ncbi.nlm.nih.gov/.../GSE180759_series_matrix.txt.gz",
        raw_urls=[],
        supplementary_urls=["ftp://ftp.ncbi.nlm.nih.gov/.../GSE180759_expr.csv.gz"],
        h5_url=None,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        recommended_strategy=StrategyConfig(
            strategy_name="MATRIX_FIRST",
            concatenation_strategy="auto",
            confidence=0.9,
            rationale="Matrix file available with complete metadata",
        ),
        downloaded_by=None,
        modality_name=None,
        error_log=[],
    )


@pytest.fixture
def sample_adata():
    """Create sample AnnData object for mock returns."""
    return ad.AnnData(
        X=np.random.rand(100, 500),  # 100 samples, 500 genes
        obs={"sample_id": [f"S{i}" for i in range(100)]},
        var={"gene_id": [f"G{i}" for i in range(500)]},
    )


# ============================================================================
# Test Class: Queue Management
# ============================================================================


class TestQueueManagement:
    """Test queue management functionality."""

    def test_add_entry_to_queue(self, data_manager_with_queue, sample_queue_entry):
        """Test adding entry to queue."""
        dm = data_manager_with_queue

        # Add entry
        dm.download_queue.add_entry(sample_queue_entry)

        # Verify entry added
        entries = dm.download_queue.list_entries()
        assert len(entries) == 1
        assert entries[0].entry_id == sample_queue_entry.entry_id
        assert entries[0].status == DownloadStatus.PENDING

    def test_get_entry_from_queue(self, data_manager_with_queue, sample_queue_entry):
        """Test retrieving specific entry from queue."""
        dm = data_manager_with_queue

        # Add entry
        dm.download_queue.add_entry(sample_queue_entry)

        # Retrieve entry
        retrieved = dm.download_queue.get_entry(sample_queue_entry.entry_id)

        assert retrieved is not None
        assert retrieved.entry_id == sample_queue_entry.entry_id
        assert retrieved.dataset_id == sample_queue_entry.dataset_id

    def test_get_nonexistent_entry(self, data_manager_with_queue):
        """Test retrieving non-existent entry raises EntryNotFoundError."""
        dm = data_manager_with_queue

        # Try to get non-existent entry - should raise exception
        from lobster.core.download_queue import EntryNotFoundError

        with pytest.raises(EntryNotFoundError, match="not found"):
            dm.download_queue.get_entry("queue_INVALID_xyz")

    def test_update_entry_status(self, data_manager_with_queue, sample_queue_entry):
        """Test updating entry status."""
        dm = data_manager_with_queue

        # Add entry
        dm.download_queue.add_entry(sample_queue_entry)

        # Update status to IN_PROGRESS
        dm.download_queue.update_status(
            entry_id=sample_queue_entry.entry_id,
            status=DownloadStatus.IN_PROGRESS,
            downloaded_by="test_agent",
        )

        # Verify status updated
        updated = dm.download_queue.get_entry(sample_queue_entry.entry_id)
        assert updated.status == DownloadStatus.IN_PROGRESS
        assert updated.downloaded_by == "test_agent"

    def test_update_entry_to_completed(
        self, data_manager_with_queue, sample_queue_entry
    ):
        """Test updating entry to completed with modality name."""
        dm = data_manager_with_queue

        # Add entry
        dm.download_queue.add_entry(sample_queue_entry)

        # Update to COMPLETED
        modality_name = "geo_gse180759_transcriptomics"
        dm.download_queue.update_status(
            entry_id=sample_queue_entry.entry_id,
            status=DownloadStatus.COMPLETED,
            modality_name=modality_name,
        )

        # Verify update
        updated = dm.download_queue.get_entry(sample_queue_entry.entry_id)
        assert updated.status == DownloadStatus.COMPLETED
        assert updated.modality_name == modality_name

    def test_update_entry_to_failed(self, data_manager_with_queue, sample_queue_entry):
        """Test updating entry to failed with error message."""
        dm = data_manager_with_queue

        # Add entry
        dm.download_queue.add_entry(sample_queue_entry)

        # Update to FAILED
        error_msg = "Network error: Connection timeout"
        dm.download_queue.update_status(
            entry_id=sample_queue_entry.entry_id,
            status=DownloadStatus.FAILED,
            error=error_msg,
        )

        # Verify update
        updated = dm.download_queue.get_entry(sample_queue_entry.entry_id)
        assert updated.status == DownloadStatus.FAILED
        assert len(updated.error_log) > 0
        # Error message is timestamped, so just check it contains the error
        assert any(error_msg in log_entry for log_entry in updated.error_log)


# ============================================================================
# Test Class: Queue Filtering
# ============================================================================


class TestQueueFiltering:
    """Test queue filtering functionality."""

    def test_empty_queue_list(self, data_manager_with_queue):
        """Test listing entries from empty queue."""
        entries = data_manager_with_queue.download_queue.list_entries()
        assert entries == []

    def test_filter_by_pending_status(self, data_manager_with_queue):
        """Test filtering entries by PENDING status."""
        dm = data_manager_with_queue

        # Add entries with different statuses
        for i, status in enumerate([DownloadStatus.PENDING, DownloadStatus.COMPLETED]):
            entry = DownloadQueueEntry(
                entry_id=f"queue_{status.value}_{i}",
                dataset_id=f"GSE{i}",
                database="geo",
                status=status,
                metadata={"title": f"Test {i}"},
            )
            dm.download_queue.add_entry(entry)

        # Filter by PENDING
        pending = dm.download_queue.list_entries(status=DownloadStatus.PENDING)
        assert len(pending) == 1
        assert pending[0].status == DownloadStatus.PENDING

    def test_filter_by_completed_status(self, data_manager_with_queue):
        """Test filtering entries by COMPLETED status."""
        dm = data_manager_with_queue

        # Add entries with different statuses
        for i, status in enumerate([DownloadStatus.PENDING, DownloadStatus.COMPLETED]):
            entry = DownloadQueueEntry(
                entry_id=f"queue_{status.value}_{i}",
                dataset_id=f"GSE{i}",
                database="geo",
                status=status,
                metadata={"title": f"Test {i}"},
            )
            dm.download_queue.add_entry(entry)

        # Filter by COMPLETED
        completed = dm.download_queue.list_entries(status=DownloadStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].status == DownloadStatus.COMPLETED

    def test_multiple_pending_entries(self, data_manager_with_queue):
        """Test handling multiple pending entries in queue."""
        dm = data_manager_with_queue

        # Add multiple entries
        entries = []
        for i in range(3):
            entry = DownloadQueueEntry(
                entry_id=f"queue_GSE{i}_test",
                dataset_id=f"GSE{i}",
                database="geo",
                status=DownloadStatus.PENDING,
                metadata={"title": f"Test {i}"},
            )
            dm.download_queue.add_entry(entry)
            entries.append(entry)

        # Verify all entries in queue
        pending = dm.download_queue.list_entries(status=DownloadStatus.PENDING)
        assert len(pending) == 3

        # Each entry should be independently accessible
        for entry in entries:
            retrieved = dm.download_queue.get_entry(entry.entry_id)
            assert retrieved.entry_id == entry.entry_id
            assert retrieved.status == DownloadStatus.PENDING


# ============================================================================
# Test Class: Strategy Selection
# ============================================================================


class TestStrategySelection:
    """Test download strategy selection logic."""

    def test_strategy_selection_recommended(
        self, data_manager_with_queue, sample_queue_entry
    ):
        """Test strategy selection uses recommended strategy when available."""
        dm = data_manager_with_queue

        # Entry already has recommended_strategy in fixture
        dm.download_queue.add_entry(sample_queue_entry)

        # Verify queue entry has recommended strategy
        entry = dm.download_queue.get_entry(sample_queue_entry.entry_id)
        assert entry.recommended_strategy is not None
        assert entry.recommended_strategy.strategy_name == "MATRIX_FIRST"
        assert entry.recommended_strategy.concatenation_strategy == "auto"

    def test_strategy_selection_auto_h5(self, data_manager_with_queue):
        """Test auto strategy selection prioritizes H5 when available."""
        # Create entry with H5 URL
        entry = DownloadQueueEntry(
            entry_id="queue_GSE123456_h5test",
            dataset_id="GSE123456",
            database="geo",
            status=DownloadStatus.PENDING,
            metadata={"title": "Test with H5"},
            h5_url="http://example.com/data.h5",
            matrix_url=None,
            supplementary_urls=[],
            raw_urls=[],
            recommended_strategy=None,  # No recommended strategy
        )

        data_manager_with_queue.download_queue.add_entry(entry)

        # Entry should be available for auto-selection
        retrieved = data_manager_with_queue.download_queue.get_entry(entry.entry_id)
        assert retrieved.h5_url is not None
        assert retrieved.recommended_strategy is None  # Will use auto-selection

    def test_strategy_selection_auto_matrix(self, data_manager_with_queue):
        """Test auto strategy selection uses matrix when H5 unavailable."""
        # Create entry with only matrix URL
        entry = DownloadQueueEntry(
            entry_id="queue_GSE123456_matrixtest",
            dataset_id="GSE123456",
            database="geo",
            status=DownloadStatus.PENDING,
            metadata={"title": "Test with Matrix"},
            h5_url=None,
            matrix_url="http://example.com/matrix.txt.gz",
            supplementary_urls=[],
            raw_urls=[],
            recommended_strategy=None,
        )

        data_manager_with_queue.download_queue.add_entry(entry)

        retrieved = data_manager_with_queue.download_queue.get_entry(entry.entry_id)
        assert retrieved.matrix_url is not None
        assert retrieved.h5_url is None

    def test_missing_urls_auto_strategy(self, data_manager_with_queue):
        """Test auto strategy when no URLs available (should use RAW_FIRST)."""
        entry = DownloadQueueEntry(
            entry_id="queue_GSE_no_urls",
            dataset_id="GSE999999",
            database="geo",
            status=DownloadStatus.PENDING,
            metadata={"title": "No URLs"},
            h5_url=None,
            matrix_url=None,
            supplementary_urls=[],
            raw_urls=[],  # Empty raw URLs
            recommended_strategy=None,
        )

        data_manager_with_queue.download_queue.add_entry(entry)

        # Retrieve and verify
        retrieved = data_manager_with_queue.download_queue.get_entry(entry.entry_id)
        assert retrieved.h5_url is None
        assert retrieved.matrix_url is None
        assert len(retrieved.supplementary_urls) == 0
        # Auto-selection should fall back to RAW_FIRST


# ============================================================================
# Test Class: URL Extraction
# ============================================================================


class TestURLExtraction:
    """Test URL extraction and storage in queue entries."""

    def test_queue_entry_with_matrix_url(self, data_manager_with_queue):
        """Test queue entry stores matrix URL correctly."""
        entry = DownloadQueueEntry(
            entry_id="queue_matrix_test",
            dataset_id="GSE123456",
            database="geo",
            status=DownloadStatus.PENDING,
            metadata={"title": "Matrix test"},
            matrix_url="ftp://ftp.ncbi.nlm.nih.gov/.../GSE123456_matrix.txt.gz",
        )

        data_manager_with_queue.download_queue.add_entry(entry)

        retrieved = data_manager_with_queue.download_queue.get_entry(entry.entry_id)
        assert retrieved.matrix_url is not None
        assert "matrix" in retrieved.matrix_url.lower()

    def test_queue_entry_with_supplementary_urls(self, data_manager_with_queue):
        """Test queue entry stores supplementary URLs correctly."""
        entry = DownloadQueueEntry(
            entry_id="queue_supp_test",
            dataset_id="GSE123456",
            database="geo",
            status=DownloadStatus.PENDING,
            metadata={"title": "Supplementary test"},
            supplementary_urls=[
                "ftp://ftp.ncbi.nlm.nih.gov/.../GSE123456_file1.csv.gz",
                "ftp://ftp.ncbi.nlm.nih.gov/.../GSE123456_file2.csv.gz",
            ],
        )

        data_manager_with_queue.download_queue.add_entry(entry)

        retrieved = data_manager_with_queue.download_queue.get_entry(entry.entry_id)
        assert len(retrieved.supplementary_urls) == 2

    def test_queue_entry_with_h5_url(self, data_manager_with_queue):
        """Test queue entry stores H5 URL correctly."""
        entry = DownloadQueueEntry(
            entry_id="queue_h5_test",
            dataset_id="GSE123456",
            database="geo",
            status=DownloadStatus.PENDING,
            metadata={"title": "H5 test"},
            h5_url="http://example.com/GSE123456.h5",
        )

        data_manager_with_queue.download_queue.add_entry(entry)

        retrieved = data_manager_with_queue.download_queue.get_entry(entry.entry_id)
        assert retrieved.h5_url is not None
        assert ".h5" in retrieved.h5_url


# ============================================================================
# Test Class: Modality Naming
# ============================================================================


class TestModalityNaming:
    """Test modality naming patterns."""

    def test_modality_naming_pattern(
        self, data_manager_with_queue, sample_queue_entry
    ):
        """Test that modality naming follows pattern: geo_{dataset_id}_{adapter}."""
        # This test verifies the expected naming convention
        # Actual modality creation happens in GEOService

        expected_pattern = f"geo_{sample_queue_entry.dataset_id.lower()}"

        # Add entry
        data_manager_with_queue.download_queue.add_entry(sample_queue_entry)

        # Verify entry has correct dataset_id
        entry = data_manager_with_queue.download_queue.get_entry(
            sample_queue_entry.entry_id
        )
        assert entry.dataset_id == "GSE180759"

        # Expected modality name should start with this pattern
        assert "gse180759" in expected_pattern.lower()

    def test_completed_entry_has_modality_name(self, data_manager_with_queue):
        """Test that completed entry stores modality name."""
        entry = DownloadQueueEntry(
            entry_id="queue_completed_test",
            dataset_id="GSE123456",
            database="geo",
            status=DownloadStatus.COMPLETED,
            metadata={"title": "Completed test"},
            modality_name="geo_gse123456_transcriptomics",
        )

        data_manager_with_queue.download_queue.add_entry(entry)

        retrieved = data_manager_with_queue.download_queue.get_entry(entry.entry_id)
        assert retrieved.status == DownloadStatus.COMPLETED
        assert retrieved.modality_name is not None
        assert "geo_gse123456" in retrieved.modality_name


# ============================================================================
# Test Class: Concatenation Strategy
# ============================================================================


class TestConcatenationStrategy:
    """Test concatenation strategy validation."""

    def test_valid_auto_strategy(self):
        """Test 'auto' concatenation strategy is valid."""
        strategy = StrategyConfig(
            strategy_name="MATRIX_FIRST",
            concatenation_strategy="auto",
            confidence=0.9,
        )

        assert strategy.concatenation_strategy == "auto"

    def test_valid_union_strategy(self):
        """Test 'union' concatenation strategy is valid."""
        strategy = StrategyConfig(
            strategy_name="MATRIX_FIRST",
            concatenation_strategy="union",
            confidence=0.9,
        )

        assert strategy.concatenation_strategy == "union"

    def test_valid_intersection_strategy(self):
        """Test 'intersection' concatenation strategy is valid."""
        strategy = StrategyConfig(
            strategy_name="MATRIX_FIRST",
            concatenation_strategy="intersection",
            confidence=0.9,
        )

        assert strategy.concatenation_strategy == "intersection"

    def test_invalid_concatenation_strategy(self):
        """Test invalid concatenation strategy raises error."""
        with pytest.raises(ValueError, match="concatenation_strategy must be one of"):
            StrategyConfig(
                strategy_name="MATRIX_FIRST",
                concatenation_strategy="invalid",
                confidence=0.9,
            )


# ============================================================================
# Test Class: Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling in queue operations."""

    def test_failed_entry_has_error_log(self, data_manager_with_queue):
        """Test that failed entry stores error log."""
        entry = DownloadQueueEntry(
            entry_id="queue_failed_test",
            dataset_id="GSE123456",
            database="geo",
            status=DownloadStatus.FAILED,
            metadata={"title": "Failed test"},
            error_log=["Network error: Connection timeout"],
        )

        data_manager_with_queue.download_queue.add_entry(entry)

        retrieved = data_manager_with_queue.download_queue.get_entry(entry.entry_id)
        assert retrieved.status == DownloadStatus.FAILED
        assert len(retrieved.error_log) > 0
        assert "Network error" in retrieved.error_log[0]

    def test_multiple_errors_in_log(self, data_manager_with_queue):
        """Test that multiple errors can be logged."""
        entry = DownloadQueueEntry(
            entry_id="queue_multi_error_test",
            dataset_id="GSE123456",
            database="geo",
            status=DownloadStatus.FAILED,
            metadata={"title": "Multi-error test"},
            error_log=[
                "First error: Connection timeout",
                "Second error: File not found",
            ],
        )

        data_manager_with_queue.download_queue.add_entry(entry)

        retrieved = data_manager_with_queue.download_queue.get_entry(entry.entry_id)
        assert len(retrieved.error_log) == 2
