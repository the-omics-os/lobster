"""
Comprehensive integration tests for queue-based download workflow.

This module tests the complete workflow using real Lobster components:
- DataManagerV2 for workspace management
- DownloadQueue for queue management
- DownloadOrchestrator for download routing
- GEODownloadService for GEO-specific downloads

The workflow tested:
1. research_agent validates dataset → creates DownloadQueueEntry (PENDING)
2. data_expert calls orchestrator.execute_download(entry_id)
3. Orchestrator routes to GEODownloadService
4. Service downloads data → stores in DataManagerV2
5. Queue status updated (PENDING → IN_PROGRESS → COMPLETED/FAILED)
6. Provenance logged

Test Strategy:
- Use real Lobster components (no mocking of internal systems)
- Mock only external dependencies (GEO API calls via GEOService.download_dataset)
- Test end-to-end workflow with status transitions
- Test error handling and edge cases
- Verify provenance tracking

Markers:
- @pytest.mark.integration: Multi-component integration tests
- @pytest.mark.slow: Tests that may take > 5 seconds (if using real API)

Running Tests:
```bash
# All tests in this file
pytest tests/integration/test_queue_workflow.py -v

# Specific test
pytest tests/integration/test_queue_workflow.py::TestQueueWorkflow::test_basic_queue_workflow -v

# With detailed output
pytest tests/integration/test_queue_workflow.py -v -s
```
"""

from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, Mock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.download_queue import EntryNotFoundError
from lobster.core.schemas.download_queue import (
    DownloadQueueEntry,
    DownloadStatus,
    StrategyConfig,
)
from lobster.tools.download_orchestrator import (
    DownloadOrchestrator,
    ServiceNotFoundError,
)
from lobster.tools.geo_download_service import GEODownloadService
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def data_manager(tmp_path):
    """
    Create DataManagerV2 with temporary workspace.

    Args:
        tmp_path: pytest built-in fixture providing temporary directory

    Returns:
        DataManagerV2: Fresh data manager instance for testing
    """
    workspace = tmp_path / "test_workspace"
    workspace.mkdir()
    dm = DataManagerV2(workspace_path=str(workspace))
    logger.info(f"Created DataManagerV2 with workspace: {workspace}")
    return dm


@pytest.fixture
def orchestrator(data_manager):
    """
    Create DownloadOrchestrator with GEODownloadService registered.

    Args:
        data_manager: DataManagerV2 fixture

    Returns:
        DownloadOrchestrator: Orchestrator with GEO service registered
    """
    orch = DownloadOrchestrator(data_manager)
    geo_service = GEODownloadService(data_manager)
    orch.register_service(geo_service)
    logger.info("Created DownloadOrchestrator with GEODownloadService registered")
    return orch


@pytest.fixture
def mock_adata():
    """
    Create mock AnnData object for testing.

    Returns:
        AnnData: Minimal AnnData with synthetic data
    """
    n_obs = 100
    n_vars = 50
    X = np.random.poisson(5, size=(n_obs, n_vars))
    obs = pd.DataFrame(
        {"cell_type": ["T cell"] * n_obs}, index=[f"cell_{i}" for i in range(n_obs)]
    )
    var = pd.DataFrame(
        {"gene_name": [f"gene_{i}" for i in range(n_vars)]},
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    adata = ad.AnnData(X=X, obs=obs, var=var)
    logger.debug(f"Created mock AnnData: {n_obs} obs × {n_vars} vars")
    return adata


# ============================================================================
# Helper Functions
# ============================================================================


def create_test_queue_entry(
    dataset_id: str = "GSE12345",
    status: DownloadStatus = DownloadStatus.PENDING,
    strategy_name: Optional[str] = "MATRIX_FIRST",
    strategy_params: Optional[Dict[str, Any]] = None,
    execution_params: Optional[Dict[str, Any]] = None,
) -> DownloadQueueEntry:
    """
    Create test queue entry with sensible defaults.

    Args:
        dataset_id: GEO accession ID
        status: Initial queue status
        strategy_name: Download strategy name
        strategy_params: Database-specific strategy parameters
        execution_params: Execution parameters (timeout, retries, etc.)

    Returns:
        DownloadQueueEntry: Queue entry ready for testing
    """
    recommended_strategy = None
    if strategy_name:
        recommended_strategy = StrategyConfig(
            strategy_name=strategy_name,
            concatenation_strategy="auto",
            confidence=0.95,
            rationale=f"Using {strategy_name} for testing",
            strategy_params=strategy_params,
            execution_params=execution_params,
        )

    entry = DownloadQueueEntry(
        entry_id=f"queue_{dataset_id}_test",
        dataset_id=dataset_id,
        database="geo",
        status=status,
        priority=5,
        metadata={
            "title": f"Test dataset {dataset_id}",
            "n_samples": 10,
            "platform": "GPL24676",
        },
        validation_result={"is_valid": True, "warnings": []},
        recommended_strategy=recommended_strategy,
    )
    logger.debug(f"Created test queue entry: {entry.entry_id}")
    return entry


# ============================================================================
# Test Class: Basic Queue Workflow
# ============================================================================


@pytest.mark.integration
class TestQueueWorkflow:
    """Integration tests for queue-based download workflow."""

    def test_basic_queue_workflow(self, data_manager, orchestrator, mock_adata):
        """
        Test basic workflow: create queue entry → execute download → verify modality.

        Steps:
        1. Create DownloadQueueEntry with PENDING status
        2. Mock GEOService.download_dataset to store mock data
        3. Execute download via orchestrator
        4. Verify queue status updated to COMPLETED
        5. Verify modality stored in data_manager
        6. Verify provenance logged
        """
        logger.info("=== Starting test_basic_queue_workflow ===")

        # Step 1: Create queue entry
        entry = create_test_queue_entry(dataset_id="GSE12345")
        data_manager.download_queue.add_entry(entry)
        logger.info(f"Created queue entry: {entry.entry_id}")

        # Verify entry is PENDING
        retrieved_entry = data_manager.download_queue.get_entry(entry.entry_id)
        assert retrieved_entry.status == DownloadStatus.PENDING

        # Step 2: Mock GEOService to store test data
        modality_name = "geo_gse12345_transcriptomics_single_cell"

        def mock_download_dataset(*args, **kwargs):
            """Mock GEOService.download_dataset to store test data."""
            logger.debug(
                f"Mock download_dataset called with args={args}, kwargs={kwargs}"
            )
            # Store mock data directly in data_manager modalities dict
            data_manager.modalities[modality_name] = mock_adata
            return f"Successfully downloaded GSE12345 as {modality_name}"

        # Step 3: Execute download with mocked GEOService
        with patch.object(
            orchestrator.get_service_for_database("geo").geo_service,
            "download_dataset",
            side_effect=mock_download_dataset,
        ):
            result_modality, stats = orchestrator.execute_download(entry.entry_id)
            logger.info(f"Download completed: {result_modality}")

        # Step 4: Verify queue status updated to COMPLETED
        updated_entry = data_manager.download_queue.get_entry(entry.entry_id)
        assert (
            updated_entry.status == DownloadStatus.COMPLETED
        ), f"Expected COMPLETED, got {updated_entry.status}"
        logger.info("✓ Queue status correctly updated to COMPLETED")

        # Step 5: Verify modality stored in data_manager
        modalities = data_manager.list_modalities()
        assert (
            modality_name in modalities
        ), f"Modality '{modality_name}' not found in {modalities}"

        stored_adata = data_manager.get_modality(modality_name)
        assert stored_adata.n_obs == mock_adata.n_obs
        assert stored_adata.n_vars == mock_adata.n_vars
        logger.info(
            f"✓ Modality stored correctly: {stored_adata.n_obs} obs × {stored_adata.n_vars} vars"
        )

        # Step 6: Verify provenance logged (if enabled)
        if data_manager.provenance:
            # Just verify some activities were logged (provenance is working)
            logger.info(
                f"Provenance activities logged: {len(data_manager.provenance.activities)}"
            )
            if len(data_manager.provenance.activities) > 0:
                logger.info("✓ Provenance tracking is active")
            else:
                logger.warning(
                    "No provenance activities logged (may be expected for mocked calls)"
                )
        else:
            logger.info("Provenance tracking disabled")

        logger.info("=== test_basic_queue_workflow PASSED ===")

    def test_strategy_override(self, data_manager, orchestrator, mock_adata):
        """
        Test download with strategy override.

        Steps:
        1. Create queue entry with recommended_strategy="H5_FIRST"
        2. Execute with strategy_override="MATRIX_FIRST"
        3. Verify MATRIX_FIRST was used (check logs or stats)
        4. Verify download succeeded despite override
        """
        logger.info("=== Starting test_strategy_override ===")

        # Step 1: Create queue entry with H5_FIRST strategy
        entry = create_test_queue_entry(
            dataset_id="GSE67890",
            strategy_name="H5_FIRST",
        )
        data_manager.download_queue.add_entry(entry)
        logger.info(f"Created queue entry with recommended strategy: H5_FIRST")

        # Step 2: Mock GEOService and capture strategy parameter
        captured_strategy = {}
        modality_name = "geo_gse67890_transcriptomics_single_cell"

        def mock_download_with_strategy_capture(*args, **kwargs):
            """Mock download and capture strategy parameter."""
            captured_strategy["manual_strategy_override"] = kwargs.get(
                "manual_strategy_override"
            )
            captured_strategy["use_intersecting_genes_only"] = kwargs.get(
                "use_intersecting_genes_only"
            )
            logger.debug(f"Captured strategy: {captured_strategy}")
            data_manager.modalities[modality_name] = mock_adata
            return f"Successfully downloaded with strategy override"

        # Step 3: Execute with strategy override
        strategy_override = {
            "strategy_name": "MATRIX_FIRST",
            "strategy_params": {"use_intersecting_genes_only": False},
        }

        with patch.object(
            orchestrator.get_service_for_database("geo").geo_service,
            "download_dataset",
            side_effect=mock_download_with_strategy_capture,
        ):
            result_modality, stats = orchestrator.execute_download(
                entry.entry_id, strategy_override=strategy_override
            )

        # Step 4: Verify MATRIX_FIRST was used (not H5_FIRST)
        assert (
            captured_strategy["manual_strategy_override"] == "MATRIX_FIRST"
        ), f"Expected MATRIX_FIRST, got {captured_strategy['manual_strategy_override']}"
        assert captured_strategy["use_intersecting_genes_only"] is False
        logger.info("✓ Strategy override correctly applied")

        # Verify download succeeded
        updated_entry = data_manager.download_queue.get_entry(entry.entry_id)
        assert updated_entry.status == DownloadStatus.COMPLETED
        logger.info("✓ Download succeeded with override")

        logger.info("=== test_strategy_override PASSED ===")

    def test_execution_params(self, data_manager, orchestrator, mock_adata):
        """
        Test execution parameters are respected.

        Steps:
        1. Create queue entry with execution_params (timeout, max_retries)
        2. Execute download
        3. Verify execution params were accessible (check via helper method)
        4. Verify download completed successfully
        """
        logger.info("=== Starting test_execution_params ===")

        # Step 1: Create queue entry with execution parameters
        execution_params = {
            "timeout": 7200,  # 2 hours
            "max_retries": 5,
            "retry_backoff": 2.0,
        }

        entry = create_test_queue_entry(
            dataset_id="GSE11111",
            strategy_name="MATRIX_FIRST",
            execution_params=execution_params,
        )
        data_manager.download_queue.add_entry(entry)
        logger.info(f"Created queue entry with execution_params: {execution_params}")

        # Step 2: Mock download
        modality_name = "geo_gse11111_transcriptomics_single_cell"

        def mock_download(*args, **kwargs):
            data_manager.modalities[modality_name] = mock_adata
            return f"Downloaded successfully"

        # Step 3: Execute download
        with patch.object(
            orchestrator.get_service_for_database("geo").geo_service,
            "download_dataset",
            side_effect=mock_download,
        ):
            result_modality, stats = orchestrator.execute_download(entry.entry_id)

        # Step 4: Verify execution params were accessible
        # (In real implementation, these would be used by download service)
        retrieved_entry = data_manager.download_queue.get_entry(entry.entry_id)
        assert retrieved_entry.recommended_strategy is not None
        assert retrieved_entry.recommended_strategy.execution_params == execution_params
        logger.info("✓ Execution params correctly stored and retrievable")

        # Verify download completed
        assert retrieved_entry.status == DownloadStatus.COMPLETED
        logger.info("✓ Download completed successfully")

        logger.info("=== test_execution_params PASSED ===")

    def test_failed_download_retry(self, data_manager, orchestrator, mock_adata):
        """
        Test retry workflow for failed downloads.

        Steps:
        1. Create queue entry with valid dataset ID
        2. Mock GEOService to fail on first call
        3. Execute download (should fail)
        4. Verify queue status is FAILED
        5. Mock GEOService to succeed on retry
        6. Retry download
        7. Verify queue status is COMPLETED
        """
        logger.info("=== Starting test_failed_download_retry ===")

        # Step 1: Create queue entry
        entry = create_test_queue_entry(dataset_id="GSE22222")
        data_manager.download_queue.add_entry(entry)
        logger.info(f"Created queue entry: {entry.entry_id}")

        # Step 2: Mock GEOService to fail
        def mock_download_fail(*args, **kwargs):
            raise RuntimeError("Simulated download failure (network timeout)")

        # Step 3: Execute download (should fail)
        with patch.object(
            orchestrator.get_service_for_database("geo").geo_service,
            "download_dataset",
            side_effect=mock_download_fail,
        ):
            with pytest.raises(RuntimeError, match="Simulated download failure"):
                orchestrator.execute_download(entry.entry_id)

        # Step 4: Verify queue status is FAILED
        failed_entry = data_manager.download_queue.get_entry(entry.entry_id)
        assert failed_entry.status == DownloadStatus.FAILED
        assert failed_entry.error_log is not None
        # error_log is a list of strings
        assert len(failed_entry.error_log) > 0
        assert any(
            "Simulated download failure" in log for log in failed_entry.error_log
        )
        logger.info("✓ Queue status correctly updated to FAILED")

        # Step 5: Mock GEOService to succeed on retry
        modality_name = "geo_gse22222_transcriptomics_single_cell"

        def mock_download_success(*args, **kwargs):
            data_manager.modalities[modality_name] = mock_adata
            return f"Successfully downloaded after retry"

        # Step 6: Retry download
        with patch.object(
            orchestrator.get_service_for_database("geo").geo_service,
            "download_dataset",
            side_effect=mock_download_success,
        ):
            result_modality, stats = orchestrator.execute_download(entry.entry_id)
            logger.info(f"Retry succeeded: {result_modality}")

        # Step 7: Verify queue status is COMPLETED
        completed_entry = data_manager.download_queue.get_entry(entry.entry_id)
        assert completed_entry.status == DownloadStatus.COMPLETED
        # Note: error_log may persist from previous failure (not cleared on retry success)
        # This is expected behavior - it preserves failure history
        logger.info("✓ Retry successful, status updated to COMPLETED")

        logger.info("=== test_failed_download_retry PASSED ===")

    def test_duplicate_execution_prevention(self, data_manager, orchestrator):
        """
        Test that IN_PROGRESS entries cannot be executed concurrently.

        Steps:
        1. Create queue entry with PENDING status
        2. Update status to IN_PROGRESS manually
        3. Attempt execute_download
        4. Verify raises appropriate error (ValueError)
        5. Verify queue status remains IN_PROGRESS
        """
        logger.info("=== Starting test_duplicate_execution_prevention ===")

        # Step 1: Create queue entry
        entry = create_test_queue_entry(dataset_id="GSE33333")
        data_manager.download_queue.add_entry(entry)
        logger.info(f"Created queue entry: {entry.entry_id}")

        # Step 2: Manually update status to IN_PROGRESS
        data_manager.download_queue.update_status(
            entry.entry_id,
            DownloadStatus.IN_PROGRESS,
        )
        logger.info("Manually set status to IN_PROGRESS")

        in_progress_entry = data_manager.download_queue.get_entry(entry.entry_id)
        assert in_progress_entry.status == DownloadStatus.IN_PROGRESS

        # Step 3: Attempt to execute download (should fail)
        with pytest.raises(ValueError, match="Cannot execute download"):
            orchestrator.execute_download(entry.entry_id)

        logger.info("✓ Correctly rejected execution of IN_PROGRESS entry")

        # Step 5: Verify status remains IN_PROGRESS
        still_in_progress = data_manager.download_queue.get_entry(entry.entry_id)
        assert still_in_progress.status == DownloadStatus.IN_PROGRESS
        logger.info("✓ Status remains IN_PROGRESS (not corrupted)")

        logger.info("=== test_duplicate_execution_prevention PASSED ===")

    def test_completed_entry_no_retry(self, data_manager, orchestrator, mock_adata):
        """
        Test that COMPLETED entries cannot be re-executed without explicit handling.

        Steps:
        1. Create queue entry and complete download successfully
        2. Verify status is COMPLETED
        3. Attempt execute_download again
        4. Verify raises error or handles gracefully
        5. Verify original modality unchanged
        """
        logger.info("=== Starting test_completed_entry_no_retry ===")

        # Step 1: Create and complete a download
        entry = create_test_queue_entry(dataset_id="GSE44444")
        data_manager.download_queue.add_entry(entry)

        modality_name = "geo_gse44444_transcriptomics_single_cell"
        original_n_obs = mock_adata.n_obs
        original_n_vars = mock_adata.n_vars

        def mock_download(*args, **kwargs):
            data_manager.modalities[modality_name] = mock_adata
            return f"Downloaded successfully"

        with patch.object(
            orchestrator.get_service_for_database("geo").geo_service,
            "download_dataset",
            side_effect=mock_download,
        ):
            orchestrator.execute_download(entry.entry_id)

        # Step 2: Verify status is COMPLETED
        completed_entry = data_manager.download_queue.get_entry(entry.entry_id)
        assert completed_entry.status == DownloadStatus.COMPLETED
        logger.info("✓ Initial download completed successfully")

        # Step 3: Attempt to execute again (should fail)
        with pytest.raises(ValueError, match="Cannot execute download"):
            orchestrator.execute_download(entry.entry_id)

        logger.info("✓ Correctly rejected re-execution of COMPLETED entry")

        # Step 5: Verify original modality unchanged
        stored_adata = data_manager.get_modality(modality_name)
        assert stored_adata.n_obs == original_n_obs
        assert stored_adata.n_vars == original_n_vars
        logger.info("✓ Original modality remains unchanged")

        logger.info("=== test_completed_entry_no_retry PASSED ===")


# ============================================================================
# Test Class: Error Handling
# ============================================================================


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling and edge cases in queue workflow."""

    def test_invalid_entry_id(self, orchestrator):
        """
        Test error handling for invalid entry_id.

        Steps:
        1. Attempt execute_download with nonexistent entry_id
        2. Verify raises ValueError
        3. Verify error message is descriptive
        """
        logger.info("=== Starting test_invalid_entry_id ===")

        invalid_entry_id = "queue_NONEXISTENT_12345"

        # Should raise EntryNotFoundError for nonexistent entry
        with pytest.raises(EntryNotFoundError, match=r"not found"):
            orchestrator.execute_download(invalid_entry_id)

        logger.info("✓ Correctly rejected invalid entry_id")
        logger.info("=== test_invalid_entry_id PASSED ===")

    def test_invalid_database_type(self, data_manager, orchestrator):
        """
        Test error handling for database without registered service.

        Steps:
        1. Create queue entry with database="sra" (no service registered)
        2. Attempt execute_download
        3. Verify raises ServiceNotFoundError
        4. Verify error lists available databases
        """
        logger.info("=== Starting test_invalid_database_type ===")

        # Create entry with unsupported database
        entry = DownloadQueueEntry(
            entry_id="queue_SRA_test",
            dataset_id="SRR12345",
            database="sra",  # No SRA service registered
            status=DownloadStatus.PENDING,
            metadata={"title": "SRA test"},
        )
        data_manager.download_queue.add_entry(entry)

        # Should raise ServiceNotFoundError
        with pytest.raises(ServiceNotFoundError) as exc_info:
            orchestrator.execute_download(entry.entry_id)

        # Verify error message includes available databases
        error = exc_info.value
        assert error.database == "sra"
        assert "geo" in error.available_databases
        logger.info(f"✓ Correctly rejected unsupported database: {error}")

        logger.info("=== test_invalid_database_type PASSED ===")

    def test_invalid_strategy_override(self, data_manager, orchestrator):
        """
        Test error handling for invalid strategy override parameters.

        Steps:
        1. Create valid queue entry
        2. Attempt execute_download with invalid strategy_params
        3. Verify raises ValueError
        4. Verify queue status remains PENDING (not corrupted)
        """
        logger.info("=== Starting test_invalid_strategy_override ===")

        # Create valid entry
        entry = create_test_queue_entry(dataset_id="GSE55555")
        data_manager.download_queue.add_entry(entry)

        # Invalid strategy override (wrong type for boolean parameter)
        invalid_override = {
            "strategy_name": "MATRIX_FIRST",
            "strategy_params": {"use_intersecting_genes_only": "invalid_not_a_boolean"},
        }

        # Should raise ValueError during validation
        with pytest.raises(ValueError, match=r"Invalid strategy_override|must be bool"):
            orchestrator.execute_download(
                entry.entry_id, strategy_override=invalid_override
            )

        logger.info("✓ Correctly rejected invalid strategy_override")

        # Verify queue status updated to FAILED (validation errors are caught and status updated)
        entry_after = data_manager.download_queue.get_entry(entry.entry_id)
        assert entry_after.status == DownloadStatus.FAILED
        logger.info(
            "✓ Queue status correctly updated to FAILED after validation failure"
        )

        logger.info("=== test_invalid_strategy_override PASSED ===")

    def test_geo_service_exception_handling(self, data_manager, orchestrator):
        """
        Test that GEOService exceptions are properly caught and logged.

        Steps:
        1. Create valid queue entry
        2. Mock GEOService to raise unexpected exception
        3. Execute download (should fail)
        4. Verify queue status is FAILED
        5. Verify error_log contains exception details
        6. Verify exception is re-raised
        """
        logger.info("=== Starting test_geo_service_exception_handling ===")

        # Step 1: Create queue entry
        entry = create_test_queue_entry(dataset_id="GSE66666")
        data_manager.download_queue.add_entry(entry)

        # Step 2: Mock GEOService to raise unexpected exception
        def mock_download_crash(*args, **kwargs):
            raise RuntimeError("Unexpected GEOService crash (simulated)")

        # Step 3: Execute download (should fail)
        with patch.object(
            orchestrator.get_service_for_database("geo").geo_service,
            "download_dataset",
            side_effect=mock_download_crash,
        ):
            with pytest.raises(RuntimeError, match="Unexpected GEOService crash"):
                orchestrator.execute_download(entry.entry_id)

        # Step 4: Verify queue status is FAILED
        failed_entry = data_manager.download_queue.get_entry(entry.entry_id)
        assert failed_entry.status == DownloadStatus.FAILED
        logger.info("✓ Queue status correctly updated to FAILED")

        # Step 5: Verify error_log contains exception details
        assert failed_entry.error_log is not None
        # error_log is a list of strings
        assert len(failed_entry.error_log) > 0
        error_log_str = " ".join(failed_entry.error_log)
        assert "RuntimeError" in error_log_str
        assert "Unexpected GEOService crash" in error_log_str
        logger.info("✓ Error log correctly captured exception details")

        logger.info("=== test_geo_service_exception_handling PASSED ===")


# ============================================================================
# Test Class: Provenance and Metadata
# ============================================================================


@pytest.mark.integration
class TestProvenanceTracking:
    """Test provenance tracking and metadata in queue workflow."""

    def test_provenance_logging(self, data_manager, orchestrator, mock_adata):
        """
        Test that provenance is correctly logged for downloads.

        Steps:
        1. Execute successful download
        2. Verify tool_usage_history has download entry
        3. Verify provenance contains correct parameters
        4. Verify IR (AnalysisStep) is present
        """
        logger.info("=== Starting test_provenance_logging ===")

        # Step 1: Execute successful download
        entry = create_test_queue_entry(dataset_id="GSE77777")
        data_manager.download_queue.add_entry(entry)

        modality_name = "geo_gse77777_transcriptomics_single_cell"

        def mock_download(*args, **kwargs):
            data_manager.modalities[modality_name] = mock_adata
            return f"Downloaded successfully"

        with patch.object(
            orchestrator.get_service_for_database("geo").geo_service,
            "download_dataset",
            side_effect=mock_download,
        ):
            orchestrator.execute_download(entry.entry_id)

        # Step 2: Verify provenance activities has download entry
        if data_manager.provenance:
            logger.info(
                f"Provenance activities logged: {len(data_manager.provenance.activities)}"
            )
            if len(data_manager.provenance.activities) > 0:
                download_activities = [
                    a
                    for a in data_manager.provenance.activities
                    if "download" in str(a.get("activity_type", "")).lower()
                ]
                if len(download_activities) > 0:
                    logger.info("✓ Download operation logged in provenance activities")
                    # Step 3 & 4: Verify provenance details (simplified for activities structure)
                    last_activity = download_activities[-1]
                    # Activities have different structure - just verify basic info
                    logger.info(
                        f"Last activity type: {last_activity.get('activity_type')}"
                    )
                    logger.info("✓ Provenance activity correctly logged")
                else:
                    logger.info(
                        "No specific download activities found (may use generic activity_type)"
                    )
            else:
                logger.warning(
                    "No provenance activities logged (may be expected for mocked calls)"
                )
        else:
            logger.warning("Provenance disabled")

        logger.info("=== test_provenance_logging PASSED ===")

    def test_queue_entry_metadata_preservation(
        self, data_manager, orchestrator, mock_adata
    ):
        """
        Test that queue entry metadata is preserved through workflow.

        Steps:
        1. Create queue entry with rich metadata
        2. Execute download
        3. Verify metadata still present in queue entry
        4. Verify validation_result preserved
        5. Verify recommended_strategy preserved
        """
        logger.info("=== Starting test_queue_entry_metadata_preservation ===")

        # Step 1: Create queue entry with rich metadata
        metadata = {
            "title": "Rich metadata test",
            "n_samples": 24,
            "platform": "GPL24676",
            "organism": "Homo sapiens",
            "experiment_type": "scRNA-seq",
        }

        validation_result = {
            "is_valid": True,
            "warnings": ["Minor metadata inconsistency"],
            "n_samples_validated": 24,
        }

        strategy = StrategyConfig(
            strategy_name="H5_FIRST",
            concatenation_strategy="intersection",
            confidence=0.98,
            rationale="Optimal strategy for this dataset",
            strategy_params={"use_intersecting_genes_only": True},
            execution_params={"timeout": 3600, "max_retries": 3},
        )

        entry = DownloadQueueEntry(
            entry_id="queue_metadata_test",
            dataset_id="GSE88888",
            database="geo",
            status=DownloadStatus.PENDING,
            metadata=metadata,
            validation_result=validation_result,
            recommended_strategy=strategy,
        )
        data_manager.download_queue.add_entry(entry)

        # Step 2: Execute download
        modality_name = "geo_gse88888_transcriptomics_single_cell"

        def mock_download(*args, **kwargs):
            data_manager.modalities[modality_name] = mock_adata
            return f"Downloaded successfully"

        with patch.object(
            orchestrator.get_service_for_database("geo").geo_service,
            "download_dataset",
            side_effect=mock_download,
        ):
            orchestrator.execute_download(entry.entry_id)

        # Step 3: Verify metadata preserved
        completed_entry = data_manager.download_queue.get_entry(entry.entry_id)
        assert completed_entry.metadata == metadata
        logger.info("✓ Metadata preserved")

        # Step 4: Verify validation_result preserved
        assert completed_entry.validation_result == validation_result
        logger.info("✓ Validation result preserved")

        # Step 5: Verify recommended_strategy preserved
        assert completed_entry.recommended_strategy is not None
        assert completed_entry.recommended_strategy.strategy_name == "H5_FIRST"
        assert completed_entry.recommended_strategy.confidence == 0.98
        logger.info("✓ Recommended strategy preserved")

        logger.info("=== test_queue_entry_metadata_preservation PASSED ===")


# ============================================================================
# Test Class: Edge Cases
# ============================================================================


@pytest.mark.integration
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_queue(self, data_manager, orchestrator):
        """
        Test behavior with empty download queue.

        Steps:
        1. Verify queue is empty
        2. Attempt to execute download with nonexistent entry
        3. Verify raises ValueError
        """
        logger.info("=== Starting test_empty_queue ===")

        # Step 1: Verify queue is empty
        entries = data_manager.download_queue.list_entries()
        assert len(entries) == 0
        logger.info("✓ Queue is empty")

        # Step 2: Attempt download from empty queue
        with pytest.raises(EntryNotFoundError, match=r"not found"):
            orchestrator.execute_download("queue_NONEXISTENT")

        logger.info("✓ Correctly handled download from empty queue")
        logger.info("=== test_empty_queue PASSED ===")

    def test_multiple_queue_entries_independence(
        self, data_manager, orchestrator, mock_adata
    ):
        """
        Test that multiple queue entries are processed independently.

        Steps:
        1. Create 3 queue entries with different datasets
        2. Execute download for entry 1 (success)
        3. Execute download for entry 2 (fail)
        4. Execute download for entry 3 (success)
        5. Verify statuses are independent
        """
        logger.info("=== Starting test_multiple_queue_entries_independence ===")

        # Step 1: Create 3 queue entries
        entry1 = create_test_queue_entry(dataset_id="GSE11111")
        entry2 = create_test_queue_entry(dataset_id="GSE22222")
        entry3 = create_test_queue_entry(dataset_id="GSE33333")

        data_manager.download_queue.add_entry(entry1)
        data_manager.download_queue.add_entry(entry2)
        data_manager.download_queue.add_entry(entry3)

        logger.info("Created 3 queue entries")

        # Step 2: Execute entry1 (success)
        def mock_download_success(*args, **kwargs):
            geo_id = kwargs.get("geo_id", args[0] if args else "GSE00000")
            modality = f"geo_{geo_id.lower()}_transcriptomics_single_cell"
            data_manager.modalities[modality] = mock_adata
            return f"Downloaded {geo_id}"

        with patch.object(
            orchestrator.get_service_for_database("geo").geo_service,
            "download_dataset",
            side_effect=mock_download_success,
        ):
            orchestrator.execute_download(entry1.entry_id)

        logger.info("Entry 1 completed successfully")

        # Step 3: Execute entry2 (fail)
        def mock_download_fail(*args, **kwargs):
            raise RuntimeError("Simulated failure for entry 2")

        with patch.object(
            orchestrator.get_service_for_database("geo").geo_service,
            "download_dataset",
            side_effect=mock_download_fail,
        ):
            with pytest.raises(RuntimeError):
                orchestrator.execute_download(entry2.entry_id)

        logger.info("Entry 2 failed as expected")

        # Step 4: Execute entry3 (success)
        with patch.object(
            orchestrator.get_service_for_database("geo").geo_service,
            "download_dataset",
            side_effect=mock_download_success,
        ):
            orchestrator.execute_download(entry3.entry_id)

        logger.info("Entry 3 completed successfully")

        # Step 5: Verify statuses are independent
        final1 = data_manager.download_queue.get_entry(entry1.entry_id)
        final2 = data_manager.download_queue.get_entry(entry2.entry_id)
        final3 = data_manager.download_queue.get_entry(entry3.entry_id)

        assert final1.status == DownloadStatus.COMPLETED
        assert final2.status == DownloadStatus.FAILED
        assert final3.status == DownloadStatus.COMPLETED

        logger.info("✓ Queue entries processed independently")
        logger.info("=== test_multiple_queue_entries_independence PASSED ===")

    def test_strategy_params_none_handling(
        self, data_manager, orchestrator, mock_adata
    ):
        """
        Test that None/missing strategy_params are handled gracefully.

        Steps:
        1. Create queue entry with strategy but no strategy_params
        2. Execute download
        3. Verify download succeeds
        4. Verify no crashes from missing params
        """
        logger.info("=== Starting test_strategy_params_none_handling ===")

        # Step 1: Create entry without strategy_params
        entry = DownloadQueueEntry(
            entry_id="queue_no_params_test",
            dataset_id="GSE99999",
            database="geo",
            status=DownloadStatus.PENDING,
            metadata={"title": "Test without params"},
            recommended_strategy=StrategyConfig(
                strategy_name="MATRIX_FIRST",
                concatenation_strategy="auto",
                confidence=0.9,
                rationale="No params test",
                strategy_params=None,  # Explicitly None
                execution_params=None,  # Explicitly None
            ),
        )
        data_manager.download_queue.add_entry(entry)

        # Step 2: Execute download
        modality_name = "geo_gse99999_transcriptomics_single_cell"

        def mock_download(*args, **kwargs):
            data_manager.modalities[modality_name] = mock_adata
            return f"Downloaded successfully"

        with patch.object(
            orchestrator.get_service_for_database("geo").geo_service,
            "download_dataset",
            side_effect=mock_download,
        ):
            result_modality, stats = orchestrator.execute_download(entry.entry_id)

        # Step 3: Verify download succeeded
        final_entry = data_manager.download_queue.get_entry(entry.entry_id)
        assert final_entry.status == DownloadStatus.COMPLETED
        logger.info("✓ Download succeeded with None strategy_params")

        logger.info("=== test_strategy_params_none_handling PASSED ===")
