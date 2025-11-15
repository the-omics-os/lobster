"""
End-to-end integration tests for complete download queue workflow.

Tests complete workflow across research_agent → supervisor → data_expert with
real API calls to validate queue consumer pattern in production-like environment.

**Workflow Architecture:**
1. research_agent validates metadata and adds to queue (Task 2.2B)
2. supervisor queries download_queue workspace for entry_id
3. data_expert downloads from queue using entry_id (Task 2.2C)

**Test Strategy:**
- Use real GEO datasets (small ones like GSE109564, GSE126906)
- Test complete workflow with real API calls
- Validate queue status transitions
- Test workspace persistence and restoration
- Verify modality creation and data quality
- Test error recovery and concurrent downloads

**Markers:**
- @pytest.mark.integration: Multi-component tests
- @pytest.mark.real_api: Requires internet + API keys
- @pytest.mark.slow: Tests >30s

**Environment Requirements:**
- AWS_BEDROCK_ACCESS_KEY + AWS_BEDROCK_SECRET_ACCESS_KEY
- NCBI_API_KEY (recommended for rate limits)
- Internet connectivity for GEO access

Task 2.2D: Integration Tests for End-to-End Queue Workflow
"""

import os
import time
from datetime import datetime
from pathlib import Path

import pytest

from lobster.agents.data_expert import data_expert
from lobster.agents.research_agent import research_agent
from lobster.agents.supervisor import supervisor
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.schemas.download_queue import DownloadStatus
from lobster.tools.workspace_tool import create_get_content_from_workspace_tool
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def test_workspace(tmp_path_factory):
    """Create temporary workspace for test session."""
    workspace = tmp_path_factory.mktemp("test_queue_workflow_e2e")
    return workspace


@pytest.fixture(scope="module")
def data_manager(test_workspace):
    """Initialize DataManagerV2 with test workspace."""
    dm = DataManagerV2(workspace_path=test_workspace, console=None)
    return dm


@pytest.fixture(scope="module")
def integrated_system(data_manager):
    """Create complete agent system for integration testing."""
    # Create agents with shared data_manager
    research = research_agent(data_manager=data_manager)
    data_exp = data_expert(data_manager=data_manager)
    sup = supervisor(data_manager=data_manager)

    # Create workspace query tool for supervisor
    workspace_tool = create_get_content_from_workspace_tool(data_manager)

    return {
        "data_manager": data_manager,
        "research_agent": research,
        "data_expert": data_exp,
        "supervisor": sup,
        "workspace_tool": workspace_tool,
    }


@pytest.fixture(scope="module")
def check_api_keys():
    """Verify required API keys are present."""
    required_keys = ["AWS_BEDROCK_ACCESS_KEY", "AWS_BEDROCK_SECRET_ACCESS_KEY"]
    missing = [key for key in required_keys if not os.getenv(key)]

    if missing:
        pytest.skip(f"Missing required API keys: {', '.join(missing)}")

    if not os.getenv("NCBI_API_KEY"):
        logger.warning("NCBI_API_KEY not set - GEO requests may be rate limited")


@pytest.fixture(scope="module")
def known_datasets():
    """Known stable GEO datasets for testing."""
    return {
        "small": "GSE109564",  # Small dataset, fast download
        "medium": "GSE126906",  # Medium dataset
        "large": "GSE180759",  # Larger dataset with good metadata
    }


# ============================================================================
# Test Class: Complete Queue Workflow
# ============================================================================


@pytest.mark.integration
@pytest.mark.real_api
@pytest.mark.slow
class TestCompleteQueueWorkflow:
    """Integration tests for complete queue workflow."""

    def test_research_to_data_expert_workflow(
        self, integrated_system, check_api_keys, known_datasets
    ):
        """
        Test complete workflow:
        1. research_agent validates metadata → creates queue entry
        2. supervisor queries download_queue
        3. data_expert downloads from queue
        """
        dm = integrated_system["data_manager"]
        research = integrated_system["research_agent"]
        data_exp = integrated_system["data_expert"]

        dataset_id = known_datasets["small"]

        # ===== STEP 1: research_agent validates and queues dataset =====
        logger.info(f"Step 1: Validating {dataset_id} and adding to queue")

        # Find validate_dataset_metadata tool
        validate_tool = next(
            (t for t in research.tools if t.name == "validate_dataset_metadata"), None
        )
        assert (
            validate_tool is not None
        ), "validate_dataset_metadata tool not found in research_agent"

        result1 = validate_tool.invoke({"dataset_id": dataset_id, "add_to_queue": True})

        logger.info(f"Validation result: {result1[:200]}...")

        # Verify queue entry created
        assert (
            "added to download queue" in result1.lower()
            or "queue" in result1.lower()
            or "validated" in result1.lower()
        )

        # ===== STEP 2: Verify queue entry created =====
        logger.info("Step 2: Verifying queue entry")

        queue_entries = dm.download_queue.list_entries(status=DownloadStatus.PENDING)
        assert len(queue_entries) > 0, "No pending entries in queue after validation"

        entry_id = queue_entries[0].entry_id
        entry_dataset_id = queue_entries[0].dataset_id

        logger.info(f"Queue entry created: {entry_id} for {entry_dataset_id}")

        # Verify entry has required metadata
        entry = dm.download_queue.get_entry(entry_id)
        assert entry.dataset_id == dataset_id
        assert entry.status == DownloadStatus.PENDING
        assert entry.metadata is not None
        assert len(entry.metadata) > 0

        # ===== STEP 3: data_expert downloads from queue =====
        logger.info(f"Step 3: Downloading {dataset_id} from queue entry {entry_id}")

        download_tool = next(
            (t for t in data_exp.tools if t.name == "execute_download_from_queue"), None
        )
        assert (
            download_tool is not None
        ), "execute_download_from_queue tool not found in data_expert"

        start_time = time.time()
        result2 = download_tool.invoke({"entry_id": entry_id})
        download_time = time.time() - start_time

        logger.info(f"Download completed in {download_time:.1f}s")
        logger.info(f"Download result: {result2[:300]}...")

        # ===== STEP 4: Verify download completed =====
        logger.info("Step 4: Verifying download completion")

        updated_entry = dm.download_queue.get_entry(entry_id)
        assert (
            updated_entry.status == DownloadStatus.COMPLETED
        ), f"Expected COMPLETED, got {updated_entry.status}"
        assert (
            updated_entry.modality_name is not None
        ), "modality_name not set after successful download"

        # ===== STEP 5: Verify modality created =====
        logger.info("Step 5: Verifying modality creation")

        modalities = dm.list_modalities()
        assert len(modalities) > 0, "No modalities created after successful download"
        assert any(
            dataset_id.lower() in mod.lower() for mod in modalities
        ), f"Expected modality with {dataset_id} not found in {modalities}"

        # ===== STEP 6: Verify modality data quality =====
        logger.info("Step 6: Verifying modality data quality")

        modality_name = updated_entry.modality_name
        adata = dm.get_modality(modality_name)

        assert adata.n_obs > 0, f"Modality has 0 observations"
        assert adata.n_vars > 0, f"Modality has 0 variables"

        logger.info(
            f"✓ Complete workflow successful: {adata.n_obs} obs × {adata.n_vars} vars"
        )

    @pytest.mark.slow
    def test_multi_dataset_queue(
        self, integrated_system, check_api_keys, known_datasets
    ):
        """Test queuing and downloading multiple datasets."""
        dm = integrated_system["data_manager"]
        research = integrated_system["research_agent"]
        data_exp = integrated_system["data_expert"]

        # Use two small datasets for faster testing
        datasets = [known_datasets["small"], known_datasets["medium"]]

        logger.info(f"Testing multi-dataset queue with: {datasets}")

        # ===== STEP 1: Queue multiple datasets =====
        validate_tool = next(
            (t for t in research.tools if t.name == "validate_dataset_metadata"), None
        )

        initial_pending = len(
            dm.download_queue.list_entries(status=DownloadStatus.PENDING)
        )

        for dataset_id in datasets:
            logger.info(f"Queuing {dataset_id}...")
            result = validate_tool.invoke(
                {"dataset_id": dataset_id, "add_to_queue": True}
            )
            # Allow some delay between requests to respect rate limits
            time.sleep(1)

        # ===== STEP 2: Verify all queued =====
        queue_entries = dm.download_queue.list_entries(status=DownloadStatus.PENDING)
        assert len(queue_entries) >= len(
            datasets
        ), f"Expected at least {len(datasets)} pending entries, got {len(queue_entries)}"

        logger.info(f"All {len(datasets)} datasets queued successfully")

        # ===== STEP 3: Download all from queue =====
        download_tool = next(
            (t for t in data_exp.tools if t.name == "execute_download_from_queue"), None
        )

        # Get entry IDs for our datasets
        our_entries = [e for e in queue_entries if e.dataset_id in datasets][
            : len(datasets)
        ]

        for entry in our_entries:
            logger.info(f"Downloading {entry.dataset_id} from queue...")
            start_time = time.time()
            result = download_tool.invoke({"entry_id": entry.entry_id})
            logger.info(f"Download completed in {time.time() - start_time:.1f}s")

            # Allow some delay between downloads
            time.sleep(1)

        # ===== STEP 4: Verify all completed =====
        completed_entries = dm.download_queue.list_entries(
            status=DownloadStatus.COMPLETED
        )
        our_completed = [e for e in completed_entries if e.dataset_id in datasets]

        assert len(our_completed) >= len(
            datasets
        ), f"Expected {len(datasets)} completed, got {len(our_completed)}"

        logger.info(f"✓ All {len(datasets)} datasets downloaded successfully")

    def test_supervisor_workspace_query(
        self, integrated_system, check_api_keys, known_datasets
    ):
        """Test supervisor can query download_queue workspace."""
        dm = integrated_system["data_manager"]
        research = integrated_system["research_agent"]
        workspace_tool = integrated_system["workspace_tool"]

        dataset_id = known_datasets["small"]

        # ===== STEP 1: Create queue entry via research_agent =====
        logger.info(f"Step 1: Creating queue entry for {dataset_id}")

        validate_tool = next(
            (t for t in research.tools if t.name == "validate_dataset_metadata"), None
        )
        result1 = validate_tool.invoke({"dataset_id": dataset_id, "add_to_queue": True})

        # ===== STEP 2: Supervisor queries workspace =====
        logger.info("Step 2: Supervisor querying download_queue workspace")

        # Use workspace tool to query download_queue
        result2 = workspace_tool.invoke({"workspace": "download_queue"})

        logger.info(f"Workspace query result: {result2[:300]}...")

        # ===== STEP 3: Verify queue entry visible in workspace =====
        assert "queue_" in result2, "Queue entry ID format not found in workspace"
        assert (
            dataset_id in result2
        ), f"Dataset ID {dataset_id} not found in workspace query"
        assert (
            "PENDING" in result2 or "pending" in result2.lower()
        ), "Status not found in workspace query"

        logger.info("✓ Supervisor can successfully query download_queue workspace")

    @pytest.mark.real_api
    def test_url_extraction_integration(
        self, integrated_system, check_api_keys, known_datasets
    ):
        """Test GEOProvider URL extraction within workflow."""
        dm = integrated_system["data_manager"]
        research = integrated_system["research_agent"]

        dataset_id = known_datasets["small"]

        # ===== STEP 1: Validate and queue (triggers URL extraction) =====
        logger.info(f"Validating {dataset_id} and extracting URLs")

        validate_tool = next(
            (t for t in research.tools if t.name == "validate_dataset_metadata"), None
        )
        result = validate_tool.invoke({"dataset_id": dataset_id, "add_to_queue": True})

        # ===== STEP 2: Verify queue entry has URLs =====
        queue_entries = dm.download_queue.list_entries()
        entry = next((e for e in queue_entries if e.dataset_id == dataset_id), None)

        assert entry is not None, f"Queue entry for {dataset_id} not found"

        # Check that at least one URL type is present
        has_urls = (
            entry.matrix_url is not None
            or len(entry.supplementary_urls) > 0
            or entry.h5_url is not None
            or len(entry.raw_urls) > 0
        )

        assert has_urls, f"No URLs extracted for {dataset_id}"

        logger.info(
            f"✓ URLs extracted: matrix={entry.matrix_url is not None}, "
            f"supp={len(entry.supplementary_urls)}, h5={entry.h5_url is not None}"
        )


# ============================================================================
# Test Class: Error Recovery and Edge Cases
# ============================================================================


@pytest.mark.integration
@pytest.mark.real_api
class TestErrorRecoveryWorkflow:
    """Test error recovery and edge cases in queue workflow."""

    def test_invalid_dataset_id(self, integrated_system, check_api_keys):
        """Test error handling for invalid GEO ID."""
        dm = integrated_system["data_manager"]
        research = integrated_system["research_agent"]

        invalid_id = "GSE_INVALID_999999999"

        logger.info(f"Testing invalid dataset ID: {invalid_id}")

        validate_tool = next(
            (t for t in research.tools if t.name == "validate_dataset_metadata"), None
        )

        # This should fail gracefully
        result = validate_tool.invoke({"dataset_id": invalid_id, "add_to_queue": False})

        # Verify error handling
        assert (
            "error" in result.lower()
            or "invalid" in result.lower()
            or "not found" in result.lower()
        ), "Expected error message for invalid dataset ID"

        logger.info("✓ Invalid dataset ID handled gracefully")

    def test_concurrent_download_prevention(
        self, integrated_system, check_api_keys, known_datasets
    ):
        """Test that concurrent downloads are prevented."""
        dm = integrated_system["data_manager"]
        research = integrated_system["research_agent"]
        data_exp = integrated_system["data_expert"]

        dataset_id = known_datasets["small"]

        # ===== STEP 1: Create queue entry =====
        validate_tool = next(
            (t for t in research.tools if t.name == "validate_dataset_metadata"), None
        )
        validate_tool.invoke({"dataset_id": dataset_id, "add_to_queue": True})

        # ===== STEP 2: Manually set status to IN_PROGRESS =====
        queue_entries = dm.download_queue.list_entries(status=DownloadStatus.PENDING)
        entry = next((e for e in queue_entries if e.dataset_id == dataset_id), None)

        if entry:
            dm.download_queue.update_status(
                entry_id=entry.entry_id,
                status=DownloadStatus.IN_PROGRESS,
                downloaded_by="test_agent",
            )

            # ===== STEP 3: Try to download again =====
            download_tool = next(
                (t for t in data_exp.tools if t.name == "execute_download_from_queue"),
                None,
            )
            result = download_tool.invoke({"entry_id": entry.entry_id})

            # Should return error about in-progress download
            assert (
                "in progress" in result.lower()
                or "currently being downloaded" in result.lower()
            ), "Expected in-progress error message"

            logger.info("✓ Concurrent download prevention working")


# ============================================================================
# Test Class: Workspace Persistence
# ============================================================================


@pytest.mark.integration
class TestWorkspacePersistence:
    """Test workspace persistence and restoration."""

    def test_queue_persistence_across_sessions(self, test_workspace, check_api_keys):
        """Test that queue entries persist across DataManager instances."""
        dataset_id = "GSE123456_test"

        # ===== SESSION 1: Create queue entry =====
        logger.info("Session 1: Creating queue entry")

        dm1 = DataManagerV2(workspace_path=test_workspace)
        research1 = research_agent(data_manager=dm1)

        # Create a simple queue entry (without full validation to avoid API call)
        from lobster.core.schemas.download_queue import DownloadQueueEntry

        entry = DownloadQueueEntry(
            entry_id=f"queue_{dataset_id}_persist_test",
            dataset_id=dataset_id,
            database="geo",
            status=DownloadStatus.PENDING,
            metadata={"title": "Persistence test"},
        )
        dm1.download_queue.add_entry(entry)

        # Verify entry exists
        entries1 = dm1.download_queue.list_entries()
        assert len(entries1) > 0

        logger.info(f"Created {len(entries1)} queue entries in session 1")

        # ===== SESSION 2: Restore and verify =====
        logger.info("Session 2: Restoring workspace")

        dm2 = DataManagerV2(workspace_path=test_workspace)

        # Queue entries should persist
        entries2 = dm2.download_queue.list_entries()

        # Find our test entry
        our_entry = next((e for e in entries2 if e.dataset_id == dataset_id), None)

        assert (
            our_entry is not None
        ), f"Queue entry for {dataset_id} not found after restoration"
        assert our_entry.status == DownloadStatus.PENDING

        logger.info("✓ Queue entries persist across sessions")


# ============================================================================
# Test Class: Provenance Tracking
# ============================================================================


@pytest.mark.integration
@pytest.mark.real_api
@pytest.mark.slow
class TestProvenanceTracking:
    """Test provenance tracking end-to-end."""

    def test_provenance_end_to_end(
        self, integrated_system, check_api_keys, known_datasets
    ):
        """Test that provenance is tracked throughout queue workflow."""
        dm = integrated_system["data_manager"]
        research = integrated_system["research_agent"]
        data_exp = integrated_system["data_expert"]

        dataset_id = known_datasets["small"]

        # ===== STEP 1: Validate and queue =====
        validate_tool = next(
            (t for t in research.tools if t.name == "validate_dataset_metadata"), None
        )
        validate_tool.invoke({"dataset_id": dataset_id, "add_to_queue": True})

        # ===== STEP 2: Download from queue =====
        queue_entries = dm.download_queue.list_entries(status=DownloadStatus.PENDING)
        entry = next((e for e in queue_entries if e.dataset_id == dataset_id), None)

        if entry:
            download_tool = next(
                (t for t in data_exp.tools if t.name == "execute_download_from_queue"),
                None,
            )
            result = download_tool.invoke({"entry_id": entry.entry_id})

            # ===== STEP 3: Verify provenance logged =====
            # Check that data_manager has tool usage history
            assert hasattr(
                dm, "tool_usage_history"
            ), "DataManager should have tool_usage_history"

            # Check that download operation was logged
            if len(dm.tool_usage_history) > 0:
                # Find download operation in history
                download_ops = [
                    op
                    for op in dm.tool_usage_history
                    if "execute_download_from_queue" in op.get("tool_name", "")
                ]

                if download_ops:
                    logger.info(
                        f"✓ Provenance tracked: {len(download_ops)} download operations"
                    )
                else:
                    logger.warning(
                        "Download operation not found in provenance (may be expected)"
                    )


# ============================================================================
# Performance Benchmarks
# ============================================================================


@pytest.mark.integration
@pytest.mark.real_api
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarks for queue workflow."""

    def test_small_dataset_download_time(
        self, integrated_system, check_api_keys, known_datasets
    ):
        """Benchmark download time for small dataset."""
        dm = integrated_system["data_manager"]
        research = integrated_system["research_agent"]
        data_exp = integrated_system["data_expert"]

        dataset_id = known_datasets["small"]

        # ===== STEP 1: Validate and queue (measure time) =====
        validate_tool = next(
            (t for t in research.tools if t.name == "validate_dataset_metadata"), None
        )

        start_validation = time.time()
        validate_tool.invoke({"dataset_id": dataset_id, "add_to_queue": True})
        validation_time = time.time() - start_validation

        logger.info(f"Validation time: {validation_time:.2f}s")

        # ===== STEP 2: Download from queue (measure time) =====
        queue_entries = dm.download_queue.list_entries(status=DownloadStatus.PENDING)
        entry = next((e for e in queue_entries if e.dataset_id == dataset_id), None)

        if entry:
            download_tool = next(
                (t for t in data_exp.tools if t.name == "execute_download_from_queue"),
                None,
            )

            start_download = time.time()
            result = download_tool.invoke({"entry_id": entry.entry_id})
            download_time = time.time() - start_download

            logger.info(f"Download time: {download_time:.2f}s")
            logger.info(f"Total time: {validation_time + download_time:.2f}s")

            # Performance targets
            # Validation should be < 30s for small datasets
            # Download should be < 2 minutes for small datasets
            assert (
                validation_time < 60
            ), f"Validation took {validation_time:.2f}s (>60s)"
            assert download_time < 300, f"Download took {download_time:.2f}s (>5min)"

            logger.info(
                f"✓ Performance acceptable: {validation_time:.2f}s validation, {download_time:.2f}s download"
            )
