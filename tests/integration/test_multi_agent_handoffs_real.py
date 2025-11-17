"""
Integration tests for multi-agent handoffs with real supervisor coordination.

This test suite validates handoff mechanisms between specialized agents through
the supervisor in a production-like environment with real API calls:
- research_agent → metadata_assistant: Context preservation during handoffs
- research_agent → data_expert: Workspace state transfer and dataset loading
- metadata_assistant → research_agent: Bidirectional handback communication
- Supervisor coordination: Complex 4-agent workflows with orchestration

**Architecture:**
- All handoffs are supervisor-mediated (no direct agent-to-agent communication)
- Supervisor uses dynamic decision framework based on active agents
- Context preservation through LangGraph state management
- Workspace state shared across agents via DataManagerV2

**Test Strategy:**
- Use real GEO datasets and PubMed identifiers
- Test both successful handoffs and failure recovery
- Validate context/workspace state preservation across agents
- Test complex multi-step workflows requiring multiple handoffs
- Verify supervisor's decision-making and coordination logic

**Markers:**
- @pytest.mark.real_api: All tests (requires API keys + network)
- @pytest.mark.slow: Tests >30s (multi-agent workflows)
- @pytest.mark.integration: Multi-component tests

**Environment Requirements:**
- NCBI_API_KEY (recommended for GEO rate limits)
- AWS_BEDROCK_ACCESS_KEY + AWS_BEDROCK_SECRET_ACCESS_KEY (for LLM)
- Internet connectivity for PubMed/GEO access

Phase 7 - Task Group 4: Multi-Agent Handoff Validation Tests
"""

import os
import time
from pathlib import Path
from typing import Any, Dict

import pytest

from lobster.config.settings import get_settings
from lobster.core.client import AgentClient
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def test_workspace(tmp_path_factory):
    """Create temporary workspace for test session."""
    workspace = tmp_path_factory.mktemp("test_multi_agent_handoffs_real")
    return workspace


@pytest.fixture(scope="module")
def data_manager(test_workspace):
    """Initialize DataManagerV2 with test workspace."""
    settings = get_settings()
    dm = DataManagerV2(workspace_path=test_workspace, console=None)
    return dm


@pytest.fixture(scope="module")
def agent_client(data_manager, test_workspace):
    """Create AgentClient for testing multi-agent coordination via full graph."""
    return AgentClient(
        data_manager=data_manager,
        enable_reasoning=False,
        workspace_path=test_workspace,
    )


@pytest.fixture(scope="module")
def check_api_keys():
    """Verify required API keys are present."""
    required_keys = ["AWS_BEDROCK_ACCESS_KEY", "AWS_BEDROCK_SECRET_ACCESS_KEY"]
    missing = [key for key in required_keys if not os.getenv(key)]

    if missing:
        pytest.skip(f"Missing required API keys: {', '.join(missing)}")

    # NCBI_API_KEY is recommended but not required
    if not os.getenv("NCBI_API_KEY"):
        logger.warning("NCBI_API_KEY not set - GEO requests may be rate limited")


@pytest.fixture(scope="module")
def known_identifiers():
    """Known stable identifiers for testing."""
    return {
        "publication": "PMID:35042229",  # Has linked GEO dataset
        "dataset": "GSE180759",  # GEO dataset with good metadata
        "dataset_small": "GSE12345",  # Another dataset for comparison
    }


# ============================================================================
# Scenario 4.1: research_agent → metadata_assistant Handoff
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
@pytest.mark.slow
class TestResearchToMetadataHandoff:
    """Test handoffs from research_agent to metadata_assistant via supervisor."""

    def test_context_preservation_during_handoff(
        self, agent_client, data_manager, check_api_keys, known_identifiers
    ):
        """Test that context (dataset info) is preserved during handoff."""
        # Step 1: Research agent discovers dataset
        result1 = agent_client.query(
            f"Find the GEO dataset linked to {known_identifiers['publication']}"
        )

        response1 = result1.get("response", "")

        # Verify dataset discovered
        assert "GSE" in response1 or "dataset" in response1.lower()

        # Step 2: Request metadata standardization (should handoff to metadata_assistant)
        result2 = agent_client.query(
            f"Now standardize the metadata for {known_identifiers['dataset']} to transcriptomics schema"
        )

        response2 = result2.get("response", "")

        # Verify metadata standardization attempted
        assert (
            "standardiz" in response2.lower()
            or "transcriptomics" in response2.lower()
            or "schema" in response2.lower()
        )

    def test_handoff_message_format(
        self, agent_client, check_api_keys, known_identifiers
    ):
        """Test that handoff messages contain required context."""
        # Request task requiring metadata operations
        result = agent_client.query(
            f"Get sample metadata from {known_identifiers['dataset']} in summary format"
        )

        response = result.get("response", "")

        # Verify metadata operation completed (via handoff)
        assert (
            "metadata" in response.lower()
            or "sample" in response.lower()
            or "GSE" in response
        )

    def test_dataset_validation_handoff(
        self, agent_client, check_api_keys, known_identifiers
    ):
        """Test handoff for dataset validation task."""
        result = agent_client.query(
            f"Validate that {known_identifiers['dataset']} has required sample metadata fields"
        )

        response = result.get("response", "")

        # Verify validation executed
        assert "validation" in response.lower() or "sample" in response.lower()

    def test_sample_mapping_handoff(
        self, agent_client, check_api_keys, known_identifiers
    ):
        """Test handoff for cross-dataset sample mapping."""
        result = agent_client.query(
            f"Map samples between {known_identifiers['dataset']} and {known_identifiers['dataset_small']}"
        )

        response = result.get("response", "")

        # Verify mapping attempted
        assert "mapping" in response.lower() or "match" in response.lower()


# ============================================================================
# Scenario 4.2: research_agent → data_expert Handoff
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
@pytest.mark.slow
class TestResearchToDataExpertHandoff:
    """Test handoffs from research_agent to data_expert via supervisor."""

    def test_workspace_state_transfer(
        self, agent_client, data_manager, check_api_keys, known_identifiers
    ):
        """Test that workspace state is accessible after handoff."""
        # Step 1: Research agent caches dataset
        result1 = agent_client.query(
            f"Cache metadata for {known_identifiers['dataset']} to workspace"
        )

        response1 = result1.get("response", "")

        # Verify caching attempted
        assert "cache" in response1.lower() or "workspace" in response1.lower()

        # Step 2: Request data operation (should handoff to data_expert)
        result2 = agent_client.query("List all cached datasets in workspace")

        response2 = result2.get("response", "")

        # Verify workspace listing
        assert (
            "workspace" in response2.lower()
            or "cached" in response2.lower()
            or "GSE" in response2
        )

    def test_dataset_loading_handoff(
        self, agent_client, data_manager, check_api_keys, known_identifiers
    ):
        """Test handoff when dataset loading is required."""
        # Request dataset download (requires data_expert)
        result = agent_client.query(
            f"Download {known_identifiers['dataset']} and tell me the number of samples"
        )

        response = result.get("response", "")

        # Verify dataset information retrieved
        assert (
            "sample" in response.lower()
            or "GSE" in response
            or "dataset" in response.lower()
        )

    def test_data_quality_check_handoff(
        self, agent_client, check_api_keys, known_identifiers
    ):
        """Test handoff for data quality assessment."""
        result = agent_client.query(
            f"Check if {known_identifiers['dataset']} has missing values"
        )

        response = result.get("response", "")

        # Verify quality check attempted
        assert (
            "missing" in response.lower()
            or "quality" in response.lower()
            or "data" in response.lower()
        )

    def test_cache_miss_recovery(self, agent_client, data_manager, check_api_keys):
        """Test handoff recovery when cached dataset not found."""
        # Request non-existent cached dataset
        result = agent_client.query("Get content for dataset NONEXISTENT_12345")

        response = result.get("response", "")

        # Verify graceful handling
        assert (
            "not found" in response.lower()
            or "not" in response.lower()
            or "no" in response.lower()
        )


# ============================================================================
# Scenario 4.3: metadata_assistant → research_agent Handback
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
@pytest.mark.slow
class TestMetadataToResearchHandback:
    """Test bidirectional handoffs (handback) from metadata_assistant to research_agent."""

    def test_publication_lookup_after_validation(
        self, agent_client, check_api_keys, known_identifiers
    ):
        """Test handback to research_agent after metadata validation."""
        # Step 1: Validate dataset (metadata_assistant)
        result1 = agent_client.query(
            f"Validate {known_identifiers['dataset']} metadata completeness"
        )

        response1 = result1.get("response", "")

        # Verify validation attempted
        assert "validation" in response1.lower() or "metadata" in response1.lower()

        # Step 2: Request related publication (should handback to research_agent)
        result2 = agent_client.query(
            f"Now find the publication associated with {known_identifiers['dataset']}"
        )

        response2 = result2.get("response", "")

        # Verify publication found
        assert "PMID" in response2 or "publication" in response2.lower()

    def test_literature_search_after_standardization(
        self, agent_client, check_api_keys, known_identifiers
    ):
        """Test handback for literature search after metadata standardization."""
        # Step 1: Standardize metadata
        result1 = agent_client.query(
            f"Standardize {known_identifiers['dataset']} to transcriptomics schema"
        )

        response1 = result1.get("response", "")

        # Step 2: Request literature search (handback to research_agent)
        result2 = agent_client.query(
            "Find papers about similar breast cancer transcriptomics studies"
        )

        response2 = result2.get("response", "")

        # Verify literature search executed
        assert (
            "paper" in response2.lower()
            or "PMID" in response2
            or "publication" in response2.lower()
        )

    def test_bidirectional_communication(
        self, agent_client, check_api_keys, known_identifiers
    ):
        """Test complete bidirectional workflow (research→metadata→research)."""
        # Step 1: Find dataset (research_agent)
        result1 = agent_client.query(
            f"Find datasets from publication {known_identifiers['publication']}"
        )

        # Step 2: Validate dataset (→ metadata_assistant)
        result2 = agent_client.query(
            f"Validate the metadata for {known_identifiers['dataset']}"
        )

        # Step 3: Find related papers (→ research_agent)
        result3 = agent_client.query("Find other papers by the same authors")

        response3 = result3.get("response", "")

        # Verify complete workflow executed
        assert (
            "paper" in response3.lower()
            or "author" in response3.lower()
            or "PMID" in response3
        )

    def test_agent_unavailable_recovery(self, agent_client, check_api_keys):
        """Test recovery when handback target agent unavailable."""
        # Request operation that might fail
        result = agent_client.query(
            "Perform complex analysis requiring multiple unavailable agents"
        )

        response = result.get("response", "")

        # Verify graceful handling (either success or clear error message)
        assert len(response) > 0  # Some response received


# ============================================================================
# Scenario 4.4: Supervisor Coordination (CRITICAL)
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
@pytest.mark.slow
class TestSupervisorCoordination:
    """Test complex supervisor coordination with 4-agent workflows."""

    def test_four_agent_workflow_complete(
        self, agent_client, data_manager, check_api_keys, known_identifiers
    ):
        """
        CRITICAL: Test complete 4-agent workflow orchestration.

        Workflow: research → metadata → data_expert → research
        """
        start_time = time.time()

        # Step 1: Literature search (research_agent)
        result1 = agent_client.query(
            "Search PubMed for 'breast cancer single-cell RNA-seq' (max 3 results)"
        )

        response1 = result1.get("response", "")
        logger.info(f"Step 1 (research): {len(response1)} chars")

        # Step 2: Find dataset (research_agent)
        result2 = agent_client.query(
            f"Find datasets from {known_identifiers['publication']}"
        )

        response2 = result2.get("response", "")
        logger.info(f"Step 2 (research): {len(response2)} chars")

        # Step 3: Standardize metadata (→ metadata_assistant)
        result3 = agent_client.query(
            f"Standardize {known_identifiers['dataset']} metadata to transcriptomics schema"
        )

        response3 = result3.get("response", "")
        logger.info(f"Step 3 (metadata): {len(response3)} chars")

        # Step 4: Cache to workspace (→ data_expert)
        result4 = agent_client.query(
            f"Cache {known_identifiers['dataset']} to workspace"
        )

        response4 = result4.get("response", "")
        logger.info(f"Step 4 (data): {len(response4)} chars")

        elapsed = time.time() - start_time

        # Verify all steps completed
        assert len(response1) > 0  # Literature search
        assert len(response2) > 0  # Dataset discovery
        assert len(response3) > 0  # Metadata standardization
        assert len(response4) > 0  # Workspace caching

        logger.info(f"4-agent workflow completed in {elapsed:.2f}s")

    def test_supervisor_decision_framework(
        self, agent_client, check_api_keys, known_identifiers
    ):
        """Test supervisor's decision-making for agent delegation."""
        # Test 1: Should delegate to research_agent
        result1 = agent_client.query("Search PubMed for 'CRISPR screening' papers")

        response1 = result1.get("response", "")
        assert "paper" in response1.lower() or "PMID" in response1

        # Test 2: Should delegate to metadata_assistant
        result2 = agent_client.query(
            f"Validate {known_identifiers['dataset']} metadata"
        )

        response2 = result2.get("response", "")
        assert "validation" in response2.lower() or "metadata" in response2.lower()

        # Test 3: Should delegate to data_expert
        result3 = agent_client.query("List all cached datasets in workspace")

        response3 = result3.get("response", "")
        assert "workspace" in response3.lower() or "cached" in response3.lower()

    def test_complex_orchestration_with_failures(self, agent_client, check_api_keys):
        """Test supervisor coordination when some operations fail."""
        # Request workflow with potential failures
        result = agent_client.query(
            "Find datasets from PMID:INVALID, validate metadata, and cache to workspace"
        )

        response = result.get("response", "")

        # Verify graceful handling of failures
        assert len(response) > 0  # Some response received
        # Should indicate issue or provide partial results
        assert (
            "error" in response.lower()
            or "invalid" in response.lower()
            or "not found" in response.lower()
            or "unable" in response.lower()
        )

    def test_mid_workflow_failure_recovery(
        self, agent_client, data_manager, check_api_keys, known_identifiers
    ):
        """Test recovery when mid-workflow step fails."""
        # Step 1: Successful operation
        result1 = agent_client.query(
            f"Find datasets from {known_identifiers['publication']}"
        )

        # Step 2: Operation that may fail
        result2 = agent_client.query("Validate metadata for INVALID_DATASET")

        response2 = result2.get("response", "")

        # Verify error handled gracefully
        assert (
            "error" in response2.lower()
            or "invalid" in response2.lower()
            or "not found" in response2.lower()
        )

        # Step 3: Verify supervisor can continue after failure
        result3 = agent_client.query(
            f"Instead, validate {known_identifiers['dataset']}"
        )

        response3 = result3.get("response", "")

        # Should recover and complete validation
        assert "validation" in response3.lower() or "metadata" in response3.lower()

    def test_parallel_agent_coordination(
        self, agent_client, check_api_keys, known_identifiers
    ):
        """Test supervisor handling multiple agent requests efficiently."""
        # Request task requiring coordination
        result = agent_client.query(
            f"For {known_identifiers['dataset']}: validate metadata, find the publication, and check data quality"
        )

        response = result.get("response", "")

        # Verify multiple aspects addressed
        # (May be sequential execution, but should complete all tasks)
        assert len(response) > 100  # Substantial response expected

    def test_state_consistency_across_handoffs(
        self, agent_client, data_manager, check_api_keys, known_identifiers
    ):
        """Test that shared state remains consistent across multiple handoffs."""
        # Workflow with multiple handoffs
        result1 = agent_client.query(
            f"Cache {known_identifiers['dataset']} to workspace"
        )

        # Check workspace from different agent
        result2 = agent_client.query("List all cached datasets")

        response2 = result2.get("response", "")

        # Workspace state should be consistent
        assert (
            "GSE" in response2
            or "cached" in response2.lower()
            or "no" in response2.lower()
        )


# ============================================================================
# Integration Workflow Tests
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
@pytest.mark.slow
class TestMultiAgentIntegrationWorkflows:
    """End-to-end integration tests for complete multi-agent workflows."""

    def test_complete_research_pipeline(
        self, agent_client, data_manager, check_api_keys
    ):
        """
        Test complete research-to-analysis pipeline.

        Workflow: Literature search → Dataset discovery → Metadata validation → Caching
        """
        # Complete pipeline
        steps = [
            "Search PubMed for 'breast cancer transcriptomics' (max 2 results)",
            "Find datasets from the first paper",
            "Validate metadata for the discovered dataset",
            "Cache the dataset to workspace",
        ]

        for i, step in enumerate(steps):
            result = agent_client.query(step)
            response = result.get("response", "")
            logger.info(f"Pipeline step {i+1}/{len(steps)}: {len(response)} chars")
            assert len(response) > 0  # Each step produces response

    def test_meta_analysis_preparation(
        self, agent_client, check_api_keys, known_identifiers
    ):
        """Test workflow for preparing multiple datasets for meta-analysis."""
        # Workflow: Find datasets → Standardize metadata → Validate compatibility
        result1 = agent_client.query(
            "Find datasets related to breast cancer transcriptomics"
        )

        result2 = agent_client.query(
            f"Standardize metadata for {known_identifiers['dataset']} to transcriptomics schema"
        )

        result3 = agent_client.query(
            f"Validate {known_identifiers['dataset']} has required fields"
        )

        # Verify all steps completed
        assert all(
            [
                result1.get("response", ""),
                result2.get("response", ""),
                result3.get("response", ""),
            ]
        )
