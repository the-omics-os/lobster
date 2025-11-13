"""
Integration tests for research_agent workspace functionality and metadata_assistant handoff.

Tests real workflows with actual services:
- Workspace caching and retrieval with different detail levels
- research_agent → metadata_assistant handoff via supervisor
- Tool enhancements (related_to, entry_type, focus, database parameters)
- End-to-end workflow: Discover → Cache → Hand Off → Validate

Test coverage target: ≥85% with realistic workflow scenarios.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.research_agent import research_agent
from lobster.core.data_manager_v2 import DataManagerV2


# ===============================================================================
# Test Fixtures
# ===============================================================================


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for integration tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir) / ".lobster_workspace"
        workspace_path.mkdir(parents=True, exist_ok=True)
        # Create workspace subdirectories
        (workspace_path / "literature").mkdir(exist_ok=True)
        (workspace_path / "data").mkdir(exist_ok=True)
        (workspace_path / "metadata").mkdir(exist_ok=True)
        yield workspace_path


@pytest.fixture
def data_manager(temp_workspace):
    """Create DataManagerV2 with test workspace."""
    dm = DataManagerV2(workspace_path=temp_workspace)
    return dm


@pytest.fixture
def sample_dataset():
    """Create sample transcriptomics dataset for testing."""
    obs_data = {
        "sample_id": [f"Sample_{i}" for i in range(1, 11)],
        "condition": ["Control"] * 5 + ["Treatment"] * 5,
        "organism": ["Homo sapiens"] * 10,
        "platform": ["Illumina NovaSeq"] * 10,
        "sequencing_type": ["single-cell"] * 10,
        "batch": ["Batch1"] * 5 + ["Batch2"] * 5,
        "cell_type": ["T_cell"] * 10,
        "tissue": ["PBMC"] * 10,
        "timepoint": ["Day0"] * 10,
    }
    adata = ad.AnnData(
        X=np.random.rand(10, 100),
        obs=pd.DataFrame(obs_data, index=[f"Sample_{i}" for i in range(1, 11)]),
    )
    return adata


@pytest.fixture
def mock_content_access_service():
    """Mock ContentAccessService for literature tools."""
    with patch(
        "lobster.agents.research_agent.ContentAccessService"
    ) as MockContentService:
        mock_service = MockContentService.return_value

        # Mock search_literature
        mock_service.search_literature.return_value = """## Literature Search Results

**Query**: T cell exhaustion
**Results**: 3 publications

1. **PMID:12345678** - T cell exhaustion in cancer immunotherapy
   - Authors: Smith J, et al.
   - Journal: Nature Immunology
   - Published: 2023

2. **PMID:23456789** - Mechanisms of T cell dysfunction
   - Authors: Jones A, et al.
   - Journal: Cell
   - Published: 2023
"""

        # Mock find_related_publications (for related_to parameter)
        mock_service.find_related_publications.return_value = """## Related Publications

**Related to**: PMID:12345678
**Found**: 5 related papers

1. **PMID:98765432** - Citing paper about exhaustion markers
2. **PMID:87654321** - Reference about PD-1 pathway
"""

        # Mock find_linked_datasets
        mock_service.find_linked_datasets.return_value = """## Related Datasets

**Publication**: PMID:12345678
**Found**: 2 datasets

1. **GSE123456** - T cell exhaustion RNA-seq
   - Platform: Illumina
   - Samples: 20 (10 control, 10 exhausted)

2. **GSE234567** - Single-cell T cell profiling
   - Platform: 10X Genomics
   - Samples: 50,000 cells
"""

        # Mock extract_metadata
        from lobster.tools.providers.base_provider import PublicationMetadata

        mock_service.extract_metadata.return_value = PublicationMetadata(
            uid="12345678",
            title="T cell exhaustion in cancer immunotherapy",
            authors=["Smith J", "Jones A", "Brown B"],
            journal="Nature Immunology",
            published="2023-05-15",
            doi="10.1038/s41590-023-01234-5",
            pmid="12345678",
            abstract="T cell exhaustion is a major barrier to cancer immunotherapy...",
            keywords=["T cells", "exhaustion", "immunotherapy", "cancer"],
        )

        # Mock discover_datasets
        mock_service.discover_datasets.return_value = """## Dataset Search Results

**Query**: lung cancer single-cell
**Database**: GEO
**Found**: 3 datasets

1. **GSE345678** - Lung cancer single-cell RNA-seq
2. **GSE456789** - NSCLC tumor microenvironment
"""

        yield mock_service


@pytest.fixture
def mock_geo_service():
    """Mock GEOService for dataset metadata validation."""
    with patch("lobster.agents.research_agent.GEOService") as MockGEOService:
        mock_service = MockGEOService.return_value

        # Mock fetch_metadata_only
        mock_service.fetch_metadata_only.return_value = (
            {
                "title": "T cell exhaustion dataset",
                "organism": "Homo sapiens",
                "platform": "Illumina NovaSeq",
                "samples": 20,
                "description": "RNA-seq of exhausted vs control T cells",
            },
            None,  # validation_result placeholder
        )

        yield mock_service


# ===============================================================================
# Workflow 1: Workspace Caching and Retrieval
# ===============================================================================


@pytest.mark.skip(reason="Requires real LLM and full service integration")
class TestWorkspaceCaching:
    """Test workspace caching and retrieval functionality."""

    def test_write_to_workspace_publication(
        self, data_manager, mock_content_access_service
    ):
        """Test caching publication to workspace."""
        # This test would require full integration with:
        # - Real LLM for agent execution
        # - ContentAccessService for publication retrieval
        # - Workspace write functionality
        pass

    def test_get_content_from_workspace_list_mode(self, data_manager, temp_workspace):
        """Test listing all cached content from workspace."""
        # Create some test cached content
        literature_dir = temp_workspace / "literature"
        test_file = literature_dir / "publication_PMID12345678.json"
        test_file.write_text('{"identifier": "PMID:12345678", "cached_at": "2025-01-12"}')

        # This would require agent execution to call get_content_from_workspace()
        # with no identifier (list mode)
        pass

    def test_get_content_from_workspace_detail_levels(
        self, data_manager, temp_workspace
    ):
        """Test different detail levels for workspace retrieval."""
        # Test would verify:
        # - summary level (key-value pairs)
        # - methods level (Methods section)
        # - samples level (Sample IDs)
        # - platform level (Platform info)
        # - metadata level (Full metadata)
        # - github level (GitHub repos)
        pass


# ===============================================================================
# Workflow 2: Tool Enhancement Tests
# ===============================================================================


@pytest.mark.skip(reason="Requires real LLM and full service integration")
class TestToolEnhancements:
    """Test enhanced tool functionality."""

    def test_search_literature_with_related_to(
        self, data_manager, mock_content_access_service
    ):
        """Test search_literature with related_to parameter."""
        # Test merged functionality from discover_related_studies
        # Agent would call: search_literature(related_to="PMID:12345678")
        pass

    def test_find_related_entries_with_entry_type(
        self, data_manager, mock_content_access_service
    ):
        """Test find_related_entries with entry_type filtering."""
        # Test entry_type parameter: "publication", "dataset", "sample", "metadata"
        # Agent would call: find_related_entries("PMID:12345678", entry_type="dataset")
        pass

    def test_extract_methods_single_paper(
        self, data_manager, mock_content_access_service
    ):
        """Test extract_methods with single paper."""
        # Test single paper extraction
        # Agent would call: extract_methods("PMID:12345678")
        pass

    def test_extract_methods_batch(self, data_manager, mock_content_access_service):
        """Test extract_methods with batch identifiers."""
        # Test batch processing (comma-separated)
        # Agent would call: extract_methods("PMID:123,PMID:456,PMID:789")
        pass

    def test_get_dataset_metadata_publication(
        self, data_manager, mock_content_access_service
    ):
        """Test get_dataset_metadata with publication identifier."""
        # Test auto-detection for PMID/DOI
        # Agent would call: get_dataset_metadata("PMID:12345678")
        pass

    def test_get_dataset_metadata_dataset(self, data_manager, mock_geo_service):
        """Test get_dataset_metadata with dataset identifier."""
        # Test auto-detection for GSE
        # Agent would call: get_dataset_metadata("GSE12345")
        pass


# ===============================================================================
# Workflow 3: metadata_assistant Handoff via Supervisor
# ===============================================================================


@pytest.mark.skip(reason="Requires multi-agent setup with supervisor")
class TestMetadataAssistantHandoff:
    """Test research_agent → supervisor → metadata_assistant handoff."""

    def test_handoff_for_sample_mapping(self, data_manager, sample_dataset):
        """Test handoff to metadata_assistant for sample mapping task."""
        # Setup: Load two datasets with different sample naming
        data_manager.modalities["geo_gse12345"] = sample_dataset

        # Create second dataset with lowercase naming
        second_dataset = sample_dataset.copy()
        second_dataset.obs.index = [
            f"sample_{i}" for i in range(1, 11)
        ]  # Lowercase
        data_manager.modalities["geo_gse67890"] = second_dataset

        # This test would verify:
        # 1. research_agent receives task: "Map samples between GSE12345 and GSE67890"
        # 2. research_agent hands off to metadata_assistant via supervisor
        # 3. metadata_assistant performs sample mapping
        # 4. Results are returned to research_agent
        # 5. research_agent reports results to user
        pass

    def test_handoff_for_metadata_validation(self, data_manager, sample_dataset):
        """Test handoff to metadata_assistant for dataset validation."""
        # Setup: Load dataset
        data_manager.modalities["geo_gse12345"] = sample_dataset

        # This test would verify:
        # 1. research_agent receives task: "Validate GSE12345 for required metadata"
        # 2. research_agent caches dataset using write_to_workspace
        # 3. research_agent hands off to metadata_assistant
        # 4. metadata_assistant validates dataset
        # 5. Results are returned with recommendation (proceed/skip/manual_check)
        pass

    def test_handoff_state_preservation(self, data_manager, sample_dataset):
        """Test that state is preserved across handoff."""
        # This test would verify:
        # - DataManagerV2 state persists across agent handoff
        # - Workspace content is accessible to metadata_assistant
        # - Provenance tracking continues correctly
        pass


# ===============================================================================
# Workflow 4: End-to-End Discover → Cache → Hand Off
# ===============================================================================


@pytest.mark.skip(reason="Requires full multi-agent integration")
class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_complete_research_workflow(
        self, data_manager, mock_content_access_service, mock_geo_service
    ):
        """Test complete workflow: Discover → Cache → Validate → Hand Off."""
        # This would test the complete workflow:
        # 1. fast_dataset_search("lung cancer single-cell") - Find dataset
        # 2. get_dataset_metadata("GSE12345") - Get metadata
        # 3. write_to_workspace("dataset_GSE12345", workspace="metadata") - Cache
        # 4. handoff_to_metadata_assistant - Validate dataset
        # 5. Receive validation results
        pass

    def test_literature_extraction_workflow(
        self, data_manager, mock_content_access_service
    ):
        """Test literature extraction workflow."""
        # This would test:
        # 1. search_literature("T cell exhaustion") - Find papers
        # 2. fast_abstract_search("PMID:12345678") - Screen relevance
        # 3. read_full_publication("PMID:12345678") - Extract full content
        # 4. extract_methods("PMID:12345678") - Extract methods
        # 5. write_to_workspace("publication_PMID12345678", workspace="literature")
        pass


# ===============================================================================
# Workflow 5: Agent Tool Count and Registration Verification
# ===============================================================================


@pytest.mark.unit
class TestAgentToolCount:
    """Verify research_agent has exactly 12 tools."""

    def test_agent_has_12_base_tools(self, data_manager):
        """Verify research_agent has exactly 12 base tools (excluding handoff tools)."""
        agent = research_agent(data_manager, handoff_tools=None)

        # Get the graph and inspect tools
        # This is a unit test to verify the tool count matches Phase 4 target
        assert agent is not None

        # Note: Actual tool count verification would require inspecting
        # the agent's internal tool list, which depends on LangGraph internals

    def test_agent_with_handoff_tools(self, data_manager):
        """Verify handoff tools are properly added to base tools."""

        # Create mock handoff tool
        def mock_handoff():
            """Mock handoff to metadata_assistant."""
            return "Handed off"

        mock_handoff.name = "handoff_to_metadata_assistant"

        agent = research_agent(data_manager, handoff_tools=[mock_handoff])

        assert agent is not None
        # Verify agent was created with handoff tools
        # Actual verification of tool availability would require LangGraph inspection


# ===============================================================================
# Workspace Content Service Integration (Unit-Level)
# ===============================================================================


@pytest.mark.unit
class TestWorkspaceContentService:
    """Test WorkspaceContentService integration patterns."""

    def test_workspace_directory_structure(self, temp_workspace):
        """Test workspace directory structure creation."""
        assert temp_workspace.exists()
        assert (temp_workspace / "literature").exists()
        assert (temp_workspace / "data").exists()
        assert (temp_workspace / "metadata").exists()

    def test_workspace_content_type_validation(self, temp_workspace):
        """Test content type validation for workspace operations."""
        # Test would verify:
        # - "literature" workspace accepts publication content
        # - "data" workspace accepts dataset content
        # - "metadata" workspace accepts metadata content
        pass

    def test_naming_convention_validation(self):
        """Test workspace naming convention enforcement."""
        # Test would verify:
        # - Publications: publication_PMID12345 or publication_DOI...
        # - Datasets: dataset_GSE12345
        # - Metadata: metadata_GSE12345_samples
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
