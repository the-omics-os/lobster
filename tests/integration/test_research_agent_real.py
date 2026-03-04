"""
Real API integration tests for research_agent.

These tests make actual API calls to verify end-to-end functionality
of the ContentAccessService within the research_agent context.

Test Paper: PMID:35042229 (Nature 2022)
- PMC ID: PMC8760896
- DOI: 10.1038/s41586-021-03852-1
- Title contains: "single-cell"

Rate Limiting: 0.5-1s sleeps between consecutive API calls
"""

import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from lobster.agents.research.research_agent import research_agent
from lobster.core.data_manager_v2 import DataManagerV2

pytestmark = [pytest.mark.integration]


@pytest.fixture
def mock_data_manager(tmp_path):
    """Mock DataManagerV2 instance."""
    mock_dm = Mock(spec=DataManagerV2)
    mock_dm.metadata_store = {}
    mock_dm.list_modalities.return_value = []
    mock_dm.workspace_path = Path(tmp_path / "workspace")
    mock_dm.cache_dir = Path(tmp_path / "cache")
    mock_dm.log_tool_usage = Mock()
    return mock_dm


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.real_api
class TestContentAccessRealAPI:
    """Real API integration tests for ContentAccessService in research_agent.

    These tests make actual API calls to verify end-to-end functionality.

    Test Paper: PMID:35042229 (Nature 2022)
    - PMC ID: PMC8760896
    - DOI: 10.1038/s41586-021-03852-1
    - Title contains: "single-cell"

    Rate Limiting: 0.5-1s sleeps between consecutive API calls
    """

    @pytest.mark.skip(reason="Requires real API access - run with pytest -m real_api")
    @patch("lobster.agents.research.research_agent.create_react_agent")
    @patch("lobster.agents.research.research_agent.create_llm")
    def test_real_literature_search(
        self, mock_create_llm, mock_create_agent, mock_data_manager
    ):
        """Test real literature search via ContentAccessService.

        Verifies:
        1. Agent initializes ContentAccessService correctly
        2. Real API call to PubMed succeeds
        3. Response time <3s
        4. Results contain valid publications
        """
        import time

        from lobster.services.data_access.content_access_service import (
            ContentAccessService,
        )

        # Rate limiting
        time.sleep(1.0)

        mock_llm = Mock()
        mock_create_llm.return_value = mock_llm
        mock_agent_instance = Mock()
        mock_create_agent.return_value = mock_agent_instance

        # Create real ContentAccessService
        content_service = ContentAccessService(data_manager=mock_data_manager)

        # Real API call
        start_time = time.time()
        results = content_service.search_literature(
            query="BRCA1 breast cancer", limit=5
        )
        elapsed = time.time() - start_time

        # Verify results
        assert results is not None
        assert "results" in results
        assert len(results["results"]) > 0
        assert elapsed < 3.0

        # Verify result structure
        first_result = results["results"][0]
        assert "pmid" in first_result or "doi" in first_result
        assert "title" in first_result

        # Rate limiting
        time.sleep(0.5)

    @pytest.mark.skip(reason="Requires real API access - run with pytest -m real_api")
    @patch("lobster.agents.research.research_agent.create_react_agent")
    @patch("lobster.agents.research.research_agent.create_llm")
    def test_real_abstract_retrieval(
        self, mock_create_llm, mock_create_agent, mock_data_manager
    ):
        """Test real abstract retrieval (Tier 1 fast access).

        Test Paper: PMID:35042229
        Expected: <1s response time, Nature journal
        """
        import time

        from lobster.services.data_access.content_access_service import (
            ContentAccessService,
        )

        # Rate limiting
        time.sleep(1.0)

        mock_llm = Mock()
        mock_create_llm.return_value = mock_llm
        mock_agent_instance = Mock()
        mock_create_agent.return_value = mock_agent_instance

        # Create real ContentAccessService
        content_service = ContentAccessService(data_manager=mock_data_manager)

        # Real API call
        start_time = time.time()
        result = content_service.get_abstract("PMID:35042229")
        elapsed = time.time() - start_time

        # Verify abstract
        assert result is not None
        assert "abstract" in result or "content" in result
        abstract = result.get("abstract") or result.get("content")
        assert len(abstract) > 200
        assert elapsed < 1.0

        # Rate limiting
        time.sleep(0.5)
