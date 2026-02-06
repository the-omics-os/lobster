"""
Unit tests for Cohere reranker.

Tests the reranking functionality with graceful degradation.
"""

import pytest
from unittest.mock import MagicMock, patch

# Check for optional dependencies
try:
    import cohere
    HAS_COHERE = True
except ImportError:
    HAS_COHERE = False


@pytest.mark.skipif(not HAS_COHERE, reason="cohere not installed (pip install lobster-ai[search])")
class TestCohereReranker:
    """Tests for CohereReranker."""

    def test_initialization_no_api_key(self):
        """Test reranker initializes but marks unavailable without API key."""
        from lobster.services.search.reranker import CohereReranker

        with patch.dict("os.environ", {}, clear=True):
            reranker = CohereReranker()

            assert not reranker.available
            assert reranker.model == "rerank-english-v3.0"

    def test_initialization_with_api_key(self):
        """Test reranker initializes with API key."""
        from lobster.services.search.reranker import CohereReranker

        reranker = CohereReranker(api_key="test-key")

        assert reranker.available
        assert reranker._client is None  # Lazy initialization

    def test_initialization_from_env(self):
        """Test reranker reads API key from environment."""
        from lobster.services.search.reranker import CohereReranker

        with patch.dict("os.environ", {"COHERE_API_KEY": "env-key"}):
            reranker = CohereReranker()

            assert reranker.available

    def test_custom_model(self):
        """Test reranker accepts custom model."""
        from lobster.services.search.reranker import CohereReranker

        reranker = CohereReranker(
            api_key="test-key",
            model="rerank-multilingual-v3.0",
        )

        assert reranker.model == "rerank-multilingual-v3.0"

    def test_rerank_unavailable_returns_original(self):
        """Test rerank returns original order when unavailable."""
        from lobster.services.search.reranker import CohereReranker

        with patch.dict("os.environ", {}, clear=True):
            reranker = CohereReranker()

            documents = [
                {"content": "First"},
                {"content": "Second"},
                {"content": "Third"},
            ]

            result = reranker.rerank("query", documents, top_k=2)

            # Should return first 2 documents in original order
            assert len(result) == 2
            assert result[0]["content"] == "First"
            assert result[1]["content"] == "Second"

    def test_rerank_empty_documents(self):
        """Test rerank handles empty document list."""
        from lobster.services.search.reranker import CohereReranker

        reranker = CohereReranker(api_key="test-key")
        result = reranker.rerank("query", [], top_k=5)

        assert result == []

    def test_rerank_single_document(self):
        """Test rerank handles single document."""
        from lobster.services.search.reranker import CohereReranker

        reranker = CohereReranker(api_key="test-key")
        documents = [{"content": "Only document"}]

        result = reranker.rerank("query", documents, top_k=5)

        assert len(result) == 1
        assert result[0]["content"] == "Only document"
        assert result[0]["relevance_score"] == 1.0

    @patch("cohere.Client")
    def test_rerank_with_api(self, mock_client_class):
        """Test rerank calls Cohere API correctly."""
        from lobster.services.search.reranker import CohereReranker

        # Setup mock
        mock_client = MagicMock()
        mock_result1 = MagicMock()
        mock_result1.index = 1
        mock_result1.relevance_score = 0.95
        mock_result2 = MagicMock()
        mock_result2.index = 0
        mock_result2.relevance_score = 0.85
        mock_response = MagicMock()
        mock_response.results = [mock_result1, mock_result2]
        mock_client.rerank.return_value = mock_response
        mock_client_class.return_value = mock_client

        reranker = CohereReranker(api_key="test-key")

        documents = [
            {"content": "First document"},
            {"content": "Second document"},
        ]

        result = reranker.rerank("test query", documents, top_k=2)

        # Should be reordered based on relevance
        assert len(result) == 2
        assert result[0]["content"] == "Second document"
        assert result[0]["relevance_score"] == 0.95
        assert result[1]["content"] == "First document"
        assert result[1]["relevance_score"] == 0.85

        mock_client.rerank.assert_called_once()

    @patch("cohere.Client")
    def test_rerank_with_min_relevance(self, mock_client_class):
        """Test rerank filters by minimum relevance."""
        from lobster.services.search.reranker import CohereReranker

        # Setup mock
        mock_client = MagicMock()
        mock_result1 = MagicMock()
        mock_result1.index = 0
        mock_result1.relevance_score = 0.9
        mock_result2 = MagicMock()
        mock_result2.index = 1
        mock_result2.relevance_score = 0.3
        mock_response = MagicMock()
        mock_response.results = [mock_result1, mock_result2]
        mock_client.rerank.return_value = mock_response
        mock_client_class.return_value = mock_client

        reranker = CohereReranker(api_key="test-key")

        documents = [
            {"content": "Relevant"},
            {"content": "Not relevant"},
        ]

        result = reranker.rerank(
            "test query",
            documents,
            top_k=2,
            min_relevance=0.5,
        )

        # Only the first result should pass threshold
        assert len(result) == 1
        assert result[0]["content"] == "Relevant"

    @patch("cohere.Client")
    def test_rerank_includes_title(self, mock_client_class):
        """Test rerank includes title in text when available."""
        from lobster.services.search.reranker import CohereReranker

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.results = [
            MagicMock(index=0, relevance_score=0.9),
        ]
        mock_client.rerank.return_value = mock_response
        mock_client_class.return_value = mock_client

        reranker = CohereReranker(api_key="test-key")

        documents = [
            {
                "content": "Content text",
                "metadata": {"title": "Document Title"},
            },
        ]

        reranker.rerank("query", documents, top_k=1)

        # Check that title was included in the text
        call_args = mock_client.rerank.call_args
        texts = call_args.kwargs["documents"]
        assert "Document Title" in texts[0]

    @patch("cohere.Client")
    def test_rerank_api_error_returns_original(self, mock_client_class):
        """Test rerank returns original order on API error."""
        from lobster.services.search.reranker import CohereReranker

        mock_client = MagicMock()
        mock_client.rerank.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client

        reranker = CohereReranker(api_key="test-key")

        documents = [
            {"content": "First"},
            {"content": "Second"},
        ]

        result = reranker.rerank("query", documents, top_k=2)

        # Should gracefully fall back to original order
        assert len(result) == 2
        assert result[0]["content"] == "First"

    def test_rerank_texts_convenience(self):
        """Test rerank_texts convenience method."""
        from lobster.services.search.reranker import CohereReranker

        with patch.dict("os.environ", {}, clear=True):
            reranker = CohereReranker()

            texts = ["First text", "Second text"]
            result = reranker.rerank_texts("query", texts, top_k=2)

            assert len(result) == 2
            assert result[0]["content"] == "First text"
