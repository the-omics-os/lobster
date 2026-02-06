"""
Unit tests for VectorSearchService.

Tests the main unified search service with all features.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestVectorSearchService:
    """Tests for VectorSearchService."""

    @pytest.fixture
    def mock_embedding_provider(self):
        """Create mock embedding provider."""
        provider = MagicMock()
        provider.model_name = "test-model"
        provider.embedding_dimension = 384
        provider.embed_text.return_value = np.zeros(384)
        provider.embed_batch.return_value = [np.zeros(384)]
        return provider

    @pytest.fixture
    def mock_backend(self):
        """Create mock vector backend."""
        backend = MagicMock()
        backend.name = "mock:test"
        backend.search.return_value = [
            {
                "id": "doc1",
                "content": "Test content 1",
                "metadata": {"key": "value1"},
                "similarity_score": 0.9,
            },
            {
                "id": "doc2",
                "content": "Test content 2",
                "metadata": {"key": "value2"},
                "similarity_score": 0.8,
            },
        ]
        return backend

    @pytest.fixture
    def mock_reranker(self):
        """Create mock reranker."""
        reranker = MagicMock()
        reranker.available = True
        reranker.rerank.return_value = [
            {
                "id": "doc1",
                "content": "Test content 1",
                "metadata": {"key": "value1"},
                "similarity_score": 0.9,
                "relevance_score": 0.95,
            },
        ]
        return reranker

    def test_initialization(self, mock_embedding_provider, mock_reranker):
        """Test service initializes correctly."""
        from lobster.services.search.vector_search_service import VectorSearchService

        service = VectorSearchService(
            embedding_provider=mock_embedding_provider,
            reranker=mock_reranker,
        )

        assert service.embedding_provider == mock_embedding_provider
        assert service.reranking_enabled

    def test_initialization_default_provider(self):
        """Test service initializes with default provider."""
        from lobster.services.search.vector_search_service import VectorSearchService

        with patch(
            "lobster.services.search.vector_search_service.get_sentence_transformers_provider"
        ) as mock_get:
            mock_provider = MagicMock()
            mock_provider.model_name = "all-MiniLM-L6-v2"
            mock_get.return_value = mock_provider

            service = VectorSearchService()

            assert service._embedding_provider == mock_provider

    def test_search_requires_backend(self, mock_embedding_provider, mock_reranker):
        """Test search raises error without backend."""
        from lobster.services.search.vector_search_service import VectorSearchService

        service = VectorSearchService(
            embedding_provider=mock_embedding_provider,
            reranker=mock_reranker,
        )

        with pytest.raises(ValueError) as exc_info:
            service.search("test query")

        assert "No backend configured" in str(exc_info.value)

    def test_search_basic(
        self, mock_embedding_provider, mock_backend, mock_reranker
    ):
        """Test basic search functionality."""
        from lobster.services.search.vector_search_service import VectorSearchService

        mock_reranker.available = False  # Disable reranking for this test

        service = VectorSearchService(
            embedding_provider=mock_embedding_provider,
            reranker=mock_reranker,
            enable_reranking=False,
        )

        response = service.search("test query", backend=mock_backend)

        assert response.query == "test query"
        assert len(response.results) > 0
        assert response.total_candidates == 2
        assert response.search_time_ms >= 0
        assert not response.reranking_applied

    def test_search_with_reranking(
        self, mock_embedding_provider, mock_backend, mock_reranker
    ):
        """Test search with reranking enabled."""
        from lobster.services.search.vector_search_service import VectorSearchService

        service = VectorSearchService(
            embedding_provider=mock_embedding_provider,
            reranker=mock_reranker,
            enable_reranking=True,
        )

        response = service.search("test query", backend=mock_backend)

        assert response.reranking_applied
        assert response.rerank_time_ms is not None
        mock_reranker.rerank.assert_called_once()

    def test_search_caching(
        self, mock_embedding_provider, mock_backend, mock_reranker
    ):
        """Test search results are cached."""
        from lobster.services.search.vector_search_service import VectorSearchService

        mock_reranker.available = False

        service = VectorSearchService(
            embedding_provider=mock_embedding_provider,
            reranker=mock_reranker,
            enable_reranking=False,
        )

        # First search
        response1 = service.search("test query", backend=mock_backend)

        # Second search with same query
        response2 = service.search("test query", backend=mock_backend)

        # Backend should only be called once (second is cached)
        assert mock_backend.search.call_count == 1
        assert response1.query == response2.query

    def test_search_cache_clear(
        self, mock_embedding_provider, mock_backend, mock_reranker
    ):
        """Test cache can be cleared."""
        from lobster.services.search.vector_search_service import VectorSearchService

        mock_reranker.available = False

        service = VectorSearchService(
            embedding_provider=mock_embedding_provider,
            reranker=mock_reranker,
            enable_reranking=False,
        )

        service.search("test query", backend=mock_backend)
        service.clear_cache()
        service.search("test query", backend=mock_backend)

        # Backend should be called twice (cache was cleared)
        assert mock_backend.search.call_count == 2

    def test_match_ontology(
        self, mock_embedding_provider, mock_reranker
    ):
        """Test ontology matching."""
        from lobster.services.search.vector_search_service import VectorSearchService

        # Create mock ChromaBackend
        with patch(
            "lobster.services.search.vector_search_service.ChromaBackend"
        ) as mock_chroma:
            mock_backend = MagicMock()
            mock_backend.name = "chroma:mondo"
            mock_backend.search.return_value = [
                {
                    "id": "MONDO:0005575",
                    "content": "Colorectal cancer definition",
                    "metadata": {"name": "colorectal cancer", "synonyms": []},
                    "similarity_score": 0.9,
                },
            ]
            mock_chroma.return_value = mock_backend

            service = VectorSearchService(
                embedding_provider=mock_embedding_provider,
                reranker=mock_reranker,
                enable_reranking=True,
            )

            matches = service.match_ontology(
                term="colorectal cancer",
                ontology="mondo",
                k=3,
                min_confidence=0.5,
            )

            assert len(matches) == 1
            assert matches[0].input_term == "colorectal cancer"
            assert matches[0].ontology_source == "MONDO"

    def test_match_ontology_unknown_ontology(
        self, mock_embedding_provider, mock_reranker
    ):
        """Test match_ontology raises error for unknown ontology."""
        from lobster.services.search.vector_search_service import VectorSearchService

        service = VectorSearchService(
            embedding_provider=mock_embedding_provider,
            reranker=mock_reranker,
        )

        with pytest.raises(ValueError) as exc_info:
            service.match_ontology("term", ontology="unknown")

        assert "Unknown ontology" in str(exc_info.value)

    def test_search_literature(
        self, mock_embedding_provider, mock_backend, mock_reranker
    ):
        """Test literature search."""
        from lobster.services.search.vector_search_service import VectorSearchService

        # Update mock backend response
        mock_backend.search.return_value = [
            {
                "id": "pmid123",
                "content": "Study abstract text",
                "metadata": {
                    "title": "Study Title",
                    "pmid": "123",
                    "doi": "10.1234/test",
                },
                "similarity_score": 0.9,
            },
        ]

        service = VectorSearchService(
            embedding_provider=mock_embedding_provider,
            reranker=mock_reranker,
            enable_reranking=False,
        )

        matches = service.search_literature(
            query="single cell RNA-seq",
            backend=mock_backend,
            k=5,
            use_reranking=False,
        )

        assert len(matches) == 1
        assert matches[0].title == "Study Title"
        assert matches[0].pmid == "123"
        assert matches[0].query == "single cell RNA-seq"

    def test_create_search_ir(self, mock_embedding_provider, mock_reranker):
        """Test IR generation for provenance."""
        from lobster.services.search.vector_search_service import VectorSearchService

        service = VectorSearchService(
            embedding_provider=mock_embedding_provider,
            reranker=mock_reranker,
        )

        ir = service.create_search_ir(
            query="test query",
            results_count=5,
            search_type="ontology",
        )

        assert ir.operation == "vector_search_ontology"
        assert "test query" in ir.description
        assert ir.exportable is False

    def test_backend_type_detection(self, mock_embedding_provider, mock_reranker):
        """Test backend type detection."""
        from lobster.services.search.vector_search_service import VectorSearchService
        from lobster.core.schemas.search import SearchBackend

        service = VectorSearchService(
            embedding_provider=mock_embedding_provider,
            reranker=mock_reranker,
        )

        mock_backend = MagicMock()

        mock_backend.name = "chroma:test"
        assert service._detect_backend_type(mock_backend) == SearchBackend.CHROMA

        mock_backend.name = "faiss:memory"
        assert service._detect_backend_type(mock_backend) == SearchBackend.FAISS

        mock_backend.name = "pgvector:postgres"
        assert service._detect_backend_type(mock_backend) == SearchBackend.PGVECTOR


class TestVectorSearchServiceOntologyMapping:
    """Tests for ontology collection mapping."""

    def test_ontology_aliases(self):
        """Test ontology aliases map correctly."""
        from lobster.services.search.vector_search_service import VectorSearchService

        mapping = VectorSearchService.ONTOLOGY_COLLECTIONS

        # Disease aliases
        assert mapping["mondo"] == "mondo"
        assert mapping["disease"] == "mondo"

        # Tissue aliases
        assert mapping["uberon"] == "uberon"
        assert mapping["tissue"] == "uberon"

        # Cell type aliases
        assert mapping["cl"] == "cell_ontology"
        assert mapping["cell_type"] == "cell_ontology"
        assert mapping["cell_ontology"] == "cell_ontology"

        # Organism aliases
        assert mapping["ncbi_taxonomy"] == "ncbi_taxonomy"
        assert mapping["organism"] == "ncbi_taxonomy"
        assert mapping["taxonomy"] == "ncbi_taxonomy"
