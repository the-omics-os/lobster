"""
Integration tests for Vector Search Pipeline.

Tests end-to-end vector search functionality including:
- Ontology matching
- Literature search
- Graceful degradation without API keys
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.integration
class TestVectorSearchIntegration:
    """Integration tests for the vector search pipeline."""

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock SentenceTransformer to avoid downloading models in tests."""
        with patch("sentence_transformers.SentenceTransformer") as mock:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock.return_value = mock_model
            yield mock_model

    @pytest.fixture
    def mock_faiss_index(self):
        """Mock FAISS index."""
        with patch("faiss.IndexFlatL2") as mock:
            mock_index = MagicMock()
            mock_index.ntotal = 0
            mock_index.add = MagicMock()

            def search_impl(query, k):
                # Return valid indices and distances
                n_docs = mock_index.ntotal
                if n_docs == 0:
                    return np.array([[-1]]), np.array([[float("inf")]])
                actual_k = min(k, n_docs)
                indices = np.arange(actual_k).reshape(1, -1)
                distances = np.linspace(0, 1, actual_k).reshape(1, -1)
                return distances, indices

            mock_index.search.side_effect = search_impl
            mock.return_value = mock_index
            yield mock_index

    def test_full_pipeline_faiss_backend(
        self, mock_sentence_transformer, mock_faiss_index
    ):
        """Test full search pipeline with FAISS backend."""
        from lobster.services.search import VectorSearchService
        from lobster.services.search.backends import FAISSBackend
        from lobster.services.search.embeddings import SentenceTransformersProvider

        # Setup embedding provider
        provider = SentenceTransformersProvider()

        # Setup FAISS backend
        backend = FAISSBackend(dimension=384)

        # Add test documents
        mock_faiss_index.ntotal = 3  # Set document count
        backend._documents = {
            "doc1": {"content": "Colorectal cancer treatment", "metadata": {}},
            "doc2": {"content": "Breast cancer research", "metadata": {}},
            "doc3": {"content": "Lung cancer diagnosis", "metadata": {}},
        }
        backend._idx_to_id = {0: "doc1", 1: "doc2", 2: "doc3"}
        backend._id_to_idx = {"doc1": 0, "doc2": 1, "doc3": 2}

        # Create service without reranking
        service = VectorSearchService(
            backend=backend,
            embedding_provider=provider,
            enable_reranking=False,
        )

        # Perform search
        response = service.search(
            query="cancer treatment",
            vector_limit=10,
            final_limit=3,
        )

        # Verify results
        assert response.query == "cancer treatment"
        assert len(response.results) <= 3
        assert response.total_candidates >= 0
        assert response.search_time_ms >= 0
        assert not response.reranking_applied

    def test_graceful_degradation_no_cohere(self, mock_sentence_transformer):
        """Test service works without COHERE_API_KEY."""
        from lobster.services.search import VectorSearchService
        from lobster.services.search.embeddings import SentenceTransformersProvider
        from lobster.services.search.reranker import CohereReranker

        # Clear COHERE_API_KEY
        with patch.dict("os.environ", {}, clear=True):
            provider = SentenceTransformersProvider()
            reranker = CohereReranker()  # Should be unavailable

            assert not reranker.available

            service = VectorSearchService(
                embedding_provider=provider,
                reranker=reranker,
                enable_reranking=True,  # Try to enable
            )

            # Reranking should be disabled
            assert not service.reranking_enabled

    @patch("chromadb.PersistentClient")
    def test_ontology_matching_flow(
        self, mock_chroma_client, mock_sentence_transformer, tmp_path
    ):
        """Test ontology matching end-to-end."""
        from lobster.services.search import VectorSearchService
        from lobster.services.search.embeddings import SentenceTransformersProvider

        # Setup mock ChromaDB
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 100
        mock_collection.query.return_value = {
            "ids": [["MONDO:0005575", "MONDO:0005061"]],
            "documents": [
                [
                    "Colorectal cancer - A cancer that begins in the colon or rectum",
                    "Carcinoma of colon - Malignant tumor of colon",
                ]
            ],
            "metadatas": [
                [
                    {"name": "colorectal cancer", "synonyms": ["CRC", "bowel cancer"]},
                    {"name": "carcinoma of colon", "synonyms": ["colon carcinoma"]},
                ]
            ],
            "distances": [[0.1, 0.2]],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client

        # Create collection directory
        (tmp_path / "mondo").mkdir()

        provider = SentenceTransformersProvider()

        with patch(
            "lobster.services.search.backends.chroma_backend.ChromaBackend.DEFAULT_CACHE_DIR",
            str(tmp_path),
        ):
            service = VectorSearchService(
                embedding_provider=provider,
                enable_reranking=False,
            )

            # Override cache dir for test
            service._ontology_backends = {}

            with patch(
                "lobster.services.search.vector_search_service.ChromaBackend"
            ) as mock_chroma_backend:
                mock_backend = MagicMock()
                mock_backend.name = "chroma:mondo"
                mock_backend.search.return_value = [
                    {
                        "id": "MONDO:0005575",
                        "content": "Colorectal cancer definition",
                        "metadata": {
                            "name": "colorectal cancer",
                            "synonyms": ["CRC"],
                        },
                        "similarity_score": 0.9,
                    },
                ]
                mock_chroma_backend.return_value = mock_backend

                matches = service.match_ontology(
                    term="colon cancer",
                    ontology="mondo",
                    k=3,
                    min_confidence=0.5,
                )

                assert len(matches) >= 1
                assert matches[0].ontology_source == "MONDO"
                assert matches[0].input_term == "colon cancer"

    def test_literature_search_flow(
        self, mock_sentence_transformer, mock_faiss_index
    ):
        """Test literature search end-to-end."""
        from lobster.services.search import VectorSearchService
        from lobster.services.search.backends import FAISSBackend
        from lobster.services.search.embeddings import SentenceTransformersProvider

        provider = SentenceTransformersProvider()
        backend = FAISSBackend(dimension=384)

        # Setup mock documents
        mock_faiss_index.ntotal = 2
        backend._documents = {
            "pmid123": {
                "content": "Single-cell RNA sequencing reveals cellular heterogeneity",
                "metadata": {
                    "title": "scRNA-seq Study",
                    "pmid": "123",
                    "doi": "10.1234/test",
                },
            },
            "pmid456": {
                "content": "Bulk RNA-seq analysis methods comparison",
                "metadata": {
                    "title": "Bulk RNA Study",
                    "pmid": "456",
                },
            },
        }
        backend._idx_to_id = {0: "pmid123", 1: "pmid456"}
        backend._id_to_idx = {"pmid123": 0, "pmid456": 1}

        service = VectorSearchService(
            embedding_provider=provider,
            enable_reranking=False,
        )

        matches = service.search_literature(
            query="single cell sequencing",
            backend=backend,
            k=5,
            use_reranking=False,
        )

        assert len(matches) <= 2
        for match in matches:
            assert hasattr(match, "title")
            assert hasattr(match, "pmid")
            assert hasattr(match, "relevance_score")

    def test_search_response_schema(
        self, mock_sentence_transformer, mock_faiss_index
    ):
        """Test search response matches schema."""
        from lobster.services.search import VectorSearchService
        from lobster.services.search.backends import FAISSBackend
        from lobster.services.search.embeddings import SentenceTransformersProvider
        from lobster.core.schemas.search import SearchResponse, SearchBackend

        provider = SentenceTransformersProvider()
        backend = FAISSBackend(dimension=384)

        mock_faiss_index.ntotal = 1
        backend._documents = {"doc1": {"content": "Test", "metadata": {}}}
        backend._idx_to_id = {0: "doc1"}
        backend._id_to_idx = {"doc1": 0}

        service = VectorSearchService(
            embedding_provider=provider,
            enable_reranking=False,
        )

        response = service.search("test", backend=backend)

        # Verify response is valid SearchResponse
        assert isinstance(response, SearchResponse)
        assert response.backend == SearchBackend.FAISS
        assert response.search_time_ms >= 0
        assert isinstance(response.results, list)


@pytest.mark.integration
class TestVectorSearchWithRealModels:
    """Integration tests that can optionally use real models.

    These tests are skipped by default. Run with:
    pytest tests/integration/test_vector_search.py -m "integration and real_api"
    """

    @pytest.mark.real_api
    @pytest.mark.slow
    def test_real_sentence_transformers(self):
        """Test with real sentence-transformers model."""
        pytest.importorskip("sentence_transformers")

        from lobster.services.search.embeddings import SentenceTransformersProvider

        provider = SentenceTransformersProvider()

        # Generate real embedding
        embedding = provider.embed_text("colorectal cancer treatment")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert not np.allclose(embedding, 0)  # Should not be all zeros

    @pytest.mark.real_api
    @pytest.mark.slow
    def test_real_faiss_search(self):
        """Test with real FAISS index."""
        pytest.importorskip("faiss")
        pytest.importorskip("sentence_transformers")

        from lobster.services.search import VectorSearchService
        from lobster.services.search.backends import FAISSBackend
        from lobster.services.search.embeddings import SentenceTransformersProvider

        provider = SentenceTransformersProvider()
        backend = FAISSBackend(dimension=384)

        # Add real documents with real embeddings
        texts = [
            "Colorectal cancer is a disease of the colon",
            "Breast cancer affects the mammary glands",
            "Lung cancer originates in lung tissue",
        ]

        embeddings = provider.embed_batch(texts)
        ids = [f"doc{i}" for i in range(len(texts))]
        metadatas = [{"index": i} for i in range(len(texts))]

        backend.add_documents(ids, embeddings, texts, metadatas)

        service = VectorSearchService(
            backend=backend,
            embedding_provider=provider,
            enable_reranking=False,
        )

        # Search should find colorectal cancer document first
        response = service.search("colon cancer treatment", backend=backend, final_limit=3)

        assert len(response.results) == 3
        # First result should be most relevant (colorectal cancer)
        assert "colorectal" in response.results[0].content.lower()
