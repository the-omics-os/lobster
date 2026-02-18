"""
Unit tests for VectorSearchService.

Uses mock embedder and mock backend to test the orchestration layer
without requiring real chromadb, torch, or sentence-transformers.
"""

import hashlib
from typing import Any

import pytest

from lobster.core.vector.backends.base import BaseVectorBackend
from lobster.core.vector.config import VectorSearchConfig
from lobster.core.vector.embeddings.base import BaseEmbedder
from lobster.core.vector.service import VectorSearchService


# ---------------------------------------------------------------------------
# Mock implementations for testing
# ---------------------------------------------------------------------------


class MockEmbedder(BaseEmbedder):
    """Deterministic mock embedder for unit tests -- no torch required."""

    DIMENSIONS = 768

    def embed_text(self, text: str) -> list[float]:
        h = hashlib.md5(text.encode()).hexdigest()
        return [int(c, 16) / 15.0 for c in h] * 48  # 768 dims

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]

    @property
    def dimensions(self) -> int:
        return self.DIMENSIONS


class MockVectorBackend(BaseVectorBackend):
    """In-memory mock backend for unit tests.

    Supports configurable search results via set_results().
    Default: returns 3 mock ontology matches.
    """

    def __init__(self) -> None:
        self._collections: dict[str, list[dict]] = {}
        self._search_results: dict[str, Any] | None = None
        self._last_search_n_results: int | None = None

    def set_results(self, results: dict[str, Any]) -> None:
        """Configure what search() returns."""
        self._search_results = results

    def add_documents(
        self,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        if collection_name not in self._collections:
            self._collections[collection_name] = []
        for i, doc_id in enumerate(ids):
            self._collections[collection_name].append(
                {
                    "id": doc_id,
                    "embedding": embeddings[i],
                    "document": documents[i] if documents else None,
                    "metadata": metadatas[i] if metadatas else None,
                }
            )

    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> dict[str, Any]:
        self._last_search_n_results = n_results

        if self._search_results is not None:
            return self._search_results

        # Default mock results: 3 ontology matches
        return {
            "ids": [["CL:0000084", "CL:0000625", "CL:0000624"]],
            "distances": [[0.1, 0.3, 0.5]],
            "documents": [["T cell", "CD8-positive T cell", "CD4-positive T cell"]],
            "metadatas": [
                [
                    {"ontology_id": "CL:0000084", "source": "CL"},
                    {"ontology_id": "CL:0000625", "source": "CL"},
                    {"ontology_id": "CL:0000624", "source": "CL"},
                ]
            ],
        }

    def delete(self, collection_name: str, ids: list[str]) -> None:
        if collection_name in self._collections:
            self._collections[collection_name] = [
                d
                for d in self._collections[collection_name]
                if d["id"] not in ids
            ]

    def count(self, collection_name: str) -> int:
        return len(self._collections.get(collection_name, []))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    return MockEmbedder()


@pytest.fixture
def mock_backend() -> MockVectorBackend:
    return MockVectorBackend()


@pytest.fixture
def config() -> VectorSearchConfig:
    return VectorSearchConfig()


@pytest.fixture
def service(config, mock_backend, mock_embedder) -> VectorSearchService:
    return VectorSearchService(
        config=config,
        backend=mock_backend,
        embedder=mock_embedder,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQueryBasic:
    """Basic query() behavior tests."""

    def test_query_returns_list_of_dicts(self, service):
        """query() should return a list of dicts."""
        results = service.query("T cell", "cell_ontology")
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_query_result_shape(self, service):
        """Each result dict has required keys: term, ontology_id, score, metadata, distance_metric."""
        results = service.query("T cell", "cell_ontology")
        assert len(results) > 0
        required_keys = {"term", "ontology_id", "score", "metadata", "distance_metric"}
        for r in results:
            assert set(r.keys()) == required_keys

    def test_query_returns_correct_terms(self, service):
        """Results should contain the mocked document texts."""
        results = service.query("T cell", "cell_ontology")
        terms = [r["term"] for r in results]
        assert "T cell" in terms
        assert "CD8-positive T cell" in terms


class TestDistanceConversion:
    """Tests for distance-to-similarity conversion."""

    def test_query_distance_to_similarity_conversion(self, service, mock_backend):
        """Distances [0.1, 0.3, 0.8] should produce scores [0.9, 0.7, 0.2]."""
        mock_backend.set_results(
            {
                "ids": [["id1", "id2", "id3"]],
                "distances": [[0.1, 0.3, 0.8]],
                "documents": [["term1", "term2", "term3"]],
                "metadatas": [
                    [
                        {"ontology_id": "T:001"},
                        {"ontology_id": "T:002"},
                        {"ontology_id": "T:003"},
                    ]
                ],
            }
        )
        results = service.query("test", "collection")
        scores = [r["score"] for r in results]
        assert scores == [0.9, 0.7, 0.2]

    def test_query_score_clamped_0_1(self, service, mock_backend):
        """Distances outside [0,1] should produce scores clamped to [0,1]."""
        mock_backend.set_results(
            {
                "ids": [["id1", "id2"]],
                "distances": [[-0.1, 1.5]],
                "documents": [["term1", "term2"]],
                "metadatas": [
                    [{"ontology_id": "T:001"}, {"ontology_id": "T:002"}]
                ],
            }
        )
        results = service.query("test", "collection")
        scores = [r["score"] for r in results]
        # distance -0.1 -> 1 - (-0.1) = 1.1 -> clamped to 1.0
        # distance 1.5 -> 1 - 1.5 = -0.5 -> clamped to 0.0
        assert scores == [1.0, 0.0]


class TestTopK:
    """Tests for top_k behavior."""

    def test_query_default_top_k_5(self, service, mock_backend):
        """query() without top_k should use default of 5."""
        service.query("test", "collection")
        assert mock_backend._last_search_n_results == 5

    def test_query_custom_top_k(self, service, mock_backend):
        """query() with top_k=10 should pass 10 to backend."""
        service.query("test", "collection", top_k=10)
        assert mock_backend._last_search_n_results == 10


class TestBatchQuery:
    """Tests for query_batch()."""

    def test_query_batch_returns_list_of_lists(self, service):
        """query_batch() returns one result list per input text."""
        results = service.query_batch(["T cell", "neuron"], "cell_ontology")
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)

    def test_query_batch_each_result_has_correct_shape(self, service):
        """Each result in the batch should have the same shape as single query."""
        results = service.query_batch(["T cell"], "cell_ontology")
        assert len(results) == 1
        required_keys = {"term", "ontology_id", "score", "metadata", "distance_metric"}
        for match in results[0]:
            assert set(match.keys()) == required_keys


class TestEmptyResults:
    """Tests for empty result handling."""

    def test_query_empty_collection(self, service, mock_backend):
        """Empty backend results should return empty list."""
        mock_backend.set_results(
            {
                "ids": [[]],
                "distances": [[]],
                "documents": [[]],
                "metadatas": [[]],
            }
        )
        results = service.query("test", "empty_collection")
        assert results == []

    def test_query_missing_keys_in_raw(self, service, mock_backend):
        """Completely empty raw results should return empty list."""
        mock_backend.set_results({})
        results = service.query("test", "collection")
        assert results == []


class TestDistanceMetric:
    """Tests for distance_metric field."""

    def test_distance_metric_in_results(self, service):
        """Every match dict should have distance_metric='cosine'."""
        results = service.query("T cell", "cell_ontology")
        for r in results:
            assert r["distance_metric"] == "cosine"


class TestServiceConstruction:
    """Tests for service construction and lazy initialization."""

    def test_service_accepts_injected_backend_and_embedder(
        self, mock_backend, mock_embedder
    ):
        """Injected backend/embedder should be used directly."""
        service = VectorSearchService(
            backend=mock_backend, embedder=mock_embedder
        )
        # No factory calls needed, should work directly
        results = service.query("test", "collection")
        assert isinstance(results, list)

    def test_service_lazy_initialization(self):
        """Without injection, _backend and _embedder should be None until first query."""
        service = VectorSearchService()
        assert service._backend is None
        assert service._embedder is None
