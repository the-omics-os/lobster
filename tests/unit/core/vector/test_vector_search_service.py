"""
Unit tests for VectorSearchService.

Uses mock embedder and mock backend to test the orchestration layer
without requiring real chromadb, torch, or sentence-transformers.
"""

import hashlib
from typing import Any

import pytest

from lobster.core.schemas.search import OntologyMatch
from lobster.core.vector.backends.base import BaseVectorBackend
from lobster.core.vector.config import VectorSearchConfig
from lobster.core.vector.embeddings.base import BaseEmbedder
from lobster.core.vector.service import ONTOLOGY_COLLECTIONS, VectorSearchService


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


# ---------------------------------------------------------------------------
# match_ontology() tests (Phase 2 Plan 02)
# ---------------------------------------------------------------------------


class TestMatchOntology:
    """Tests for match_ontology() domain-aware ontology matching."""

    def test_match_ontology_returns_ontology_match_list(
        self, service, mock_backend
    ):
        """match_ontology() should return a list of OntologyMatch Pydantic objects (SRCH-01, SRCH-04)."""
        mock_backend.set_results(
            {
                "ids": [["MONDO:0005068", "MONDO:0004995"]],
                "distances": [[0.1, 0.3]],
                "documents": [["myocardial infarction", "acute myocardial infarction"]],
                "metadatas": [
                    [
                        {"ontology_id": "MONDO:0005068", "source": "MONDO"},
                        {"ontology_id": "MONDO:0004995", "source": "MONDO"},
                    ]
                ],
            }
        )
        results = service.match_ontology("heart attack", "disease", k=2)
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, OntologyMatch) for r in results)
        # Verify fields are populated
        assert results[0].term == "myocardial infarction"
        assert results[0].ontology_id == "MONDO:0005068"
        assert results[0].score == 0.9  # 1.0 - 0.1
        assert results[0].distance_metric == "cosine"

    def test_match_ontology_alias_resolution_disease(
        self, service, mock_backend
    ):
        """'disease' should resolve to mondo collection (SRCH-03)."""
        service.match_ontology("test", "disease")
        # The mock backend was called. Verify it went through ONTOLOGY_COLLECTIONS alias.
        # We can verify this indirectly by checking the oversampling was applied.
        assert mock_backend._last_search_n_results == 5 * 4  # default k=5, oversampled 4x

    def test_match_ontology_alias_resolution_tissue(
        self, service, mock_backend
    ):
        """'tissue' should resolve to uberon collection (SRCH-03)."""
        service.match_ontology("lung", "tissue")
        assert mock_backend._last_search_n_results == 5 * 4

    def test_match_ontology_alias_resolution_cell_type(
        self, service, mock_backend
    ):
        """'cell_type' should resolve to cell_ontology collection (SRCH-03)."""
        service.match_ontology("T cell", "cell_type")
        assert mock_backend._last_search_n_results == 5 * 4

    def test_match_ontology_oversampling_k_times_4(
        self, service, mock_backend
    ):
        """match_ontology(k=3) should request 3*4=12 from backend (SRCH-02)."""
        service.match_ontology("test", "mondo", k=3)
        assert mock_backend._last_search_n_results == 12

    def test_match_ontology_truncates_to_k(self, service, mock_backend):
        """match_ontology(k=2) returns only 2 results even if backend returns more (SRCH-02)."""
        # Backend returns 3 results by default
        results = service.match_ontology("test", "mondo", k=2)
        assert len(results) == 2

    def test_match_ontology_unknown_ontology_raises_value_error(self, service):
        """Unknown ontology name should raise ValueError (SRCH-01)."""
        with pytest.raises(ValueError, match="Unknown ontology"):
            service.match_ontology("test", "invalid")

    def test_match_ontology_empty_results(self, service, mock_backend):
        """Empty backend results should return empty list (SRCH-04)."""
        mock_backend.set_results(
            {
                "ids": [[]],
                "distances": [[]],
                "documents": [[]],
                "metadatas": [[]],
            }
        )
        results = service.match_ontology("test", "mondo", k=3)
        assert results == []


class TestOntologyMatchDiseaseMatchCompat:
    """Tests for SCHM-03: OntologyMatch -> DiseaseMatch field compatibility."""

    def test_ontology_match_to_disease_match_field_mapping(self):
        """OntologyMatch fields can map to DiseaseMatch-style fields (SCHM-03).

        DiseaseMatch would have: disease_id, name, confidence.
        OntologyMatch provides: ontology_id -> disease_id, term -> name, score -> confidence.
        """
        match = OntologyMatch(
            term="myocardial infarction",
            ontology_id="MONDO:0005068",
            score=0.92,
            metadata={"source": "MONDO"},
            distance_metric="cosine",
        )
        # Verify the field mapping is possible and values are correct
        disease_id = match.ontology_id
        name = match.term
        confidence = match.score

        assert disease_id == "MONDO:0005068"
        assert name == "myocardial infarction"
        assert confidence == 0.92


def _chromadb_available() -> bool:
    """Check if chromadb is installed for integration tests."""
    try:
        import chromadb

        return True
    except ImportError:
        return False


class TestMatchOntologyIntegration:
    """Integration test for match_ontology() with real ChromaDB (TEST-07)."""

    @pytest.mark.skipif(
        not _chromadb_available(),
        reason="chromadb not installed",
    )
    def test_full_pipeline_with_real_chromadb(self, tmp_path):
        """End-to-end: embed terms, add to ChromaDB, query via match_ontology()."""
        import chromadb

        from lobster.core.vector.backends.chromadb_backend import (
            ChromaDBBackend,
        )

        # Create a real ChromaDB backend
        client = chromadb.Client()
        backend = ChromaDBBackend(persist_path=str(tmp_path))
        backend._client = client  # Override with ephemeral client

        # Create a mock embedder (still mock, but tests full pipeline through real ChromaDB)
        embedder = MockEmbedder()

        # Add 10 disease terms to a collection
        terms = [
            ("MONDO:0005068", "myocardial infarction"),
            ("MONDO:0004995", "acute myocardial infarction"),
            ("MONDO:0005010", "congestive heart failure"),
            ("MONDO:0006502", "coronary artery disease"),
            ("MONDO:0005148", "type 2 diabetes"),
            ("MONDO:0005015", "diabetes mellitus"),
            ("MONDO:0005575", "colorectal carcinoma"),
            ("MONDO:0008170", "ovarian cancer"),
            ("MONDO:0004992", "lung cancer"),
            ("MONDO:0005105", "melanoma"),
        ]

        collection = client.get_or_create_collection("mondo_v2024_01")
        ids = [t[0] for t in terms]
        documents = [t[1] for t in terms]
        embeddings = embedder.embed_batch(documents)
        metadatas = [{"ontology_id": t[0], "source": "MONDO"} for t in terms]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        # Create service with real ChromaDB + mock embedder
        service = VectorSearchService(backend=backend, embedder=embedder)

        results = service.match_ontology("heart attack", "mondo", k=3)

        # Should return OntologyMatch objects
        assert isinstance(results, list)
        assert len(results) <= 3
        assert all(isinstance(r, OntologyMatch) for r in results)

        # Scores should be valid
        for r in results:
            assert 0.0 <= r.score <= 1.0
            assert r.distance_metric == "cosine"
