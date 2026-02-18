"""
Unit tests for vector search Pydantic schemas and enums.

Tests the type system defined in lobster.core.schemas.search:
OntologyMatch, SearchResult, LiteratureMatch, SearchResponse,
SearchBackend, EmbeddingProvider, RerankerType.
"""

import pytest
from pydantic import ValidationError

from lobster.core.schemas.search import (
    EmbeddingProvider,
    LiteratureMatch,
    OntologyMatch,
    RerankerType,
    SearchBackend,
    SearchResponse,
    SearchResult,
)


class TestOntologyMatch:
    """Tests for the OntologyMatch Pydantic model."""

    def test_ontology_match_creation(self):
        """Create OntologyMatch with all fields and verify model_dump()."""
        match = OntologyMatch(
            term="colorectal carcinoma",
            ontology_id="MONDO:0005575",
            score=0.9234,
            metadata={"synonyms": ["bowel cancer"], "source": "MONDO"},
            distance_metric="cosine",
        )
        dump = match.model_dump()
        assert dump["term"] == "colorectal carcinoma"
        assert dump["ontology_id"] == "MONDO:0005575"
        assert dump["score"] == 0.9234
        assert dump["metadata"]["synonyms"] == ["bowel cancer"]
        assert dump["distance_metric"] == "cosine"

    def test_ontology_match_defaults(self):
        """Create with only required fields, verify defaults."""
        match = OntologyMatch(
            term="T cell",
            ontology_id="CL:0000084",
            score=0.85,
        )
        assert match.metadata == {}
        assert match.distance_metric == "cosine"

    def test_ontology_match_score_bounds_valid(self):
        """Score of 0.0 and 1.0 should be accepted."""
        match_zero = OntologyMatch(
            term="test", ontology_id="TEST:0001", score=0.0
        )
        assert match_zero.score == 0.0

        match_one = OntologyMatch(
            term="test", ontology_id="TEST:0001", score=1.0
        )
        assert match_one.score == 1.0

    def test_ontology_match_score_rejects_negative(self):
        """Score below 0 should be rejected by Pydantic ge=0.0."""
        with pytest.raises(ValidationError):
            OntologyMatch(
                term="test", ontology_id="TEST:0001", score=-0.1
            )

    def test_ontology_match_score_rejects_above_one(self):
        """Score above 1 should be rejected by Pydantic le=1.0."""
        with pytest.raises(ValidationError):
            OntologyMatch(
                term="test", ontology_id="TEST:0001", score=1.1
            )

    def test_ontology_match_score_rounding(self):
        """Score should be rounded to 4 decimal places via model_post_init."""
        match = OntologyMatch(
            term="test", ontology_id="TEST:0001", score=0.123456789
        )
        assert match.score == 0.1235


class TestSearchResult:
    """Tests for the SearchResult Pydantic model."""

    def test_search_result_creation(self):
        """Create SearchResult with query, collection, matches, top_k."""
        matches = [
            OntologyMatch(
                term="heart attack",
                ontology_id="MONDO:0005068",
                score=0.95,
            ),
            OntologyMatch(
                term="myocardial infarction",
                ontology_id="MONDO:0005068",
                score=0.88,
            ),
        ]
        result = SearchResult(
            query="heart attack",
            collection="mondo_v2024_01",
            matches=matches,
            top_k=5,
        )
        assert result.query == "heart attack"
        assert result.collection == "mondo_v2024_01"
        assert len(result.matches) == 2
        assert result.top_k == 5
        assert result.total_in_collection is None

    def test_search_result_with_total(self):
        """SearchResult can include total_in_collection for diagnostics."""
        result = SearchResult(
            query="test",
            collection="test_collection",
            matches=[],
            top_k=5,
            total_in_collection=60000,
        )
        assert result.total_in_collection == 60000


class TestSearchResponse:
    """Tests for the SearchResponse Pydantic model."""

    def test_search_response_creation(self):
        """Create SearchResponse with results, backend, embedding_provider."""
        result = SearchResult(
            query="test",
            collection="test_collection",
            matches=[],
            top_k=5,
        )
        response = SearchResponse(
            results=[result],
            backend=SearchBackend.chromadb,
            embedding_provider=EmbeddingProvider.sapbert,
        )
        assert len(response.results) == 1
        assert response.backend == SearchBackend.chromadb
        assert response.embedding_provider == EmbeddingProvider.sapbert
        assert response.reranker is None


class TestLiteratureMatch:
    """Tests for the LiteratureMatch Pydantic model."""

    def test_literature_match_creation(self):
        """Create LiteratureMatch and verify fields."""
        match = LiteratureMatch(
            title="CRISPR-Cas9 gene editing for sickle cell disease",
            pmid="12345678",
            score=0.92,
            metadata={"journal": "Nature", "year": 2024},
        )
        assert match.title == "CRISPR-Cas9 gene editing for sickle cell disease"
        assert match.pmid == "12345678"
        assert match.score == 0.92
        assert match.metadata["journal"] == "Nature"
        assert match.distance_metric == "cosine"

    def test_literature_match_defaults(self):
        """LiteratureMatch with only required fields."""
        match = LiteratureMatch(
            title="Some paper",
            score=0.5,
        )
        assert match.pmid is None
        assert match.metadata == {}
        assert match.distance_metric == "cosine"


class TestEnums:
    """Tests for the vector search enums."""

    def test_search_backend_enum_values(self):
        """Verify all 3 SearchBackend members."""
        assert SearchBackend.chromadb.value == "chromadb"
        assert SearchBackend.faiss.value == "faiss"
        assert SearchBackend.pgvector.value == "pgvector"
        assert len(SearchBackend) == 3

    def test_embedding_provider_enum_values(self):
        """Verify all 3 EmbeddingProvider members."""
        assert EmbeddingProvider.sapbert.value == "sapbert"
        assert EmbeddingProvider.minilm.value == "minilm"
        assert EmbeddingProvider.openai.value == "openai"
        assert len(EmbeddingProvider) == 3

    def test_reranker_type_enum_values(self):
        """Verify all 3 RerankerType members."""
        assert RerankerType.cross_encoder.value == "cross_encoder"
        assert RerankerType.cohere.value == "cohere"
        assert RerankerType.none.value == "none"
        assert len(RerankerType) == 3
