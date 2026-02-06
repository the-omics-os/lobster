"""
Vector Search Schema Definitions

Unified schemas for vector search operations supporting both
literature search and ontology standardization use cases.

Design follows BioAgents pattern with two-stage pipeline:
1. Vector search (broad, fast)
2. Reranking (precise, optional)
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SearchBackend(str, Enum):
    """Available vector search backends."""

    CHROMA = "chroma"  # ChromaDB (ontology, persistent)
    FAISS = "faiss"  # FAISS (literature, ephemeral)
    PGVECTOR = "pgvector"  # PostgreSQL (cloud, future)


class EmbeddingProvider(str, Enum):
    """Available embedding providers."""

    SENTENCE_TRANSFORMERS = "sentence_transformers"  # Local, free, 384 dims
    OPENAI = "openai"  # API, paid, 1536 dims


class SearchResult(BaseModel):
    """
    Single search result from vector search.

    Attributes:
        id: Unique identifier for the document
        content: Full text content of the document
        metadata: Additional metadata (source, ontology_id, etc.)
        similarity_score: Cosine similarity from vector search (0.0-1.0)
        relevance_score: Reranker score, only present if reranking enabled
    """

    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    similarity_score: float = Field(ge=0.0, le=1.0)
    relevance_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Reranker score (only if reranking enabled)",
    )


class SearchResponse(BaseModel):
    """
    Complete search response with results and metadata.

    Provides transparency about the search pipeline configuration
    and performance metrics for debugging and optimization.
    """

    query: str
    results: List[SearchResult]
    backend: SearchBackend
    embedding_provider: EmbeddingProvider
    reranking_applied: bool = False
    total_candidates: int  # Before reranking
    search_time_ms: int
    rerank_time_ms: Optional[int] = None


class OntologyMatch(BaseModel):
    """
    Ontology matching result.

    Compatible with existing DiseaseOntologyService API.
    Enables seamless Phase 1 (keyword) to Phase 2 (embedding) migration.

    Attributes:
        input_term: Original query term from user
        matched_term: Canonical term from ontology
        ontology_id: Formal ontology identifier (e.g., "MONDO:0005575")
        ontology_source: Ontology name (e.g., "MONDO", "UBERON", "CL")
        confidence: Match confidence (0.0-1.0)
        synonyms: Known synonyms for the matched term
        definition: Ontology definition text
    """

    input_term: str
    matched_term: str
    ontology_id: str
    ontology_source: str
    confidence: float = Field(ge=0.0, le=1.0)
    synonyms: List[str] = Field(default_factory=list)
    definition: Optional[str] = None


class LiteratureMatch(BaseModel):
    """
    Literature search result.

    Designed for research_agent semantic search over cached abstracts.

    Attributes:
        query: Original search query
        title: Publication title
        abstract: Publication abstract text
        doi: Digital Object Identifier (optional)
        pmid: PubMed ID (optional)
        relevance_score: Combined similarity/reranking score
        matched_terms: Terms that contributed to the match
    """

    query: str
    title: str
    abstract: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    relevance_score: float = Field(ge=0.0, le=1.0)
    matched_terms: List[str] = Field(default_factory=list)


class VectorSearchConfig(BaseModel):
    """
    Configuration for vector search service.

    Environment variable mapping:
    - embedding_provider: LOBSTER_EMBEDDING_PROVIDER
    - enable_reranking: LOBSTER_ENABLE_RERANKING
    - ontology_cache_dir: LOBSTER_ONTOLOGY_CACHE_DIR
    """

    embedding_provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    enable_reranking: bool = True
    vector_search_limit: int = Field(default=20, ge=1, le=100)
    final_result_limit: int = Field(default=5, ge=1, le=50)
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    reranker_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    cache_ttl_seconds: int = Field(default=300, ge=0)  # 5 minutes
    ontology_cache_dir: str = "~/.lobster/ontology_cache"
