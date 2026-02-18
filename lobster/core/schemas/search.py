"""
Pydantic schemas and enums for vector search results.

Defines the type system for semantic search infrastructure including
ontology matching, literature search, and backend/embedder configuration.
These models serve as the interface contracts between vector search backends,
embedding providers, and consuming agents (annotation, metadata, research).

Part of Phase 1 (Foundation) — used by all subsequent vector search phases.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SearchBackend(str, Enum):
    """Supported vector database backends for semantic search."""

    chromadb = "chromadb"
    faiss = "faiss"
    pgvector = "pgvector"


class EmbeddingProvider(str, Enum):
    """Supported embedding model providers for text vectorization."""

    sapbert = "sapbert"
    minilm = "minilm"
    openai = "openai"


class RerankerType(str, Enum):
    """Supported reranker strategies for result refinement."""

    cross_encoder = "cross_encoder"
    cohere = "cohere"
    none = "none"


class OntologyMatch(BaseModel):
    """
    A single ontology term match from vector similarity search.

    Represents one result from querying a biomedical ontology collection
    (e.g., MONDO disease terms, CL cell types, UBERON tissues).
    Flat structure for simplicity — no nested result hierarchies.
    """

    term: str = Field(
        description="The matched ontology term text (e.g., 'colorectal carcinoma')"
    )
    ontology_id: str = Field(
        description="Ontology identifier (e.g., 'MONDO:0005575', 'CL:0000084')"
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Similarity score between 0 and 1, rounded to 4 decimal places",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata (synonyms, cross-references, source ontology)",
    )
    distance_metric: str = Field(
        default="cosine",
        description="Distance metric used for scoring (cosine, l2, ip)",
    )

    def model_post_init(self, __context: Any) -> None:
        """Round score to 4 decimal places for consistent display."""
        object.__setattr__(self, "score", round(self.score, 4))


class SearchResult(BaseModel):
    """
    Aggregated search results for a single query against one collection.

    Groups all matches for a given query, including diagnostics about
    the collection size and requested top-k.
    """

    query: str = Field(
        description="The original search query text"
    )
    collection: str = Field(
        description="Name of the vector collection searched (e.g., 'mondo_diseases')"
    )
    matches: list[OntologyMatch] = Field(
        description="Ranked list of matching ontology terms"
    )
    top_k: int = Field(
        description="Number of results requested"
    )
    total_in_collection: int | None = Field(
        default=None,
        description="Total number of documents in the collection (for diagnostics)",
    )


class LiteratureMatch(BaseModel):
    """
    A single literature match from vector similarity search.

    Represents one publication result from querying a literature collection.
    Defined per SCHM-01 requirement; full implementation deferred to
    literature search phase.
    """

    title: str = Field(
        description="Publication title"
    )
    pmid: str | None = Field(
        default=None,
        description="PubMed identifier (if available)",
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Similarity score between 0 and 1",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata (authors, journal, year, abstract snippet)",
    )
    distance_metric: str = Field(
        default="cosine",
        description="Distance metric used for scoring (cosine, l2, ip)",
    )


class SearchResponse(BaseModel):
    """
    Complete response from a vector search operation.

    Wraps one or more SearchResult objects with metadata about which
    backend, embedding provider, and reranker were used. Enables
    result provenance and reproducibility.
    """

    results: list[SearchResult] = Field(
        description="List of per-collection search results"
    )
    backend: SearchBackend = Field(
        description="Vector database backend used for this search"
    )
    embedding_provider: EmbeddingProvider = Field(
        description="Embedding model provider used to vectorize the query"
    )
    reranker: RerankerType | None = Field(
        default=None,
        description="Reranker applied to results (None if no reranking)",
    )
