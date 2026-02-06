"""
Unified Vector Search Service

Serves both literature search and ontology standardization.
Implements BioAgents two-stage pipeline: vector search -> reranking.

Key Features:
- Pluggable backends (ChromaDB, FAISS)
- Embedding provider abstraction (local or API)
- Optional Cohere reranking
- 5-minute result caching
- Compatible with DiseaseOntologyService API
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from lobster.core.analysis_ir import AnalysisStep
from lobster.core.schemas.search import (
    EmbeddingProvider,
    LiteratureMatch,
    OntologyMatch,
    SearchBackend,
    SearchResponse,
    SearchResult,
)
from lobster.services.search.backends.base import BaseVectorBackend
from lobster.services.search.backends.chroma_backend import ChromaBackend
from lobster.services.search.embeddings.base import BaseEmbeddingProvider
from lobster.services.search.embeddings.sentence_transformers import (
    get_sentence_transformers_provider,
)
from lobster.services.search.reranker import CohereReranker

logger = logging.getLogger(__name__)


class VectorSearchService:
    """
    Unified vector search service.

    Use Cases:
    1. Ontology matching (disease, tissue, cell type, organism)
    2. Literature search (abstracts, methods sections)
    3. Custom document search (uploaded files)

    Features:
    - Pluggable backends (ChromaDB, FAISS, pgvector)
    - Embedding provider abstraction (local or API)
    - Optional Cohere reranking
    - Caching for repeated queries

    Usage:
        # Basic search
        service = VectorSearchService()
        response = service.search("colorectal cancer", backend=my_backend)

        # Ontology matching (compatible with DiseaseOntologyService)
        matches = service.match_ontology("colorectal cancer", ontology="mondo")

        # Literature search
        matches = service.search_literature("single-cell RNA-seq", backend=my_backend)
    """

    DEFAULT_VECTOR_LIMIT = 20
    DEFAULT_FINAL_LIMIT = 5
    CACHE_TTL_SECONDS = 300  # 5 minutes

    # Ontology collection mapping
    ONTOLOGY_COLLECTIONS = {
        "mondo": "mondo",
        "disease": "mondo",
        "uberon": "uberon",
        "tissue": "uberon",
        "cl": "cell_ontology",
        "cell_type": "cell_ontology",
        "cell_ontology": "cell_ontology",
        "ncbi_taxonomy": "ncbi_taxonomy",
        "organism": "ncbi_taxonomy",
        "taxonomy": "ncbi_taxonomy",
    }

    def __init__(
        self,
        backend: Optional[BaseVectorBackend] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        reranker: Optional[CohereReranker] = None,
        enable_reranking: bool = True,
    ):
        """
        Initialize VectorSearchService.

        Args:
            backend: Default vector backend (optional)
            embedding_provider: Embedding provider (defaults to sentence-transformers)
            reranker: Cohere reranker (defaults to new instance)
            enable_reranking: Whether to enable reranking (default True)
        """
        self._backend = backend
        self._embedding_provider = (
            embedding_provider or get_sentence_transformers_provider()
        )
        self._reranker = reranker or CohereReranker()
        self._enable_reranking = enable_reranking and self._reranker.available
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._ontology_backends: Dict[str, ChromaBackend] = {}

        logger.info(
            f"VectorSearchService initialized: "
            f"embedding_provider={self._embedding_provider.model_name}, "
            f"reranking={'enabled' if self._enable_reranking else 'disabled'}"
        )

    @property
    def embedding_provider(self) -> BaseEmbeddingProvider:
        """Return the embedding provider."""
        return self._embedding_provider

    @property
    def reranking_enabled(self) -> bool:
        """Return whether reranking is enabled."""
        return self._enable_reranking

    # =========================================================================
    # GENERIC SEARCH
    # =========================================================================

    def search(
        self,
        query: str,
        backend: Optional[BaseVectorBackend] = None,
        vector_limit: Optional[int] = None,
        final_limit: Optional[int] = None,
        use_reranking: Optional[bool] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> SearchResponse:
        """
        Two-stage vector search with optional reranking.

        Stage 1: Vector similarity search (fast, broad)
        Stage 2: Cohere reranking (precise, slower)

        Args:
            query: Search query text
            backend: Vector backend to use (default: self._backend)
            vector_limit: Initial vector search results (default: 20)
            final_limit: Final results after reranking (default: 5)
            use_reranking: Enable reranking (default: self._enable_reranking)
            filter_metadata: Optional metadata filter for vector search

        Returns:
            SearchResponse with results and metadata

        Raises:
            ValueError: If no backend configured
        """
        vector_limit = vector_limit or self.DEFAULT_VECTOR_LIMIT
        final_limit = final_limit or self.DEFAULT_FINAL_LIMIT
        use_reranking = (
            use_reranking if use_reranking is not None else self._enable_reranking
        )
        backend = backend or self._backend

        if backend is None:
            raise ValueError(
                "No backend configured for search. "
                "Pass a backend parameter or set self._backend"
            )

        # Check cache
        cache_key = self._make_cache_key(
            query, vector_limit, final_limit, use_reranking, backend.name
        )
        cached = self._get_cached(cache_key)
        if cached:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return cached

        start_time = time.time()

        # Stage 1: Vector search
        query_embedding = self._embedding_provider.embed_text(query)
        vector_results = backend.search(
            query_embedding=query_embedding,
            k=vector_limit,
            filter_metadata=filter_metadata,
        )

        search_time_ms = int((time.time() - start_time) * 1000)
        total_candidates = len(vector_results)

        # Stage 2: Reranking (optional)
        rerank_time_ms = None
        reranking_applied = False
        if use_reranking and len(vector_results) > 1 and self._reranker.available:
            rerank_start = time.time()
            reranked = self._reranker.rerank(
                query=query,
                documents=vector_results,
                top_k=final_limit,
            )
            rerank_time_ms = int((time.time() - rerank_start) * 1000)
            final_results = reranked
            reranking_applied = True
        else:
            final_results = vector_results[:final_limit]

        # Build response
        results = [
            SearchResult(
                id=r["id"],
                content=r["content"],
                metadata=r.get("metadata", {}),
                similarity_score=r.get("similarity_score", 0.0),
                relevance_score=r.get("relevance_score"),
            )
            for r in final_results
        ]

        # Detect backend type
        backend_type = self._detect_backend_type(backend)
        embedding_type = self._detect_embedding_type()

        response = SearchResponse(
            query=query,
            results=results,
            backend=backend_type,
            embedding_provider=embedding_type,
            reranking_applied=reranking_applied,
            total_candidates=total_candidates,
            search_time_ms=search_time_ms,
            rerank_time_ms=rerank_time_ms,
        )

        # Cache result
        self._set_cached(cache_key, response)

        logger.debug(
            f"Search completed: query='{query[:30]}...', "
            f"candidates={total_candidates}, "
            f"returned={len(results)}, "
            f"time={search_time_ms}ms"
            + (f"+{rerank_time_ms}ms rerank" if rerank_time_ms else "")
        )

        return response

    # =========================================================================
    # ONTOLOGY MATCHING (Compatible with DiseaseOntologyService API)
    # =========================================================================

    def match_ontology(
        self,
        term: str,
        ontology: str,
        k: int = 3,
        min_confidence: float = 0.5,
    ) -> List[OntologyMatch]:
        """
        Match free-text term to ontology concepts.

        Compatible with existing DiseaseOntologyService.match_disease() API.
        Enables seamless Phase 1 (keyword) to Phase 2 (embedding) migration.

        Args:
            term: Free-text term to match (e.g., "brain cortex", "colorectal cancer")
            ontology: Ontology to search ("mondo", "uberon", "cl", "ncbi_taxonomy")
            k: Number of candidates to consider
            min_confidence: Minimum confidence threshold

        Returns:
            List of OntologyMatch sorted by confidence

        Raises:
            ValueError: If unknown ontology specified
        """
        # Get ontology-specific backend
        backend = self._get_ontology_backend(ontology)

        # Search with reranking for better precision
        response = self.search(
            query=term,
            backend=backend,
            vector_limit=k * 2,  # Get more candidates for filtering
            final_limit=k,
            use_reranking=True,  # Always rerank for ontology matching
        )

        # Convert to OntologyMatch format
        collection_name = self.ONTOLOGY_COLLECTIONS.get(ontology.lower(), ontology)
        matches = []
        for result in response.results:
            confidence = result.relevance_score or result.similarity_score

            if confidence >= min_confidence:
                matches.append(
                    OntologyMatch(
                        input_term=term,
                        matched_term=result.metadata.get(
                            "name", result.content[:50]
                        ),
                        ontology_id=result.id,
                        ontology_source=collection_name.upper(),
                        confidence=confidence,
                        synonyms=result.metadata.get("synonyms", []),
                        definition=result.content,
                    )
                )

        return matches

    def _get_ontology_backend(self, ontology: str) -> ChromaBackend:
        """
        Get or create ChromaDB backend for ontology.

        Caches backends to avoid repeated initialization.

        Args:
            ontology: Ontology name or alias

        Returns:
            ChromaBackend for the ontology

        Raises:
            ValueError: If unknown ontology
        """
        ontology_lower = ontology.lower()
        collection = self.ONTOLOGY_COLLECTIONS.get(ontology_lower)

        if not collection:
            available = ", ".join(sorted(set(self.ONTOLOGY_COLLECTIONS.values())))
            raise ValueError(
                f"Unknown ontology: {ontology}. Available: {available}"
            )

        if collection not in self._ontology_backends:
            self._ontology_backends[collection] = ChromaBackend(
                collection_name=collection
            )

        return self._ontology_backends[collection]

    # =========================================================================
    # LITERATURE SEARCH
    # =========================================================================

    def search_literature(
        self,
        query: str,
        backend: BaseVectorBackend,
        k: int = 5,
        use_reranking: bool = True,
    ) -> List[LiteratureMatch]:
        """
        Search literature by semantic similarity.

        Args:
            query: Research question or topic
            backend: Pre-populated literature backend
            k: Number of results
            use_reranking: Enable Cohere reranking

        Returns:
            List of LiteratureMatch sorted by relevance
        """
        response = self.search(
            query=query,
            backend=backend,
            vector_limit=k * 4,
            final_limit=k,
            use_reranking=use_reranking,
        )

        matches = []
        for result in response.results:
            matches.append(
                LiteratureMatch(
                    query=query,
                    title=result.metadata.get("title", ""),
                    abstract=result.content,
                    doi=result.metadata.get("doi"),
                    pmid=result.metadata.get("pmid"),
                    relevance_score=result.relevance_score or result.similarity_score,
                    matched_terms=result.metadata.get("matched_terms", []),
                )
            )

        return matches

    # =========================================================================
    # CACHING
    # =========================================================================

    def _make_cache_key(
        self,
        query: str,
        vector_limit: int,
        final_limit: int,
        use_reranking: bool,
        backend_name: str,
    ) -> str:
        """Create cache key from search parameters."""
        return f"{query}_{vector_limit}_{final_limit}_{use_reranking}_{backend_name}"

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached result if not expired."""
        if key in self._cache:
            timestamp, value = self._cache[key]
            if time.time() - timestamp < self.CACHE_TTL_SECONDS:
                return value
            del self._cache[key]
        return None

    def _set_cached(self, key: str, value: Any) -> None:
        """Cache result with timestamp."""
        self._cache[key] = (time.time(), value)

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        logger.debug("Search cache cleared")

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _detect_backend_type(self, backend: BaseVectorBackend) -> SearchBackend:
        """Detect backend type from instance."""
        backend_name = backend.name.lower()
        if "chroma" in backend_name:
            return SearchBackend.CHROMA
        elif "faiss" in backend_name:
            return SearchBackend.FAISS
        elif "pgvector" in backend_name:
            return SearchBackend.PGVECTOR
        return SearchBackend.FAISS  # Default

    def _detect_embedding_type(self) -> EmbeddingProvider:
        """Detect embedding provider type."""
        model_name = self._embedding_provider.model_name.lower()
        if "openai" in model_name or "embedding-3" in model_name:
            return EmbeddingProvider.OPENAI
        return EmbeddingProvider.SENTENCE_TRANSFORMERS

    # =========================================================================
    # IR GENERATION (for provenance)
    # =========================================================================

    def create_search_ir(
        self,
        query: str,
        results_count: int,
        search_type: str,
    ) -> AnalysisStep:
        """
        Create IR for search operation (orchestration-only, not executable).

        Args:
            query: Search query
            results_count: Number of results returned
            search_type: Type of search ("ontology", "literature", "generic")

        Returns:
            AnalysisStep for provenance tracking
        """
        return AnalysisStep(
            operation=f"vector_search_{search_type}",
            tool_name=f"VectorSearchService.{search_type}",
            description=f"Semantic search for: {query[:50]}... ({results_count} results)",
            library="lobster.services.search",
            code_template=f"""# Vector search performed
# Query: {{{{ query }}}}
# Results: {{{{ results_count }}}}
# Type: {search_type}
""",
            imports=[],
            parameters={"query": query, "results_count": results_count},
            parameter_schema={},
            input_entities=["query"],
            output_entities=["search_results"],
            exportable=False,  # Orchestration step
        )
