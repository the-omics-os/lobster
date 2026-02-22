"""
Vector search service — the main orchestration layer for semantic search.

VectorSearchService coordinates the full query flow: embed query text,
search vector backend, convert distances to similarity scores, and return
flat match dictionaries. Supports both single-query and batch-query modes.

This is the primary API that agents (annotation, metadata, research) will
call to perform semantic search against ontology and literature collections.

Design:
    - Accepts injected backend/embedder for testing (mock objects)
    - Falls back to config-driven factory creation when not injected
    - Lazy initialization: no heavy deps loaded until first query
    - Distance-to-similarity conversion with clamping to [0, 1]
    - match_ontology() provides domain-aware alias resolution and oversampling
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lobster.core.schemas.search import OntologyMatch
    from lobster.services.vector.backends.base import BaseVectorBackend
    from lobster.services.vector.embeddings.base import BaseEmbedder
    from lobster.services.vector.rerankers.base import BaseReranker


# ---------------------------------------------------------------------------
# Ontology collection alias map
# ---------------------------------------------------------------------------

ONTOLOGY_COLLECTIONS: dict[str, str] = {
    # Primary names (canonical)
    "mondo": "mondo_v2024_01",
    "uberon": "uberon_v2024_01",
    "cell_ontology": "cell_ontology_v2024_01",
    # Aliases (domain-friendly shortcuts)
    "disease": "mondo_v2024_01",
    "tissue": "uberon_v2024_01",
    "cell_type": "cell_ontology_v2024_01",
}
"""Maps ontology names and aliases to versioned collection names.

Three primary ontologies (mondo, uberon, cell_ontology) plus three
human-friendly aliases (disease, tissue, cell_type). Agents and
DiseaseOntologyService use aliases; the underlying query() uses
the resolved versioned collection name.
"""


class VectorSearchService:
    """
    Orchestrates vector-based semantic search over biomedical collections.

    The service manages the embed -> search -> format pipeline:
    1. Embed query text using the configured embedding provider
    2. Search the vector backend for nearest neighbors
    3. Convert raw distances to similarity scores (0-1)
    4. Return flat match dictionaries

    Args:
        config: Search configuration. If None, reads from environment.
        backend: Pre-configured vector backend (for testing/DI).
            If provided, skips config factory creation.
        embedder: Pre-configured embedder (for testing/DI).
            If provided, skips config factory creation.

    Example::

        # Production: config-driven (lazy loading)
        service = VectorSearchService()
        matches = service.query("heart attack", "mondo_v2024_01")

        # Testing: inject mocks
        service = VectorSearchService(backend=mock_backend, embedder=mock_embedder)
        matches = service.query("test", "collection")
    """

    def __init__(
        self,
        config: Any | None = None,
        backend: BaseVectorBackend | None = None,
        embedder: BaseEmbedder | None = None,
        reranker: BaseReranker | None = None,
    ) -> None:
        # Lazy import to avoid pulling in pydantic at module level
        if config is None:
            from lobster.services.vector.config import VectorSearchConfig

            config = VectorSearchConfig.from_env()

        self._config = config
        self._backend = backend
        self._embedder = embedder
        self._reranker = reranker
        self._reranker_resolved: bool = reranker is not None

    def _get_backend(self) -> BaseVectorBackend:
        """Get or create the vector backend (lazy initialization)."""
        if self._backend is None:
            self._backend = self._config.create_backend()
        return self._backend

    def _get_embedder(self) -> BaseEmbedder:
        """Get or create the embedder (lazy initialization)."""
        if self._embedder is None:
            self._embedder = self._config.create_embedder()
        return self._embedder

    def _get_reranker(self) -> BaseReranker | None:
        """Get or create the reranker (lazy initialization).

        Returns None if reranking is disabled (RerankerType.none).
        Uses a resolved flag to distinguish 'not checked' from 'checked and is None'.
        """
        if not self._reranker_resolved:
            self._reranker = self._config.create_reranker()
            self._reranker_resolved = True
        return self._reranker

    def query(
        self,
        text: str,
        collection: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search a collection for terms similar to the query text.

        Args:
            text: Query text (e.g., "heart attack", "CD8+ T cell").
            collection: Name of the vector collection to search.
            top_k: Number of results to return. Defaults to config.default_top_k (5).

        Returns:
            list[dict]: Flat match dictionaries, each with keys:
                - term (str): Matched document text
                - ontology_id (str): Ontology identifier from metadata
                - score (float): Cosine similarity (0-1), rounded to 4 decimals
                - metadata (dict): Full metadata dict from the backend
                - distance_metric (str): Always "cosine"
        """
        top_k = top_k or self._config.default_top_k

        # Embed
        query_embedding = self._get_embedder().embed_text(text)

        # Search
        raw_results = self._get_backend().search(
            collection, query_embedding, n_results=top_k
        )

        # Format
        return self._format_results(raw_results, query_text=text)

    def query_batch(
        self,
        texts: list[str],
        collection: str,
        top_k: int | None = None,
    ) -> list[list[dict[str, Any]]]:
        """
        Search a collection for multiple query texts.

        Args:
            texts: List of query texts.
            collection: Name of the vector collection to search.
            top_k: Number of results per query. Defaults to config.default_top_k (5).

        Returns:
            list[list[dict]]: One list of match dicts per query text.
        """
        top_k = top_k or self._config.default_top_k

        # Batch embed
        embeddings = self._get_embedder().embed_batch(texts)

        # Search and format each
        results = []
        for text, embedding in zip(texts, embeddings):
            raw = self._get_backend().search(
                collection, embedding, n_results=top_k
            )
            results.append(self._format_results(raw, query_text=text))

        return results

    def match_ontology(
        self,
        term: str,
        ontology: str,
        k: int = 5,
    ) -> list[OntologyMatch]:
        """
        Match a biomedical term to ontology concepts with alias resolution.

        Domain-aware API that wraps query() with:
        1. Alias resolution (e.g., "disease" -> "mondo_v2024_01")
        2. Oversampling (requests k*4 from backend for future reranking)
        3. Typed OntologyMatch Pydantic objects as return value
        4. Truncation to k results

        This is the primary method agents and DiseaseOntologyService call
        for semantic ontology matching.

        Args:
            term: Biomedical term to match (e.g., "heart attack", "T cell").
            ontology: Ontology name or alias. Supported values:
                - "mondo", "disease" -> MONDO disease ontology
                - "uberon", "tissue" -> UBERON tissue ontology
                - "cell_ontology", "cell_type" -> Cell Ontology
            k: Number of results to return (default 5).

        Returns:
            list[OntologyMatch]: Ranked list of typed ontology matches,
                truncated to k results.

        Raises:
            ValueError: If ontology name is not in ONTOLOGY_COLLECTIONS.

        Example::

            service = VectorSearchService()
            matches = service.match_ontology("heart attack", "disease", k=3)
            for m in matches:
                print(f"{m.term} ({m.ontology_id}): {m.score}")
        """
        from lobster.core.schemas.search import OntologyMatch as _OntologyMatch

        # Resolve ontology alias to versioned collection name
        collection = ONTOLOGY_COLLECTIONS.get(ontology)
        if collection is None:
            available = ", ".join(sorted(ONTOLOGY_COLLECTIONS.keys()))
            raise ValueError(
                f"Unknown ontology '{ontology}'. "
                f"Available options: {available}"
            )

        # Oversample: request k*4 from backend (for reranking headroom)
        oversampled_k = k * 4
        raw_matches = self.query(term, collection, top_k=oversampled_k)

        # Reranking step (between search and truncation)
        reranker = self._get_reranker()
        if reranker is not None and len(raw_matches) > 1:
            from lobster.services.vector.rerankers.base import normalize_scores

            documents = [m["term"] for m in raw_matches]
            reranked = reranker.rerank(term, documents, top_k=k)

            # Normalize scores to [0, 1] for OntologyMatch compatibility
            reranked = normalize_scores(reranked)

            # Rebuild OntologyMatch list in reranked order
            results: list[OntologyMatch] = []
            for entry in reranked:
                original = raw_matches[entry["corpus_id"]]
                results.append(
                    _OntologyMatch(
                        term=original["term"],
                        ontology_id=original["ontology_id"],
                        score=entry["score"],
                        metadata=original["metadata"],
                        distance_metric=original["distance_metric"],
                    )
                )
            return results[:k]

        # No reranker: convert flat dicts to typed OntologyMatch Pydantic objects
        results_list: list[OntologyMatch] = []
        for match_dict in raw_matches:
            results_list.append(
                _OntologyMatch(
                    term=match_dict["term"],
                    ontology_id=match_dict["ontology_id"],
                    score=match_dict["score"],
                    metadata=match_dict["metadata"],
                    distance_metric=match_dict["distance_metric"],
                )
            )

        # Truncate to requested k
        return results_list[:k]

    def _format_results(
        self, raw: dict[str, Any], query_text: str
    ) -> list[dict[str, Any]]:
        """
        Convert raw backend results to flat match dictionaries.

        ChromaDB returns column-oriented results::

            {
                "ids": [["id1", "id2"]],
                "distances": [[0.1, 0.3]],
                "documents": [["term1", "term2"]],
                "metadatas": [[{"ontology_id": "..."}, ...]],
            }

        This method converts distances to similarity scores using:
            score = max(0.0, min(1.0, 1.0 - distance))

        Args:
            raw: Raw backend search results (column-oriented).
            query_text: The original query text (for diagnostics).

        Returns:
            list[dict]: Flat match dicts with term, ontology_id, score,
                metadata, distance_metric.
        """
        # Handle empty results
        ids = raw.get("ids", [[]])
        if not ids or not ids[0]:
            return []

        # Extract the first (and only) query's results
        result_ids = ids[0]
        distances = raw.get("distances", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]

        matches = []
        for i, doc_id in enumerate(result_ids):
            distance = distances[i] if i < len(distances) else 0.0
            document = documents[i] if i < len(documents) else ""
            metadata = metadatas[i] if i < len(metadatas) else {}

            # Convert cosine distance to similarity, clamped to [0, 1]
            score = max(0.0, min(1.0, 1.0 - distance))
            score = round(score, 4)

            # Extract ontology_id from metadata, fall back to document ID.
            # Supports multiple metadata schemas:
            #   - Lobster build script: {"ontology_id": "..."}  or {"term_id": "..."}
            #   - SRAgent tarballs:     {"id": "..."}
            ontology_id = ""
            if metadata and isinstance(metadata, dict):
                ontology_id = (
                    metadata.get("ontology_id")
                    or metadata.get("id")
                    or metadata.get("term_id")
                    or doc_id
                )
            else:
                ontology_id = doc_id
                metadata = {}

            matches.append(
                {
                    "term": document or "",
                    "ontology_id": ontology_id,
                    "score": score,
                    "metadata": metadata,
                    "distance_metric": "cosine",
                }
            )

        return matches
