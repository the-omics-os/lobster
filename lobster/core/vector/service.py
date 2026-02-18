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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lobster.core.vector.backends.base import BaseVectorBackend
    from lobster.core.vector.embeddings.base import BaseEmbedder


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
    ) -> None:
        # Lazy import to avoid pulling in pydantic at module level
        if config is None:
            from lobster.core.vector.config import VectorSearchConfig

            config = VectorSearchConfig.from_env()

        self._config = config
        self._backend = backend
        self._embedder = embedder

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

            # Extract ontology_id from metadata, fall back to document ID
            ontology_id = ""
            if metadata and isinstance(metadata, dict):
                ontology_id = metadata.get("ontology_id", doc_id)
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
