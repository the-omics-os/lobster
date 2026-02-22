"""
Vector search commands for CLI and slash-command interfaces.

Provides a shared function that queries ALL ontology collections
(MONDO, UBERON, Cell Ontology) and returns aggregated results as JSON.

No AgentClient or workspace dependency — VectorSearchService is fully standalone.
"""

from __future__ import annotations

from typing import Any

# Canonical collection keys — NOT aliases (disease, tissue, cell_type)
# which would double-query the same data.
_CANONICAL_COLLECTIONS = ("mondo", "uberon", "cell_ontology")


def vector_search_all_collections(
    query_text: str,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Search all ontology collections for a biomedical term.

    Args:
        query_text: Biomedical term to search (e.g., "glioblastoma", "CD8+ T cell").
        top_k: Number of results per collection. Defaults to service default (5).

    Returns:
        dict with keys: query, results (per-collection), top_k, backend, embedding_provider.

    Raises:
        ImportError: If chromadb is not installed (vector-search extra).
    """
    try:
        from lobster.services.vector.service import (
            ONTOLOGY_COLLECTIONS,
            VectorSearchService,
        )
    except ImportError:
        raise ImportError(
            "Vector search requires the [vector-search] extra.\n"
            "Install with: pip install 'lobster-ai[vector-search]'"
        )

    service = VectorSearchService()
    effective_top_k = top_k if top_k is not None else 5

    results: dict[str, list[dict[str, Any]]] = {}
    for key in _CANONICAL_COLLECTIONS:
        versioned_name = ONTOLOGY_COLLECTIONS[key]
        try:
            matches = service.query(query_text, versioned_name, top_k=effective_top_k)
            # Slim each match: keep only term, ontology_id, score
            results[key] = [
                {
                    "term": m["term"],
                    "ontology_id": m["ontology_id"],
                    "score": m["score"],
                }
                for m in matches
            ]
        except Exception:
            # Graceful per-collection failure — don't crash the whole search
            results[key] = []

    return {
        "query": query_text,
        "results": results,
        "top_k": effective_top_k,
        "backend": "chromadb",
        "embedding_provider": service._config.embedding_provider.value,
    }
