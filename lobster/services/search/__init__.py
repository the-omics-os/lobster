"""
Vector Search Service Module

Unified vector search infrastructure for Lobster AI.
Supports both literature search and ontology standardization.

Usage:
    from lobster.services.search import VectorSearchService
    from lobster.services.search.backends import ChromaBackend, FAISSBackend
    from lobster.services.search.embeddings import (
        SentenceTransformersProvider,
        OpenAIEmbeddingProvider,
    )

Dependencies (optional):
    pip install lobster-ai[search]
"""

from lobster.services.search.vector_search_service import VectorSearchService

__all__ = [
    "VectorSearchService",
]
