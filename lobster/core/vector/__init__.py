"""
Vector search infrastructure for Lobster AI.

Provides pluggable vector database backends and embedding providers
for semantic search across biomedical ontologies, literature, and datasets.

Public API is exposed via __all__ but imports are lazy — importing this
module does NOT load chromadb, torch, sentence-transformers, or any other
heavy dependency. Classes are resolved on first access via __getattr__.

Usage::

    from lobster.core.vector import VectorSearchService, VectorSearchConfig
    from lobster.core.vector import ONTOLOGY_COLLECTIONS
    from lobster.core.vector.backends.base import BaseVectorBackend
    from lobster.core.vector.embeddings.base import BaseEmbedder
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lobster.core.vector.backends.base import BaseVectorBackend
    from lobster.core.vector.config import VectorSearchConfig
    from lobster.core.vector.embeddings.base import BaseEmbedder
    from lobster.core.vector.rerankers.base import BaseReranker
    from lobster.core.vector.service import VectorSearchService

__all__ = [
    "BaseReranker",
    "BaseVectorBackend",
    "BaseEmbedder",
    "ONTOLOGY_COLLECTIONS",
    "VectorSearchService",
    "VectorSearchConfig",
]


def __getattr__(name: str):
    if name == "VectorSearchService":
        from lobster.core.vector.service import VectorSearchService

        return VectorSearchService
    if name == "VectorSearchConfig":
        from lobster.core.vector.config import VectorSearchConfig

        return VectorSearchConfig
    if name == "BaseVectorBackend":
        from lobster.core.vector.backends.base import BaseVectorBackend

        return BaseVectorBackend
    if name == "BaseEmbedder":
        from lobster.core.vector.embeddings.base import BaseEmbedder

        return BaseEmbedder
    if name == "BaseReranker":
        from lobster.core.vector.rerankers.base import BaseReranker

        return BaseReranker
    if name == "ONTOLOGY_COLLECTIONS":
        from lobster.core.vector.service import ONTOLOGY_COLLECTIONS

        return ONTOLOGY_COLLECTIONS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
