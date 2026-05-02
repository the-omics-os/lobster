"""
Vector search infrastructure for Lobster AI.

Provides pluggable vector database backends and embedding providers
for semantic search across biomedical ontologies, literature, and datasets.

Public API is exposed via __all__ but imports are lazy — importing this
module does NOT load chromadb, torch, sentence-transformers, or any other
heavy dependency. Classes are resolved on first access via __getattr__.

Usage::

    from lobster.vector import VectorSearchService, VectorSearchConfig
    from lobster.vector import ONTOLOGY_COLLECTIONS
    from lobster.vector.backends.base import BaseVectorBackend
    from lobster.vector.embeddings.base import BaseEmbedder
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lobster.vector.artifact import ArtifactMetadata, CollectionUnavailable
    from lobster.vector.backends.base import BaseVectorBackend
    from lobster.vector.config import VectorSearchConfig
    from lobster.vector.embeddings.base import BaseEmbedder
    from lobster.vector.rerankers.base import BaseReranker
    from lobster.vector.service import VectorSearchService

__all__ = [
    "ArtifactMetadata",
    "BaseReranker",
    "BaseVectorBackend",
    "BaseEmbedder",
    "CollectionUnavailable",
    "ONTOLOGY_COLLECTIONS",
    "VectorSearchService",
    "VectorSearchConfig",
]


def __getattr__(name: str):
    if name == "VectorSearchService":
        from lobster.vector.service import VectorSearchService

        return VectorSearchService
    if name == "VectorSearchConfig":
        from lobster.vector.config import VectorSearchConfig

        return VectorSearchConfig
    if name == "BaseVectorBackend":
        from lobster.vector.backends.base import BaseVectorBackend

        return BaseVectorBackend
    if name == "BaseEmbedder":
        from lobster.vector.embeddings.base import BaseEmbedder

        return BaseEmbedder
    if name == "BaseReranker":
        from lobster.vector.rerankers.base import BaseReranker

        return BaseReranker
    if name == "ONTOLOGY_COLLECTIONS":
        from lobster.vector.service import ONTOLOGY_COLLECTIONS

        return ONTOLOGY_COLLECTIONS
    if name == "ArtifactMetadata":
        from lobster.vector.artifact import ArtifactMetadata

        return ArtifactMetadata
    if name == "CollectionUnavailable":
        from lobster.vector.artifact import CollectionUnavailable

        return CollectionUnavailable
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
