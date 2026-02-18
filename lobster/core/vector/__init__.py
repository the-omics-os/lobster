"""
Vector search infrastructure for Lobster AI.

Provides pluggable vector database backends and embedding providers
for semantic search across biomedical ontologies, literature, and datasets.

Public API is exposed via __all__ but imports are lazy — importing this
module does NOT load chromadb, torch, sentence-transformers, or any other
heavy dependency. Callers import specific classes directly:

    from lobster.core.vector.backends.base import BaseVectorBackend
    from lobster.core.vector.embeddings.base import BaseEmbedder

The VectorSearchService and VectorSearchConfig names are reserved for
Phase 2 implementation and listed in __all__ for forward compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lobster.core.vector.backends.base import BaseVectorBackend
    from lobster.core.vector.embeddings.base import BaseEmbedder

__all__ = [
    "BaseVectorBackend",
    "BaseEmbedder",
    "VectorSearchService",
    "VectorSearchConfig",
]
