"""
Vector Search Backends

Pluggable backend implementations for different storage strategies:
- ChromaBackend: Persistent storage for ontology embeddings (lazy download)
- FAISSBackend: In-memory ephemeral storage for literature search

Usage:
    from lobster.services.search.backends import ChromaBackend, FAISSBackend
"""

from lobster.services.search.backends.base import BaseVectorBackend
from lobster.services.search.backends.chroma_backend import ChromaBackend
from lobster.services.search.backends.faiss_backend import FAISSBackend

__all__ = [
    "BaseVectorBackend",
    "ChromaBackend",
    "FAISSBackend",
]
