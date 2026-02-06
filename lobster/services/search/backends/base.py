"""
Abstract Vector Search Backend

Base class for all vector search backends.
Provides a unified interface for different storage solutions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class BaseVectorBackend(ABC):
    """
    Abstract base class for vector search backends.

    All backends must implement:
    - add_documents: Add documents with embeddings
    - search: Find similar documents
    - delete: Remove documents
    - count: Return document count

    Implementations:
    - ChromaBackend: Persistent storage for ontology (lazy download)
    - FAISSBackend: In-memory ephemeral storage for literature
    - PgVectorBackend: PostgreSQL for cloud deployment (future)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return backend name for logging and debugging."""
        pass

    @abstractmethod
    def add_documents(
        self,
        ids: List[str],
        embeddings: List[np.ndarray],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add documents to the index.

        Args:
            ids: Unique identifiers for each document
            embeddings: Pre-computed embedding vectors
            texts: Document content
            metadatas: Optional metadata for each document

        Raises:
            ValueError: If list lengths don't match
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of dicts with keys:
            - id: str
            - content: str
            - metadata: dict
            - similarity_score: float (0.0-1.0)
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by ID.

        Args:
            ids: List of document IDs to delete
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Return total document count.

        Returns:
            Number of documents in the index
        """
        pass

    def clear(self) -> None:
        """
        Clear all documents from the index.

        Default implementation deletes all documents.
        Subclasses may override for efficiency.
        """
        # Default: no-op (subclasses should override)
        pass

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}(count={self.count()})"
