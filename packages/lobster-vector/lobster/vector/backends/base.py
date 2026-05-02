"""
Abstract base class for vector database backends.

Defines the contract that all vector storage implementations must follow,
enabling pluggable backends (ChromaDB, FAISS, pgvector) with a consistent API.
Backend implementations are discovered via entry points and loaded lazily
to avoid importing heavy dependencies at startup.

Part of Phase 1 (Foundation) — implementations added in Phase 2+.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseVectorBackend(ABC):
    """
    Abstract interface for vector database backends.

    All vector storage implementations must subclass this and implement
    the four core operations: add, search, delete, count. The interface
    uses simple Python types (lists, dicts) to avoid coupling to any
    specific backend's data model.

    Implementations should handle their own connection management and
    resource cleanup.
    """

    @abstractmethod
    def add_documents(
        self,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Add documents with embeddings to a collection.

        Creates the collection if it does not exist. If a document with
        a given ID already exists, it is overwritten (upsert semantics).

        Args:
            collection_name: Name of the target collection.
            ids: Unique identifiers for each document. Must be same length
                as embeddings.
            embeddings: Pre-computed embedding vectors. Each inner list
                must have the same dimensionality.
            documents: Optional raw text documents corresponding to each
                embedding. Stored alongside vectors for retrieval.
            metadatas: Optional metadata dicts for each document. Used for
                filtering and returned with search results.

        Raises:
            ValueError: If ids, embeddings, documents, or metadatas have
                mismatched lengths.
            ConnectionError: If the backend is unreachable.
        """
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> dict[str, Any]:
        """
        Search a collection by vector similarity.

        Returns raw backend results in a column-oriented format compatible
        with ChromaDB's response structure. Callers should normalize these
        results into SearchResult/OntologyMatch models.

        Args:
            collection_name: Name of the collection to search.
            query_embedding: Query vector. Must match the dimensionality
                of stored embeddings.
            n_results: Maximum number of results to return.

        Returns:
            dict[str, Any]: Raw results with keys:
                - "ids": list[list[str]] — matched document IDs
                - "distances": list[list[float]] — distance scores
                - "documents": list[list[str | None]] — document texts
                - "metadatas": list[list[dict | None]] — metadata dicts

        Raises:
            ValueError: If the collection does not exist.
            ValueError: If query_embedding dimensionality does not match
                the collection's embeddings.
        """
        pass

    @abstractmethod
    def delete(
        self,
        collection_name: str,
        ids: list[str],
    ) -> None:
        """
        Delete documents from a collection by ID.

        Silently ignores IDs that do not exist in the collection.

        Args:
            collection_name: Name of the collection.
            ids: List of document IDs to delete.

        Raises:
            ValueError: If the collection does not exist.
            ConnectionError: If the backend is unreachable.
        """
        pass

    @abstractmethod
    def count(
        self,
        collection_name: str,
    ) -> int:
        """
        Count the number of documents in a collection.

        Args:
            collection_name: Name of the collection.

        Returns:
            int: Number of documents in the collection.

        Raises:
            ValueError: If the collection does not exist.
        """
        pass

    def collection_exists(self, collection_name: str) -> bool:
        """
        Check whether a collection exists in the backend.

        Default implementation attempts count() and catches exceptions.
        Backends may override this with a more efficient native check.

        Args:
            collection_name: Name of the collection to check.

        Returns:
            bool: True if the collection exists, False otherwise.
        """
        try:
            self.count(collection_name)
            return True
        except Exception:
            return False
