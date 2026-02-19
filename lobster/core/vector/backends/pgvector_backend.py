"""
pgvector backend stub for future v2.0 cloud deployment.

This module provides a placeholder PgVectorBackend that raises
NotImplementedError on all operations. It exists so that the
SearchBackend.pgvector enum value can be instantiated via the
VectorSearchConfig factory without breaking, while clearly
communicating that the implementation is planned for v2.0.

No external dependencies are required for this stub.
"""

from __future__ import annotations

from typing import Any

from lobster.core.vector.backends.base import BaseVectorBackend


class PgVectorBackend(BaseVectorBackend):
    """
    Placeholder vector backend for PostgreSQL pgvector extension.

    All operations raise NotImplementedError with guidance to use
    chromadb or faiss backends in the meantime. Full implementation
    is planned for Omics-OS Cloud v2.0.
    """

    _MSG = (
        "pgvector backend is planned for v2.0. "
        "Use LOBSTER_VECTOR_BACKEND=chromadb (default) "
        "or LOBSTER_VECTOR_BACKEND=faiss for current backends."
    )

    def add_documents(
        self,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Raise NotImplementedError — pgvector is planned for v2.0."""
        raise NotImplementedError(self._MSG)

    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> dict[str, Any]:
        """Raise NotImplementedError — pgvector is planned for v2.0."""
        raise NotImplementedError(self._MSG)

    def delete(
        self,
        collection_name: str,
        ids: list[str],
    ) -> None:
        """Raise NotImplementedError — pgvector is planned for v2.0."""
        raise NotImplementedError(self._MSG)

    def count(
        self,
        collection_name: str,
    ) -> int:
        """Raise NotImplementedError — pgvector is planned for v2.0."""
        raise NotImplementedError(self._MSG)
