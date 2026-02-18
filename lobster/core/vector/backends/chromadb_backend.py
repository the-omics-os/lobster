"""
ChromaDB vector database backend with PersistentClient and cosine HNSW.

Uses ChromaDB's PersistentClient for durable local vector storage at
``~/.lobster/vector_store/`` by default. All collections use cosine
distance with HNSW indexing, which returns ``1 - cosine_similarity``
as the distance score. The conversion from distance to similarity
happens in the VectorSearchService layer, not here.

Client initialization is fully lazy — chromadb is not imported until
the first operation that requires a database connection. This keeps
``import lobster`` fast even when vector-search extras are installed.

Batch operations automatically chunk at 5000 documents to stay within
ChromaDB's recommended batch limits.
"""

import logging
from pathlib import Path
from typing import Any

from lobster.core.vector.backends.base import BaseVectorBackend

logger = logging.getLogger(__name__)

# ChromaDB recommended maximum batch size
_BATCH_SIZE = 5000


class ChromaDBBackend(BaseVectorBackend):
    """
    Vector storage backend using ChromaDB with persistent local storage.

    Stores vectors in a local SQLite+HNSW database. Collections use
    cosine distance by default, matching SapBERT's training objective.

    Requires ``chromadb``. Install with::

        pip install 'lobster-ai[vector-search]'

    Example::

        backend = ChromaDBBackend()  # Uses ~/.lobster/vector_store/
        backend.add_documents(
            "ontology_terms",
            ids=["CL:0000084"],
            embeddings=[[0.1] * 768],
            documents=["T cell"],
            metadatas=[{"ontology_id": "CL:0000084"}],
        )
        results = backend.search("ontology_terms", query_embedding=[0.1] * 768)

    Args:
        persist_path: Directory for ChromaDB storage. Defaults to
            ``~/.lobster/vector_store/``. Created with parents if needed.
    """

    def __init__(self, persist_path: str | None = None) -> None:
        if persist_path is None:
            persist_path = str(Path.home() / ".lobster" / "vector_store")

        # Create directory before PersistentClient needs it
        Path(persist_path).mkdir(parents=True, exist_ok=True)

        self._persist_path = persist_path
        self._client = None

    def _get_client(self):
        """
        Get or create the ChromaDB PersistentClient.

        Imports chromadb lazily to avoid import-time overhead.
        Thread-safe via GIL for client creation.

        Returns:
            chromadb.PersistentClient: The database client.

        Raises:
            ImportError: If chromadb is not installed.
        """
        if self._client is not None:
            return self._client

        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "ChromaDB vector store is required. "
                "Install with: pip install 'lobster-ai[vector-search]'"
            )

        self._client = chromadb.PersistentClient(path=self._persist_path)
        logger.info("Initialized ChromaDB at %s", self._persist_path)
        return self._client

    def _get_or_create_collection(self, name: str):
        """
        Get or create a collection with cosine HNSW space.

        All collections use cosine distance (``hnsw:space = cosine``),
        which returns ``1 - cosine_similarity`` as the distance score.

        Args:
            name: Collection name.

        Returns:
            chromadb.Collection: The collection handle.
        """
        client = self._get_client()
        return client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

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

        Automatically chunks large batches into groups of 5000 to stay
        within ChromaDB's recommended limits.

        Args:
            collection_name: Target collection (created if absent).
            ids: Unique document identifiers.
            embeddings: Pre-computed embedding vectors.
            documents: Optional raw text for each document.
            metadatas: Optional metadata dicts for filtering.

        Raises:
            ImportError: If chromadb is not installed.
            ValueError: If input lengths are mismatched.
        """
        collection = self._get_or_create_collection(collection_name)

        if len(ids) <= _BATCH_SIZE:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        else:
            # Process in chunks to avoid ChromaDB batch limits
            for start in range(0, len(ids), _BATCH_SIZE):
                end = start + _BATCH_SIZE
                chunk_docs = documents[start:end] if documents else None
                chunk_meta = metadatas[start:end] if metadatas else None
                collection.add(
                    ids=ids[start:end],
                    embeddings=embeddings[start:end],
                    documents=chunk_docs,
                    metadatas=chunk_meta,
                )
            logger.info(
                "Added %d documents to '%s' in %d chunks",
                len(ids),
                collection_name,
                (len(ids) + _BATCH_SIZE - 1) // _BATCH_SIZE,
            )

    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> dict[str, Any]:
        """
        Search a collection by vector similarity.

        Returns raw ChromaDB column-oriented results. Distance values are
        cosine distances (``1 - similarity``); conversion to similarity
        scores happens in the VectorSearchService layer.

        Args:
            collection_name: Collection to search.
            query_embedding: Query vector (must match collection dimensionality).
            n_results: Maximum results to return.

        Returns:
            dict with keys: ``ids``, ``distances``, ``documents``, ``metadatas``.
            Each value is a list of lists (one inner list per query).

        Raises:
            ImportError: If chromadb is not installed.
        """
        collection = self._get_or_create_collection(collection_name)
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    def delete(
        self,
        collection_name: str,
        ids: list[str],
    ) -> None:
        """
        Delete documents from a collection by ID.

        Silently ignores IDs that do not exist.

        Args:
            collection_name: Target collection.
            ids: Document IDs to delete.

        Raises:
            ImportError: If chromadb is not installed.
        """
        collection = self._get_or_create_collection(collection_name)
        collection.delete(ids=ids)

    def count(
        self,
        collection_name: str,
    ) -> int:
        """
        Count documents in a collection.

        Args:
            collection_name: Target collection.

        Returns:
            int: Number of documents in the collection.

        Raises:
            ImportError: If chromadb is not installed.
        """
        collection = self._get_or_create_collection(collection_name)
        return collection.count()

    def collection_exists(self, collection_name: str) -> bool:
        """
        Check whether a collection exists.

        Uses ChromaDB's native get_collection (more efficient than the
        default count-based check in the base class).

        Args:
            collection_name: Collection name to check.

        Returns:
            bool: True if the collection exists.
        """
        client = self._get_client()
        try:
            client.get_collection(collection_name)
            return True
        except ValueError:
            return False
