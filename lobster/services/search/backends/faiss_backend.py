"""
FAISS Backend for Literature Search

In-memory ephemeral storage for literature search.
Optimized for fast similarity search without persistence.

Key Features:
- In-memory (fast, no disk I/O)
- Ephemeral (no persistence between sessions)
- Efficient for small to medium datasets
- Supports exact nearest neighbor search
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from lobster.services.search.backends.base import BaseVectorBackend

logger = logging.getLogger(__name__)


class FAISSBackend(BaseVectorBackend):
    """
    FAISS in-memory vector store.

    Used for:
    - Literature search (cached abstracts)
    - Custom document search
    - Temporary collections

    Features:
    - In-memory storage (fast)
    - No persistence (ephemeral)
    - Efficient cosine similarity

    Usage:
        backend = FAISSBackend(dimension=384)
        backend.add_documents(ids, embeddings, texts, metadatas)
        results = backend.search(query_embedding, k=5)
    """

    def __init__(self, dimension: int = 384):
        """
        Initialize FAISS backend.

        Args:
            dimension: Embedding vector dimension
                      Should match the embedding provider dimension
        """
        self.dimension = dimension
        self._index = None
        self._documents: Dict[str, Dict[str, Any]] = {}  # id -> doc data
        self._id_to_idx: Dict[str, int] = {}  # id -> faiss index
        self._idx_to_id: Dict[int, str] = {}  # faiss index -> id

        logger.debug(f"FAISSBackend initialized with dimension={dimension}")

    @property
    def name(self) -> str:
        """Return backend name."""
        return "faiss:memory"

    def _get_index(self):
        """
        Lazy initialize FAISS index.

        Uses IndexFlatIP (inner product) with normalized vectors
        for cosine similarity.
        """
        if self._index is None:
            try:
                import faiss

                # Use L2 index for simplicity (we normalize vectors for cosine)
                self._index = faiss.IndexFlatL2(self.dimension)
                logger.debug(f"Created FAISS index with dimension={self.dimension}")
            except ImportError as e:
                raise ImportError(
                    "faiss-cpu not installed. "
                    "Install with: pip install lobster-ai[search]"
                ) from e
        return self._index

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """
        L2 normalize vectors for cosine similarity.

        Args:
            vectors: Input vectors (N, D) or (D,)

        Returns:
            Normalized vectors
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return vectors / norms

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[np.ndarray],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add documents to FAISS index.

        Args:
            ids: Unique identifiers for each document
            embeddings: Pre-computed embedding vectors
            texts: Document content
            metadatas: Optional metadata for each document

        Raises:
            ValueError: If list lengths don't match or dimension mismatch
        """
        if len(ids) != len(embeddings) or len(ids) != len(texts):
            raise ValueError("ids, embeddings, and texts must have same length")

        if metadatas and len(metadatas) != len(ids):
            raise ValueError("metadatas must have same length as ids")

        index = self._get_index()

        # Convert and validate embeddings
        vectors = np.array(embeddings, dtype=np.float32)
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: got {vectors.shape[1]}, "
                f"expected {self.dimension}"
            )

        # Normalize for cosine similarity
        vectors = self._normalize(vectors)

        # Add to index
        start_idx = index.ntotal
        index.add(vectors)

        # Store document data and ID mappings
        for i, doc_id in enumerate(ids):
            idx = start_idx + i
            self._id_to_idx[doc_id] = idx
            self._idx_to_id[idx] = doc_id
            self._documents[doc_id] = {
                "content": texts[i],
                "metadata": metadatas[i] if metadatas else {},
            }

        logger.debug(f"Added {len(ids)} documents to FAISS index")

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
            filter_metadata: Optional metadata filter (post-search filtering)

        Returns:
            List of dicts with id, content, metadata, similarity_score
        """
        index = self._get_index()

        if index.ntotal == 0:
            return []

        # Normalize query for cosine similarity
        query = self._normalize(query_embedding.reshape(1, -1).astype(np.float32))

        # Search (request more if we need to filter)
        search_k = min(k * 3, index.ntotal) if filter_metadata else min(k, index.ntotal)
        distances, indices = index.search(query, search_k)

        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue

            doc_id = self._idx_to_id.get(int(idx))
            if doc_id is None:
                continue

            doc = self._documents[doc_id]

            # Apply metadata filter if provided
            if filter_metadata:
                match = all(
                    doc["metadata"].get(key) == value
                    for key, value in filter_metadata.items()
                )
                if not match:
                    continue

            # Convert L2 distance to similarity score
            # For normalized vectors: distance = 2 - 2*cos(theta)
            # So similarity = 1 - distance/2
            distance = distances[0][i]
            similarity = max(0.0, min(1.0, 1.0 - distance / 2))

            results.append(
                {
                    "id": doc_id,
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "similarity_score": similarity,
                }
            )

            if len(results) >= k:
                break

        return results

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by ID.

        Note: FAISS IndexFlatL2 doesn't support deletion.
        We mark documents as deleted and filter them out on search.
        For full deletion, rebuild the index.
        """
        for doc_id in ids:
            if doc_id in self._documents:
                del self._documents[doc_id]
                idx = self._id_to_idx.pop(doc_id, None)
                if idx is not None:
                    self._idx_to_id.pop(idx, None)

        logger.debug(f"Marked {len(ids)} documents as deleted")

    def count(self) -> int:
        """Return total document count."""
        return len(self._documents)

    def clear(self) -> None:
        """Clear all documents and reset index."""
        self._index = None
        self._documents.clear()
        self._id_to_idx.clear()
        self._idx_to_id.clear()
        logger.debug("Cleared FAISS index")

    def add_document(
        self,
        doc_id: str,
        text: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Convenience method to add a single document.

        Args:
            doc_id: Unique identifier
            text: Document content
            embedding: Pre-computed embedding
            metadata: Optional metadata
        """
        self.add_documents(
            ids=[doc_id],
            embeddings=[embedding],
            texts=[text],
            metadatas=[metadata] if metadata else None,
        )
