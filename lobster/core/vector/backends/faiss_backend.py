"""
FAISS vector database backend with IndexIDMap and L2 normalization.

Uses FAISS IndexIDMap wrapping IndexFlatL2 for in-memory vector search.
All vectors are L2-normalized before indexing, so squared L2 distances
can be converted to cosine distances via ``distance / 2.0``. This
conversion happens here (not in the service layer) to match the
ChromaDB backend's contract of returning cosine distances.

FAISS is imported lazily inside ``_ensure_faiss()`` to avoid
import-time overhead when the faiss backend is not selected.

String document IDs are mapped to sequential integer IDs internally,
since FAISS only supports int64 IDs. The mapping is maintained per
collection in companion dicts.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from lobster.core.vector.backends.base import BaseVectorBackend

logger = logging.getLogger(__name__)


class FAISSBackend(BaseVectorBackend):
    """
    Vector storage backend using FAISS with in-memory IndexFlatL2.

    Vectors are L2-normalized before insertion and search so that
    squared L2 distances can be converted to cosine distances.
    Uses IndexIDMap to allow explicit integer ID assignment and
    single-vector deletion without index shifting.

    Requires ``faiss-cpu`` (or ``faiss-gpu``). Install with::

        pip install faiss-cpu

    Example::

        backend = FAISSBackend()
        backend.add_documents(
            "ontology_terms",
            ids=["CL:0000084"],
            embeddings=[[0.1] * 768],
            documents=["T cell"],
            metadatas=[{"ontology_id": "CL:0000084"}],
        )
        results = backend.search("ontology_terms", query_embedding=[0.1] * 768)
    """

    def __init__(self) -> None:
        self._faiss = None
        self._collections: dict[str, dict[str, Any]] = {}

    def _ensure_faiss(self):
        """
        Lazily import faiss, raising a helpful error if not installed.

        Returns:
            The faiss module.

        Raises:
            ImportError: If faiss-cpu (or faiss-gpu) is not installed.
        """
        if self._faiss is not None:
            return self._faiss

        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS is required for the faiss backend. "
                "Install with: pip install faiss-cpu"
            )

        self._faiss = faiss
        return self._faiss

    def _get_collection(self, collection_name: str) -> dict[str, Any]:
        """
        Get an existing collection by name.

        Args:
            collection_name: Name of the collection.

        Returns:
            The collection dict with index and mappings.

        Raises:
            ValueError: If the collection does not exist.
        """
        if collection_name not in self._collections:
            raise ValueError(
                f"Collection '{collection_name}' does not exist. "
                f"Add documents first to create it."
            )
        return self._collections[collection_name]

    def add_documents(
        self,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Add documents with embeddings to a FAISS collection.

        Creates the collection (IndexIDMap wrapping IndexFlatL2) on first
        add. Vectors are L2-normalized before insertion. Existing IDs are
        overwritten (upsert semantics).

        Args:
            collection_name: Target collection (created if absent).
            ids: Unique string identifiers for each document.
            embeddings: Pre-computed embedding vectors.
            documents: Optional raw text for each document.
            metadatas: Optional metadata dicts for filtering.

        Raises:
            ImportError: If faiss is not installed.
        """
        faiss = self._ensure_faiss()

        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)

        # Create collection on first add
        if collection_name not in self._collections:
            dim = vectors.shape[1]
            inner_index = faiss.IndexFlatL2(dim)
            index = faiss.IndexIDMap(inner_index)
            self._collections[collection_name] = {
                "index": index,
                "id_to_int": {},
                "int_to_id": {},
                "documents": {},
                "metadatas": {},
                "next_int_id": 0,
            }

        coll = self._collections[collection_name]

        # Upsert: remove existing vectors for IDs that already exist
        for str_id in ids:
            if str_id in coll["id_to_int"]:
                old_int_id = coll["id_to_int"][str_id]
                coll["index"].remove_ids(np.array([old_int_id], dtype=np.int64))
                del coll["int_to_id"][old_int_id]
                del coll["id_to_int"][str_id]
                coll["documents"].pop(old_int_id, None)
                coll["metadatas"].pop(old_int_id, None)

        # Assign sequential integer IDs
        int_ids = []
        for i, str_id in enumerate(ids):
            int_id = coll["next_int_id"]
            coll["next_int_id"] += 1
            int_ids.append(int_id)

            coll["id_to_int"][str_id] = int_id
            coll["int_to_id"][int_id] = str_id

            if documents is not None:
                coll["documents"][int_id] = documents[i]

            if metadatas is not None:
                coll["metadatas"][int_id] = metadatas[i]

        coll["index"].add_with_ids(
            vectors, np.array(int_ids, dtype=np.int64)
        )

    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> dict[str, Any]:
        """
        Search a FAISS collection by vector similarity.

        Query vector is L2-normalized before search. Squared L2 distances
        are converted to cosine distances via ``distance / 2.0`` (valid
        because all stored vectors are also L2-normalized).

        Returns column-oriented results matching the ChromaDB format.

        Args:
            collection_name: Collection to search.
            query_embedding: Query vector.
            n_results: Maximum results to return.

        Returns:
            dict with keys: ids, distances, documents, metadatas.
            Each value is a list of lists (one inner list per query).

        Raises:
            ValueError: If the collection does not exist.
        """
        faiss = self._ensure_faiss()
        coll = self._get_collection(collection_name)

        # Guard for empty index
        if coll["index"].ntotal == 0:
            return {
                "ids": [[]],
                "distances": [[]],
                "documents": [[]],
                "metadatas": [[]],
            }

        # Clamp n_results to available vectors
        n_results = min(n_results, coll["index"].ntotal)

        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)

        distances, indices = coll["index"].search(query, n_results)

        # Convert squared L2 to cosine distance:
        # For L2-normalized vectors: squared_L2 = 2 * (1 - cos_sim) = 2 * cosine_distance
        # Therefore: cosine_distance = squared_L2 / 2.0
        cosine_distances = (distances[0] / 2.0).tolist()

        # Map integer indices back to string IDs, documents, metadatas
        result_ids = []
        result_docs = []
        result_metas = []
        for int_id in indices[0]:
            int_id = int(int_id)
            str_id = coll["int_to_id"].get(int_id, "")
            result_ids.append(str_id)
            result_docs.append(coll["documents"].get(int_id))
            result_metas.append(coll["metadatas"].get(int_id))

        return {
            "ids": [result_ids],
            "distances": [cosine_distances],
            "documents": [result_docs],
            "metadatas": [result_metas],
        }

    def delete(
        self,
        collection_name: str,
        ids: list[str],
    ) -> None:
        """
        Delete documents from a FAISS collection by ID.

        Silently ignores IDs that do not exist in the collection.

        Args:
            collection_name: Target collection.
            ids: Document IDs to delete.

        Raises:
            ValueError: If the collection does not exist.
        """
        coll = self._get_collection(collection_name)

        for str_id in ids:
            if str_id not in coll["id_to_int"]:
                continue  # Silently skip non-existent IDs

            int_id = coll["id_to_int"][str_id]
            coll["index"].remove_ids(np.array([int_id], dtype=np.int64))
            del coll["int_to_id"][int_id]
            del coll["id_to_int"][str_id]
            coll["documents"].pop(int_id, None)
            coll["metadatas"].pop(int_id, None)

    def count(
        self,
        collection_name: str,
    ) -> int:
        """
        Count documents in a FAISS collection.

        Args:
            collection_name: Target collection.

        Returns:
            int: Number of vectors in the collection's index.

        Raises:
            ValueError: If the collection does not exist.
        """
        coll = self._get_collection(collection_name)
        return coll["index"].ntotal

    def collection_exists(self, collection_name: str) -> bool:
        """
        Check whether a collection exists.

        More efficient than the base class try/except approach.

        Args:
            collection_name: Collection name to check.

        Returns:
            bool: True if the collection exists.
        """
        return collection_name in self._collections
