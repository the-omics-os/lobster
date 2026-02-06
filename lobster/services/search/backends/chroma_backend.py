"""
ChromaDB Backend for Ontology Search

Persistent storage for pre-built ontology embeddings.
Supports lazy download from GitHub releases following BioAgents pattern.

Key Features:
- Persistent storage (~/.lobster/ontology_cache/)
- Lazy download from GitHub releases
- Pre-built embeddings (no runtime embedding cost)
- Supports multiple ontologies (MONDO, UBERON, CL, NCBI)
"""

import logging
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from lobster.services.search.backends.base import BaseVectorBackend

logger = logging.getLogger(__name__)


class ChromaBackend(BaseVectorBackend):
    """
    ChromaDB persistent vector store.

    Used for:
    - Disease ontology (MONDO)
    - Tissue ontology (UBERON)
    - Cell type ontology (CL)
    - Organism taxonomy (NCBI)

    Features:
    - Persistent storage (~/.lobster/ontology_cache/)
    - Lazy download from GitHub releases
    - Pre-built embeddings (no runtime embedding cost)

    Usage:
        # Load ontology collection (auto-downloads if needed)
        backend = ChromaBackend("mondo")
        results = backend.search(query_embedding, k=5)

        # Custom cache directory
        backend = ChromaBackend("uberon", cache_dir="/custom/path")
    """

    DEFAULT_CACHE_DIR = "~/.lobster/ontology_cache"
    GITHUB_RELEASE_URL = (
        "https://github.com/the-omics-os/lobster-ontologies/releases/download"
    )
    RELEASE_VERSION = "v1.0.0"

    def __init__(
        self,
        collection_name: str,
        cache_dir: Optional[str] = None,
        auto_download: bool = True,
    ):
        """
        Initialize ChromaDB backend.

        Args:
            collection_name: Name of the collection (e.g., "mondo", "uberon")
            cache_dir: Directory for persistent storage
                      Defaults to ~/.lobster/ontology_cache
            auto_download: Whether to auto-download missing collections
                          Set to False for tests or offline mode
        """
        self.collection_name = collection_name
        self.cache_dir = Path(cache_dir or self.DEFAULT_CACHE_DIR).expanduser()
        self.auto_download = auto_download
        self._client = None
        self._collection = None

        logger.debug(
            f"ChromaBackend initialized: collection={collection_name}, "
            f"cache_dir={self.cache_dir}"
        )

    @property
    def name(self) -> str:
        """Return backend name."""
        return f"chroma:{self.collection_name}"

    def _ensure_downloaded(self) -> None:
        """
        Download pre-built ChromaDB if not present.

        Follows BioAgents lazy download pattern.
        """
        collection_path = self.cache_dir / self.collection_name

        if collection_path.exists():
            logger.debug(f"Collection exists at {collection_path}")
            return

        if not self.auto_download:
            raise FileNotFoundError(
                f"Ontology collection '{self.collection_name}' not found at "
                f"{collection_path}. Set auto_download=True or manually download."
            )

        logger.info(f"Downloading ontology collection: {self.collection_name}")
        self._download_collection()

    def _download_collection(self) -> None:
        """
        Download and extract collection from GitHub releases.

        Downloads tarball, extracts to cache directory.
        """
        url = (
            f"{self.GITHUB_RELEASE_URL}/{self.RELEASE_VERSION}/"
            f"{self.collection_name}_chroma.tar.gz"
        )

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            logger.info(f"Downloading from {url}")
            try:
                urllib.request.urlretrieve(url, tmp.name)

                logger.info(f"Extracting to {self.cache_dir}")
                with tarfile.open(tmp.name, "r:gz") as tar:
                    tar.extractall(self.cache_dir)

                logger.info(f"Downloaded {self.collection_name} ontology")
            except Exception as e:
                logger.error(f"Failed to download {self.collection_name}: {e}")
                raise

    def _get_collection(self):
        """
        Lazy load ChromaDB collection.

        Returns:
            chromadb Collection object
        """
        if self._collection is not None:
            return self._collection

        self._ensure_downloaded()

        try:
            import chromadb

            persist_path = str(self.cache_dir / self.collection_name)
            self._client = chromadb.PersistentClient(path=persist_path)

            # Get or create collection
            try:
                self._collection = self._client.get_collection(self.collection_name)
            except Exception:
                # Collection might not exist yet (for tests)
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )

            logger.info(
                f"Loaded ChromaDB collection: {self.collection_name} "
                f"({self._collection.count()} documents)"
            )
            return self._collection

        except ImportError as e:
            raise ImportError(
                "chromadb not installed. Install with: pip install lobster-ai[search]"
            ) from e

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[np.ndarray],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add documents to ChromaDB collection.

        Args:
            ids: Unique identifiers for each document
            embeddings: Pre-computed embedding vectors
            texts: Document content
            metadatas: Optional metadata for each document
        """
        if len(ids) != len(embeddings) or len(ids) != len(texts):
            raise ValueError("ids, embeddings, and texts must have same length")

        collection = self._get_collection()
        collection.add(
            ids=ids,
            embeddings=[e.tolist() for e in embeddings],
            documents=texts,
            metadatas=metadatas,
        )
        logger.debug(f"Added {len(ids)} documents to {self.collection_name}")

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
            filter_metadata: Optional metadata filter (ChromaDB where clause)

        Returns:
            List of dicts with id, content, metadata, similarity_score
        """
        collection = self._get_collection()

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=filter_metadata,
        )

        # Format results
        formatted = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances, convert to similarity
                distance = results["distances"][0][i] if results["distances"] else 0
                # For cosine distance: similarity = 1 - distance
                similarity = max(0.0, min(1.0, 1.0 - distance))

                formatted.append(
                    {
                        "id": doc_id,
                        "content": (
                            results["documents"][0][i]
                            if results["documents"]
                            else ""
                        ),
                        "metadata": (
                            results["metadatas"][0][i]
                            if results["metadatas"]
                            else {}
                        ),
                        "similarity_score": similarity,
                    }
                )

        return formatted

    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        collection = self._get_collection()
        collection.delete(ids=ids)
        logger.debug(f"Deleted {len(ids)} documents from {self.collection_name}")

    def count(self) -> int:
        """Return total document count."""
        collection = self._get_collection()
        return collection.count()

    def clear(self) -> None:
        """Clear all documents from the collection."""
        collection = self._get_collection()
        # Get all IDs and delete them
        all_docs = collection.get()
        if all_docs["ids"]:
            collection.delete(ids=all_docs["ids"])
        logger.debug(f"Cleared collection {self.collection_name}")
