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

Ontology collections (mondo, uberon, cell_ontology) are auto-downloaded
from S3 as pre-built tarballs on first use, cached at
``~/.lobster/ontology_cache/``. This eliminates the 10-15 minute
cold-start embedding time for fresh installs.
"""

import logging
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from lobster.services.vector.backends.base import BaseVectorBackend

logger = logging.getLogger(__name__)

# ChromaDB recommended maximum batch size
_BATCH_SIZE = 5000

# S3 URLs for pre-built ontology embedding tarballs
ONTOLOGY_TARBALLS: dict[str, str] = {
    "mondo_v2024_01": "https://lobster-ontology-data.s3.amazonaws.com/v2/mondo_openai_1536.tar.gz",
    "uberon_v2024_01": "https://lobster-ontology-data.s3.amazonaws.com/v2/uberon_openai_1536.tar.gz",
    "cell_ontology_v2024_01": "https://lobster-ontology-data.s3.amazonaws.com/v2/cell_ontology_openai_1536.tar.gz",
}

# Local cache directory for downloaded ontology tarballs
ONTOLOGY_CACHE_DIR = Path.home() / ".lobster" / "ontology_cache"


def _download_with_progress(url: str, dest: Path) -> None:
    """
    Download a file from *url* to *dest* with a Rich progress bar.

    Uses a temporary file (``dest.with_suffix('.tmp')``) during download
    and atomically renames to *dest* only on success.  The ``.tmp`` file
    is cleaned up on any error to prevent partial-file corruption.

    Args:
        url: HTTP(S) URL to download.
        dest: Local file path to write.

    Raises:
        requests.HTTPError: If the server returns a non-2xx status.
        requests.ConnectionError: If a network error occurs.
    """
    import requests
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(".tmp")

    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                f"Downloading {dest.name}...",
                total=total or None,
            )
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))

        # Atomic rename on success
        tmp_path.rename(dest)
    except Exception:
        # Clean up partial download
        if tmp_path.exists():
            tmp_path.unlink()
        raise


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

    def _ensure_ontology_data(self, collection_name: str) -> bool:
        """
        Auto-download pre-built ontology data from S3 if needed.

        Checks whether *collection_name* is a known ontology collection
        and, if the local ChromaDB store does not yet contain data for it,
        downloads the corresponding tarball from S3 (with Rich progress),
        extracts it to a temporary directory, opens a **separate**
        PersistentClient on the extracted data, and copies all documents
        into this backend's own persist path.

        The downloaded tarball is cached at ``~/.lobster/ontology_cache/``
        so subsequent calls skip the download.

        Args:
            collection_name: The collection to check.

        Returns:
            True if ontology data is now available (either it was already
            present or was successfully downloaded).  False if this is not
            an ontology collection or if the download/extraction failed
            (in which case a warning is logged and the caller should
            proceed with an empty collection).
        """
        if collection_name not in ONTOLOGY_TARBALLS:
            return False  # Not an ontology collection — nothing to do

        # Already populated?  Check directly via the client to avoid
        # recursion through _get_or_create_collection -> _ensure_ontology_data.
        client = self._get_client()
        try:
            existing_coll = client.get_collection(collection_name)
            if existing_coll.count() > 0:
                return True
        except (ValueError, Exception):
            pass  # Collection doesn't exist yet — proceed to download

        url = ONTOLOGY_TARBALLS[collection_name]
        tarball_filename = url.rsplit("/", 1)[-1]  # e.g. mondo_sapbert_768.tar.gz
        tarball_path = ONTOLOGY_CACHE_DIR / tarball_filename

        # --- Step 1: Download tarball if not cached ---
        if not tarball_path.exists():
            logger.info(
                "Downloading pre-built ontology data for '%s' from S3...",
                collection_name,
            )
            try:
                _download_with_progress(url, tarball_path)
            except Exception as exc:
                logger.warning(
                    "Failed to download ontology data for %s: %s. "
                    "Falling back to empty collection.",
                    collection_name,
                    exc,
                )
                return False

        # --- Step 2: Extract tarball to a temporary directory ---
        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp(prefix="lobster_ontology_")
            try:
                with tarfile.open(tarball_path) as tar:
                    tar.extractall(path=temp_dir, filter="data")
            except (tarfile.ReadError, tarfile.CompressionError) as exc:
                logger.warning(
                    "Corrupt tarball for %s: %s. Removing from cache.",
                    collection_name,
                    exc,
                )
                if tarball_path.exists():
                    tarball_path.unlink()
                return False

            # --- Step 3: Load extracted data into this backend's persist path ---
            # The tarball contains a directory like mondo_sapbert_768/ which
            # is a complete ChromaDB PersistentClient directory.
            extracted_name = tarball_filename.replace(".tar.gz", "")
            extracted_dir = Path(temp_dir) / extracted_name

            if not extracted_dir.exists():
                # Try finding the first directory in temp_dir
                subdirs = [
                    p for p in Path(temp_dir).iterdir() if p.is_dir()
                ]
                if subdirs:
                    extracted_dir = subdirs[0]
                else:
                    logger.warning(
                        "No extracted directory found for %s.", collection_name
                    )
                    return False

            import chromadb

            source_client = chromadb.PersistentClient(path=str(extracted_dir))
            try:
                source_coll = source_client.get_collection(collection_name)
            except (ValueError, Exception):
                # SRAgent tarballs use short names (e.g. "mondo") while
                # we ask for versioned names (e.g. "mondo_v2024_01").
                # Fall back to the first available collection in the tarball.
                available = [c.name for c in source_client.list_collections()]
                if available:
                    logger.info(
                        "Collection '%s' not found in tarball; "
                        "using '%s' (available: %s)",
                        collection_name,
                        available[0],
                        available,
                    )
                    source_coll = source_client.get_collection(available[0])
                else:
                    logger.warning(
                        "No collections in tarball for '%s'.",
                        collection_name,
                    )
                    del source_client
                    return False

            # Paginate extraction from source to avoid SQLite variable
            # limits on large collections (e.g. MONDO has 52K+ terms).
            total_source = source_coll.count()
            page_size = _BATCH_SIZE
            total_loaded = 0

            for offset in range(0, total_source, page_size):
                data = source_coll.get(
                    include=["embeddings", "documents", "metadatas"],
                    limit=page_size,
                    offset=offset,
                )
                if not data["ids"]:
                    break

                self.add_documents(
                    collection_name,
                    ids=data["ids"],
                    embeddings=data["embeddings"],
                    documents=data["documents"],
                    metadatas=data["metadatas"],
                )
                total_loaded += len(data["ids"])

            if total_loaded > 0:
                logger.info(
                    "Loaded %d ontology terms into '%s' from pre-built data.",
                    total_loaded,
                    collection_name,
                )

            del source_client
            return True

        except Exception as exc:
            logger.warning(
                "Failed to load ontology data for %s: %s. "
                "Falling back to empty collection.",
                collection_name,
                exc,
            )
            return False
        finally:
            # Clean up temporary extraction directory
            if temp_dir is not None:
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)

    def _get_or_create_collection(self, name: str):
        """
        Get or create a collection with cosine HNSW space.

        For ontology collections, auto-downloads pre-built data from S3
        on first use so that fresh installs do not require local embedding.

        All collections use cosine distance (``hnsw:space = cosine``),
        which returns ``1 - cosine_similarity`` as the distance score.

        Args:
            name: Collection name.

        Returns:
            chromadb.Collection: The collection handle.
        """
        # Auto-download ontology data if this is a known ontology collection
        self._ensure_ontology_data(name)

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
        except (ValueError, Exception):
            # ChromaDB <1.0 raises ValueError; >=1.5 raises NotFoundError.
            # Catch broadly to support both versions.
            return False
