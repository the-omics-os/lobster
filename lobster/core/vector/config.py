"""
Configuration for vector search infrastructure.

VectorSearchConfig reads environment variables to determine which backend
and embedding provider to use, and provides factory methods for creating
configured instances. All imports of heavy dependencies (chromadb, torch,
sentence-transformers) are lazy — they only happen when create_backend()
or create_embedder() is called.

Environment variables:
    LOBSTER_VECTOR_BACKEND: Backend name (default: "chromadb")
    LOBSTER_EMBEDDING_PROVIDER: Embedder name (default: "sapbert")
    LOBSTER_VECTOR_STORE_PATH: Persist directory (default: ~/.lobster/vector_store/)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from lobster.core.schemas.search import EmbeddingProvider, SearchBackend

if TYPE_CHECKING:
    from lobster.core.vector.backends.base import BaseVectorBackend
    from lobster.core.vector.embeddings.base import BaseEmbedder


class VectorSearchConfig(BaseModel):
    """
    Configuration for vector search backend and embedding provider.

    Use ``from_env()`` for environment-variable-driven creation, or
    construct directly for testing / explicit configuration.
    """

    backend: SearchBackend = Field(
        default=SearchBackend.chromadb,
        description="Vector database backend to use",
    )
    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.sapbert,
        description="Embedding model provider for text vectorization",
    )
    persist_path: str = Field(
        default="",
        description="Path for persistent vector storage (resolved in from_env)",
    )
    default_top_k: int = Field(
        default=5,
        description="Default number of results to return per query",
    )

    @classmethod
    def from_env(cls) -> VectorSearchConfig:
        """
        Create a VectorSearchConfig from environment variables.

        Reads:
            LOBSTER_VECTOR_BACKEND -> backend (default "chromadb")
            LOBSTER_EMBEDDING_PROVIDER -> embedding_provider (default "sapbert")
            LOBSTER_VECTOR_STORE_PATH -> persist_path (default ~/.lobster/vector_store/)

        Returns:
            VectorSearchConfig: Configuration instance.
        """
        backend_str = os.environ.get("LOBSTER_VECTOR_BACKEND", "chromadb")
        embedder_str = os.environ.get("LOBSTER_EMBEDDING_PROVIDER", "sapbert")
        persist_path = os.environ.get(
            "LOBSTER_VECTOR_STORE_PATH",
            str(Path.home() / ".lobster" / "vector_store"),
        )

        return cls(
            backend=SearchBackend(backend_str),
            embedding_provider=EmbeddingProvider(embedder_str),
            persist_path=persist_path,
        )

    def create_backend(self) -> BaseVectorBackend:
        """
        Create a configured vector backend instance.

        Imports the backend implementation lazily to avoid loading
        heavy dependencies until they are actually needed.

        Returns:
            BaseVectorBackend: Configured backend instance.

        Raises:
            ValueError: If the configured backend is not yet implemented.
        """
        if self.backend == SearchBackend.chromadb:
            from lobster.core.vector.backends.chromadb_backend import (
                ChromaDBBackend,
            )

            return ChromaDBBackend(persist_path=self.persist_path)

        raise ValueError(
            f"Unsupported backend: {self.backend}. Available: chromadb"
        )

    def create_embedder(self) -> BaseEmbedder:
        """
        Create a configured embedding provider instance.

        Imports the embedder implementation lazily to avoid loading
        heavy dependencies (torch, sentence-transformers) until needed.

        Returns:
            BaseEmbedder: Configured embedder instance.

        Raises:
            ValueError: If the configured provider is not yet implemented.
        """
        if self.embedding_provider == EmbeddingProvider.sapbert:
            from lobster.core.vector.embeddings.sapbert import SapBERTEmbedder

            return SapBERTEmbedder()

        raise ValueError(
            f"Unsupported embedding provider: {self.embedding_provider}. "
            f"Available: sapbert"
        )
