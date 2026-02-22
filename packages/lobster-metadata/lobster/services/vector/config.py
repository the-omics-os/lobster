"""
Configuration for vector search infrastructure.

VectorSearchConfig reads environment variables to determine which backend,
embedding provider, and reranker to use, and provides factory methods for
creating configured instances. All imports of heavy dependencies (chromadb,
torch, sentence-transformers, cohere) are lazy -- they only happen when
create_backend(), create_embedder(), or create_reranker() is called.

Environment variables:
    LOBSTER_VECTOR_BACKEND: Backend name (default: "chromadb")
    LOBSTER_EMBEDDING_PROVIDER: Embedder name (default: "openai")
    LOBSTER_VECTOR_STORE_PATH: Persist directory (default: ~/.lobster/vector_store/)
    LOBSTER_RERANKER: Reranker strategy (default: "none")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from lobster.core.schemas.search import (
    EmbeddingProvider,
    RerankerType,
    SearchBackend,
)

if TYPE_CHECKING:
    from lobster.services.vector.backends.base import BaseVectorBackend
    from lobster.services.vector.embeddings.base import BaseEmbedder
    from lobster.services.vector.rerankers.base import BaseReranker


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
        default=EmbeddingProvider.openai,
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
    reranker: RerankerType = Field(
        default=RerankerType.none,
        description="Reranking strategy for search results",
    )

    @classmethod
    def from_env(cls) -> VectorSearchConfig:
        """
        Create a VectorSearchConfig from environment variables.

        Reads:
            LOBSTER_VECTOR_BACKEND -> backend (default "chromadb")
            LOBSTER_EMBEDDING_PROVIDER -> embedding_provider (default "openai")
            LOBSTER_VECTOR_STORE_PATH -> persist_path (default ~/.lobster/vector_store/)
            LOBSTER_RERANKER -> reranker (default "none")

        Returns:
            VectorSearchConfig: Configuration instance.
        """
        backend_str = os.environ.get("LOBSTER_VECTOR_BACKEND", "chromadb")
        embedder_str = os.environ.get("LOBSTER_EMBEDDING_PROVIDER", "openai")
        persist_path = os.environ.get(
            "LOBSTER_VECTOR_STORE_PATH",
            str(Path.home() / ".lobster" / "vector_store"),
        )
        reranker_str = os.environ.get("LOBSTER_RERANKER", "none")

        try:
            reranker = RerankerType(reranker_str)
        except ValueError:
            reranker = RerankerType.none

        return cls(
            backend=SearchBackend(backend_str),
            embedding_provider=EmbeddingProvider(embedder_str),
            persist_path=persist_path,
            reranker=reranker,
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
            from lobster.services.vector.backends.chromadb_backend import (
                ChromaDBBackend,
            )

            return ChromaDBBackend(persist_path=self.persist_path)

        if self.backend == SearchBackend.faiss:
            from lobster.services.vector.backends.faiss_backend import FAISSBackend

            return FAISSBackend()

        if self.backend == SearchBackend.pgvector:
            from lobster.services.vector.backends.pgvector_backend import (
                PgVectorBackend,
            )

            return PgVectorBackend()

        raise ValueError(
            f"Unsupported backend: {self.backend}. "
            f"Available: chromadb, faiss, pgvector"
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
            from lobster.services.vector.embeddings.sapbert import SapBERTEmbedder

            return SapBERTEmbedder()

        if self.embedding_provider == EmbeddingProvider.minilm:
            from lobster.services.vector.embeddings.minilm import MiniLMEmbedder

            return MiniLMEmbedder()

        if self.embedding_provider == EmbeddingProvider.openai:
            from lobster.services.vector.embeddings.openai_embedder import (
                OpenAIEmbedder,
            )

            return OpenAIEmbedder()

        raise ValueError(
            f"Unsupported embedding provider: {self.embedding_provider}. "
            f"Available: sapbert, minilm, openai"
        )

    def create_reranker(self) -> BaseReranker | None:
        """
        Create a configured reranker instance, or None if reranker=none.

        Imports the reranker implementation lazily to avoid loading
        heavy dependencies (torch, cohere) until they are actually needed.

        Returns:
            BaseReranker | None: Configured reranker instance, or None
                if reranking is disabled (RerankerType.none).

        Raises:
            ValueError: If the configured reranker is not supported.
        """
        if self.reranker == RerankerType.none:
            return None

        if self.reranker == RerankerType.cross_encoder:
            from lobster.services.vector.rerankers.cross_encoder_reranker import (
                CrossEncoderReranker,
            )

            return CrossEncoderReranker()

        if self.reranker == RerankerType.cohere:
            from lobster.services.vector.rerankers.cohere_reranker import (
                CohereReranker,
            )

            return CohereReranker()

        raise ValueError(
            f"Unsupported reranker: {self.reranker}. "
            f"Available: none, cross_encoder, cohere"
        )
