"""
Unit tests for VectorSearchConfig and ONTOLOGY_COLLECTIONS.

Tests environment variable reading, factory methods, default values,
ontology collection alias resolution, and reranker config/factory.
"""

import os

import pytest

from lobster.core.schemas.search import EmbeddingProvider, RerankerType, SearchBackend
from lobster.core.vector.config import VectorSearchConfig
from lobster.core.vector.service import ONTOLOGY_COLLECTIONS


class TestFromEnv:
    """Tests for VectorSearchConfig.from_env()."""

    def test_from_env_defaults(self, monkeypatch):
        """With no env vars, from_env() returns chromadb + sapbert defaults."""
        monkeypatch.delenv("LOBSTER_VECTOR_BACKEND", raising=False)
        monkeypatch.delenv("LOBSTER_EMBEDDING_PROVIDER", raising=False)
        monkeypatch.delenv("LOBSTER_VECTOR_STORE_PATH", raising=False)

        config = VectorSearchConfig.from_env()
        assert config.backend == SearchBackend.chromadb
        assert config.embedding_provider == EmbeddingProvider.sapbert
        assert config.persist_path.endswith("vector_store")
        assert ".lobster" in config.persist_path

    def test_from_env_custom_backend(self, monkeypatch):
        """LOBSTER_VECTOR_BACKEND=faiss -> SearchBackend.FAISS."""
        monkeypatch.setenv("LOBSTER_VECTOR_BACKEND", "faiss")
        monkeypatch.delenv("LOBSTER_EMBEDDING_PROVIDER", raising=False)
        monkeypatch.delenv("LOBSTER_VECTOR_STORE_PATH", raising=False)

        config = VectorSearchConfig.from_env()
        assert config.backend == SearchBackend.faiss

    def test_from_env_custom_path(self, monkeypatch):
        """LOBSTER_VECTOR_STORE_PATH=/tmp/custom -> config.persist_path."""
        monkeypatch.setenv("LOBSTER_VECTOR_STORE_PATH", "/tmp/custom")
        monkeypatch.delenv("LOBSTER_VECTOR_BACKEND", raising=False)
        monkeypatch.delenv("LOBSTER_EMBEDDING_PROVIDER", raising=False)

        config = VectorSearchConfig.from_env()
        assert config.persist_path == "/tmp/custom"

    def test_from_env_custom_embedder(self, monkeypatch):
        """LOBSTER_EMBEDDING_PROVIDER=minilm -> EmbeddingProvider.MINILM."""
        monkeypatch.setenv("LOBSTER_EMBEDDING_PROVIDER", "minilm")
        monkeypatch.delenv("LOBSTER_VECTOR_BACKEND", raising=False)
        monkeypatch.delenv("LOBSTER_VECTOR_STORE_PATH", raising=False)

        config = VectorSearchConfig.from_env()
        assert config.embedding_provider == EmbeddingProvider.minilm


class TestDefaults:
    """Tests for VectorSearchConfig default values."""

    def test_default_top_k_is_5(self):
        """Default top_k should be 5 per user decision."""
        config = VectorSearchConfig()
        assert config.default_top_k == 5

    def test_default_backend(self):
        """Default backend should be chromadb."""
        config = VectorSearchConfig()
        assert config.backend == SearchBackend.chromadb

    def test_default_embedder(self):
        """Default embedding provider should be sapbert."""
        config = VectorSearchConfig()
        assert config.embedding_provider == EmbeddingProvider.sapbert


class TestFactoryMethods:
    """Tests for create_backend() and create_embedder() factory methods."""

    def test_create_backend_faiss(self):
        """create_backend() with faiss returns FAISSBackend."""
        try:
            import faiss
        except ImportError:
            pytest.skip("faiss-cpu not installed")

        config = VectorSearchConfig(backend=SearchBackend.faiss)
        backend = config.create_backend()

        from lobster.core.vector.backends.faiss_backend import FAISSBackend

        assert isinstance(backend, FAISSBackend)

    def test_create_backend_pgvector(self):
        """create_backend() with pgvector returns PgVectorBackend."""
        config = VectorSearchConfig(backend=SearchBackend.pgvector)
        backend = config.create_backend()

        from lobster.core.vector.backends.pgvector_backend import PgVectorBackend

        assert isinstance(backend, PgVectorBackend)

    def test_create_embedder_minilm(self):
        """create_embedder() with minilm returns MiniLMEmbedder.

        Note: This only checks instantiation, not model loading.
        MiniLMEmbedder uses lazy model loading so no torch is needed here.
        """
        config = VectorSearchConfig(embedding_provider=EmbeddingProvider.minilm)
        embedder = config.create_embedder()

        from lobster.core.vector.embeddings.minilm import MiniLMEmbedder

        assert isinstance(embedder, MiniLMEmbedder)

    def test_create_embedder_openai(self):
        """create_embedder() with openai returns OpenAIEmbedder.

        Note: This only checks instantiation, not client creation.
        OpenAIEmbedder uses lazy client init so no openai package needed here.
        """
        config = VectorSearchConfig(embedding_provider=EmbeddingProvider.openai)
        embedder = config.create_embedder()

        from lobster.core.vector.embeddings.openai_embedder import OpenAIEmbedder

        assert isinstance(embedder, OpenAIEmbedder)

    def test_from_env_openai_embedder(self, monkeypatch):
        """LOBSTER_EMBEDDING_PROVIDER=openai -> EmbeddingProvider.openai."""
        monkeypatch.setenv("LOBSTER_EMBEDDING_PROVIDER", "openai")
        monkeypatch.delenv("LOBSTER_VECTOR_BACKEND", raising=False)
        monkeypatch.delenv("LOBSTER_VECTOR_STORE_PATH", raising=False)

        config = VectorSearchConfig.from_env()
        assert config.embedding_provider == EmbeddingProvider.openai

    def test_create_backend_chromadb(self, tmp_path):
        """create_backend() with chromadb returns ChromaDBBackend."""
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb not installed")

        config = VectorSearchConfig(persist_path=str(tmp_path / "vectors"))
        backend = config.create_backend()

        from lobster.core.vector.backends.chromadb_backend import (
            ChromaDBBackend,
        )

        assert isinstance(backend, ChromaDBBackend)

    def test_create_embedder_sapbert(self):
        """create_embedder() with sapbert returns SapBERTEmbedder.

        Note: This only checks instantiation, not model loading.
        SapBERTEmbedder uses lazy model loading so no torch is needed here.
        """
        config = VectorSearchConfig()
        embedder = config.create_embedder()

        from lobster.core.vector.embeddings.sapbert import SapBERTEmbedder

        assert isinstance(embedder, SapBERTEmbedder)


# ---------------------------------------------------------------------------
# ONTOLOGY_COLLECTIONS tests (Phase 2 Plan 02)
# ---------------------------------------------------------------------------


class TestOntologyCollections:
    """Tests for ONTOLOGY_COLLECTIONS constant (TEST-05)."""

    def test_ontology_collections_has_six_entries(self):
        """ONTOLOGY_COLLECTIONS should have exactly 6 entries (3 primary + 3 aliases)."""
        assert len(ONTOLOGY_COLLECTIONS) == 6

    def test_ontology_collections_aliases_resolve_correctly(self):
        """Aliases should resolve to the same collection as primary names (TEST-05)."""
        # Primary names
        assert "mondo" in ONTOLOGY_COLLECTIONS
        assert "uberon" in ONTOLOGY_COLLECTIONS
        assert "cell_ontology" in ONTOLOGY_COLLECTIONS

        # Aliases
        assert "disease" in ONTOLOGY_COLLECTIONS
        assert "tissue" in ONTOLOGY_COLLECTIONS
        assert "cell_type" in ONTOLOGY_COLLECTIONS

        # Alias resolution matches primary
        assert ONTOLOGY_COLLECTIONS["disease"] == ONTOLOGY_COLLECTIONS["mondo"]
        assert ONTOLOGY_COLLECTIONS["tissue"] == ONTOLOGY_COLLECTIONS["uberon"]
        assert ONTOLOGY_COLLECTIONS["cell_type"] == ONTOLOGY_COLLECTIONS["cell_ontology"]

    def test_ontology_collections_versioned_names(self):
        """Collection names should be versioned (e.g., mondo_v2024_01)."""
        assert ONTOLOGY_COLLECTIONS["mondo"] == "mondo_v2024_01"
        assert ONTOLOGY_COLLECTIONS["uberon"] == "uberon_v2024_01"
        assert ONTOLOGY_COLLECTIONS["cell_ontology"] == "cell_ontology_v2024_01"


# ---------------------------------------------------------------------------
# Reranker config tests (Phase 4 Plan 02)
# ---------------------------------------------------------------------------


class TestRerankerConfig:
    """Tests for reranker configuration, env var reading, and factory."""

    def test_default_reranker_is_none(self):
        """VectorSearchConfig() default reranker should be RerankerType.none."""
        config = VectorSearchConfig()
        assert config.reranker == RerankerType.none

    def test_from_env_reads_lobster_reranker(self, monkeypatch):
        """LOBSTER_RERANKER=cross_encoder -> config.reranker == RerankerType.cross_encoder."""
        monkeypatch.setenv("LOBSTER_RERANKER", "cross_encoder")
        monkeypatch.delenv("LOBSTER_VECTOR_BACKEND", raising=False)
        monkeypatch.delenv("LOBSTER_EMBEDDING_PROVIDER", raising=False)
        monkeypatch.delenv("LOBSTER_VECTOR_STORE_PATH", raising=False)

        config = VectorSearchConfig.from_env()
        assert config.reranker == RerankerType.cross_encoder

    def test_from_env_default_reranker_none(self, monkeypatch):
        """No LOBSTER_RERANKER env var -> RerankerType.none."""
        monkeypatch.delenv("LOBSTER_RERANKER", raising=False)
        monkeypatch.delenv("LOBSTER_VECTOR_BACKEND", raising=False)
        monkeypatch.delenv("LOBSTER_EMBEDDING_PROVIDER", raising=False)
        monkeypatch.delenv("LOBSTER_VECTOR_STORE_PATH", raising=False)

        config = VectorSearchConfig.from_env()
        assert config.reranker == RerankerType.none

    def test_from_env_invalid_reranker_falls_back_to_none(self, monkeypatch):
        """LOBSTER_RERANKER=invalid -> falls back to RerankerType.none (no crash)."""
        monkeypatch.setenv("LOBSTER_RERANKER", "invalid_reranker")
        monkeypatch.delenv("LOBSTER_VECTOR_BACKEND", raising=False)
        monkeypatch.delenv("LOBSTER_EMBEDDING_PROVIDER", raising=False)
        monkeypatch.delenv("LOBSTER_VECTOR_STORE_PATH", raising=False)

        config = VectorSearchConfig.from_env()
        assert config.reranker == RerankerType.none

    def test_create_reranker_none_returns_none(self):
        """config.reranker=none -> create_reranker() returns None."""
        config = VectorSearchConfig(reranker=RerankerType.none)
        assert config.create_reranker() is None

    def test_create_reranker_cross_encoder(self):
        """config.reranker=cross_encoder -> create_reranker() returns CrossEncoderReranker.

        Note: Only checks instantiation -- CrossEncoderReranker uses lazy
        model loading so no torch is needed for this test.
        """
        config = VectorSearchConfig(reranker=RerankerType.cross_encoder)
        reranker = config.create_reranker()

        from lobster.core.vector.rerankers.cross_encoder_reranker import (
            CrossEncoderReranker,
        )

        assert isinstance(reranker, CrossEncoderReranker)

    def test_create_reranker_cohere(self):
        """config.reranker=cohere -> create_reranker() returns CohereReranker.

        CohereReranker only imports cohere lazily on rerank(), so this
        always works regardless of cohere package availability.
        """
        config = VectorSearchConfig(reranker=RerankerType.cohere)
        reranker = config.create_reranker()

        from lobster.core.vector.rerankers.cohere_reranker import CohereReranker

        assert isinstance(reranker, CohereReranker)
