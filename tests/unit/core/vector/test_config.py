"""
Unit tests for VectorSearchConfig and ONTOLOGY_COLLECTIONS.

Tests environment variable reading, factory methods, default values,
and ontology collection alias resolution.
"""

import os

import pytest

from lobster.core.schemas.search import EmbeddingProvider, SearchBackend
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

    def test_create_backend_unsupported(self):
        """Backend=faiss should raise ValueError (not yet implemented)."""
        config = VectorSearchConfig(backend=SearchBackend.faiss)
        with pytest.raises(ValueError, match="Unsupported backend"):
            config.create_backend()

    def test_create_embedder_unsupported(self):
        """Provider=minilm should raise ValueError (not yet implemented)."""
        config = VectorSearchConfig(embedding_provider=EmbeddingProvider.minilm)
        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            config.create_embedder()

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
