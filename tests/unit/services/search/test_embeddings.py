"""
Unit tests for embedding providers.

Tests the embedding provider abstraction layer including
local (sentence-transformers) and API (OpenAI) providers.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# Check for optional dependencies
try:
    import sentence_transformers
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class TestBaseEmbeddingProvider:
    """Tests for BaseEmbeddingProvider interface."""

    def test_interface_contract(self):
        """Verify the abstract interface defines required methods."""
        from lobster.services.search.embeddings.base import BaseEmbeddingProvider

        # Check abstract methods exist
        assert hasattr(BaseEmbeddingProvider, "model_name")
        assert hasattr(BaseEmbeddingProvider, "embedding_dimension")
        assert hasattr(BaseEmbeddingProvider, "embed_text")
        assert hasattr(BaseEmbeddingProvider, "embed_batch")


@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed (pip install lobster-ai[search])")
class TestSentenceTransformersProvider:
    """Tests for SentenceTransformersProvider."""

    def test_initialization(self):
        """Test provider initializes without loading model."""
        from lobster.services.search.embeddings.sentence_transformers import (
            SentenceTransformersProvider,
        )

        provider = SentenceTransformersProvider()

        assert provider.model_name == "all-MiniLM-L6-v2"
        assert provider.embedding_dimension == 384
        # Model should not be loaded yet (lazy initialization)
        assert provider._model is None

    def test_custom_model(self):
        """Test provider accepts custom model name."""
        from lobster.services.search.embeddings.sentence_transformers import (
            SentenceTransformersProvider,
        )

        provider = SentenceTransformersProvider(model_name="all-mpnet-base-v2")

        assert provider.model_name == "all-mpnet-base-v2"
        assert provider.embedding_dimension == 768

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_text(self, mock_st_class):
        """Test single text embedding."""
        from lobster.services.search.embeddings.sentence_transformers import (
            SentenceTransformersProvider,
        )

        # Setup mock
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        provider = SentenceTransformersProvider()
        result = provider.embed_text("test query")

        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)
        mock_model.encode.assert_called_once()

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_batch(self, mock_st_class):
        """Test batch embedding."""
        from lobster.services.search.embeddings.sentence_transformers import (
            SentenceTransformersProvider,
        )

        # Setup mock
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((3, 384))
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        provider = SentenceTransformersProvider()
        result = provider.embed_batch(["text1", "text2", "text3"])

        assert len(result) == 3
        assert all(isinstance(e, np.ndarray) for e in result)

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_batch_empty(self, mock_st_class):
        """Test batch embedding with empty list."""
        from lobster.services.search.embeddings.sentence_transformers import (
            SentenceTransformersProvider,
        )

        provider = SentenceTransformersProvider()
        result = provider.embed_batch([])

        assert result == []

    def test_singleton_caching(self):
        """Test get_sentence_transformers_provider caches instances."""
        from lobster.services.search.embeddings.sentence_transformers import (
            get_sentence_transformers_provider,
        )

        provider1 = get_sentence_transformers_provider("all-MiniLM-L6-v2")
        provider2 = get_sentence_transformers_provider("all-MiniLM-L6-v2")

        assert provider1 is provider2

    def test_repr(self):
        """Test string representation."""
        from lobster.services.search.embeddings.sentence_transformers import (
            SentenceTransformersProvider,
        )

        provider = SentenceTransformersProvider()

        repr_str = repr(provider)
        assert "SentenceTransformersProvider" in repr_str
        assert "all-MiniLM-L6-v2" in repr_str


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed (pip install lobster-ai[search])")
class TestOpenAIEmbeddingProvider:
    """Tests for OpenAIEmbeddingProvider."""

    def test_initialization_no_api_key(self):
        """Test provider raises error without API key."""
        from lobster.services.search.embeddings.openai_embeddings import (
            OpenAIEmbeddingProvider,
        )

        # Remove API key from environment
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                OpenAIEmbeddingProvider()

            assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_initialization_with_api_key(self):
        """Test provider initializes with API key."""
        from lobster.services.search.embeddings.openai_embeddings import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(api_key="test-key")

        assert provider.model_name == "text-embedding-3-small"
        assert provider.embedding_dimension == 1536
        assert provider._client is None  # Lazy initialization

    def test_custom_model(self):
        """Test provider accepts custom model name."""
        from lobster.services.search.embeddings.openai_embeddings import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model_name="text-embedding-3-large",
        )

        assert provider.model_name == "text-embedding-3-large"
        assert provider.embedding_dimension == 3072

    @patch("openai.OpenAI")
    def test_embed_text(self, mock_openai_class):
        """Test single text embedding."""
        from lobster.services.search.embeddings.openai_embeddings import (
            OpenAIEmbeddingProvider,
        )

        # Setup mock
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        provider = OpenAIEmbeddingProvider(api_key="test-key")
        result = provider.embed_text("test query")

        assert isinstance(result, np.ndarray)
        assert result.shape == (1536,)
        mock_client.embeddings.create.assert_called_once()

    @patch("openai.OpenAI")
    def test_embed_batch(self, mock_openai_class):
        """Test batch embedding."""
        from lobster.services.search.embeddings.openai_embeddings import (
            OpenAIEmbeddingProvider,
        )

        # Setup mock
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536),
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        provider = OpenAIEmbeddingProvider(api_key="test-key")
        result = provider.embed_batch(["text1", "text2"])

        assert len(result) == 2
        assert all(isinstance(e, np.ndarray) for e in result)

    @patch("openai.OpenAI")
    def test_embed_batch_empty(self, mock_openai_class):
        """Test batch embedding with empty list."""
        from lobster.services.search.embeddings.openai_embeddings import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(api_key="test-key")
        result = provider.embed_batch([])

        assert result == []
