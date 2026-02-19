"""
Unit tests for embedding providers.

Tests SapBERT embedder with mocked sentence-transformers to avoid
torch dependency. Validates lazy loading, CLS pooling configuration,
dimensions, batch_size, and import guard behavior.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lobster.core.vector.embeddings.base import BaseEmbedder
from lobster.core.vector.embeddings.sapbert import SapBERTEmbedder


# ---------------------------------------------------------------------------
# TestSapBERTEmbedderLazyLoading
# ---------------------------------------------------------------------------


class TestSapBERTEmbedderLazyLoading:
    """Verify model is loaded lazily (not on __init__)."""

    def test_model_not_loaded_on_init(self):
        """SapBERTEmbedder should NOT load model on construction."""
        embedder = SapBERTEmbedder()
        assert embedder._model is None

    def test_model_loaded_on_first_embed_text(self):
        """First embed_text() call should trigger model loading."""
        embedder = SapBERTEmbedder()

        mock_transformer = MagicMock()
        mock_pooling = MagicMock()
        mock_st = MagicMock()
        mock_st.encode.return_value = np.zeros(768)

        with patch(
            "lobster.core.vector.embeddings.sapbert.SentenceTransformer",
            create=True,
        ) as mock_st_cls, patch(
            "lobster.core.vector.embeddings.sapbert.Transformer",
            create=True,
        ) as mock_t_cls, patch(
            "lobster.core.vector.embeddings.sapbert.Pooling",
            create=True,
        ) as mock_p_cls, patch.dict(
            "sys.modules",
            {
                "sentence_transformers": MagicMock(
                    SentenceTransformer=mock_st_cls,
                    models=MagicMock(
                        Transformer=mock_t_cls, Pooling=mock_p_cls
                    ),
                ),
                "sentence_transformers.models": MagicMock(
                    Transformer=mock_t_cls, Pooling=mock_p_cls
                ),
            },
        ):
            mock_st_cls.return_value = mock_st
            mock_t_cls.return_value = mock_transformer
            mock_p_cls.return_value = mock_pooling

            # Force fresh load
            embedder._model = None
            embedder._load_model()

            mock_st_cls.assert_called_once()

    def test_model_loaded_once_across_calls(self):
        """Multiple embed_text() calls should load model only once."""
        embedder = SapBERTEmbedder()

        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(768)

        # Simulate already-loaded model
        embedder._model = mock_model

        embedder.embed_text("test1")
        embedder.embed_text("test2")

        # encode called twice, but model stays the same object
        assert mock_model.encode.call_count == 2
        assert embedder._model is mock_model


# ---------------------------------------------------------------------------
# TestSapBERTEmbedderConfig
# ---------------------------------------------------------------------------


class TestSapBERTEmbedderConfig:
    """Verify SapBERT configuration constants and CLS pooling."""

    def test_cls_pooling_configured(self):
        """SapBERT must use CLS pooling (not mean pooling)."""
        embedder = SapBERTEmbedder()

        mock_transformer = MagicMock()
        mock_pooling = MagicMock()
        mock_st = MagicMock()
        mock_st.encode.return_value = np.zeros(768)

        with patch.dict(
            "sys.modules",
            {
                "sentence_transformers": MagicMock(
                    SentenceTransformer=MagicMock(return_value=mock_st),
                    models=MagicMock(
                        Transformer=MagicMock(return_value=mock_transformer),
                        Pooling=MagicMock(return_value=mock_pooling),
                    ),
                ),
                "sentence_transformers.models": MagicMock(
                    Transformer=MagicMock(return_value=mock_transformer),
                    Pooling=MagicMock(return_value=mock_pooling),
                ),
            },
        ) as modules:
            embedder._load_model()

            # The Pooling constructor should have been called with pooling_mode="cls"
            pooling_cls = modules["sentence_transformers.models"].Pooling
            pooling_cls.assert_called_once_with(
                word_embedding_dimension=768,
                pooling_mode="cls",
            )

    def test_model_name_is_sapbert(self):
        """Model name should be the full SapBERT HuggingFace path."""
        assert (
            SapBERTEmbedder.MODEL_NAME
            == "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        )

    def test_dimensions_is_768(self):
        """SapBERT produces 768-dimensional embeddings."""
        embedder = SapBERTEmbedder()
        assert embedder.dimensions == 768

    def test_batch_size_128(self):
        """embed_batch should use batch_size=128 per model card."""
        embedder = SapBERTEmbedder()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((2, 768))
        embedder._model = mock_model

        embedder.embed_batch(["a", "b"])
        mock_model.encode.assert_called_once_with(
            ["a", "b"], convert_to_numpy=True, batch_size=128
        )


# ---------------------------------------------------------------------------
# TestSapBERTEmbedderImportGuard
# ---------------------------------------------------------------------------


class TestSapBERTEmbedderImportGuard:
    """Verify helpful ImportError when sentence-transformers is missing."""

    def test_import_error_has_helpful_message(self):
        """ImportError should mention pip install and vector-search."""
        embedder = SapBERTEmbedder()
        embedder._model = None  # Ensure fresh

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def mock_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("No module named 'sentence_transformers'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="pip install"):
                embedder._load_model()


# ---------------------------------------------------------------------------
# TestBaseEmbedderContract
# ---------------------------------------------------------------------------


class TestBaseEmbedderContract:
    """Verify abstract base class contract enforcement."""

    def test_abc_cannot_be_instantiated(self):
        """BaseEmbedder is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEmbedder()

    def test_sapbert_is_valid_subclass(self):
        """SapBERTEmbedder is a valid BaseEmbedder subclass."""
        embedder = SapBERTEmbedder()
        assert isinstance(embedder, BaseEmbedder)

    def test_mock_embedder_satisfies_contract(self):
        """A properly implemented mock embedder should be a valid BaseEmbedder."""

        class _MockEmbedder(BaseEmbedder):
            def embed_text(self, text: str) -> list[float]:
                return [0.0] * 768

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [self.embed_text(t) for t in texts]

            @property
            def dimensions(self) -> int:
                return 768

        mock = _MockEmbedder()
        assert isinstance(mock, BaseEmbedder)
        assert mock.dimensions == 768
        assert len(mock.embed_text("test")) == 768
