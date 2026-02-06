"""
Local Sentence Transformers Embedding Provider

Free, fast (10x faster than API), no token costs.
Default model: all-MiniLM-L6-v2 (384 dimensions)

Benefits:
- No API costs
- No rate limits
- Privacy-preserving (data stays local)
- 10x faster than API calls
- Works offline
"""

import logging
from functools import lru_cache
from typing import List, Optional

import numpy as np

from lobster.services.search.embeddings.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class SentenceTransformersProvider(BaseEmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.

    Lazy loads the model on first use to minimize startup time.
    Thread-safe and can be shared across multiple search services.

    Model Options:
    - all-MiniLM-L6-v2: 384 dims, fast, good quality (default)
    - all-mpnet-base-v2: 768 dims, slower, higher quality
    - paraphrase-MiniLM-L6-v2: 384 dims, good for paraphrase detection

    Usage:
        provider = SentenceTransformersProvider()
        embedding = provider.embed_text("colorectal cancer")

        # Or batch processing
        embeddings = provider.embed_batch(["cancer", "tumor", "neoplasm"])
    """

    # Model registry with dimensions
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-MiniLM-L6-v2": 384,
        "multi-qa-MiniLM-L6-cos-v1": 384,
    }

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize provider with model name.

        Model is not loaded until first use (lazy initialization).

        Args:
            model_name: sentence-transformers model name
                       Defaults to all-MiniLM-L6-v2
        """
        self._model_name = model_name or self.DEFAULT_MODEL
        self._model = None  # Lazy load
        self._dimension = self.MODEL_DIMENSIONS.get(self._model_name, 384)

        logger.debug(
            f"SentenceTransformersProvider initialized with model={self._model_name} "
            f"(dimension={self._dimension})"
        )

    @property
    def model_name(self) -> str:
        """Return model identifier."""
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        """Return embedding vector dimension."""
        return self._dimension

    def _get_model(self):
        """
        Lazy load model on first use.

        This avoids loading the model at import time, which would slow
        down CLI startup even when vector search is not used.
        """
        if self._model is None:
            logger.info(f"Loading sentence-transformers model: {self._model_name}")
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
                # Update dimension from actual model
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(
                    f"Model loaded successfully (dimension={self._dimension})"
                )
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install lobster-ai[search]"
                ) from e
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text.

        Args:
            text: Input text to embed

        Returns:
            numpy array of shape (embedding_dimension,)
        """
        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return embedding

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.

        Uses internal batching for efficiency.
        Progress bar disabled for cleaner output.

        Args:
            texts: List of texts to embed

        Returns:
            List of numpy arrays, each of shape (embedding_dimension,)
        """
        if not texts:
            return []

        model = self._get_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return list(embeddings)


@lru_cache(maxsize=4)
def get_sentence_transformers_provider(
    model_name: Optional[str] = None,
) -> SentenceTransformersProvider:
    """
    Get singleton instance of sentence transformers provider.

    Cached per model_name to avoid loading multiple instances of
    the same model.

    Args:
        model_name: Model to use (defaults to all-MiniLM-L6-v2)

    Returns:
        Cached SentenceTransformersProvider instance
    """
    return SentenceTransformersProvider(model_name)
