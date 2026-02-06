"""
OpenAI API Embedding Provider

Higher quality embeddings, requires API key and incurs costs.
Model: text-embedding-3-small (1536 dimensions)

Benefits:
- Higher quality embeddings
- Better semantic understanding
- Larger context window

Costs:
- $0.00002 per 1K tokens
- Rate limits apply
- Requires OPENAI_API_KEY environment variable
"""

import logging
import os
from typing import List, Optional

import numpy as np

from lobster.services.search.embeddings.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI API embedding provider.

    Lazy loads the client on first use. Requires OPENAI_API_KEY
    environment variable or explicit api_key parameter.

    Model Options:
    - text-embedding-3-small: 1536 dims, good balance of quality/cost (default)
    - text-embedding-3-large: 3072 dims, highest quality
    - text-embedding-ada-002: 1536 dims, legacy model

    Usage:
        # With environment variable OPENAI_API_KEY
        provider = OpenAIEmbeddingProvider()
        embedding = provider.embed_text("colorectal cancer")

        # With explicit API key
        provider = OpenAIEmbeddingProvider(api_key="sk-...")
    """

    # Model registry with dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize provider with model name and API key.

        Client is not created until first use (lazy initialization).

        Args:
            model_name: OpenAI embedding model name
                       Defaults to text-embedding-3-small
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var

        Raises:
            ValueError: If no API key provided and OPENAI_API_KEY not set
        """
        self._model_name = model_name or self.DEFAULT_MODEL
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None  # Lazy load
        self._dimension = self.MODEL_DIMENSIONS.get(self._model_name, 1536)

        if not self._api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Either set the environment variable or pass api_key parameter."
            )

        logger.debug(
            f"OpenAIEmbeddingProvider initialized with model={self._model_name} "
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

    def _get_client(self):
        """
        Lazy load OpenAI client on first use.

        This avoids importing openai at module level, which would
        require the package to be installed even when not used.
        """
        if self._client is None:
            logger.info(f"Initializing OpenAI client for model: {self._model_name}")
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self._api_key)
                logger.info("OpenAI client initialized successfully")
            except ImportError as e:
                raise ImportError(
                    "openai package not installed. "
                    "Install with: pip install lobster-ai[search]"
                ) from e
        return self._client

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text.

        Args:
            text: Input text to embed

        Returns:
            numpy array of shape (embedding_dimension,)
        """
        client = self._get_client()
        response = client.embeddings.create(
            model=self._model_name,
            input=text,
            encoding_format="float",
        )
        return np.array(response.data[0].embedding)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.

        OpenAI API supports batch requests natively.

        Args:
            texts: List of texts to embed

        Returns:
            List of numpy arrays, each of shape (embedding_dimension,)
        """
        if not texts:
            return []

        client = self._get_client()
        response = client.embeddings.create(
            model=self._model_name,
            input=texts,
            encoding_format="float",
        )
        return [np.array(item.embedding) for item in response.data]
