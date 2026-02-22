"""
OpenAI API-based text embedder with lazy client initialization.

Uses the OpenAI Embeddings API (text-embedding-3-small by default) for
cloud-based embedding generation. The OpenAI client is initialized lazily
on first use to avoid unnecessary imports and API connections.

Requires the ``openai`` package and a valid ``OPENAI_API_KEY`` environment
variable. The OpenAI client reads the API key from the environment
automatically — no explicit key passing is needed.

This provider is suitable for users who prefer cloud-based embeddings,
need high-quality general-purpose vectors, or want to avoid running
local models. The trade-off is API cost and network latency vs. the
free, offline operation of SapBERT or MiniLM.
"""

import logging

from lobster.services.vector.embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)

MODEL_NAME = "text-embedding-3-small"
DIMENSIONS = 1536


class OpenAIEmbedder(BaseEmbedder):
    """
    Cloud-based text embedder using the OpenAI Embeddings API.

    Produces 1536-dimensional embeddings via text-embedding-3-small
    (default) or a user-specified model. The OpenAI client is created
    lazily on first embed call to avoid importing ``openai`` or
    establishing API connections at construction time.

    Requires ``openai`` package and ``OPENAI_API_KEY`` env var::

        pip install openai
        export OPENAI_API_KEY=sk-...

    Example::

        embedder = OpenAIEmbedder()
        # No client created yet
        vec = embedder.embed_text("gene expression analysis")
        # Client created on first call, vec is list[float] of length 1536

        # Custom model
        embedder = OpenAIEmbedder(model="text-embedding-3-large")
    """

    MODEL_NAME = MODEL_NAME
    DIMENSIONS = DIMENSIONS

    def __init__(self, model: str | None = None) -> None:
        self._client = None
        self._model_name = model or self.MODEL_NAME

    def _get_client(self):
        """Get or create the OpenAI client lazily. Thread-safe via GIL."""
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI embeddings require the openai package. "
                "Install with: pip install openai"
            )

        self._client = OpenAI()

        logger.info(
            "Created OpenAI client for model %s. This is a one-time operation.",
            self._model_name,
        )

        return self._client

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text string into a 1536-d vector via OpenAI API.

        Triggers client creation on first call.

        Args:
            text: Input text to embed.

        Returns:
            list[float]: 1536-dimensional embedding vector.

        Raises:
            ImportError: If openai package not installed.
            openai.AuthenticationError: If OPENAI_API_KEY is invalid.
            openai.RateLimitError: If API rate limit exceeded.
        """
        client = self._get_client()
        response = client.embeddings.create(
            model=self._model_name, input=text
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts into 1536-d vectors via OpenAI API.

        Sends all texts in a single API call for efficiency. The OpenAI
        API handles batching internally.

        Args:
            texts: List of input texts to embed.

        Returns:
            list[list[float]]: List of 1536-dimensional embedding vectors.

        Raises:
            ImportError: If openai package not installed.
            openai.AuthenticationError: If OPENAI_API_KEY is invalid.
            openai.RateLimitError: If API rate limit exceeded.
        """
        client = self._get_client()
        response = client.embeddings.create(
            model=self._model_name, input=texts
        )
        return [item.embedding for item in response.data]

    @property
    def dimensions(self) -> int:
        """Return embedding dimensionality (1536 for text-embedding-3-small)."""
        return self.DIMENSIONS
