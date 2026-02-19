"""
MiniLM general-purpose text embedder with lazy model loading.

Uses the sentence-transformers/all-MiniLM-L6-v2 model, a lightweight
384-dimensional model trained on 1B+ sentence pairs. Unlike SapBERT
(which is biomedical-specific), MiniLM provides strong general-purpose
embeddings suitable for non-biomedical text, mixed-domain queries, and
scenarios where a smaller, faster model is preferred.

IMPORTANT: MiniLM uses MEAN pooling (the SentenceTransformer default).
Do NOT configure CLS pooling like SapBERT — simply load the model with
``SentenceTransformer(MODEL_NAME)`` and mean pooling is applied
automatically by the pre-trained model configuration.

Model loading is fully lazy — no heavy dependencies (torch,
sentence-transformers) are imported until the first embed_text() or
embed_batch() call. This keeps ``import lobster`` fast even when
vector-search extras are installed.
"""

import logging

from lobster.core.vector.embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSIONS = 384


class MiniLMEmbedder(BaseEmbedder):
    """
    General-purpose text embedder using all-MiniLM-L6-v2.

    Produces 384-dimensional embeddings from text using mean pooling.
    The model is loaded lazily on first use to avoid slow imports and
    unnecessary GPU/memory allocation. This is a good fallback for
    non-biomedical text where SapBERT's domain-specific training does
    not apply.

    Requires ``sentence-transformers`` and ``torch``. Install with::

        pip install sentence-transformers

    Example::

        embedder = MiniLMEmbedder()
        # Model NOT loaded yet
        vec = embedder.embed_text("machine learning classifier")
        # Model loaded on first call, vec is list[float] of length 384
    """

    MODEL_NAME = MODEL_NAME
    DIMENSIONS = DIMENSIONS

    def __init__(self) -> None:
        self._model = None

    def _load_model(self) -> None:
        """Load the MiniLM model on first use. Thread-safe via GIL."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "MiniLM embeddings require sentence-transformers. "
                "Install with: pip install sentence-transformers"
            )

        # MiniLM uses mean pooling (the default for SentenceTransformer).
        # Simply load the model directly — no custom Transformer/Pooling modules needed.
        self._model = SentenceTransformer(self.MODEL_NAME)

        logger.info(
            "Loaded MiniLM model (%s). This is a one-time operation.",
            self.MODEL_NAME,
        )

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text string into a 384-d vector.

        Triggers model loading on first call.

        Args:
            text: Input text to embed (e.g., "machine learning classifier",
                "protein folding", "CRISPR gene editing").

        Returns:
            list[float]: 384-dimensional embedding vector.

        Raises:
            ImportError: If sentence-transformers/torch not installed.
            RuntimeError: If model inference fails.
        """
        self._load_model()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts into 384-d vectors.

        Uses batch_size=128 for efficient throughput, matching the
        convention used by other Lobster embedders.

        Args:
            texts: List of input texts to embed.

        Returns:
            list[list[float]]: List of 384-dimensional embedding vectors.

        Raises:
            ImportError: If sentence-transformers/torch not installed.
            RuntimeError: If model inference fails.
        """
        self._load_model()
        embeddings = self._model.encode(
            texts, convert_to_numpy=True, batch_size=128
        )
        return embeddings.tolist()

    @property
    def dimensions(self) -> int:
        """Return embedding dimensionality (384 for MiniLM)."""
        return self.DIMENSIONS
