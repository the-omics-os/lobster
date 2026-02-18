"""
SapBERT biomedical entity embedder with lazy model loading.

Uses the cambridgeltl/SapBERT-from-PubMedBERT-fulltext model, which was
trained on 4M+ UMLS synonym pairs for biomedical entity linking. SapBERT
produces 768-dimensional embeddings optimized for matching biomedical terms
to their correct ontology concepts.

IMPORTANT: SapBERT was trained with CLS-token pooling, NOT mean pooling.
Using mean pooling degrades entity-linking performance significantly.
See: https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext

Model loading is fully lazy — no heavy dependencies (torch,
sentence-transformers) are imported until the first embed_text() or
embed_batch() call. This keeps ``import lobster`` fast even when
vector-search extras are installed.
"""

import logging

from lobster.core.vector.embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)

MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
DIMENSIONS = 768


class SapBERTEmbedder(BaseEmbedder):
    """
    Biomedical entity embedder using SapBERT (Self-Alignment Pretraining for BERT).

    Produces 768-dimensional embeddings from biomedical text using CLS-token
    pooling. The model is loaded lazily on first use to avoid slow imports
    and unnecessary GPU/memory allocation.

    Requires ``sentence-transformers`` and ``torch``. Install with::

        pip install 'lobster-ai[vector-search]'

    Example::

        embedder = SapBERTEmbedder()
        # Model NOT loaded yet
        vec = embedder.embed_text("CD8+ T cell")
        # Model loaded on first call, vec is list[float] of length 768
    """

    MODEL_NAME = MODEL_NAME
    DIMENSIONS = DIMENSIONS

    def __init__(self) -> None:
        self._model = None

    def _load_model(self) -> None:
        """Load the SapBERT model on first use. Thread-safe via GIL."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            from sentence_transformers.models import Pooling, Transformer
        except ImportError:
            raise ImportError(
                "SapBERT embeddings require sentence-transformers and PyTorch. "
                "Install with: pip install 'lobster-ai[vector-search]'"
            )

        # CRITICAL: SapBERT was trained with CLS-token pooling.
        # Using mean pooling would degrade entity-linking performance.
        transformer = Transformer(self.MODEL_NAME)
        pooling = Pooling(
            word_embedding_dimension=self.DIMENSIONS,
            pooling_mode="cls",
        )
        self._model = SentenceTransformer(modules=[transformer, pooling])

        logger.info(
            "Loaded SapBERT model (%s). This is a one-time operation.",
            self.MODEL_NAME,
        )

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single biomedical text string into a 768-d vector.

        Triggers model loading on first call.

        Args:
            text: Biomedical text to embed (e.g., "CD8+ T cell",
                "acute myeloid leukemia", "BRCA1").

        Returns:
            list[float]: 768-dimensional embedding vector.

        Raises:
            ImportError: If sentence-transformers/torch not installed.
            RuntimeError: If model inference fails.
        """
        self._load_model()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple biomedical texts into 768-d vectors.

        Uses batch_size=128 per SapBERT model card recommendation to
        balance throughput and memory usage.

        Args:
            texts: List of biomedical texts to embed.

        Returns:
            list[list[float]]: List of 768-dimensional embedding vectors.

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
        """Return embedding dimensionality (768 for SapBERT)."""
        return self.DIMENSIONS
