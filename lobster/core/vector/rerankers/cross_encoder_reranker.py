"""
Cross-encoder reranker with lazy model loading.

Uses the ``cross-encoder/ms-marco-MiniLM-L-6-v2`` model (22.7M params, ~91MB)
from the sentence-transformers library. The cross-encoder scores each
(query, document) pair through a full transformer forward pass, providing
significantly better relevance ranking than bi-encoder cosine similarity
alone (typically 10-15% NDCG improvement).

Model loading is fully lazy -- no heavy dependencies (torch,
sentence-transformers) are imported until the first ``rerank()`` call.
This keeps ``import lobster`` fast even when vector-search extras are
installed.

Requires ``sentence-transformers`` and ``torch``. Install with::

    pip install 'lobster-ai[vector-search]'
"""

import logging
from typing import Any

from lobster.core.vector.rerankers.base import BaseReranker

logger = logging.getLogger(__name__)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.

    Scores (query, document) pairs through a full transformer forward pass.
    The model is loaded lazily on first ``rerank()`` call to avoid slow imports
    and unnecessary GPU/memory allocation.

    Edge cases:
        - Empty document list: returns []
        - Single document: returns it with score=1.0 (no model loaded)

    Example::

        reranker = CrossEncoderReranker()
        # Model NOT loaded yet
        results = reranker.rerank("heart attack", ["MI", "diabetes", "ACS"])
        # Model loaded on first call
        # results: [{"corpus_id": 0, "score": 8.6, "text": "MI"}, ...]
    """

    MODEL_NAME = MODEL_NAME

    def __init__(self) -> None:
        self._model = None

    def _load_model(self) -> None:
        """Load the cross-encoder model on first use. Thread-safe via GIL."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "Cross-encoder reranking requires sentence-transformers and PyTorch. "
                "Install with: pip install 'lobster-ai[vector-search]'"
            )

        self._model = CrossEncoder(self.MODEL_NAME)

        logger.info(
            "Loaded cross-encoder model (%s). This is a one-time operation.",
            self.MODEL_NAME,
        )

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank documents by cross-encoder relevance to query.

        Scores each (query, document) pair through a full transformer pass.
        Results are returned sorted by score (highest first).

        Args:
            query: The search query text.
            documents: List of document texts to rerank.
            top_k: Number of top results to return. None = return all.

        Returns:
            list[dict]: Sorted results with corpus_id, score, and text keys.

        Raises:
            ImportError: If sentence-transformers/torch not installed.
        """
        # Edge case: empty list
        if not documents:
            return []

        # Edge case: single document -- skip model loading
        if len(documents) == 1:
            return [{"corpus_id": 0, "score": 1.0, "text": documents[0]}]

        self._load_model()

        results = self._model.rank(
            query, documents, top_k=top_k, return_documents=True
        )

        return [
            {
                "corpus_id": r["corpus_id"],
                "score": float(r["score"]),
                "text": r["text"],
            }
            for r in results
        ]
