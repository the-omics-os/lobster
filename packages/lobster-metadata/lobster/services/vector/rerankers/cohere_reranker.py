"""
Cohere API-based reranker with graceful degradation.

Uses the Cohere Rerank API (v2) to rerank search results by relevance.
Requires a ``COHERE_API_KEY`` or ``CO_API_KEY`` environment variable.

**Graceful degradation (RANK-02):** When the API key is missing or the
cohere package is not installed, the reranker logs a warning and returns
documents in their original order with synthetic descending scores.
This ensures the search pipeline never fails due to a missing optional
dependency.

The Cohere client is initialized lazily on first ``rerank()`` call.

Requires ``cohere>=5.0``. Install with::

    pip install cohere
"""

import logging
import os
from typing import Any

from lobster.services.vector.rerankers.base import BaseReranker

logger = logging.getLogger(__name__)


class CohereReranker(BaseReranker):
    """
    Cohere API-based reranker with graceful degradation.

    When the API key is missing or the cohere package is not installed,
    falls back to returning documents in original order with synthetic
    descending scores (1.0, 0.99, 0.98, ...).

    The model name defaults to ``rerank-v4.0-pro`` but can be overridden
    via the ``COHERE_RERANK_MODEL`` environment variable.

    Edge cases:
        - Empty document list: returns []
        - Single document: returns it with score=1.0 (no API call)
        - Missing API key: returns original order with synthetic scores
        - Missing cohere package: returns original order with synthetic scores

    Example::

        reranker = CohereReranker()
        # Client NOT initialized yet
        results = reranker.rerank("heart attack", ["MI", "diabetes", "ACS"])
        # Client initialized on first call (or graceful degradation)
    """

    MODEL = "rerank-v4.0-pro"

    def __init__(self) -> None:
        self._client = None
        self._available: bool | None = None  # None = not checked yet

    def _init_client(self) -> bool:
        """
        Initialize Cohere client. Returns True if available, False if degraded.

        Only checks once -- subsequent calls return the cached result.
        """
        if self._available is not None:
            return self._available

        api_key = os.environ.get("COHERE_API_KEY") or os.environ.get(
            "CO_API_KEY"
        )
        if not api_key:
            logger.warning(
                "Cohere API key not found (COHERE_API_KEY or CO_API_KEY). "
                "Reranking disabled -- returning original result order."
            )
            self._available = False
            return False

        try:
            import cohere

            self._client = cohere.ClientV2(api_key=api_key)
            self._available = True
            return True
        except ImportError:
            logger.warning(
                "cohere package not installed. "
                "Install with: pip install cohere"
            )
            self._available = False
            return False

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank documents by Cohere API relevance to query.

        Falls back to original order with synthetic scores if API
        key is missing or cohere package is not installed.

        Args:
            query: The search query text.
            documents: List of document texts to rerank.
            top_k: Number of top results to return. None = return all.

        Returns:
            list[dict]: Sorted results with corpus_id, score, and text keys.
        """
        # Edge case: empty list
        if not documents:
            return []

        # Edge case: single document -- skip API call
        if len(documents) == 1:
            return [{"corpus_id": 0, "score": 1.0, "text": documents[0]}]

        if not self._init_client():
            # Graceful degradation: return original order with synthetic scores
            effective_docs = documents[:top_k] if top_k else documents
            return [
                {
                    "corpus_id": i,
                    "score": 1.0 - (i * 0.01),
                    "text": doc,
                }
                for i, doc in enumerate(effective_docs)
            ]

        # Read model name (allow env var override)
        model = os.environ.get("COHERE_RERANK_MODEL", self.MODEL)

        response = self._client.rerank(
            model=model,
            query=query,
            documents=documents,
            top_n=top_k,
        )

        return [
            {
                "corpus_id": r.index,
                "score": r.relevance_score,
                "text": documents[r.index],
            }
            for r in response.results
        ]
