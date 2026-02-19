"""
Abstract base class for result rerankers.

Defines the contract that all reranker implementations must follow,
enabling pluggable rerankers (CrossEncoder, Cohere) with a consistent API.
Reranker implementations are loaded lazily to avoid importing heavy
dependencies (torch, sentence-transformers, cohere) at startup.

Also provides a ``normalize_scores()`` helper function for converting
unbounded reranker scores (e.g., cross-encoder raw logits) to the
[0.0, 1.0] range required by OntologyMatch.

Part of Phase 4 (Performance) -- implementations in this module.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseReranker(ABC):
    """
    Abstract interface for result rerankers.

    All reranker implementations must subclass this and implement
    the ``rerank()`` method. The interface uses simple Python types
    (list of dicts) to avoid coupling to any specific ML framework.

    Implementations should handle model/client loading, caching,
    and graceful degradation (e.g., missing API key) internally.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank documents by relevance to query.

        Args:
            query: The search query text.
            documents: List of document texts to rerank.
            top_k: Number of top results to return. None = return all.

        Returns:
            list[dict]: Sorted results (most relevant first), each with keys:
                - corpus_id (int): Original index in the documents list
                - score (float): Relevance score (higher = more relevant)
                - text (str): Document text
        """
        pass

    @property
    def name(self) -> str:
        """
        Human-readable name for this reranker.

        Default implementation returns the class name. Subclasses may
        override to provide a more descriptive name.

        Returns:
            str: Reranker name for logging and diagnostics.
        """
        return self.__class__.__name__


def normalize_scores(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalize reranker scores to [0.0, 1.0] using min-max normalization.

    Cross-encoder rerankers produce raw logit scores that are unbounded
    (e.g., 8.6, -4.3). This function maps them to [0, 1] for compatibility
    with OntologyMatch's score field (ge=0.0, le=1.0).

    If all scores are equal, all are set to 1.0 (single-result or uniform).
    Scores are rounded to 4 decimal places for consistent display.

    Args:
        results: List of reranker result dicts, each with a ``score`` key.

    Returns:
        list[dict]: The same list with scores normalized in-place.
    """
    if not results:
        return results

    scores = [r["score"] for r in results]
    min_s = min(scores)
    max_s = max(scores)

    if max_s == min_s:
        # All scores equal -- set all to 1.0
        for r in results:
            r["score"] = 1.0
    else:
        for r in results:
            r["score"] = round((r["score"] - min_s) / (max_s - min_s), 4)

    return results
