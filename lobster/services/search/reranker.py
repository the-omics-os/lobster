"""
Cohere Reranker for Two-Stage Search

Improves search precision by reranking initial vector search results.
Follows BioAgents pattern: vector search (20) -> rerank (top 5).

Key Features:
- Graceful degradation (works without COHERE_API_KEY)
- Configurable thresholds
- Supports various document formats
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CohereReranker:
    """
    Cohere reranking for improved search precision.

    Two-stage pipeline:
    1. Vector search returns 20+ candidates
    2. Reranker scores each candidate against query
    3. Return top-k by relevance score

    Model: rerank-english-v3.0

    Usage:
        # With COHERE_API_KEY environment variable
        reranker = CohereReranker()

        if reranker.available:
            reranked = reranker.rerank(
                query="colorectal cancer treatment",
                documents=vector_results,
                top_k=5
            )

        # Graceful degradation without API key
        else:
            print("Reranking unavailable, using vector search order")
    """

    DEFAULT_MODEL = "rerank-english-v3.0"

    # Alternative models
    AVAILABLE_MODELS = {
        "rerank-english-v3.0": "English, highest quality",
        "rerank-multilingual-v3.0": "Multilingual support",
        "rerank-english-v2.0": "English, legacy",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize Cohere reranker.

        Args:
            api_key: Cohere API key. If not provided, uses COHERE_API_KEY env var
            model: Reranking model. Defaults to rerank-english-v3.0
        """
        self._api_key = api_key or os.environ.get("COHERE_API_KEY")
        self._model = model or self.DEFAULT_MODEL
        self._client = None

        if not self._api_key:
            logger.warning(
                "COHERE_API_KEY not set. Reranking will be disabled. "
                "Set the environment variable to enable reranking."
            )
        else:
            logger.debug(f"CohereReranker initialized with model={self._model}")

    @property
    def available(self) -> bool:
        """
        Check if reranking is available.

        Returns:
            True if COHERE_API_KEY is set, False otherwise
        """
        return self._api_key is not None

    @property
    def model(self) -> str:
        """Return the reranking model name."""
        return self._model

    def _get_client(self):
        """
        Lazy load Cohere client.

        Returns:
            cohere.Client instance
        """
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "Cannot create Cohere client without API key. "
                    "Set COHERE_API_KEY environment variable."
                )
            try:
                import cohere

                self._client = cohere.Client(self._api_key)
                logger.debug("Cohere client initialized successfully")
            except ImportError as e:
                raise ImportError(
                    "cohere package not installed. "
                    "Install with: pip install lobster-ai[search]"
                ) from e
        return self._client

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
        min_relevance: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents by relevance to query.

        Args:
            query: Search query
            documents: List of documents (must have 'content' key)
            top_k: Number of results to return
            min_relevance: Minimum relevance score threshold (0.0-1.0)

        Returns:
            Reranked documents with added 'relevance_score' key

        Note:
            Returns original order if reranking is unavailable
        """
        if not self.available:
            logger.warning("Reranking unavailable, returning original order")
            return documents[:top_k]

        if not documents:
            return []

        if len(documents) == 1:
            # No need to rerank single document
            doc = documents[0].copy()
            doc["relevance_score"] = 1.0
            return [doc]

        client = self._get_client()

        # Prepare texts for reranking
        texts = []
        for doc in documents:
            content = doc.get("content", "")
            # Include title if available
            title = doc.get("metadata", {}).get("title", "")
            if title:
                text = f"{title}\n{content}"
            else:
                text = content
            texts.append(text)

        try:
            response = client.rerank(
                model=self._model,
                query=query,
                documents=texts,
                top_n=min(top_k, len(documents)),
                return_documents=False,  # We have original docs
            )

            # Map results back to original documents
            reranked = []
            for result in response.results:
                if result.relevance_score >= min_relevance:
                    doc = documents[result.index].copy()
                    doc["relevance_score"] = result.relevance_score
                    reranked.append(doc)

            logger.debug(
                f"Reranked {len(documents)} documents, "
                f"returned {len(reranked)} above threshold {min_relevance}"
            )

            return reranked

        except Exception as e:
            logger.error(f"Reranking failed: {e}. Returning original order.")
            return documents[:top_k]

    def rerank_texts(
        self,
        query: str,
        texts: List[str],
        top_k: int = 5,
        min_relevance: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Convenience method to rerank plain texts.

        Args:
            query: Search query
            texts: List of text strings
            top_k: Number of results to return
            min_relevance: Minimum relevance score threshold

        Returns:
            List of dicts with 'content' and 'relevance_score' keys
        """
        documents = [{"content": text} for text in texts]
        return self.rerank(query, documents, top_k, min_relevance)
