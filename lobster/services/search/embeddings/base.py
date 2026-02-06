"""
Abstract Embedding Provider

Base class for all embedding providers (local and API-based).
Follows the provider pattern from BioAgents with lazy initialization.
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseEmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    All providers must implement:
    - model_name: Identifier for the embedding model
    - embedding_dimension: Vector dimension output
    - embed_text: Generate embedding for single text
    - embed_batch: Generate embeddings for multiple texts

    Design Principles:
    - Lazy initialization: Models loaded on first use
    - Thread-safe: Can be used across threads
    - Batch support: Efficient batch embedding generation
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Return model identifier.

        Examples:
            - "all-MiniLM-L6-v2" (sentence-transformers)
            - "text-embedding-3-small" (OpenAI)
        """
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """
        Return embedding vector dimension.

        Examples:
            - 384 (all-MiniLM-L6-v2)
            - 1536 (text-embedding-3-small)
        """
        pass

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text.

        Args:
            text: Input text to embed

        Returns:
            numpy array of shape (embedding_dimension,)
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.

        More efficient than calling embed_text in a loop.
        Implementations should handle batching internally.

        Args:
            texts: List of texts to embed

        Returns:
            List of numpy arrays, each of shape (embedding_dimension,)
        """
        pass

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"{self.__class__.__name__}(model={self.model_name}, "
            f"dim={self.embedding_dimension})"
        )
