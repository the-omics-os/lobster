"""
Abstract base class for text embedding providers.

Defines the contract that all embedding implementations must follow,
enabling pluggable embedders (SapBERT, MiniLM, OpenAI) with a consistent API.
Embedder implementations are discovered via entry points and loaded lazily
to avoid importing heavy dependencies (torch, sentence-transformers) at startup.

Part of Phase 1 (Foundation) — implementations added in Phase 3+.
"""

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """
    Abstract interface for text embedding providers.

    All embedding implementations must subclass this and implement
    embed_text, embed_batch, and the dimensions property. The interface
    uses simple Python types (lists of floats) to avoid coupling to
    any specific ML framework's tensor types.

    Implementations should handle model loading, caching, and device
    management (CPU/GPU) internally.
    """

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text string into a vector.

        Args:
            text: Input text to embed. Should be preprocessed
                (lowercased, trimmed) by the caller if needed.

        Returns:
            list[float]: Embedding vector of length self.dimensions.

        Raises:
            ValueError: If text is empty.
            RuntimeError: If the model fails to produce an embedding.
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts into vectors.

        Implementations should batch efficiently (e.g., using the model's
        native batch inference) rather than calling embed_text in a loop.

        Args:
            texts: List of input texts to embed.

        Returns:
            list[list[float]]: List of embedding vectors, one per input text.
                Each vector has length self.dimensions.

        Raises:
            ValueError: If texts list is empty.
            RuntimeError: If the model fails to produce embeddings.
        """
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """
        Return the dimensionality of the embedding vectors.

        This is a fixed property of the model (e.g., 768 for SapBERT,
        384 for MiniLM, 1536 for OpenAI text-embedding-ada-002).

        Returns:
            int: Number of dimensions in the output embedding vectors.
        """
        pass

    @property
    def name(self) -> str:
        """
        Human-readable name for this embedding provider.

        Default implementation returns the class name. Subclasses may
        override to provide a more descriptive name (e.g., "SapBERT-PubMedBERT").

        Returns:
            str: Provider name for logging and diagnostics.
        """
        return self.__class__.__name__
