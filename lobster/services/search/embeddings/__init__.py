"""
Embedding Providers

Abstraction over embedding generation with support for:
- SentenceTransformersProvider: Local, free, no API costs (default)
- OpenAIEmbeddingProvider: Cloud API, higher quality, paid

Usage:
    from lobster.services.search.embeddings import (
        get_embedding_provider,
        SentenceTransformersProvider,
        OpenAIEmbeddingProvider,
    )
"""

from lobster.services.search.embeddings.base import BaseEmbeddingProvider
from lobster.services.search.embeddings.sentence_transformers import (
    SentenceTransformersProvider,
    get_sentence_transformers_provider,
)

__all__ = [
    "BaseEmbeddingProvider",
    "SentenceTransformersProvider",
    "get_sentence_transformers_provider",
]

# Conditional imports for optional providers
try:
    from lobster.services.search.embeddings.openai_embeddings import (
        OpenAIEmbeddingProvider,
    )

    __all__.append("OpenAIEmbeddingProvider")
except ImportError:
    pass  # OpenAI not installed
