"""
Text embedding provider implementations.

Provides BaseEmbedder ABC and provider-specific implementations.
Implementations are loaded lazily — importing this package does NOT
trigger torch or sentence-transformers imports.
"""

# base.py imports only abc + typing, so binding the ABC here costs nothing and
# keeps the heavy implementations lazy. Without this, __all__ names an unbound
# symbol and `from lobster.vector.embeddings import *` raises AttributeError.
from lobster.vector.embeddings.base import BaseEmbedder

__all__ = ["BaseEmbedder"]
