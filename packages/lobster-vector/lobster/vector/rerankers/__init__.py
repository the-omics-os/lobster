"""
Result reranker implementations.

Provides BaseReranker ABC and reranker-specific implementations.
Implementations are loaded lazily — importing this package does NOT
trigger torch, sentence-transformers, or cohere imports.
"""

# base.py imports only abc + typing, so binding the ABC here costs nothing and
# keeps the heavy implementations lazy. Without this there was no __all__ at all,
# so `from lobster.vector.rerankers import *` bound nothing and failed silently.
from lobster.vector.rerankers.base import BaseReranker

__all__ = ["BaseReranker"]
