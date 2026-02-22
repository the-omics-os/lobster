"""
Result reranker implementations.

Provides BaseReranker ABC and reranker-specific implementations.
Implementations are loaded lazily — importing this package does NOT
trigger torch, sentence-transformers, or cohere imports.
"""
