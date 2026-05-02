"""
Text embedding provider implementations.

Provides BaseEmbedder ABC and provider-specific implementations.
Implementations are loaded lazily — importing this package does NOT
trigger torch or sentence-transformers imports.
"""

__all__ = ["BaseEmbedder"]
