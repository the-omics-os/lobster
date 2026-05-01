"""
Vector database backend implementations.

Provides BaseVectorBackend ABC and backend-specific implementations.
Implementations are loaded lazily — importing this package does NOT
trigger chromadb, faiss, or psycopg2 imports.
"""

__all__ = ["BaseVectorBackend"]
