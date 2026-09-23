"""
Vector database backend implementations.

Provides BaseVectorBackend ABC and backend-specific implementations.
Implementations are loaded lazily — importing this package does NOT
trigger chromadb, faiss, or psycopg2 imports.
"""

# base.py imports only abc + typing, so binding the ABC here costs nothing and
# keeps the heavy implementations lazy. Without this, __all__ names an unbound
# symbol and `from lobster.vector.backends import *` raises AttributeError.
from lobster.vector.backends.base import BaseVectorBackend

__all__ = ["BaseVectorBackend"]
