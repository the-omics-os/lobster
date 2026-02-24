"""Backward-compat shim. Use lobster.services.vector.backends.chromadb_backend instead."""

import warnings as _w

_w.warn(
    "Import from 'lobster.services.vector.backends.chromadb_backend' instead of "
    "'lobster.core.vector.backends.chromadb_backend'. Shim removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
from lobster.services.vector.backends.chromadb_backend import *  # noqa: F401,F403
from lobster.services.vector.backends.chromadb_backend import (
    ONTOLOGY_CACHE_DIR,
    ONTOLOGY_TARBALLS,
    ChromaDBBackend,
)
