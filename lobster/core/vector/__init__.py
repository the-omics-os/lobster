"""Backward-compat shim. Use lobster.services.vector instead."""
import warnings as _w

_w.warn(
    "Import from 'lobster.services.vector' instead of "
    "'lobster.core.vector'. Shim removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
from lobster.services.vector import *  # noqa: F401,F403
from lobster.services.vector import ONTOLOGY_COLLECTIONS, VectorSearchConfig, VectorSearchService
