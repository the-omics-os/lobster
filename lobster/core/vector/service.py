"""Backward-compat shim. Use lobster.services.vector.service instead."""
import warnings as _w

_w.warn(
    "Import from 'lobster.services.vector.service' instead of "
    "'lobster.core.vector.service'. Shim removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
from lobster.services.vector.service import *  # noqa: F401,F403
from lobster.services.vector.service import ONTOLOGY_COLLECTIONS, VectorSearchService
