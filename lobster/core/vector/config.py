"""Backward-compat shim. Use lobster.services.vector.config instead."""

import warnings as _w

_w.warn(
    "Import from 'lobster.services.vector.config' instead of "
    "'lobster.core.vector.config'. Shim removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
from lobster.services.vector.config import *  # noqa: F401,F403
from lobster.services.vector.config import VectorSearchConfig
