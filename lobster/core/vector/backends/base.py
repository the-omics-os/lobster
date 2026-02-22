"""Backward-compat shim. Use lobster.services.vector.backends.base instead."""
import warnings as _w

_w.warn(
    "Import from 'lobster.services.vector.backends.base' instead of "
    "'lobster.core.vector.backends.base'. Shim removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
from lobster.services.vector.backends.base import *  # noqa: F401,F403
from lobster.services.vector.backends.base import BaseVectorBackend
