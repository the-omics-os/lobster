"""DEPRECATED: Vector search has moved to lobster.vector.

This module re-exports from lobster.vector for backward compatibility.
A DeprecationWarning is emitted on first import.

Migration: replace ``from lobster.services.vector import ...``
with ``from lobster.vector import ...``
"""
import warnings as _warnings

_warnings.warn(
    "lobster.services.vector is deprecated. Use 'from lobster.vector import ...' instead. "
    "This compatibility shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from lobster.vector import *  # noqa: F401, F403
from lobster.vector import __all__  # noqa: F401
