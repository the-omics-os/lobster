"""Backward-compat shim. Use lobster.core.runtime.data_manager instead."""

import warnings as _w

_w.warn(
    "Import from 'lobster.core.runtime.data_manager' instead of "
    "'lobster.core.data_manager_v2'. Shim will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
from lobster.core.runtime.data_manager import *  # noqa: F401,F403
