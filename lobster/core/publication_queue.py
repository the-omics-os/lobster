"""Backward-compat shim. Use lobster.core.queues.publication_queue instead."""
import warnings as _w

_w.warn(
    "Import from 'lobster.core.queues.publication_queue' instead of "
    "'lobster.core.publication_queue'. Shim will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
from lobster.core.queues.publication_queue import *  # noqa: F401,F403
