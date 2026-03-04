"""Backward-compat shim. Use lobster.core.queues.queue_storage instead."""
import warnings as _w

_w.warn(
    "Import from 'lobster.core.queues.queue_storage' instead of "
    "'lobster.core.queue_storage'. Shim will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
from lobster.core.queues.queue_storage import *  # noqa: F401,F403
