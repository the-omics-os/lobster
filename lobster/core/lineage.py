"""Backward-compat shim. Use lobster.core.provenance.lineage instead."""

import warnings as _w

_w.warn(
    "Import from 'lobster.core.provenance.lineage' instead of "
    "'lobster.core.lineage'. Shim will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
from lobster.core.provenance.lineage import *  # noqa: F401,F403
