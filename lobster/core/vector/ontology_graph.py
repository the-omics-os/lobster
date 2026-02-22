"""Backward-compat shim. Use lobster.services.vector.ontology_graph instead."""
import warnings as _w

_w.warn(
    "Import from 'lobster.services.vector.ontology_graph' instead of "
    "'lobster.core.vector.ontology_graph'. Shim removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
from lobster.services.vector.ontology_graph import *  # noqa: F401,F403
from lobster.services.vector.ontology_graph import OBO_URLS, get_neighbors, load_ontology_graph
