"""
Ontology graph loading and traversal for biomedical ontologies.

Provides OBO file parsing via obonet and graph traversal using networkx
for MONDO, Uberon, and Cell Ontology. Agents use this to retrieve
parent/child/sibling context for ontology terms (e.g., "colorectal cancer
IS_A intestinal cancer").

**Edge direction convention (OBO format):**
    In OBO ontologies, ``is_a`` edges point FROM child TO parent.
    So ``graph.successors(term)`` yields **parents** and
    ``graph.predecessors(term)`` yields **children**.

Usage::

    from lobster.services.vector.ontology_graph import (
        load_ontology_graph, get_neighbors, OBO_URLS,
    )

    graph = load_ontology_graph("mondo")
    neighbors = get_neighbors(graph, "MONDO:0005575")
    # {"parents": [...], "children": [...], "siblings": [...]}

obonet is import-guarded: importing this module does NOT require obonet.
The dependency is only needed when ``load_ontology_graph`` is called.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import networkx as nx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OBO Foundry download URLs for supported ontologies (GRPH-03)
# ---------------------------------------------------------------------------

OBO_URLS: dict[str, str] = {
    "mondo": "https://purl.obolibrary.org/obo/mondo.obo",
    "uberon": "https://purl.obolibrary.org/obo/uberon/uberon-basic.obo",
    "cell_ontology": "https://purl.obolibrary.org/obo/cl.obo",
}

# ---------------------------------------------------------------------------
# Graph loading with process-lifetime caching (GRPH-01)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=3)
def load_ontology_graph(ontology: str) -> nx.MultiDiGraph:
    """Load and cache an OBO ontology as a networkx MultiDiGraph.

    Parameters
    ----------
    ontology : str
        One of the keys in :data:`OBO_URLS` (``"mondo"``, ``"uberon"``,
        ``"cell_ontology"``).

    Returns
    -------
    networkx.MultiDiGraph
        The parsed ontology graph where edges point from child to parent
        (``is_a`` relationship).

    Raises
    ------
    ValueError
        If *ontology* is not a recognised key in :data:`OBO_URLS`.
    ImportError
        If ``obonet`` is not installed.
    """
    if ontology not in OBO_URLS:
        available = ", ".join(sorted(OBO_URLS.keys()))
        raise ValueError(
            f"Unknown ontology {ontology!r}. Available: {available}"
        )

    try:
        import obonet  # noqa: F811 — import-guarded
    except ImportError:
        raise ImportError(
            "Ontology graph requires obonet. "
            "Install with: pip install 'lobster-ai[vector-search]'"
        )

    url = OBO_URLS[ontology]
    logger.info("Loading ontology graph %r from %s ...", ontology, url)

    graph = obonet.read_obo(url)

    logger.info(
        "Loaded %s: %d terms, %d edges",
        ontology,
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph


# ---------------------------------------------------------------------------
# Graph traversal helpers (GRPH-02)
# ---------------------------------------------------------------------------


def _format_term(graph: nx.MultiDiGraph, tid: str) -> dict[str, str]:
    """Return ``{"term_id": tid, "name": <name>}`` for a graph node."""
    name = graph.nodes[tid].get("name", "") if tid in graph.nodes else ""
    return {"term_id": tid, "name": name}


def get_neighbors(
    graph: nx.MultiDiGraph,
    term_id: str,
    depth: int = 1,
    relation: str = "all",
) -> dict[str, list[dict[str, str]]]:
    """Return parent, child, and sibling terms for *term_id*.

    **OBO edge direction:** edges go FROM child TO parent (``is_a``).

    * **Parents** (``graph.successors``): follow edges forward.
    * **Children** (``graph.predecessors``): follow edges backward.
    * **Siblings**: other children of the immediate parents (always depth=1).

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        An ontology graph as returned by :func:`load_ontology_graph`.
    term_id : str
        The ontology term identifier (e.g. ``"MONDO:0005575"``).
    depth : int, optional
        Traversal depth for parents/children. ``1`` = immediate only,
        ``>1`` = transitive up to *depth* hops.  Default ``1``.
    relation : str, optional
        Currently unused; reserved for future relation filtering.
        Default ``"all"``.

    Returns
    -------
    dict
        ``{"parents": [...], "children": [...], "siblings": [...]}``
        where each entry is a list of ``{"term_id": str, "name": str}``
        dicts. Returns empty lists if *term_id* is not in the graph.
    """
    import networkx as nx  # noqa: F811 — import inside function body

    empty: dict[str, list[dict[str, str]]] = {
        "parents": [],
        "children": [],
        "siblings": [],
    }

    if term_id not in graph.nodes:
        return empty

    # --- Parents (successors in OBO) ---
    if depth <= 1:
        parent_ids = list(graph.successors(term_id))
    else:
        # nx.descendants follows edges forward (successors) transitively
        parent_ids = list(nx.descendants(graph, term_id))

    # --- Children (predecessors in OBO) ---
    if depth <= 1:
        child_ids = list(graph.predecessors(term_id))
    else:
        # nx.ancestors follows edges backward (predecessors) transitively
        child_ids = list(nx.ancestors(graph, term_id))

    # --- Siblings: other children of immediate parents (always depth=1) ---
    immediate_parent_ids = list(graph.successors(term_id))
    sibling_ids: list[str] = []
    seen: set[str] = set()
    for pid in immediate_parent_ids:
        for sibling in graph.predecessors(pid):
            if sibling != term_id and sibling not in seen:
                sibling_ids.append(sibling)
                seen.add(sibling)

    return {
        "parents": [_format_term(graph, tid) for tid in parent_ids],
        "children": [_format_term(graph, tid) for tid in child_ids],
        "siblings": [_format_term(graph, tid) for tid in sibling_ids],
    }
