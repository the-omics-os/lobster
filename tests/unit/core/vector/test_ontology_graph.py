"""
Unit tests for lobster.core.vector.ontology_graph module.

Tests cover:
- OBO_URLS completeness and correctness (GRPH-03)
- load_ontology_graph error handling and caching (GRPH-01)
- get_neighbors parent/child/sibling traversal (GRPH-02)
- Edge cases for leaf and root nodes

All tests use mocked obonet — no real network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from lobster.core.vector.ontology_graph import (
    OBO_URLS,
    _format_term,
    get_neighbors,
    load_ontology_graph,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_graph() -> nx.MultiDiGraph:
    """Create a small test MultiDiGraph mimicking OBO edge direction.

    OBO direction: child -> parent (is_a)

    Structure::

        grandparent
          ^       ^
          |       |
        parent1  parent2
          ^  ^      ^
          |  |      |
       child1 child2 child3
    """
    g = nx.MultiDiGraph()

    # Edges: child -> parent (is_a)
    g.add_edge("child1", "parent1")
    g.add_edge("child2", "parent1")
    g.add_edge("child3", "parent2")
    g.add_edge("parent1", "grandparent")
    g.add_edge("parent2", "grandparent")

    # Node names
    g.nodes["child1"]["name"] = "Child One"
    g.nodes["child2"]["name"] = "Child Two"
    g.nodes["child3"]["name"] = "Child Three"
    g.nodes["parent1"]["name"] = "Parent One"
    g.nodes["parent2"]["name"] = "Parent Two"
    g.nodes["grandparent"]["name"] = "Grandparent"

    return g


@pytest.fixture(autouse=True)
def _clear_lru_cache():
    """Clear the lru_cache on load_ontology_graph before and after each test."""
    load_ontology_graph.cache_clear()
    yield
    load_ontology_graph.cache_clear()


# ---------------------------------------------------------------------------
# TestOBOUrls (GRPH-03)
# ---------------------------------------------------------------------------


class TestOBOUrls:
    """Verify OBO_URLS constant covers required ontologies."""

    def test_obo_urls_has_mondo(self):
        assert "mondo" in OBO_URLS
        assert OBO_URLS["mondo"] == "https://purl.obolibrary.org/obo/mondo.obo"

    def test_obo_urls_has_uberon(self):
        assert "uberon" in OBO_URLS
        assert OBO_URLS["uberon"] == "https://purl.obolibrary.org/obo/uberon/uberon-basic.obo"

    def test_obo_urls_has_cell_ontology(self):
        assert "cell_ontology" in OBO_URLS
        assert OBO_URLS["cell_ontology"] == "https://purl.obolibrary.org/obo/cl.obo"

    def test_obo_urls_exactly_three_entries(self):
        assert len(OBO_URLS) == 3


# ---------------------------------------------------------------------------
# TestLoadOntologyGraph (GRPH-01)
# ---------------------------------------------------------------------------


class TestLoadOntologyGraph:
    """Verify load_ontology_graph error handling and delegation to obonet."""

    def test_load_unknown_ontology_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown ontology 'bogus'"):
            load_ontology_graph("bogus")

    def test_load_unknown_ontology_lists_available(self):
        with pytest.raises(ValueError, match="Available:"):
            load_ontology_graph("not_an_ontology")

    def test_load_missing_obonet_raises_import_error(self):
        """Simulate obonet not installed by patching __import__."""
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def mock_import(name, *args, **kwargs):
            if name == "obonet":
                raise ImportError("No module named 'obonet'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="pip install"):
                load_ontology_graph("mondo")

    @patch("lobster.core.vector.ontology_graph.obonet", create=True)
    def test_load_calls_obonet_read_obo(self, mock_obonet_module):
        """Verify obonet.read_obo is called with the correct URL."""
        mock_graph = MagicMock(spec=nx.MultiDiGraph)
        mock_graph.number_of_nodes.return_value = 100
        mock_graph.number_of_edges.return_value = 200

        # Patch the import inside the function
        with patch.dict("sys.modules", {"obonet": mock_obonet_module}):
            mock_obonet_module.read_obo.return_value = mock_graph
            result = load_ontology_graph("mondo")

        mock_obonet_module.read_obo.assert_called_once_with(
            "https://purl.obolibrary.org/obo/mondo.obo"
        )
        assert result is mock_graph

    @patch.dict("sys.modules", {"obonet": MagicMock()})
    def test_load_returns_graph(self):
        """Verify load returns the graph object from obonet."""
        import sys

        mock_obonet = sys.modules["obonet"]
        mock_graph = MagicMock(spec=nx.MultiDiGraph)
        mock_graph.number_of_nodes.return_value = 50
        mock_graph.number_of_edges.return_value = 80
        mock_obonet.read_obo.return_value = mock_graph

        result = load_ontology_graph("uberon")
        assert result is mock_graph


# ---------------------------------------------------------------------------
# TestGetNeighbors (GRPH-02)
# ---------------------------------------------------------------------------


class TestGetNeighbors:
    """Verify get_neighbors traversal logic with a small test graph."""

    def test_parents_depth_1(self, sample_graph):
        result = get_neighbors(sample_graph, "child1", depth=1)
        parent_ids = [p["term_id"] for p in result["parents"]]
        assert "parent1" in parent_ids
        # child1 only has parent1 as direct parent
        assert len(parent_ids) == 1

    def test_children_depth_1(self, sample_graph):
        result = get_neighbors(sample_graph, "parent1", depth=1)
        child_ids = [c["term_id"] for c in result["children"]]
        assert "child1" in child_ids
        assert "child2" in child_ids
        assert len(child_ids) == 2

    def test_siblings(self, sample_graph):
        result = get_neighbors(sample_graph, "child1", depth=1)
        sibling_ids = [s["term_id"] for s in result["siblings"]]
        assert "child2" in sibling_ids
        # child1 should not appear as its own sibling
        assert "child1" not in sibling_ids

    def test_parents_depth_2(self, sample_graph):
        result = get_neighbors(sample_graph, "child1", depth=2)
        parent_ids = {p["term_id"] for p in result["parents"]}
        assert "parent1" in parent_ids
        assert "grandparent" in parent_ids

    def test_missing_term_returns_empty(self, sample_graph):
        result = get_neighbors(sample_graph, "NONEXISTENT:999")
        assert result["parents"] == []
        assert result["children"] == []
        assert result["siblings"] == []

    def test_format_includes_name(self, sample_graph):
        result = get_neighbors(sample_graph, "child1", depth=1)
        parent = result["parents"][0]
        assert "term_id" in parent
        assert "name" in parent
        assert parent["term_id"] == "parent1"
        assert parent["name"] == "Parent One"


# ---------------------------------------------------------------------------
# TestLruCache (GRPH-01 caching)
# ---------------------------------------------------------------------------


class TestLruCache:
    """Verify load_ontology_graph uses lru_cache for process-lifetime caching."""

    @patch.dict("sys.modules", {"obonet": MagicMock()})
    def test_load_ontology_graph_cached(self):
        """Calling load_ontology_graph twice should invoke read_obo only once."""
        import sys

        mock_obonet = sys.modules["obonet"]
        mock_graph = MagicMock(spec=nx.MultiDiGraph)
        mock_graph.number_of_nodes.return_value = 10
        mock_graph.number_of_edges.return_value = 20
        mock_obonet.read_obo.return_value = mock_graph

        result1 = load_ontology_graph("mondo")
        result2 = load_ontology_graph("mondo")

        assert result1 is result2
        mock_obonet.read_obo.assert_called_once()


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for leaf and root nodes."""

    def test_get_neighbors_leaf_node(self, sample_graph):
        """A leaf node (child1) has no children."""
        result = get_neighbors(sample_graph, "child1", depth=1)
        assert result["children"] == []

    def test_get_neighbors_root_node(self, sample_graph):
        """A root node (grandparent) has no parents."""
        result = get_neighbors(sample_graph, "grandparent", depth=1)
        assert result["parents"] == []

    def test_format_term_missing_name(self, sample_graph):
        """Node without name attribute returns empty string."""
        # Add a node with no "name" attribute
        sample_graph.add_node("nameless_node")
        result = _format_term(sample_graph, "nameless_node")
        assert result["term_id"] == "nameless_node"
        assert result["name"] == ""

    def test_format_term_not_in_graph(self, sample_graph):
        """Term not in graph returns empty name."""
        result = _format_term(sample_graph, "MISSING:001")
        assert result["term_id"] == "MISSING:001"
        assert result["name"] == ""
