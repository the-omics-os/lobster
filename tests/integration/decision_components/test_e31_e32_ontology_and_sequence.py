"""E31: an ontology term picked in the browser; E32: a sequence region.

On the real binary under a pty, with the scripted supervisor. The ontology
is a small made-up one (no download); the sequence is synthetic.
"""

from __future__ import annotations

import networkx as nx

from lobster.services.interaction import view_sources
from lobster.services.interaction.view_sources import Ontology, OntologyUnavailable
from tests.integration.decision_components._harness import (
    ask,
    new_id,
    tool_results,
)


def _toy() -> Ontology:
    graph = nx.MultiDiGraph()
    for term, name, parent, relation in [
        ("TOY:1", "anatomical entity", None, None),
        ("TOY:2", "organ", "TOY:1", "is_a"),
        ("TOY:3", "liver", "TOY:2", "is_a"),
        ("TOY:4", "lung", "TOY:2", "is_a"),
        ("TOY:5", "liver lobule", "TOY:3", "is_a"),
        ("TOY:7", "liver parenchyma", "TOY:3", "part_of"),
    ]:
        graph.add_node(term, name=name)
        if parent:
            graph.add_edge(term, parent, key=relation)
    return Ontology("toy", graph)


def _tissue(**field):
    return {
        "fields": [
            {
                "key": "tissue",
                "type": "choice",
                "view": "ontology_browser",
                "question": "Which tissue is S01 from?",
                "options_from": {"ontology": "toy", "suggest": ["liver", "lung"]},
                **field,
            }
        ]
    }


def test_e31_a_finer_term_is_picked_in_the_browser(start_tui, monkeypatch):
    """The suggestions come first; Browse opens the tree on the last one
    highlighted (lung); up is liver parenchyma (a part of liver), which
    space picks, by its id."""
    monkeypatch.setattr(view_sources, "load_ontology", lambda name: _toy())
    cid = new_id()
    tui = start_tui([ask((cid, "Which tissue is S01 from?", _tissue()))])
    tui.ask_supervisor()
    render = tui.wait_render(1)
    assert render.component == "ontology_browser"
    assert render.data["suggested"] == ["TOY:3", "TOY:4"]
    tui.wait_screen("Suggested terms:")
    tui.key("down", "down", "enter")
    tui.wait_screen("Ontology Browser")
    tui.key("up", "space")
    assert tui.wait_turn_end() == ""
    result = tool_results(tui.client)[cid]
    assert result["action"] == "submit"
    assert result["answer"] == {"tissue": {"id": "TOY:7"}}


def test_e31_without_the_ontology_the_user_gets_a_plain_choice_and_a_notice(
    start_tui, monkeypatch
):
    def missing(name):
        raise OntologyUnavailable(
            "ontology browsing isn't installed (the obonet package)"
        )

    monkeypatch.setattr(view_sources, "load_ontology", missing)
    cid = new_id()
    tui = start_tui([ask((cid, "Which tissue is S01 from?", _tissue()))])
    tui.ask_supervisor()
    render = tui.wait_render(1)
    assert render.component == "choice"
    tui.wait_screen("Lobster: ontology browsing isn't installed")
    tui.wait_screen("lobster-ai[vector-search]")
    tui.key("down", "enter")
    assert tui.wait_turn_end() == ""
    result = tool_results(tui.client)[cid]
    assert result["answer"] == {"tissue": {"id": "lung"}}
    assert "names, not toy ids" in result["note"]


FASTA = ">construct_1 synthetic\nACGTACGTACGGGTTTAACCGGTTAAACCC\n"


def test_e32_a_region_is_picked_and_its_letters_shown(start_tui, workspace):
    (workspace / "data").mkdir(exist_ok=True)
    (workspace / "data" / "construct.fasta").write_text(FASTA)
    context = {
        "fields": [
            {
                "key": "guide",
                "type": "choice",
                "view": "sequence",
                "question": "Which guide should we use?",
                "sequence_from": {"file": "data/construct.fasta"},
                "options": [
                    {"id": "g1", "label": "guide 1", "start": 1, "end": 5},
                    {"id": "g2", "label": "guide 2", "start": 11, "end": 17},
                ],
            }
        ]
    }
    cid = new_id()
    tui = start_tui([ask((cid, "Which guide should we use?", context))])
    tui.ask_supervisor()
    render = tui.wait_render(1)
    assert render.component == "sequence_view"
    assert render.data["sequence"] == "ACGTACGTACGGGTTTAACCGGTTAAACCC"
    tui.wait_screen("guide 2")
    tui.key("down")
    tui.wait_screen("GGGTTTA")  # the highlighted region's letters
    tui.key("enter")
    assert tui.wait_turn_end() == ""
    result = tool_results(tui.client)[cid]
    assert result["answer"] == {"guide": {"id": "g2"}}
