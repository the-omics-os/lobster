"""The ontology_browser and sequence views: where their data comes from
(view_sources), their checks, and how the Go TUI and classic UI get them.
A made-up ontology and synthetic sequences only; no network."""

import builtins
import sys
import types

import networkx as nx
import pytest
from pydantic import ValidationError

from lobster.cli_internal import classic_interaction
from lobster.cli_internal.decision_screens import _from_reply, _render_one, collect
from lobster.services.interaction import view_sources
from lobster.services.interaction.component_schemas import (
    AskUserContext,
    DecisionRequest,
)
from lobster.services.interaction.view_sources import (
    Ontology,
    OntologyUnavailable,
    expand_view_sources,
    read_fasta,
    reread_sequences,
)
from lobster.tools.user_interaction import (
    _build_request,
    _with_notes,
    create_ask_user_tool,
)


def _toy() -> Ontology:
    graph = nx.MultiDiGraph()

    def term(term_id, name, parent=None, **data):
        graph.add_node(term_id, name=name, **data)
        if parent:
            graph.add_edge(term_id, parent, key="is_a")

    term("TOY:1", "anatomical entity")
    term("TOY:2", "organ", "TOY:1")
    term(
        "TOY:3",
        "liver",
        "TOY:2",
        synonym=['"hepar" EXACT []', '"liverish" RELATED []'],
        **{"def": '"An organ that clears things." [TOY:x]'},
    )
    term("TOY:4", "lung", "TOY:2")
    term("TOY:5", "liver lobule", "TOY:3")
    term("TOY:6", "old liver part", "TOY:3", is_obsolete="true")
    graph.add_node("TOY:7", name="liver parenchyma")
    graph.add_edge("TOY:7", "TOY:3", key="part_of")  # a part: a finer term
    graph.add_edge("TOY:5", "TOY:2", key="part_of")  # not a parent
    return Ontology("toy", graph)


def _ontology_field(**field):
    return {
        "key": "tissue",
        "type": "choice",
        "view": "ontology_browser",
        "options_from": {"ontology": "toy", "suggest": ["liver", "lung"]},
        **field,
    }


def _expand(context, workspace=None):
    return expand_view_sources(context, workspace, load=lambda name: _toy())


# --- ontology terms ----------------------------------------------------------


def test_ontology_finds_terms_by_id_name_and_exact_synonym():
    toy = _toy()
    assert toy.find("TOY:3") == "TOY:3"
    assert toy.find("  Liver ") == "TOY:3"
    assert toy.find("hepar") == "TOY:3"
    assert toy.find("liverish") is None  # only EXACT synonyms
    assert toy.find("old liver part") is None  # obsolete
    assert toy.find("TOY:6") is None


def test_suggested_names_become_real_ids_with_their_neighbourhood():
    context, problems, notes = _expand({"fields": [_ontology_field()]})
    assert problems == [] and notes == []
    options = {o["id"]: o for o in context["fields"][0]["options"]}
    # The path to the top, the suggestions, then liver's kinds and parts
    # (part_of under a suggestion); no obsolete term, and part_of never
    # makes a parent on the path to the top.
    assert list(options) == ["TOY:1", "TOY:2", "TOY:3", "TOY:4", "TOY:5", "TOY:7"]
    assert options["TOY:5"]["parent"] == "TOY:3"
    assert options["TOY:7"]["parent"] == "TOY:3"
    assert "parent" not in options["TOY:1"]
    assert options["TOY:3"]["details"].startswith("suggested · An organ")
    asked = AskUserContext.model_validate(context)
    assert asked.fields[0].options_from.ontology == "toy"


def test_an_unknown_name_is_a_problem_for_the_supervisor():
    field = _ontology_field(options_from={"ontology": "toy", "suggest": ["livr"]})
    _, problems, _ = _expand({"fields": [field]})
    assert problems[0].path == "fields.0.options_from.suggest"
    assert "no term named 'livr'" in problems[0].message


def test_options_written_by_the_supervisor_are_refused():
    _, problems, _ = _expand({"fields": [_ontology_field(options=["UBERON:1"])]})
    assert "leave options out" in problems[0].message


def test_an_unknown_ontology_is_a_problem():
    def unknown(name):
        raise ValueError(f"unknown ontology {name!r}; available: uberon")

    _, problems, _ = expand_view_sources(
        {"fields": [_ontology_field()]}, None, load=unknown
    )
    assert problems[0].path == "fields.0.options_from.ontology"


def test_a_default_by_name_becomes_the_term_id():
    field = _ontology_field(for_each=["S01", "S02"], default={"S01": "hepar"})
    context, problems, _ = _expand({"fields": [field]})
    assert problems == []
    assert context["fields"][0]["default"] == {"S01": {"id": "TOY:3"}}


def test_without_the_ontology_it_is_a_plain_choice_with_a_notice():
    def missing(name):
        raise OntologyUnavailable(
            "ontology browsing isn't installed (the obonet package)"
        )

    field = _ontology_field(
        for_each=["S01", "S02"], default={"S01": "liver", "S02": "heart"}
    )
    context, problems, notes = expand_view_sources(
        {"fields": [field]}, None, load=missing
    )
    assert problems == []
    asked = AskUserContext.model_validate(context).fields[0]
    assert asked.view == "default"
    assert [o.id for o in asked.options] == ["liver", "lung"]
    assert asked.default == {"S01": {"id": "liver"}}  # "heart" isn't shown
    assert asked.details.startswith("Lobster: ontology browsing isn't installed")
    assert "lobster-ai[vector-search]" in asked.details
    assert "names, not toy ids" in notes[0]


def test_the_supervisor_gets_the_note_with_the_answer():
    built = {"notes": ["The ontology view isn't available here."]}
    result = _with_notes({"action": "submit", "note": "Ask over all."}, built)
    assert result["note"] == "Ask over all. The ontology view isn't available here."
    assert _with_notes({"action": "submit"}, {"notes": []}) == {"action": "submit"}


def test_load_ontology_without_lobster_vector_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "lobster.vector.ontology_graph", None)
    monkeypatch.setattr(view_sources, "_LOADED", {})
    with pytest.raises(OntologyUnavailable, match="lobster-vector"):
        view_sources.load_ontology("uberon")


def test_load_ontology_downloads_once_into_the_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(view_sources, "ONTOLOGY_CACHE", tmp_path)
    monkeypatch.setattr(view_sources, "_LOADED", {})
    fake = types.ModuleType("obonet")
    fake.read_obo = lambda path: _toy().graph
    monkeypatch.setitem(sys.modules, "obonet", fake)
    downloads = []

    class Response:
        def __init__(self):
            self.chunks = [b"format-version: 1.2\n", b""]

        def read(self, size):
            return self.chunks.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(url, timeout):
        downloads.append(url)
        return Response()

    monkeypatch.setattr(view_sources.urllib.request, "urlopen", urlopen)
    loaded = view_sources.load_ontology("uberon")
    assert loaded.find("liver") == "TOY:3"
    assert (tmp_path / "uberon.obo").is_file()
    view_sources._LOADED.clear()
    view_sources.load_ontology("uberon")
    assert len(downloads) == 1  # the second load reads the cached file
    assert not list(tmp_path.glob("*.part"))


def test_a_failed_download_is_unavailable_and_leaves_no_file(monkeypatch, tmp_path):
    monkeypatch.setattr(view_sources, "ONTOLOGY_CACHE", tmp_path)
    monkeypatch.setattr(view_sources, "_LOADED", {})
    monkeypatch.setitem(sys.modules, "obonet", types.ModuleType("obonet"))

    def offline(url, timeout):
        raise OSError("no network")

    monkeypatch.setattr(view_sources.urllib.request, "urlopen", offline)
    with pytest.raises(OntologyUnavailable, match="couldn't download"):
        view_sources.load_ontology("uberon")
    assert list(tmp_path.iterdir()) == []


# --- view checks ---------------------------------------------------------------


def _choice(**field):
    return {"fields": [{"key": "k", "type": "choice", **field}]}


@pytest.mark.parametrize(
    "field, message",
    [
        (
            {"options": [{"id": "a"}, {"id": "b", "parent": "a"}]},
            "parent is only for view 'ontology_browser'",
        ),
        (
            {"options": [{"id": "a", "start": 1, "end": 3}]},
            "start and end are only for view 'sequence'",
        ),
        (
            {
                "view": "ontology_browser",
                "options": [{"id": "a"}],
                "options_from": {"ontology": "toy", "suggest": ["a"]},
                "count": {"min": 1},
            },
            "leave count out",
        ),
        (
            {
                "view": "ontology_browser",
                "options": [{"id": "a", "parent": "b"}, {"id": "b", "parent": "a"}],
                "options_from": {"ontology": "toy", "suggest": ["a"]},
            },
            "loop",
        ),
        (
            {"view": "ontology_browser", "options": [{"id": "a"}]},
            "needs options_from",
        ),
        (
            {"options": [{"id": "a", "start": 5, "end": 2}]},
            "starts at 5, after its end 2",
        ),
    ],
)
def test_view_keys_are_checked(field, message):
    with pytest.raises(ValidationError, match=message):
        AskUserContext.model_validate(_choice(**field))


def test_neither_view_can_be_a_dashboard_section():
    context, _, _ = _expand(
        {
            "view": "dashboard",
            "fields": [
                {**_ontology_field(), "question": "Tissue?"},
                {"key": "n", "type": "number", "question": "How many?"},
            ],
        }
    )
    with pytest.raises(ValidationError, match="can't be in a dashboard yet"):
        AskUserContext.model_validate(context)


# --- sequences -----------------------------------------------------------------

FASTA = ">construct_1 synthetic\nACGTACGTAC\nGGGTTTAACC\n>protein_1\nMKVLA\n"


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "construct.fasta").write_text(FASTA)
    return tmp_path


def _sequence_field(**field):
    return {
        "key": "guide",
        "type": "choice",
        "view": "sequence",
        "sequence_from": {"file": "data/construct.fasta", "id": "construct_1"},
        "options": [
            {"id": "g1", "label": "guide 1", "start": 1, "end": 5},
            {"id": "g2", "label": "guide 2", "start": 11, "end": 20},
        ],
        **field,
    }


def test_read_fasta_reads_a_record_and_its_kind(workspace):
    assert read_fasta(workspace, "data/construct.fasta", "construct_1") == (
        "construct_1",
        "ACGTACGTACGGGTTTAACC",
        "dna",
    )
    assert read_fasta(workspace, "data/construct.fasta", "protein_1")[2] == "protein"


@pytest.mark.parametrize(
    "record, message",
    [(None, "has 2 records"), ("nope", "has no record 'nope'")],
)
def test_read_fasta_says_which_record_is_missing(workspace, record, message):
    with pytest.raises(ValueError, match=message):
        read_fasta(workspace, "data/construct.fasta", record)


def test_read_fasta_stays_inside_the_workspace(workspace):
    with pytest.raises(ValueError):
        read_fasta(workspace, "../outside.fasta", None)


def test_the_sequence_is_read_and_its_file_watched(workspace):
    context, problems, _ = expand_view_sources(
        {"fields": [_sequence_field()]}, workspace
    )
    assert problems == []
    assert context["data_files"] == ["data/construct.fasta"]
    source = AskUserContext.model_validate(context).fields[0].sequence_from
    assert (source.sequence, source.kind) == ("ACGTACGTACGGGTTTAACC", "dna")


def test_letters_from_the_supervisor_are_refused(workspace):
    field = _sequence_field(
        sequence_from={"file": "data/construct.fasta", "sequence": "ACGT"}
    )
    _, problems, _ = expand_view_sources({"fields": [field]}, workspace)
    assert "read from the file" in problems[0].message


def test_a_region_past_the_end_is_refused(workspace):
    field = _sequence_field(options=[{"id": "g1", "start": 18, "end": 21}])
    context, _, _ = expand_view_sources({"fields": [field]}, workspace)
    with pytest.raises(ValidationError, match="past the end of the sequence"):
        AskUserContext.model_validate(context)


def test_a_changed_sequence_is_read_again_on_a_reask(workspace):
    request = _build_request(
        "Which guide?", {"fields": [_sequence_field()]}, "dec-1", None, workspace
    )
    (workspace / "data" / "construct.fasta").write_text(
        FASTA.replace("ACGTACGTAC", "TTTTACGTAC")
    )
    again = reread_sequences(request, workspace)
    assert again.fields[0].sequence_from.sequence.startswith("TTTTA")
    assert "changed" in again.problems[-1].message


def test_a_sequence_that_no_longer_fits_keeps_the_old_one(workspace):
    request = _build_request(
        "Which guide?", {"fields": [_sequence_field()]}, "dec-1", None, workspace
    )
    (workspace / "data" / "construct.fasta").write_text(">construct_1\nACGT\n")
    again = reread_sequences(request, workspace)
    assert again.fields[0].sequence_from.sequence == "ACGTACGTACGGGTTTAACC"
    assert "couldn't read data/construct.fasta again" in again.problems[-1].message


# --- surfaces ------------------------------------------------------------------


def _request(context, workspace=None, monkeypatch=None) -> DecisionRequest:
    if monkeypatch is not None:
        monkeypatch.setattr(view_sources, "load_ontology", lambda name: _toy())
    built = _build_request("Which tissue?", context, "dec-1", None, workspace)
    assert isinstance(built, DecisionRequest), built
    return built


def test_go_gets_the_tree_and_the_suggestions(monkeypatch):
    request = _request({"fields": [_ontology_field()]}, monkeypatch=monkeypatch)
    component, data = _render_one(request.fields[0])
    assert component == "ontology_browser"
    assert data["suggested"] == ["TOY:3", "TOY:4"]
    assert data["ontology"] == "toy"
    assert {"id": "TOY:5", "label": "liver lobule", "parent": "TOY:3"} in data[
        "options"
    ]
    reply = {"selected": [{"id": "TOY:5"}]}
    assert _from_reply(request.fields[0], component, data, reply) == {"id": "TOY:5"}


def test_each_item_gets_its_own_browser(monkeypatch):
    request = _request(
        {"fields": [_ontology_field(for_each=["S01", "S02"])]},
        monkeypatch=monkeypatch,
    )
    asked = []

    def render(component, data):
        asked.append((component, data["question"]))
        return "submit", {"selected": [{"id": "TOY:4"}]}

    action, answer = collect(request.to_wire(), render)
    assert action == "submit"
    assert answer == {"tissue": {"S01": {"id": "TOY:4"}, "S02": {"id": "TOY:4"}}}
    assert [c for c, _ in asked] == ["ontology_browser", "ontology_browser"]
    assert asked[0][1].startswith("S01")


def test_go_gets_the_letters_and_the_regions(workspace):
    request = _request({"fields": [_sequence_field()]}, workspace)
    component, data = _render_one(request.fields[0])
    assert component == "sequence_view"
    assert data["sequence"] == "ACGTACGTACGGGTTTAACC"
    assert (data["kind"], data["name"]) == ("dna", "construct_1")
    assert data["options"][1] == {
        "id": "g2",
        "label": "guide 2",
        "start": 11,
        "end": 20,
    }


def test_classic_lists_the_regions_with_their_letters(workspace, capsys, monkeypatch):
    request = _request({"fields": [_sequence_field()]}, workspace)
    monkeypatch.setattr(builtins, "input", lambda prompt: "2")
    action, answer = collect(request.to_wire(), classic_interaction._render)
    assert (action, answer) == ("submit", {"guide": {"id": "g2"}})
    shown = capsys.readouterr().out
    assert "construct_1: DNA, 20 letters" in shown
    assert "11–20: GGGTTTAACC" in shown


def test_classic_shows_suggestions_then_the_tree(capsys, monkeypatch):
    request = _request({"fields": [_ontology_field()]}, monkeypatch=monkeypatch)
    # 1-2 the suggestions, then the tree: 3 anatomical entity, 4 organ,
    # 5 liver, 6 liver lobule, 7 liver parenchyma, 8 lung.
    monkeypatch.setattr(builtins, "input", lambda prompt: "8")
    action, answer = collect(request.to_wire(), classic_interaction._render)
    assert (action, answer) == ("submit", {"tissue": {"id": "TOY:4"}})
    shown = capsys.readouterr().out
    assert "1. ★ liver (TOY:3)" in shown
    assert "      liver lobule (TOY:5)" in shown


def test_the_docstring_lists_both_views():
    description = " ".join(create_ask_user_tool().description.split())
    assert '"ontology_browser" (choice with options_from)' in description
    assert '"sequence" (choice with sequence_from)' in description
    assert "options_from, sequence_from (choice)" in description
