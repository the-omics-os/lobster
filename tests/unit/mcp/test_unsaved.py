"""A turn that made data and published none says so."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lobster.cli_internal import mcp_record


def entry(name, parent=None):
    return {
        "name": name,
        "base_name": name,
        "version": 1,
        "processing_step": "x",
        "parent_modality": parent,
        "created_at": f"2026-11-13T10:00:{len(name):02d}",
        "shape": {"n_obs": 3, "n_vars": 2},
    }


def later_turn(before, after, data_files):
    state = {"now": before}
    dm = SimpleNamespace(
        list_modalities_with_lineage=lambda: [entry(n) for n in state["now"]],
        provenance=SimpleNamespace(activities=[]),
    )
    turn = mcp_record.Turn.begin(dm, 2)
    state["now"] = after
    found = mcp_record.collect(
        SimpleNamespace(data_manager=dm),
        turn,
        {"files_written": []},
        exported=True,
        open_question=False,
        data_files=data_files,
    )
    return [w["code"] for w in found["warnings"]]


@pytest.mark.parametrize(
    "data_files",
    [{}, {"a": "/out/a.h5ad"}],
    ids=["no file", "only an earlier turn's file"],
)
def test_a_later_turn_that_made_data_and_published_none_says_so(data_files):
    assert "no_data_file_saved" in later_turn(["a"], ["a", "b"], data_files)


@pytest.mark.parametrize(
    "after, data_files",
    [(["a"], {}), (["a", "b"], {"b": "/out/b.h5ad"})],
    ids=["no new data", "the new data was published"],
)
def test_a_later_turn_is_not_warned_when_it_need_not_be(after, data_files):
    assert "no_data_file_saved" not in later_turn(["a"], after, data_files)


def test_the_turn_notes_the_modalities_it_began_with():
    dm = SimpleNamespace(
        list_modalities_with_lineage=lambda: [entry("a"), entry("b")],
        provenance=SimpleNamespace(activities=[]),
    )
    assert mcp_record.Turn.begin(dm, 2).modalities_before == ("a", "b")
    assert mcp_record.Turn.begin(dm, 1).modalities_before == ()


def test_a_file_read_back_without_lineage_is_not_the_newest_data():
    """After a restart the engine stamps such a file with the time it read it."""
    raw = {**entry("raw"), "created_at": "2026-11-13T12:00:00"}
    raw["step_summary"] = "[Legacy] Loaded from raw.h5ad"
    worked = {**entry("clustered"), "created_at": "2026-11-13T10:00:00"}
    worked["shape"] = {"n_obs": 2612, "n_vars": 9}
    dm = SimpleNamespace(list_modalities_with_lineage=lambda: [raw, worked])
    entries, lineage = mcp_record._chain(dm)
    assert lineage[0]["name"] == "clustered"
    only = SimpleNamespace(list_modalities_with_lineage=lambda: [raw])
    assert mcp_record._chain(only)[1][0]["name"] == "raw"
