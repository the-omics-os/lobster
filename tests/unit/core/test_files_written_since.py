"""``files_written_since`` lists what is on disk, not what is in memory. Real disk."""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

import anndata as ad
import numpy as np
import plotly.graph_objects as go
import pytest

from lobster.core.figures.errors import FigureError, StorageError
from lobster.core.runtime.data_manager import DataManagerV2


@pytest.fixture()
def dm(tmp_path):
    return DataManagerV2(workspace_path=tmp_path)


def _write(path, age_seconds=0, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    stamp = time.time() - age_seconds
    os.utime(path, (stamp, stamp))
    return path


def _names(entries):
    return [e["relative"] for e in entries]


def test_a_fresh_workspace_lists_nothing(dm):
    assert dm.files_written_since() == []


def test_a_file_older_than_since_is_left_out_and_a_newer_one_is_listed(dm):
    _write(dm.data_dir / "old.h5ad", age_seconds=3600)
    _write(dm.data_dir / "new.h5ad")
    cutoff = time.time() - 60
    assert _names(dm.files_written_since(cutoff)) == ["data/new.h5ad"]
    assert _names(dm.files_written_since()) == ["data/old.h5ad", "data/new.h5ad"]


def test_an_overwritten_file_counts_as_written(dm):
    path = _write(dm.exports_dir / "report.txt", age_seconds=3600)
    cutoff = time.time() - 60
    assert dm.files_written_since(cutoff) == []
    _write(path, text="again")
    [entry] = dm.files_written_since(cutoff)
    assert entry["relative"] == "exports/report.txt"
    assert entry["category"] == "exports" and entry["size_bytes"] == 5


def test_a_deleted_file_is_dropped(dm):
    path = _write(dm.data_dir / "gone.h5ad")
    path.unlink()
    assert dm.files_written_since() == []


def test_a_modality_that_only_lives_in_memory_is_not_a_file(dm):
    dm.store_modality("pbmc", ad.AnnData(np.ones((3, 3), dtype=np.float32)))
    assert "pbmc" in dm.list_modalities()
    assert dm.files_written_since() == []


def test_saving_the_modality_makes_it_a_file(dm):
    dm.store_modality("pbmc", ad.AnnData(np.ones((3, 3), dtype=np.float32)))
    dm.save_modality("pbmc", "pbmc.h5ad")
    [entry] = dm.files_written_since()
    assert entry["relative"] == "data/pbmc.h5ad" and entry["category"] == "data"
    assert os.path.isabs(entry["path"]) and entry["size_bytes"] > 0
    parsed = datetime.fromisoformat(entry["modified"])
    assert parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0)


def test_internals_and_cache_never_appear(dm):
    _write(dm.workspace_path / ".lobster" / "state.json")
    _write(dm.cache_dir / "scratch.bin")
    assert dm.files_written_since() == []
    assert dm.files_written_since(categories=("data",)) == []


def test_a_managed_figure_carries_its_ref(dm):
    fig = go.Figure(go.Bar(x=[1], y=[2]))
    plot_id, _, _ = dm.plot_manager.add_plot(plot=fig, title="A", source="t")
    [res] = dm.plot_manager.commit_staged(
        {"id": "a-1"}, tool_call_id="c1", staged_ids=[plot_id]
    )
    [entry] = dm.files_written_since()
    assert entry["category"] == "plots" and entry["figure_ref"] == res.ref
    assert entry["relative"].endswith("figure.json")


def test_a_new_revision_is_listed_after_the_cutoff(dm):
    def commit(call):
        fig = go.Figure(go.Bar(x=[1], y=[2]))
        pid, _, _ = dm.plot_manager.add_plot(plot=fig, title=call, source="t")
        [res] = dm.plot_manager.commit_staged(
            {"id": call}, tool_call_id=call, staged_ids=[pid]
        )
        return res.ref

    first = commit("c1")
    for item in dm.files_written_since():
        os.utime(item["path"], (time.time() - 3600,) * 2)
    cutoff = time.time() - 60
    assert dm.files_written_since(cutoff) == []
    second = commit("c2")
    assert [e["figure_ref"] for e in dm.files_written_since(cutoff)] == [second]
    assert first != second


def test_a_timezone_aware_datetime_is_accepted_and_a_naive_one_refused(dm):
    _write(dm.data_dir / "a.h5ad")
    aware = datetime.now(timezone.utc) - timedelta(minutes=1)
    assert _names(dm.files_written_since(aware)) == ["data/a.h5ad"]
    with pytest.raises(ValueError):
        dm.files_written_since(datetime.now())


def test_an_unknown_category_is_refused(dm):
    with pytest.raises(ValueError):
        dm.files_written_since(categories=("cache",))


def test_a_storage_fault_is_raised_not_a_partial_list(dm, monkeypatch):
    _write(dm.data_dir / "a.h5ad")

    def broken():
        raise StorageError("catalog unreadable")

    monkeypatch.setattr(dm, "_saved_figure_records", broken)
    with pytest.raises(FigureError):
        dm.files_written_since()


def test_the_supervisor_prompt_treats_a_modality_name_as_no_evidence_of_a_file():
    # Prompt wording only: that a model obeys it is not provable offline. The wording must
    # hold on every surface, so it neither says a modality is saved nor that it is not.
    from lobster.agents.supervisor import _build_agent_result_memory

    text = " ".join(_build_agent_result_memory().split())
    assert "A modality NAME is not evidence of a file" in text
    assert (
        "the manifest lists it under files or a tool result names the written path"
        in text
    )
    assert (
        "never state that it is NOT saved unless the manifest or a tool result shows that"
        in text
    )
    assert "held in memory" not in text and "It is NOT a file" not in text
