"""QC plots of a turn get an HTML view at ``<output_dir>/plots``.

Real figures are committed through the data manager's plot manager on a temp
workspace and viewed through the engine's own ``ensure_html_renditions``; no
model is involved.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from types import SimpleNamespace

import plotly.graph_objects as go
import pytest

from lobster.cli_internal import mcp_record, mcp_stdio
from lobster.cli_internal.mcp_stdio import RunRequest
from lobster.core.figures import renditions
from lobster.core.figures.errors import StorageError
from lobster.core.runs import new_run_id
from lobster.core.runtime.data_manager import DataManagerV2
from tests.unit.mcp.test_run import (  # noqa: F401  (fixtures)
    COMPLETE,
    FakeDataManager,
    data,
    engine,
)


def commit(dm, title, color="red"):
    call = f"call-{title}-{color}"
    fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4], marker={"color": color}))
    plot_id, _, _ = dm.plot_manager.add_plot(plot=fig, title=title, source="test")
    [res] = dm.plot_manager.commit_staged(
        {"id": f"act-{call}"}, tool_call_id=call, staged_ids=[plot_id]
    )
    return res.ref


@pytest.fixture
def session(tmp_path):
    out = tmp_path / "out"
    return out, DataManagerV2(workspace_path=out / ".lobster_workspace")


def test_a_committed_figure_gets_an_html_view_in_output_dir(session):
    out, dm = session
    old = commit(dm, "Before the turn")
    turn = mcp_record.Turn.begin(dm)
    ref = commit(dm, "QC Plots - pbmc3k")

    views = mcp_record.publish_plot_views(dm, turn, out)

    [view] = views.published
    path = out / "plots" / "QC-Plots-pbmc3k.html"
    assert Path(view["path"]) == path and path.is_file()
    assert view["category"] == "plots" and view["format"] == "html"
    assert view["title"] == "QC Plots - pbmc3k"
    assert view["size_bytes"] == path.stat().st_size > 1000
    assert view["figure_ref"] == ref and old != ref
    assert not re.findall(r"<script[^>]*\bsrc=", path.read_text(encoding="utf-8"))
    assert (views.total, views.failed) == (1, 0)
    # Only the figure of this turn: the earlier one got no view.
    assert [p.name for p in (out / "plots").iterdir()] == ["QC-Plots-pbmc3k.html"]


def test_a_view_is_a_copy_the_user_can_keep(session):
    out, dm = session
    turn = mcp_record.Turn.begin(dm)
    commit(dm, "QC")
    [view] = mcp_record.publish_plot_views(dm, turn, out).published
    assert Path(view["path"]).stat().st_nlink == 1
    assert Path(view["source"]).is_file() and Path(view["source"]) != Path(view["path"])


def test_a_view_takes_the_mode_of_a_new_file(session):
    out, dm = session
    turn = mcp_record.Turn.begin(dm)
    commit(dm, "QC")
    old = os.umask(0o022)
    try:
        [view] = mcp_record.publish_plot_views(dm, turn, out).published
    finally:
        os.umask(old)
    # The session writes its pages private (0600); the user's copy is not.
    assert Path(view["source"]).stat().st_mode & 0o777 == 0o600
    assert Path(view["path"]).stat().st_mode & 0o777 == 0o644


def test_a_turn_without_a_figure_has_no_view_and_no_folder(session):
    out, dm = session
    turn = mcp_record.Turn.begin(dm)
    views = mcp_record.publish_plot_views(dm, turn, out)
    assert (views.published, views.total, views.failed) == ([], 0, 0)
    assert not (out / "plots").exists()


def test_two_figures_with_one_label_do_not_overwrite_each_other(session):
    out, dm = session
    turn = mcp_record.Turn.begin(dm)
    commit(dm, "QC", color="red")
    commit(dm, "QC", color="blue")
    views = mcp_record.publish_plot_views(dm, turn, out)
    names = sorted(Path(v["path"]).name for v in views.published)
    assert len(names) == 2 and len(set(names)) == 2
    assert "QC.html" in names


def test_the_label_never_leaves_the_plots_folder(session):
    out, dm = session
    turn = mcp_record.Turn.begin(dm)
    commit(dm, "../../etc/pa ss‮wd\n*?")
    [view] = mcp_record.publish_plot_views(dm, turn, out).published
    assert Path(view["path"]).parent == out / "plots"
    assert re.fullmatch(r"[A-Za-z0-9._-]+\.html", Path(view["path"]).name)
    assert "‮" not in view["title"] and "\n" not in view["title"]


def test_at_most_six_figures_get_a_view_and_the_total_is_visible(session, monkeypatch):
    out, dm = session
    # A small page instead of the 5 MB one: the cap is what is tested here.
    monkeypatch.setattr(
        renditions,
        "render_html",
        lambda fig, cfg, div_id: f"<html>{div_id}</html>".encode(),
    )
    turn = mcp_record.Turn.begin(dm)
    for i in range(8):
        commit(dm, f"QC {i}", color=f"#00000{i}")
    views = mcp_record.publish_plot_views(dm, turn, out)
    assert len(views.published) == mcp_record.MAX_PLOT_VIEWS == 6
    assert (views.total, views.failed) == (8, 0)
    assert len(list((out / "plots").iterdir())) == 6


def test_a_view_that_fails_is_counted_and_the_others_stay(session, monkeypatch):
    out, dm = session
    turn = mcp_record.Turn.begin(dm)
    commit(dm, "First")
    bad = commit(dm, "Second", color="blue")
    store = dm.figure_store
    add = store.add_rendition

    def failing(ref, *args, **kwargs):
        if str(ref) == bad:
            raise RuntimeError("disk full at /secret/place")
        return add(ref, *args, **kwargs)

    monkeypatch.setattr(store, "add_rendition", failing)
    views = mcp_record.publish_plot_views(dm, turn, out)
    assert [v["title"] for v in views.published] == ["First"]
    assert (views.total, views.failed) == (2, 1)
    assert not (out / "plots" / "Second.html").exists()


def test_a_session_without_a_figure_store_fails_each_view_not_the_run(tmp_path):
    entry = {"category": "plots", "figure_ref": "fig_a@rev_b", "path": "/x"}
    dm = SimpleNamespace(
        workspace_path=tmp_path,
        files_written_since=lambda since, categories=(): [entry],
    )
    views = mcp_record.publish_plot_views(dm, mcp_record.Turn.begin(dm), tmp_path)
    assert (views.published, views.total, views.failed) == ([], 1, 1)


def test_a_plot_file_that_is_not_a_managed_figure_is_ignored(tmp_path):
    legacy = {"category": "plots", "path": "/w/plots/old.png"}
    dm = SimpleNamespace(
        workspace_path=tmp_path,
        files_written_since=lambda since, categories=(): [legacy],
    )
    views = mcp_record.publish_plot_views(dm, mcp_record.Turn.begin(dm), tmp_path)
    assert (views.published, views.total, views.failed) == ([], 0, 0)


# what the collector says about them --------------------------------------------


def sections(views, **overrides):
    client = SimpleNamespace(data_manager=SimpleNamespace())
    result = {
        "files_written": [{"category": "plots", "path": "/p", "figure_ref": "a@b"}]
    }
    args = {
        "analysis": "scrna_qc",
        "exported": True,
        "open_question": False,
        "plots": views,
        **overrides,
    }
    return mcp_record.collect(client, mcp_record.Turn.begin(None), result, **args)


def codes(found):
    return [w["code"] for w in found["warnings"]]


def test_a_failed_view_is_a_warning_with_the_numbers_only():
    found = sections(mcp_record.PlotViews([], total=3, failed=2))
    [warning] = [w for w in found["warnings"] if w["code"] == "plot_view_failed"]
    assert warning["message"] == (
        "2 of 3 plot views could not be made; the figures are saved in the "
        "session folder."
    )
    assert "plots_truncated" not in found


def test_more_figures_than_views_is_visible_to_the_host():
    found = sections(mcp_record.PlotViews([{}] * 6, total=9, failed=0))
    assert found["plots_truncated"] is True and found["plots_total"] == 9
    assert "plot_view_failed" not in codes(found)
    assert mcp_record.status_fields(found)["plots_total"] == 9
    record = mcp_record.result_record({"run_id": "r"}, found, {})
    assert record["plots_truncated"] is True and record["plots_total"] == 9


def test_no_qc_plots_is_only_for_the_first_turn_of_a_recipe_that_draws_them():
    nothing = {"files_written": []}
    client = SimpleNamespace(data_manager=SimpleNamespace())

    def run(number, analysis="scrna_qc"):
        turn = mcp_record.Turn.begin(None, number)
        return codes(
            mcp_record.collect(
                client,
                turn,
                nothing,
                analysis=analysis,
                exported=True,
                open_question=False,
                plots=mcp_record.PlotViews([], total=0, failed=0),
            )
        )

    assert "no_qc_plots" in run(1)
    assert "no_qc_plots" not in run(2)  # a later turn need not draw plots
    assert "no_qc_plots" not in run(1, analysis="other")


# wiring in a completed run -----------------------------------------------------


def test_published_views_are_listed_in_files_and_in_result_json(
    engine, tmp_path, data, monkeypatch
):
    _opened, script = engine
    script["events"] = [COMPLETE]
    out = tmp_path / "out"
    monkeypatch.setattr(
        FakeDataManager,
        "files_written_since",
        lambda self, since=None, **kwargs: [],
        raising=False,
    )

    def fake_views(data_manager, turn, output_dir):
        page = output_dir / "plots" / "QC.html"
        page.parent.mkdir()
        page.write_text("<html></html>")
        entry = {
            "path": str(page),
            "category": "plots",
            "size_bytes": page.stat().st_size,
            "modified": "2026-11-13T10:00:00+00:00",
            "source": str(output_dir / ".lobster_workspace" / "r.html"),
            "format": "html",
            "title": "QC",
            "figure_ref": "fig@rev",
        }
        return mcp_record.PlotViews([entry], total=7, failed=0)

    monkeypatch.setattr(mcp_record, "publish_plot_views", fake_views)
    request = RunRequest(new_run_id(), str(data), str(out), "scrna_qc")
    status = mcp_stdio.run_engine(request, threading.Event())

    plot = next(f for f in status["files"] if f["category"] == "plots")
    assert plot == {
        "path": str(out / "plots" / "QC.html"),
        "category": "plots",
        "size_bytes": 13,
        "format": "html",
        "title": "QC",
    }
    assert status["plots_truncated"] is True and status["plots_total"] == 7
    on_disk = json.loads(Path(status["result_path"]).read_text())
    listed = next(f for f in on_disk["files_written"] if f["category"] == "plots")
    assert listed["path"] == plot["path"] and listed["format"] == "html"
    assert listed["relative"] == "plots/QC.html" and listed["figure_ref"] == "fig@rev"
    assert "no_qc_plots" not in [w["code"] for w in status["warnings"]]
    assert on_disk["plots_total"] == 7


def test_a_view_that_cannot_be_read_does_not_fail_the_run(
    engine, tmp_path, data, monkeypatch
):
    _opened, script = engine
    script["events"] = [COMPLETE]

    def broken(self, since=None, **kwargs):
        raise StorageError("saved figures could not be listed")

    monkeypatch.setattr(FakeDataManager, "files_written_since", broken, raising=False)
    request = RunRequest(new_run_id(), str(data), str(tmp_path / "out"), "scrna_qc")
    status = mcp_stdio.run_engine(request, threading.Event())
    assert status["status"] == "completed"
    [warning] = [w for w in status["warnings"] if w["code"] == "metrics_unavailable"]
    assert "plots" in warning["message"]
