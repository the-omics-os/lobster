"""Saved plots are visible to the agent, and the plot tool tells the truth.

Regression for the Cloud report "the agent couldn't confirm the plot file":
the HTML was written to ``<workspace>/plots`` but the workspace listing (and
therefore the handoff artifact manifest) never looked there, and the tool
returned no path. PNG export failures logged an apparently empty reason
because the exporter starts its message with a newline.

Deterministic; real PlotManager / DataManagerV2 / plotly, no model calls.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import plotly.graph_objects as go
import pytest

from lobster.agents.artifact_manifest import diff_snapshots, take_snapshot
from lobster.core.plot_manager import PlotManager, _one_line_reason
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.tools.plot_tools import create_plot_tools

KALEIDO_MESSAGE = (
    "\nImage export requires the Kaleido package, v1.0.0 or greater,\n"
    '    $ pip install --upgrade "kaleido>=1"\n'
)


@pytest.fixture
def no_png(monkeypatch):
    """Make PNG export fail the way an old kaleido does in the Cloud worker."""
    import plotly.io as pio

    def _fail(*_args, **_kwargs):
        raise ValueError(KALEIDO_MESSAGE)

    monkeypatch.setattr(pio, "write_image", _fail)


@pytest.fixture
def png_ok(monkeypatch):
    import plotly.io as pio

    def _write(_fig, path, *_args, **_kwargs):
        open(path, "wb").write(b"\x89PNG")

    monkeypatch.setattr(pio, "write_image", _write)


def _bar() -> go.Figure:
    return go.Figure(go.Bar(x=["A", "B", "C"], y=[3, 5, 2]))


def _tool(workspace, plot_manager):
    from lobster.core.figures.store import LocalFigureStore

    dm = SimpleNamespace(
        plot_manager=plot_manager, figure_store=LocalFigureStore(workspace)
    )
    tool = next(
        t
        for t in create_plot_tools(dm, workspace)
        if t.name == "create_interactive_plot"
    )
    return tool


def _invoke(tool, title="Simple Bar Chart", call_id="call_1"):
    args = {
        "plot_type": "bar",
        "data_json": json.dumps(
            [{"label": k, "value": v} for k, v in (("A", 3), ("B", 5), ("C", 2))]
        ),
        "x": "label",
        "y": "value",
        "title": title,
    }
    if call_id is None:
        return tool.invoke(args)
    msg = tool.invoke(
        {"args": args, "id": call_id, "name": tool.name, "type": "tool_call"}
    )
    return msg.content


# --- one-line reason -------------------------------------------------------


def test_one_line_reason_skips_leading_newline_and_trailing_comma():
    assert (
        _one_line_reason(ValueError(KALEIDO_MESSAGE))
        == "Image export requires the Kaleido package, v1.0.0 or greater"
    )


def test_one_line_reason_falls_back_to_type_name():
    assert _one_line_reason(RuntimeError("  \n ")) == "RuntimeError"


@pytest.mark.parametrize(
    "exc, expected",
    [
        (
            PermissionError(13, "Permission denied", "/abs/ws/p.html"),
            "Permission denied: p.html",
        ),
        (PermissionError(13, None, "/abs/ws/p.html"), "PermissionError: p.html"),
        (OSError(28, "No space left", b"/abs/ws/p.png"), "No space left: p.png"),
        (OSError(9, "Bad file descriptor", 7), "Bad file descriptor"),
        (OSError("\ndisk full"), "disk full"),
    ],
)
def test_one_line_reason_for_os_errors_never_shows_a_path(exc, expected):
    reason = _one_line_reason(exc)
    assert reason == expected
    assert "/abs/" not in reason


@pytest.mark.parametrize(
    "message, expected",
    [
        (
            "Failed to write /Users/someone/ws/plots/plot_1.html: disk full",
            "Failed to write plot_1.html: disk full",
        ),
        ("C:\\Users\\someone\\ws\\p.png not writable", "p.png not writable"),
        ("path '/var/folders/ab/T/tmpx/file.json' bad", "path 'file.json' bad"),
        (
            "\nChrome is required at /opt/google/chrome/chrome",
            "Chrome is required at chrome",
        ),
        ("path:/home/someone/ws/plot.png bad", "path:plot.png bad"),
        ("\\\\server\\share\\dir\\p.png gone", "p.png gone"),
        ("/abs/dir/ trailing", "dir trailing"),
        ("cannot read file:///Users/someone/ws/p.png", "cannot read file:p.png"),
        ("//server/share/dir/p.png gone", "p.png gone"),
    ],
)
def test_one_line_reason_strips_directories_from_any_message(message, expected):
    assert _one_line_reason(ValueError(message)) == expected


@pytest.mark.parametrize(
    "message",
    [
        "see https://plotly.com/python/static-image-export/ for help",
        "saved under plots/plot_1.html",
        "and / or 1 / 2",
        "1 /2",
        "~/plots/a.html",
        "see https://example.org/a/b?next=/c/d for help",
        "/srv",
    ],
)
def test_one_line_reason_keeps_urls_relative_paths_and_slashes(message):
    assert _one_line_reason(RuntimeError(message)) == message


def test_one_line_reason_os_error_without_filename_strips_directories():
    assert _one_line_reason(OSError("cannot open /tmp/x/y.png")) == "cannot open y.png"


def test_one_line_reason_is_bounded():
    assert len(_one_line_reason(ValueError("x" * 1000))) == 200


# --- PlotManager per-plot outcome -----------------------------------------


def test_tool_commits_through_trusted_save_and_names_the_revision(tmp_path):
    pm = PlotManager(workspace_path=tmp_path)
    out = _invoke(_tool(tmp_path, pm))
    assert out.startswith("Plot: Simple Bar Chart | 3 rows | saved: f_")
    ref = out.split("saved: ", 1)[1].split(" → ")[0]
    path = out.split(" → ", 1)[1]
    assert (tmp_path / path).is_file()
    assert not list((tmp_path / "plots").glob("*.html"))  # no flat HTML copies anymore
    from lobster.core.figures.store import LocalFigureStore

    execution_id = LocalFigureStore(tmp_path).read_record(ref)["execution"][
        "execution_id"
    ]
    assert (
        LocalFigureStore(tmp_path).read_execution(execution_id)["activity_unavailable"]
        == "provenance_disabled"
    )


def test_tool_without_tool_call_context_saves_nothing(tmp_path):
    pm = PlotManager(workspace_path=tmp_path)
    with pytest.raises(
        ValueError, match="ToolCall"
    ):  # LangChain refuses calls without a tool call id
        _invoke(_tool(tmp_path, pm), call_id=None)
    assert not (tmp_path / "plots").exists()
    plot_id, _, _ = pm.add_plot(plot=_bar(), title="direct")
    from lobster.core.figures.errors import AttemptContextError

    with pytest.raises(
        AttemptContextError
    ):  # direct core call without context also refuses
        pm.commit_staged(None, tool_call_id="", staged_ids=[plot_id])
    assert not (tmp_path / "plots").exists()


def test_tool_reports_the_current_plot_not_an_earlier_one(tmp_path):
    pm = PlotManager(workspace_path=tmp_path)
    tool = _tool(tmp_path, pm)
    first = _invoke(tool, title="First", call_id="call_a")
    second = _invoke(tool, title="Second", call_id="call_b")
    ref1 = first.split("saved: ", 1)[1].split(" → ")[0]
    ref2 = second.split("saved: ", 1)[1].split(" → ")[0]
    assert ref1 != ref2 and ref1 not in second


# --- workspace listing + artifact manifest ----------------------------------


def test_list_workspace_files_has_empty_plots_before_first_save(tmp_path):
    dm = DataManagerV2(workspace_path=tmp_path)
    files = dm.list_workspace_files()
    assert files["plots"] == []
    assert {"data", "exports", "cache"} <= set(files)


def test_list_workspace_files_lists_managed_and_legacy_plots(tmp_path):
    dm = DataManagerV2(workspace_path=tmp_path)
    plot_id, _, _ = dm.plot_manager.add_plot(plot=_bar(), title="Bars", source="test")
    [res] = dm.plot_manager.commit_staged(
        {"id": "a1"}, tool_call_id="call_1", staged_ids=[plot_id]
    )
    (tmp_path / "plots" / "plot_1_Old.html").write_text("<html></html>")

    plots = dm.list_workspace_files()["plots"]

    managed = [p for p in plots if not p["legacy"]]
    legacy = [p for p in plots if p["legacy"]]
    assert [p["figure_ref"] for p in managed] == [res.ref]
    assert managed[0]["label"] == "Bars" and managed[0]["size"] > 0
    assert [p["name"] for p in legacy] == ["plot_1_Old.html"]


def test_artifact_manifest_reports_a_saved_plot(tmp_path):
    dm = DataManagerV2(workspace_path=tmp_path)
    before = take_snapshot(dm)

    plot_id, _, _ = dm.plot_manager.add_plot(plot=_bar(), title="Bars", source="test")
    [res] = dm.plot_manager.commit_staged(
        {"id": "a1"}, tool_call_id="call_1", staged_ids=[plot_id]
    )
    diff = diff_snapshots(before, take_snapshot(dm))

    assert not before.failed
    # contract §8: a saved revision is reported by ref, not as its file path
    assert diff.figures_saved == [f"Bars ({res.ref})"]
    assert diff.files_written == []


def test_tool_failure_reply_carries_codes_not_exception_text(tmp_path, monkeypatch):
    from lobster.core.figures.store import LocalFigureStore

    def boom(*_a, **_k):
        raise OSError(f"CANARY {tmp_path}/secret.csv")

    monkeypatch.setattr(LocalFigureStore, "commit", boom)
    pm = PlotManager(workspace_path=tmp_path)
    out = _invoke(_tool(tmp_path, pm))
    assert out == "Plot not saved: Simple Bar Chart | failed: storage:unexpected_error"
    assert "CANARY" not in out and str(tmp_path) not in out
    assert pm.latest_plots == []


# --- pending publication is "confirmation pending", never a definitive absence ----------


def _pending_reason():
    from lobster.core.figures.errors import DurabilityError, safe_reason

    return safe_reason(DurabilityError("fsync unconfirmed"))


def _returned_pending(monkeypatch):
    """The store's own uncertain-acknowledgement result, returned through the saver."""
    from lobster.core.figures.errors import DurabilityError
    from lobster.core.figures.store import LocalFigureStore

    def pending(self, candidate, **_kw):
        return self._pending(candidate, DurabilityError("fsync unconfirmed"))

    monkeypatch.setattr(LocalFigureStore, "commit", pending)


def _raise_from_commit(monkeypatch, results):
    from lobster.core.figures.errors import FigureError

    def raised(self, *_a, **_k):
        raise FigureError(
            "CANARY raised detail",
            details={"reason": "no_figure_available", "results": results},
        )

    monkeypatch.setattr(PlotManager, "commit_staged", raised)


FORGED_REF = "f_forgedaaaa@r_forgedbbbb"


def test_a_returned_pending_save_says_confirmation_pending(tmp_path, monkeypatch):
    from lobster.core.figures.saver import HEADER_PENDING

    _returned_pending(monkeypatch)
    out = _invoke(_tool(tmp_path, PlotManager(workspace_path=tmp_path)))
    assert out.startswith("Plot: Simple Bar Chart | 3 rows | ")
    assert HEADER_PENDING in out
    assert f"confirmation pending: {_pending_reason()}" in out
    assert "not saved" not in out.lower() and "No figure saved" not in out


def test_a_raised_pending_says_confirmation_pending_without_raised_refs(
    tmp_path, monkeypatch
):
    from lobster.core.figures.saver import HEADER_PENDING

    _raise_from_commit(
        monkeypatch,
        [
            {
                "candidate_index": 0,
                "status": "pending",
                "reason": _pending_reason(),
                "ref": FORGED_REF,
                "path": "plots/forged/figure.json",
            }
        ],
    )
    pm = PlotManager(workspace_path=tmp_path)
    out = _invoke(_tool(tmp_path, pm))
    assert HEADER_PENDING in out
    assert f"pending: {_pending_reason()}" in out
    assert not out.startswith("Plot not saved") and "not saved" not in out.lower()
    assert FORGED_REF not in out and "forged" not in out and "CANARY" not in out
    assert pm.latest_plots == []


def test_a_raised_mixed_pending_and_failed_is_pending_not_absent(tmp_path, monkeypatch):
    from lobster.core.figures.saver import HEADER_PENDING

    _raise_from_commit(
        monkeypatch,
        [
            {"status": "pending", "reason": _pending_reason(), "ref": FORGED_REF},
            {"status": "failed", "reason": "storage:unexpected_error"},
        ],
    )
    out = _invoke(_tool(tmp_path, PlotManager(workspace_path=tmp_path)))
    assert HEADER_PENDING in out and not out.startswith("Plot not saved")
    assert "failed: storage:unexpected_error" in out
    assert FORGED_REF not in out


def test_a_raised_failure_without_pending_stays_not_saved(tmp_path, monkeypatch):
    from lobster.core.figures.saver import HEADER_PENDING

    _raise_from_commit(
        monkeypatch,
        [{"status": "failed", "reason": "storage:unexpected_error", "ref": FORGED_REF}],
    )
    out = _invoke(_tool(tmp_path, PlotManager(workspace_path=tmp_path)))
    assert out == "Plot not saved: Simple Bar Chart | failed: storage:unexpected_error"
    assert HEADER_PENDING not in out


def test_a_raised_available_status_never_authenticates_a_save(tmp_path, monkeypatch):
    _raise_from_commit(
        monkeypatch,
        [{"status": "available", "reason": None, "ref": FORGED_REF, "path": "x"}],
    )
    out = _invoke(_tool(tmp_path, PlotManager(workspace_path=tmp_path)))
    assert out == "Plot not saved: Simple Bar Chart | unconfirmed: reason_withheld"
    assert "available" not in out
    assert FORGED_REF not in out and "saved: f_" not in out


def test_a_raised_pending_with_a_forged_available_never_shows_available(
    tmp_path, monkeypatch
):
    from lobster.core.figures.saver import HEADER_PENDING

    _raise_from_commit(
        monkeypatch,
        [
            {"status": "pending", "reason": _pending_reason()},
            {"status": "available", "reason": "storage_failed", "ref": FORGED_REF},
        ],
    )
    out = _invoke(_tool(tmp_path, PlotManager(workspace_path=tmp_path)))
    assert out == (
        f"Plot: Simple Bar Chart | {HEADER_PENDING} | pending: {_pending_reason()}; "
        "unconfirmed: reason_withheld"
    )
    assert "available" not in out and FORGED_REF not in out


def test_a_returned_unknown_reason_is_withheld(tmp_path, monkeypatch):
    from lobster.core.figures.records import CommitResult

    def returned(self, *_a, **_k):
        return [CommitResult(0, "pending", reason=f"CANARY {tmp_path}")]

    monkeypatch.setattr(PlotManager, "commit_staged", returned)
    out = _invoke(_tool(tmp_path, PlotManager(workspace_path=tmp_path)))
    assert "confirmation pending: reason_withheld" in out
    assert "CANARY" not in out and str(tmp_path) not in out
