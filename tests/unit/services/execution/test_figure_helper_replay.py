"""Attempts that use the attempt-only figure helpers stay out of exports and replays.

The runtime setup of a code attempt injects the figure helpers (submit, load, list,
templates) and the attempt cache path ``EXEC_CACHE``; the exported setup defines none of
them, so an exported or replayed cell that uses one cannot run as recorded. Such an attempt
keeps its provenance but is neither exportable nor replayable, with a fixed reason. Code that
uses none of them keeps its flags. Parent-side only: no child process is launched.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import anndata
import numpy as np
import plotly.io as pio
import pytest

from lobster.core.figures import child_api
from lobster.core.notebooks.exporter import NotebookExporter
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.execution import custom_code_execution_service as svc

REF = "f_aaaaaaaaaa@r_bbbbbbbbbb"

DEPENDENT = {
    "submit": "import plotly.graph_objects as go\nsubmit_figure(go.Figure(), label='x')\n",
    "alias": "import plotly.graph_objects as go\npublish_plot(go.Figure(), 'x')\n",
    "edit": f"fig = load_figure('{REF}')\nsubmit_figure(fig, parent='{REF}')\n",
    "edit_config": f"cfg = load_figure_config('{REF}')\n",
    "list": "refs = [f['ref'] for f in list_figures()]\n",
    "source": "df = load_source('sha256-ab')\n",
    "template": "t = get_template('proteomics.volcano')\n",
    "template_list": "names = list_templates()\n",
    "house_style": "import plotly.graph_objects as go\napply_house_style(go.Figure())\n",
    "unavailable": "try:\n    x = 1\nexcept FigureUnavailable:\n    pass\n",
    "snapshot_path": "p = EXEC_CACHE / 'figures' / 'index.json'\n",
    "candidates_path": "p = EXEC_CACHE / 'candidates'\n",
    "as_value": "helper = submit_figure\n",
    "nested": "def make():\n    return list_figures()\n",
}

SELF_CONTAINED = {
    "arith": "result = 2 + 2\n",
    "plot_without_submit": "import plotly.graph_objects as go\nfig = go.Figure()\n",
    "comment_only": "# submit_figure is not called here\nresult = 1\n",
    "string_only": "note = 'call submit_figure later'\n",
    "attribute": "class A:\n    submit_figure = 1\n\nresult = A().submit_figure\n",
}


def _ir(tmp_path, code, persist=False):
    dm = MagicMock()
    dm.workspace_path = tmp_path
    return svc.CustomCodeExecutionService(dm)._create_ir(
        code=code,
        description="d",
        modality_name=None,
        load_workspace_files=False,
        workspace_keys=None,
        persist=persist,
        stats={"success": True, "duration_seconds": 0.0, "warnings": []},
    )


def _selected(ir, select_on):
    exporter = NotebookExporter.__new__(NotebookExporter)
    pairs = exporter._get_exportable_activity_ir_pairs(
        [{"type": "custom_code", "ir": ir.to_dict()}], select_on=select_on
    )
    return len(pairs) == 1


@pytest.mark.parametrize("persist", [False, True])
@pytest.mark.parametrize("case", sorted(DEPENDENT))
def test_a_helper_dependent_attempt_is_neither_exported_nor_replayed(
    tmp_path, case, persist
):
    ir = _ir(tmp_path, DEPENDENT[case], persist=persist)
    assert ir.exportable is False and ir.replayable is False
    assert ir.execution_context["replay_excluded"] == svc.REPLAY_EXCLUDED_FIGURE_HELPERS
    assert not _selected(ir, "exportable") and not _selected(ir, "replayable")
    assert ir.parameters["code"] == DEPENDENT[case]  # provenance keeps the code


@pytest.mark.parametrize("persist", [False, True])
@pytest.mark.parametrize("case", sorted(SELF_CONTAINED))
def test_self_contained_code_keeps_its_flags(tmp_path, case, persist):
    ir = _ir(tmp_path, SELF_CONTAINED[case], persist=persist)
    assert ir.exportable is persist and ir.replayable is True
    assert "replay_excluded" not in ir.execution_context
    assert _selected(ir, "replayable") and _selected(ir, "exportable") is persist


def test_the_names_cover_everything_the_child_setup_injects(tmp_path):
    default = pio.templates.default
    namespace = {}
    try:
        child_api.install(namespace, tmp_path)
    finally:
        pio.templates.default = default  # the install sets the house style
    public = {name for name in namespace if not name.startswith("_")}
    assert {"submit_figure", "publish_plot", "load_figure"} <= public
    assert public <= svc.FIGURE_HELPER_NAMES
    assert "EXEC_CACHE" in svc.FIGURE_HELPER_NAMES


def _execute(tmp_path, monkeypatch, code):
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.store_modality(name="m", adata=anndata.AnnData(np.ones((2, 2))), step="t")
    service = svc.CustomCodeExecutionService(dm)

    def fake_child(code, context):  # the child changed the data
        modified = context["exec_cache_path"] / ".modified_adata.h5ad"
        anndata.AnnData(np.zeros((2, 2))).write_h5ad(modified)
        context["completion"] = svc.COMPLETION_COMPLETE
        return None, "", "", None

    monkeypatch.setattr(service, "_execute_in_namespace", fake_child)
    _, stats, ir = service.execute(code, modality_name="m", load_workspace_files=False)
    return dm, stats, ir


def test_excluding_a_data_changing_attempt_says_the_replay_is_incomplete(
    tmp_path, monkeypatch
):
    dm, stats, ir = _execute(tmp_path, monkeypatch, DEPENDENT["list"])
    assert "m_custom" in dm.list_modalities() and ir.output_entities == ["m_custom"]
    assert ir.replayable is False
    assert ir.execution_context["replay_incomplete"] is True
    assert stats["replay_excluded"] == svc.REPLAY_EXCLUDED_FIGURE_HELPERS
    assert stats["replay_incomplete"] is True


def test_a_self_contained_data_changing_attempt_is_not_marked(tmp_path, monkeypatch):
    dm, stats, ir = _execute(tmp_path, monkeypatch, "x = 1\n")
    assert "m_custom" in dm.list_modalities() and ir.replayable is True
    assert "replay_incomplete" not in ir.execution_context
    assert stats["replay_excluded"] is None and stats["replay_incomplete"] is False
