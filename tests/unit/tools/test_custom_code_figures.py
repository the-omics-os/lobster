"""execute_custom_code + managed figures, end to end (contract §6.1–§6.4).

Real DataManagerV2, real CustomCodeExecutionService (real child subprocess), real store.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from lobster.core.figures.ids import FigureRef
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.execution.custom_code_execution_service import (
    CustomCodeExecutionService,
)
from lobster.tools.custom_code_tool import create_execute_custom_code_tool

FIG = "import plotly.graph_objects as go\n"


@pytest.fixture()
def env(tmp_path):
    dm = DataManagerV2(workspace_path=tmp_path)
    tool = create_execute_custom_code_tool(
        dm, CustomCodeExecutionService(dm), agent_name="proteomics_expert"
    )
    return dm, tool


def _run(tool, code, call_id="call_1", **args):
    async def go():
        msg = await tool.ainvoke(
            {
                "args": {"python_code": code, "load_workspace_files": False, **args},
                "id": call_id,
                "name": tool.name,
                "type": "tool_call",
            }
        )
        return json.loads(msg.content)

    return asyncio.run(go())


def test_only_submitted_figures_are_saved_and_unsubmitted_are_named(env):
    dm, tool = env
    code = (
        FIG
        + "fig = go.Figure(go.Bar(x=['a'], y=[1]))\n"
        + "fig_qc = go.Figure(go.Scatter(x=[1], y=[2]))\n"
        + "submit_figure(fig, intent='counts per group', label='Counts')\n"
        + "result = 'ok'\n"
    )
    out = _run(tool, code)

    assert out["success"] is True
    assert out["figure_outcome"] == "available" and len(out["figure_refs"]) == 1
    assert out["unsubmitted_figures"] == (
        "Not saved: fig_qc. Call submit_figure(fig, source=…, intent=…) to save a figure."
    )
    [rec] = dm.figure_store.list().items
    assert rec["label"] == "Counts" and rec["change"]["request"] == "counts per group"
    assert rec["change"]["actor"] == "expert:proteomics_expert"
    # no automatic capture, no child writes into plots/
    assert not list((dm.workspace_path / "plots").glob("auto_*"))
    assert "captured_plots" not in out


def test_execution_record_links_the_logged_activity_and_exact_code(env):
    dm, tool = env
    code = FIG + "submit_figure(go.Figure(go.Bar(x=['a'], y=[1])))\n"
    out = _run(tool, code)
    record = dm.figure_store.read_record(out["figure_refs"][0])
    execution = dm.figure_store.read_execution(record["execution"]["execution_id"])
    code_path = dm.figure_store.root / "_code" / f"{execution['code_id']}.py"
    assert code_path.read_bytes() == code.encode("utf-8")
    activity = json.loads(
        (dm.workspace_path / execution["activity_record"]["path"]).read_text()
    )
    assert activity["id"] == execution["activity_id"]
    assert (
        activity["type"] == "execute_custom_code"
        and activity["agent"] == "proteomics_expert"
    )
    assert execution["tool"] == "execute_custom_code" and execution["persist"] is False


def test_failed_code_saves_nothing_even_after_submit(env):
    dm, tool = env
    code = (
        FIG
        + "submit_figure(go.Figure(go.Bar(x=['a'], y=[1])))\nraise RuntimeError('boom')\n"
    )
    out = _run(tool, code)
    assert out["success"] is False
    assert out["figures"].startswith("No figure saved: the code failed")
    assert dm.figure_store.list().items == []


def test_zero_submits_with_figures_says_no_figure_saved(env):
    _, tool = env
    out = _run(tool, FIG + "fig = go.Figure()\n")
    assert out["figures"] == "No figure saved."
    assert "fig" in out["unsubmitted_figures"]


def test_publish_plot_alias_and_same_call_retry_is_idempotent(env):
    dm, tool = env
    code = FIG + "publish_plot(go.Figure(go.Bar(x=['a'], y=[1])), 'Alias')\n"
    first = _run(tool, code, call_id="call_r")
    again = _run(tool, code, call_id="call_r")
    assert first["figure_refs"] == again["figure_refs"]
    assert len(dm.figure_store.list().items) == 1
    assert dm.figure_store.list().items[0]["label"] == "Alias"


def test_child_never_imports_lobster_core_or_config(env):
    """The child loads the helpers by path; only the PEP 420 namespace (and the optional
    templates package) may appear — never lobster.core / lobster.config (dotenv) code.
    """
    _, tool = env
    out = _run(
        tool,
        "import sys\nresult = sorted(m for m in sys.modules if m == 'lobster' or m.startswith('lobster.'))\n",
    )
    assert out["success"] is True
    allowed = {"lobster"}
    assert [
        m
        for m in out["result"]
        if m not in allowed and not m.startswith("lobster.figure_templates")
    ] == []


def test_snapshot_literal_refs_heads_and_edits(env):
    dm, tool = env
    first = _run(
        tool, FIG + "submit_figure(go.Figure(go.Bar(x=['a'], y=[1])), label='v1')\n"
    )
    r1 = first["figure_refs"][0]
    # edit R1 → R2 (same figure, kind edited); R1 becomes a non-head
    edit = (
        FIG
        + f"base = load_figure('{r1}')\n"
        + "base.update_layout(title='edited')\n"
        + f"submit_figure(base, parent='{r1}', label='v2')\n"
    )
    second = _run(tool, edit, call_id="call_2")
    r2 = second["figure_refs"][0]
    assert FigureRef.parse(r2).figure_id == FigureRef.parse(r1).figure_id
    assert dm.figure_store.read_record(r2)["change"]["kind"] == "edited"
    # a dynamically built non-head ref is not_requested (never "missing"); heads always load
    fid, rid = r1.split("@")
    dynamic = (
        f"parts = [{fid!r}, {rid!r}]\n"
        + "try:\n    load_figure('@'.join(parts))\n    dyn = 'loaded'\nexcept FigureUnavailable as e:\n    dyn = e.reason\n"
        + f"head = type(load_figure({r2!r})).__name__\n"
        + "result = {'dyn': dyn, 'head': head, 'n': len(list_figures())}\n"
    )
    out = _run(tool, dynamic, call_id="call_3")
    assert out["result"] == {"dyn": "not_requested", "head": "Figure", "n": 2}
    # writing the non-head ref literally materializes it at the next attempt
    out = _run(tool, f"result = type(load_figure({r1!r})).__name__\n", call_id="call_4")
    assert out["result"] == "Figure"


def test_revision_created_after_attempt_start_is_seen_by_the_next_attempt(env):
    dm, tool = env
    probe = (
        "try:\n    load_figure(REF)\n    seen = 'loaded'\nexcept FigureUnavailable as e:\n    seen = e.status\n"
        "result = seen\n"
    )
    plot_id, _, _ = dm.plot_manager.add_plot(
        plot=__import__("plotly.graph_objects", fromlist=["Figure"]).Figure(),
        title="api",
    )
    missing = "f_zzzzzzzzzz@r_zzzzzzzzzz"
    out = _run(tool, f"REF = '{missing}'\n" + probe, call_id="call_a")
    assert out["result"] == "not_in_snapshot"
    [res] = dm.plot_manager.commit_staged(
        {"id": "x"}, tool_call_id="api_1", staged_ids=[plot_id]
    )
    out2 = _run(tool, f"REF = '{res.ref}'\n" + probe, call_id="call_b")
    assert out2["result"] == "loaded"


def test_materialization_cache_is_keyed_by_instance(env):
    import anndata
    import numpy as np

    dm, tool = env
    dm.store_modality(name="m", adata=anndata.AnnData(np.ones((2, 2))), step="t")
    out1 = _run(
        tool, "result = float(adata.X.sum())\n", call_id="c1", modality_name="m"
    )
    dm.store_modality(name="m", adata=anndata.AnnData(np.full((2, 2), 3.0)), step="t")
    out2 = _run(
        tool, "result = float(adata.X.sum())\n", call_id="c2", modality_name="m"
    )
    assert (out1["result"], out2["result"]) == (4.0, 12.0)  # stale cache never served


@pytest.mark.parametrize(
    "tail, state",
    [
        ("import sys\nsys.exit(0)\n", "completion_unconfirmed"),
        ("import os, sys\nsys.stdout.flush()\nos._exit(0)\n", "completion_unconfirmed"),
    ],
)
def test_exit_zero_without_normal_completion_saves_nothing(env, tail, state):
    dm, tool = env
    code = FIG + "submit_figure(go.Figure(go.Bar(x=['a'], y=[1])))\n" + tail
    out = _run(tool, code, call_id=f"call_{state}_{len(tail)}")
    assert out["figures"] == (
        "No figure saved: the run did not finish normally, so 1 submitted "
        f"figure(s) were discarded ({state})."
    )
    assert "figure_refs" not in out
    assert dm.figure_store.list().items == []


def _tamper_result_after_child_exit(monkeypatch, mutate):
    """Real child runs to normal completion; then the on-disk result envelope is changed
    before the parent reads it (the subprocess boundary is the only seam)."""
    from pathlib import Path

    from lobster.services.execution import custom_code_execution_service as svc

    real_run = svc.subprocess.run

    def run(argv, *a, **k):
        proc = real_run(argv, *a, **k)
        mutate(Path(argv[1]).parent / ".execution_result.json")
        return proc

    monkeypatch.setattr(svc.subprocess, "run", run)


@pytest.mark.parametrize(
    "mutate, state",
    [
        (lambda p: p.unlink(), "result_missing"),
        (
            lambda p: p.write_bytes(b'{"schema": "lobster.execution_res'),
            "result_unreadable",
        ),
        (lambda p: p.write_bytes(b"\xff\xfe not json"), "result_unreadable"),
        (lambda p: p.write_bytes(b"[]"), "result_unreadable"),
        (lambda p: p.write_bytes(b'{"result": 1, "type": "int"}'), "result_unreadable"),
        (
            lambda p: p.write_bytes(
                b'{"schema": "lobster.execution_result/1", "has_result": "yes",'
                b' "result": 1, "type": "int"}'
            ),
            "result_unreadable",
        ),
        (
            lambda p: p.write_bytes(
                b'{"schema": "lobster.execution_result/1", "has_result": false,'
                b' "result": 5, "type": "int"}'
            ),
            "result_unreadable",
        ),
    ],
)
def test_missing_or_bad_result_envelope_after_normal_exit_saves_nothing(
    env, monkeypatch, mutate, state
):
    dm, tool = env
    _tamper_result_after_child_exit(monkeypatch, mutate)
    code = FIG + "submit_figure(go.Figure(go.Bar(x=['a'], y=[1])))\nresult = 1\n"
    out = _run(tool, code, call_id=f"call_bad_{state}")
    assert out["figures"] == (
        "No figure saved: the run did not finish normally, so 1 submitted "
        f"figure(s) were discarded ({state})."
    )
    assert dm.figure_store.list().items == []


def test_plot_only_run_without_result_still_saves(env):
    dm, tool = env
    out = _run(tool, FIG + "submit_figure(go.Figure(go.Bar(x=['a'], y=[1])))\n")
    assert out["success"] is True and out["figure_outcome"] == "available"
    assert len(dm.figure_store.list().items) == 1


def test_unknown_completion_state_is_never_echoed():
    from lobster.tools.custom_code_tool import _commit_submitted_figures

    report = _commit_submitted_figures(
        None,
        [],
        {"completion": "synthetic_sensitive_marker_123", "figures_submitted": 2},
        activity=None,
        tool_call_id="c",
        agent_name="a",
        code="",
        modality_name=None,
        workspace_key=None,
        persist=False,
    )
    assert "synthetic_sensitive_marker_123" not in str(report)
    assert report["figures"].endswith("(completion_unconfirmed).")


def test_marker_nonce_in_user_code_is_not_replaced(env):
    _, tool = env
    out = _run(tool, "result = '__LOBSTER_RUN_NONCE__'\n", call_id="call_nonce")
    assert out["success"] is True and out["result"] == "__LOBSTER_RUN_NONCE__"


@pytest.mark.parametrize(
    "envelope",
    [
        b'{"schema": "lobster.execution_result/1", "has_result": true, "result": 1, "type": 5}',
        b'{"schema": "lobster.execution_result/1", "has_result": true, "result": 1, "type": ""}',
        b'{"schema": "lobster.execution_result/1", "has_result": false, "result": null, "type": "int"}',
    ],
)
def test_result_envelope_type_is_validated(env, monkeypatch, envelope):
    dm, tool = env
    _tamper_result_after_child_exit(monkeypatch, lambda p: p.write_bytes(envelope))
    code = FIG + "submit_figure(go.Figure(go.Bar(x=['a'], y=[1])))\nresult = 1\n"
    out = _run(tool, code, call_id="call_type")
    assert out["figures"].endswith("(result_unreadable).")
    assert dm.figure_store.list().items == []


@pytest.mark.parametrize("completion", [["x"], {"a": 1}, 7, None, b"complete"])
def test_non_string_or_unhashable_completion_never_throws(completion):
    from lobster.tools.custom_code_tool import _commit_submitted_figures

    report = _commit_submitted_figures(
        None,
        [],
        {"completion": completion, "figures_submitted": [1]},
        activity=None,
        tool_call_id="c",
        agent_name="a",
        code="",
        modality_name=None,
        workspace_key=None,
        persist=False,
    )
    assert (
        report["figures"]
        == (
            "No figure saved: the run did not finish normally, so 0 submitted figure(s) "
            "were discarded (completion_unconfirmed)."
        )
        or report == {}
    )


# --- freshness (a): an unknown identity is NEVER reused by name ------------------

_READ_SUM = "result = float(adata.X.sum())\n"


def test_unknown_identity_never_reuses_a_name_only_materialization(env):
    import anndata
    import numpy as np

    dm, tool = env
    dm.modalities["m"] = anndata.AnnData(
        np.ones((2, 2))
    )  # no lineage: identity unknown
    assert dm.resolve_input_identity("m") is None
    assert _run(tool, _READ_SUM, call_id="c1", modality_name="m")["result"] == 4.0
    dm.modalities["m"] = anndata.AnnData(np.full((2, 2), 3.0))  # still unknown
    assert _run(tool, _READ_SUM, call_id="c2", modality_name="m")["result"] == 12.0


def test_a_new_store_event_is_read_and_nothing_persists_across_attempts(env):
    import anndata
    import numpy as np

    dm, tool = env
    dm.store_modality(name="m", adata=anndata.AnnData(np.ones((2, 2))), step="t")
    assert _run(tool, _READ_SUM, call_id="c1", modality_name="m")["result"] == 4.0
    assert _run(tool, _READ_SUM, call_id="c2", modality_name="m")["result"] == 4.0
    dm.store_modality(name="m", adata=anndata.AnnData(np.full((2, 2), 3.0)), step="t")
    assert _run(tool, _READ_SUM, call_id="c3", modality_name="m")["result"] == 12.0
    execution = dm.workspace_path / "cache" / "execution"
    # per-attempt inputs are removed with the attempt; no session-wide copy is written
    assert not list(execution.rglob("*.h5ad"))
    assert not (execution / "materialized").exists()


def test_inputs_are_materialized_once_per_attempt_and_costs_stay_in_delivery_stats(env):
    import anndata
    import numpy as np

    dm, _ = env
    dm.store_modality(name="m", adata=anndata.AnnData(np.ones((2, 2))), step="t")
    service = CustomCodeExecutionService(dm)
    # referenced twice (modality_name + a literal path): one materialization in this attempt
    code = (
        "import anndata\nb = anndata.read_h5ad('m.h5ad')\nresult = float(b.X.sum())\n"
    )
    result, stats, _ = service.execute(
        code, modality_name="m", load_workspace_files=False
    )
    assert result == 4.0
    [entry] = stats["materialized_inputs"]
    assert entry["name"] == "m" and entry["bytes"] > 0 and entry["seconds"] >= 0


def test_a_failed_materialization_leaves_no_attempt_directory(env, monkeypatch):
    import anndata
    import numpy as np

    from lobster.services.execution.custom_code_execution_service import (
        CodeExecutionError,
    )

    dm, _ = env
    dm.store_modality(name="m", adata=anndata.AnnData(np.ones((2, 2))), step="t")
    service = CustomCodeExecutionService(dm)

    def boom(name, materialized_dir, *_):
        (materialized_dir / "partial.h5ad").write_bytes(b"x")
        raise RuntimeError("disk full")

    monkeypatch.setattr(service, "_materialize_modality", boom)
    with pytest.raises(CodeExecutionError):
        service.execute("result = 1\n", modality_name="m", load_workspace_files=False)
    execution = dm.workspace_path / "cache" / "execution"
    assert not execution.exists() or not any(execution.iterdir())


# --- Journey: create -> save -> fresh reload (worker replacement) -> child edit ----


def _tree_hashes(root):
    import hashlib

    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_journey_create_save_fresh_reload_child_edit_keeps_the_original(tmp_path):
    """Real disk. The original revision's bytes never change; the edit is a new revision."""
    import gc

    import anndata
    import numpy as np

    from lobster.core.figures.renditions import ensure_html_rendition
    from lobster.core.figures.store import LocalFigureStore

    # 1. an expert creates and saves a figure from a stored modality (with source data)
    dm1 = DataManagerV2(workspace_path=tmp_path)
    dm1.store_modality(name="m", adata=anndata.AnnData(np.ones((2, 2))), step="t")
    tool1 = create_execute_custom_code_tool(
        dm1, CustomCodeExecutionService(dm1), agent_name="proteomics_expert"
    )
    created = _run(
        tool1,
        FIG
        + "fig = go.Figure(go.Bar(x=['a', 'b'], y=[float(adata.X.sum()), 1.0]))\n"
        + "submit_figure(fig, source=[{'x': 'a', 'y': 4.0}, {'x': 'b', 'y': 1.0}],"
        + " intent='totals', label='Totals')\n",
        call_id="c1",
        modality_name="m",
    )
    assert created["figure_outcome"] == "available"
    [r1] = created["figure_refs"]
    r1_dir = tmp_path / "plots" / r1.replace("@", "/")
    before = _tree_hashes(r1_dir)
    exact = dm1.figure_store.read(r1)
    edges = dm1.figure_store.read_record(r1)["inputs"]
    assert [e["name"] for e in edges] == ["m"]

    # 2. worker replacement: nothing survives in memory
    del tool1, dm1
    gc.collect()
    dm2 = DataManagerV2(workspace_path=tmp_path)
    tool2 = create_execute_custom_code_tool(
        dm2, CustomCodeExecutionService(dm2), agent_name="proteomics_expert"
    )
    [item] = dm2.figure_store.list().items
    assert (f"{item['figure_id']}@{item['revision_id']}", item["availability"]) == (
        r1,
        "available",
    )

    # 3. a child edit after the reload: copy first, original preserved
    edited = _run(
        tool2,
        FIG
        + f"base = load_figure({r1!r})\n"
        + "base.update_traces(marker_color='orange')\n"
        + f"submit_figure(base, parent={r1!r}, label='Totals (orange)')\n",
        call_id="c2",
    )
    assert edited["figure_outcome"] == "available"
    [r2] = edited["figure_refs"]
    assert r2 != r1 and r2.split("@")[0] == r1.split("@")[0]

    # 4. both surfaces read the same truth from a third, fresh store; views are appended only
    fresh = LocalFigureStore(tmp_path, session_id=str(dm2.session_id))
    assert fresh.read(r1) == exact  # byte-identical original
    source = fresh.read_record(r1)["source"]
    assert source["source_id"] and fresh.read_source(
        source["source_id"]
    )  # sha-verified
    rec2 = fresh.read_record(r2)
    assert rec2["change"]["kind"] == "edited"
    assert rec2["parents"] == [r1.split("@")[1]]
    assert rec2["inputs"] == edges  # inherited from the parent
    assert fresh.read(r2)[1] != exact[1]
    for ref in (r1, r2):
        assert ensure_html_rendition(fresh, ref).status == "available"
    after = _tree_hashes(r1_dir)
    assert {k: v for k, v in after.items() if not k.startswith("renditions/")} == before


# --- freshness: every attempt reads the CURRENT data ---------------------------
#
# Were strict FRESHNESS_TARGET xfails; fixed by per-attempt materialization.


def _lineaged(dm, value):
    import anndata
    import numpy as np

    dm.store_modality(name="m", adata=anndata.AnnData(np.full((2, 2), value)), step="t")
    return dm.resolve_input_identity("m").instance_id


def test_in_place_change_then_reassignment_is_read_fresh(env):
    dm, tool = env
    lineage = _lineaged(dm, 1.0)
    assert _run(tool, _READ_SUM, call_id="c1", modality_name="m")["result"] == 4.0
    ad = dm.modalities["m"]
    ad.X[:] = 3.0
    dm.modalities["m"] = ad
    assert dm.resolve_input_identity("m").instance_id == lineage  # unchanged id
    assert _run(tool, _READ_SUM, call_id="c2", modality_name="m")["result"] == 12.0


def test_in_place_change_without_reassignment_is_read_fresh(env):
    dm, tool = env
    _lineaged(dm, 1.0)
    assert _run(tool, _READ_SUM, call_id="c1", modality_name="m")["result"] == 4.0
    dm.modalities["m"].X[:] = 3.0
    assert _run(tool, _READ_SUM, call_id="c2", modality_name="m")["result"] == 12.0


def test_a_rewritten_backed_file_with_the_same_lineage_is_read_fresh(env):
    import anndata
    import numpy as np

    dm, tool = env
    path = dm.data_dir / "m.h5ad"
    path.parent.mkdir(parents=True, exist_ok=True)

    def write(value):
        ad = anndata.AnnData(np.full((2, 2), value))
        ad.uns["lineage"] = {"instance_id": "same_lineage"}
        ad.write_h5ad(path)
        backed = anndata.read_h5ad(path, backed="r")
        dm.modalities["m"] = backed
        return backed

    first = write(1.0)
    assert dm.resolve_input_identity("m").instance_id == "same_lineage"
    assert _run(tool, _READ_SUM, call_id="c1", modality_name="m")["result"] == 4.0
    first.file.close()
    write(3.0)  # the mutable local file changed; its lineage did not
    assert _run(tool, _READ_SUM, call_id="c2", modality_name="m")["result"] == 12.0


# --- config on edit: an omitted config keeps the parent's EXACT config ------------

# Was a strict CONFIG_TARGET xfail; fixed in the child/tool layer.


def _parent_with_config(dm, config_bytes):
    import hashlib

    from lobster.core.figures.records import Candidate, ExecutionInfo
    from lobster.core.figures.saver import TrustedSaver

    seed = (
        "s:seed_" + hashlib.sha256(config_bytes).hexdigest()[:12]
    )  # one attempt per seed
    info = ExecutionInfo(
        seed, "seed", None, {}, False, None, None, "1.2.0.dev0", "7.1.0"
    )
    cand = Candidate(
        f"{seed}:0",
        0,
        b'{"data":[{"type":"bar","x":["a"],"y":[1]}],"layout":{}}',
        config_bytes,
    )
    [res] = TrustedSaver(dm.figure_store).commit_attempt(info, [cand], succeeded=True)
    assert res.status == "available"
    return res.ref


def test_title_only_child_edit_keeps_the_parent_config_exactly(env):
    dm, tool = env
    parent_config = b'{"displaylogo":false,"scrollZoom":true}'
    r1 = _parent_with_config(dm, parent_config)
    out = _run(
        tool,
        FIG
        + f"base = load_figure({r1!r})\n"
        + "base.update_layout(title='renamed')\n"
        + f"submit_figure(base, parent={r1!r})\n",
        call_id="c_edit",
    )
    [r2] = out["figure_refs"]
    assert dm.figure_store.read(r2)[2] == parent_config
    assert dm.figure_store.read(r1)[2] == parent_config  # original untouched


def _edit_code(r1, config_expr=None, extra=""):
    cfg = "" if config_expr is None else f", config={config_expr}"
    return (
        FIG
        + f"base = load_figure({r1!r})\n"
        + "base.update_layout(title='renamed')\n"
        + extra
        + f"submit_figure(base, parent={r1!r}{cfg})\n"
    )


@pytest.mark.parametrize(
    "parent_config",
    [
        b'{ "displaylogo" : false,\n  "scrollZoom": true }\n',
        b"{}",
        b'{"responsive":true}',
    ],
)
def test_omitted_or_none_config_keeps_the_exact_parent_bytes(env, parent_config):
    dm, tool = env
    r1 = _parent_with_config(dm, parent_config)
    for call, expr in (("c_omit", None), ("c_none", "None")):
        [r2] = _run(tool, _edit_code(r1, expr), call_id=call)["figure_refs"]
        assert (
            dm.figure_store.read(r2)[2] == parent_config
        )  # byte-exact, whitespace kept
    assert dm.figure_store.read(r1)[2] == parent_config  # original unchanged


def test_explicit_empty_config_clears_and_an_explicit_config_is_saved_exactly(env):
    dm, tool = env
    r1 = _parent_with_config(dm, b'{"displaylogo":false}')
    [cleared] = _run(tool, _edit_code(r1, "{}"), call_id="c_clear")["figure_refs"]
    assert dm.figure_store.read(cleared)[2] == b"{}"
    code = _edit_code(
        r1,
        "cfg",
        extra=f"cfg = load_figure_config({r1!r})\ncfg['scrollZoom'] = True\n",
    )
    [edited] = _run(tool, code, call_id="c_cfg")["figure_refs"]
    assert dm.figure_store.read(edited)[2] == b'{"displaylogo":false,"scrollZoom":true}'
    assert dm.figure_store.read(r1)[2] == b'{"displaylogo":false}'


def test_a_new_figure_without_config_gets_empty_config(env):
    dm, tool = env
    out = _run(tool, FIG + "submit_figure(go.Figure(go.Bar(y=[1])))\n", call_id="c_new")
    assert dm.figure_store.read(out["figure_refs"][0])[2] == b"{}"


@pytest.mark.parametrize(
    "bad",
    [
        "{'x': float('nan')}",
        "{'x': {1, 2}}",
        "['not', 'a', 'dict']",
        "{'big': 'x' * (64 * 1024)}",
    ],
)
def test_an_invalid_config_raises_in_the_child_and_leaves_no_half_submission(env, bad):
    dm, tool = env
    code = (
        FIG
        + "try:\n"
        + f"    submit_figure(go.Figure(go.Bar(y=[1])), config={bad})\n"
        + "    first = 'accepted'\n"
        + "except (TypeError, ValueError) as e:\n"
        + "    first = 'refused'\n"
        + "idx = submit_figure(go.Figure(go.Bar(y=[2])), label='ok')\n"
        + "result = [first, idx]\n"
    )
    out = _run(tool, code, call_id="c_bad")
    assert out["result"] == ["refused", 0]  # counter not advanced by the refused call
    [ref] = out["figure_refs"]
    assert dm.figure_store.read_record(ref)["label"] == "ok"


def test_unreadable_parents_are_typed_refusals_never_empty_config(env, monkeypatch):
    from lobster.core.figures import store as store_mod
    from lobster.core.figures.errors import DurabilityError

    dm, tool = env
    good = _parent_with_config(dm, b'{"displaylogo":false}')
    tampered = _parent_with_config(dm, b'{"scrollZoom":true}')
    pending = _parent_with_config(dm, b'{"responsive":true}')
    missing = "f_missing000@r_missing000"
    cfg = dm.figure_store._rev_dir(store_mod.FigureRef.parse(tampered)) / "config.json"
    cfg.write_bytes(b'{"scrollZoom":false}')
    real_read = type(dm.figure_store).read

    def read(self, ref):
        if str(ref) == pending:
            raise DurabilityError("pending", details={"reason": "dir_fsync_failed"})
        return real_read(self, ref)

    code = FIG + "".join(
        f"submit_figure(go.Figure(go.Bar(y=[{i}])), parent={ref!r})\n"
        for i, ref in enumerate((missing, good, tampered, pending))
    )
    monkeypatch.setattr(type(dm.figure_store), "read", read)
    out = _run(tool, code, call_id="c_multi")
    monkeypatch.undo()
    # only index 1 (readable parent) is saved; its config is the parent's exact bytes
    [ref] = out["figure_refs"]
    assert dm.figure_store.read(ref)[2] == b'{"displaylogo":false}'
    assert dm.figure_store.read_record(ref)["execution"]["candidate_index"] == 1
    for reason in (
        "figure_not_found:revision_not_found",
        "fingerprint_mismatch:config_changed",
        "durability_failed:dir_fsync_failed",
    ):
        assert reason in out["figures"]
    assert not any(
        dm.figure_store.read_record(r)["config"]["sha256"]
        == "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"  # sha256({})
        for r in out["figure_refs"]
    )


@pytest.mark.parametrize(
    "has_config, config_file, ok",
    [
        (True, b'{"displaylogo":false}', True),
        (False, None, True),
        (True, None, False),  # declared but missing
        (False, b"{}", False),  # present but undeclared
        (True, b"[1]", False),  # not an object
        (True, b'{"x":NaN}', False),  # not strict JSON
        (True, b'{"x":"' + b"a" * (64 * 1024) + b'"}', False),  # over the 64 KiB cap
    ],
)
def test_submitted_config_is_structurally_all_or_nothing(
    tmp_path, has_config, config_file, ok
):
    from lobster.core.figures.records import read_submitted_figures

    cdir = tmp_path / "candidates" / "0"
    cdir.mkdir(parents=True)
    (cdir / "figure.json").write_bytes(b'{"data":[{"type":"bar","y":[1]}],"layout":{}}')
    meta = {"index": 0, "label": None, "request": None, "resolved_intent": None}
    meta.update({"parent": None, "has_source": False, "has_config": has_config})
    if config_file is not None:
        (cdir / "config.json").write_bytes(config_file)
    (cdir / "candidate.json").write_text(json.dumps(meta))
    figures, _, error = read_submitted_figures(tmp_path)
    if ok:
        assert error is None and figures[0].config_bytes == config_file
    else:
        assert (figures, error) == ([], "figure_submission_invalid")


# --- freshness seams: fail explicitly, never older data, nothing left behind --


def _no_attempt_dirs(dm):
    execution = dm.workspace_path / "cache" / "execution"
    return not execution.exists() or not any(execution.iterdir())


def test_an_in_memory_serialization_failure_fails_and_never_reads_an_older_file(
    env, monkeypatch
):
    import anndata
    import numpy as np

    from lobster.core.backends.h5ad_backend import H5ADBackend
    from lobster.services.execution.custom_code_execution_service import (
        CodeExecutionError,
    )

    dm, _ = env
    dm.data_dir.mkdir(parents=True, exist_ok=True)
    anndata.AnnData(np.ones((2, 2))).write_h5ad(dm.data_dir / "m.h5ad")  # older on disk
    dm.store_modality(name="m", adata=anndata.AnnData(np.full((2, 2), 3.0)), step="t")

    def fail(self, *a, **k):
        raise OSError("no space left")

    monkeypatch.setattr(H5ADBackend, "save", fail)
    service = CustomCodeExecutionService(dm)
    with pytest.raises(CodeExecutionError, match="serialize"):
        service.execute(_READ_SUM, modality_name="m", load_workspace_files=False)
    assert _no_attempt_dirs(dm)


def test_a_cost_probe_failure_is_typed_and_leaves_no_attempt_directory(
    env, monkeypatch
):
    import anndata
    import numpy as np

    from lobster.services.execution.custom_code_execution_service import (
        CodeExecutionError,
    )

    dm, _ = env
    dm.store_modality(name="m", adata=anndata.AnnData(np.ones((2, 2))), step="t")
    service = CustomCodeExecutionService(dm)
    monkeypatch.setattr(
        service, "_materialize_modality", lambda name, d, *_: d / "vanished.h5ad"
    )
    with pytest.raises(CodeExecutionError):
        service.execute("result = 1\n", modality_name="m", load_workspace_files=False)
    assert _no_attempt_dirs(dm)


def test_a_preflight_failure_leaves_no_attempt_directory(env):
    from lobster.services.execution.custom_code_execution_service import (
        CodeValidationError,
    )

    dm, _ = env
    service = CustomCodeExecutionService(dm)
    with pytest.raises(CodeValidationError):
        service.execute(
            "import anndata\nb = anndata.read_h5ad('unknown.h5ad')\n",
            load_workspace_files=False,
        )
    assert _no_attempt_dirs(dm)


@pytest.fixture
def any_name(monkeypatch):
    """Admit a name the modality-name check (1.2.1) refuses, as a name arriving by
    another route would: the downstream guard under test must still hold."""
    monkeypatch.setattr(
        "lobster.core.runtime.data_manager.check_modality_name", lambda name: None
    )


@pytest.mark.parametrize("names", [("x..y", "x_y"), ("a/m", "m")])
@pytest.mark.usefixtures("any_name")
def test_input_names_sharing_one_file_name_fail_closed_before_the_child(env, names):
    import anndata
    import numpy as np

    from lobster.services.execution.custom_code_execution_service import (
        CodeValidationError,
    )

    dm, _ = env
    first, second = names
    dm.store_modality(name=first, adata=anndata.AnnData(np.ones((2, 2))), step="t")
    dm.store_modality(
        name=second, adata=anndata.AnnData(np.full((2, 2), 3.0)), step="t"
    )
    service = CustomCodeExecutionService(dm)
    code = (
        f"import anndata\nb = anndata.read_h5ad('{second}.h5ad')\n"
        "result = [float(adata.X.sum()), float(b.X.sum())]\n"
    )
    with pytest.raises(CodeValidationError, match="same file name"):
        service.execute(code, modality_name=first, load_workspace_files=False)
    assert _no_attempt_dirs(dm)


# --- spill fix A: a DM-held input never falls to an older provider file ------------


@pytest.fixture()
def backed_small(monkeypatch):
    """Tiny fixtures load backed='r' (only the 1 GB size test is patched)."""
    import lobster.core.runtime.data_manager as dmmod

    monkeypatch.setattr(dmmod, "_should_use_backed_mode", lambda path: True)


def _spilled_backed_dm(tmp_path, older):
    import anndata
    import numpy as np

    dm = DataManagerV2(workspace_path=tmp_path, backed_mode=True)
    dm.store_modality(name="m", adata=anndata.AnnData(np.full((2, 2), 3.0)), step="t")
    assert dm._spill_modality("m") is not None  # real spill; reload will be backed
    if older:
        dm.data_dir.mkdir(parents=True, exist_ok=True)
        anndata.AnnData(np.ones((2, 2))).write_h5ad(dm.data_dir / "m.h5ad")
    return dm


@pytest.mark.parametrize("older", [True, False])
def test_a_spilled_backed_input_is_the_current_data_never_an_older_file(
    tmp_path, backed_small, older
):
    import hashlib

    dm = _spilled_backed_dm(tmp_path, older)
    spill_file = dm._modality_spilled["m"]
    old = dm.data_dir / "m.h5ad"
    old_sha = hashlib.sha256(old.read_bytes()).hexdigest() if older else None
    result, _, _ = CustomCodeExecutionService(dm).execute(
        _READ_SUM, modality_name="m", load_workspace_files=False
    )
    assert result == 12.0  # current; not 4.0 (older file), not a refusal (no file)
    assert not dm.modalities["m"].isbacked  # existing materialize_modality semantics
    assert spill_file.exists()  # the original file is kept
    if older:
        assert hashlib.sha256(old.read_bytes()).hexdigest() == old_sha
    assert _no_attempt_dirs(dm)


def test_a_failed_backed_conversion_is_explicit_and_keeps_the_original(
    tmp_path, backed_small, monkeypatch
):
    from lobster.services.execution.custom_code_execution_service import (
        CodeExecutionError,
    )

    dm = _spilled_backed_dm(tmp_path, older=True)
    spill_file = dm._modality_spilled["m"]

    def fail(name):
        raise MemoryError("no room")

    monkeypatch.setattr(dm, "materialize_modality", fail)
    with pytest.raises(CodeExecutionError, match="into memory"):
        CustomCodeExecutionService(dm).execute(
            _READ_SUM, modality_name="m", load_workspace_files=False
        )
    assert _no_attempt_dirs(dm)
    assert spill_file.exists()
    entry = dm.modalities["m"]  # restored by get_modality; conversion failed
    assert entry.isbacked and entry.file.is_open  # handle left as the method leaves it
    assert float(entry.X[:].sum()) == 12.0


# --- input edges captured at materialization from the object actually used -------

_SUBMIT_SUM = (
    FIG
    + "v = float(adata.X[:].sum())\n"
    + "submit_figure(go.Figure(go.Bar(y=[v])), label='p')\n"
    + "result = v\n"
)


def _edges(dm, out):
    [ref] = out["figure_refs"]
    return [
        (e["name"], e["instance_id"], e["origin"], e["verified"])
        for e in dm.figure_store.read_record(ref)["inputs"]
    ]


def _stored(dm, name, value):
    import anndata
    import numpy as np

    dm.store_modality(
        name=name, adata=anndata.AnnData(np.full((2, 2), value)), step="t"
    )
    return dm.resolve_input_identity(name).instance_id


def _tool_with(dm):
    service = CustomCodeExecutionService(dm)
    tool = create_execute_custom_code_tool(dm, service, agent_name="proteomics_expert")
    return service, tool


def test_known_cold_identities_are_kept(tmp_path):
    from lobster.core.backends.h5ad_modality_backend import H5ADModalityBackend

    dm = DataManagerV2(workspace_path=tmp_path / "a")
    sid = _stored(dm, "m", 3.0)
    dm._spill_modality("m")
    assert dm.resolve_input_identity("m") is None  # cold
    out = _run(_tool_with(dm)[1], _SUBMIT_SUM, call_id="c1", modality_name="m")
    assert _edges(dm, out) == [("m", sid, "store_event", True)]

    src = DataManagerV2(workspace_path=tmp_path / "src")
    bid = _stored(src, "m", 3.0)
    (tmp_path / "be").mkdir()
    src.modalities["m"].write_h5ad(tmp_path / "be" / "m.h5ad")
    dm2 = DataManagerV2(workspace_path=tmp_path / "b")
    dm2.set_modality_backend(H5ADModalityBackend(tmp_path / "be"))
    out = _run(_tool_with(dm2)[1], _SUBMIT_SUM, call_id="c2", modality_name="m")
    assert _edges(dm2, out) == [("m", bid, "store_event", True)]


def test_a_spilled_backed_input_records_the_store_event_it_was_read_from(
    tmp_path, backed_small
):
    import anndata

    dm = _spilled_backed_dm(tmp_path, older=True)
    sid = anndata.read_h5ad(dm._modality_spilled["m"]).uns["lineage"]["instance_id"]
    out = _run(_tool_with(dm)[1], _SUBMIT_SUM, call_id="c1", modality_name="m")
    assert out["result"] == 12.0
    assert _edges(dm, out) == [("m", sid, "store_event", True)]


def test_every_materialized_input_gets_an_edge_sorted_by_name(env):
    dm, _ = env
    zid = _stored(dm, "zeta", 3.0)
    aid = _stored(dm, "alpha", 1.0)
    code = (
        "import anndata\nb = anndata.read_h5ad('alpha.h5ad')\n"
        + _SUBMIT_SUM.replace("adata.X[:].sum()", "adata.X[:].sum() + b.X.sum()")
    )
    out = _run(_tool_with(dm)[1], code, call_id="c1", modality_name="zeta")
    assert out["result"] == 16.0
    assert _edges(dm, out) == [
        ("alpha", aid, "store_event", True),
        ("zeta", zid, "store_event", True),
    ]


def test_an_unknown_identity_is_kept_as_an_unverified_edge_with_a_warning(env):
    import anndata
    import numpy as np

    dm, _ = env
    dm.modalities["m"] = anndata.AnnData(np.ones((2, 2)))  # no lineage, no provider id
    out = _run(_tool_with(dm)[1], _SUBMIT_SUM, call_id="c1", modality_name="m")
    assert _edges(dm, out) == [("m", None, "declared", False)]
    assert out["input_identity"] == (
        "Input identity unknown (recorded as unverified): m"
    )
    assert dm.figure_store.dependents("m") == []  # never deletable by name


def test_a_provider_file_never_takes_a_lineage_identity(env):
    import anndata
    import numpy as np

    dm, _ = env
    sid = _stored(dm, "m", 1.0)
    dm.data_dir.mkdir(parents=True, exist_ok=True)
    f = anndata.AnnData(np.ones((2, 2)))
    f.uns["lineage"] = {"instance_id": "lineage_in_the_file"}
    f.write_h5ad(dm.data_dir / "filed.h5ad")  # only on disk: the provider supplies it
    assert "filed" not in dm.list_modalities()
    code = "import anndata\nb = anndata.read_h5ad('filed.h5ad')\n" + _SUBMIT_SUM
    out = _run(_tool_with(dm)[1], code, call_id="c1", modality_name="m")
    assert _edges(dm, out) == [
        ("filed", None, "declared", False),
        ("m", sid, "store_event", True),
    ]


@pytest.mark.parametrize("after", ["store", "evict"])
def test_a_store_or_eviction_after_materialization_cannot_rebind_the_edge(
    env, monkeypatch, after
):
    dm, _ = env
    sid = _stored(dm, "m", 1.0)
    service, tool = _tool_with(dm)
    real = service._execute_in_namespace

    def run_then_change(*a, **k):
        res = real(*a, **k)
        if after == "store":
            _stored(dm, "m", 3.0)  # a new store event while the attempt was running
        else:
            dm._spill_modality("m")
        return res

    monkeypatch.setattr(service, "_execute_in_namespace", run_then_change)
    out = _run(tool, _SUBMIT_SUM, call_id="c1", modality_name="m")
    assert out["result"] == 4.0
    assert _edges(dm, out) == [("m", sid, "store_event", True)]


def test_an_edit_merges_captured_edges_before_the_parents(env):
    dm, _ = env
    id1 = _stored(dm, "m", 1.0)
    tool = _tool_with(dm)[1]
    [r1] = _run(tool, _SUBMIT_SUM, call_id="c1", modality_name="m")["figure_refs"]
    edit = FIG + f"base = load_figure({r1!r})\nsubmit_figure(base, parent={r1!r})\n"
    out = _run(tool, edit, call_id="c2", modality_name="m")
    assert _edges(dm, out) == [
        ("m", id1, "store_event", True)
    ]  # same edge, no downgrade
    id2 = _stored(dm, "m", 3.0)
    out = _run(tool, edit, call_id="c3", modality_name="m")
    assert _edges(dm, out) == [
        ("m", id2, "store_event", True),  # captured edge first
        ("m", id1, "store_event", True),  # then the parent's, never dropped
    ]


def test_a_retry_whose_captured_edge_changed_is_the_existing_plan_conflict(env):
    dm, _ = env
    _stored(dm, "m", 1.0)
    tool = _tool_with(dm)[1]
    first = _run(tool, _SUBMIT_SUM, call_id="same", modality_name="m")
    assert first["figure_outcome"] == "available"
    _stored(dm, "m", 1.0)  # same data, new store event
    retry = _run(tool, _SUBMIT_SUM, call_id="same", modality_name="m")
    assert "execution_plan_changed" in retry["figures"]
    assert retry["figure_refs"] == []


def test_journey_reply_manifest_restart_edit_and_export_name_the_same_bytes(tmp_path):
    """Journey from reply to export, engine side, real disk and a real child process.

    Custom code saves a figure; the reply, the artifact manifest and the overview all name
    the same F@R; a fresh DataManagerV2 on the same workspace lists and reads it; an edit
    is a new revision and the original's bytes never change; the export package holds the
    exact saved bytes of both. Fingerprints prove bytes only, never display.
    """
    import gc
    import zipfile

    import anndata
    import numpy as np

    from lobster.agents.artifact_manifest import (
        diff_snapshots,
        render_manifest,
        take_snapshot,
    )
    from lobster.core.figures.listing import describe_workspace_figures

    def run_with_manifest(dm, tool, code, call_id, **args):
        before = take_snapshot(dm)
        reply = _run(tool, code, call_id=call_id, **args)
        return reply, render_manifest(diff_snapshots(before, take_snapshot(dm)))

    # 12. custom code saves a figure
    dm1 = DataManagerV2(workspace_path=tmp_path)
    dm1.store_modality(name="m", adata=anndata.AnnData(np.ones((2, 2))), step="t")
    tool1 = create_execute_custom_code_tool(
        dm1, CustomCodeExecutionService(dm1), agent_name="proteomics_expert"
    )
    created, manifest1 = run_with_manifest(
        dm1,
        tool1,
        FIG
        + "fig = go.Figure(go.Bar(x=['a', 'b'], y=[float(adata.X.sum()), 1.0]))\n"
        + "submit_figure(fig, label='Totals')\n",
        "c1",
        modality_name="m",
    )
    assert created["figure_outcome"] == "available"
    [r1] = created["figure_refs"]
    exact1 = dm1.figure_store.read(r1)
    r1_dir = tmp_path / "plots" / r1.replace("@", "/")
    tree1 = _tree_hashes(r1_dir)

    # 13. the reply, the manifest and the overview name the same F@R
    assert f'Created {r1} ("Totals")' in created["figures"]
    assert f"Figures saved: Totals ({r1})" in manifest1
    assert r1 in describe_workspace_figures(dm1)

    # 14. restart: a new DataManagerV2 on the same workspace lists and reads the same bytes
    del tool1, dm1
    gc.collect()
    dm2 = DataManagerV2(workspace_path=tmp_path)
    assert r1 in describe_workspace_figures(dm2)
    assert dm2.figure_store.read(r1) == exact1

    # 15. an edit is a new revision of the same figure; the original is preserved
    tool2 = create_execute_custom_code_tool(
        dm2, CustomCodeExecutionService(dm2), agent_name="proteomics_expert"
    )
    edited, manifest2 = run_with_manifest(
        dm2,
        tool2,
        FIG
        + f"base = load_figure({r1!r})\n"
        + "base.update_layout(title='Totals per group')\n"
        + f"submit_figure(base, parent={r1!r}, label='Totals (titled)')\n",
        "c2",
    )
    [r2] = edited["figure_refs"]
    assert r2 != r1 and r2.split("@")[0] == r1.split("@")[0]
    assert f"Figures saved: Totals (titled) ({r2})" in manifest2
    assert r1 not in manifest2  # the manifest names only what this call saved
    overview = describe_workspace_figures(dm2)
    assert r1 in overview and r2 in overview
    assert dm2.figure_store.read(r1) == exact1
    assert _tree_hashes(r1_dir) == tree1
    exact2 = dm2.figure_store.read(r2)
    assert exact2[1] != exact1[1]

    # 16. the export package holds the exact saved bytes of both revisions
    zip_path = dm2.create_data_package(output_dir=str(tmp_path / "out"))
    with zipfile.ZipFile(zip_path) as zf:
        index = {e["figure_ref"]: e for e in json.loads(zf.read("plots/index.json"))}
        for ref, exact in ((r1, exact1), (r2, exact2)):
            fid, rid = ref.split("@")
            for name, data in zip(
                ("revision.json", "figure.json", "config.json"), exact
            ):
                assert zf.read(f"plots/{fid}/{rid}/{name}") == data
    assert set(index) == {r1, r2}
    assert index[r2]["parents"] == [r1.split("@")[1]]
