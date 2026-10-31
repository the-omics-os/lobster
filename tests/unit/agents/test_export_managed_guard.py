"""Native exports never write into the managed figure folder (figure contract §1).

A model-supplied export path (or a modality name used as a file name) is checked with
the generic writer rule before anything is created: a managed target is refused with a
fixed message and nothing is written; an ordinary export is unchanged, including the
path the tool reports.
"""

import os
import threading
from importlib.util import find_spec
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.core.export_scope import WORKSPACE_EXPORT_MESSAGE
from lobster.core.figures.export_guard import MANAGED_EXPORT_MESSAGE
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.testing.provenance_replay import factory_tools

HAS_TX = find_spec("lobster.agents.transcriptomics") is not None
HAS_ML = find_spec("lobster.agents.machine_learning") is not None
needs_tx = pytest.mark.skipif(not HAS_TX, reason="transcriptomics package absent")
needs_ml = pytest.mark.skipif(not HAS_ML, reason="ML package absent")

DE_MODULE = "lobster.agents.transcriptomics.de_analysis_expert"
ANNOTATION_MODULE = "lobster.agents.transcriptomics.annotation_expert"
ML_MODULE = "lobster.agents.machine_learning.machine_learning_expert"
RESULTS_KEY = "de_results_condition_sample_b_vs_sample_a"
SAVED = b'{"saved": "figure"}'
REFUSAL = "inside the managed figure folder"


@pytest.fixture
def session(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    adata = ad.AnnData(
        X=np.ones((6, 2)),
        obs=pd.DataFrame(
            {
                "condition": ["sample_a"] * 3 + ["sample_b"] * 3,
                "cell_type_manual": ["type_a"] * 3 + ["type_b"] * 3,
            },
            index=[f"sample_{i}" for i in range(6)],
        ),
        var=pd.DataFrame(index=["gene_a", "gene_b"]),
    )
    adata.uns["is_pseudobulk"] = True
    results = pd.DataFrame(
        {"log2FoldChange": [1.0, -1.0], "padj": [0.01, 0.02]},
        index=pd.Index(["gene_a", "gene_b"], name="gene"),
    )
    adata.uns[RESULTS_KEY] = {"results_df": results, "parameters": {}}
    dm.store_modality("de_results", adata)
    rev = dm.workspace_path / "plots" / "fig_a" / "rev_1"
    rev.mkdir(parents=True)
    (rev / "figure.json").write_bytes(SAVED)
    dm.exports_dir.mkdir(parents=True, exist_ok=True)
    os.symlink(rev, dm.exports_dir / "alias")  # an alias INTO a saved revision
    return dm, cwd


def _snapshot(dm):
    """Every entry under plots/ with its bytes (dirs as None)."""
    plots = dm.workspace_path / "plots"
    return {
        str(p.relative_to(plots)): (None if p.is_dir() else p.read_bytes())
        for p in sorted(plots.rglob("*"))
    }


def _managed_targets(dm, name):
    """Managed spellings for an export named ``name`` (relative to exports/)."""
    ws = dm.workspace_path
    return {
        "dotdot-existing": "../plots/fig_a/rev_1/figure.json",
        "dotdot-new": f"../plots/fig_a/rev_1/{name}",
        "dotdot-new-figure": f"../plots/fig_new/rev_9/{name}",
        "absolute": str(ws / "plots" / "fig_a" / "rev_1" / "figure.json"),
        "alias": f"alias/{name}",
    }


def _assert_refused(dm, cwd, before, response):
    assert REFUSAL in response, response
    assert _snapshot(dm) == before  # nothing written, nothing created
    assert (
        dm.workspace_path / "plots" / "fig_a" / "rev_1" / "figure.json"
    ).read_bytes() == SAVED
    assert list(cwd.iterdir()) == []


TARGET_IDS = ["dotdot-existing", "dotdot-new", "dotdot-new-figure", "absolute", "alias"]


@needs_tx
@pytest.mark.parametrize("which", TARGET_IDS)
@pytest.mark.parametrize(
    "tool_name", ["export_de_results", "extract_and_export_de_results"]
)
@pytest.mark.parametrize("fmt", ["csv", "xlsx"])
def test_de_exports_refuse_a_managed_path(session, tool_name, which, fmt):
    from lobster.agents.transcriptomics.de_analysis_expert import de_analysis_expert

    dm, cwd = session
    before = _snapshot(dm)
    tools = factory_tools(de_analysis_expert, DE_MODULE, dm)
    args = {
        "modality_name": "de_results",
        "output_format": fmt,
        "output_path": _managed_targets(dm, f"x.{fmt}")[which],
    }
    if tool_name == "extract_and_export_de_results":
        args["shrink_lfc"] = False
    _assert_refused(dm, cwd, before, tools[tool_name].invoke(args))


@needs_tx
@pytest.mark.parametrize("which", TARGET_IDS)
@pytest.mark.parametrize("fmt", ["json", "csv"])
def test_annotation_export_refuses_a_managed_path(session, which, fmt):
    from lobster.agents.transcriptomics.annotation_expert import annotation_expert

    dm, cwd = session
    before = _snapshot(dm)
    tools = factory_tools(annotation_expert, ANNOTATION_MODULE, dm)
    response = tools["export_annotation_mapping"].invoke(
        {
            "modality_name": "de_results",
            "format": fmt,
            "output_filename": _managed_targets(dm, f"m.{fmt}")[which],
        }
    )
    _assert_refused(dm, cwd, before, response)


@needs_ml
@pytest.mark.parametrize(
    "output_dir",
    ["../plots/fig_a/rev_1", "../plots", "../plots/fig_new/rev_9", "alias", "ABS"],
)
@pytest.mark.parametrize("fmt", ["numpy", "csv"])
def test_ml_export_refuses_a_managed_directory(session, output_dir, fmt):
    from lobster.agents.machine_learning.machine_learning_expert import (
        machine_learning_expert,
    )

    dm, cwd = session
    if output_dir == "ABS":
        output_dir = str(dm.workspace_path / "plots" / "fig_a" / "rev_1")
    before = _snapshot(dm)
    tools = factory_tools(machine_learning_expert, ML_MODULE, dm)
    response = tools["export_for_ml_framework"].invoke(
        {
            "modality_name": "de_results",
            "format": fmt,
            "include_labels": False,
            "output_dir": output_dir,
        }
    )
    _assert_refused(dm, cwd, before, response)


@pytest.mark.parametrize(
    "path",
    ["../plots/fig_a/rev_1/m.h5ad", "../plots/fig_new/rev_9/m.h5ad", "ABS"],
)
def test_save_modality_refuses_a_managed_path(session, path):
    from lobster.core.figures.errors import ManagedPathError

    dm, cwd = session
    if path == "ABS":
        path = str(dm.workspace_path / "plots" / "fig_a" / "rev_1" / "m.h5ad")
    before = _snapshot(dm)
    with pytest.raises(ManagedPathError):
        dm.save_modality("de_results", path)
    assert _snapshot(dm) == before


def test_save_mudata_refuses_a_managed_path(session):
    pytest.importorskip("mudata")
    from lobster.core.figures.errors import ManagedPathError

    dm, cwd = session
    before = _snapshot(dm)
    with pytest.raises(ManagedPathError):
        dm.save_mudata("../plots/fig_a/rev_1/m.h5mu", modalities=["de_results"])
    assert _snapshot(dm) == before


@pytest.fixture
def any_name(monkeypatch):
    """Admit a name the modality-name check (1.2.1) refuses, as a name arriving by
    another route would: the downstream guard under test must still hold."""
    monkeypatch.setattr(
        "lobster.core.runtime.data_manager.check_modality_name", lambda name: None
    )


@needs_tx
def test_a_modality_named_into_plots_is_not_saved_there(session, monkeypatch, any_name):
    """A modality NAME is used as a file name by the DE save path."""
    from unittest.mock import Mock

    from lobster.agents.transcriptomics.de_analysis_expert import de_analysis_expert

    dm, cwd = session
    name = "../plots/fig_a/rev_1/evil"
    dm.store_modality(name, dm.get_modality("de_results").copy())
    before = _snapshot(dm)
    results = dm.get_modality("de_results").uns[RESULTS_KEY]["results_df"]
    science = Mock()
    science.validate_experimental_design.return_value = {"valid": True}
    science.run_pydeseq2_from_pseudobulk.return_value = (
        results,
        {
            "n_genes_tested": 2,
            "n_significant_genes": 2,
            "n_upregulated": 1,
            "n_downregulated": 1,
            "top_upregulated": ["gene_a"],
            "top_downregulated": ["gene_b"],
        },
    )
    monkeypatch.setattr(f"{DE_MODULE}.BulkRNASeqService", lambda **_: science)
    tools = factory_tools(de_analysis_expert, DE_MODULE, dm)
    response = tools["run_differential_expression"].invoke(
        {
            "modality_name": name,
            "groupby": "condition",
            "group1": "sample_a",
            "group2": "sample_b",
            "save_result": True,
        }
    )
    assert _snapshot(dm) == before, response
    assert list(cwd.iterdir()) == []


# ------------------------------------------------------------ ordinary exports unchanged


@needs_tx
@pytest.mark.parametrize(
    "tool_name", ["export_de_results", "extract_and_export_de_results"]
)
@pytest.mark.parametrize("output", ["relative", "subdir", "absolute"])
def test_de_ordinary_export_is_unchanged(session, tool_name, output):
    from lobster.agents.transcriptomics.de_analysis_expert import de_analysis_expert

    dm, cwd = session
    tools = factory_tools(de_analysis_expert, DE_MODULE, dm)
    rel = {"relative": "t.csv", "subdir": "tables/t.csv"}.get(output)
    target = dm.exports_dir / (rel or "abs.csv")
    args = {"modality_name": "de_results", "output_path": rel or str(target)}
    if tool_name == "extract_and_export_de_results":
        args["shrink_lfc"] = False
    response = tools[tool_name].invoke(args)
    assert target.is_file(), response
    reported = response.split("- Output: ", 1)[1].splitlines()[0]
    assert reported == str(target.resolve())  # the returned path is NOT changed


@needs_tx
def test_annotation_ordinary_export_is_unchanged(session):
    from lobster.agents.transcriptomics.annotation_expert import annotation_expert

    dm, cwd = session
    tools = factory_tools(annotation_expert, ANNOTATION_MODULE, dm)
    response = tools["export_annotation_mapping"].invoke(
        {"modality_name": "de_results", "format": "json", "output_filename": "m.json"}
    )
    target = dm.exports_dir / "m.json"
    assert target.is_file(), response
    reported = response.split("- Output file: ", 1)[1].splitlines()[0]
    assert reported == str(target.resolve())


@needs_ml
def test_ml_ordinary_export_is_unchanged(session):
    from lobster.agents.machine_learning.machine_learning_expert import (
        machine_learning_expert,
    )

    dm, cwd = session
    tools = factory_tools(machine_learning_expert, ML_MODULE, dm)
    response = tools["export_for_ml_framework"].invoke(
        {
            "modality_name": "de_results",
            "format": "csv",
            "include_labels": False,
            "output_dir": "ml",
        }
    )
    target = dm.exports_dir / "ml" / "de_results_ml_data.csv"
    assert target.is_file(), response
    reported = response.split("**Export directory**: ", 1)[1].splitlines()[0]
    assert reported == str(dm.exports_dir / "ml")


def test_save_modality_ordinary_path_is_unchanged(session):
    dm, cwd = session
    returned = dm.save_modality("de_results", "m.h5ad")
    assert returned == str(dm.data_dir / "m.h5ad")
    assert (dm.data_dir / "m.h5ad").is_file()


# ------------------------------------------------------- the XLSX -> CSV fallback target


def _no_openpyxl(monkeypatch):
    def refuse(*_a, **_k):
        raise ImportError("openpyxl")

    monkeypatch.setattr(pd.DataFrame, "to_excel", refuse)


@needs_tx
def test_csv_fallback_refuses_a_derived_managed_target(session, monkeypatch):
    """foo.xlsx is ordinary, but its fallback foo.csv is an alias of a saved figure."""
    from lobster.agents.transcriptomics.de_analysis_expert import de_analysis_expert

    dm, cwd = session
    os.symlink(
        dm.workspace_path / "plots" / "fig_a" / "rev_1" / "figure.json",
        dm.exports_dir / "foo.csv",
    )
    _no_openpyxl(monkeypatch)
    before = _snapshot(dm)
    tools = factory_tools(de_analysis_expert, DE_MODULE, dm)
    response = tools["extract_and_export_de_results"].invoke(
        {
            "modality_name": "de_results",
            "shrink_lfc": False,
            "output_format": "xlsx",
            "output_path": "foo.xlsx",
        }
    )
    _assert_refused(dm, cwd, before, response)
    assert not (dm.exports_dir / "foo.xlsx").exists()


@needs_tx
def test_csv_fallback_ordinary_export_is_unchanged(session, monkeypatch):
    from lobster.agents.transcriptomics.de_analysis_expert import de_analysis_expert

    dm, cwd = session
    _no_openpyxl(monkeypatch)
    tools = factory_tools(de_analysis_expert, DE_MODULE, dm)
    response = tools["extract_and_export_de_results"].invoke(
        {
            "modality_name": "de_results",
            "shrink_lfc": False,
            "output_format": "xlsx",
            "output_path": "bar.xlsx",
        }
    )
    target = dm.exports_dir / "bar.csv"
    assert target.is_file(), response
    reported = response.split("- Output: ", 1)[1].splitlines()[0]
    assert reported == str(target.resolve())


# ------------------------------------------- a modality name in a service file name


@pytest.mark.parametrize("name", ["plots/fig_a/rev_1/m", "test_modality"])
def test_pathway_csv_never_lands_in_the_managed_folder(tmp_path, monkeypatch, name):
    from lobster.core.figures.errors import ManagedPathError
    from lobster.services.analysis import pathway_enrichment_bridge_service as svc

    ws = tmp_path / "ws"
    rev = ws / "plots" / "fig_a" / "rev_1"
    rev.mkdir(parents=True)
    (rev / "figure.json").write_bytes(SAVED)
    monkeypatch.setattr(svc, "resolve_workspace", lambda *a, **k: ws)
    adata = ad.AnnData(
        X=np.ones((2, 2)),
        obs=pd.DataFrame(index=["s1", "s2"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    df = pd.DataFrame(
        {"pathway_name": ["p"], "fdr": [0.01], "gene_overlap": ["g1"], "source": ["go"]}
    )
    stats = {
        "n_genes_selected": 1,
        "n_pathways_significant": 1,
        "fdr_threshold": 0.05,
        "sources": ["go"],
        "organism": "human",
        "timestamp": "t",
    }
    before = sorted(str(p) for p in (ws / "plots").rglob("*"))
    service = svc.PathwayEnrichmentBridgeService(timeout=1)
    if name.startswith("plots/"):
        with pytest.raises(ManagedPathError):
            service._store_results(adata, df, name, "sel", stats)
        assert sorted(str(p) for p in (ws / "plots").rglob("*")) == before
        assert (rev / "figure.json").read_bytes() == SAVED
    else:
        result = service._store_results(adata, df, name, "sel", stats)
        target = ws / f"{name}_pathway_enrichment.csv"
        assert target.is_file()
        assert result.uns["pathway_enrichment"]["csv_file"] == str(target)


# ------------------------------------- a publication entry id in a metadata file name


def _process_entry_with(dm, monkeypatch, entry_id):
    from lobster.core.schemas.publication_queue import PublicationQueueEntry
    from lobster.services.orchestration import publication_processing_service as pps

    dm.publication_queue.add_entry(
        PublicationQueueEntry(entry_id=entry_id, pmid="12345678", title="t")
    )
    service = pps.PublicationProcessingService(data_manager=dm)
    monkeypatch.setattr(
        service.content_service,
        "get_full_content",
        lambda **_: {"content": "abstract text", "summary": "s"},
    )
    return service.process_entry(entry_id, extraction_tasks="metadata")


@pytest.mark.parametrize(
    "entry_id",
    [
        "pub_queue_pmid_x/../../plots/fig_a/rev_1/figure",  # into a saved revision
        "pub_queue_pmid_x/../../outside",  # out of metadata/
    ],
)
def test_a_crafted_entry_id_writes_no_metadata_file(session, monkeypatch, entry_id):
    dm, cwd = session
    # The first component must exist for the OS to walk through it: model another
    # write having created it.
    (dm.workspace_path / "metadata" / "pub_queue_pmid_x").mkdir(parents=True)
    logged = []
    monkeypatch.setattr(dm, "log_tool_usage", lambda *a, **k: logged.append(k))
    before = _snapshot(dm)
    outcome = _process_entry_with(dm, monkeypatch, entry_id)
    assert _snapshot(dm) == before, outcome.response_markdown
    assert (
        dm.workspace_path / "plots" / "fig_a" / "rev_1" / "figure.json"
    ).read_bytes() == SAVED
    assert not (dm.workspace_path / "outside_metadata.json").exists()
    # Intentional response change (P1-2 F5): the early refusal replaces the late
    # persistence warning; the queue record is untouched and nothing is logged.
    assert logged == []
    assert dm.publication_queue.get_entry(entry_id).status == "pending"
    assert outcome.final_status == "failed"
    # The exact fixed outcome (names added by the fix, imported after the behaviour).
    from lobster.services.orchestration.publication_processing_service import (
        ENTRY_ID_REFUSED,
        entry_refused,
    )

    assert outcome.response_markdown == ENTRY_ID_REFUSED
    assert entry_refused(outcome)


def test_an_ordinary_entry_id_still_writes_its_metadata_file(session, monkeypatch):
    dm, cwd = session
    outcome = _process_entry_with(dm, monkeypatch, "pub_queue_pmid_12345678")
    target = dm.workspace_path / "metadata" / "pub_queue_pmid_12345678_metadata.json"
    assert target.is_file(), outcome.response_markdown
    assert '"content": "abstract text"' in target.read_text()


# --- lane B (P1-2) ----------------------------------------------------------------
# Real committed revision (B0 fixture), whole-workspace snapshots, the real tools.
# Faked only at external boundaries: HTTP (urllib / requests / httpx), the pysradb
# import, and the DE computation service (as in the cases above).

PUB_MODULE = "lobster.services.orchestration.publication_processing_service"
RESEARCH_MODULE = "lobster.agents.research.research_agent"
PRJ = "PRJNA123456"
REFUSAL_LOG = "publication entry refused (reason=%s)"


@pytest.fixture
def lane_b(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tests.unit.export_guard_fixtures import saved_revision

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    rev = saved_revision(dm.workspace_path)
    logged = []
    monkeypatch.setattr(dm, "log_tool_usage", lambda *a, **k: logged.append(k))
    return SimpleNamespace(
        dm=dm, ws=dm.workspace_path, rev=rev, logged=logged, cwd=cwd, tmp=tmp_path
    )


def _tree(s):
    from tests.unit.export_guard_fixtures import workspace_tree

    return workspace_tree(s.ws)


def _rev_file(s):
    return s.rev.directory / sorted(s.rev.files)[0]


def _rev_spelling(s, name):
    return f"plots/{s.rev.directory.parent.name}/{s.rev.directory.name}/{name}"


def _link(link, target):
    from tests.unit.export_guard_fixtures import alias

    return alias(link, target)


@pytest.fixture
def net(monkeypatch):
    """Every HTTP request (URL + body) is recorded and answered 404, offline."""
    import urllib.error
    import urllib.request

    import requests

    seen = []

    def opener_open(self, fullurl, data=None, timeout=None):
        url = getattr(fullurl, "full_url", str(fullurl))
        seen.append(
            f"{url} {data if data is not None else getattr(fullurl, 'data', None)!r}"
        )
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    def session_send(self, request, **kwargs):
        seen.append(f"{request.url} {request.body!r}")
        response = requests.Response()
        response.status_code, response.url, response.request = 404, request.url, request
        response._content = b""
        return response

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", opener_open)
    monkeypatch.setattr(requests.Session, "send", session_send)
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx is a dependency
        return seen

    def client_send(self, request, **kwargs):
        seen.append(f"{request.url} {request.content!r}")
        return httpx.Response(404, request=request)

    monkeypatch.setattr(httpx.Client, "send", client_send)
    return seen


@pytest.fixture
def sraweb(monkeypatch):
    """``pysradb.SRAweb`` at its import boundary; ``run`` decides each answer."""
    import sys
    import types

    state = types.SimpleNamespace(calls=[], run=None)

    def default(project):
        return pd.DataFrame([{"run_accession": "SRR1", "study_accession": project}])

    state.run = default

    class SRAweb:
        def sra_metadata(self, project, detailed=True):
            state.calls.append(project)
            return state.run(project)

    module = types.ModuleType("pysradb")
    module.SRAweb = SRAweb
    monkeypatch.setitem(sys.modules, "pysradb", module)
    return state


def _queue(s, entry_id, pmid, text=None):
    """Queue an entry; ``text`` is served by the real content cache (no HTTP)."""
    from lobster.core.schemas.publication_queue import PublicationQueueEntry

    s.dm.publication_queue.add_entry(
        PublicationQueueEntry(entry_id=entry_id, pmid=pmid, title="t")
    )
    if text is not None:
        source = f"PMID:{pmid}"
        s.dm.cache_publication_content(
            source, {"content": text, "source": source}, format="json"
        )
    s.logged.clear()


def _record(s, entry_id):
    return s.dm.publication_queue.get_entry(entry_id).model_dump()


def _service(s):
    from lobster.services.orchestration import publication_processing_service as pps

    return pps.PublicationProcessingService(data_manager=s.dm)


def _research_tools(s):
    from lobster.agents.research.research_agent import research_agent

    return factory_tools(research_agent, RESEARCH_MODULE, s.dm)


def _refusal_records(caplog):
    return [r for r in caplog.records if r.msg == REFUSAL_LOG]


def _hostile_ids(s):
    return {
        "separator": "pub_queue_x/../../" + _rev_spelling(s, "m"),
        "alias": "pub_queue_plain",
    }


def _arm(s, kind, entry_id):
    if kind == "alias":  # a plain id whose metadata file is a link into a revision
        _link(s.ws / "metadata" / f"{entry_id}_metadata.json", _rev_file(s))


@pytest.mark.parametrize("kind", ["separator", "alias"])
def test_lane_b_refused_entry_sends_nothing_writes_nothing_logs_nothing(
    lane_b, net, kind
):
    s = lane_b
    entry_id = _hostile_ids(s)[kind]
    _queue(s, entry_id, "11111111")
    _arm(s, kind, entry_id)
    tool = _research_tools(s)["process_publication_entry"]  # factory set-up first
    record, before = _record(s, entry_id), _tree(s)
    response = tool.invoke({"entry_id": entry_id})
    assert net == []  # no request at all
    assert _record(s, entry_id) == record and s.logged == []
    assert _tree(s) == before and s.rev.unchanged()
    assert list(s.cwd.iterdir()) == []
    if kind == "alias":
        assert response == MANAGED_EXPORT_MESSAGE
    else:
        from lobster.services.orchestration.publication_processing_service import (
            ENTRY_ID_REFUSED,
        )

        assert response == ENTRY_ID_REFUSED


def test_lane_b_parallel_tool_never_sends_or_shows_a_refused_entry(lane_b, net, capsys):
    s = lane_b
    hostile = _hostile_ids(s)["separator"]
    _queue(s, hostile, "11111111")
    _queue(s, "pub_queue_ok", "22222222", text="ordinary abstract")
    record = _record(s, hostile)
    response = _research_tools(s)["process_publication_queue"].invoke(
        {
            "parallel_workers": 2,
            "extraction_tasks": "ncbi_enrich,metadata",
        }
    )
    shown = response + "".join(capsys.readouterr())
    assert "processed=2" in response
    assert hostile not in shown and "11111111" not in shown
    assert not [r for r in net if "11111111" in r]
    assert [r for r in net if "22222222" in r]  # the ordinary entry was enriched
    assert _record(s, hostile) == record
    # One per-entry log (the ordinary entry) and the batch log; none for the refused.
    assert sorted(k["tool_name"] for k in s.logged) == [
        "process_publication_entry",
        "process_publication_queue",
    ]
    assert not [k for k in s.logged if hostile in repr(k) or "11111111" in repr(k)]
    assert s.rev.unchanged()


def _sra_target(s):
    """The SRA samples file exactly as the content writer names it (sanitized)."""
    from lobster.services.data_access.workspace_content_service import (
        WorkspaceContentService,
    )

    name = WorkspaceContentService(s.dm)._sanitize_filename(f"sra_{PRJ}_samples")
    return s.ws / "metadata" / f"{name}.json"


def test_lane_b_late_sra_refusal_crosses_both_catches(lane_b, net, sraweb, caplog):
    import logging

    s = lane_b
    _queue(s, "pub_queue_sra", "33333333", text=f"reads in {PRJ}")
    service = _service(s)  # its set-up creates its folders first
    _link(_sra_target(s), _rev_file(s))
    record, before = _record(s, "pub_queue_sra"), _tree(s)
    with caplog.at_level(logging.WARNING, logger=PUB_MODULE):
        outcome = service.process_entry(
            "pub_queue_sra", extraction_tasks="identifiers,fetch_sra_metadata"
        )
    assert sraweb.calls == [PRJ]  # the late write was reached
    # Either generic catch would have continued into the ordinary reply.
    assert _record(s, "pub_queue_sra") == record and s.logged == []
    assert _tree(s) == before and s.rev.unchanged()
    assert outcome.final_status == "failed"
    assert outcome.response_markdown == MANAGED_EXPORT_MESSAGE
    assert [r.args for r in _refusal_records(caplog)] == [("managed_path",)]
    from lobster.services.orchestration.publication_processing_service import (
        entry_refused,
    )

    assert entry_refused(outcome)


def test_lane_b_late_persistence_retarget_is_refused(lane_b, net, sraweb):
    """An L1-shaped deterministic retarget between preflight and write (no race claim)."""
    s = lane_b
    entry_id = "pub_queue_late"
    _queue(s, entry_id, "44444444", text=f"reads in {PRJ}")
    target = s.ws / "metadata" / f"{entry_id}_metadata.json"

    def retarget(project):
        _link(target, _rev_file(s))
        return pd.DataFrame()

    sraweb.run = retarget
    record = _record(s, entry_id)
    outcome = _service(s).process_entry(
        entry_id, extraction_tasks="metadata,identifiers,fetch_sra_metadata"
    )
    assert sraweb.calls == [PRJ]
    assert _record(s, entry_id) == record and s.logged == []
    assert s.rev.unchanged()
    assert outcome.response_markdown == MANAGED_EXPORT_MESSAGE
    from lobster.services.orchestration.publication_processing_service import (
        entry_refused,
    )

    assert entry_refused(outcome)


def test_regression_lane_b_unrelated_sra_errors_keep_their_text(lane_b, net, sraweb):
    s = lane_b
    _queue(s, "pub_queue_lib", "55555555", text=f"reads in {PRJ}")

    def library_down(project):
        raise RuntimeError("library down")

    sraweb.run = library_down
    outcome = _service(s).process_entry(
        "pub_queue_lib", extraction_tasks="identifiers,fetch_sra_metadata"
    )
    assert f"  - {PRJ}: ✗ Failed (library down)" in outcome.response_markdown
    assert MANAGED_EXPORT_MESSAGE not in outcome.response_markdown

    class _Unprintable(Exception):
        def __str__(self):  # the per-project handler's own message fails
            raise RuntimeError("step")

    def unprintable(project):
        raise _Unprintable()

    _queue(s, "pub_queue_step", "66666666", text=f"reads in {PRJ}")
    sraweb.run = unprintable
    outcome = _service(s).process_entry(
        "pub_queue_step", extraction_tasks="identifiers,fetch_sra_metadata"
    )
    assert "✗ SRA metadata fetch failed: step" in outcome.response_markdown
    assert MANAGED_EXPORT_MESSAGE not in outcome.response_markdown


def test_lane_b_refusal_warning_is_logged_once_without_id_or_path(lane_b, net, caplog):
    import logging

    s = lane_b
    ids = _hostile_ids(s)
    for kind, entry_id in ids.items():
        _queue(s, entry_id, "11111111")
        _arm(s, kind, entry_id)
    service = _service(s)
    expected = {("entry_id_not_plain",), ("managed_path",)}

    def check(records):
        assert sorted(r.args for r in records) == sorted(expected)
        for r in records:
            text = r.getMessage() + repr(r.args)
            assert not any(i in text for i in ids.values()) and str(s.ws) not in text
            assert r.exc_info is None

    with caplog.at_level(logging.WARNING, logger=PUB_MODULE):
        service._process_entry_internal(ids["separator"])
        service._process_entry_internal(ids["alias"])
    check(_refusal_records(caplog))
    caplog.clear()

    with caplog.at_level(logging.WARNING, logger=PUB_MODULE):
        summary = service.process_queue_entries(status_filter="pending")
    check(_refusal_records(caplog))
    assert not any(i in summary for i in ids.values())
    from lobster.services.orchestration.publication_processing_service import (
        REFUSED_ENTRY_LABEL,
    )

    assert summary.count(f"- Entry ID: {REFUSED_ENTRY_LABEL}") == 2
    caplog.clear()

    with caplog.at_level(logging.WARNING, logger=PUB_MODULE):
        result = service.process_entries_parallel(
            list(ids.values()), extraction_tasks="ncbi_enrich", show_progress=False
        )
    check(_refusal_records(caplog))
    assert result.failed == 2 and net == []
    assert s.logged == [] and s.rev.unchanged()


# --- lane B: DE save boundary (F7) and the XLSX derived name ----------------------


def _de_science(monkeypatch, results):
    from unittest.mock import Mock

    science = Mock()
    science.validate_experimental_design.return_value = {"valid": True}
    stats = {
        "n_genes_tested": 2,
        "n_significant_genes": 2,
        "n_upregulated": 1,
        "n_downregulated": 1,
        "top_upregulated": ["gene_a"],
        "top_downregulated": ["gene_b"],
    }
    science.run_pydeseq2_from_pseudobulk.return_value = (results, stats)
    monkeypatch.setattr(f"{DE_MODULE}.BulkRNASeqService", lambda **_: science)
    return science


def _de_modality(pseudobulk=True, group1="sample_a"):
    adata = ad.AnnData(
        X=np.ones((6, 2)),
        obs=pd.DataFrame(
            {"condition": [group1] * 3 + ["sample_b"] * 3},
            index=[f"sample_{i}" for i in range(6)],
        ),
        var=pd.DataFrame(index=["gene_a", "gene_b"]),
    )
    if pseudobulk:
        adata.uns["is_pseudobulk"] = True
    results = pd.DataFrame(
        {"log2FoldChange": [1.0, -1.0], "padj": [0.01, 0.02]},
        index=pd.Index(["gene_a", "gene_b"], name="gene"),
    )
    adata.uns[RESULTS_KEY] = {"results_df": results, "parameters": {}}
    return adata, results


@pytest.fixture
def de(lane_b, monkeypatch):
    adata, results = _de_modality()
    lane_b.dm.store_modality("de_results", adata)
    lane_b.science = _de_science(monkeypatch, results)
    return lane_b


def _de_tools(s):
    from lobster.agents.transcriptomics.de_analysis_expert import de_analysis_expert

    return factory_tools(de_analysis_expert, DE_MODULE, s.dm)


def _assert_de_refused(s, before, response, name="de_results"):
    adata = s.dm.get_modality(name)
    assert response == MANAGED_EXPORT_MESSAGE
    assert "de_current_results_key" not in adata.uns
    assert s.logged == []
    assert _tree(s) == before and s.rev.unchanged()
    assert list(s.cwd.iterdir()) == []


DE_ARGS = {
    "modality_name": "de_results",
    "groupby": "condition",
    "group1": "sample_a",
    "group2": "sample_b",
    "save_result": True,
}


@needs_tx
@pytest.mark.parametrize("which", ["csv", "h5ad"])
def test_lane_b_pseudobulk_de_refuses_every_save_target_before_analysis(de, which):
    s = de
    if which == "csv":
        _link(s.dm.exports_dir / "de_results_de_results.csv", _rev_file(s))
    else:
        _link(s.dm.data_dir / "de_results_with_de_results.h5ad", _rev_file(s))
    dtype = s.dm.get_modality("de_results").obs["condition"].dtype
    modalities, before = list(s.dm.list_modalities()), _tree(s)
    response = _de_tools(s)["run_differential_expression"].invoke(DE_ARGS)
    s.science.run_pydeseq2_from_pseudobulk.assert_not_called()
    assert s.dm.get_modality("de_results").obs["condition"].dtype == dtype
    assert list(s.dm.list_modalities()) == modalities
    _assert_de_refused(s, before, response)


@needs_tx
def test_lane_b_direct_bulk_de_never_saves_into_a_revision_by_a_group_label(
    lane_b, monkeypatch
):
    """A group label is data: since 1.2.1 it is encoded into one plain result name
    (Sol WP-IMPL-03), so a label spelled into a revision cannot reach it. The run
    completes and its file stays a plain name in data/."""
    import types

    from lobster.core.export_scope import plain_name_part

    s = lane_b
    group1 = "x/../../" + _rev_spelling(s, "y")
    adata, results = _de_modality(pseudobulk=False, group1=group1)
    s.dm.store_modality("bulk", adata)
    science = _de_science(monkeypatch, results)
    stats = {
        "method": "t_test",
        "n_genes_tested": 2,
        "n_significant_genes": 1,
        "n_upregulated": 1,
        "n_downregulated": 0,
        "top_upregulated": ["gene_a"],
        "top_downregulated": [],
        "de_results_key": "de_results_bulk",
    }
    science.run_differential_expression_analysis.return_value = (
        adata.copy(),
        stats,
        types.SimpleNamespace(code_template="", parameters={}),
    )
    monkeypatch.setattr(s.dm, "log_tool_usage", lambda *a, **k: None)
    response = _de_tools(s)["run_differential_expression"].invoke(
        {**DE_ARGS, "modality_name": "bulk", "group1": group1, "method": "t_test"}
    )
    name = f"bulk_de_{plain_name_part(group1)}_vs_sample_b"
    assert name in s.dm.list_modalities(), response
    assert (s.dm.data_dir / f"{name}.h5ad").is_file()
    assert s.rev.unchanged()
    assert list(s.cwd.iterdir()) == []


@needs_tx
@pytest.mark.parametrize("which", ["csv", "h5ad"])
def test_lane_b_formula_de_refuses_each_save_target_before_analysis(de, which):
    s = de
    if which == "csv":
        _link(s.dm.exports_dir / "de_results_formula_de_results.csv", _rev_file(s))
    else:
        _link(s.dm.data_dir / "de_results_with_formula_results.h5ad", _rev_file(s))
    before = _tree(s)
    response = _de_tools(s)["run_de_with_formula"].invoke(
        {
            "pseudobulk_modality": "de_results",
            "formula": "~condition",
            "contrast": ["condition", "sample_b", "sample_a"],
        }
    )
    s.science.validate_experimental_design.assert_not_called()
    s.science.run_pydeseq2_from_pseudobulk.assert_not_called()
    _assert_de_refused(s, before, response)


@needs_tx
def test_lane_b_ordinary_pseudobulk_and_direct_saves_are_unchanged(de, monkeypatch):
    import types

    s = de
    response = _de_tools(s)["run_differential_expression"].invoke(DE_ARGS)
    assert (s.dm.exports_dir / "de_results_de_results.csv").is_file(), response
    assert (s.dm.data_dir / "de_results_with_de_results.h5ad").is_file()
    assert "run_differential_expression" in [k["tool_name"] for k in s.logged]

    adata, results = _de_modality(pseudobulk=False)
    s.dm.store_modality("bulk", adata)
    science = _de_science(monkeypatch, results)
    stats = {
        "method": "t_test",
        "n_genes_tested": 2,
        "n_significant_genes": 1,
        "n_upregulated": 1,
        "n_downregulated": 0,
        "top_upregulated": ["gene_a"],
        "top_downregulated": [],
        "de_results_key": "de_results_bulk",
    }
    ir = types.SimpleNamespace(code_template="", parameters={})
    science.run_differential_expression_analysis.return_value = (
        adata.copy(),
        stats,
        ir,
    )
    response = _de_tools(s)["run_differential_expression"].invoke(
        {**DE_ARGS, "modality_name": "bulk", "method": "t_test"}
    )
    assert "bulk_de_sample_a_vs_sample_b" in s.dm.list_modalities(), response
    assert (s.dm.data_dir / "bulk_de_sample_a_vs_sample_b.h5ad").is_file()
    assert s.rev.unchanged()


@needs_tx
def test_lane_b_ordinary_formula_save_is_unchanged(de):
    s = de
    response = _de_tools(s)["run_de_with_formula"].invoke(
        {
            "pseudobulk_modality": "de_results",
            "formula": "~condition",
            "contrast": ["condition", "sample_b", "sample_a"],
        }
    )
    assert (s.dm.exports_dir / "de_results_formula_de_results.csv").is_file(), response
    assert (s.dm.data_dir / "de_results_with_formula_results.h5ad").is_file()
    assert s.rev.unchanged()


@needs_tx
def test_lane_b_xlsx_resolved_fallback_name_is_checked_before_mkdir(de):
    s = de
    real = s.dm.exports_dir / "data.xlsx"
    real.mkdir(parents=True)
    _link(s.dm.exports_dir / "lnk", real)
    _link(s.dm.exports_dir / "data.csv", s.rev.directory)
    before = _tree(s)
    response = _de_tools(s)["extract_and_export_de_results"].invoke(
        {
            "modality_name": "de_results",
            "shrink_lfc": False,
            "output_format": "xlsx",
            "output_path": "lnk/new/x.xlsx",
        }
    )
    assert response == MANAGED_EXPORT_MESSAGE
    assert not (real / "new").exists()
    assert _tree(s) == before and s.rev.unchanged() and s.logged == []


@needs_tx
def test_lane_b_ordinary_xlsx_fallback_without_excel_writers(de, monkeypatch):
    import sys

    s = de
    # Blocked at the import boundary, including submodules cached by earlier tests.
    for name in [
        m for m in list(sys.modules) if m.split(".")[0] in ("openpyxl", "xlsxwriter")
    ] + ["openpyxl", "xlsxwriter"]:
        monkeypatch.setitem(sys.modules, name, None)
    response = _de_tools(s)["extract_and_export_de_results"].invoke(
        {
            "modality_name": "de_results",
            "shrink_lfc": False,
            "output_format": "xlsx",
            "output_path": "plain.xlsx",
        }
    )
    target = s.dm.exports_dir / "plain.csv"
    assert target.is_file(), response
    assert not (s.dm.exports_dir / "plain.xlsx").exists()
    assert s.rev.unchanged()


# --- B2 (P1-2 lane B fixes) -------------------------------------------------------


@needs_tx
@pytest.mark.parametrize(
    "tool_name", ["export_de_results", "extract_and_export_de_results"]
)
def test_b2_de_export_folder_passing_through_a_revision_is_refused(de, tool_name):
    s = de
    spelled = f"../{_rev_spelling(s, 'NEW')}/../../../../exports/out.csv"
    assert os.path.normpath(s.dm.exports_dir / spelled) == str(
        s.dm.exports_dir / "out.csv"
    )
    args = {"modality_name": "de_results", "output_path": spelled}
    if tool_name == "extract_and_export_de_results":
        args["shrink_lfc"] = False
    before = _tree(s)
    response = _de_tools(s)[tool_name].invoke(args)
    assert response == MANAGED_EXPORT_MESSAGE
    assert not (s.rev.directory / "NEW").exists()
    assert _tree(s) == before and s.rev.unchanged() and s.logged == []


@needs_tx
@pytest.mark.parametrize(
    "tool_name", ["export_de_results", "extract_and_export_de_results"]
)
def test_b2_de_export_to_the_workspace_root_is_unchanged(de, tool_name):
    s = de
    args = {"modality_name": "de_results", "output_path": str(s.ws / "root.csv")}
    if tool_name == "extract_and_export_de_results":
        args["shrink_lfc"] = False
    response = _de_tools(s)[tool_name].invoke(args)
    assert (s.ws / "root.csv").is_file(), response
    assert response != MANAGED_EXPORT_MESSAGE and s.rev.unchanged()


@needs_tx
@pytest.mark.parametrize("case", ["nul", "loop"])
def test_b2_xlsx_unresolvable_destination_gets_the_fixed_refusal(de, case, caplog):
    import logging

    s = de
    if case == "nul":
        spelled = "x\x00.xlsx"
    else:
        _link(s.dm.exports_dir / "loop", s.dm.exports_dir / "loop")
        spelled = "loop/x.xlsx"
    before = _tree(s)
    with caplog.at_level(logging.ERROR, logger=DE_MODULE):
        response = _de_tools(s)["extract_and_export_de_results"].invoke(
            {
                "modality_name": "de_results",
                "shrink_lfc": False,
                "output_format": "xlsx",
                "output_path": spelled,
            }
        )
    # The workspace-scope check (1.2.1) runs first and fails closed on a spelling
    # it cannot resolve (a NUL byte, a link loop): its fixed message.
    assert response == WORKSPACE_EXPORT_MESSAGE
    assert caplog.records == []
    assert _tree(s) == before and s.rev.unchanged() and s.logged == []


@needs_tx
def test_b2_de_refusal_logs_no_modality_name(lane_b, monkeypatch, caplog, any_name):
    import logging

    s = lane_b
    name = "x/../../" + _rev_spelling(s, "m")
    adata, results = _de_modality()
    s.dm.store_modality(name, adata)
    _de_science(monkeypatch, results)
    with caplog.at_level(logging.DEBUG, logger=DE_MODULE):
        response = _de_tools(s)["run_differential_expression"].invoke(
            {**DE_ARGS, "modality_name": name}
        )
    assert response == MANAGED_EXPORT_MESSAGE
    assert not [r for r in caplog.records if name in r.getMessage()]


def _lock_alias(s):
    """The publication queue's lock file as a dangling link into the revision."""
    from pathlib import Path

    lock = Path(s.dm.publication_queue._file_lock_path)
    if os.path.lexists(lock):
        lock.unlink()  # the real lock file left by the set-up
    _link(lock, s.rev.directory / "new.lock")
    return s.rev.directory / "new.lock"


@pytest.mark.parametrize("mode", ["processing", "status_override"])
def test_b2_publication_queue_lock_alias_is_refused_before_anything(lane_b, net, mode):
    s = lane_b
    entry_id = "pub_queue_lock"
    _queue(s, entry_id, "77777777", text="ordinary abstract")
    managed = _lock_alias(s)
    tool = _research_tools(s)["process_publication_entry"]  # factory set-up first
    record, before = _record(s, entry_id), _tree(s)
    args = {"entry_id": entry_id}
    if mode == "status_override":
        args["status_override"] = "failed"
    response = tool.invoke(args)
    assert not os.path.exists(managed)  # the lock file was never created
    assert net == []
    assert _record(s, entry_id) == record and s.logged == []
    assert _tree(s) == before and s.rev.unchanged()
    assert response == MANAGED_EXPORT_MESSAGE


def test_b2_publication_service_refuses_a_lock_alias_before_any_task(lane_b, net):
    s = lane_b
    entry_id = "pub_queue_lock"
    _queue(s, entry_id, "77777777", text="ordinary abstract")
    managed = _lock_alias(s)
    service = _service(s)
    record, before = _record(s, entry_id), _tree(s)
    outcome = service.process_entry(entry_id, extraction_tasks="metadata")
    assert not os.path.exists(managed)
    assert net == [] and s.logged == []
    assert _record(s, entry_id) == record
    assert _tree(s) == before and s.rev.unchanged()
    assert outcome.final_status == "failed"
    assert outcome.response_markdown == MANAGED_EXPORT_MESSAGE


def test_b2_ordinary_publication_status_override_is_unchanged(lane_b, net):
    s = lane_b
    entry_id = "pub_queue_plain"
    _queue(s, entry_id, "88888888")
    response = _research_tools(s)["process_publication_entry"].invoke(
        {"entry_id": entry_id, "status_override": "failed", "error_message": "e"}
    )
    assert "Publication Status Updated" in response, response
    assert str(s.dm.publication_queue.get_entry(entry_id).status).endswith("failed")
    assert [k["tool_name"] for k in s.logged] == ["process_publication_entry"]


# --- B3 (lane B2 delta reviews) ----------------------------------------------------


def _queue_folder_through_a_revision(s):
    """The queue folder keeps its ordinary spelling but is a link into the revision;
    every name the queue update would check points back out to ordinary files."""
    import shutil

    folder = Path(s.dm.publication_queue.queue_file).parent
    safe = s.ws / "queue-safe"
    shutil.move(str(folder), str(safe))
    _link(folder, s.rev.directory)
    for name in (
        "publication_queue.jsonl",
        "publication_queue.lock",
        "publication_queue.jsonl.tmp",
        "backups",
    ):
        _link(s.rev.directory / name, safe / name)
    return safe


@pytest.mark.parametrize("mode", ["status_override", "service"])
def test_b3_queue_folder_alias_with_outward_names_is_refused(lane_b, net, mode):
    s = lane_b
    entry_id = "pub_queue_dir"
    _queue(s, entry_id, "66666666", text="ordinary abstract")
    record = _record(s, entry_id)
    _queue_folder_through_a_revision(s)
    tool = _research_tools(s)["process_publication_entry"]  # factory set-up first
    service = _service(s)
    before = _tree(s)
    if mode == "status_override":
        response = tool.invoke(
            {"entry_id": entry_id, "status_override": "failed", "error_message": "e"}
        )
    else:
        outcome = service.process_entry(entry_id, extraction_tasks="metadata")
        assert outcome.final_status == "failed"
        response = outcome.response_markdown
    assert _tree(s) == before  # no temporary file, the queue link is not replaced
    assert net == [] and s.logged == []
    assert _record(s, entry_id) == record
    assert response == MANAGED_EXPORT_MESSAGE


def _frozen_queue_clock(monkeypatch, module):
    from datetime import datetime as real

    class _Clock(real):
        @classmethod
        def now(cls, tz=None):
            return real(2026, 10, 28, 9, 0, 0, 123456)

    monkeypatch.setattr(module, "datetime", _Clock)
    return "20261028_090000_123456"


def test_b3_enabled_publication_backup_leaf_alias_is_refused(lane_b, net, monkeypatch):
    from lobster.core.queues import publication_queue as pq

    s = lane_b
    entry_id = "pub_queue_backup"
    _queue(s, entry_id, "55555555")
    record = _record(s, entry_id)
    tool = _research_tools(s)["process_publication_entry"]
    stamp = _frozen_queue_clock(monkeypatch, pq)
    monkeypatch.setenv("LOBSTER_ENABLE_QUEUE_BACKUPS", "true")
    backups = Path(s.dm.publication_queue.backup_dir)
    _link(backups / f"publication_queue_backup_{stamp}.jsonl", _rev_file(s))
    before = _tree(s)
    response = tool.invoke(
        {"entry_id": entry_id, "status_override": "failed", "error_message": "e"}
    )
    assert _tree(s) == before and s.rev.unchanged()
    assert _record(s, entry_id) == record and s.logged == []
    assert response == MANAGED_EXPORT_MESSAGE


def test_b3_enabled_publication_backup_is_written_as_before(lane_b, net, monkeypatch):
    from lobster.core.queues import publication_queue as pq

    s = lane_b
    entry_id = "pub_queue_backup"
    _queue(s, entry_id, "55555555")
    queue_file = Path(s.dm.publication_queue.queue_file)
    queued, queued_mtime = queue_file.read_bytes(), queue_file.stat().st_mtime_ns
    stamp = _frozen_queue_clock(monkeypatch, pq)
    monkeypatch.setenv("LOBSTER_ENABLE_QUEUE_BACKUPS", "true")
    response = _research_tools(s)["process_publication_entry"].invoke(
        {"entry_id": entry_id, "status_override": "failed", "error_message": "e"}
    )
    assert "Publication Status Updated" in response, response
    backup = (
        Path(s.dm.publication_queue.backup_dir)
        / f"publication_queue_backup_{stamp}.jsonl"
    )
    assert backup.is_file() and not backup.is_symlink()
    assert backup.read_bytes() == queued
    assert backup.stat().st_mtime_ns == queued_mtime  # metadata copied, as copy2
    assert [k["tool_name"] for k in s.logged] == ["process_publication_entry"]


class _Prepared:
    """A queue preparation result (the metadata fetch is outside this test)."""

    def __init__(self, entry):
        self.queue_entry = entry
        self.url_data = None


def _download_entry(entry_id, accession):
    from lobster.core.schemas.download_queue import DownloadQueueEntry, DownloadStatus

    return DownloadQueueEntry(
        entry_id=entry_id,
        dataset_id=accession,
        database="geo",
        priority=5,
        status=DownloadStatus.PENDING,
    )


@pytest.fixture
def prepare_service(monkeypatch):
    from lobster.services.data_access import queue_preparation_service as qps

    class _Service:
        def __init__(self, data_manager, *args, **kwargs):
            pass

        def detect_database(self, accession):
            return "geo"

        def list_supported_databases(self):
            return ["geo"]

        def prepare(self, accession, database=None, priority=5):
            return _Prepared(_download_entry(f"queue_{accession}", accession))

    monkeypatch.setattr(qps, "QueuePreparationService", _Service)


def test_b3_enabled_download_backup_leaf_alias_is_refused(
    lane_b, net, monkeypatch, prepare_service
):
    from lobster.core.queues import download_queue as dq

    s = lane_b
    queue = s.dm.download_queue
    queue.add_entry(_download_entry("queue_GSE1", "GSE1"))  # a non-empty queue
    entries = [e.model_dump() for e in queue.list_entries()]
    tool = _research_tools(s)["prepare_dataset_download"]
    stamp = _frozen_queue_clock(monkeypatch, dq)
    monkeypatch.setenv("LOBSTER_ENABLE_QUEUE_BACKUPS", "true")
    _link(Path(queue.backup_dir) / f"queue_backup_{stamp}.jsonl", _rev_file(s))
    s.logged.clear()
    before = _tree(s)
    response = tool.invoke({"accession": "GSE2"})
    assert _tree(s) == before and s.rev.unchanged()
    assert [e.model_dump() for e in queue.list_entries()] == entries
    assert s.logged == []
    assert response == MANAGED_EXPORT_MESSAGE


def test_b3_enabled_download_backup_is_written_as_before(
    lane_b, net, monkeypatch, prepare_service
):
    from lobster.core.queues import download_queue as dq

    s = lane_b
    queue = s.dm.download_queue
    queue.add_entry(_download_entry("queue_GSE1", "GSE1"))
    queued = Path(queue.queue_file).read_bytes()
    stamp = _frozen_queue_clock(monkeypatch, dq)
    monkeypatch.setenv("LOBSTER_ENABLE_QUEUE_BACKUPS", "true")
    response = _research_tools(s)["prepare_dataset_download"].invoke(
        {"accession": "GSE2"}
    )
    assert "## Download Prepared: GSE2" in response, response
    backup = Path(queue.backup_dir) / f"queue_backup_{stamp}.jsonl"
    assert backup.is_file() and not backup.is_symlink()
    assert backup.read_bytes() == queued
    assert {e.dataset_id for e in queue.list_entries()} == {"GSE1", "GSE2"}


# --- B4 (lane B3 delta review) -----------------------------------------------------


def _download_queue_folder_through_a_revision(s, queue):
    """``.lobster/queues -> revision``; the download queue's names point back out."""
    import shutil

    folder = Path(queue.queue_file).parent
    safe = s.ws / "queue-safe"
    shutil.move(str(folder), str(safe))
    _link(folder, s.rev.directory)
    for name in ("download_queue.jsonl", "download_queue.lock", "backups"):
        _link(s.rev.directory / name, safe / name)


@pytest.mark.parametrize("arrangement", ["backup_folder", "queue_folder"])
def test_b4_download_queue_folder_alias_is_refused(
    lane_b, net, monkeypatch, prepare_service, arrangement
):
    s = lane_b
    queue = s.dm.download_queue
    queue.add_entry(_download_entry("queue_GSE1", "GSE1"))  # a non-empty queue
    entries = [e.model_dump() for e in queue.list_entries()]
    tool = _research_tools(s)["prepare_dataset_download"]
    monkeypatch.setenv("LOBSTER_ENABLE_QUEUE_BACKUPS", "true")
    if arrangement == "backup_folder":
        backups = Path(queue.backup_dir)
        backups.rmdir()
        _link(backups, s.rev.directory)
    else:
        _download_queue_folder_through_a_revision(s, queue)
    s.logged.clear()
    before = _tree(s)
    response = tool.invoke({"accession": "GSE2"})
    assert _tree(s) == before and s.rev.unchanged()
    assert [e.model_dump() for e in queue.list_entries()] == entries
    assert s.logged == [] and response == MANAGED_EXPORT_MESSAGE


class _Abstract:
    def __init__(self, *args, **kwargs):
        pass

    def get_abstract(self, identifier):
        from types import SimpleNamespace

        return SimpleNamespace(
            pmid="44444444",
            doi=None,
            title="t",
            authors=["a"],
            journal="j",
            published="2026-01-01",
            abstract="abstract",
        )


def test_b4_new_publication_entry_with_a_backup_leaf_alias_is_refused(
    lane_b, net, monkeypatch
):
    from lobster.core.queues import publication_queue as pq

    s = lane_b
    _queue(s, "pub_queue_other", "55555555")  # a non-empty queue
    entries = [e.model_dump() for e in s.dm.publication_queue.list_entries()]
    monkeypatch.setattr(_research_module(), "AbstractProvider", _Abstract)
    tool = _research_tools(s)["process_publication_entry"]
    stamp = _frozen_queue_clock(monkeypatch, pq)
    monkeypatch.setenv("LOBSTER_ENABLE_QUEUE_BACKUPS", "true")
    backups = Path(s.dm.publication_queue.backup_dir)
    _link(backups / f"publication_queue_backup_{stamp}.jsonl", _rev_file(s))
    before = _tree(s)
    response = tool.invoke({"entry_id": "PMID:44444444"})
    assert _tree(s) == before and s.rev.unchanged()
    assert [e.model_dump() for e in s.dm.publication_queue.list_entries()] == entries
    assert s.logged == [] and response == MANAGED_EXPORT_MESSAGE


def _research_module():
    import importlib

    return importlib.import_module(RESEARCH_MODULE)


@pytest.fixture
def geo_validation(monkeypatch):
    """GEO metadata and its validation (both outside this test) for one accession."""
    from types import SimpleNamespace

    from lobster.services.data_access import geo_service
    from lobster.services.metadata import metadata_validation_service as mvs

    monkeypatch.setattr(
        geo_service.GEOService,
        "fetch_metadata_only",
        lambda self, identifier: ({"title": "t", "samples": {}}, None),
    )
    monkeypatch.setattr(
        mvs.MetadataValidationService,
        "validate_dataset_metadata",
        lambda self, **kwargs: SimpleNamespace(recommendation="proceed"),
    )
    monkeypatch.setattr(
        mvs.MetadataValidationService,
        "format_validation_report",
        lambda self, result, identifier: "## Validation report",
    )


@pytest.mark.parametrize("branch", ["stored_metadata", "validated"])
def test_b4_metadata_validation_queue_add_refusal_is_the_fixed_message(
    lane_b, net, monkeypatch, prepare_service, geo_validation, branch
):
    from lobster.core.queues import download_queue as dq

    s = lane_b
    queue = s.dm.download_queue
    queue.add_entry(_download_entry("queue_GSE1", "GSE1"))
    entries = [e.model_dump() for e in queue.list_entries()]
    if branch == "stored_metadata":
        s.dm.metadata_store["GSE3"] = {"metadata": {"title": "t"}}
    tool = _research_tools(s)["validate_dataset_metadata"]
    stamp = _frozen_queue_clock(monkeypatch, dq)
    monkeypatch.setenv("LOBSTER_ENABLE_QUEUE_BACKUPS", "true")
    _link(Path(queue.backup_dir) / f"queue_backup_{stamp}.jsonl", _rev_file(s))
    s.logged.clear()
    before = _tree(s)
    response = tool.invoke({"identifier": "GSE3", "required_fields": "tissue"})
    assert _tree(s) == before and s.rev.unchanged()
    assert [e.model_dump() for e in queue.list_entries()] == entries
    assert response == MANAGED_EXPORT_MESSAGE


@pytest.mark.parametrize("branch", ["stored_metadata", "validated"])
def test_b4_metadata_validation_ordinary_queue_add_is_unchanged(
    lane_b, net, monkeypatch, prepare_service, geo_validation, branch
):
    s = lane_b
    if branch == "stored_metadata":
        s.dm.metadata_store["GSE3"] = {"metadata": {"title": "t"}}
    monkeypatch.setenv("LOBSTER_ENABLE_QUEUE_BACKUPS", "true")
    response = _research_tools(s)["validate_dataset_metadata"].invoke(
        {"identifier": "GSE3", "required_fields": "tissue"}
    )
    assert response != MANAGED_EXPORT_MESSAGE and "Error" not in response, response
    assert [e.dataset_id for e in s.dm.download_queue.list_entries()] == ["GSE3"]


@pytest.mark.parametrize("module", ["publication_queue", "download_queue"])
def test_b4_backup_copy_failure_leaves_no_partial_file_or_descriptor(
    tmp_path, monkeypatch, module
):
    import importlib

    queues = importlib.import_module(f"lobster.core.queues.{module}")
    source = tmp_path / "queue.jsonl"
    source.write_text('{"x": 1}\n')
    target = tmp_path / "backup.jsonl"

    def broken(src, dst, *args, **kwargs):
        dst.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(queues.shutil, "copyfileobj", broken)
    fds = sorted(os.listdir("/dev/fd"))
    with pytest.raises(OSError, match="disk full"):
        queues._copy_to_new_file(source, target)
    assert not os.path.lexists(target)
    assert sorted(os.listdir("/dev/fd")) == fds


@pytest.mark.parametrize("module", ["publication_queue", "download_queue"])
def test_b4_unreadable_backup_source_creates_nothing(tmp_path, module):
    import importlib

    queues = importlib.import_module(f"lobster.core.queues.{module}")
    target = tmp_path / "backup.jsonl"
    fds = sorted(os.listdir("/dev/fd"))
    with pytest.raises(FileNotFoundError):
        queues._copy_to_new_file(tmp_path / "missing.jsonl", target)
    assert not os.path.lexists(target)
    assert sorted(os.listdir("/dev/fd")) == fds


# --- B5 (lane B4 delta review): every queue write is checked by the queue itself ---

DATA_MODULE = "lobster.agents.data_expert.data_expert"


class _DownloadService:
    """The download service the orchestrator routes to (each call is recorded)."""

    def __init__(self):
        self.calls = []

    def download_dataset(self, entry, strategy_override=None):
        self.calls.append(entry.entry_id)
        return ad.AnnData(X=np.ones((3, 2))), {"strategy_used": "auto"}, None


@pytest.fixture
def download_service(monkeypatch):
    from lobster.tools.download_orchestrator import DownloadOrchestrator

    service = _DownloadService()
    monkeypatch.setattr(
        DownloadOrchestrator, "get_service_for_database", lambda self, db: service
    )
    return service


def _data_tools(s):
    from lobster.agents.data_expert.data_expert import data_expert
    from lobster.tools.download_orchestrator import DownloadOrchestrator

    tools = factory_tools(data_expert, DATA_MODULE, s.dm)
    DownloadOrchestrator(s.dm)  # service set-up (its caches) before any snapshot
    return tools


def _queues_folder_is_a_revision(s):
    """Before the queues are first used: ``.lobster/queues -> revision``."""
    lobster = s.ws / ".lobster"
    lobster.mkdir(exist_ok=True)
    assert not (lobster / "queues").exists()
    _link(lobster / "queues", s.rev.directory)


def _pending(s):
    queue = s.dm.download_queue
    queue.add_entry(_download_entry("queue_GSE1", "GSE1"))
    return queue


def _status(queue):
    status = queue.get_entry("queue_GSE1").status
    return getattr(status, "value", status)


def _dangling_lock_alias(s, queue):
    lock = Path(queue._file_lock_path)
    if lock.exists() or lock.is_symlink():
        lock.unlink()
    target = s.rev.directory / "new.lock"
    _link(lock, target)
    return target


@pytest.mark.parametrize("which", ["download_queue", "publication_queue"])
def test_b5_first_queue_use_through_a_revision_creates_nothing(lane_b, which):
    from lobster.core.figures.errors import ManagedPathError

    s = lane_b
    _queues_folder_is_a_revision(s)
    before = _tree(s)
    with pytest.raises(ManagedPathError):
        getattr(s.dm, which)
    assert _tree(s) == before and s.rev.unchanged()


def test_b5_first_queue_use_in_download_preparation_is_the_fixed_message(
    lane_b, net, prepare_service
):
    s = lane_b
    _queues_folder_is_a_revision(s)
    tool = _research_tools(s)["prepare_dataset_download"]
    s.logged.clear()
    before = _tree(s)
    response = tool.invoke({"accession": "GSE2"})
    assert _tree(s) == before and s.rev.unchanged()
    assert s.logged == [] and response == MANAGED_EXPORT_MESSAGE


@pytest.mark.parametrize("arrangement", ["queue_folder", "dangling_lock"])
def test_b5_queue_download_with_backups_off_is_refused_before_the_download(
    lane_b, net, monkeypatch, download_service, arrangement
):
    s = lane_b
    monkeypatch.delenv("LOBSTER_ENABLE_QUEUE_BACKUPS", raising=False)
    queue = _pending(s)
    tool = _data_tools(s)["execute_download_from_queue"]
    lock_target = None
    if arrangement == "queue_folder":
        _download_queue_folder_through_a_revision(s, queue)
    else:
        lock_target = _dangling_lock_alias(s, queue)
    s.logged.clear()
    before = _tree(s)
    response = tool.invoke({"entry_id": "queue_GSE1"})
    assert response == MANAGED_EXPORT_MESSAGE
    assert _tree(s) == before and s.rev.unchanged()
    assert lock_target is None or not os.path.lexists(lock_target)
    assert download_service.calls == [] and s.logged == []
    assert _status(queue) == "pending" and "queue_GSE1" not in "".join(s.dm.modalities)


def test_b5_queue_download_backup_collision_is_the_fixed_message(
    lane_b, net, monkeypatch, download_service, caplog
):
    from lobster.core.queues import download_queue as dq

    s = lane_b
    queue = _pending(s)
    tool = _data_tools(s)["execute_download_from_queue"]
    stamp = _frozen_queue_clock(monkeypatch, dq)
    monkeypatch.setenv("LOBSTER_ENABLE_QUEUE_BACKUPS", "true")
    _link(Path(queue.backup_dir) / f"queue_backup_{stamp}.jsonl", _rev_file(s))
    s.logged.clear()
    before = _tree(s)
    with caplog.at_level("INFO"):
        response = tool.invoke({"entry_id": "queue_GSE1"})
    assert response == MANAGED_EXPORT_MESSAGE  # no "Download Failed", no retry advice
    assert _tree(s) == before and s.rev.unchanged()
    assert download_service.calls == [] and _status(queue) == "pending"
    assert [r for r in caplog.records if "failed" in r.getMessage().lower()] == []


def test_b5_ordinary_queue_download_is_unchanged(
    lane_b, net, monkeypatch, download_service
):
    monkeypatch.delenv("LOBSTER_ENABLE_QUEUE_BACKUPS", raising=False)
    s = lane_b
    queue = _pending(s)
    response = _data_tools(s)["execute_download_from_queue"].invoke(
        {"entry_id": "queue_GSE1"}
    )
    assert "Download completed successfully" in response, response
    assert download_service.calls == ["queue_GSE1"]
    assert _status(queue) == "completed" and "geo_GSE1" in s.dm.modalities


@pytest.mark.parametrize("module", ["download_queue", "publication_queue"])
@pytest.mark.parametrize(
    "operation", ["add_entry", "update_status", "remove_entry", "clear_queue"]
)
def test_b5_every_queue_change_checks_the_lock_first(lane_b, module, operation):
    from lobster.core.figures.errors import ManagedPathError
    from lobster.core.schemas.download_queue import DownloadStatus
    from lobster.core.schemas.publication_queue import (
        PublicationQueueEntry,
        PublicationStatus,
    )

    s = lane_b
    queue = getattr(s.dm, module)
    if module == "download_queue":
        first, second = (_download_entry(f"queue_GSE{i}", f"GSE{i}") for i in (1, 2))
        status = DownloadStatus.IN_PROGRESS
    else:
        first, second = (
            PublicationQueueEntry(entry_id=f"pub_{i}", pmid=f"1000000{i}", title="t")
            for i in (1, 2)
        )
        status = PublicationStatus.METADATA_EXTRACTED
    queue.add_entry(first)
    target = _dangling_lock_alias(s, queue)
    before = _tree(s)
    with pytest.raises(ManagedPathError):
        if operation == "add_entry":
            queue.add_entry(second)
        elif operation == "update_status":
            queue.update_status(first.entry_id, status)
        elif operation == "remove_entry":
            queue.remove_entry(first.entry_id)
        else:
            queue.clear_queue()
    assert _tree(s) == before and s.rev.unchanged()
    assert not os.path.lexists(target)


@pytest.mark.parametrize("module", ["download_queue", "publication_queue"])
def test_b5_a_named_guard_workspace_is_checked_at_any_location(lane_b, module):
    import importlib

    from lobster.core.figures.errors import ManagedPathError

    s = lane_b
    cls = getattr(
        importlib.import_module(f"lobster.core.queues.{module}"),
        "DownloadQueue" if module == "download_queue" else "PublicationQueue",
    )
    before = _tree(s)
    with pytest.raises(ManagedPathError):
        cls(s.rev.directory / "elsewhere" / "q.jsonl", guard_workspace=s.ws)
    assert _tree(s) == before and s.rev.unchanged()


@pytest.mark.parametrize("module", ["download_queue", "publication_queue"])
def test_b5_a_queue_at_another_location_is_not_checked(tmp_path, module):
    """Residual, named: outside ``<workspace>/.lobster/queues/`` with no named
    workspace, the queue has no workspace to check against and behaves as before."""
    import importlib

    queues = importlib.import_module(f"lobster.core.queues.{module}")
    cls = (
        queues.DownloadQueue if module == "download_queue" else queues.PublicationQueue
    )
    queue = cls(tmp_path / "elsewhere" / f"{module}.jsonl")
    assert queue.guard_workspace is None
    assert queue.queue_file.is_file() and Path(queue.backup_dir).is_dir()
    assert queue.clear_queue() == 0


def _queue_class(module):
    import importlib

    queues = importlib.import_module(f"lobster.core.queues.{module}")
    return (
        queues.DownloadQueue if module == "download_queue" else queues.PublicationQueue
    )


@pytest.mark.parametrize("module", ["download_queue", "publication_queue"])
def test_b6_a_relative_spelling_of_the_standard_queue_is_checked(
    lane_b, monkeypatch, module
):
    """``queues/<file>`` from inside ``<workspace>/.lobster`` is the standard queue."""
    from lobster.core.figures.errors import ManagedPathError

    s = lane_b
    _queues_folder_is_a_revision(s)
    monkeypatch.chdir(s.ws / ".lobster")
    before = _tree(s)
    with pytest.raises(ManagedPathError) as refused:
        _queue_class(module)(Path("queues") / f"{module}.jsonl")
    assert "Error: " + str(refused.value) == MANAGED_EXPORT_MESSAGE
    assert _tree(s) == before and s.rev.unchanged()


@pytest.mark.parametrize("module", ["download_queue", "publication_queue"])
def test_b6_an_ordinary_relative_standard_queue_is_unchanged(
    lane_b, monkeypatch, module
):
    s = lane_b
    lobster = s.ws / ".lobster"
    lobster.mkdir(exist_ok=True)
    monkeypatch.chdir(lobster)
    queue = _queue_class(module)(Path("queues") / f"{module}.jsonl")
    assert queue.guard_workspace == Path.cwd().parent
    assert (lobster / "queues" / f"{module}.jsonl").is_file()
    assert (lobster / "queues" / "backups").is_dir()
    assert queue.clear_queue() == 0


META_MODULE = "lobster.agents.metadata_assistant.metadata_assistant"
HAS_META = find_spec("lobster.agents.metadata_assistant") is not None
needs_meta = pytest.mark.skipif(not HAS_META, reason="lobster-metadata not installed")
OUTPUT_KEY = "aggregated_filtered_samples"
STOP_NOTE = "Some queue entries may already have been updated."
DISPATCH_LOGS = ("Using parallel processing", "Starting batch processing")


def _metadata_queue_tool(s):
    from lobster.agents.metadata_assistant.metadata_assistant import (
        metadata_assistant,
    )

    return factory_tools(metadata_assistant, META_MODULE, s.dm)[
        "process_metadata_queue"
    ]


def _pending_publications(s, count=1, keys=()):
    """Ordinary PENDING entries ``pub_1..``; without workspace keys nothing is read."""
    from lobster.core.schemas.publication_queue import PublicationQueueEntry

    queue = s.dm.publication_queue
    for i in range(1, count + 1):
        queue.add_entry(
            PublicationQueueEntry(
                entry_id=f"pub_{i}",
                pmid=f"{10000000 + i}",
                title="t",
                workspace_metadata_keys=list(keys),
            )
        )
    return queue


def _states(queue):
    return {
        e.entry_id: (
            getattr(e.status, "value", e.status),
            getattr(e.handoff_status, "value", e.handoff_status),
        )
        for e in queue.list_entries()
    }


def _changed(before, after):
    return sum(1 for entry_id, state in before.items() if after[entry_id] != state)


def _executor_threads():
    import threading

    return {t for t in threading.enumerate() if t.name.startswith("ThreadPoolExecutor")}


class _Progress:
    """The progress display (UI boundary). ``on_start``/``on_complete`` may place an
    alias while the workers run, i.e. after the tool's own check."""

    def __init__(self, on_start=None, on_complete=None):
        self.on_start, self.on_complete = on_start, on_complete
        self.started, self.placed = [], []
        # The workers call in from their own threads: check and place atomically.
        self._lock = threading.Lock()

    def worker_start(self, worker_id, title):
        with self._lock:
            self.started.append(worker_id)
            if self.on_start and not self.placed:
                self.placed.append(self.on_start())

    def worker_complete(self, worker_id, status, elapsed):
        with self._lock:
            if self.on_complete and not self.placed:
                placed = self.on_complete()
                if placed is not None:
                    self.placed.append(placed)

    def worker_done(self, worker_id):
        pass


def _fake_progress(monkeypatch, progress):
    import contextlib
    import importlib

    @contextlib.contextmanager
    def parallel_workers_progress(workers, total):
        yield progress

    monkeypatch.setattr(
        importlib.import_module(META_MODULE),
        "parallel_workers_progress",
        parallel_workers_progress,
    )
    return progress


def _process(tool, workers):
    args = {"status_filter": "pending"}
    if workers:
        args["parallel_workers"] = workers
    return tool.invoke(args)


def _dispatch_logs(caplog):
    return [r for r in caplog.records if r.getMessage().startswith(DISPATCH_LOGS)]


def _output_alias(s):
    """``metadata/<output>.json`` points to an absent file inside the revision."""
    target = s.rev.directory / "new.json"
    _link(s.ws / "metadata" / f"{OUTPUT_KEY}.json", target)
    return target


@needs_meta
@pytest.mark.parametrize("workers", [2, None], ids=["parallel", "sequential"])
@pytest.mark.parametrize("arrange", ["lock", "output"])
def test_b6_a_refused_queue_run_dispatches_nothing(
    lane_b, monkeypatch, caplog, workers, arrange
):
    import logging

    s = lane_b
    tool = _metadata_queue_tool(s)
    queue = _pending_publications(s)
    progress = _fake_progress(monkeypatch, _Progress())
    target = _dangling_lock_alias(s, queue) if arrange == "lock" else _output_alias(s)
    state, before = _states(queue), _tree(s)
    with caplog.at_level(logging.INFO):
        assert _process(tool, workers) == MANAGED_EXPORT_MESSAGE
    assert progress.started == [] and _dispatch_logs(caplog) == []
    assert _tree(s) == before and s.rev.unchanged()
    assert not os.path.lexists(target)
    assert _states(queue) == state and OUTPUT_KEY not in s.dm.metadata_store


@needs_meta
def test_b6_a_refusal_inside_the_parallel_workers_waits_for_them(lane_b, monkeypatch):
    s = lane_b
    tool = _metadata_queue_tool(s)
    queue = _pending_publications(s)
    progress = _fake_progress(
        monkeypatch, _Progress(on_start=lambda: _dangling_lock_alias(s, queue))
    )
    state, threads = _states(queue), _executor_threads()
    # Entries were dispatched: the reply says updates may exist (here none do).
    assert _process(tool, 2) == f"{MANAGED_EXPORT_MESSAGE}\n{STOP_NOTE}"
    assert _executor_threads() <= threads  # every worker has finished
    assert len(progress.placed) == 1 and not os.path.lexists(progress.placed[0])
    assert s.rev.unchanged()
    assert _states(queue) == state and OUTPUT_KEY not in s.dm.metadata_store


@needs_meta
def test_b6_a_late_parallel_refusal_notes_the_earlier_updates(lane_b, monkeypatch):
    """45 entries, 2 workers: a worker writes its first 20 updates, then the lock is
    aliased; the remaining updates are refused (completed work is not undone)."""
    s = lane_b
    tool = _metadata_queue_tool(s)
    queue = _pending_publications(s, count=45)
    before = _states(queue)

    def after_the_first_write():
        if _changed(before, _states(queue)):
            return _dangling_lock_alias(s, queue)
        return None

    progress = _fake_progress(monkeypatch, _Progress(on_complete=after_the_first_write))
    threads = _executor_threads()
    response = _process(tool, 2)
    assert len(progress.placed) == 1 and not os.path.lexists(progress.placed[0])
    updated = _changed(before, _states(queue))
    assert 0 < updated < 45
    assert response == f"{MANAGED_EXPORT_MESSAGE}\n{STOP_NOTE}"
    assert _executor_threads() <= threads
    assert s.rev.unchanged() and OUTPUT_KEY not in s.dm.metadata_store


@needs_meta
def test_b6_a_late_sequential_refusal_records_no_failure(lane_b, monkeypatch, caplog):
    """The second entry's queue update is refused: no failure status is attempted."""
    import logging

    from lobster.services.data_access.workspace_content_service import (
        WorkspaceContentService,
    )

    s = lane_b
    tool = _metadata_queue_tool(s)
    queue = _pending_publications(s, count=2, keys=["sra_PRJ1_samples"])
    before, placed = _states(queue), []

    def read_content(self, identifier, content_type=None, **kwargs):
        # The workspace read (boundary): the second entry's read places the alias.
        if len(placed) == 0 and _changed(before, _states(queue)):
            placed.append(_dangling_lock_alias(s, queue))
        return None

    monkeypatch.setattr(WorkspaceContentService, "read_content", read_content)
    with caplog.at_level(logging.INFO):
        response = _process(tool, None)
    assert response == f"{MANAGED_EXPORT_MESSAGE}\n{STOP_NOTE}"
    assert len(placed) == 1 and not os.path.lexists(placed[0])
    assert _states(queue) == {**before, "pub_1": ("completed", "metadata_complete")}
    assert not [
        r for r in caplog.records if "Failed to process entry" in r.getMessage()
    ]
    assert s.rev.unchanged()


@needs_meta
@pytest.mark.parametrize("workers", [2, None], ids=["parallel", "sequential"])
def test_b6_ordinary_queue_processing_is_unchanged(lane_b, workers):
    s = lane_b
    tool = _metadata_queue_tool(s)
    queue = _pending_publications(s)
    response = _process(tool, workers)
    assert MANAGED_EXPORT_MESSAGE not in response and STOP_NOTE not in response
    if workers:
        assert response.startswith("## Queue Processing Complete (Parallel Mode)")
        assert "**Successful**: 1" in response
        assert _states(queue) == {"pub_1": ("completed", "metadata_complete")}
    else:  # an entry without workspace keys is skipped, as before
        assert _states(queue) == {"pub_1": ("pending", "not_ready")}
    assert s.rev.unchanged()
