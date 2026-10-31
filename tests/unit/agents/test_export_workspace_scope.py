"""Tool exports stay inside the session workspace (export write policy, 1.2.1).

A model-supplied export path, or a modality name used in a file name, is checked
before anything is created: a target outside the workspace (absolute, '..', or a
link leading out) is refused with one fixed message and nothing is written, unless
the person listed the folder in LOBSTER_EXPORT_ROOTS. Ordinary exports under
exports/ are covered by test_export_managed_guard.py and stay unchanged.
"""

import os
from importlib.util import find_spec
from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import lobster.core.export_scope as scope
from lobster.core.export_scope import (
    WORKSPACE_EXPORT_MESSAGE,
    InvalidModalityNameError,
    WorkspaceExportError,
)
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.testing.provenance_replay import factory_tools
from tests.unit.services.test_scvi_export_guard import fake_scvi  # noqa: F401
from tests.unit.services.test_native_export_sink_guard import indra  # noqa: F401

HAS_TX = find_spec("lobster.agents.transcriptomics") is not None
HAS_ML = find_spec("lobster.agents.machine_learning") is not None
needs_tx = pytest.mark.skipif(not HAS_TX, reason="transcriptomics package absent")
needs_ml = pytest.mark.skipif(not HAS_ML, reason="ML package absent")

DE_MODULE = "lobster.agents.transcriptomics.de_analysis_expert"
ANNOTATION_MODULE = "lobster.agents.transcriptomics.annotation_expert"
ML_MODULE = "lobster.agents.machine_learning.machine_learning_expert"
RESULTS_KEY = "de_results_condition_sample_b_vs_sample_a"
#: A modality name that leads out of the workspace (data/../../outside/evil).
ESCAPING_NAME = "../../outside/evil"
#: The auto-save summary line for one refused target outside the workspace.
NOT_SAVED_OUTSIDE = "Not saved: 1 modalities refused (outside the workspace)"


def _modality():
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
    return adata, results


class Session:
    def __init__(self, tmp_path):
        self.root = tmp_path
        self.cwd = tmp_path / "cwd"
        self.outside = tmp_path / "outside"
        self.cwd.mkdir()
        self.outside.mkdir()
        self.dm = DataManagerV2(workspace_path=tmp_path / "ws")
        self.adata, self.results = _modality()
        self.dm.store_modality("de_results", self.adata)
        self.dm.exports_dir.mkdir(parents=True, exist_ok=True)
        os.symlink(self.outside, self.dm.exports_dir / "door")  # a link leading out

    def nothing_outside(self):
        assert sorted(p.name for p in self.root.iterdir()) == ["cwd", "outside", "ws"]
        assert list(self.outside.iterdir()) == []
        assert list(self.cwd.iterdir()) == []

    def targets(self, name):
        """Spellings, relative to exports/, of a file outside the workspace."""
        return {
            "absolute": str(self.outside / name),
            "dotdot": f"../../outside/{name}",
            "dotdot-root": f"../../{name}",
            "link": f"door/{name}",
        }


TARGET_IDS = ["absolute", "dotdot", "dotdot-root", "link"]


@pytest.fixture(autouse=True)
def no_roots(monkeypatch):
    monkeypatch.setattr(scope, "_EXPORT_ROOTS", ())


@pytest.fixture
def s(tmp_path, monkeypatch):
    session = Session(tmp_path)
    monkeypatch.chdir(session.cwd)
    return session


def _de_science(monkeypatch, results):
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


def _de_tools(s):
    from lobster.agents.transcriptomics.de_analysis_expert import de_analysis_expert

    return factory_tools(de_analysis_expert, DE_MODULE, s.dm)


# ------------------------------------------------------------- D1/D2 DE table exports


@needs_tx
@pytest.mark.parametrize("which", TARGET_IDS)
@pytest.mark.parametrize(
    "tool_name", ["export_de_results", "extract_and_export_de_results"]
)
@pytest.mark.parametrize("fmt", ["csv", "xlsx"])
def test_de_exports_refuse_a_target_outside_the_workspace(s, tool_name, which, fmt):
    args = {
        "modality_name": "de_results",
        "output_format": fmt,
        "output_path": s.targets(f"x.{fmt}")[which],
    }
    if tool_name == "extract_and_export_de_results":
        args["shrink_lfc"] = False
    response = _de_tools(s)[tool_name].invoke(args)
    assert response == WORKSPACE_EXPORT_MESSAGE
    assert str(s.root) not in response
    s.nothing_outside()


@needs_tx
@pytest.mark.parametrize(
    "tool_name", ["export_de_results", "extract_and_export_de_results"]
)
def test_de_export_to_a_listed_root_is_written(s, monkeypatch, tool_name):
    monkeypatch.setattr(scope, "_EXPORT_ROOTS", (s.outside,))
    target = s.outside / "t.csv"
    args = {"modality_name": "de_results", "output_path": str(target)}
    if tool_name == "extract_and_export_de_results":
        args["shrink_lfc"] = False
    response = _de_tools(s)[tool_name].invoke(args)
    assert target.is_file(), response
    assert response.split("- Output: ", 1)[1].splitlines()[0] == str(target.resolve())


# ------------------------------------------------------------- D3 annotation mapping


@needs_tx
@pytest.mark.parametrize("which", TARGET_IDS)
@pytest.mark.parametrize("fmt", ["json", "csv"])
def test_annotation_export_refuses_a_target_outside_the_workspace(s, which, fmt):
    from lobster.agents.transcriptomics.annotation_expert import annotation_expert

    tools = factory_tools(annotation_expert, ANNOTATION_MODULE, s.dm)
    response = tools["export_annotation_mapping"].invoke(
        {
            "modality_name": "de_results",
            "format": fmt,
            "output_filename": s.targets(f"m.{fmt}")[which],
        }
    )
    assert response == WORKSPACE_EXPORT_MESSAGE
    s.nothing_outside()


@needs_tx
def test_annotation_export_to_a_listed_root_is_written(s, monkeypatch):
    from lobster.agents.transcriptomics.annotation_expert import annotation_expert

    monkeypatch.setattr(scope, "_EXPORT_ROOTS", (s.outside,))
    tools = factory_tools(annotation_expert, ANNOTATION_MODULE, s.dm)
    target = s.outside / "m.json"
    response = tools["export_annotation_mapping"].invoke(
        {
            "modality_name": "de_results",
            "format": "json",
            "output_filename": str(target),
        }
    )
    assert target.is_file(), response


# ------------------------------------------------------------- D4/N3 ML export folder


@needs_ml
@pytest.mark.parametrize("which", TARGET_IDS)
@pytest.mark.parametrize("fmt", ["numpy", "csv"])
def test_ml_export_refuses_a_folder_outside_the_workspace(s, which, fmt):
    from lobster.agents.machine_learning.machine_learning_expert import (
        machine_learning_expert,
    )

    tools = factory_tools(machine_learning_expert, ML_MODULE, s.dm)
    response = tools["export_for_ml_framework"].invoke(
        {
            "modality_name": "de_results",
            "format": fmt,
            "include_labels": False,
            "output_dir": s.targets("ml")[which],
        }
    )
    assert WORKSPACE_EXPORT_MESSAGE in response, response
    s.nothing_outside()


@needs_ml
def test_ml_export_refuses_a_modality_name_leading_out(s):
    from lobster.agents.machine_learning.machine_learning_expert import (
        machine_learning_expert,
    )

    s.dm.modalities["../../../outside/evil"] = s.adata.copy()  # direct registration
    tools = factory_tools(machine_learning_expert, ML_MODULE, s.dm)
    response = tools["export_for_ml_framework"].invoke(
        {
            "modality_name": "../../../outside/evil",
            "format": "csv",
            "include_labels": False,
            "output_dir": "ml",
        }
    )
    assert WORKSPACE_EXPORT_MESSAGE in response, response
    s.nothing_outside()


@needs_ml
def test_ml_export_to_a_listed_root_is_written(s, monkeypatch):
    from lobster.agents.machine_learning.machine_learning_expert import (
        machine_learning_expert,
    )

    monkeypatch.setattr(scope, "_EXPORT_ROOTS", (s.outside,))
    tools = factory_tools(machine_learning_expert, ML_MODULE, s.dm)
    response = tools["export_for_ml_framework"].invoke(
        {
            "modality_name": "de_results",
            "format": "csv",
            "include_labels": False,
            "output_dir": str(s.outside / "ml"),
        }
    )
    assert (s.outside / "ml" / "de_results_ml_data.csv").is_file(), response


# ------------------------------------------------------------- N1/N2 DE run saves


DE_ARGS = {
    "groupby": "condition",
    "group1": "sample_a",
    "group2": "sample_b",
    "save_result": True,
}


@needs_tx
def test_pseudobulk_de_refuses_a_modality_name_leading_out(s, monkeypatch):
    science = _de_science(monkeypatch, s.results)
    s.dm.modalities[ESCAPING_NAME] = s.adata.copy()  # direct registration
    modalities = list(s.dm.list_modalities())
    response = _de_tools(s)["run_differential_expression"].invoke(
        {**DE_ARGS, "modality_name": ESCAPING_NAME}
    )
    assert response == WORKSPACE_EXPORT_MESSAGE
    science.run_pydeseq2_from_pseudobulk.assert_not_called()
    assert list(s.dm.list_modalities()) == modalities
    s.nothing_outside()


@needs_tx
def test_formula_de_refuses_a_modality_name_leading_out(s, monkeypatch):
    science = _de_science(monkeypatch, s.results)
    s.dm.modalities[ESCAPING_NAME] = s.adata.copy()
    response = _de_tools(s)["run_de_with_formula"].invoke(
        {
            "pseudobulk_modality": ESCAPING_NAME,
            "formula": "~condition",
            "contrast": ["condition", "sample_b", "sample_a"],
        }
    )
    assert response == WORKSPACE_EXPORT_MESSAGE
    science.validate_experimental_design.assert_not_called()
    science.run_pydeseq2_from_pseudobulk.assert_not_called()
    s.nothing_outside()


# ------------------------------------------------------------- N5 pathway CSV


def test_pathway_csv_refuses_a_modality_name_leading_out(tmp_path, monkeypatch):
    from lobster.services.analysis import pathway_enrichment_bridge_service as svc

    ws, outside = tmp_path / "ws", tmp_path / "outside"
    ws.mkdir()
    outside.mkdir()
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
    service = svc.PathwayEnrichmentBridgeService(timeout=1)
    with pytest.raises(WorkspaceExportError):
        service._store_results(adata, df, "../outside/evil", "sel", stats)
    assert list(outside.iterdir()) == []
    assert "pathway_enrichment" not in adata.uns


@needs_ml
def test_pathway_tool_refuses_a_name_leading_out_before_the_request(s, request):
    from tests.unit.services.test_native_export_sink_guard import (
        _pathway_tool,
        _selected_adata,
    )

    requests_sent = request.getfixturevalue("indra")  # records any request
    adata = _selected_adata()
    s.dm.modalities[ESCAPING_NAME] = adata  # direct registration
    response = _pathway_tool(s.dm).invoke({"modality_name": ESCAPING_NAME})
    assert response == WORKSPACE_EXPORT_MESSAGE
    assert requests_sent == []  # refused before the request
    assert "pathway_enrichment" not in adata.uns
    s.nothing_outside()


# ------------------------------------------------------------- DM names and N6 saves


@pytest.mark.parametrize("name", ["a/b", "../x", "..", "a\\b", "a\x00b", ""])
def test_store_modality_refuses_a_name_that_is_not_one_plain_name(s, name):
    before = list(s.dm.list_modalities())
    with pytest.raises(InvalidModalityNameError):
        s.dm.store_modality(name, s.adata.copy())
    assert list(s.dm.list_modalities()) == before


@pytest.mark.parametrize("name", ["a/b", "../x", "."])
def test_load_modality_refuses_a_name_that_is_not_one_plain_name(s, name):
    before = list(s.dm.list_modalities())
    with pytest.raises(InvalidModalityNameError):
        s.dm.load_modality(name, s.adata.copy(), adapter="transcriptomics_single_cell")
    assert list(s.dm.list_modalities()) == before


@pytest.mark.parametrize("name", ["my data", "pbmc3k.v2", "geo_gse12345_filtered"])
def test_ordinary_names_are_stored(s, name):
    s.dm.store_modality(name, s.adata.copy())
    assert name in s.dm.list_modalities()


@pytest.mark.parametrize(
    "path", ["../../outside/x.h5ad", "../../x.h5ad", "../../ws_other/x.h5ad"]
)
def test_save_modality_refuses_a_relative_path_leading_out(s, path):
    with pytest.raises(WorkspaceExportError):
        s.dm.save_modality("de_results", path)
    s.nothing_outside()


def test_save_modality_refuses_a_relative_path_through_a_link_leading_out(s):
    s.dm.data_dir.mkdir(parents=True, exist_ok=True)
    os.symlink(s.outside, s.dm.data_dir / "door")
    with pytest.raises(WorkspaceExportError):
        s.dm.save_modality("de_results", "door/x.h5ad")
    s.nothing_outside()


def test_save_modality_keeps_an_absolute_user_path(s):
    """A person's absolute path (CLI/SDK) is an explicit user path: unchanged."""
    target = s.outside / "kept.h5ad"
    returned = s.dm.save_modality("de_results", str(target))
    assert target.is_file()
    assert returned == str(target)


# ------------------------------------------------------------- N4 scVI model folder


@needs_ml
def test_scvi_refuses_a_model_folder_leading_out(s, request):
    from tests.unit.services.test_scvi_export_guard import _tool

    fake = request.getfixturevalue("fake_scvi")

    s.dm.modalities[ESCAPING_NAME] = s.adata.copy()  # direct registration
    response = _tool(s.dm).invoke({"modality_name": ESCAPING_NAME, "max_epochs": 1})
    assert response == WORKSPACE_EXPORT_MESSAGE
    assert fake.setup_calls == [] and fake.instances == []
    assert "X_scvi" not in s.dm.modalities[ESCAPING_NAME].obsm
    s.nothing_outside()


@needs_ml
def test_scvi_service_refuses_a_save_path_outside_the_workspace(s):
    from lobster.services.analysis.scvi_embedding_service import ScviEmbeddingService

    with pytest.raises(WorkspaceExportError):
        ScviEmbeddingService().train_scvi_embedding(
            s.adata.copy(),
            save_path=str(s.outside / "m"),
            workspace_path=s.dm.workspace_path,
        )
    s.nothing_outside()


# ------------------------------------------------------------- auto-save of a name leading out


@pytest.mark.parametrize("which", ["dotdot", "absolute"])
def test_autosave_skips_a_name_leading_out_and_saves_the_rest(s, which):
    """A name that arrived by direct registration is not auto-saved outside; the
    other modalities are still saved (one refusal does not stop the run)."""
    name = ESCAPING_NAME if which == "dotdot" else str(s.outside / "abs")
    s.dm.modalities[name] = s.adata.copy()  # direct registration
    s.dm.mark_modality_dirty(name)
    s.dm.store_modality("plain", s.adata.copy())
    result = s.dm.auto_save_state()
    assert (s.dm.data_dir / "plain_autosave.h5ad").is_file()
    assert result.count(NOT_SAVED_OUTSIDE) == 1
    assert name in s.dm._modality_dirty  # not saved: stays modified
    s.nothing_outside()


def test_autosave_skips_a_spilled_modality_whose_name_leads_out(tmp_path, monkeypatch):
    from tests.unit.core.test_dm_save_guard import (
        SPILL_SHAPE,
        _adata,
        _session,
    )

    monkeypatch.setattr(  # admit the name, as a direct registration would
        "lobster.core.runtime.data_manager.check_modality_name", lambda name: None
    )
    sp = _session(tmp_path, monkeypatch, budget_of=1.5)
    outside = tmp_path / "outside"
    outside.mkdir()
    sp.dm.store_modality(name=ESCAPING_NAME, adata=_adata(0, *SPILL_SHAPE))
    sp.dm.store_modality(name="other", adata=_adata(1, *SPILL_SHAPE))
    assert ESCAPING_NAME in sp.dm._modality_spilled
    dirty, spilled = set(sp.dm._modality_dirty), dict(sp.dm._modality_spilled)
    result = sp.dm.auto_save_state()
    assert list(outside.iterdir()) == []
    assert (sp.dm.data_dir / "other_autosave.h5ad").is_file()
    assert result.count(NOT_SAVED_OUTSIDE) == 1
    # The refused one stays spilled and as it was; only "other" was saved.
    assert set(sp.dm._modality_dirty) == dirty - {"other"}
    assert dict(sp.dm._modality_spilled) == spilled


# ------------------------------------------------------------- Sol r0 findings


BACK_IN = "../../outside/CREATED/../back/sub/x.{fmt}"


def _only_the_link_outside(s):
    assert sorted(p.name for p in s.outside.iterdir()) == ["back"]
    assert list(s.cwd.iterdir()) == []


@needs_tx
@pytest.mark.parametrize(
    "tool_name", ["export_de_results", "extract_and_export_de_results"]
)
def test_de_export_creates_no_folder_outside_on_the_way_back_in(s, tool_name):
    """WP-IMPL-01: the final file resolves inside, a folder made on the way is not."""
    os.symlink(s.dm.exports_dir, s.outside / "back")
    args = {"modality_name": "de_results", "output_path": BACK_IN.format(fmt="csv")}
    if tool_name == "extract_and_export_de_results":
        args["shrink_lfc"] = False
    response = _de_tools(s)[tool_name].invoke(args)
    assert response == WORKSPACE_EXPORT_MESSAGE
    _only_the_link_outside(s)
    assert not (s.dm.exports_dir / "sub").exists()


@needs_tx
@pytest.mark.parametrize("shape", ["link", "dotdot"])
def test_annotation_export_creates_no_folder_outside_on_the_way_back_in(s, shape):
    from lobster.agents.transcriptomics.annotation_expert import annotation_expert

    os.symlink(s.dm.exports_dir, s.outside / "back")
    spelled = (
        BACK_IN.format(fmt="json")
        if shape == "link"
        else "../../outside/CREATED/../../ws/exports/sub/x.json"
    )
    tools = factory_tools(annotation_expert, ANNOTATION_MODULE, s.dm)
    response = tools["export_annotation_mapping"].invoke(
        {"modality_name": "de_results", "format": "json", "output_filename": spelled}
    )
    assert response == WORKSPACE_EXPORT_MESSAGE
    _only_the_link_outside(s)
    assert not (s.dm.exports_dir / "sub").exists()


def test_spilled_autosave_creates_no_folder_outside_on_the_way_back_in(
    tmp_path, monkeypatch
):
    from tests.unit.core.test_dm_save_guard import SPILL_SHAPE, _adata, _session

    monkeypatch.setattr(  # admit the name, as a direct registration would
        "lobster.core.runtime.data_manager.check_modality_name", lambda name: None
    )
    sp = _session(tmp_path, monkeypatch, budget_of=1.5)
    outside = tmp_path / "outside"
    outside.mkdir()
    sp.dm.data_dir.mkdir(parents=True, exist_ok=True)
    os.symlink(sp.dm.data_dir, outside / "back")
    name = "../../outside/CREATED/../back/sub/name"
    sp.dm.store_modality(name=name, adata=_adata(0, *SPILL_SHAPE))
    sp.dm.store_modality(name="other", adata=_adata(1, *SPILL_SHAPE))
    assert name in sp.dm._modality_spilled
    dirty, spilled = set(sp.dm._modality_dirty), dict(sp.dm._modality_spilled)
    result = sp.dm.auto_save_state()
    assert sorted(p.name for p in outside.iterdir()) == ["back"]
    assert not (sp.dm.data_dir / "sub").exists()
    assert result.count(NOT_SAVED_OUTSIDE) == 1
    assert set(sp.dm._modality_dirty) == dirty - {"other"}
    assert dict(sp.dm._modality_spilled) == spilled


def _scvi_child_link(s):
    victim = s.outside / "victim.pt"
    victim.write_bytes(b"keep")
    folder = s.dm.workspace_path / "models" / "de_results_scvi_model"
    folder.mkdir(parents=True)
    os.symlink(victim, folder / "model.pt")
    return victim, folder


@needs_ml
def test_scvi_tool_refuses_a_saved_file_linked_outside(s, request):
    """WP-IMPL-02: the folder is inside, its model.pt leads out."""
    from tests.unit.services.test_scvi_export_guard import _tool

    fake = request.getfixturevalue("fake_scvi")
    victim, _ = _scvi_child_link(s)
    response = _tool(s.dm).invoke({"modality_name": "de_results", "max_epochs": 1})
    assert response == WORKSPACE_EXPORT_MESSAGE
    assert victim.read_bytes() == b"keep"
    assert fake.setup_calls == [] and fake.instances == []
    assert "X_scvi" not in s.dm.modalities["de_results"].obsm


@needs_ml
def test_scvi_service_refuses_a_saved_file_linked_outside(s, request):
    from lobster.services.analysis.scvi_embedding_service import ScviEmbeddingService

    fake = request.getfixturevalue("fake_scvi")
    victim, folder = _scvi_child_link(s)
    with pytest.raises(WorkspaceExportError):
        ScviEmbeddingService().train_scvi_embedding(
            s.adata.copy(), save_path=str(folder), workspace_path=s.dm.workspace_path
        )
    assert victim.read_bytes() == b"keep"
    assert fake.setup_calls == [] and fake.instances == []


@needs_ml
def test_scvi_saved_file_linked_into_a_listed_root_is_written(s, monkeypatch, request):
    from tests.unit.services.test_scvi_export_guard import _tool

    request.getfixturevalue("fake_scvi")
    monkeypatch.setattr(scope, "_EXPORT_ROOTS", (s.outside,))
    victim, _ = _scvi_child_link(s)
    response = _tool(s.dm).invoke({"modality_name": "de_results", "max_epochs": 1})
    assert victim.read_bytes() == b"model", response


def _single_cell(s, monkeypatch, labels):
    import types

    science = _de_science(monkeypatch, s.results)
    monkeypatch.setattr(s.dm, "log_tool_usage", lambda *a, **k: None)  # mock IR
    adata = s.adata.copy()
    adata.uns.pop("is_pseudobulk")
    adata.obs["condition"] = [labels[0]] * 3 + [labels[1]] * 3
    s.dm.store_modality("cells", adata)
    stats = {
        "method": "t_test",
        "n_genes_tested": 2,
        "n_significant_genes": 1,
        "n_upregulated": 1,
        "n_downregulated": 0,
        "top_upregulated": ["gene_a"],
        "top_downregulated": [],
        "de_results_key": "de_results_cells",
    }
    science.run_differential_expression_analysis.return_value = (
        adata.copy(),
        stats,
        types.SimpleNamespace(code_template="", parameters={}),
    )
    return science


@needs_tx
@pytest.mark.parametrize("save", [False, True])
def test_single_cell_de_with_a_slash_in_a_group_label(s, monkeypatch, save):
    """WP-IMPL-03: a group label is data, not a modality name. It is encoded in the
    derived name (and file name); the analysis result is kept."""
    science = _single_cell(s, monkeypatch, ("CD4/CD8", "other"))
    response = _de_tools(s)["run_differential_expression"].invoke(
        {
            **DE_ARGS,
            "modality_name": "cells",
            "group1": "CD4/CD8",
            "group2": "other",
            "method": "t_test",
            "save_result": save,
        }
    )
    science.run_differential_expression_analysis.assert_called_once()
    assert science.run_differential_expression_analysis.call_args.kwargs["group1"] == (
        "CD4/CD8"
    )
    name = "cells_de_CD4%2FCD8_vs_other"
    assert name in s.dm.list_modalities(), response
    assert f"'{name}'" in response
    assert (s.dm.data_dir / f"{name}.h5ad").is_file() is save
    s.nothing_outside()


@needs_tx
def test_group_labels_that_differ_only_in_a_slash_stay_apart(s, monkeypatch):
    _single_cell(s, monkeypatch, ("A/B", "A%2FB"))
    tools = _de_tools(s)
    args = {**DE_ARGS, "modality_name": "cells", "method": "t_test"}
    tools["run_differential_expression"].invoke(
        {**args, "group1": "A/B", "group2": "A%2FB"}
    )
    tools["run_differential_expression"].invoke(
        {**args, "group1": "A%2FB", "group2": "A/B"}
    )
    names = set(s.dm.list_modalities())
    assert {"cells_de_A%2FB_vs_A%252FB", "cells_de_A%252FB_vs_A%2FB"} <= names


@needs_tx
def test_a_group_label_leading_out_is_encoded_and_saved_inside(s, monkeypatch):
    science = _single_cell(s, monkeypatch, ("../../../../outside/a", "other"))
    response = _de_tools(s)["run_differential_expression"].invoke(
        {
            **DE_ARGS,
            "modality_name": "cells",
            "group1": "../../../../outside/a",
            "group2": "other",
            "method": "t_test",
        }
    )
    science.run_differential_expression_analysis.assert_called_once()
    name = "cells_de_..%2F..%2F..%2F..%2Foutside%2Fa_vs_other"
    assert (s.dm.data_dir / f"{name}.h5ad").is_file(), response
    s.nothing_outside()


@needs_tx
def test_bulk_de_direct_with_a_slash_in_a_group_label(s, monkeypatch):
    science = _single_cell(s, monkeypatch, ("CD4/CD8", "other"))
    response = _de_tools(s)["run_bulk_de_direct"].invoke(
        {
            "modality_name": "cells",
            "group_key": "condition",
            "group1": "CD4/CD8",
            "group2": "other",
        }
    )
    science.run_differential_expression_analysis.assert_called_once()
    name = "cells_de_CD4%2FCD8_vs_other"
    assert name in s.dm.list_modalities(), response
    assert (s.dm.data_dir / f"{name}.h5ad").is_file()
    s.nothing_outside()


@needs_tx
def test_a_derived_name_over_the_limit_is_refused_before_the_analysis(s, monkeypatch):
    science = _single_cell(s, monkeypatch, ("g" * 190, "other"))
    response = _de_tools(s)["run_differential_expression"].invoke(
        {
            **DE_ARGS,
            "modality_name": "cells",
            "group1": "g" * 190,
            "group2": "other",
            "method": "t_test",
        }
    )
    assert response == "Error: " + scope.MODALITY_NAME_REASON
    science.run_differential_expression_analysis.assert_not_called()


@needs_tx
def test_annotation_export_through_a_link_loop_is_the_fixed_refusal(s, caplog):
    """WP-IMPL-04: no path-bearing error text or log."""
    import logging

    from lobster.agents.transcriptomics.annotation_expert import annotation_expert

    os.symlink("loop", s.dm.exports_dir / "loop")
    tools = factory_tools(annotation_expert, ANNOTATION_MODULE, s.dm)
    with caplog.at_level(logging.DEBUG):
        response = tools["export_annotation_mapping"].invoke(
            {
                "modality_name": "de_results",
                "format": "json",
                "output_filename": "loop/x.json",
            }
        )
    assert response == WORKSPACE_EXPORT_MESSAGE
    assert not [r for r in caplog.records if str(s.root) in r.getMessage()]
