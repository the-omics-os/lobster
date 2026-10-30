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


@needs_tx
def test_single_cell_de_refuses_a_group_name_leading_out(s, monkeypatch):
    """The group names are tool arguments and part of the h5ad name (N6)."""
    science = _de_science(monkeypatch, s.results)
    adata = s.adata.copy()
    adata.uns.pop("is_pseudobulk")
    # A real group value; data/cells_de_<group>_vs_sample_b.h5ad leads to outside/.
    group = "../../../../outside/a"
    adata.obs["condition"] = [group] * 3 + ["sample_b"] * 3
    s.dm.store_modality("cells", adata)
    response = _de_tools(s)["run_differential_expression"].invoke(
        {**DE_ARGS, "modality_name": "cells", "group1": group}
    )
    assert response == WORKSPACE_EXPORT_MESSAGE
    science.run_pydeseq2_from_pseudobulk.assert_not_called()
    assert "cells" in s.dm.list_modalities()
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
