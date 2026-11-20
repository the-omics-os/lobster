"""A group column with an unused category: DE runs on the groups present.

A categorical column keeps a level with no sample after subsetting. Since the
replicate check counts only the groups present, it reached pyDESeq2, whose
design then had a column of zeros ("Singular matrix"). The fit now drops the
unused levels on a copy; the stored modality keeps them.
"""

import pandas as pd
import pytest

from lobster.agents.transcriptomics.de_analysis_expert import de_analysis_expert
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.testing.provenance_replay import factory_tools

from .test_replicate_check_compared_groups import _groups

DE = "lobster.agents.transcriptomics.de_analysis_expert"


def _with_unused_level():
    adata = _groups(6, 0, 6)
    adata.obs["condition"] = pd.Categorical(
        adata.obs["condition"], categories=["ctrl", "other", "treat"]
    )
    adata.uns["is_pseudobulk"] = True
    return adata


def _tools(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # BulkRNASeqService writes ./bulk_results
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.store_modality("bulk", _with_unused_level())
    return dm, factory_tools(de_analysis_expert, DE, dm)


@pytest.mark.parametrize("method", ["deseq2", "wilcoxon", "t_test"])
def test_a_two_group_de_with_an_unused_level_completes(tmp_path, monkeypatch, method):
    dm, tools = _tools(tmp_path, monkeypatch)

    reply = tools["run_differential_expression"].invoke(
        {
            "modality_name": "bulk",
            "groupby": "condition",
            "group1": "ctrl",
            "group2": "treat",
            "method": method,
            "save_result": False,
        }
    )

    assert reply.startswith("## Differential Expression Analysis Complete"), reply
    assert list(dm.get_modality("bulk").obs["condition"].cat.categories) == [
        "ctrl",
        "other",
        "treat",
    ]


def test_the_formula_de_with_an_unused_level_completes(tmp_path, monkeypatch):
    _, tools = _tools(tmp_path, monkeypatch)

    reply = tools["run_de_with_formula"].invoke(
        {
            "pseudobulk_modality": "bulk",
            "formula": "~condition",
            "contrast": ["condition", "treat", "ctrl"],
            "save_result": False,
        }
    )

    assert "Singular" not in reply and "failed" not in reply.lower(), reply
    assert "Genes tested: 2,000" in reply, reply


@pytest.mark.parametrize("tool", ["prepare_de_design", "suggest_de_formula"])
def test_the_design_tools_do_not_count_an_unused_level(tmp_path, monkeypatch, tool):
    _, tools = _tools(tmp_path, monkeypatch)
    args = (
        {"formula": "~condition", "contrast": ["condition", "treat", "ctrl"]}
        if tool == "prepare_de_design"
        else {"groupby": "condition"}
    )

    reply = tools[tool].invoke({"modality_name": "bulk", **args})

    assert "'other'" not in reply, reply
    assert "0 replicate" not in reply, reply


def test_the_fit_leaves_the_callers_metadata_alone(tmp_path, monkeypatch):
    from lobster.services.analysis.bulk_rnaseq_service import BulkRNASeqService

    monkeypatch.chdir(tmp_path)
    adata = _with_unused_level()
    metadata = adata.obs.copy()
    counts = pd.DataFrame(adata.X.T, index=adata.var_names, columns=adata.obs_names)

    BulkRNASeqService(results_dir=str(tmp_path / "results")).run_pydeseq2_analysis(
        count_matrix=counts,
        metadata=metadata,
        formula="~condition",
        contrast=["condition", "treat", "ctrl"],
        shrink_lfc=False,
    )

    assert list(metadata["condition"].cat.categories) == ["ctrl", "other", "treat"]
