"""A publication export keeps the same genes the DE reply counted as significant.

The DE counts use padj < alpha (strict) and, with an LFC threshold, |LFC| >= it;
the export kept a gene at exactly padj == alpha, so its count was one higher.
"""

from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.transcriptomics.de_analysis_expert import de_analysis_expert
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.analysis.bulk_rnaseq_service import BulkRNASeqService
from lobster.testing.provenance_replay import factory_tools, replay_ir

MODULE = "lobster.agents.transcriptomics.de_analysis_expert"
GENES = ["A", "B", "C", "D", "E"]
# B sits exactly at padj 0.05 and C exactly at |LFC| 1.0.
TABLE = pd.DataFrame(
    {
        "log2FoldChange": [2.0, 3.0, 1.0, -0.5, 2.0],
        "padj": [0.01, 0.05, 0.04, 0.03, 0.2],
    },
    index=pd.Index(GENES, name="gene"),
)
# Counted by hand: padj < 0.05 keeps A, C, D; |LFC| >= 1.0 then keeps A, C.
SIGNIFICANT = ["A", "C", "D"]
SIGNIFICANT_LFC = ["A", "C"]


def _run(tmp_path, monkeypatch, lfc_threshold):
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    counts = np.random.default_rng(0).integers(1, 50, (6, len(GENES)))
    adata = ad.AnnData(
        X=counts.astype(float),
        obs=pd.DataFrame(
            {"condition": ["ctrl"] * 3 + ["treat"] * 3},
            index=[f"s{i}" for i in range(6)],
        ),
        var=pd.DataFrame(index=GENES),
    )
    adata.uns["is_pseudobulk"] = True
    dm.store_modality("pb", adata)
    # Only the pyDESeq2 fit is replaced; the service counts the table itself.
    fitter = Mock(return_value=(TABLE.copy(), {}, None))
    monkeypatch.setattr(BulkRNASeqService, "run_pydeseq2_analysis", fitter)
    log = Mock(wraps=dm.log_tool_usage)
    monkeypatch.setattr(dm, "log_tool_usage", log)
    tools = factory_tools(de_analysis_expert, MODULE, dm)
    de_reply = tools["run_de_with_formula"].invoke(
        {
            "pseudobulk_modality": "pb",
            "formula": "~condition",
            "contrast": ["condition", "treat", "ctrl"],
            "lfc_threshold": lfc_threshold,
            "save_results": False,
        }
    )
    out = dm.workspace_path / "exports" / "pub.csv"
    export_reply = tools["extract_and_export_de_results"].invoke(
        {
            "modality_name": "pb",
            "shrink_lfc": False,
            "output_path": str(out),
            "padj_threshold": 0.05,
            "lfc_threshold": lfc_threshold,
        }
    )
    ir = log.call_args.kwargs["ir"]
    return de_reply, export_reply, out, ir, dm.get_modality("pb")


@pytest.mark.parametrize(
    "lfc_threshold, de_line, expected",
    [
        (0.0, "- Significant genes (FDR < 0.05): 3\n", SIGNIFICANT),
        (1.0, "- Significant + |LFC| >= 1.0: 2\n", SIGNIFICANT_LFC),
    ],
    ids=["padj-only", "padj-and-lfc"],
)
def test_export_keeps_the_genes_the_de_reply_counted(
    tmp_path, monkeypatch, lfc_threshold, de_line, expected
):
    de_reply, export_reply, out, ir, adata = _run(tmp_path, monkeypatch, lfc_threshold)

    assert de_line in de_reply, de_reply
    assert f"- Genes after filtering: {len(expected)}\n" in export_reply, export_reply
    assert sorted(pd.read_csv(out, index_col=0).index) == expected

    # The notebook cell keeps the same genes (it writes relative to its folder).
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    monkeypatch.chdir(replay_dir)
    replay_ir(ir, adata=adata)
    replayed = pd.read_csv(replay_dir / ir.parameters["output_path"], index_col=0)
    assert sorted(replayed.index) == expected
