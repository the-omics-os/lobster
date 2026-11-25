"""Enrichment beyond max_genes takes the proteins with the smallest adjusted p-value.

The gene list was cut alphabetically, so which proteins went into enrichment
depended on their names; the notebook's replay cut a set, in another order.
"""

import sys
import types
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd

from lobster.services.analysis.proteomics_pathway_service import (
    ProteomicsPathwayService,
)
from lobster.testing.provenance_replay import replay_ir

N_GENES = 650
MAX_GENES = 500


def _fixture():
    """Records for 650 genes, and each gene's best (padj, LFC) by construction."""
    rng = np.random.default_rng(11)
    names = np.array([f"G{i:04d}" for i in range(N_GENES)])
    # Few distinct values, so padj ties and |LFC| ties both reach the cut
    padj = rng.choice([1e-9, 1e-5, 1e-3, 0.01, 0.04], N_GENES)
    lfc = rng.choice([-3.0, -1.5, 1.0, 1.5, 3.0], N_GENES)
    padj[rng.choice(N_GENES, 30, replace=False)] = np.nan

    records = []
    for i, name in enumerate(names):
        # Protein-group and UniProt-mnemonic names clean to the same symbol
        shown = [name, f"{name};P{i:05d}", f"{name}_HUMAN"][i % 3]
        records.append(
            {"protein": shown, "p_adjusted": padj[i], "log2_fold_change": lfc[i]}
        )
        if i % 4 == 0 and not np.isnan(padj[i]):
            # The same gene from a second comparison, worse than its best
            records.append(
                {"protein": name, "p_adjusted": padj[i] * 2, "log2_fold_change": 9.0}
            )
    order = rng.permutation(len(records))
    records = [records[k] for k in order]

    adata = ad.AnnData(np.zeros((2, 1)))
    adata.uns["differential_expression"] = {"significant_results": records}
    return adata, names, padj, lfc


def _expected(names, padj, lfc):
    known = ~np.isnan(padj)
    k_names, k_padj, k_lfc = names[known], padj[known], lfc[known]
    order = np.lexsort((k_names, -np.abs(k_lfc), k_padj))
    assert k_padj[order][MAX_GENES - 1] == k_padj[order][MAX_GENES]
    assert known.sum() > MAX_GENES
    return k_names[order][:MAX_GENES].tolist()


def _run(adata):
    received = []

    def ora(self, adata, gene_list, **_):
        received.append(list(gene_list))
        return adata, {}, None

    with patch(
        "lobster.services.analysis.pathway_enrichment_service."
        "PathwayEnrichmentService.over_representation_analysis",
        ora,
    ):
        _, stats, ir = ProteomicsPathwayService().run_enrichment(
            adata, max_genes=MAX_GENES
        )
    return received[0], stats, ir


def test_enrichment_beyond_max_genes_takes_the_smallest_padj_ties_by_larger_lfc():
    adata, names, padj, lfc = _fixture()
    expected = _expected(names, padj, lfc)

    received, stats, _ = _run(adata)

    assert set(received) == set(expected)
    assert received == expected
    assert stats["n_genes_input"] == MAX_GENES
    assert stats["n_genes_significant"] == N_GENES


def test_the_notebook_replay_takes_the_same_genes(monkeypatch):
    adata, *_ = _fixture()
    received, _, ir = _run(adata)

    replayed = []

    def enrichr(gene_list, **_):
        replayed.append(list(gene_list))
        return types.SimpleNamespace(results=pd.DataFrame({"Adjusted P-value": []}))

    monkeypatch.setitem(sys.modules, "gseapy", types.SimpleNamespace(enrichr=enrichr))
    replay_ir(ir, adata=adata)

    assert replayed == [received]


def test_the_reply_says_which_proteins_went_in(tmp_path):
    import importlib
    from unittest.mock import MagicMock

    module = importlib.import_module("lobster.agents.proteomics.de_analysis_expert")
    adata, *_ = _fixture()
    service = MagicMock()
    service.run_enrichment.return_value = (
        adata,
        {"n_genes_input": MAX_GENES, "n_genes_significant": N_GENES},
        None,
    )
    manager = MagicMock()
    manager.list_modalities.return_value = ["proteins"]
    manager.get_modality.return_value = adata
    with (
        patch.object(module, "create_llm", return_value=MagicMock()),
        patch.object(module, "create_react_agent", side_effect=lambda **kw: kw),
        patch(
            "lobster.services.analysis.proteomics_pathway_service."
            "ProteomicsPathwayService",
            return_value=service,
        ),
    ):
        agent = module.de_analysis_expert(manager, workspace_path=tmp_path)
        tool = {t.name: t for t in agent["tools"]}["run_pathway_enrichment"]
        reply = tool.invoke({"modality_name": "proteins"})

    assert (
        "- Proteins analyzed: 500 (the 500 with the smallest adjusted p-value, "
        "out of 650 significant)\n" in reply
    ), reply
