"""The wilcoxon / t-test reply says how many genes were tested, of how many, and why
the others were not.

"Genes tested: N" left out the genes the expression pre-filter dropped (no value
above 1.0 in at least 2 samples) and those whose fold change is not a number (one
group's mean below -1; exactly -1 gives -inf, which is kept). The line is now
"Genes tested: N of M (K below the expression threshold, J whose fold change is
not a number)", the parts that are zero left out, and N is the stored table's row
count.
"""

import pytest

from lobster.agents.transcriptomics.de_analysis_expert import de_analysis_expert
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.testing.provenance_replay import factory_tools

from .test_bulk_rank_no_marker_filter import DE, _c2


def _with_untested(below, signed):
    """c2 with ``below`` genes all zero and ``signed`` genes at -5 in ctrl and 10 in
    treat (above the threshold in 6 samples, a fold change that is not a number)."""
    adata = _c2()
    end = adata.n_vars
    adata.X[:, end - below - signed : end - signed] = 0
    if signed:
        adata.X[:6, end - signed :] = -5
        adata.X[6:, end - signed :] = 10
    return adata


def _run(tmp_path, monkeypatch, adata, method, pseudobulk):
    monkeypatch.chdir(tmp_path)  # BulkRNASeqService writes ./bulk_results
    if pseudobulk:
        adata.uns["is_pseudobulk"] = True
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.store_modality("bulk", adata)
    reply = factory_tools(de_analysis_expert, DE, dm)[
        "run_differential_expression"
    ].invoke(
        {
            "modality_name": "bulk",
            "groupby": "condition",
            "group1": "ctrl",
            "group2": "treat",
            "method": method,
            "save_result": False,
        }
    )
    return dm, reply


def _tested_line(reply):
    lines = [line for line in reply.splitlines() if "Genes tested" in line]
    assert len(lines) == 1, reply
    return lines[0]


def _stored_rows(dm):
    (result,) = [m for m in dm.list_modalities() if m != "bulk"]
    stored = dm.get_modality(result)
    table = stored.uns[stored.uns["de_current_results_key"]]
    return len(next(iter(table.values()))), stored.n_vars


CASES = {
    "all_tested": (0, 0, "- Genes tested: 2,000 of 2,000"),
    "below_threshold": (
        30,
        0,
        "- Genes tested: 1,970 of 2,000 (30 below the expression threshold)",
    ),
    "no_finite_fold_change": (
        0,
        20,
        "- Genes tested: 1,980 of 2,000 (20 whose fold change is not a number)",
    ),
    "both": (
        30,
        20,
        "- Genes tested: 1,950 of 2,000 (30 below the expression threshold, "
        "20 whose fold change is not a number)",
    ),
}


@pytest.mark.parametrize("pseudobulk", [False, True], ids=["bulk", "pseudobulk"])
@pytest.mark.parametrize("method", ["wilcoxon", "t_test"])
@pytest.mark.parametrize("case", list(CASES))
def test_the_reply_says_how_many_genes_were_tested_of_how_many(
    tmp_path, monkeypatch, case, method, pseudobulk
):
    below, signed, line = CASES[case]
    adata = _with_untested(below, signed)

    dm, reply = _run(tmp_path, monkeypatch, adata, method, pseudobulk)

    assert "Complete" in reply, reply
    assert _tested_line(reply) == line
    tested = 2000 - below - signed
    assert _stored_rows(dm) == (tested, tested)
