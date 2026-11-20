"""The replicate check of a two-group DE counts the groups the test uses.

Wilcoxon and the t-test compare two groups only, so a third group with one
sample refused a ctrl vs treat comparison. deseq2 fits every group of the
column (design ~condition), but a level it does not compare is fitted on its
own and leaves the contrast unchanged, so it is not checked either. One
replicate reads "1 replicate", not "1 replicates".
"""

import pandas as pd
import pytest

from lobster.agents.transcriptomics.de_analysis_expert import (
    _count_biological_replicates,
)

from .test_bulk_rank_no_marker_filter import _c2
from .test_de_genes_tested_line import _run


def _groups(ctrl, other, treat):
    adata = _c2()
    adata.obs["condition"] = ["ctrl"] * ctrl + ["other"] * other + ["treat"] * treat
    return adata


@pytest.mark.parametrize("method", ["wilcoxon", "t_test"])
def test_a_short_group_not_compared_does_not_refuse(tmp_path, monkeypatch, method):
    _, reply = _run(tmp_path, monkeypatch, _groups(5, 1, 6), method, pseudobulk=True)

    assert "Insufficient replicates" not in reply, reply
    assert "Genes tested" in reply


@pytest.mark.parametrize("method", ["wilcoxon", "t_test"])
def test_a_short_compared_group_refuses_naming_only_it(tmp_path, monkeypatch, method):
    _, reply = _run(tmp_path, monkeypatch, _groups(1, 5, 6), method, pseudobulk=True)

    assert reply.startswith(
        "**Insufficient replicates**: Group 'ctrl' has only 1 replicate (minimum 3"
    ), reply
    assert "'other'" not in reply


def test_deseq2_does_not_count_a_short_group_it_does_not_compare(tmp_path, monkeypatch):
    """deseq2 fits a 1-sample level exactly and the contrast is unchanged; the
    minimum of 3 is for the groups compared."""
    _, reply = _run(tmp_path, monkeypatch, _groups(5, 1, 6), "deseq2", pseudobulk=True)

    assert reply.startswith("## Differential Expression Analysis Complete"), reply
    assert "- Contrast: treat vs ctrl in condition\n" in reply


def _rows(a, b):
    return pd.DataFrame({"condition": ["a"] * a + ["b"] * b})


def test_one_replicate_is_singular():
    result = _count_biological_replicates(_rows(1, 4), "condition")

    assert "Group 'a' has only 1 replicate (minimum 3" in result["errors"][0]


def test_one_biological_replicate_is_singular():
    meta = pd.DataFrame(
        {"condition": ["a"] * 3 + ["b"] * 4, "donor_id": ["d0"] * 3 + list("wxyz")}
    )

    result = _count_biological_replicates(meta, "condition")

    assert "Group 'a' has only 1 biological replicate (" in result["errors"][0]


def test_the_power_warning_for_one_replicate_is_singular():
    result = _count_biological_replicates(_rows(1, 4), "condition", min_replicates=1)

    assert any(w.startswith("Group 'a' has 1 replicate. ") for w in result["warnings"])


def test_deseq2_with_a_short_compared_group_adds_no_note(tmp_path, monkeypatch):
    _, reply = _run(tmp_path, monkeypatch, _groups(1, 5, 6), "deseq2", pseudobulk=True)

    assert reply.startswith(
        "**Insufficient replicates**: Group 'ctrl' has only 1 replicate (minimum 3"
    ), reply
