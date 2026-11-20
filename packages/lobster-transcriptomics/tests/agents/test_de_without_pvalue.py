"""A gene with no p-value no longer wipes out a wilcoxon / t-test result.

A t-test gives no p-value (NaN) for a gene with the same value in every sample.
Benjamini-Hochberg over a list holding one NaN made every adjusted p-value NaN,
so every gene was dropped and the refusal named a false cause. statsmodels 0.14
also divided by zero on an empty list (no gene left after the pre-filter). The
correction now runs over the finite p-values only; a gene without one is counted
as such and is not reported.
"""

import numpy as np
import pytest

from lobster.services.analysis.bulk_rnaseq_service import BulkRNASeqService

from .test_bulk_de_empty_filter import counts
from .test_bulk_rank_no_marker_filter import _c2
from .test_de_genes_tested_line import _run, _stored_rows, _tested_line

FILTERS = {
    "min_expression_threshold": 1.0,
    "min_fold_change": 1.0,
    "min_pct_expressed": 0.0,
    "max_out_pct_expressed": 1.0,
}


@pytest.fixture
def statsmodels_0_14(monkeypatch):
    """multipletests as statsmodels 0.14 has it: an empty list divides by zero."""
    import statsmodels.stats.multitest as multitest

    real = multitest.multipletests

    def multipletests(pvals, *args, **kwargs):
        if len(pvals) == 0:
            raise ZeroDivisionError("float division by zero")
        return real(pvals, *args, **kwargs)

    monkeypatch.setattr(multitest, "multipletests", multipletests)


def _constant(adata, n, value=5.0):
    """The last ``n`` genes hold ``value`` in every sample."""
    adata.X[:, adata.n_vars - n :] = value
    return adata


@pytest.mark.parametrize("method", ["deseq2_like", "wilcoxon", "t_test"])
def test_no_gene_left_after_the_pre_filter_is_an_empty_result(
    tmp_path, monkeypatch, statsmodels_0_14, method
):
    monkeypatch.chdir(tmp_path)
    adata = counts("all_zero")

    out, stats, _ = BulkRNASeqService(
        results_dir=str(tmp_path / "results")
    ).run_differential_expression_analysis(
        adata=adata,
        groupby="condition",
        group1="ctrl",
        group2="treat",
        method=method,
        **FILTERS,
    )

    assert out.n_vars == 0 and stats["n_genes_tested"] == 0


@pytest.mark.parametrize("method", ["wilcoxon", "t_test"])
def test_no_gene_left_is_refused_whichever_statsmodels(
    tmp_path, monkeypatch, statsmodels_0_14, method
):
    adata = counts("all_zero")
    adata.obs = adata.obs.copy()

    _, reply = _run(tmp_path, monkeypatch, adata, method, pseudobulk=True)

    assert reply.startswith("**DE refused: no gene was tested.**"), reply
    assert "expression pre-filter kept none" in reply


def test_a_gene_without_a_p_value_leaves_the_others_tested(tmp_path, monkeypatch):
    adata = _constant(_c2(), 1)

    dm, reply = _run(tmp_path, monkeypatch, adata, "t_test", pseudobulk=False)

    assert "Complete" in reply, reply
    assert _tested_line(reply) == (
        "- Genes tested: 1,999 of 2,000 (1 without a p-value)"
    )
    assert _stored_rows(dm) == (1999, 1999)
    (result,) = [m for m in dm.list_modalities() if m != "bulk"]
    assert np.isfinite(dm.get_modality(result).var["padj"]).all()


def test_wilcoxon_tests_a_gene_with_the_same_value_everywhere(tmp_path, monkeypatch):
    adata = _constant(_c2(), 1)

    dm, reply = _run(tmp_path, monkeypatch, adata, "wilcoxon", pseudobulk=False)

    assert _tested_line(reply) == "- Genes tested: 2,000 of 2,000"


def test_a_missing_value_drops_only_its_gene(tmp_path, monkeypatch):
    adata = _c2()
    adata.X[0, -1] = np.nan  # its mean, so its fold change, is not a number

    dm, reply = _run(tmp_path, monkeypatch, adata, "wilcoxon", pseudobulk=False)

    assert "Complete" in reply, reply
    assert _tested_line(reply) == (
        "- Genes tested: 1,999 of 2,000 (1 whose fold change is not a number)"
    )
    assert _stored_rows(dm) == (1999, 1999)


def test_the_line_names_each_reason_a_gene_was_not_tested(tmp_path, monkeypatch):
    adata = _constant(_c2(), 3)
    adata.X[:, :30] = 0  # below the expression threshold
    adata.X[:6, 30:50] = -5  # a fold change that is not a number
    adata.X[6:, 30:50] = 10

    dm, reply = _run(tmp_path, monkeypatch, adata, "t_test", pseudobulk=False)

    assert _tested_line(reply) == (
        "- Genes tested: 1,947 of 2,000 (30 below the expression threshold, "
        "20 whose fold change is not a number, 3 without a p-value)"
    )


def test_no_gene_with_a_p_value_is_refused_naming_why(tmp_path, monkeypatch):
    adata = _constant(counts("bulk_all_expressed"), 500)

    _, reply = _run(tmp_path, monkeypatch, adata, "t_test", pseudobulk=True)

    assert reply.startswith("**DE refused: no gene was tested.**"), reply
    assert (
        "Of 500 genes, 500 passed the expression pre-filter, but none has a "
        "p-value: a t-test gives none for a gene with the same value in every "
        "sample. So no result can be reported." in reply
    ), reply


def test_a_refusal_for_both_reasons_counts_each(tmp_path, monkeypatch):
    adata = _constant(counts("bulk_all_expressed"), 200)
    adata.X[:3, :300] = -5  # the other 300: a fold change that is not a number
    adata.X[3:, :300] = 10

    _, reply = _run(tmp_path, monkeypatch, adata, "t_test", pseudobulk=True)

    assert reply.startswith("**DE refused: no gene was tested.**"), reply
    assert (
        "Of 500 genes, 500 passed the expression pre-filter, but none can be "
        "reported: 200 have no p-value (a t-test gives none for a gene with the "
        "same value in every sample) and the fold change of the other 300 is not "
        "a number (one group's mean is below -1, as with centred or signed "
        "values)." in reply
    ), reply


@pytest.mark.parametrize("method", ["deseq2_like", "wilcoxon", "t_test"])
def test_the_exported_code_adjusts_as_the_analysis_does(tmp_path, monkeypatch, method):
    monkeypatch.chdir(tmp_path)
    adata = _constant(_c2(), 1)
    service = BulkRNASeqService(results_dir=str(tmp_path / "results"))
    out, stats, ir = service.run_differential_expression_analysis(
        adata=adata.copy(),
        groupby="condition",
        group1="ctrl",
        group2="treat",
        method=method,
        **FILTERS,
    )
    scope = {"adata": adata.copy()}

    exec(compile(ir.render(), "<de>", "exec"), scope)  # nosec B102

    # a wilcoxon gives the constant gene p = 1; a t-test gives it none
    assert out.n_vars == (2000 if method == "wilcoxon" else 1999)
    replayed = scope["results_df"]["padj"].dropna()
    assert len(replayed) == out.n_vars
    np.testing.assert_allclose(
        replayed.loc[out.var_names].to_numpy(), out.var["padj"].to_numpy()
    )
