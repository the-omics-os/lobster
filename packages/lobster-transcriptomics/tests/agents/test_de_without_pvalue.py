"""A gene with no p-value no longer wipes out a wilcoxon / t-test result.

A t-test gives no p-value (NaN) for a gene with the same value in every sample.
Benjamini-Hochberg over a list holding one NaN made every adjusted p-value NaN,
so every gene was dropped and the refusal named a false cause. statsmodels 0.14
also divided by zero on an empty list (no gene left after the pre-filter). The
correction now runs over the finite p-values only; a gene without one is counted
as such and is not reported.
"""

import numpy as np
import pandas as pd
import pytest

from lobster.services.analysis.bulk_rnaseq_service import BulkRNASeqService

from .test_bulk_de_empty_filter import counts
from .test_bulk_rank_no_marker_filter import _c2
from .test_de_genes_tested_line import _run, _stored_rows, _tested_line

MISSING = (
    "Remove the genes with missing values, or fill them, before testing; or "
    "import complete raw counts and use deseq2. No result modality was stored."
)
SIGNED = (
    "Wilcoxon and t-test expect counts or log-scale values. Use deseq2 on raw "
    "counts, or run on unscaled values. No result modality was stored."
)
BOTH = (
    "Remove the genes with missing values, and run on counts or log-scale values "
    "rather than centred or scaled ones; deseq2 on complete raw counts avoids "
    "both. No result modality was stored."
)

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
        "a number (300 with a group mean below -1). " + SIGNED in reply
    ), reply


def _analysed_and_exported(tmp_path, monkeypatch, adata, method):
    monkeypatch.chdir(tmp_path)
    service = BulkRNASeqService(results_dir=str(tmp_path / "results"))
    out, _, ir = service.run_differential_expression_analysis(
        adata=adata.copy(),
        groupby="condition",
        group1="ctrl",
        group2="treat",
        method=method,
        **FILTERS,
    )
    scope = {"adata": adata.copy()}
    exec(compile(ir.render(), "<de>", "exec"), scope)  # nosec B102
    return out, pd.DataFrame(scope["adata_de"].uns["de_results_ctrl_vs_treat"])


@pytest.mark.parametrize("method", ["deseq2_like", "wilcoxon", "t_test"])
def test_the_exported_code_adjusts_as_the_analysis_does(tmp_path, monkeypatch, method):
    adata = _constant(_c2(), 1)
    adata.X[0, 0] = np.nan  # no p-value from any test, and no fold change

    out, exported = _analysed_and_exported(tmp_path, monkeypatch, adata, method)

    # a wilcoxon gives the constant gene p = 1; a t-test gives it none
    assert out.n_vars == (1999 if method == "wilcoxon" else 1998)
    replayed = exported["padj"].dropna()
    assert len(replayed) == out.n_vars
    np.testing.assert_allclose(
        replayed.loc[out.var_names].to_numpy(), out.var["padj"].to_numpy()
    )


@pytest.mark.parametrize("method", ["deseq2_like", "wilcoxon", "t_test"])
def test_the_exported_code_runs_when_no_gene_has_a_p_value(
    tmp_path, monkeypatch, statsmodels_0_14, method
):
    adata = counts("bulk_all_expressed")
    adata.X[:] = np.nan

    out, exported = _analysed_and_exported(tmp_path, monkeypatch, adata, method)

    assert out.n_vars == 0
    assert exported["padj"].isna().all()


def test_a_count_of_one_reads_in_the_singular(tmp_path, monkeypatch):
    adata = counts("survivor")
    adata.X[:, 0] = 5  # the same value everywhere: no p-value
    adata.X[:3, 1] = -5  # a fold change that is not a number
    adata.X[3:, 1] = 10

    _, reply = _run(tmp_path, monkeypatch, adata, "t_test", pseudobulk=True)

    assert (
        "Of 2 genes, 2 passed the expression pre-filter, but none can be "
        "reported: 1 has no p-value (a t-test gives none for a gene with the same "
        "value in every sample) and the fold change of the other 1 is not a "
        "number" in reply
    ), reply


def test_one_gene_reads_in_the_singular(tmp_path, monkeypatch):
    adata = counts("all_zero")[:, :1].copy()

    _, reply = _run(tmp_path, monkeypatch, adata, "t_test", pseudobulk=True)

    assert "Of 1 gene, no gene has a value above" in reply, reply


@pytest.mark.parametrize("method", ["wilcoxon", "t_test"])
def test_a_refusal_for_missing_values_names_them(tmp_path, monkeypatch, method):
    adata = counts("bulk_all_expressed")
    adata.X[0, :] = np.nan  # one sample has no values

    _, reply = _run(tmp_path, monkeypatch, adata, method, pseudobulk=True)

    assert reply.startswith("**DE refused: no gene was tested.**"), reply
    assert (
        "Of 500 genes, 500 passed the expression pre-filter, but none has a fold "
        "change that is a number (500 with a missing value): the fold change "
        "log2((mean2 + 1) / (mean1 + 1)) is not a number when a gene has a "
        "missing value. So no result can be reported. " + MISSING in reply
    ), reply
    assert "below -1" not in reply and "centred" not in reply, reply


def test_a_refusal_for_both_reasons_names_missing_values(tmp_path, monkeypatch):
    adata = counts("bulk_all_expressed")
    adata.X[:, :200] = 5  # no p-value
    adata.X[0, 200:] = np.nan  # the other 300: a missing value

    _, reply = _run(tmp_path, monkeypatch, adata, "t_test", pseudobulk=True)

    assert (
        "the fold change of the other 300 is not a number (300 with a missing "
        "value). " + MISSING in reply
    ), reply


def test_a_refusal_for_both_fold_change_causes_counts_each(tmp_path, monkeypatch):
    adata = counts("bulk_all_expressed")
    adata.X[0, :300] = np.nan  # 300: a missing value
    adata.X[:3, 300:] = -5  # the other 200: a group mean below -1
    adata.X[3:, 300:] = 10

    _, reply = _run(tmp_path, monkeypatch, adata, "t_test", pseudobulk=True)

    assert (
        "but none has a fold change that is a number (300 with a missing value, "
        "200 with a group mean below -1): the fold change log2((mean2 + 1) / "
        "(mean1 + 1)) is not a number when a gene has a missing value, or when "
        "one group's mean is below -1, as with centred or signed values. So no "
        "result can be reported. " + BOTH in reply
    ), reply


def test_an_input_with_every_value_missing_says_so(tmp_path, monkeypatch):
    adata = counts("bulk_all_expressed")
    adata.X[:] = np.nan

    _, reply = _run(tmp_path, monkeypatch, adata, "t_test", pseudobulk=True)

    assert (
        "Of 500 genes, no gene has a value above 1.0 in at least 2 samples "
        "(every value is missing), so the expression pre-filter kept none. " + MISSING
        in reply
    ), reply


@pytest.mark.parametrize("method", ["deseq2_like", "wilcoxon", "t_test"])
def test_the_exported_code_tests_the_genes_the_analysis_tested(
    tmp_path, monkeypatch, method
):
    """30 genes below the expression threshold: the exported code leaves them out
    of the test and of the correction, as the analysis does."""
    adata = _c2()
    adata.X[:, :30] = np.arange(12)[:, None] % 3 * 0.5  # 0 to 1: varies, never above

    out, exported = _analysed_and_exported(tmp_path, monkeypatch, adata, method)

    replayed = exported["padj"].dropna()
    assert out.n_vars == 1970
    assert sorted(replayed.index) == sorted(out.var_names)
    np.testing.assert_allclose(
        replayed.loc[out.var_names].to_numpy(), out.var["padj"].to_numpy()
    )


@pytest.mark.parametrize("method", ["wilcoxon", "t_test"])
def test_the_exported_code_makes_the_call_the_analysis_made(
    tmp_path, monkeypatch, method
):
    monkeypatch.chdir(tmp_path)
    called = {
        "groupby": "condition",
        "group1": "ctrl",
        "group2": "treat",
        "method": method,
        "min_expression_threshold": 2.5,
        "min_fold_change": 1.7,
        "min_pct_expressed": 0.3,
        "max_out_pct_expressed": 0.8,
    }
    service = BulkRNASeqService(results_dir=str(tmp_path / "results"))
    _, _, ir = service.run_differential_expression_analysis(adata=_c2(), **called)
    seen = []

    def record(self, adata, **kwargs):
        seen.append(kwargs)
        return adata, {}, None

    monkeypatch.setattr(
        BulkRNASeqService, "run_differential_expression_analysis", record
    )
    exec(compile(ir.render(), "<de>", "exec"), {"adata": _c2()})  # nosec B102

    assert seen == [called]


def test_a_missing_value_in_the_second_group_counts_too(tmp_path, monkeypatch):
    adata = counts("bulk_all_expressed")
    adata.X[-1, :] = np.nan  # a treat sample has no values

    _, reply = _run(tmp_path, monkeypatch, adata, "t_test", pseudobulk=True)

    assert "(500 with a missing value)" in reply, reply


@pytest.mark.parametrize(
    "case", ["low_and_one_missing", "no_gene"], ids=["partly_missing", "no_gene"]
)
def test_a_pre_filter_refusal_says_every_value_is_missing_only_then(
    tmp_path, monkeypatch, case
):
    adata = counts("bulk_all_expressed")
    if case == "no_gene":
        adata = adata[:, :0].copy()
    else:
        adata.X = adata.X / 1e4  # every value <= 1
        adata.X[0, 0] = np.nan

    _, reply = _run(tmp_path, monkeypatch, adata, "t_test", pseudobulk=True)

    assert "the expression pre-filter kept none. " + SIGNED in reply, reply
    assert "every value is missing" not in reply, reply
