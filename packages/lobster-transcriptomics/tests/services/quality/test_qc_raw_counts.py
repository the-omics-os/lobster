"""Count-based QC metrics come from raw counts, never from normalized values.

A re-assessment of a log-normalized modality used to sum log1p values as "total
counts" and divide mitochondrial log1p sums by them, so the report showed a median
of about 2,100 "counts" where the cells had about 5,700, and a mitochondrial
percentage about three times too low.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

from lobster.services.quality.quality_service import QualityService

GENES = ["MT-ND1", "MT-CO1", "RPL3", "RPS6", "ACTB", "GAPDH", "MALAT1", "HBB"]
GENES += [f"GENE{i}" for i in range(24)]
COUNT_COLUMNS = ("n_counts", "mt_pct", "ribo_pct", "housekeeping_score", "percent_hb")
PARAMS = dict(
    min_genes=1,
    max_genes=100,
    max_mt_pct=20.0,
    max_ribo_pct=50.0,
    min_housekeeping_score=1.0,
)


def _counts(n_obs=40, seed=7):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(4, size=(n_obs, len(GENES))).astype(np.float32)
    counts[:, :2] *= 3  # a visible mitochondrial fraction
    return AnnData(
        X=counts,
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=GENES),
    )


def _log_cp10k(raw, sparse=False, log=True):
    totals = raw.X.sum(axis=1, keepdims=True)
    values = raw.X / np.maximum(totals, 1) * 1e4
    if log:
        values = np.log1p(values)
    norm = AnnData(
        X=sp.csr_matrix(values) if sparse else values,
        obs=raw.obs.copy(),
        var=raw.var.copy(),
    )
    if log:
        norm.uns["log1p"] = {"base": None}
    return norm


def _replay(ir, adata):
    namespace = {"adata": adata.copy(), "np": np}
    for helper in ir.helper_code:
        exec(
            helper, namespace
        )  # nosec B102 # Trusted repository IR on synthetic inputs.
    exec(
        ir.render(), namespace
    )  # nosec B102 # Trusted repository IR on synthetic inputs.
    return namespace["adata"]


def _assert_count_columns_equal(result, expected):
    for column in COUNT_COLUMNS:
        np.testing.assert_allclose(
            result.obs[column].to_numpy(dtype=float),
            expected.obs[column].to_numpy(dtype=float),
            rtol=1e-5,
            err_msg=column,
        )


@pytest.mark.parametrize("sparse", [False, True])
def test_normalized_with_counts_layer_uses_the_counts(sparse):
    raw = _counts()
    expected, _, _ = QualityService().assess_quality(raw, **PARAMS)
    norm = _log_cp10k(raw, sparse=sparse)
    norm.layers["counts"] = sp.csr_matrix(raw.X) if sparse else raw.X.copy()

    result, stats, ir = QualityService().assess_quality(norm, **PARAMS)

    _assert_count_columns_equal(result, expected)
    assert stats["count_source"] == "layers['counts']"
    assert stats["count_metrics_unavailable"] == []
    np.testing.assert_allclose(
        stats["qc_distributions"]["total_counts"]["median"],
        float(np.median(raw.X.sum(axis=1))),
        rtol=1e-6,
    )
    pd.testing.assert_frame_equal(_replay(ir, norm).obs, result.obs)


def test_normalized_with_raw_uses_raw_and_its_own_genes():
    # raw keeps every gene; the normalized modality dropped four of them,
    # including one mitochondrial gene, so its own genes would undercount.
    raw = _counts()
    expected, _, _ = QualityService().assess_quality(raw, **PARAMS)
    norm = _log_cp10k(raw)
    norm.raw = raw
    norm = norm[:, [g for g in GENES if g not in ("MT-CO1", "GENE0", "GENE1", "GENE2")]]
    norm = norm.copy()

    result, stats, ir = QualityService().assess_quality(norm, **PARAMS)

    for column in ("n_counts", "mt_pct", "ribo_pct", "housekeeping_score"):
        np.testing.assert_allclose(
            result.obs[column].to_numpy(dtype=float),
            expected.obs[column].to_numpy(dtype=float),
            rtol=1e-5,
            err_msg=column,
        )
    assert stats["count_source"] == "raw"
    pd.testing.assert_frame_equal(_replay(ir, norm).obs, result.obs)


def test_normalized_with_carried_obs_totals_uses_them():
    raw = _counts()
    norm = _log_cp10k(raw)
    totals = raw.X.sum(axis=1)
    norm.obs["total_counts"] = totals
    norm.obs["pct_counts_mt"] = raw.X[:, :2].sum(axis=1) / totals * 100

    result, stats, ir = QualityService().assess_quality(norm, **PARAMS)

    np.testing.assert_allclose(result.obs["n_counts"], totals, rtol=1e-6)
    np.testing.assert_allclose(result.obs["mt_pct"], norm.obs["pct_counts_mt"])
    assert result.obs["ribo_pct"].isna().all()
    assert result.obs["housekeeping_score"].isna().all()
    assert stats["count_source"] == "obs"
    assert stats["count_metrics_unavailable"] == ["ribo_pct", "housekeeping_score"]
    pd.testing.assert_frame_equal(_replay(ir, norm).obs, result.obs)


@pytest.mark.parametrize("log", [True, False], ids=["log1p", "cp10k_only"])
@pytest.mark.parametrize("sparse", [False, True])
def test_normalized_without_counts_marks_metrics_unavailable(log, sparse):
    norm = _log_cp10k(_counts(), sparse=sparse, log=log)

    result, stats, ir = QualityService().assess_quality(norm, **PARAMS)

    for column in COUNT_COLUMNS:
        assert result.obs[column].isna().all(), column
    assert stats["count_source"] is None
    assert stats["count_metrics_unavailable"] == [
        "total_counts",
        "mt_pct",
        "ribo_pct",
        "housekeeping_score",
    ]
    assert stats["qc_distributions"]["total_counts"]["median"] is None
    assert stats["qc_distributions"]["mt_pct"]["median"] is None
    assert stats["quality_status"] != "Pass"
    assert "need raw counts" in stats["quality_status"]
    assert "count-based metrics need raw counts" in stats["qc_summary"]
    assert "this modality is normalized" in stats["qc_summary"]
    assert "mitochondrial gene expression" not in stats["qc_summary"]
    # qc_pass is judged on genes per cell only, not on numbers that do not exist.
    n_genes = np.asarray((norm.X > 0).sum(axis=1)).ravel()
    np.testing.assert_array_equal(
        result.obs["qc_pass"].to_numpy(),
        (n_genes >= PARAMS["min_genes"]) & (n_genes <= PARAMS["max_genes"]),
    )
    pd.testing.assert_frame_equal(_replay(ir, norm).obs, result.obs)


@pytest.mark.parametrize("sparse", [False, True])
def test_raw_counts_modality_is_unchanged(sparse):
    raw = _counts()
    X = raw.X
    if sparse:
        raw.X = sp.csr_matrix(X)

    result, stats, ir = QualityService().assess_quality(raw, **PARAMS)

    totals = X.sum(axis=1)
    np.testing.assert_allclose(result.obs["n_counts"], totals)
    np.testing.assert_allclose(
        result.obs["mt_pct"], X[:, :2].sum(axis=1) / (totals + 1e-8) * 100
    )
    np.testing.assert_allclose(
        result.obs["ribo_pct"], X[:, 2:4].sum(axis=1) / (totals + 1e-8) * 100
    )
    np.testing.assert_allclose(result.obs["housekeeping_score"], X[:, 4:7].sum(axis=1))
    assert stats["count_source"] == "X"
    assert stats["count_metrics_unavailable"] == []
    assert stats["quality_status"] in ("Pass", "Warning")
    pd.testing.assert_frame_equal(_replay(ir, raw).obs, result.obs)
