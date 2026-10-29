"""Real Scanpy marker producer -> native write and recorded IR, benign S6 only.

Original-versus-filtered ranking/count/extraction policy intentionally stays
outside these assertions (S7). No model invocation or malicious fixture.
"""

from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from lobster.core.backends.h5ad_backend import H5ADBackend
from lobster.services.analysis import enhanced_singlecell_service as service_module
from lobster.services.analysis.enhanced_singlecell_service import (
    EnhancedSingleCellService,
)
from lobster.testing.provenance_replay import replay_ir

NUMERIC_FIELDS = ("scores", "pvals", "pvals_adj", "logfoldchanges")


def _prepared(raw):
    rng = np.random.default_rng(17)
    # Every gene is positive in every cell: max_out_pct=0 eliminates all
    # filtered names deterministically, without engineering a hostile payload.
    values = np.log1p(rng.uniform(1.1, 3.3, (16, 6)))
    adata = ad.AnnData(values)
    adata.obs_names = [f"cell_{i}" for i in range(16)]
    adata.var_names = [f"GENE_{i}" for i in range(6)]
    adata.obs["leiden"] = pd.Categorical(["B"] * 8 + ["A"] * 8, categories=["B", "A"])
    adata.uns["log1p"] = {"base": None}
    if raw:
        adata.raw = adata.copy()
    return adata


def _reference(source):
    result = source.copy()
    sc.tl.rank_genes_groups(
        result,
        groupby="leiden",
        method="wilcoxon",
        n_genes=4,
        use_raw=source.raw is not None,
    )
    sc.tl.filter_rank_genes_groups(
        result,
        min_fold_change=1.5,
        min_in_group_fraction=0.25,
        max_out_group_fraction=0.0,
    )
    return result


def _assert_numeric_and_original_ranking(actual, reference):
    for key in ("rank_genes_groups", "rank_genes_groups_filtered"):
        assert set(actual.uns[key]) == set(reference.uns[key])
        for field in NUMERIC_FIELDS:
            assert actual.uns[key][field].dtype == reference.uns[key][field].dtype
            for group in reference.uns[key][field].dtype.names:
                np.testing.assert_array_equal(
                    actual.uns[key][field][group], reference.uns[key][field][group]
                )
    for group in reference.uns["rank_genes_groups"]["names"].dtype.names:
        assert (
            actual.uns["rank_genes_groups"]["names"][group].tolist()
            == reference.uns["rank_genes_groups"]["names"][group].tolist()
        )


def test_marker_producer_origin_is_this_checkout():
    root = Path(__file__).resolve().parents[4]
    assert Path(service_module.__file__).resolve().is_relative_to(root)


@pytest.mark.parametrize("raw", [False, True], ids=["x", "prepared_raw"])
def test_real_marker_producer_has_write_safe_names_without_science_rewrite(
    tmp_path, raw
):
    source = _prepared(raw)
    snapshot = source.copy()
    reference = _reference(source)
    for group in reference.uns["rank_genes_groups_filtered"]["names"].dtype.names:
        assert pd.isna(
            reference.uns["rank_genes_groups_filtered"]["names"][group]
        ).all()

    with (
        patch.object(sc.tl, "rank_genes_groups", wraps=sc.tl.rank_genes_groups) as rank,
        patch.object(
            sc.tl, "filter_rank_genes_groups", wraps=sc.tl.filter_rank_genes_groups
        ) as filter_markers,
    ):
        result, stats, ir = EnhancedSingleCellService().find_marker_genes(
            source, groupby="leiden", n_genes=4, max_out_pct=0.0
        )
    assert rank.call_count == 1 and filter_markers.call_count == 1
    assert rank.call_args.kwargs == {
        "groupby": "leiden",
        "method": "wilcoxon",
        "n_genes": 4,
        "use_raw": raw,
        "key_added": "rank_genes_groups",
    }
    assert filter_markers.call_args.kwargs == {
        "key": "rank_genes_groups",
        "key_added": "rank_genes_groups_filtered",
        "min_fold_change": 1.5,
        "min_in_group_fraction": 0.25,
        "max_out_group_fraction": 0.0,
    }
    _assert_numeric_and_original_ranking(result, reference)
    names = result.uns["rank_genes_groups_filtered"]["names"]
    assert names.shape == reference.uns["rank_genes_groups_filtered"]["names"].shape
    assert (
        names.dtype.names
        == reference.uns["rank_genes_groups_filtered"]["names"].dtype.names
    )
    for group in names.dtype.names:
        assert names[group].tolist() == [""] * 4
    assert (
        "rank_genes_groups" not in source.uns
        and "rank_genes_groups_filtered" not in source.uns
    )
    np.testing.assert_array_equal(source.X, snapshot.X)
    np.testing.assert_array_equal(result.X, snapshot.X)
    if raw:
        np.testing.assert_array_equal(source.raw.X, snapshot.raw.X)
        np.testing.assert_array_equal(result.raw.X, snapshot.raw.X)

    # No backend sanitizer is credited here: the producer object writes directly.
    direct = tmp_path / "producer.h5ad"
    result.write_h5ad(direct)
    restored = ad.read_h5ad(direct)
    _assert_numeric_and_original_ranking(restored, reference)
    H5ADBackend(base_path=tmp_path).save(result, "backend.h5ad")

    # Execute the actual recorded imports/helpers, not an invented replay API.
    namespace = {"adata": source.copy()}
    for statement in ir.imports:
        exec(compile(statement, "<marker-ir-import>", "exec"), namespace)
    for helper in getattr(ir, "helper_code", []):
        exec(compile(helper, "<marker-ir-helper>", "exec"), namespace)
    replayed = replay_ir(ir, **namespace)["adata"]
    _assert_numeric_and_original_ranking(replayed, reference)
    for group in names.dtype.names:
        assert (
            replayed.uns["rank_genes_groups_filtered"]["names"][group].tolist()
            == [""] * 4
        )
    replayed.write_h5ad(tmp_path / "replayed.h5ad")
    assert stats["filtering_params"] == {
        "min_fold_change": 1.5,
        "min_pct": 0.25,
        "max_out_pct": 0.0,
    }
