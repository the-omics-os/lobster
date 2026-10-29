"""Benign S6 marker-name serialization; no ranking-selection/science changes.

Native round trips cover the two standard Scanpy result keys only. Missing
name slots are empty text in the serialization copy, not fabricated genes.
Numeric scientific NaNs and the resident/source object must remain unchanged.
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from lobster.core.backends import h5ad_backend as backend_module
from lobster.core.backends.h5ad_backend import H5ADBackend
from lobster.core.utils import h5ad_utils

GROUPS = ("B", "A")
KEYS = ("rank_genes_groups", "rank_genes_groups_filtered")


def _ranking(missing):
    names = np.empty(4, dtype=[(group, object) for group in GROUPS]).view(np.recarray)
    names["B"] = ["GENE_B", missing, "nan", ""]
    names["A"] = [missing, "GENE_A_LONG", "nan", ""]
    numeric = np.empty(4, dtype=[(group, np.float64) for group in GROUPS]).view(
        np.recarray
    )
    numeric["B"] = [1.0, np.nan, 3.0, 4.0]
    numeric["A"] = [np.nan, 2.0, 3.0, 4.0]
    return {
        "names": names,
        "scores": numeric.copy(),
        "pvals": numeric.copy(),
        "pvals_adj": numeric.copy(),
        "logfoldchanges": numeric.copy(),
        "params": {"groupby": "leiden", "method": "wilcoxon", "reference": "rest"},
    }


def _adata(key, missing):
    adata = ad.AnnData(X=np.array([[1.25, 2.5], [3.25, 4.5]], dtype=np.float64))
    adata.obs_names = ["cell_0", "cell_1"]
    adata.var_names = ["GENE_B", "GENE_A_LONG"]
    adata.uns[key] = _ranking(missing)
    adata.uns["scientific_numeric_array"] = np.array([0.0, np.nan, 2.0])
    return adata


def _assert_numeric(actual, original):
    for name in ("scores", "pvals", "pvals_adj", "logfoldchanges"):
        assert actual[name].dtype == original[name].dtype
        assert actual[name].dtype.names == GROUPS
        for group in GROUPS:
            np.testing.assert_array_equal(actual[name][group], original[name][group])


def _assert_source_names(actual, original, missing):
    assert actual.dtype == original.dtype
    assert actual.shape == original.shape
    assert actual.dtype.names == GROUPS
    for group, missing_row in (("B", 1), ("A", 0)):
        for row in range(len(actual)):
            if row == missing_row:
                if missing is None:
                    assert actual[group][row] is None
                else:
                    assert np.isnan(actual[group][row])
            else:
                assert actual[group][row] == original[group][row]


def test_s6_modules_come_from_this_checkout():
    root = Path(__file__).resolve().parents[3]
    for module in (backend_module, h5ad_utils):
        assert Path(module.__file__).resolve().is_relative_to(root)


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize(
    "missing",
    [float("nan"), np.float32("nan"), None],
    ids=["float_nan", "numpy_nan", "none"],
)
def test_native_backend_round_trip_missing_marker_names_preserves_science(
    tmp_path, key, missing
):
    original = _adata(key, missing)
    snapshot = original.copy()
    target = tmp_path / "markers.h5ad"
    backend = H5ADBackend(base_path=tmp_path)

    try:
        backend.save(original, target)
    finally:
        # Also checked on RED: a failed write may not repair the resident source.
        _assert_source_names(
            original.uns[key]["names"], snapshot.uns[key]["names"], missing
        )
        _assert_numeric(original.uns[key], snapshot.uns[key])
        np.testing.assert_array_equal(original.X, snapshot.X)
        assert list(original.obs_names) == list(snapshot.obs_names)
        assert list(original.var_names) == list(snapshot.var_names)

    restored = ad.read_h5ad(target)
    names = restored.uns[key]["names"]
    assert names.shape == (4,)
    assert names.dtype.names == GROUPS
    assert names["B"].tolist() == ["GENE_B", "", "nan", ""]
    assert names["A"].tolist() == ["", "GENE_A_LONG", "nan", ""]
    _assert_numeric(restored.uns[key], snapshot.uns[key])
    np.testing.assert_array_equal(
        restored.uns["scientific_numeric_array"],
        snapshot.uns["scientific_numeric_array"],
    )
    np.testing.assert_array_equal(restored.X, snapshot.X)
    assert set(restored.uns) == set(snapshot.uns)


@pytest.mark.parametrize("key", KEYS)
def test_serialization_sanitizer_repairs_names_on_copy_only(key):
    original = _adata(key, float("nan"))
    snapshot = original.copy()
    serialized = H5ADBackend.sanitize_anndata(original.copy())
    assert serialized.uns[key]["names"]["B"].tolist() == ["GENE_B", "", "nan", ""]
    assert serialized.uns[key]["names"]["A"].tolist() == ["", "GENE_A_LONG", "nan", ""]
    _assert_source_names(
        original.uns[key]["names"], snapshot.uns[key]["names"], float("nan")
    )
    _assert_numeric(serialized.uns[key], snapshot.uns[key])


def test_unknown_object_name_payload_is_not_blanket_repaired():
    original = _adata("unrelated_payload", float("nan"))
    snapshot = original.copy()
    serialized = H5ADBackend.sanitize_anndata(original.copy())
    _assert_source_names(
        serialized.uns["unrelated_payload"]["names"],
        snapshot.uns["unrelated_payload"]["names"],
        float("nan"),
    )
    _assert_numeric(
        serialized.uns["unrelated_payload"], snapshot.uns["unrelated_payload"]
    )


def test_existing_valid_text_marker_names_round_trip_is_unchanged(tmp_path):
    original = _adata("rank_genes_groups", "VALID_LABEL")
    snapshot = original.copy()
    target = tmp_path / "valid.h5ad"
    H5ADBackend(base_path=tmp_path).save(original, target)
    restored = ad.read_h5ad(target)
    for group in GROUPS:
        assert (
            restored.uns["rank_genes_groups"]["names"][group].tolist()
            == snapshot.uns["rank_genes_groups"]["names"][group].tolist()
        )
    _assert_numeric(
        restored.uns["rank_genes_groups"], snapshot.uns["rank_genes_groups"]
    )
