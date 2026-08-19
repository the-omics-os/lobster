"""The H5AD write sanitizer must not corrupt string identifier columns.

``sanitize_anndata``'s object branch calls ``pd.to_numeric`` on single-type
columns, which rewrites numeric-looking text IDs ("001" -> 1, losing the
leading zero). That branch is gated on ``dtype == "object"`` and is deliberately
NOT broadened to pandas-3 ``str`` / ``StringDtype``: those columns are natively
H5AD-writable, so routing them through the coercion would corrupt identifiers
for no benefit. These tests pin that decision (a future broadening would
re-introduce the corruption) and confirm the object mixed-type path still
produces a writable object.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.core.backends.h5ad_backend import H5ADBackend
from lobster.core.utils.h5ad_utils import is_arrow_dtype


def _adata(var: pd.DataFrame) -> ad.AnnData:
    var.index = [f"g{i}" for i in range(len(var))]
    return ad.AnnData(X=np.ones((2, len(var)), dtype=float), var=var)


def test_string_id_column_is_preserved_not_coerced():
    # StringDtype numeric-looking IDs must survive the sanitizer unchanged.
    var = pd.DataFrame({"sample_id": pd.array(["001", "002", "003"], dtype="string")})
    adata = _adata(var)
    H5ADBackend.sanitize_anndata(adata)
    assert list(adata.var["sample_id"]) == [
        "001",
        "002",
        "003",
    ], f"leading-zero string IDs were corrupted: {list(adata.var['sample_id'])}"


def test_sanitized_mixed_object_column_is_writable(tmp_path):
    # Mixed-type object column: the sanitizer stringifies it (None -> "NA") so it
    # can be written. Characterizes the object path the gate intentionally keeps.
    var = pd.DataFrame({"mixed": pd.Series(["ok", 1, None], dtype=object)})
    adata = _adata(var)
    H5ADBackend.sanitize_anndata(adata)
    out = tmp_path / "sanitized.h5ad"
    adata.write_h5ad(out)  # must not raise
    back = ad.read_h5ad(out)
    assert back.n_vars == 3
    assert list(back.var["mixed"]) == ["ok", "1", "NA"]


def test_object_leading_zero_ids_are_preserved(tmp_path):
    # OBJECT-dtype (not StringDtype) numeric-looking IDs with leading zeros hit
    # the sanitizer's object branch, where the old code ran unconditional
    # pd.to_numeric and rewrote "001" -> 1. The conservative coercion must leave
    # them untouched. This is the pre-existing corruption tracked as a follow-on.
    var = pd.DataFrame({"sample_id": pd.Series(["001", "002", "003"], dtype=object)})
    adata = _adata(var)
    H5ADBackend.sanitize_anndata(adata)
    assert list(adata.var["sample_id"]) == [
        "001",
        "002",
        "003",
    ], f"object leading-zero IDs were coerced: {list(adata.var['sample_id'])}"
    out = tmp_path / "ids.h5ad"
    adata.write_h5ad(out)  # must remain writable
    assert list(ad.read_h5ad(out).var["sample_id"]) == ["001", "002", "003"]


@pytest.mark.parametrize(
    "values",
    [
        ["001", "002"],  # leading zeros
        [" 1 ", "2 "],  # surrounding whitespace (identity includes spaces)
        ["1", ""],  # empty-string token (parses to NaN, must not coerce)
        ["1.50", "2.50"],  # non-canonical decimal spelling
        ["1e3", "2e3"],  # exponent form
        ["+1", "+2"],  # explicit sign
    ],
)
def test_object_non_roundtrip_strings_are_preserved(values):
    # Numeric-looking object strings whose text does not survive coercion must
    # be kept verbatim (identifier safety). Regression guard for the Codex
    # findings: whitespace/empty-string false-accept and exponent/sign forms.
    var = pd.DataFrame({"col": pd.Series(values, dtype=object)})
    adata = _adata(var)
    H5ADBackend.sanitize_anndata(adata)
    assert list(adata.var["col"]) == values


@pytest.mark.parametrize(
    "values,expected",
    [
        (["1", "2", "3"], [1, 2, 3]),
        (["1.0", "2.0"], [1.0, 2.0]),  # canonical float spelling must coerce
        (["0.0", "5.0"], [0.0, 5.0]),
        (["9223372036854775807", "1"], [9223372036854775807, 1]),  # exact int64
    ],
)
def test_object_canonical_numeric_strings_coerce(values, expected):
    # Canonical numeric spellings still coerce — no false-reject from the guard.
    var = pd.DataFrame({"col": pd.Series(values, dtype=object)})
    adata = _adata(var)
    H5ADBackend.sanitize_anndata(adata)
    assert list(adata.var["col"]) == expected


def test_object_string_infinity_does_not_crash(tmp_path):
    # Regression: the lossless helper must not call int() on float infinity
    # (OverflowError). "inf"/"-inf" coerce to numeric infinities as before.
    var = pd.DataFrame({"col": pd.Series(["inf", "-inf"], dtype=object)})
    adata = _adata(var)
    H5ADBackend.sanitize_anndata(adata)  # must not raise
    out = tmp_path / "inf.h5ad"
    adata.write_h5ad(out)  # must remain writable


def test_is_arrow_dtype_recognizes_string_extension_dtypes():
    # StringDtype (and, under pandas 3, the default `str` dtype) must be
    # recognized so convert_arrow_to_standard materializes it to object before
    # write. Plain object / numeric must NOT be treated as extension dtypes.
    assert is_arrow_dtype(pd.Series(pd.array(["a", "b"], dtype="string"))) is True
    assert is_arrow_dtype(pd.Index(pd.array(["g0", "g1"], dtype="string"))) is True
    assert is_arrow_dtype(pd.Series(["a", "b"], dtype=object)) is False
    assert is_arrow_dtype(pd.Series([1, 2, 3])) is False


def test_save_roundtrips_string_dtype_column_and_index(tmp_path):
    # Full H5ADBackend.save() path (P0). A StringDtype var column + index must
    # write and read back with values preserved. If is_arrow_dtype fails to
    # recognize the extension dtype, save() forces infer_string=False and anndata
    # refuses to write the nullable-string arrays.
    var = pd.DataFrame(
        {"sample_id": pd.array(["001", "002", "003"], dtype="string")},
        index=pd.Index(pd.array(["g0", "g1", "g2"], dtype="string"), name="gene_id"),
    )
    adata = ad.AnnData(X=np.ones((2, 3), dtype=float), var=var)
    backend = H5ADBackend(base_path=str(tmp_path))
    out = tmp_path / "string_dtype.h5ad"
    backend.save(adata, str(out))  # must not raise
    back = ad.read_h5ad(out)
    assert list(back.var["sample_id"]) == ["001", "002", "003"]
    assert list(back.var_names) == ["g0", "g1", "g2"]
