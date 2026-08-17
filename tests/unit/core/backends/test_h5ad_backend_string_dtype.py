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

from lobster.core.backends.h5ad_backend import H5ADBackend


def _adata(var: pd.DataFrame) -> ad.AnnData:
    var.index = [f"g{i}" for i in range(len(var))]
    return ad.AnnData(X=np.ones((2, len(var)), dtype=float), var=var)


def test_string_id_column_is_preserved_not_coerced():
    # StringDtype numeric-looking IDs must survive the sanitizer unchanged.
    var = pd.DataFrame({"sample_id": pd.array(["001", "002", "003"], dtype="string")})
    adata = _adata(var)
    H5ADBackend.sanitize_anndata(adata)
    assert list(adata.var["sample_id"]) == ["001", "002", "003"], (
        f"leading-zero string IDs were corrupted: {list(adata.var['sample_id'])}"
    )


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
