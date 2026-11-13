"""A SHA-256 over what an AnnData object holds, independent of how it is stored.

Version ``v2`` covers the whole object except ``uns``: ``X``, the cell and gene names,
the ``obs`` and ``var`` columns, ``layers``, ``obsm``/``varm``, ``obsp``/``varp`` and
``raw``. ``uns`` is left out on purpose: it holds the session's own records, colours
and parameters, and a copy that differs only there is the same loaded data.

Every part is encoded with a kind tag and exact values, so that two objects share a
digest only when they hold the same values:

* matrices (dense of either memory order, CSR, CSC, COO) become one canonical form:
  per row block, the number of stored entries per row, their column indices (int32) and
  their values. Entries equal to ``+0.0`` are dropped; ``-0.0`` is kept;
* booleans are tagged ``b``; signed integers ``i`` (int64) and unsigned ``u`` (uint64),
  exactly; floats ``f`` (float64: exact for float16 and float32) with one NaN; complex
  ``c`` (complex128) with one NaN per part; no value is converted to another kind;
* strings are UTF-8, length-prefixed, with a distinct mark for a missing value; a
  categorical column is hashed as its values, so a column and its categorical copy
  (what an h5ad file holds) agree;
* a part of a type not listed here makes the digest fail (``ValueError``), never guess.

Nothing is copied in full: matrices are read per block of rows.
"""

import hashlib
from typing import Any

import numpy as np
import scipy.sparse as sp

# Off for 1.3.0: an object loaded from memory is not anchored to an --input, and no
# digest is computed at load. Read by the load (data_manager) and by the replay check.
IN_MEMORY_INPUTS = False

DIGEST = "v2"
SCOPE = "X+obs_names+var_names+obs+var+layers+obsm+varm+obsp+varp+raw " "(uns excluded)"
_PREFIX = f"lobster-content-{DIGEST}".encode()
_BLOCK = 1 << 22  # matrix elements per block
_NAN = np.float64("nan")
_MISSING = (1 << 64) - 1


def _tag(h, text: str) -> None:
    encoded = text.encode("utf-8")
    h.update(len(encoded).to_bytes(8, "little"))
    h.update(encoded)


def _names(h, names: Any) -> None:
    h.update(len(names).to_bytes(8, "little"))
    for name in names:
        _string(h, name)


def _string(h, value: Any) -> None:
    if value is None or (isinstance(value, float) and value != value):
        h.update(_MISSING.to_bytes(8, "little"))
        return
    if type(value).__name__ == "NAType":
        h.update(_MISSING.to_bytes(8, "little"))
        return
    if not isinstance(value, str):
        raise ValueError(f"not a string: {type(value).__name__}")
    encoded = value.encode("utf-8", "surrogatepass")
    h.update(len(encoded).to_bytes(8, "little"))
    h.update(encoded)


def _kind(dtype: np.dtype) -> str:
    kind = dtype.kind
    if kind not in "biufc" or dtype.itemsize > (16 if kind == "c" else 8):
        raise ValueError(f"not an exact kind: {dtype}")
    return kind


def _values(h, kind: str, values: np.ndarray) -> None:
    """Hash numeric values in their kind (no conversion to another kind)."""
    if kind == "b":
        out = values.astype(np.uint8)
    elif kind == "i":
        out = values.astype(np.int64, copy=False)
    elif kind == "u":
        out = values.astype(np.uint64, copy=False)
    elif kind == "f":
        out = values.astype(np.float64)
        out[np.isnan(out)] = _NAN
    else:  # c
        out = values.astype(np.complex128)
        view = out.view(np.float64)
        view[np.isnan(view)] = _NAN
    h.update(memoryview(np.ascontiguousarray(out).reshape(-1)).cast("B"))


def _stored(kind: str, values: np.ndarray) -> np.ndarray:
    """Which entries a canonical matrix holds: everything but ``+0.0``."""
    if kind == "f":
        nonzero = values != 0
        # -0.0 is kept: only look at signs when some entry is zero
        return nonzero if nonzero.all() else nonzero | np.signbit(values)
    if kind == "c":
        return (values != 0) | np.signbit(values.real) | np.signbit(values.imag)
    return values != 0


def _matrix(h, matrix: Any) -> None:
    """A 2-D matrix of any layout in one canonical form, block by block of rows."""
    if sp.issparse(matrix):
        matrix = matrix if sp.isspmatrix_csr(matrix) else matrix.tocsr()
        kind = _kind(matrix.dtype)
    else:
        matrix = np.asarray(matrix)
        if matrix.ndim != 2:
            raise ValueError("not a matrix")
        kind = _kind(matrix.dtype)
    rows, cols = matrix.shape
    _tag(h, f"matrix:{kind}:{rows}x{cols}")
    step = max(1, _BLOCK // max(cols, 1))
    for start in range(0, rows, step):
        stop = min(start + step, rows)
        if sp.issparse(matrix):
            block = matrix[start:stop]
            block.sum_duplicates()
            block.sort_indices()
            counts = np.diff(block.indptr)
            row_of = np.repeat(np.arange(stop - start), counts)
            keep = _stored(kind, block.data)
            data, where, row_of = block.data[keep], block.indices[keep], row_of[keep]
        else:
            block = matrix[start:stop]
            keep = _stored(kind, block)
            if keep.all():  # a dense block without zeros: every entry, row by row
                height = stop - start
                counts = np.full(height, cols, dtype=np.int64)
                where = np.tile(np.arange(cols, dtype=np.int32), height)
                h.update(counts.tobytes())
                h.update(where.tobytes())
                _values(h, kind, block)
                continue
            row_of, where = np.nonzero(keep)
            data = block[keep]
        h.update(np.bincount(row_of, minlength=stop - start).astype(np.int64).tobytes())
        h.update(where.astype(np.int32).tobytes())
        _values(h, kind, data)


def _column(h, series: Any) -> None:
    import pandas as pd

    values = series
    if isinstance(series.dtype, pd.CategoricalDtype):
        values = series.astype(object)
    dtype = values.dtype
    if dtype == np.dtype(object) or pd.api.types.is_string_dtype(dtype):
        _tag(h, "strings")
        _names(h, values.to_numpy(dtype=object))
        return
    if pd.api.types.is_extension_array_dtype(dtype):
        mask = values.isna().to_numpy()
        base = values.fillna(0).to_numpy()
        _tag(h, "masked")
        h.update(mask.astype(np.uint8).tobytes())
        _tag(h, f"values:{_kind(base.dtype)}")
        _values(h, _kind(base.dtype), base)
        return
    array = values.to_numpy()
    _tag(h, f"values:{_kind(array.dtype)}")
    _values(h, _kind(array.dtype), array)


def _frame(h, frame: Any) -> None:
    _tag(h, f"frame:{len(frame)}")
    names = sorted(map(str, frame.columns))
    if len(set(names)) != len(frame.columns):
        raise ValueError("column names repeat")
    _names(h, names)
    for name in names:
        _column(h, frame[name])


def _part(h, value: Any) -> None:
    import pandas as pd

    if value is None:
        _tag(h, "none")
    elif isinstance(value, pd.DataFrame):
        _frame(h, value)
    else:
        _matrix(h, value)


def _mapping(h, label: str, mapping: Any) -> None:
    keys = sorted(k for k in mapping.keys() if k is not None)  # None aliases X
    _tag(h, f"{label}:{len(keys)}")
    _names(h, keys)
    for key in keys:
        _part(h, mapping[key])


def content_digest(adata: Any) -> str:
    """Hex SHA-256 over the whole object but ``uns`` (see the module). ``ValueError``
    when a part cannot be encoded exactly."""
    if getattr(adata, "isbacked", False):
        raise ValueError("backed data")
    h = hashlib.sha256()
    h.update(_PREFIX)
    _tag(h, "X")
    _part(h, adata.X)
    _tag(h, "obs_names")
    _names(h, adata.obs_names)
    _tag(h, "var_names")
    _names(h, adata.var_names)
    _tag(h, "obs")
    _frame(h, adata.obs)
    _tag(h, "var")
    _frame(h, adata.var)
    for label in ("layers", "obsm", "varm", "obsp", "varp"):
        _mapping(h, label, getattr(adata, label))
    raw = getattr(adata, "raw", None)
    _tag(h, "raw")
    if raw is None:
        _tag(h, "none")
    else:
        _part(h, raw.X)
        _names(h, raw.var_names)
        _frame(h, raw.var)
        _mapping(h, "raw.varm", raw.varm)
    return h.hexdigest()
