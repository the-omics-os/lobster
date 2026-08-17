"""Semantic dtype predicates for pandas 2/3 compatibility.

Historically, code across the engine asked "is this column textual?" by testing
``series.dtype == object``. That is a type-identity check standing in for a
semantic question, and it breaks in two directions:

* It misses ``StringDtype`` (``"string"``, the pandas-2 opt-in) and the inferred
  ``str`` dtype that pandas 3 produces by default (PDEP-14). A string-typed flag
  column then slips past the guard and, in the proteomics filters, reaches
  ``astype(bool)`` — where every non-empty token, including ``"0"`` and
  ``"False"``, becomes ``True`` and ``adata[:, ~mask]`` empties the matrix.

This module is the single, tested home for the replacement predicates so the
same fix is not re-derived (and re-broken) in six places. ``core`` owns it;
packages import from here.

Note on ``is_string_dtype`` alone: it is **False** for an ``object`` Series that
contains any non-string element (``NaN``, mixed types) — verified under pandas
2.3.3 and 3.0.5. So it is *not* a drop-in replacement for ``== object``: using
it by itself silently skips object columns with missing values, which are
common. Use :func:`is_text_dtype`, which also accepts plain ``object``.
"""
from __future__ import annotations

import pandas as pd
from pandas.api import types as pdt

# MaxQuant-style presence flags ("+" means yes); matched case-insensitively.
_FLAG_TRUE_TOKENS = frozenset({"+", "true", "1", "yes"})


def is_text_dtype(obj) -> bool:
    """True for columns that may hold text: ``object`` (including
    object-with-NaN and mixed-type object), pandas ``StringDtype``, and the
    pandas-3 ``str`` dtype. Accepts a Series, Index, array, or dtype.

    This is the faithful replacement for ``dtype == "object"``: it preserves the
    old object-column behavior *and* additionally recognizes the real string
    dtypes that ``== object`` missed. ``is_string_dtype`` alone would drop
    object columns that contain missing values.
    """
    return pdt.is_object_dtype(obj) or pdt.is_string_dtype(obj)


def boolean_flag_mask(col: pd.Series) -> pd.Series:
    """Normalize a presence/absence flag column (``is_contaminant``,
    ``is_reverse``, ``is_chimera``, ``qc_pass`` …) to a boolean mask, routing on
    the *meaning* of the dtype rather than on ``dtype == object``.

    Flag columns arrive in many encodings depending on the writer and on the
    pandas/anndata versions that round-tripped the artifact: genuine ``bool``,
    numeric ``0``/``1``, text tokens (``+``, ``True``, ``1``, ``yes``), and
    categorical wrappers around any of those. Truthy ⇔ ``+`` / ``true`` (any
    case) / ``1`` / ``yes`` for text, any nonzero for numeric, ``True`` for
    bool; missing (``NaN``/``NA``) is never flagged.

    Verified correct — and free of downcast ``FutureWarning``s — under pandas
    2.3.3 and 3.0.5.
    """

    def _from_text(series: pd.Series) -> pd.Series:
        # object / StringDtype / pandas-3 str: fillna("") does not downcast.
        return (
            series.fillna("").astype(str).str.strip().str.lower().isin(_FLAG_TRUE_TOKENS)
        )

    def _from_numeric(series: pd.Series) -> pd.Series:
        # Missing -> not flagged; any nonzero -> flagged. Coerce to float first
        # so fillna(0) never lands in a bool/int dtype that would reject it.
        # Flags are real-valued: a complex column's imaginary part is dropped by
        # the float cast (with a ComplexWarning), so a purely-imaginary value
        # reads as 0 — acceptable, since flag columns are never complex.
        return (
            pd.to_numeric(series, errors="coerce").astype("float64").fillna(0.0).ne(0.0)
        )

    def _from_bool(series: pd.Series) -> pd.Series:
        # Operate in nullable-boolean space to avoid the object-dtype downcast
        # warning that plain ``object.fillna(False)`` raises.
        return series.astype("boolean").fillna(False).astype(bool)

    # Categorical hides the value type on the flat predicates (a categorical of
    # ints reports neither numeric nor string), so decide from the category
    # dtype and operate on a NaN-safe object materialization.
    if isinstance(col.dtype, pd.CategoricalDtype):
        categories_dtype = col.cat.categories.dtype
        values = col.astype(object)
        if pdt.is_bool_dtype(categories_dtype):
            return _from_bool(values)
        if pdt.is_numeric_dtype(categories_dtype):
            return _from_numeric(values)
        return _from_text(values)

    # bool must precede numeric: is_numeric_dtype is True for bool dtypes.
    if pdt.is_bool_dtype(col):
        return _from_bool(col)
    if pdt.is_numeric_dtype(col):
        return _from_numeric(col)
    # Pure string dtypes (object-all-str, StringDtype, pandas-3 str).
    if pdt.is_string_dtype(col):
        return _from_text(col)

    # Remaining: object-with-NaN, or object holding python numbers / mixed types
    # (is_string_dtype is False for these). Prefer numeric when every
    # non-missing value is numeric-coercible; otherwise treat as text tokens.
    # Never fall through to astype(bool).
    coerced = pd.to_numeric(col, errors="coerce")
    non_missing = col.notna()
    if non_missing.sum() == 0 or bool((coerced.notna() | ~non_missing).all()):
        return _from_numeric(col)
    return _from_text(col)
