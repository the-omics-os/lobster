"""Regression tests for the contaminant / reverse-hit flag filter dtype guard.

Assignment 1 (P0). The MS filters in ``shared_tools.py`` route an
``is_contaminant`` / ``is_reverse`` flag column to a boolean mask and drop the
flagged features (``adata[:, ~mask]``). The historical guard branched on
``dtype == object``; a ``StringDtype`` (pandas 2 opt-in) or the pandas-3
inferred ``str`` dtype fell through to ``astype(bool)``, where every non-empty
token — including ``'0'`` and ``'False'`` — becomes ``True`` and ``~mask``
empties the matrix.

Every case is asserted on **features retained**, not "no exception", and runs
under both ``pd.options.future.infer_string`` regimes via ``pandas_infer_string``
so the vulnerable path is exercised, not just the object path.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.core.utils.dtype_guards import boolean_flag_mask


def _col(values, encoding):
    """Build a flag column of `values` in the requested dtype encoding.

    `values` is a list over {True (flagged), False (not), None (missing)}.
    """
    tokens = {True: "+", False: "", None: None}
    nums = {True: 1, False: 0, None: None}
    if encoding == "text_plus":
        return pd.Series([tokens[v] for v in values])
    if encoding == "text_truefalse":
        return pd.Series(
            [{True: "True", False: "False", None: None}[v] for v in values]
        )
    if encoding == "text_10":
        return pd.Series([{True: "1", False: "0", None: None}[v] for v in values])
    if encoding == "string_dtype":
        return pd.Series(pd.array([tokens[v] for v in values], dtype="string"))
    if encoding == "bool":
        return pd.Series([False if v is None else v for v in values], dtype=bool)
    if encoding == "nullable_boolean":
        return pd.Series([pd.NA if v is None else v for v in values], dtype="boolean")
    if encoding == "numeric":
        return pd.Series(
            [np.nan if v is None else nums[v] for v in values], dtype="float64"
        )
    if encoding == "categorical_text":
        return pd.Series(pd.Categorical([tokens[v] for v in values]))
    if encoding == "categorical_numeric":
        return pd.Series(pd.Categorical([nums[v] for v in values]))
    if encoding == "categorical_bool":
        return pd.Series(
            pd.Categorical([None if v is None else bool(v) for v in values])
        )
    if encoding == "categorical_numeric_string":
        # Categorical over the STRINGS "0"/"1" (categories.dtype == object).
        return pd.Series(
            pd.Categorical([{True: "1", False: "0", None: None}[v] for v in values])
        )
    if encoding == "categorical_bool_string":
        # Categorical over the STRINGS "True"/"False" (categories.dtype == object).
        return pd.Series(
            pd.Categorical(
                [{True: "True", False: "False", None: None}[v] for v in values]
            )
        )
    raise AssertionError(encoding)


ENCODINGS = [
    "text_plus",
    "text_truefalse",
    "text_10",
    "string_dtype",
    "bool",
    "nullable_boolean",
    "numeric",
    "categorical_text",
    "categorical_numeric",
    "categorical_bool",
    "categorical_numeric_string",
    "categorical_bool_string",
]


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_flag_mask_marks_only_flagged(encoding, pandas_infer_string):
    # pattern: flagged, not, flagged, not  -> keep (~mask) = [F, T, F, T]
    col = _col([True, False, True, False], encoding)
    mask = boolean_flag_mask(col)
    assert list(mask) == [True, False, True, False], f"{encoding}: {list(mask)}"


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_missing_is_not_flagged(encoding, pandas_infer_string):
    if encoding == "bool":
        pytest.skip("numpy bool cannot represent missing")
    col = _col([True, None, False, None], encoding)
    mask = boolean_flag_mask(col)
    # flagged only at index 0; missing -> not flagged
    assert list(mask) == [True, False, False, False], f"{encoding}: {list(mask)}"


@pytest.mark.parametrize("encoding", ["string_dtype", "text_plus"])
def test_zero_and_false_tokens_are_not_flagged(encoding, pandas_infer_string):
    """The exact corruption case: '0' and 'False' must NOT count as flagged."""
    col = (
        pd.Series(pd.array(["0", "False", "+"], dtype="string"))
        if encoding == "string_dtype"
        else pd.Series(["0", "False", "+"])
    )
    mask = boolean_flag_mask(col)
    assert list(mask) == [False, False, True]


def test_categorical_without_empty_string_does_not_raise(pandas_infer_string):
    """Old code did ``fillna("")`` on a categorical; if "" was not already a
    category, pandas 2.3.3 raised TypeError. New code must handle it."""
    col = pd.Series(pd.Categorical(["+", "x", "+", "x"]))  # no "" category
    mask = boolean_flag_mask(col)
    assert list(mask) == [True, False, True, False]


@pytest.mark.parametrize(
    "encoding",
    ["string_dtype", "text_plus", "categorical_text", "categorical_numeric_string"],
)
def test_anndata_features_retained_through_filter(encoding, pandas_infer_string):
    """Integration: apply the exact call-site expression to a real AnnData and
    assert the non-contaminant features survive (not a zeroed matrix)."""
    X = np.arange(12, dtype=float).reshape(3, 4)
    var = pd.DataFrame(index=[f"P{i}" for i in range(4)])
    var["is_contaminant"] = _col([True, False, True, False], encoding).values
    adata = ad.AnnData(X=X, var=var)

    mask = boolean_flag_mask(adata.var["is_contaminant"])
    kept = adata[:, ~mask].copy()

    assert kept.n_vars == 2, f"{encoding}: matrix emptied to {kept.n_vars} vars"
    assert list(kept.var_names) == ["P1", "P3"]
