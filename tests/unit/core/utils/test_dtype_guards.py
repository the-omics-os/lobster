"""Tests for the canonical dtype guards (lobster.core.utils.dtype_guards).

Covers the two predicates that replace ``dtype == "object"`` across the engine,
including the object-with-NaN and object-numeric edge cases that a naive
``is_string_dtype``-only swap gets wrong.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from lobster.core.utils.dtype_guards import boolean_flag_mask, is_text_dtype


@pytest.mark.parametrize(
    "data,expected",
    [
        (pd.Series(["a", "b"], dtype=object), True),
        (pd.Series(["a", np.nan, "b"], dtype=object), True),   # object-with-NaN
        (pd.Series([1, 2, None], dtype=object), True),         # non-string object
        (pd.Series(pd.array(["a", "b"], dtype="string")), True),
        (pd.Series([1.0, 2.0]), False),
        (pd.Series([True, False]), False),
        (pd.Series([1, 2], dtype="Int64"), False),
        (pd.Series(pd.Categorical(["a", "b"])), True),
        (pd.Series(pd.Categorical([1, 2])), False),
    ],
)
def test_is_text_dtype(data, expected):
    # is_text_dtype must preserve the old ``== object`` reach (incl. object with
    # NaN/mixed) AND additionally recognize StringDtype / pandas-3 str.
    assert is_text_dtype(data) == expected


def _keep(col):
    return list(~boolean_flag_mask(col))


def test_flag_mask_object_numeric_values():
    # object dtype holding python numbers: nonzero flagged, 0/NaN not.
    assert list(boolean_flag_mask(pd.Series([2, 0, None], dtype=object))) == [True, False, False]
    assert list(boolean_flag_mask(pd.Series([1.0, 0.0, np.nan], dtype=object))) == [True, False, False]


def test_flag_mask_object_string_with_nan():
    col = pd.Series(["+", np.nan, "0"], dtype=object)
    assert list(boolean_flag_mask(col)) == [True, False, False]


def test_flag_mask_all_missing_and_empty():
    assert list(boolean_flag_mask(pd.Series([None, None], dtype=object))) == [False, False]
    assert list(boolean_flag_mask(pd.Series([], dtype=object))) == []


@pytest.mark.parametrize(
    "col,expected",
    [
        (pd.Series(["+", "", "+", ""]), [True, False, True, False]),
        (pd.Series(["1", "0", "1", "0"]), [True, False, True, False]),
        (pd.Series(["True", "False", "True", "False"]), [True, False, True, False]),
        (pd.Series(["YES", "no", "y", "Nope"]), [True, False, False, False]),  # 'y' not a token
        (pd.Series([True, False, True, False]), [True, False, True, False]),
        (pd.Series([True, pd.NA, False], dtype="boolean"), [True, False, False]),
        (pd.Series([1, 0, 2, 0]), [True, False, True, False]),
        (pd.Series([1.0, np.nan, 0.0]), [True, False, False]),
        (pd.Series(pd.array(["+", "", "+"], dtype="string")), [True, False, True]),
        (pd.Series(pd.Categorical(["+", "", "+"])), [True, False, True]),
        (pd.Series(pd.Categorical([1, 0, 1])), [True, False, True]),
        (pd.Series(pd.Categorical([True, False])), [True, False]),
    ],
)
def test_flag_mask_matrix_no_downcast_warning(col, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        assert list(boolean_flag_mask(col)) == expected
