"""Regression tests for intensity/metadata column classification dtype guard.

Assignment 1 (P0). ``_separate_intensity_metadata_columns`` in the proteomics
and metabolomics adapters classifies a column as metadata when it is textual
and not numeric-encoded, otherwise treats it as intensity and runs
``pd.to_numeric(errors="coerce")`` over it.

The historical guard tested ``df[col].dtype == "object"``. Under pandas 3 a
text column has dtype ``str`` (and under pandas 2 an explicit ``StringDtype``
is ``string``), so a text metadata column that did not match a name pattern was
NOT recognized as metadata, stayed in the intensity set, and was silently
coerced to all-NaN. These tests assert the text column lands in metadata under
both ``pd.options.future.infer_string`` regimes.
"""

import numpy as np
import pandas as pd
import pytest

from lobster.core.adapters.metabolomics_adapter import MetabolomicsAdapter
from lobster.core.adapters.proteomics_adapter import ProteomicsAdapter


def _adapter(cls):
    # __init__ needs schema/config wiring we don't want here; the method under
    # test only uses self._is_numeric_string_column, which is stateless.
    return cls.__new__(cls)


@pytest.mark.parametrize("cls", [ProteomicsAdapter, MetabolomicsAdapter])
def test_text_column_classified_as_metadata(cls, pandas_infer_string):
    # "extra_col" matches no name pattern in either adapter, so classification
    # falls to the dtype branch — exactly the path that regressed under pandas 3.
    df = pd.DataFrame(
        {
            "extra_col": pd.Series(["batchA", "batchB", "batchA"]),  # real text
            "sample_1": [10.0, 20.0, 30.0],
            "sample_2": [1.5, 2.5, 3.5],
            "num_as_text": pd.Series(["1.1", "2.2", "3.3"]),  # numeric-encoded
        }
    )
    adapter = _adapter(cls)
    intensity_df, metadata_df = adapter._separate_intensity_metadata_columns(df)

    assert metadata_df is not None
    assert "extra_col" in metadata_df.columns
    # Must NOT leak into intensity (where to_numeric would blank it to all-NaN).
    assert "extra_col" not in intensity_df.columns
    # Numeric-encoded text is intensity and survives coercion (not all-NaN).
    assert "num_as_text" in intensity_df.columns
    assert not intensity_df["num_as_text"].isna().all()
    assert {"sample_1", "sample_2"}.issubset(intensity_df.columns)


@pytest.mark.parametrize("cls", [ProteomicsAdapter, MetabolomicsAdapter])
def test_object_metadata_with_missing_is_not_blanked(cls, pandas_infer_string):
    """Regression: object-dtype text with a missing value. is_string_dtype is
    False for object-with-NaN, so an is_string_dtype-only guard would misroute
    this to intensity and blank it; is_text_dtype keeps it as metadata."""
    df = pd.DataFrame(
        {
            "extra_col": pd.Series(["batchA", None, "batchB"], dtype=object),
            "s1": [1.0, 2.0, 3.0],
        }
    )
    intensity_df, metadata_df = _adapter(cls)._separate_intensity_metadata_columns(df)
    assert metadata_df is not None and "extra_col" in metadata_df.columns
    assert "extra_col" not in intensity_df.columns


@pytest.mark.parametrize("cls", [ProteomicsAdapter, MetabolomicsAdapter])
def test_numeric_string_column_not_blanked(cls, pandas_infer_string):
    """A column of numeric-encoded strings must become real numbers, never NaN."""
    df = pd.DataFrame(
        {
            "label": pd.Series(["m1", "m2"]),
            "abundance": pd.Series(["100.0", "200.0"]),
        }
    )
    intensity_df, metadata_df = _adapter(cls)._separate_intensity_metadata_columns(df)
    assert "abundance" in intensity_df.columns
    assert list(intensity_df["abundance"]) == [100.0, 200.0]
    assert metadata_df is not None and "label" in metadata_df.columns


@pytest.mark.parametrize("cls", [ProteomicsAdapter, MetabolomicsAdapter])
def test_late_text_value_routes_column_to_metadata(cls, pandas_infer_string):
    """Position-independence (follow-on to the P0 fix). A real text token AFTER
    the first ten numeric-looking rows must still route the column to metadata,
    not intensity. The prior ``head(10)`` sampling classified it as intensity
    and the downstream coerce blanked the token to NaN."""
    values = [str(i) for i in range(10)] + ["control"]  # 10 numeric, then text
    df = pd.DataFrame(
        {
            "mostly_numeric": pd.Series(values),  # matches no name pattern
            "s1": [float(i) for i in range(11)],
        }
    )
    adapter = _adapter(cls)
    # The predicate itself must be position-independent.
    assert adapter._is_numeric_string_column(df["mostly_numeric"]) is False
    intensity_df, metadata_df = adapter._separate_intensity_metadata_columns(df)
    assert metadata_df is not None and "mostly_numeric" in metadata_df.columns
    assert "mostly_numeric" not in intensity_df.columns
    # The text token is preserved, not silently coerced to NaN.
    assert "control" in list(metadata_df["mostly_numeric"])


@pytest.mark.parametrize("cls", [ProteomicsAdapter, MetabolomicsAdapter])
def test_fully_numeric_string_column_beyond_ten_rows_is_intensity(
    cls, pandas_infer_string
):
    """The full scan must not create false metadata: a column of >10
    numeric-encoded strings stays intensity and coerces to real numbers."""
    values = [str(float(i)) for i in range(15)]
    df = pd.DataFrame({"abundance": pd.Series(values), "label": pd.Series(["x"] * 15)})
    adapter = _adapter(cls)
    assert adapter._is_numeric_string_column(df["abundance"]) is True
    intensity_df, _ = adapter._separate_intensity_metadata_columns(df)
    assert "abundance" in intensity_df.columns
    assert not intensity_df["abundance"].isna().all()


@pytest.mark.parametrize("cls", [ProteomicsAdapter, MetabolomicsAdapter])
def test_empty_string_token_routes_column_to_metadata(cls, pandas_infer_string):
    """'' coerces to NaN WITHOUT raising, so a column containing it is not fully
    numeric and must route to metadata (token preserved), not intensity where
    the empty string would be blanked. Codex follow-on finding."""
    df = pd.DataFrame({"maybe_num": pd.Series(["1", "2", ""]), "s1": [1.0, 2.0, 3.0]})
    adapter = _adapter(cls)
    assert adapter._is_numeric_string_column(df["maybe_num"]) is False
    intensity_df, metadata_df = adapter._separate_intensity_metadata_columns(df)
    assert metadata_df is not None and "maybe_num" in metadata_df.columns
    assert "maybe_num" not in intensity_df.columns
