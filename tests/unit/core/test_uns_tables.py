"""A table in ``uns`` is kept as a table when anndata can store it, else stored as text.

A save never fails where it succeeded before, and the text is exactly the form
the session writer produced before tables were kept (``str(frame)``).
"""

import logging

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.core.backends.h5ad_backend import H5ADBackend

TABLES = {
    "plain": pd.DataFrame({"g": ["a", "b"], "x": [1.5, np.nan]}),
    "bracket_col": pd.DataFrame({"condition[T.treated]": [1, 0]}),
    "multiindex_cols": pd.DataFrame(
        [[1, 2]], columns=pd.MultiIndex.from_tuples([("a", "x"), ("a", "y")])
    ),
    "multiindex_rows": pd.DataFrame(
        {"v": [1, 2]}, index=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)])
    ),
    "dup_index": pd.DataFrame({"v": [1, 2]}, index=["a", "a"]),
    "empty": pd.DataFrame(),
    "empty_cols": pd.DataFrame(index=["a", "b"]),
    "dict_cells": pd.DataFrame({"d": [{"a": 1}, None]}),
    "nullable_int": pd.DataFrame({"i": pd.array([1, None], dtype="Int64")}),
    "nullable_bool": pd.DataFrame({"b": pd.array([True, None], dtype="boolean")}),
    "datetime": pd.DataFrame({"t": pd.to_datetime(["2026-01-01", None])}),
    "cat_int": pd.DataFrame({"c": pd.Categorical([1, 2, 1])}),
    "complex": pd.DataFrame({"z": [1 + 2j, 3j]}),
    "none_col": pd.DataFrame({None: [1, 2]}),
    "str_dtype": pd.DataFrame({"s": pd.array(["a", None], dtype="string")}),
    "big_int": pd.DataFrame({"o": [2**70, 1]}),
    "tz_aware": pd.DataFrame({"t": pd.to_datetime(["2026-01-01"]).tz_localize("UTC")}),
    "interval": pd.DataFrame({"i": pd.interval_range(0, 2)}),
    "period": pd.DataFrame({"p": pd.period_range("2026-01", periods=2, freq="M")}),
    "wide": pd.DataFrame(np.ones((3, 3000))),
    "long": pd.DataFrame({"x": np.arange(50000.0)}),
    "non_ascii": pd.DataFrame({"é基因": [1.0]}, index=["ß"]),
    "zero_rows": pd.DataFrame(
        {"p": pd.Series([], dtype=float), "g": pd.Series([], dtype=object)}
    ),
}
TEXT = {
    "slash_col": pd.DataFrame({"a/b": [1, 2], "c": [3, 4]}),
    "dup_cols": pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"]),
    "dup_after_str": pd.DataFrame([[1, 2], [3, 4]], columns=[1, "1"]),
    "list_cells": pd.DataFrame({"l": [[1, 2], [3]]}),
    "array_cells": pd.DataFrame({"l": [np.array([1, 2]), np.array([3])]}),
    "cat_mixed": pd.DataFrame({"c": pd.Categorical(["a", 1, "a"])}),
    "index_col_clash": pd.DataFrame({"index": [1, 2], "_index": [3, 4]}),
    "named_index_clash": pd.DataFrame(
        {"gene": [1, 2]}, index=pd.Index(["a", "b"], name="gene")
    ),
    "empty_str_col": pd.DataFrame({"": [1, 2]}),
    "cell_holds_table": pd.DataFrame({"c": [pd.DataFrame({"a": [1]})]}),
    "sparse": pd.DataFrame({"s": pd.arrays.SparseArray([0, 0, 1])}),
}


def _save_and_read(tmp_path, uns):
    adata = ad.AnnData(np.ones((2, 2), dtype="float32"))
    adata.uns.update(uns)
    path = tmp_path / "t.h5ad"
    H5ADBackend(base_path=tmp_path).save(adata, path)
    return ad.read_h5ad(path)


@pytest.mark.parametrize("name", sorted(TABLES))
def test_a_table_that_can_be_stored_is_read_back_as_a_table(tmp_path, name):
    frame = TABLES[name]
    before = frame.copy(deep=True)
    back = _save_and_read(tmp_path, {"k": {"t": frame}, "top": frame})
    for table in (back.uns["k"]["t"], back.uns["top"]):
        assert isinstance(table, pd.DataFrame), type(table)
        assert list(table.columns) == [str(c) for c in frame.columns]
        assert list(table.index) == [str(i) for i in frame.index]
        assert table.shape == frame.shape
        for position, column in enumerate(frame.columns):
            original = frame.iloc[:, position]
            if original.dtype.kind in "fi":
                np.testing.assert_array_equal(
                    table.iloc[:, position].to_numpy(dtype=float),
                    original.to_numpy(dtype=float),
                )
    pd.testing.assert_frame_equal(frame, before)  # the caller's table is untouched


@pytest.mark.parametrize("name", sorted(TEXT))
def test_a_table_that_cannot_be_stored_is_saved_as_the_text_it_always_was(
    tmp_path, name, caplog
):
    frame = TEXT[name]
    with caplog.at_level(logging.WARNING, logger="lobster.core.utils.h5ad_utils"):
        back = _save_and_read(tmp_path, {"k": {"t": frame}, "top": frame})
    assert back.uns["k"]["t"] == str(frame)
    assert back.uns["top"] == str(frame)
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "uns[k/t]" in messages and "uns[top]" in messages
    assert "stored as text" in messages


def test_the_warning_gives_the_reason(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="lobster.core.utils.h5ad_utils"):
        _save_and_read(tmp_path, {"k": TEXT["dup_cols"]})
    assert any("repeated column names" in r.getMessage() for r in caplog.records)


def test_a_table_in_a_list_or_tuple_is_the_text_it_always_was(tmp_path):
    frame = pd.DataFrame(
        {"s": ["a", None], "x": [1.5, np.nan]}
    )  # cleaning would show ''
    assert "None" in str(frame)
    back = _save_and_read(
        tmp_path, {"a": {"l": [frame]}, "b": [frame, 1], "c": (frame,)}
    )
    assert list(back.uns["a"]["l"]) == [str(frame)]
    assert list(back.uns["b"]) == [str(frame), "1"]
    assert list(back.uns["c"]) == [str(frame)]


def test_a_series_in_uns_is_text_as_before(tmp_path):
    series = pd.Series([1, 2], name="x")
    assert _save_and_read(tmp_path, {"k": {"s": series}}).uns["k"]["s"] == str(series)


def test_a_table_the_probe_refuses_is_text_not_an_error(tmp_path, monkeypatch):
    from lobster.core.utils import h5ad_utils

    monkeypatch.setattr(h5ad_utils, "_table_round_trip_problem", lambda c: "refused")
    frame = TABLES["plain"]
    assert _save_and_read(tmp_path, {"k": frame}).uns["k"] == str(frame)


def test_a_table_that_does_not_read_back_equal_is_text(tmp_path, monkeypatch, caplog):
    import anndata.io

    real = anndata.io.read_elem

    def altered(group):
        back = real(group)
        back.iloc[0, 0] = "changed"
        return back

    monkeypatch.setattr(anndata.io, "read_elem", altered)
    frame = TABLES["plain"]
    with caplog.at_level(logging.WARNING, logger="lobster.core.utils.h5ad_utils"):
        back = _save_and_read(tmp_path, {"k": frame})
    assert back.uns["k"] == str(frame)
    assert any("uns[k]" in r.getMessage() for r in caplog.records)


def test_the_callers_table_is_never_changed_by_the_cleaning():
    from lobster.core.utils.h5ad_utils import sanitize_value

    frame = pd.DataFrame(
        {"s": ["a", None], "n": [1, 2], "c": pd.Categorical(["x", "y"])},
        index=pd.Index([10, 20]),
    )
    columns, dtypes, index = list(frame.columns), frame.dtypes.to_dict(), frame.index
    cleaned = sanitize_value({"k": frame})["k"]
    assert cleaned is not frame
    assert list(cleaned.index) == ["10", "20"]  # the copy was cleaned ...
    assert frame.index is index and frame.index.name is None  # ... this one was not
    assert list(frame.columns) == columns and frame.dtypes.to_dict() == dtypes
    assert frame["s"].tolist() == ["a", None]


def test_an_arrow_backed_numeric_column_comes_back_numeric(tmp_path):
    pa = pytest.importorskip("pyarrow")
    frame = pd.DataFrame(
        {
            "f": pd.array([1.5, None, 3.0], dtype=pd.ArrowDtype(pa.float64())),
            "i": pd.array([1, 2, 3], dtype=pd.ArrowDtype(pa.int64())),
        }
    )
    back = _save_and_read(tmp_path, {"k": frame}).uns["k"]
    assert isinstance(back, pd.DataFrame)
    assert back["i"].dtype.kind == "i" and list(back["i"]) == [1, 2, 3]
    assert back["f"].dtype.kind == "f" and back["f"].isna().tolist() == [
        False,
        True,
        False,
    ]
