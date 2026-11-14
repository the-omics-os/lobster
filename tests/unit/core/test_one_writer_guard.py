"""A session and an exported notebook save through one function, ``write_session_h5ad``."""

import ast
import inspect
import sys
import textwrap

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.core.backends import h5ad_backend
from lobster.core.backends.h5ad_backend import H5ADBackend
from lobster.core.notebooks.exporter import _guarded_save_code

FALLBACK = (
    "Written without the session writer (lobster not importable); "
    "the saved file may differ from the session's."
)


def _calls(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            names.add(
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", "")
            )
    return names


def test_session_and_notebook_share_one_writer():
    save = ast.parse(textwrap.dedent(inspect.getsource(H5ADBackend.save)))
    calls = _calls(save)
    assert "write_session_h5ad" in calls
    # no second preparation or write of its own
    assert not calls & {"sanitize_anndata", "_repair_log1p_base", "write_h5ad"}
    assert not calls & {"_convert_arrow_to_standard", "var_names_make_unique"}
    cell = _guarded_save_code('output_path = "x.h5ad"')
    assert "from lobster.core.backends.h5ad_backend import write_session_h5ad" in cell
    assert "write_session_h5ad(adata, output_path" in cell
    for inlined in ("sanitize", "repair_log1p", "_convert_arrow"):
        assert inlined not in cell
    assert callable(h5ad_backend.write_session_h5ad)


def _run_cell(adata, path, compression=None):
    code = _guarded_save_code(f"output_path = {str(path)!r}", compression)
    namespace = {"adata": adata}
    exec(compile(code, "<save cell>", "exec"), namespace)  # nosec B102 - our own cell
    return namespace


def _adata():
    adata = ad.AnnData(np.arange(12, dtype="float32").reshape(4, 3))
    adata.uns["log1p"] = {"base": None}
    return adata


def test_a_float_key_in_uns_is_written_by_the_session_writer(tmp_path):
    adata = _adata()
    adata.uns["clustering_results"] = {0.5: {"n": 3}, 1.0: {"n": 5}}
    out = tmp_path / "o.h5ad"
    _run_cell(adata, out)
    back = ad.read_h5ad(out)
    assert "lobster_unsaved_uns_keys" not in back.uns
    assert "clustering_results" in back.uns


def test_the_cell_falls_back_with_one_line_when_lobster_is_not_importable(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setitem(sys.modules, "lobster.core.backends.h5ad_backend", None)
    out = tmp_path / "o.h5ad"
    _run_cell(_adata(), out)
    printed = capsys.readouterr().out
    assert printed.count(FALLBACK) == 1
    assert out.exists()


def test_the_cell_does_not_print_the_fallback_line_when_the_writer_is_there(
    tmp_path, capsys
):
    _run_cell(_adata(), tmp_path / "o.h5ad")
    assert FALLBACK not in capsys.readouterr().out


# --- a DataFrame in uns is saved as a table, by the session and by the notebook cell ----------


def _de_frame():
    import pandas as pd

    return pd.DataFrame(
        {
            "log2FoldChange": [1.5, -0.5, np.nan],
            "padj": [0.01, 0.2, 0.9],
            "contrast": ["a_vs_b", "a_vs_b", None],
            "flag": [True, False, True],
            "group": pd.Categorical(["x", "y", "x"]),
        },
        index=["g1", "g2", "g3"],
    )


def _with_uns_frame():
    adata = _adata()
    adata.uns["de_iteration_results_1"] = {"results_df": _de_frame(), "n": 3}
    return adata


def _check_frame(back):
    import pandas as pd

    frame = back.uns["de_iteration_results_1"]["results_df"]
    assert isinstance(frame, pd.DataFrame), type(frame)
    assert list(frame.index) == ["g1", "g2", "g3"]
    assert list(frame["padj"]) == [0.01, 0.2, 0.9]
    assert list(frame["log2FoldChange"][:2]) == [1.5, -0.5]
    assert list(frame["flag"]) == [True, False, True]
    assert list(frame["contrast"][:2]) == ["a_vs_b", "a_vs_b"]


def test_a_dataframe_in_uns_survives_the_session_writer(tmp_path):
    from lobster.core.backends.h5ad_backend import write_session_h5ad

    adata = _with_uns_frame()
    write_session_h5ad(adata, tmp_path / "u.h5ad")
    _check_frame(ad.read_h5ad(tmp_path / "u.h5ad"))
    # the live object is untouched
    live = adata.uns["de_iteration_results_1"]["results_df"]
    assert list(live["contrast"])[2] is None and list(live.index) == ["g1", "g2", "g3"]


def test_a_dataframe_in_uns_survives_the_backend_save(tmp_path):
    H5ADBackend(base_path=tmp_path).save(_with_uns_frame(), tmp_path / "b.h5ad")
    _check_frame(ad.read_h5ad(tmp_path / "b.h5ad"))


def test_a_dataframe_in_uns_survives_the_notebook_save_cell(tmp_path):
    namespace = _run_cell(_with_uns_frame(), tmp_path / "c.h5ad")
    assert "lobster_unsaved_uns_keys" not in namespace["adata"].uns
    _check_frame(ad.read_h5ad(tmp_path / "c.h5ad"))


def _awkward_frame():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "name": pd.array(["a", None, "c"], dtype="string"),
            "cat": pd.Categorical(pd.array(["u", "v", "u"], dtype="string")),
            "n": pd.array([1, None, 3], dtype="Int64"),
            "when": pd.to_datetime(["2026-01-01", None, "2026-01-03"]),
            "mixed": [1, "x", None],
            7: [0.5, np.nan, 2.5],
        },
        index=pd.Index([10, 20, 30], name="rank"),
    )
    return frame


def test_a_table_in_uns_gets_the_same_cleaning_as_obs_and_var(tmp_path):
    import pandas as pd

    from lobster.core.backends.h5ad_backend import write_session_h5ad

    adata = _adata()
    adata.uns["top"] = _awkward_frame()
    adata.uns["outer"] = {"inner": {"tbl": _awkward_frame()}}
    before = adata.uns["top"].copy()
    write_session_h5ad(adata, tmp_path / "t.h5ad")
    back = ad.read_h5ad(tmp_path / "t.h5ad")
    for frame in (back.uns["top"], back.uns["outer"]["inner"]["tbl"]):
        assert isinstance(frame, pd.DataFrame)
        assert list(frame.index) == ["10", "20", "30"] and frame.index.name == "rank"
        assert list(frame.columns) == ["name", "cat", "n", "when", "mixed", "7"]
        assert list(frame["name"]) == ["a", "", "c"]  # missing text -> ""
        assert list(frame["cat"]) == ["u", "v", "u"]
        assert frame["7"].isna().tolist() == [False, True, False]  # numbers stay NaN
        assert list(frame["mixed"]) == ["1", "x", ""]
        assert list(frame["when"])[1] == ""
    pd.testing.assert_frame_equal(
        adata.uns["top"], before
    )  # the live frame is untouched


@pytest.mark.xfail(
    strict=True,
    reason="known limit: numbers nested in uns dicts are stored as text by the session writer",
)
def test_numbers_nested_in_uns_come_back_as_numbers(tmp_path):
    adata = _adata()
    adata.uns["iteration_comparisons"] = {"c": {"summary_stats": {"n_sig_1": 12}}}
    _run_cell(adata, tmp_path / "n.h5ad")
    back = ad.read_h5ad(tmp_path / "n.h5ad")
    assert back.uns["iteration_comparisons"]["c"]["summary_stats"]["n_sig_1"] == 12


# --- nullable integer columns (Int8..UInt64), with and without pd.NA ---------------------

NULLABLE_INTS = [
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
]


def _with_nullable_int(dtype, axis, missing):
    adata = _adata()
    table = getattr(adata, axis)
    values = [1, None, 3, 4] if missing else [1, 2, 3, 4]
    values = values[: len(table)]
    table["n"] = pd.array(values, dtype=dtype)
    return adata


def _write(way, adata, tmp_path):
    from lobster.core.backends.h5ad_backend import write_session_h5ad

    path = tmp_path / f"{way}.h5ad"
    if way == "writer":
        write_session_h5ad(adata, path)
    elif way == "save":
        H5ADBackend(base_path=tmp_path).save(adata, path)
    else:
        _run_cell(adata, path)
    return path


@pytest.mark.parametrize("way", ["writer", "save", "cell"])
@pytest.mark.parametrize("axis", ["obs", "var"])
@pytest.mark.parametrize("dtype", NULLABLE_INTS)
@pytest.mark.parametrize("missing", [True, False])
def test_a_nullable_integer_column_is_written_with_its_type_and_its_missing_values(
    tmp_path, dtype, axis, way, missing
):
    adata = _with_nullable_int(dtype, axis, missing)
    before = getattr(adata, axis)["n"].copy()
    back = getattr(ad.read_h5ad(_write(way, adata, tmp_path)), axis)["n"]
    assert str(back.dtype) == dtype
    assert back.isna().tolist() == before.isna().tolist()
    assert back.dropna().tolist() == before.dropna().tolist()
    pd.testing.assert_series_equal(
        getattr(adata, axis)["n"], before
    )  # caller untouched


# --- index names and None values: kept when anndata writes and reads them back unchanged --

WAYS = ["writer", "save", "cell"]


def _unnamed():
    adata = _adata()
    assert adata.obs.index.name is None and adata.var.index.name is None
    return adata


@pytest.mark.parametrize("way", WAYS)
def test_an_unnamed_index_is_written_and_read_back_unnamed(tmp_path, way):
    adata = _unnamed()
    back = ad.read_h5ad(_write(way, adata, tmp_path))
    assert back.obs.index.name is None and back.var.index.name is None
    assert (
        adata.obs.index.name is None and adata.var.index.name is None
    )  # caller's object


@pytest.mark.parametrize("way", WAYS)
@pytest.mark.parametrize("axis, column", [("obs", "index"), ("var", "gene_id")])
def test_an_unnamed_index_and_a_column_named_like_its_default_name_is_saved(
    tmp_path, way, axis, column
):
    adata = _unnamed()
    table = getattr(adata, axis)
    table[column] = np.arange(len(table))
    back = getattr(ad.read_h5ad(_write(way, adata, tmp_path)), axis)
    assert list(back.columns) == [column]
    assert back[column].tolist() == list(range(len(table)))
    assert back.index.name is None


@pytest.mark.parametrize("way", WAYS)
def test_an_empty_index_name_is_saved_under_the_default_names(tmp_path, way):
    adata = _unnamed()
    adata.obs.index.name = ""
    adata.var.index.name = ""
    back = ad.read_h5ad(_write(way, adata, tmp_path))
    assert back.obs.index.name == "index" and back.var.index.name == "gene_id"


@pytest.mark.parametrize("way", WAYS)
def test_a_named_index_keeps_its_name(tmp_path, way):
    adata = _unnamed()
    adata.obs.index.name, adata.var.index.name = "cells", "genes"
    back = ad.read_h5ad(_write(way, adata, tmp_path))
    assert back.obs.index.name == "cells" and back.var.index.name == "genes"


@pytest.mark.parametrize("way", WAYS)
def test_a_none_value_in_a_uns_dict_stays_none(tmp_path, way):
    adata = _unnamed()
    adata.uns["other"] = {"x": None, "deeper": {"y": None}}
    adata.uns["top"] = None
    back = ad.read_h5ad(_write(way, adata, tmp_path)).uns
    assert back["log1p"] == {"base": None}
    assert back["other"]["x"] is None and back["other"]["deeper"]["y"] is None
    assert back["top"] is None
    assert adata.uns["log1p"] == {"base": None}  # caller's object


def test_a_none_log1p_base_survives_the_load_repair(tmp_path):
    path = _write("save", _unnamed(), tmp_path)
    loaded = H5ADBackend(base_path=tmp_path).load(path)
    assert loaded.uns["log1p"] == {"base": None}


@pytest.mark.parametrize("way", WAYS)
def test_a_none_inside_a_list_or_tuple_stays_text(tmp_path, way):
    adata = _unnamed()
    adata.uns["lists"] = {"l": [1, None, "x"], "t": (1, None)}
    back = ad.read_h5ad(_write(way, adata, tmp_path)).uns["lists"]
    assert back["l"].tolist() == ["1", "", "x"]
    assert back["t"].tolist() == ["1", ""]


# --- when anndata cannot keep the native form, the earlier behaviour applies ---


@pytest.fixture
def no_native(monkeypatch):
    """Make the writer believe this anndata cannot round-trip an unnamed index or None."""
    from lobster.core.utils import h5ad_utils

    # bound names only: monkeypatch restores them, the probe's cache is not touched
    monkeypatch.setattr(h5ad_utils, "anndata_keeps_native", lambda: False)
    monkeypatch.setattr(h5ad_backend, "anndata_keeps_native", lambda: False)


def test_the_probe_finds_the_pinned_anndata_native():
    from lobster.core.utils.h5ad_utils import anndata_keeps_native

    assert anndata_keeps_native() is True


@pytest.mark.parametrize("way", WAYS)
def test_without_native_support_an_unnamed_index_is_named(tmp_path, no_native, way):
    back = ad.read_h5ad(_write(way, _unnamed(), tmp_path))
    assert back.obs.index.name == "index" and back.var.index.name == "gene_id"


@pytest.mark.parametrize("way", WAYS)
def test_without_native_support_a_none_in_uns_is_empty_text(tmp_path, no_native, way):
    adata = _unnamed()
    adata.uns["other"] = {"x": None, "deeper": {"y": None}}
    adata.uns["top"] = None
    back = ad.read_h5ad(_write(way, adata, tmp_path)).uns
    assert back["other"]["x"] == "" and back["other"]["deeper"]["y"] == ""
    assert back["top"] == ""
    assert "base" not in back["log1p"]  # the load repair drops the empty base


@pytest.mark.parametrize("way", WAYS)
@pytest.mark.parametrize("axis, column", [("obs", "index"), ("var", "gene_id")])
def test_without_native_support_an_index_next_to_a_column_of_its_name_fails(
    tmp_path, no_native, way, axis, column
):
    adata = _unnamed()
    table = getattr(adata, axis)
    table[column] = np.arange(len(table))
    with pytest.raises(Exception):
        _write(way, adata, tmp_path)


@pytest.mark.parametrize("way", WAYS)
def test_the_save_is_native_again_after_the_forced_fallback(tmp_path, way):
    adata = _unnamed()
    adata.uns["other"] = {"x": None}
    back = ad.read_h5ad(_write(way, adata, tmp_path))
    assert back.obs.index.name is None and back.var.index.name is None
    assert back.uns["other"]["x"] is None


# --- categorical columns with pandas-3 string categories keep what the session writer keeps ---


def _string_categorical(values, categories, ordered=False):
    """A categorical whose categories are pandas' default string dtype (the notebook kernel's)."""
    from lobster.core.utils.h5ad_utils import is_arrow_dtype

    with pd.option_context("future.infer_string", True):
        column = pd.Series(
            pd.Categorical(values, categories=categories, ordered=ordered)
        )
    assert is_arrow_dtype(column.cat.categories)
    return column


def _with_categorical(axis, column):
    adata = _adata()
    table = getattr(adata, axis)
    table["group"] = column.set_axis(table.index).to_numpy()
    if not isinstance(table["group"].dtype, pd.CategoricalDtype):
        table["group"] = pd.Categorical(
            column.to_numpy(),
            categories=column.cat.categories,
            ordered=column.cat.ordered,
        )
    assert isinstance(table["group"].cat.categories.dtype, pd.StringDtype)
    return adata


NUMBER_ORDER = ["0", "1", "2", "10", "11", "12"]


@pytest.mark.parametrize("way", WAYS)
@pytest.mark.parametrize("axis", ["obs", "var"])
def test_ten_or_more_categories_keep_their_number_order(tmp_path, way, axis):
    n = 4 if axis == "obs" else 3
    column = _string_categorical(NUMBER_ORDER[:n], NUMBER_ORDER)
    back = getattr(
        ad.read_h5ad(_write(way, _with_categorical(axis, column), tmp_path)), axis
    )
    assert list(back["group"].cat.categories) == NUMBER_ORDER


@pytest.mark.parametrize("way", WAYS)
@pytest.mark.parametrize("axis", ["obs", "var"])
def test_unused_categories_are_kept(tmp_path, way, axis):
    n = 4 if axis == "obs" else 3
    column = _string_categorical(["a", "b", "a", "b"][:n], ["b", "a", "unused"])
    back = getattr(
        ad.read_h5ad(_write(way, _with_categorical(axis, column), tmp_path)), axis
    )
    assert list(back["group"].cat.categories) == ["b", "a", "unused"]


@pytest.mark.parametrize("way", WAYS)
@pytest.mark.parametrize("axis", ["obs", "var"])
def test_an_ordered_categorical_stays_ordered(tmp_path, way, axis):
    n = 4 if axis == "obs" else 3
    column = _string_categorical(
        ["lo", "hi", "lo", "hi"][:n], ["lo", "hi"], ordered=True
    )
    back = getattr(
        ad.read_h5ad(_write(way, _with_categorical(axis, column), tmp_path)), axis
    )
    assert back["group"].cat.ordered and list(back["group"].cat.categories) == [
        "lo",
        "hi",
    ]


@pytest.mark.parametrize("way", WAYS)
@pytest.mark.parametrize("axis", ["obs", "var"])
def test_a_missing_value_stays_missing_not_an_empty_category(tmp_path, way, axis):
    n = 4 if axis == "obs" else 3
    column = _string_categorical(["a", None, "b", "a"][:n], ["a", "b"])
    back = getattr(
        ad.read_h5ad(_write(way, _with_categorical(axis, column), tmp_path)), axis
    )
    assert back["group"].isna().tolist() == [False, True, False, False][:n]
    assert list(back["group"].cat.categories) == ["a", "b"]  # no '' category


def test_a_missing_value_in_a_table_in_uns_stays_missing(tmp_path):
    table = pd.DataFrame(
        {"group": _string_categorical(["a", None, "10", "2"], ["a", "2", "10"])}
    )
    adata = _adata()
    adata.uns["tbl"] = table
    back = ad.read_h5ad(_write("writer", adata, tmp_path)).uns["tbl"]
    assert isinstance(back, pd.DataFrame)
    assert back["group"].isna().tolist() == [False, True, False, False]
    assert list(back["group"].cat.categories) == ["a", "2", "10"]


def test_a_table_in_uns_keeps_category_order_and_unused_categories(tmp_path):
    table = pd.DataFrame(
        {"group": _string_categorical(["10", "2"], ["2", "10", "unused"], ordered=True)}
    )
    adata = _adata()
    adata.uns["tbl"] = table
    back = ad.read_h5ad(_write("writer", adata, tmp_path)).uns["tbl"]["group"]
    assert list(back.cat.categories) == ["2", "10", "unused"] and back.cat.ordered


@pytest.mark.parametrize("axis", ["obs", "var"])
@pytest.mark.parametrize(
    "values, categories, ordered",
    [
        ([3, None, 2, 3], [3, 1, 2, 7], False),  # int, not sorted, 1 and 7 unused
        ([2.5, 0.5, None, 2.5], [2.5, 0.5, 9.5], True),  # float, ordered
        ([True, None, False, True], [True, False], False),  # bool
    ],
    ids=["int", "float", "bool"],
)
def test_categories_that_are_not_pandas_strings_are_left_alone(
    axis, values, categories, ordered
):
    from lobster.core.utils.h5ad_utils import convert_arrow_to_standard

    adata = _adata()
    table = getattr(adata, axis)
    # string inference off, as the lobster command line sets it
    with pd.option_context("future.infer_string", False):
        table["group"] = pd.Categorical(
            values[: len(table)], categories=categories, ordered=ordered
        )
        returned = getattr(convert_arrow_to_standard(adata), axis)["group"]
    pd.testing.assert_series_equal(returned, table["group"])


@pytest.mark.parametrize("way", WAYS)
def test_object_categories_are_written_as_before(tmp_path, way):
    adata = _adata()
    adata.obs["group"] = pd.Categorical(
        ["10", None, "2", "10"], categories=["2", "10", "unused"], ordered=True
    )
    assert adata.obs["group"].cat.categories.dtype == object
    back = ad.read_h5ad(_write(way, adata, tmp_path)).obs["group"]
    assert list(back.cat.categories) == ["2", "10", "unused"] and back.cat.ordered
    assert back.isna().tolist() == [False, True, False, False]


def test_categories_that_repeat_as_text_fall_back_to_the_rebuild(caplog):
    import logging
    from types import SimpleNamespace

    from lobster.core.utils.h5ad_utils import plain_categorical

    column = SimpleNamespace(
        cat=SimpleNamespace(
            categories=[1, "1"], codes=np.array([0, 1, -1]), ordered=False
        ),
        to_numpy=lambda dtype=None, na_value=None: np.array(["1", "1", ""]),
    )
    with caplog.at_level(logging.DEBUG, logger="lobster.core.utils.h5ad_utils"):
        result = plain_categorical(column)
    assert list(result.categories) == ["", "1"] and list(result) == ["1", "1", ""]
    assert any("repeat" in record.message for record in caplog.records)


def test_the_inspect_tool_counts_the_same_categories_for_both_write_paths(tmp_path):
    from lobster.cli_internal.mcp_inspect import inspect

    values, categories = ["a", None, "b", "a"], ["a", "b"]
    session = _adata()
    session.obs["group"] = pd.Categorical(values, categories=categories)
    assert session.obs["group"].cat.categories.dtype == object
    notebook = _with_categorical("obs", _string_categorical(values, categories))
    counts = [
        inspect(_write(way, adata, tmp_path), "obs:group", 10)["n_categories"]
        for way, adata in (("writer", session), ("cell", notebook))
    ]
    assert counts == [2, 2]
