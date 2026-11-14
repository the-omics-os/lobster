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
