"""Boolean columns must survive an H5AD round trip as booleans.

The writer and reader must preserve boolean dtypes so downstream masks remain valid.
Native boolean dtypes can be serialized directly; conversion should be limited to object
columns that need sanitization.
"""

import anndata
import numpy as np
import pandas as pd
import pytest

from lobster.core.backends.h5ad_backend import H5ADBackend


@pytest.fixture
def backend(tmp_path):
    return H5ADBackend(base_path=tmp_path)


def _adata(n_obs=6, n_vars=4):
    return anndata.AnnData(np.random.rand(n_obs, n_vars).astype("float32"))


class TestBoolRoundTrip:
    """bool in -> bool out, for .obs and .var, with and without None."""

    def test_var_bool_stays_bool(self, backend, tmp_path):
        adata = _adata()
        adata.var["highly_variable"] = pd.Series(
            [True, False, True, False], index=adata.var_names
        )

        backend.save(adata, tmp_path / "v.h5ad")
        loaded = backend.load(tmp_path / "v.h5ad")

        assert loaded.var["highly_variable"].dtype == bool

    def test_obs_bool_stays_bool(self, backend, tmp_path):
        adata = _adata()
        adata.obs["keep"] = pd.Series([True] * 3 + [False] * 3, index=adata.obs_names)

        backend.save(adata, tmp_path / "o.h5ad")
        loaded = backend.load(tmp_path / "o.h5ad")

        assert loaded.obs["keep"].dtype == bool

    def test_nullable_boolean_preserves_na(self, backend, tmp_path):
        """A nullable `boolean` column keeps pd.NA rather than gaining an 'NA' string.

        The old sanitizer mapped None to the literal 'NA', producing a THREE-valued
        string column -- so even a naive `== 'True'` comparison silently mapped NA to
        False.
        """
        adata = _adata()
        adata.var["hv"] = pd.array([True, None, False, True], dtype="boolean")

        backend.save(adata, tmp_path / "n.h5ad")
        loaded = backend.load(tmp_path / "n.h5ad")

        assert loaded.var["hv"].dtype == "boolean"
        assert bool(loaded.var["hv"].isna().iloc[1])
        assert loaded.var["hv"].isna().sum() == 1

    def test_mask_indexing_works_after_roundtrip(self, backend, tmp_path):
        """The regression as the caller experienced it: a mask, not a dtype check."""
        adata = _adata(n_vars=5)
        adata.var["highly_variable"] = pd.Series(
            [True, False, True, False, True], index=adata.var_names
        )
        expected = list(adata.var_names[adata.var["highly_variable"]])

        backend.save(adata, tmp_path / "m.h5ad")
        loaded = backend.load(tmp_path / "m.h5ad")

        # This raised IndexError before the fix.
        assert list(loaded.var_names[loaded.var["highly_variable"]]) == expected

    def test_values_survive_not_just_dtype(self, backend, tmp_path):
        """A column of the right dtype but wrong values would pass the checks above."""
        adata = _adata(n_vars=4)
        original = [True, False, False, True]
        adata.var["flag"] = pd.Series(original, index=adata.var_names)

        backend.save(adata, tmp_path / "val.h5ad")
        loaded = backend.load(tmp_path / "val.h5ad")

        assert list(loaded.var["flag"]) == original


class TestNonBoolColumnsUnaffected:
    """The fix narrows the sanitizer; it must not stop sanitizing what still needs it."""

    def test_object_dtype_python_bools_still_stringified(self, backend, tmp_path):
        """`object` dtype holding Python bools genuinely cannot be written by HDF5.

        Raw anndata raises "Can't implicitly convert non-string objects to strings", so
        this narrow case is what the original blanket guard was really for. Asserting it
        SAVES is the point -- the dtype it lands on is incidental.
        """
        adata = _adata(n_vars=3)
        adata.var["mixed"] = pd.Series(
            [True, False, None], index=adata.var_names, dtype=object
        )

        backend.save(adata, tmp_path / "obj.h5ad")
        loaded = backend.load(tmp_path / "obj.h5ad")

        assert loaded.var["mixed"] is not None

    def test_object_dtype_numpy_bools_still_stringified(self, backend, tmp_path):
        """`np.bool_` is NOT a subclass of `bool`.

        A `type(x) is bool` identity check silently skips numpy-bool object columns and
        turns a wrong-answer bug into a write crash, which is why the guard uses
        `isinstance(x, (bool, np.bool_))`.
        """
        adata = _adata(n_vars=3)
        adata.var["np_mixed"] = pd.Series(
            [np.bool_(True), np.bool_(False), None],
            index=adata.var_names,
            dtype=object,
        )

        backend.save(adata, tmp_path / "npobj.h5ad")
        loaded = backend.load(tmp_path / "npobj.h5ad")

        assert loaded.var["np_mixed"] is not None

    def test_genuine_string_column_containing_true_is_untouched(
        self, backend, tmp_path
    ):
        """Why the fix narrows save() instead of adding a heuristic inverse to load().

        Any load-side "was this really a bool?" test can misfire on a real string column
        whose values happen to include 'True'. Removing the asymmetry avoids needing the
        heuristic at all.
        """
        adata = _adata(n_vars=4)
        values = ["True", "False", "maybe", "True"]
        adata.var["verdict"] = pd.Series(values, index=adata.var_names)

        backend.save(adata, tmp_path / "s.h5ad")
        loaded = backend.load(tmp_path / "s.h5ad")

        assert list(loaded.var["verdict"].astype(str)) == values

    def test_numeric_and_categorical_columns_survive(self, backend, tmp_path):
        adata = _adata(n_vars=4)
        adata.var["score"] = pd.Series([1.5, 2.5, 3.5, 4.5], index=adata.var_names)
        adata.var["group"] = pd.Series(
            ["a", "b", "a", "b"], index=adata.var_names
        ).astype("category")

        backend.save(adata, tmp_path / "mix.h5ad")
        loaded = backend.load(tmp_path / "mix.h5ad")

        assert np.issubdtype(loaded.var["score"].dtype, np.number)
        assert list(loaded.var["score"]) == [1.5, 2.5, 3.5, 4.5]
        assert list(loaded.var["group"].astype(str)) == ["a", "b", "a", "b"]


class TestScanpyIdiom:
    """The end-to-end shape of the reported failure."""

    def test_hvg_style_subset_after_roundtrip(self, backend, tmp_path):
        """Subsetting on a boolean .var column is what the agent was trying to do."""
        adata = _adata(n_obs=10, n_vars=8)
        flags = [True, False, True, True, False, False, True, False]
        adata.var["highly_variable"] = pd.Series(flags, index=adata.var_names)

        backend.save(adata, tmp_path / "hvg.h5ad")
        loaded = backend.load(tmp_path / "hvg.h5ad")

        subset = loaded[:, loaded.var["highly_variable"]]
        assert subset.n_vars == sum(flags)
