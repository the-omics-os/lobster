"""Compare a replay's outputs with the reference, under a ComparisonPolicy.

The comparator walks two trees of values (the reference captured from the live
session, and what the replayed notebook produced) and returns one
:class:`~lobster.testing.replay_contract.SlotResult` per required path.

Supported values: ``AnnData`` (slots ``X``, ``layers``, ``obs``, ``var``,
``obsm``, ``obsp``, ``varm``, ``varp``, ``uns``, ``raw`` with ``X``/``var``/
``varm``); pandas ``DataFrame``, ``Series`` and ``Index`` with NumPy, categorical,
nullable (masked) or string dtypes; plain NumPy arrays and scalars; SciPy
sparse matrices; mappings, lists and tuples; Python scalars; and output files
(:func:`output_files`). Anything else, including NumPy array subclasses such
as masked arrays, pandas ``MultiIndex`` and other extension dtypes, is
reported as NOT_COMPARED and never converted to text to make it comparable.

Rules that keep a wrong result from passing:

* inventories are exact: keys, columns and files present on one side only are
  a mismatch, and axis order matters (nothing is sorted before comparing);
* types and dtypes compare exactly, scalars included (``np.int32`` is not
  ``np.int64``, ``np.float64`` is not Python ``float``, datetime units
  differ), as do integers, labels, categories (inventory, order and the
  ordered flag), index and column names, and missing-value positions;
* floats use the policy's predeclared tolerance (exact by default) at native
  precision; NaN must sit at the same positions and infinities must be
  identical; complex values compare their real and imaginary parts separately;
* axis labels, index and column names and category labels are identities and
  always compare exactly, whatever tolerance the values have;
* keys and column labels must be addressable by a policy path: a label
  containing ``/``, ``[`` or ``]``, or two labels with the same text form, is
  reported as not compared rather than risk a rule applying to the wrong one;
* a required path missing from the reference is NOT_COMPARED, and missing from
  the output is a MISMATCH;
* agreement that no value exists (``None``, an absent ``raw``, an empty slot)
  is structural, not evidence: a comparison made only of that is never a
  MATCH;
* output files: CSV/TSV are parsed strictly (a broken quote is an error) and
  compared cell by cell as text (no type inference, so ``001`` is not ``1``
  and ``NA`` is not ``NULL``; quoting and line endings do not matter); JSON
  must be strict
  (duplicate keys and NaN/Infinity tokens are refused; numbers decode to IEEE
  doubles); H5AD is loaded with anndata; anything else is compared byte for
  byte. :func:`output_files` refuses a path that reaches the output root
  through a symbolic link below the caller's trusted anchor, and never
  follows a link inside the root.

What this module cannot do:

* It compares the values it is given. If the reference comes from memory and
  the output from disk, both must go through a codec that is lossless on its
  own; a codec that loses the same information on both sides produces a false
  MATCH that no comparator can see.
* It cannot show that the reference and the output were produced
  independently (for example that the notebook did not read the reference).
  The harness that stages and runs the notebook must ensure that.
* ``SlotResult.compared`` counts comparison operations. The same array stored
  under two names is counted twice; it is not a count of independent evidence.

Scientific packages are imported lazily, so importing this module is cheap.
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from lobster.testing.replay_contract import (
    ComparisonPolicy,
    ContractError,
    SlotResult,
    SlotStatus,
    SparsePolicy,
    split_path,
    strict_json_loads,
)

MAX_FINDINGS = 20

_MISSING = object()


@dataclass(frozen=True)
class OutputFile:
    """A file a replay wrote (or the reference copy of it)."""

    path: Path


def output_files(directory: Path, *, anchor: Path) -> Dict[str, Any]:
    """Nested mapping of every file below ``directory`` to an OutputFile.

    Directories become mappings, so ``files/split/train.csv`` addresses
    ``<directory>/split/train.csv``.

    ``anchor`` is a directory the caller trusts (for example the run's
    staging root). It may itself sit behind system links such as macOS
    ``/tmp``. ``directory`` must lie inside it, and no path component from the
    anchor down to ``directory`` may be a symbolic link. Below the root, a
    link is kept as an entry (and later reported as not compared) rather
    than followed. Compare OutputFile objects only from this function.
    """
    directory = Path(directory)
    _check_no_links_below(Path(anchor), directory)
    return _walk_files(directory)


def _check_no_links_below(anchor: Path, directory: Path) -> None:
    anchor = anchor.absolute()
    directory = directory.absolute()
    if not anchor.is_dir():
        raise ValueError(f"anchor is not a directory: {anchor}")
    try:
        parts = directory.relative_to(anchor).parts
    except ValueError:
        raise ValueError(f"{directory} is not inside the anchor {anchor}") from None
    if any(part in ("..", ".") for part in parts):
        raise ValueError(f"output path is not normalized: {directory}")
    current = anchor
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"output path goes through a symbolic link: {current}")
    if not directory.is_dir():
        raise ValueError(f"output root is not a directory: {directory}")


def _walk_files(directory: Path) -> Dict[str, Any]:
    tree: Dict[str, Any] = {}
    for entry in sorted(directory.iterdir()):
        if entry.is_dir() and not entry.is_symlink():
            tree[entry.name] = _walk_files(entry)
        else:
            tree[entry.name] = OutputFile(entry)
    return tree


def compare(
    expected: Any, actual: Any, policy: ComparisonPolicy
) -> Tuple[SlotResult, ...]:
    """Compare ``actual`` with ``expected`` for every path the policy requires."""
    if not isinstance(policy, ComparisonPolicy):
        raise TypeError("policy must be a ComparisonPolicy")
    results = []
    for path in policy.required:
        rtol, atol = policy.tolerance_for(path)
        reference = _resolve(expected, path)
        output = _resolve(actual, path)
        if reference is _MISSING:
            results.append(
                SlotResult(
                    path,
                    SlotStatus.NOT_COMPARED,
                    rtol=rtol,
                    atol=atol,
                    not_compared=(f"{path}: the reference has no value here",),
                )
            )
            continue
        if output is _MISSING:
            results.append(
                SlotResult(
                    path,
                    SlotStatus.MISMATCH,
                    rtol=rtol,
                    atol=atol,
                    diffs=(f"{path}: missing from the output",),
                )
            )
            continue
        walk = _Walk(policy)
        walk.value(reference, output, path)
        results.append(walk.result(path, rtol, atol))
    return tuple(results)


# --------------------------------------------------------------------------
# Tree navigation
# --------------------------------------------------------------------------


def _module_type(module: str, name: str):
    loaded = sys.modules.get(module)
    return getattr(loaded, name, None) if loaded is not None else None


def _is(value: Any, module: str, name: str) -> bool:
    cls = _module_type(module, name)
    return cls is not None and isinstance(value, cls)


def _issparse(value: Any) -> bool:
    sparse = sys.modules.get("scipy.sparse")
    return sparse is not None and sparse.issparse(value)


def _anndata_tree(adata: Any) -> Dict[str, Any]:
    raw = adata.raw
    return {
        "X": adata.X,
        # Recent anndata lists a ``None`` layer that aliases X; X is compared
        # once, under "X", and an alias is not separate evidence.
        "layers": {k: adata.layers[k] for k in adata.layers.keys() if k is not None},
        "obs": adata.obs,
        "var": adata.var,
        "obsm": dict(adata.obsm),
        "obsp": dict(adata.obsp),
        "varm": dict(adata.varm),
        "varp": dict(adata.varp),
        "uns": dict(adata.uns),
        "raw": (
            None
            if raw is None
            else {"X": raw.X, "var": raw.var, "varm": dict(raw.varm)}
        ),
    }


def _child(node: Any, key: str) -> Any:
    if _is(node, "anndata", "AnnData"):
        return _anndata_tree(node).get(key, _MISSING)
    if isinstance(node, Mapping):
        return node.get(key, _MISSING) if key in node else _MISSING
    if _is(node, "pandas", "DataFrame"):
        columns = list(node.columns)
        if columns.count(key) == 1:
            return node[key]
    return _MISSING


def _resolve(tree: Any, path: str) -> Any:
    node = tree
    for part in split_path(path):
        node = _child(node, part)
        if node is _MISSING:
            return _MISSING
    return node


def _type_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


_PYTHON_SCALARS = {
    bool: "bool",
    int: "int",
    float: "float",
    complex: "complex",
    str: "str",
    bytes: "bytes",
}


def _kind(value: Any) -> str:
    np = sys.modules.get("numpy")
    if value is None:
        return "none"
    if isinstance(value, OutputFile):
        return "file"
    if _is(value, "anndata", "AnnData"):
        return "anndata"
    if _is(value, "pandas", "DataFrame"):
        return "dataframe"
    if _is(value, "pandas", "Series"):
        return "series"
    if _is(value, "pandas", "Index"):
        return "index"
    if _issparse(value):
        return "sparse"
    if np is not None:
        if type(value) is np.ndarray:
            return "array"
        if isinstance(value, np.ndarray):
            return "unknown"  # subclasses (masked arrays, matrix) change semantics
        if isinstance(value, np.generic):
            return "npscalar"
    if type(value) in _PYTHON_SCALARS:
        return _PYTHON_SCALARS[type(value)]
    if isinstance(value, Mapping):
        return "mapping"
    if type(value) in (list, tuple):
        return "sequence"
    pandas = sys.modules.get("pandas")
    if pandas is not None and (value is pandas.NA or value is pandas.NaT):
        return "missing"
    if isinstance(value, (_dt.datetime, _dt.date, _dt.timedelta)):
        return "datetime"
    return "unknown"


def _masked_dtype_class():
    dtypes = sys.modules.get("pandas.core.dtypes.dtypes")
    return getattr(dtypes, "BaseMaskedDtype", None) if dtypes else None


# --------------------------------------------------------------------------
# Comparison walk
# --------------------------------------------------------------------------


class _Walk:
    def __init__(self, policy: ComparisonPolicy) -> None:
        self.policy = policy
        self.compared = 0
        self.diffs: List[str] = []
        self.diff_count = 0
        self.not_compared: List[str] = []
        # Axis labels, names and category labels are identities: they compare
        # exactly whatever tolerance the surrounding values have.
        self.exact_depth = 0

    def tolerance(self, path: str) -> Tuple[float, float]:
        if self.exact_depth:
            return (0.0, 0.0)
        return self.policy.tolerance_for(path)

    # -- findings ---------------------------------------------------------

    def diff(self, path: str, message: str) -> None:
        self.diff_count += 1
        if len(self.diffs) < MAX_FINDINGS:
            self.diffs.append(f"{path}: {message}")

    def skip(self, path: str, message: str) -> None:
        if len(self.not_compared) < MAX_FINDINGS:
            self.not_compared.append(f"{path}: {message}")

    def result(self, path: str, rtol: float, atol: float) -> SlotResult:
        diffs = list(self.diffs)
        if self.diff_count > len(diffs):
            diffs.append(f"... and {self.diff_count - len(diffs)} more differences")
        if diffs:
            status = SlotStatus.MISMATCH
        elif self.not_compared:
            status = SlotStatus.NOT_COMPARED
        else:
            status = SlotStatus.MATCH
        return SlotResult(
            path,
            status,
            compared=self.compared,
            rtol=rtol,
            atol=atol,
            diffs=tuple(diffs),
            not_compared=tuple(self.not_compared),
        )

    def excluded(self, path: str) -> bool:
        return self.policy.exclusion_for(path) is not None

    # -- dispatch ---------------------------------------------------------

    def value(self, expected: Any, actual: Any, path: str) -> None:
        kind = _kind(expected)
        if kind == "unknown":
            self.skip(path, f"unsupported type {_type_name(expected)}")
            return
        other = _kind(actual)
        if other != kind:
            self.diff(
                path,
                f"type differs: expected {_type_name(expected)}, "
                f"got {_type_name(actual)}",
            )
            return
        getattr(self, f"_{kind}")(expected, actual, path)

    # -- scalars ----------------------------------------------------------

    def _none(self, expected, actual, path):
        """Both absent: structural agreement, not compared evidence."""

    def _missing(self, expected, actual, path):
        if expected is not actual:
            self.diff(path, f"expected {expected!r}, got {actual!r}")

    def _exact(self, expected, actual, path):
        self.compared += 1
        if expected != actual:
            self.diff(path, f"expected {expected!r}, got {actual!r}")

    _bool = _int = _str = _bytes = _exact

    def _datetime(self, expected, actual, path):
        self.compared += 1
        if type(expected) is not type(actual):
            self.diff(path, f"type differs: {_type_name(expected)}")
        elif expected != actual or repr(expected) != repr(actual):
            # repr also separates equal instants in different time zones.
            self.diff(path, f"expected {expected!r}, got {actual!r}")

    def _float(self, expected, actual, path):
        self.compared += 1
        if not self._close(expected, actual, path):
            rtol, atol = self.tolerance(path)
            self.diff(
                path,
                f"expected {expected!r}, got {actual!r} (rtol={rtol}, atol={atol})",
            )

    def _complex(self, expected, actual, path):
        self.compared += 1
        for part in ("real", "imag"):
            if not self._close(
                getattr(expected, part), getattr(actual, part), f"{path}[{part}]"
            ):
                self.diff(path, f"expected {expected!r}, got {actual!r}")
                return

    def _close(self, e: float, a: float, path: str) -> bool:
        """Python floats are IEEE doubles, so this loses no precision."""
        rtol, atol = self.tolerance(path)
        if math.isnan(e) or math.isnan(a):
            return math.isnan(e) and math.isnan(a)
        if math.isinf(e) or math.isinf(a):
            return e == a
        return abs(a - e) <= atol + rtol * abs(e)

    def _npscalar(self, expected, actual, path):
        if expected.dtype != actual.dtype:
            self.diff(path, f"dtype differs: {expected.dtype} vs {actual.dtype}")
            return
        np = sys.modules["numpy"]
        self._array_values(np.asarray(expected), np.asarray(actual), path)

    # -- containers -------------------------------------------------------

    def _mapping(self, expected, actual, path):
        keys = []
        for side in (expected, actual):
            bad = [k for k in side if not _addressable(k)]
            if bad:
                self.skip(
                    path,
                    f"this mapping was not compared: the keys {bad[:5]!r} cannot "
                    f"be named in a comparison rule (they contain '/', '[' or ']', "
                    f"or are empty, padded, '.', '..' or not text)",
                )
                return
            keys.append({k for k in side if not self.excluded(f"{path}/{k}")})
        want, got = keys
        for key in sorted(want - got):
            self.diff(f"{path}/{key}", "missing from the output")
        for key in sorted(got - want):
            self.diff(f"{path}/{key}", "not in the reference")
        for key in sorted(want & got):
            self.value(expected[key], actual[key], f"{path}/{key}")

    def _sequence(self, expected, actual, path):
        if type(expected) is not type(actual):
            self.diff(path, f"type differs: {_type_name(expected)}")
            return
        if len(expected) != len(actual):
            self.diff(path, f"length differs: {len(expected)} vs {len(actual)}")
            return
        for index, (e, a) in enumerate(zip(expected, actual)):
            self.value(e, a, f"{path}[{index}]")

    def _anndata(self, expected, actual, path):
        self._mapping(_anndata_tree(expected), _anndata_tree(actual), path)

    # -- arrays -----------------------------------------------------------

    def _array(self, expected, actual, path):
        if expected.shape != actual.shape:
            self.diff(path, f"shape differs: {expected.shape} vs {actual.shape}")
            return
        if expected.dtype != actual.dtype:
            self.diff(path, f"dtype differs: {expected.dtype} vs {actual.dtype}")
            return
        self._array_values(expected, actual, path)

    def _array_values(self, expected, actual, path):
        """Compare two plain arrays of equal shape and dtype."""
        np = sys.modules["numpy"]
        kind = expected.dtype.kind
        if kind in "biuUS":
            self.compared += expected.size
            unequal = np.asarray(expected != actual)
            if unequal.any():
                self.diff(
                    path,
                    f"{int(unequal.sum())} of {expected.size} values differ "
                    f"(first at index {_first(unequal)})",
                )
        elif kind == "f":
            self.compared += expected.size
            self._float_array(np, expected, actual, path)
        elif kind == "c":
            # Real and imaginary parts carry their own missing values and
            # tolerances; a NaN in one part must not hide the other part.
            self.compared += expected.size
            self._float_array(np, expected.real, actual.real, f"{path}[real]")
            self._float_array(np, expected.imag, actual.imag, f"{path}[imag]")
        elif kind in "mM":
            self.compared += expected.size
            nat_e, nat_a = np.isnat(expected), np.isnat(actual)
            if (nat_e != nat_a).any():
                self.diff(path, "missing (NaT) positions differ")
            elif (expected[~nat_e] != actual[~nat_a]).any():
                self.diff(path, "datetime values differ")
        elif kind == "O":
            flat_e, flat_a = expected.ravel(), actual.ravel()
            for index in range(flat_e.size):
                self.value(flat_e[index], flat_a[index], f"{path}[{index}]")
        elif kind == "V" and expected.dtype.names:
            # Structured array (rank_genes_groups): field by field. The dtypes are
            # equal, so the fields, their order and their types are the same.
            for field in expected.dtype.names:
                self._array_values(expected[field], actual[field], f"{path}[{field}]")
        else:
            self.skip(path, f"unsupported dtype {expected.dtype}")

    def _float_array(self, np, expected, actual, path):
        rtol, atol = self.tolerance(path)
        nan_e, nan_a = np.isnan(expected), np.isnan(actual)
        if (nan_e != nan_a).any():
            self.diff(path, f"NaN positions differ (first at {_first(nan_e != nan_a)})")
            return
        inf_e, inf_a = np.isinf(expected), np.isinf(actual)
        if (inf_e != inf_a).any() or (expected[inf_e] != actual[inf_a]).any():
            self.diff(path, "infinite values differ")
            return
        finite = ~(nan_e | inf_e)
        e, a = expected[finite], actual[finite]
        with np.errstate(all="ignore"):
            error = np.abs(a - e)
            allowed = atol + rtol * np.abs(e)
            bad = ~(error <= allowed)
        if bad.any():
            worst = float(np.max(error[bad]))
            self.diff(
                path,
                f"{int(bad.sum())} of {e.size} values differ beyond "
                f"rtol={rtol}, atol={atol} (largest difference {worst:.6g})",
            )

    def _sparse(self, expected, actual, path):
        if expected.shape != actual.shape:
            self.diff(path, f"shape differs: {expected.shape} vs {actual.shape}")
            return
        if expected.dtype != actual.dtype:
            self.diff(path, f"dtype differs: {expected.dtype} vs {actual.dtype}")
            return
        representation = self.policy.sparse is SparsePolicy.REPRESENTATION
        if representation:
            if type(expected) is not type(actual):
                self.diff(path, f"sparse type differs: {_type_name(expected)}")
                return
            fields = {"csr": "indptr indices", "csc": "indptr indices"}.get(
                expected.format, "row col" if expected.format == "coo" else None
            )
            if fields is None:
                self.skip(path, f"sparse format {expected.format} not supported")
                return
            e_store, a_store = expected, actual
        else:
            # Values only: format and explicit zeros do not matter. The
            # nonzero pattern is compared exactly, so an exact zero against a
            # tiny value is a mismatch even inside a float tolerance.
            fields = "indptr indices"
            e_store, a_store = _canonical_csr(expected), _canonical_csr(actual)
        for name in fields.split():
            if not self._structure(
                getattr(e_store, name), getattr(a_store, name), path, representation
            ):
                return
        e_data, a_data = e_store.data, a_store.data
        if e_data.shape != a_data.shape:
            self.diff(path, "number of stored values differs")
            return
        before = self.compared
        self._array_values(e_data, a_data, path)
        # Every cell of the matrix was compared, stored or implicit zero.
        self.compared = before + int(expected.shape[0]) * int(expected.shape[1])

    def _structure(self, expected, actual, path, check_dtype) -> bool:
        np = sys.modules["numpy"]
        if check_dtype and expected.dtype != actual.dtype:
            self.diff(path, f"index dtype differs: {expected.dtype} vs {actual.dtype}")
            return False
        if expected.shape != actual.shape or not np.array_equal(expected, actual):
            self.diff(path, "positions of stored values differ")
            return False
        return True

    # -- pandas -----------------------------------------------------------

    def _dataframe(self, expected, actual, path):
        self._index(expected.index, actual.index, f"{path}[index]")
        e_cols, a_cols = expected.columns, actual.columns
        if not self._axis_metadata(e_cols, a_cols, f"{path}[columns]"):
            return
        for side in (e_cols, a_cols):
            if side.has_duplicates:
                self.skip(path, "duplicate column labels")
                return
            # Policy paths name columns by text, so every label must have its
            # own unambiguous text form.
            texts = [str(c) for c in side]
            bad = [c for c in side if not _addressable(str(c))]
            if bad or len(set(texts)) != len(texts):
                reason = (
                    f"the column labels {bad[:5]!r} cannot be named in a "
                    f"comparison rule (they contain '/', '[' or ']', or are "
                    f"empty, padded, '.' or '..')"
                    if bad
                    else "two column labels have the same text"
                )
                self.skip(path, f"this table was not compared: {reason}")
                return
        want = [c for c in e_cols if not self.excluded(f"{path}/{c}")]
        got = [c for c in a_cols if not self.excluded(f"{path}/{c}")]
        # Labels compare with their types: 1, 1.0 and True are different.
        if [(type(c), c) for c in want] != [(type(c), c) for c in got]:
            missing = [c for c in want if c not in got]
            extra = [c for c in got if c not in want]
            detail = (
                f"missing {missing!r}, unexpected {extra!r}"
                if missing or extra
                else "same labels in a different order or with other types"
            )
            self.diff(path, f"columns differ: {detail}")
            return
        for column in want:
            self._column(expected[column], actual[column], f"{path}/{column}")

    def _series(self, expected, actual, path):
        self._index(expected.index, actual.index, f"{path}[index]")
        if type(expected.name) is not type(actual.name) or (
            expected.name != actual.name
        ):
            self.diff(path, f"name differs: {expected.name!r} vs {actual.name!r}")
        self._column(expected, actual, path)

    def _axis_metadata(self, expected, actual, path) -> bool:
        """Type, names, dtype and categories of an Index; False if incomparable."""
        if type(expected) is not type(actual):
            self.diff(path, f"type differs: {_type_name(expected)}")
            return False
        if _is(expected, "pandas", "MultiIndex"):
            self.skip(path, "MultiIndex is not supported")
            return False
        if _typed(expected.names) != _typed(actual.names):
            self.diff(path, f"names differ: {list(expected.names)!r}")
        if expected.dtype != actual.dtype or str(expected.dtype) != str(actual.dtype):
            self.diff(path, f"dtype differs: {expected.dtype} vs {actual.dtype}")
            return False
        if _is(expected, "pandas", "CategoricalIndex"):
            self.exact_depth += 1
            try:
                self._categories(expected.dtype, actual.dtype, path)
            finally:
                self.exact_depth -= 1
        return True

    def _categories(self, expected, actual, path):
        if expected.ordered != actual.ordered:
            self.diff(path, "categorical 'ordered' flag differs")
        self._index(expected.categories, actual.categories, f"{path}[categories]")

    def _index(self, expected, actual, path):
        self.exact_depth += 1
        try:
            if self._axis_metadata(expected, actual, path):
                self._extension_values(expected, actual, path, categories=False)
        finally:
            self.exact_depth -= 1

    def _column(self, expected, actual, path):
        if expected.dtype != actual.dtype or str(expected.dtype) != str(actual.dtype):
            self.diff(path, f"dtype differs: {expected.dtype} vs {actual.dtype}")
            return
        self._extension_values(expected, actual, path)

    def _extension_values(self, owner_e, owner_a, path, categories=True):
        """Compare two Series/Index values of equal dtype without losing metadata."""
        np = sys.modules["numpy"]
        pd = sys.modules["pandas"]
        dtype = owner_e.dtype
        expected, actual = owner_e.array, owner_a.array
        if len(expected) != len(actual):
            self.diff(path, f"length differs: {len(expected)} vs {len(actual)}")
            return
        masked = _masked_dtype_class()
        if isinstance(dtype, pd.CategoricalDtype):
            if categories:  # an Index has compared them with its metadata
                self._categories(dtype, actual.dtype, path)
            self._array_values(
                np.asarray(expected.codes), np.asarray(actual.codes), f"{path}[codes]"
            )
        elif masked is not None and isinstance(dtype, masked):
            if not self._same_missing(np, expected, actual, path):
                return
            zero = dtype.numpy_dtype.type(0)
            self._array_values(
                expected.to_numpy(dtype=dtype.numpy_dtype, na_value=zero),
                actual.to_numpy(dtype=dtype.numpy_dtype, na_value=zero),
                path,
            )
        elif isinstance(dtype, pd.StringDtype):
            if not self._same_missing(np, expected, actual, path):
                return
            keep = ~np.asarray(expected.isna())
            self._array_values(
                np.asarray(expected[keep], dtype=object),
                np.asarray(actual[keep], dtype=object),
                path,
            )
        elif isinstance(dtype, np.dtype):
            self._array_values(
                np.asarray(owner_e.to_numpy()), np.asarray(owner_a.to_numpy()), path
            )
        else:
            self.skip(path, f"unsupported pandas dtype {dtype}")

    def _same_missing(self, np, expected, actual, path) -> bool:
        if (np.asarray(expected.isna()) != np.asarray(actual.isna())).any():
            self.diff(path, "missing (NA) positions differ")
            return False
        return True

    # -- files ------------------------------------------------------------

    def _file(self, expected, actual, path):
        e_path, a_path = expected.path, actual.path
        for side in (e_path, a_path):
            if side.is_symlink() or not side.is_file():
                self.skip(path, f"not a regular file: {side.name}")
                return
        suffix = e_path.suffix.lower()
        if a_path.suffix.lower() != suffix:
            self.diff(path, f"file type differs: {e_path.name} vs {a_path.name}")
            return
        readers = {".csv": _read_csv, ".tsv": _read_tsv, ".json": _read_json}
        if suffix in readers:
            try:
                reference = readers[suffix](e_path)
            except ValueError as error:
                self.skip(path, f"reference {e_path.name} cannot be read: {error}")
                return
            try:
                output = readers[suffix](a_path)
            except ValueError as error:
                self.diff(path, f"output {a_path.name} cannot be read: {error}")
                return
            self.value(reference, output, path)
        elif suffix == ".h5ad":
            import anndata

            self.value(anndata.read_h5ad(e_path), anndata.read_h5ad(a_path), path)
        else:
            self.compared += 1
            if _sha256(e_path) != _sha256(a_path):
                self.diff(path, "file contents differ")


def _read_delimited(path: Path, delimiter: str) -> List[List[str]]:
    """Rows of cell text, with no type inference or missing-value guessing."""
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter=delimiter, strict=True)
            return [list(row) for row in reader]
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError(str(error)) from error


def _read_csv(path: Path) -> List[List[str]]:
    return _read_delimited(path, ",")


def _read_tsv(path: Path) -> List[List[str]]:
    return _read_delimited(path, "\t")


def _read_json(path: Path) -> Any:
    try:
        return strict_json_loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, ContractError) as error:
        raise ValueError(str(error)) from error


def _addressable(key: Any) -> bool:
    """A key a policy path can name without colliding with element notation."""
    return (
        isinstance(key, str)
        and key != ""
        and not any(ch in key for ch in "/[]")
        and key.strip() == key
        and key not in (".", "..")
    )


def _typed(names: Any) -> List[Tuple[type, Any]]:
    return [(type(name), name) for name in names]


def _first(mask: Any) -> Any:
    np = sys.modules["numpy"]
    mask = np.asarray(mask)
    if mask.ndim == 0:
        return ()
    index = np.argwhere(mask)[0]
    return tuple(int(i) for i in index) if index.size > 1 else int(index[0])


def _canonical_csr(matrix: Any) -> Any:
    canon = matrix.tocsr(copy=True)
    canon.sum_duplicates()
    canon.eliminate_zeros()
    canon.sort_indices()
    return canon


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["MAX_FINDINGS", "OutputFile", "compare", "output_files"]
