"""What ``lobster_inspect`` reads out of an ``.h5ad`` file, and nothing else.

The file is opened read-only with ``h5py`` and the listing comes from the
structure alone: shapes, names and dtypes. The expression matrix ``X`` is never
read, and neither are layers, ``obsm`` or ``uns`` (``anndata.read_h5ad`` in
backed mode loads those into memory, which can be gigabytes). Only a column
asked for with ``column`` is read, and only that one.

Every name and value in a reply comes from the file, so each is cut and stripped
of control, format and line-separator characters before it is returned. A
number that is not finite is counted, never returned: the reply is always valid
JSON.
"""

from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_ITEMS = 50
MAX_ITEMS = 200
MAX_REPLY_CHARS = 20_000
TEXT_MAX_CHARS = 100
# The link check walks at most this many members, this deep (a cycle of hard
# links would otherwise never end).
MAX_LINK_CHECKS = 200_000
MAX_TREE_DEPTH = 16
# A column is read whole (the server answers one call at a time), so one of
# more rows than this is refused rather than allowed to hold the server.
MAX_COLUMN_ROWS = 5_000_000


class InspectRefused(Exception):
    """A refusal: ``code`` names it, the message is one line."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")


def inspect(path: Path, column: Optional[str], max_items: int) -> Dict[str, Any]:
    """The reply for *path* (an existing ``.h5ad`` file): its top-level listing,
    or the summary of one ``obs:<name>`` / ``var:<name>`` column. At most
    ``MAX_REPLY_CHARS`` characters once it is JSON."""
    if path.suffix != ".h5ad":
        raise InspectRefused("not_an_h5ad", f"{_text(path.name)} is not an .h5ad file")
    import h5py

    try:
        if not path.is_file():
            raise InspectRefused("unreadable", f"{_text(path.name)} is not a file")
        if not h5py.is_hdf5(path):
            raise InspectRefused("not_an_h5ad", f"{_text(path.name)} is not HDF5")
        handle = h5py.File(path, "r")
    except OSError as exc:
        raise InspectRefused("unreadable", _text(exc, 200)) from None
    try:
        _refuse_links(handle)
        for name in ("obs", "var"):
            if not isinstance(handle.get(name), h5py.Group):
                raise InspectRefused(
                    "not_an_h5ad", f"{_text(path.name)} has no AnnData obs and var"
                )
        items = max_items
        while True:
            try:
                reply = (
                    _column(handle, column, items)
                    if column is not None
                    else _listing(handle, items)
                )
            except InspectRefused:
                raise
            except Exception as exc:  # a layout this reader does not know
                raise InspectRefused("unreadable", _text(exc, 200)) from None
            reply = {"path": str(path), "format": "h5ad", **reply}
            if len(json.dumps(reply, allow_nan=False)) <= MAX_REPLY_CHARS or items <= 1:
                return reply
            items = max(1, items // 2)  # the lists shrink until the reply fits
    finally:
        handle.close()


def _refuse_links(handle: Any) -> None:
    """Refuse a file with an external or soft link where this module reads.

    h5py follows an external link into any file the process can open, so a file
    in an output folder could make a column of another file readable. The obs and
    var trees are walked whole (a column is read from there); the other groups
    only for their direct members, which is as far as the listing looks.
    """
    import h5py

    def walk(group: Any, depth: int, budget: List[int]) -> None:
        for name in group:
            budget[0] -= 1
            if budget[0] < 0:
                raise InspectRefused("unreadable", "the file has too many members")
            if not isinstance(group.get(name, getlink=True), h5py.HardLink):
                raise InspectRefused(
                    "unreadable",
                    "the file holds an HDF5 external or soft link, which is not followed",
                )
            node = group[name]
            if depth and isinstance(node, h5py.Group):
                walk(node, depth - 1, budget)

    budget = [MAX_LINK_CHECKS]
    walk(handle, 0, budget)
    for name in ("obs", "var", "raw", "layers", "obsm", "varm", "obsp", "varp", "uns"):
        if name in handle and isinstance(handle[name], h5py.Group):
            walk(handle[name], MAX_TREE_DEPTH if name in ("obs", "var") else 0, budget)


# --- the listing -------------------------------------------------------------


def _listing(f: Any, items: int) -> Dict[str, Any]:
    reply: Dict[str, Any] = {
        "n_obs": _length(f["obs"]),
        "n_vars": _length(f["var"]),
    }
    if "X" in f:
        reply["x"] = _matrix(f["X"])
    for axis in ("obs", "var"):
        names = _column_names(f[axis])
        reply[axis] = {
            "n_columns": len(names),
            "columns": [_column_info(f[axis][n], n) for n in names[:items]],
        }
        if len(names) > items:
            reply[axis]["columns_truncated"] = True
    for group in ("layers", "obsm", "varm"):
        reply[group] = _members(f, group, items, shapes=True)
    for group in ("obsp", "varp"):
        reply[group] = _members(f, group, items, shapes=False)
    uns = _members(f, "uns", items, shapes=False)
    reply["uns"] = {"keys": [m["name"] for m in uns["items"]], **_cut(uns)}
    if "raw" in f and "X" in f["raw"]:
        reply["raw"] = {"x": _matrix(f["raw"]["X"])}
    return reply


def _cut(members: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in members.items() if k != "items"}


def _members(f: Any, group: str, items: int, *, shapes: bool) -> Dict[str, Any]:
    """The first *items* names of a group (with shapes when asked), and how many
    there are when it is longer."""
    if group not in f:
        return {"items": []}
    keys = sorted(f[group].keys())
    shown = []
    for key in keys[:items]:
        entry: Dict[str, Any] = {"name": _text(key)}
        if shapes:
            entry.update(_matrix(f[group][key]))
        shown.append(entry)
    out: Dict[str, Any] = {"items": shown}
    if len(keys) > items:
        out.update(truncated=True, total=len(keys))
    return out


def _matrix(node: Any) -> Dict[str, Any]:
    """Shape and kind of a matrix node, from attributes only."""
    import h5py

    if isinstance(node, h5py.Dataset):
        return {"shape": [int(n) for n in node.shape], "kind": "dense"}
    shape = node.attrs.get("shape")
    kind = _attr_text(node.attrs.get("encoding-type")) or "group"
    out: Dict[str, Any] = {"kind": kind}
    if shape is not None:
        out["shape"] = [int(n) for n in shape]
    return out


def _length(frame: Any) -> int:
    """Rows of a dataframe group: the length of its index, which pandas 3 can
    store as a nullable-string group (``values`` and ``mask``) or as categories."""
    index = frame[_attr_text(frame.attrs.get("_index")) or "_index"]
    if hasattr(index, "shape"):
        return int(index.shape[0])
    return int((index["values"] if "values" in index else index["codes"]).shape[0])


def _column_names(frame: Any) -> List[str]:
    return [_attr_text(n) for n in frame.attrs.get("column-order", [])]


def _attr_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return "" if value is None else str(value)


def _column_info(node: Any, name: str) -> Dict[str, Any]:
    import h5py

    info: Dict[str, Any] = {"name": _text(name)}
    encoding = _attr_text(node.attrs.get("encoding-type"))
    if isinstance(node, h5py.Group) and encoding == "categorical":
        info.update(kind="categorical", n_categories=int(node["categories"].shape[0]))
    elif isinstance(node, h5py.Group) and encoding.startswith("nullable-"):
        values = node["values"]
        info.update(kind=_kind(encoding, values), dtype=_dtype(values))
    elif isinstance(node, h5py.Dataset):
        info.update(kind=_kind(encoding, node), dtype=_dtype(node))
    else:
        info["kind"] = "other"
    return info


def _kind(encoding: str, dataset: Any) -> str:
    if "string" in encoding or dataset.dtype.kind in "OSU":
        return "string"
    return {"b": "boolean", "i": "numeric", "u": "numeric", "f": "numeric"}.get(
        dataset.dtype.kind, "other"
    )


def _dtype(dataset: Any) -> str:
    return "string" if dataset.dtype.kind in "OSU" else str(dataset.dtype)


# --- one column --------------------------------------------------------------


def _column(f: Any, spec: str, items: int) -> Dict[str, Any]:
    axis, _, name = spec.partition(":")
    if axis not in ("obs", "var") or not name:
        raise InspectRefused(
            "unknown_column", "column must look like obs:<name> or var:<name>"
        )
    names = _column_names(f[axis])
    if name not in names:
        shown = ", ".join(_text(n) for n in names[:items])
        more = " ..." if len(names) > items else ""
        raise InspectRefused(
            "unknown_column",
            f"{axis} has no column {_text(name)}; it has: {shown}{more}",
        )
    rows = _length(f[axis])
    if rows > MAX_COLUMN_ROWS:
        raise InspectRefused(
            "unreadable",
            f"{axis} has {rows:,} rows; a column of more than "
            f"{MAX_COLUMN_ROWS:,} rows is not read",
        )
    import pandas as pd
    from anndata.io import read_elem

    series = pd.Series(read_elem(f[axis][name]))
    info = _column_info(f[axis][name], name)
    reply: Dict[str, Any] = {
        "column": f"{axis}:{_text(name)}",
        **{k: v for k, v in info.items() if k != "name"},
        "n": int(len(series)),
        "n_missing": int(series.isna().sum()),
    }
    if info["kind"] == "numeric":
        reply.update(_numeric(series))
    else:
        counts = series.dropna().astype(str).value_counts()
        reply["n_unique"] = int(len(counts))
        reply["values"] = [
            {"value": _text(v), "count": int(c)} for v, c in counts.head(items).items()
        ]
        if len(counts) > items:
            reply["values_truncated"] = True
    return reply


def _numeric(series: Any) -> Dict[str, Any]:
    import numpy as np

    values = series.to_numpy(dtype="float64", na_value=np.nan)
    finite = values[np.isfinite(values)]
    out: Dict[str, Any] = {
        "n_nan": int(np.isnan(values).sum()),
        "n_inf": int(np.isinf(values).sum()),
        "n_unique": int(len(np.unique(finite))),
    }
    if len(finite):
        out.update(
            min=_num(finite.min()),
            max=_num(finite.max()),
            mean=_num(finite.mean()),
            median=_num(np.median(finite)),
        )
    return out


def _num(value: Any) -> Optional[float]:
    number = float(value)
    return number if math.isfinite(number) else None


def _text(value: Any, limit: int = TEXT_MAX_CHARS) -> str:
    """*value* as one short printable line: control, format and line-separator
    characters become spaces."""
    text = "".join(
        " " if unicodedata.category(c) in ("Cc", "Cf", "Zl", "Zp") else c
        for c in str(value)
    )
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."
