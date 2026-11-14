"""Declared column roles for a retained drawing table; never infer from values."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

from .records import json_blob_bytes

SCHEMA = "lobster.plot_data/1"
ROLES = {"x", "y", "z", "color", "size", "id", "label", "filter"}
DATA_FIELDS = (
    "type",
    "x",
    "y",
    "z",
    "ids",
    "customdata",
    "values",
    "labels",
    "dimensions",
    "locations",
    "lat",
    "lon",
    "r",
    "theta",
)


def _parsed(figure: Figure | dict[str, Any]) -> dict[str, Any]:
    if isinstance(figure, dict):
        return figure
    from .serialize import figure_to_bytes

    return cast(dict[str, Any], json.loads(figure_to_bytes(figure)))


def _normalize_numbers(value: Any) -> Any:
    # JSON.stringify turns 1.0 and -0.0 into 1 and 0. These are the same data;
    # distinct large integer values must remain distinct.
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_normalize_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_numbers(item) for key, item in value.items()}
    return value


def _binding(
    figure: dict[str, Any], roles: list[dict[str, Any]], *, normalize: bool = True
) -> str:
    traces = []
    for trace in figure.get("data", []):
        values = {key: trace[key] for key in DATA_FIELDS if key in trace}
        # A later constant marker-size edit no longer represents a source column.
        if any(item.get("role") == "size" for item in roles):
            values["marker_size"] = trace.get("marker", {}).get("size")
        traces.append(values)
    return hashlib.sha256(
        json_blob_bytes(_normalize_numbers(traces) if normalize else traces)
    ).hexdigest()


def record_column_roles(
    figure: Figure,
    roles: list[dict[str, Any]],
    *,
    transformations: list[dict[str, Any]] | None = None,
    selection: dict[str, Any] | None = None,
) -> None:
    """Annotate a generated figure; trusted saving retains this as figure metadata.

    Each role names a source column and its use. Transformations must be supplied
    by the drawing function, not inferred from axis labels or matching values.
    The binding prevents stale template metadata from describing changed traces.
    """
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("role"), str)
        or item["role"] not in ROLES
        or not isinstance(item.get("column"), str)
        or not item["column"]
        for item in roles
    ):
        raise ValueError("Invalid plot column roles")
    parsed = _parsed(figure)
    metadata = dict(figure.layout.meta or {})
    metadata["lobster_data"] = {
        "schema": SCHEMA,
        "roles": roles,
        "transformations": transformations or [],
        "selection": selection or {},
        "binding": _binding(parsed, roles),
    }
    figure.update_layout(meta=metadata)


def column_mapping(figure: dict[str, Any], columns: list[str]) -> dict[str, Any] | None:
    """Return only applicable declared metadata; absent/stale/ambiguous is unknown."""
    metadata = figure.get("layout", {}).get("meta")
    mapping = metadata.get("lobster_data") if isinstance(metadata, dict) else None
    if not isinstance(mapping, dict) or mapping.get("schema") != SCHEMA:
        return None
    roles = mapping.get("roles")
    if not isinstance(roles, list) or not roles or len(set(columns)) != len(columns):
        return None
    if not isinstance(mapping.get("transformations", []), list) or not isinstance(
        mapping.get("selection", {}), dict
    ):
        return None
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("role"), str)
        or item["role"] not in ROLES
        or item.get("column") not in columns
        for item in roles
    ):
        return None
    # Accept already-persisted hashes as well as normalized new bindings.
    if mapping.get("binding") != _binding(figure, roles) and mapping.get(
        "binding"
    ) != _binding(figure, roles, normalize=False):
        return None
    return mapping


def refresh_column_mapping(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Normalize an inherited mapping only when the actual data/roles are unchanged.

    An old raw-number hash can be upgraded on a new revision without rewriting
    its parent. Do not resurrect invalid metadata or rebind genuinely changed data.
    """
    before_meta = before.get("layout", {}).get("meta")
    after_meta = after.get("layout", {}).get("meta")
    if not isinstance(before_meta, dict) or not isinstance(after_meta, dict):
        return False
    old, new = before_meta.get("lobster_data"), after_meta.get("lobster_data")
    if not isinstance(old, dict) or not isinstance(new, dict):
        return False
    roles = old.get("roles")
    if not isinstance(roles, list):
        return False
    columns = list(
        dict.fromkeys(
            item["column"]
            for item in roles
            if isinstance(item, dict) and isinstance(item.get("column"), str)
        )
    )
    if column_mapping(before, columns) is None:
        return False
    if {k: v for k, v in old.items() if k != "binding"} != {
        k: v for k, v in new.items() if k != "binding"
    }:
        return False
    binding = _binding(before, roles)
    if binding != _binding(after, roles) or new.get("binding") == binding:
        return False
    after_meta["lobster_data"] = {**new, "binding": binding}
    return True
