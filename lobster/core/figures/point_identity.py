"""Carry declared table identities into point traces without matching coordinates."""

from __future__ import annotations

from typing import Any

# Only conventional identity names are candidates for automatic selection.
# Axis values, row indices and arbitrary unique columns are never guessed as IDs.
_ID_NAMES = {
    "id",
    "point_id",
    "entity_id",
    "sample_id",
    "feature_id",
    "gene_id",
    "protein_id",
    "peptide_id",
    "cell_id",
    "barcode",
    "accession",
    "gene_symbol",
}


def _valid_ids(series: Any) -> bool:
    values = series.astype(str)
    return len(series) > 0 and not (
        series.isna().any()
        or values.str.strip().eq("").any()
        or values.duplicated().any()
    )


def identity_candidates(table: Any) -> list[str]:
    """Return unambiguous-looking column names with complete, unique values."""
    return [
        column
        for column in table.columns
        if isinstance(column, str)
        and column.casefold().replace(" ", "_").replace("-", "_") in _ID_NAMES
        and _valid_ids(table[column])
    ]


def resolve_identity_column(table: Any, requested: str | None = None) -> str | None:
    if requested is not None:
        if requested not in table.columns:
            raise ValueError(f"Identity column {requested!r} was not found")
        if not _valid_ids(table[requested]):
            raise ValueError(
                "Entity IDs must be present, nonblank and unique; row positions are not IDs"
            )
        return requested
    candidates = identity_candidates(table)
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple identity columns are available: {candidates}. Specify id_col explicitly."
        )
    return candidates[0] if len(candidates) == 1 else None


def identity_table(table: Any, column: str) -> tuple[Any, str]:
    """Add a private string ID channel without changing columns used as axes."""
    private = "_lobster_point_id"
    while private in table.columns:
        private += "_"
    return table.assign(**{private: table[column].astype(str)}), private


def bind_point_ids(figure: Any, column: str, *, entity_kind: str = "entity") -> None:
    """Plotly Express carries this column in custom_data[0] through grouping/sorting.

    Call only after drawing with custom_data=[the private string ID column].
    This deliberately never joins on x/y values.
    """
    for trace in figure.data:
        if trace.type not in ("scatter", "scattergl"):
            continue
        ids = [row[0] for row in trace.customdata]
        if any(not isinstance(value, str) or not value.strip() for value in ids) or len(
            set(ids)
        ) != len(ids):
            raise ValueError("Plot trace did not retain unique string IDs")
        trace.ids = ids
        trace.meta = {
            **(trace.meta or {}),
            "lobster_ids": {
                "entity_kind": entity_kind,
                "id_column": column,
                "comparison": None,
                "customdata_index": 0,
            },
        }
