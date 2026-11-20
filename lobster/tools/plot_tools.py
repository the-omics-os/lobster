"""
Plot tools factory for agents that need general-purpose interactive visualization.

Creates Plotly figures from tabular data (JSON or workspace files) and registers
them with PlotManager for canvas delivery. Scoped to 6 reliable plot types.

Architecture:
    Factory Pattern: create_plot_tools(data_manager, workspace_path) returns list of @tool functions
    Path Safety: File paths resolved relative to workspace_path, traversal blocked
    AQUADIF: UTILITY category, no provenance IR
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Callable, List

from langchain_core.tools import InjectedToolCallId, tool

logger = logging.getLogger(__name__)

MAX_INLINE_BYTES = 100_000
MAX_FILE_BYTES = 10_000_000
MAX_ROWS = 50_000
SAMPLE_THRESHOLD = 10_000

VALID_PLOT_TYPES = {"scatter", "line", "bar", "histogram", "box", "heatmap"}

REQUIRED_COLUMNS = {
    "scatter": ["x", "y"],
    "line": ["x", "y"],
    "bar": ["x", "y"],
    "histogram": ["x"],
    "box": ["y"],
    "heatmap": [],
}


def _resolve_safe_path(workspace_path: Path, relative_path: str) -> Path:
    """Resolve path safely within workspace boundary."""
    input_path = Path(relative_path).expanduser()
    if input_path.is_absolute():
        resolved = input_path.resolve()
    else:
        resolved = (workspace_path / input_path).resolve()

    workspace_resolved = workspace_path.resolve()
    if resolved != workspace_resolved and workspace_resolved not in resolved.parents:
        raise ValueError(f"Path '{relative_path}' resolves outside workspace boundary.")
    return resolved


def _load_data(data_json: str, data_file: str, workspace_path: Path):
    """Load data from inline JSON or workspace file. Returns DataFrame or error string."""
    import pandas as pd

    if not data_json and not data_file:
        return "ERROR: Provide either data_json or data_file."

    if data_json:
        if len(data_json.encode()) > MAX_INLINE_BYTES:
            return f"ERROR: data_json exceeds {MAX_INLINE_BYTES // 1000}KB limit."
        try:
            records = json.loads(data_json)
            if not isinstance(records, list):
                return "ERROR: data_json must be a JSON array of objects."
            return pd.DataFrame(records)
        except (json.JSONDecodeError, ValueError) as e:
            return f"ERROR: Invalid JSON: {e}"

    try:
        resolved = _resolve_safe_path(workspace_path, data_file)
    except ValueError as e:
        return f"ERROR: {e}"

    if not resolved.exists():
        return f"ERROR: File not found: {data_file}"

    file_size = resolved.stat().st_size
    if file_size > MAX_FILE_BYTES:
        return f"ERROR: File {data_file} is {file_size // 1_000_000}MB, exceeds 10MB limit."

    suffix = resolved.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(resolved)
        elif suffix in (".tsv", ".txt"):
            return pd.read_csv(resolved, sep="\t")
        elif suffix == ".json":
            return pd.read_json(resolved)
        elif suffix in (".xls", ".xlsx"):
            return pd.read_excel(resolved)
        elif suffix == ".parquet":
            return pd.read_parquet(resolved)
        else:
            return pd.read_csv(resolved)
    except Exception as e:
        return f"ERROR: Could not read {data_file}: {e}"


def _create_figure(
    plot_type: str,
    df,
    x: str,
    y: str,
    z: str,
    color: str,
    size: str,
    title: str,
    id_col: str | None = None,
):
    """Dispatch to Plotly Express. Returns Figure or error string."""
    import plotly.express as px
    import plotly.graph_objects as go

    required = REQUIRED_COLUMNS[plot_type]
    missing = []
    for col_name in required:
        col_val = {"x": x, "y": y, "z": z}.get(col_name, "")
        if not col_val:
            missing.append(col_name)
    if missing:
        return f"ERROR: {plot_type} requires columns: {missing}"

    kwargs = {"title": title}
    if color and color in df.columns:
        kwargs["color"] = color

    try:
        if plot_type == "scatter":
            if size and size in df.columns:
                kwargs["size"] = size
            if len(df) > SAMPLE_THRESHOLD:
                kwargs["render_mode"] = "webgl"
            if id_col:
                from lobster.core.figures.point_identity import identity_table

                df, identity_field = identity_table(df, id_col)
                kwargs["custom_data"] = [identity_field]
            fig = px.scatter(df, x=x, y=y, **kwargs)
            if id_col:
                from lobster.core.figures.point_identity import bind_point_ids

                bind_point_ids(fig, id_col)
        elif plot_type == "line":
            fig = px.line(df, x=x, y=y, **kwargs)
        elif plot_type == "bar":
            fig = px.bar(df, x=x, y=y, **kwargs)
        elif plot_type == "histogram":
            fig = px.histogram(df, x=x, **kwargs)
        elif plot_type == "box":
            fig = px.box(df, y=y, x=x if x else None, **kwargs)
        elif plot_type == "heatmap":
            if z:
                if z not in df.columns:
                    return f"ERROR: Column '{z}' not found. Available: {list(df.columns)[:10]}"
                if not x or not y:
                    return "ERROR: Heatmap with z requires x (columns) and y (index) for pivot."
                if x not in df.columns or y not in df.columns:
                    return f"ERROR: x='{x}' or y='{y}' not found in columns."
                pivot = df.pivot_table(index=y, columns=x, values=z)
                fig = go.Figure(
                    data=go.Heatmap(
                        z=pivot.values, x=list(pivot.columns), y=list(pivot.index)
                    )
                )
                fig.update_layout(title=title)
            else:
                numeric_cols = df.select_dtypes(include="number").columns.tolist()
                if len(numeric_cols) < 2:
                    return "ERROR: Correlation heatmap requires at least 2 numeric columns."
                fig = px.imshow(df[numeric_cols].corr(), title=title)
        else:
            return f"ERROR: Unsupported plot_type: {plot_type}"
    except Exception as e:
        return f"ERROR: Plot creation failed: {e}"

    return fig


def _describe_commit(results) -> str:
    """State exactly what the trusted save step published. Never names an unwritten file.

    Uses the central outcome vocabulary: a pending publication is "confirmation pending"
    (another caller may already have it confirmed), never a definitive absence.
    """
    from lobster.core.figures.errors import safe_result_reason
    from lobster.core.figures.records import STATUS_AVAILABLE, STATUS_PENDING
    from lobster.core.figures.saver import (
        HEADER_NONE,
        HEADER_PENDING,
        WITHHELD,
        outcome_state,
    )

    parts = []
    for res in results:
        reason = safe_result_reason(res.reason) or WITHHELD
        if res.status == STATUS_AVAILABLE:
            parts.append(f"saved: {res.ref} → {res.path}")
        elif res.status == STATUS_PENDING:
            parts.append(f"confirmation pending: {reason}")
        else:
            parts.append(f"not saved ({_safe_status(res.status)}): {reason}")
    state = outcome_state(results)
    if state == "pending":
        parts.insert(0, HEADER_PENDING)
    elif state == "none":
        parts.insert(0, HEADER_NONE)
    return " | ".join(parts)


def _safe_status(status) -> str:
    from lobster.core.figures.records import STATUS_FAILED, STATUSES

    return status if status in STATUSES else STATUS_FAILED


def _raised_pending(exc: BaseException) -> bool:
    """True if a raised figure fault carries a pending publication (status only: raised
    details never authenticate a ref, a path or an available save)."""
    from lobster.core.figures.errors import FigureError
    from lobster.core.figures.records import STATUS_PENDING

    details = getattr(exc, "details", None)
    results = details.get("results") if isinstance(details, dict) else None
    return (
        isinstance(exc, FigureError)
        and isinstance(results, list)
        and any(
            isinstance(r, dict) and r.get("status") == STATUS_PENDING for r in results
        )
    )


def _raised_summary(exc: BaseException) -> str:
    """The central ``failure_summary`` of a raised fault, with raised ``available`` statuses
    shown as ``unconfirmed: reason_withheld``: raised details never confirm a save."""
    from lobster.core.figures.errors import FigureError
    from lobster.core.figures.records import STATUS_AVAILABLE
    from lobster.core.figures.saver import WITHHELD, failure_summary

    details = getattr(exc, "details", None)
    results = details.get("results") if isinstance(details, dict) else None
    if not (
        isinstance(exc, FigureError)
        and isinstance(results, list)
        and results
        and all(isinstance(r, dict) for r in results)
    ):
        return failure_summary(exc)
    parts = []
    for r in results:
        if r.get("status") == STATUS_AVAILABLE:
            parts.append(f"unconfirmed: {WITHHELD}")
        else:  # status + reason only, through the central formatter
            single = {"status": r.get("status"), "reason": r.get("reason")}
            parts.append(
                failure_summary(FigureError("", details={"results": [single]}))
            )
    return "; ".join(parts)


def create_plot_tools(data_manager, workspace_path: Path) -> List[Callable]:
    """Factory: create interactive plot tools bound to a data_manager and workspace.

    Args:
        data_manager: DataManagerV2 instance (must have .plot_manager attribute)
        workspace_path: Root directory for resolving file paths

    Returns:
        List of tool callables with AQUADIF metadata
    """

    @tool
    def inspect_plot_data(data_file: str) -> str:
        """Inspect a workspace table before plotting: columns, numeric columns and sample rows.

        data_file is relative to the Lobster data workspace, e.g. data/points.csv.
        Uses the same file root as create_interactive_plot; no AnnData is needed.
        """
        df = _load_data("", data_file, workspace_path)
        if isinstance(df, str):
            return df
        from lobster.core.figures.point_identity import identity_candidates

        candidates = identity_candidates(df)
        return json.dumps(
            {
                "id_candidates": candidates,
                "suggested_id_column": candidates[0] if len(candidates) == 1 else None,
                "data_file": data_file,
                "rows": len(df),
                "columns": [str(c) for c in df.columns],
                "numeric_columns": list(df.select_dtypes(include="number").columns),
                "preview": json.loads(df.head(8).to_json(orient="records")),
            }
        )

    @tool
    def create_interactive_plot(
        plot_type: str,
        data_json: str = "",
        data_file: str = "",
        x: str = "",
        y: str = "",
        z: str = "",
        color: str = "",
        size: str = "",
        title: str = "Plot",
        x_label: str = "",
        y_label: str = "",
        related_data_files: List[str] | None = None,
        id_col: str | None = None,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> str:
        """Create AND save a managed interactive Plotly figure from a workspace table.

        Use inspect_plot_data first to identify column names. The reply includes
        the saved figure revision reference; no extra export or code step is needed.

        Args:
            plot_type: scatter, line, bar, histogram, box, or heatmap
            data_json: Inline JSON (list-of-dicts). Max 100KB.
            data_file: Workspace-relative path to CSV/TSV/JSON/Excel/Parquet.
            x: Column name for x-axis.
            y: Column name for y-axis.
            z: Column for heatmap values.
            color: Column for color grouping.
            size: Column for marker size (scatter only).
            id_col: Identity column for searchable scatter points. If omitted, a
                single valid conventional ID column is selected automatically.
                inspect_plot_data lists candidates; ask when several are plausible.
                IDs must be nonblank and unique before sampling. Never use row numbers.
            title: Plot title.
            x_label: Custom x-axis label.
            y_label: Custom y-axis label.
            related_data_files: Explicit supporting workspace tables to snapshot. Only
                include files linked by this analysis or named by the user; never guess.
        """
        if plot_type not in VALID_PLOT_TYPES:
            return f"ERROR: plot_type must be one of {sorted(VALID_PLOT_TYPES)}"

        df = _load_data(data_json, data_file, workspace_path)
        if isinstance(df, str):
            return df

        if len(df) > MAX_ROWS:
            return f"ERROR: {len(df)} rows exceeds {MAX_ROWS} limit. Aggregate or filter first."
        from lobster.core.figures.point_identity import resolve_identity_column

        if id_col is not None and plot_type != "scatter":
            return "ERROR: id_col is supported for scatter point plots only."
        try:
            id_col = (
                resolve_identity_column(df, id_col) if plot_type == "scatter" else None
            )
        except ValueError as exc:
            return f"ERROR: {exc}"
        original_df = df.copy()
        sampled = False
        if len(df) > SAMPLE_THRESHOLD:
            if plot_type in ("line", "bar", "heatmap"):
                df = df.head(SAMPLE_THRESHOLD)
            else:
                df = df.sample(n=SAMPLE_THRESHOLD, random_state=42)
            sampled = True

        fig = _create_figure(plot_type, df, x, y, z, color, size, title, id_col=id_col)
        if isinstance(fig, str):
            return fig

        if x_label:
            fig.update_layout(xaxis_title=x_label)
        if y_label:
            fig.update_layout(yaxis_title=y_label)

        from lobster.core.figures.data_view import record_column_roles

        role_columns = {"x": x}
        if plot_type != "histogram":
            role_columns["y"] = y
        if plot_type != "heatmap":
            role_columns["color"] = color
        if plot_type == "scatter":
            role_columns["size"] = size
            role_columns["id"] = id_col
        if plot_type == "heatmap":
            role_columns["z"] = z
        roles = [
            {"column": column, "role": role}
            for role, column in role_columns.items()
            if column and column in df.columns
        ]
        if roles:
            operations = []
            if plot_type in ("histogram", "box", "heatmap"):
                operations.append(
                    {
                        "operation": plot_type,
                        "description": "Display summarizes the supplied table.",
                    }
                )
            record_column_roles(
                fig,
                roles,
                transformations=operations,
                selection={
                    "input_rows": len(original_df),
                    "plotted_rows": len(df),
                    "sampled": sampled,
                    "method": (
                        (
                            "first rows"
                            if plot_type in ("line", "bar", "heatmap")
                            else "random sample (seed 42)"
                        )
                        if sampled
                        else "all supplied rows"
                    ),
                },
            )

        plot_manager = getattr(data_manager, "plot_manager", None)
        if not plot_manager:
            return "ERROR: plot_manager not available on data_manager"

        from lobster.core.figures.attachments import save_attachment
        from lobster.core.figures.payload import source_to_bytes
        from lobster.core.figures.records import SourcePayload

        store = data_manager.figure_store
        attachments = []
        try:
            attachments.append(
                save_attachment(
                    store,
                    df.to_csv(index=False).encode(),
                    label="plotted_data.csv",
                    role="plotted_data",
                    columns=list(map(str, df.columns)),
                    description="Exact table supplied to the plotting operation after any sampling.",
                    origin=data_file or "inline data",
                )
            )
            if sampled:
                attachments.append(
                    save_attachment(
                        store,
                        original_df.to_csv(index=False).encode(),
                        label="full_input.csv",
                        role="related_data",
                        columns=list(map(str, original_df.columns)),
                        description="Full input table before plot sampling.",
                        origin=data_file or "inline data",
                    )
                )
            for relative in related_data_files or []:
                path = _resolve_safe_path(workspace_path, relative)
                if path.stat().st_size > MAX_FILE_BYTES:
                    raise ValueError("Related data file exceeds 10 MB")
                attachments.append(
                    save_attachment(
                        store,
                        path.read_bytes(),
                        label=path.name,
                        role="related_data",
                        origin=relative,
                        description="Explicit supporting analysis artifact.",
                    )
                )
            declared = {
                "attachments": attachments,
                "plot_type": plot_type,
                "x": x,
                "y": y,
                "z": z,
                "color": color,
                "size": size,
                "sampled": sampled,
                "id_col": id_col,
                "column_mapping": (fig.layout.meta or {}).get("lobster_data"),
            }
            retained_source = SourcePayload(source_to_bytes(df, declared), declared)
        except (OSError, ValueError) as exc:
            return f"ERROR: Could not retain plot data: {exc}"

        plot_id, stats, _ = plot_manager.add_plot(
            plot=fig,
            title=title,
            source="data_expert",
            retained_source=retained_source,
            dataset_info={
                "rows": len(df),
                "columns": list(df.columns)[:20],
                "plot_type": plot_type,
                "sampled": sampled,
            },
        )

        if not plot_id:
            return f"ERROR: plot registration failed: {stats.get('error', 'unknown')}"

        suffix = f" (sampled to {SAMPLE_THRESHOLD} rows)" if sampled else ""
        try:
            # UTILITY tool without provenance: the execution record says so honestly.
            results = plot_manager.commit_staged(
                None,
                tool_call_id=tool_call_id,
                staged_ids=[plot_id],
                tool_name="create_interactive_plot",
            )
        except Exception as exc:
            from lobster.core.figures.saver import HEADER_PENDING

            plot_manager.discard_staged([plot_id])
            # fixed-vocabulary codes only, never exception text, raised refs or a raised
            # "available"
            summary = _raised_summary(exc)
            if _raised_pending(exc):
                return f"Plot: {title} | {HEADER_PENDING} | {summary}"
            return f"Plot not saved: {title} | {summary}"
        return f"Plot: {title} | {len(df)} rows{suffix} | " + _describe_commit(results)

    create_interactive_plot.metadata = {"categories": ["UTILITY"], "provenance": False}
    create_interactive_plot.tags = ["UTILITY"]
    inspect_plot_data.metadata = {"categories": ["UTILITY"], "provenance": False}
    inspect_plot_data.tags = ["UTILITY"]

    return [create_interactive_plot, inspect_plot_data]
