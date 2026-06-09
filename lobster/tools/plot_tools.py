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
from typing import Callable, List

from langchain_core.tools import tool

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
        raise ValueError(
            f"Path '{relative_path}' resolves outside workspace boundary."
        )
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


def _create_figure(plot_type: str, df, x: str, y: str, z: str, color: str, size: str, title: str):
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
            fig = px.scatter(df, x=x, y=y, **kwargs)
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
                fig = go.Figure(data=go.Heatmap(z=pivot.values, x=list(pivot.columns), y=list(pivot.index)))
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


def create_plot_tools(data_manager, workspace_path: Path) -> List[Callable]:
    """Factory: create interactive plot tools bound to a data_manager and workspace.

    Args:
        data_manager: DataManagerV2 instance (must have .plot_manager attribute)
        workspace_path: Root directory for resolving file paths

    Returns:
        List of tool callables with AQUADIF metadata
    """

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
    ) -> str:
        """Create interactive Plotly plot from tabular data. Registers with canvas.

        Args:
            plot_type: scatter, line, bar, histogram, box, or heatmap
            data_json: Inline JSON (list-of-dicts). Max 100KB.
            data_file: Workspace-relative path to CSV/TSV/JSON/Excel/Parquet.
            x: Column name for x-axis.
            y: Column name for y-axis.
            z: Column for heatmap values.
            color: Column for color grouping.
            size: Column for marker size (scatter only).
            title: Plot title.
            x_label: Custom x-axis label.
            y_label: Custom y-axis label.
        """
        if plot_type not in VALID_PLOT_TYPES:
            return f"ERROR: plot_type must be one of {sorted(VALID_PLOT_TYPES)}"

        df = _load_data(data_json, data_file, workspace_path)
        if isinstance(df, str):
            return df

        if len(df) > MAX_ROWS:
            return f"ERROR: {len(df)} rows exceeds {MAX_ROWS} limit. Aggregate or filter first."
        sampled = False
        if len(df) > SAMPLE_THRESHOLD:
            if plot_type in ("line", "bar", "heatmap"):
                df = df.head(SAMPLE_THRESHOLD)
            else:
                df = df.sample(n=SAMPLE_THRESHOLD, random_state=42)
            sampled = True

        fig = _create_figure(plot_type, df, x, y, z, color, size, title)
        if isinstance(fig, str):
            return fig

        if x_label:
            fig.update_layout(xaxis_title=x_label)
        if y_label:
            fig.update_layout(yaxis_title=y_label)

        plot_manager = getattr(data_manager, "plot_manager", None)
        if not plot_manager:
            return "ERROR: plot_manager not available on data_manager"

        plot_id, stats, _ = plot_manager.add_plot(
            plot=fig,
            title=title,
            source="data_expert",
            dataset_info={
                "rows": len(df),
                "columns": list(df.columns)[:20],
                "plot_type": plot_type,
                "sampled": sampled,
            },
        )

        if not plot_id:
            return f"ERROR: plot registration failed: {stats.get('error', 'unknown')}"

        plot_manager.save_plots_to_workspace()

        suffix = f" (sampled to {SAMPLE_THRESHOLD} rows)" if sampled else ""
        return f"Plot registered: {plot_id} | {title} | {len(df)} rows{suffix}"

    create_interactive_plot.metadata = {"categories": ["UTILITY"], "provenance": False}
    create_interactive_plot.tags = ["UTILITY"]

    return [create_interactive_plot]
