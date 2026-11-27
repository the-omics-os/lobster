"""Generic descriptive charts. Imports stay lazy for on-demand discovery."""

from lobster.figure_templates import figure_template
from lobster.figure_templates._style import (
    DIVERGING,
    HOUSE_COLORS,
    SEQUENTIAL_EXPRESSION,
    value_scale,
)

TEMPLATE_MANIFEST = [
    {"name": f"generic.{name}", "version": "1", "purpose": purpose, "callable": name}
    for name, purpose in (
        ("scatter", "Scatter of tabular values, optionally grouped."),
        ("line", "Line chart in supplied row order, optionally grouped."),
        ("bar", "Bar chart of supplied values (no implicit aggregation)."),
        ("histogram", "Descriptive histogram with explicit bin count."),
        ("box", "Box distributions by category."),
        ("heatmap", "Draw a supplied matrix without clustering or fitting."),
        ("correlation", "Descriptive correlation of selected numeric columns."),
    )
]


def _xy(kind, data, x, y, color, id_col, entity_kind, title, **kwargs):
    import pandas as pd
    import plotly.express as px

    table = pd.DataFrame(data).copy(deep=True)
    if id_col is not None:
        if table[id_col].isna().any() or table[id_col].astype(str).duplicated().any():
            raise ValueError(
                "Entity IDs must be present and unique; row positions are not IDs"
            )
        table[id_col] = table[id_col].astype(str)
    if color is not None and color in table.columns:
        if kind == "line" or table[color].dtype.kind not in "iufc":
            # px pins every trace to circle/solid; mapping the same column makes
            # it cycle the template's symbols and dashes in step with the colours.
            # px.line draws one line per colour value, numeric ones included.
            if kind == "scatter":
                kwargs["symbol"] = color
            elif kind == "line":
                kwargs["line_dash"] = color
        else:
            # Continuous colour on marks: grey start, or diverging when signed
            scale, limit = value_scale(table[color], unsigned=SEQUENTIAL_EXPRESSION)
            kwargs["color_continuous_scale"] = scale
            if limit:
                kwargs.update(range_color=[-limit, limit], color_continuous_midpoint=0)
    figure = getattr(px, kind)(
        table,
        x=x,
        y=y,
        color=color,
        title=title,
        custom_data=[id_col] if id_col is not None else None,
        color_discrete_sequence=HOUSE_COLORS,
        **kwargs,
    )
    for trace in figure.data:
        trace.uid = "class:" + (str(trace.name) if trace.name else "series")
        if id_col is not None:
            trace.meta = {
                "lobster_ids": {
                    "entity_kind": entity_kind,
                    "comparison": None,
                    "customdata_index": 0,
                }
            }
    return figure


@figure_template("generic.scatter")
def scatter(
    data,
    *,
    x,
    y,
    color=None,
    size=None,
    id_col=None,
    entity_kind="entity",
    title="Scatter",
):
    return _xy("scatter", data, x, y, color, id_col, entity_kind, title, size=size)


@figure_template("generic.line")
def line(data, *, x, y, color=None, id_col=None, entity_kind="entity", title="Line"):
    return _xy("line", data, x, y, color, id_col, entity_kind, title)


@figure_template("generic.bar")
def bar(data, *, x, y, color=None, id_col=None, entity_kind="entity", title="Bar"):
    return _xy("bar", data, x, y, color, id_col, entity_kind, title)


@figure_template("generic.histogram")
def histogram(data, *, x, color=None, nbins=30, title="Histogram"):
    import pandas as pd
    import plotly.express as px

    if not isinstance(nbins, int) or nbins < 1:
        raise ValueError("nbins must be a positive integer")
    figure = px.histogram(
        pd.DataFrame(data).copy(deep=True),
        x=x,
        color=color,
        nbins=nbins,
        title=title,
        color_discrete_sequence=HOUSE_COLORS,
    )
    for trace in figure.data:
        trace.uid = "class:" + (str(trace.name) if trace.name else "distribution")
    return figure  # Aggregated bins do not claim single-entity selection.


@figure_template("generic.box")
def box(data, *, x, y, color=None, points=False, title="Box distributions"):
    import pandas as pd
    import plotly.express as px

    figure = px.box(
        pd.DataFrame(data).copy(deep=True),
        x=x,
        y=y,
        color=color,
        points=points,
        title=title,
        color_discrete_sequence=HOUSE_COLORS,
    )
    for trace in figure.data:
        trace.uid = "class:" + (str(trace.name) if trace.name else "distribution")
    return figure


@figure_template("generic.heatmap")
def heatmap(data, *, x_labels=None, y_labels=None, colorscale=None, title="Heatmap"):
    import numpy as np
    import plotly.graph_objects as go

    matrix = np.asarray(data)
    if matrix.ndim != 2:
        raise ValueError("Heatmap data must be a two-dimensional matrix")
    if x_labels is not None and len(x_labels) != matrix.shape[1]:
        raise ValueError("x_labels do not match matrix columns")
    if y_labels is not None and len(y_labels) != matrix.shape[0]:
        raise ValueError("y_labels do not match matrix rows")
    scale = {"colorscale": colorscale}
    if colorscale is None and matrix.dtype.kind in "fiu":
        # A signed matrix diverges around 0 over a symmetric range.
        chosen, limit = value_scale(matrix)
        scale = {"colorscale": chosen}
        if limit:
            scale.update(zmid=0, zmin=-limit, zmax=limit)
    figure = go.Figure(
        go.Heatmap(z=matrix, x=x_labels, y=y_labels, uid="matrix:values", **scale)
    )
    figure.update_layout(title=title)
    return figure


@figure_template("generic.correlation")
def correlation(data, *, columns=None, method="pearson", title="Correlation"):
    import pandas as pd
    import plotly.graph_objects as go

    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("Use pearson, spearman or kendall correlation")
    table = pd.DataFrame(data).copy(deep=True)
    selected = (
        table[columns] if columns is not None else table.select_dtypes(include="number")
    )
    if selected.shape[1] == 0:
        raise ValueError("Correlation requires numeric columns")
    corr = selected.corr(method=method)
    figure = go.Figure(
        go.Heatmap(
            z=corr.to_numpy(),
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            zmid=0,
            zmin=-1,
            zmax=1,
            colorscale=DIVERGING,
            uid="matrix:correlation",
        )
    )
    figure.update_layout(title=title)
    return figure  # Pairwise descriptive transform; no inference/testing.
