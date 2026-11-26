"""Omics-OS house style and stable entity labels for Plotly figures.

Values: Omics-OS design tokens omics.plotly.json (design@c04f862), hard-coded
so the engine does not depend on the design repository.
"""

from __future__ import annotations

HOUSE_STYLE_NAME = "omics_os"

# Categories in this order, graphite first; above 14 categories, cycle them.
# Never sample a value scale for categories.
HOUSE_COLORS = [
    "#404040",
    "#CC2C18",
    "#0BAA6B",
    "#CC9D0D",
    "#C91A6D",
    "#F7A696",
    "#93DAB3",
    "#F2C777",
    "#8C1E10",
    "#005732",
    "#844B00",
    "#8E0049",
    "#F3A3BB",
    "#8C8C8C",
]

_SEQUENTIAL_TAIL = ["#FFE6E1", "#F7A696", "#EE6853", "#CC2C18", "#8C1E10", "#431009"]


def _stops(colors):
    return [[i / (len(colors) - 1), color] for i, color in enumerate(colors)]


SEQUENTIAL = _stops(["#FFFFFF", *_SEQUENTIAL_TAIL])
SEQUENTIAL_EXPRESSION = _stops(["#E0E0E0", *_SEQUENTIAL_TAIL])
# White at 0.5: use with zmid 0 and a symmetric range.
DIVERGING = _stops(
    ["#005732", "#0BAA6B", "#D9F7E5", "#FFFFFF", "#FDE3EC", "#C91A6D", "#8E0049"]
)

INK = "#121212"
TICK = "#636363"
GRID = "#EBEBEB"
AXIS = "#E0E0E0"
NO_CALL = "#E0E0E0"
UP = "#C91A6D"
DOWN = "#0BAA6B"
KEY = "#CC2C18"

# Red and green must never be the only difference between two series, so the
# symbol and dash cycles run in step with the colorway (one entry per colour).
SCATTER_SYMBOLS = [
    "circle",
    "square",
    "diamond",
    "triangle-up",
    "triangle-down",
    "cross",
    "x",
    "star",
    "pentagon",
    "hexagon",
    "triangle-left",
    "triangle-right",
    "hourglass",
    "bowtie",
]
LINE_DASHES = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]


def _house_template():
    import plotly.graph_objects as go

    axis = {
        "showgrid": True,
        "gridcolor": GRID,
        "linecolor": AXIS,
        "zerolinecolor": AXIS,
        "tickcolor": AXIS,
        "tickfont": {"color": TICK},
        "title": {"font": {"color": INK}},
    }
    colorbar = {"outlinewidth": 0, "tickfont": {"color": TICK}}
    scatter_cycle = [
        {
            "marker": {"symbol": symbol},
            "line": {"dash": LINE_DASHES[i % len(LINE_DASHES)]},
        }
        for i, symbol in enumerate(SCATTER_SYMBOLS)
    ]
    return go.layout.Template(
        layout={
            "font": {"family": "Manrope, Arial, sans-serif", "size": 14, "color": INK},
            "paper_bgcolor": "#FFFFFF",
            "plot_bgcolor": "#FFFFFF",
            "colorway": HOUSE_COLORS,
            "colorscale": {
                "sequential": SEQUENTIAL,
                "sequentialminus": SEQUENTIAL,
                "diverging": DIVERGING,
            },
            "coloraxis": {"colorscale": SEQUENTIAL, "colorbar": colorbar},
            "xaxis": axis,
            "yaxis": axis,
            "margin": {"l": 65, "r": 30, "t": 70, "b": 65},
            "legend": {"bgcolor": "rgba(0,0,0,0)"},
            "shapedefaults": {"line": {"color": INK, "dash": "dash"}},
        },
        data={
            "scatter": scatter_cycle,
            "scattergl": scatter_cycle,
            "heatmap": [{"colorscale": SEQUENTIAL, "colorbar": colorbar}],
            "contour": [{"colorscale": SEQUENTIAL, "colorbar": colorbar}],
        },
    )


def initialize_house_style():
    """Set the code namespace's default, including fully custom Plotly figures."""
    import plotly.io as pio

    pio.templates[HOUSE_STYLE_NAME] = _house_template()
    pio.templates.default = HOUSE_STYLE_NAME


def apply_house_style(figure):
    """Embed the style in a normal mutable figure; no I/O or dataset writes."""
    figure.update_layout(template=_house_template())
    return figure


def set_trace_identity(trace, entity_ids, *, entity_kind, comparison=None):
    """Entity labels in column zero; the revision source supplies their namespace.

    No source identity is guessed here. With no frozen source in the envelope,
    these are labels only, not bound scientific selections. Aggregate marks
    must not call this function without an explicit membership mapping.
    """
    trace.customdata = [[str(entity_id)] for entity_id in entity_ids]
    meta = dict(trace.meta or {})
    meta["lobster_ids"] = {
        "entity_kind": entity_kind,
        "comparison": comparison,
        "customdata_index": 0,
    }
    trace.meta = meta
