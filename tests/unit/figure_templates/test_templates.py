"""Every shipped drawing template: fixture, recipe, style, identities and purity."""

import copy
import json
import numbers
import subprocess
import sys
from importlib.util import find_spec

import anndata as ad
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pytest

from lobster.figure_templates import (
    figure_template,
    get_template,
    initialize_house_style,
    list_templates,
    show_template_source,
)


@pytest.fixture
def cells():
    data = ad.AnnData(
        np.array([[0, 2], [4, 6], [2, 0], [0, 4]], dtype=float),
        obs=pd.DataFrame(
            {"leiden": ["b", "b", "a", "a"], "sample": ["s1", "s2", "s1", "s2"]},
            index=["cell-z", "cell-a", "cell-q", "cell-m"],
        ),
        var=pd.DataFrame(index=["gene-1", "gene-2"]),
    )
    data.obsm["X_umap"] = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    data.uns["pca"] = {"variance_ratio": [0.7, 0.2, 0.1]}
    data.obs["n_counts"] = data.X.sum(axis=1)
    data.obs["n_genes"] = (data.X > 0).sum(axis=1)
    data.obs["gene_detection_rate"] = data.obs["n_genes"] / data.n_vars * 100
    data.obs["log10_counts"] = np.log10(data.obs["n_counts"] + 1)
    data.obs["log10_genes"] = np.log10(data.obs["n_genes"] + 1)
    for column in ("percent_mito", "percent_ribo", "percent_hb"):
        data.obs[column] = [0, 2, 4, 6]
    return data


def _fixture(name, cells):
    if name.startswith("sc."):
        args = (
            {"genes": ["gene-1", "gene-2"]}
            if name.split(".")[1] in {"violin", "feature", "dot", "heatmap"}
            else {}
        )
        return cells, args
    if name == "proteomics.volcano":
        return pd.DataFrame(
            {
                "protein": ["up", "down", "ns"],
                "comparison": ["A_vs_B"] * 3,
                "log2_fold_change": [2, -2, 0.5],
                "p_adjusted": [0.01, 0.001, 0.7],
            }
        ), {"comparison": "A_vs_B"}
    if name == "proteomics.pathway_bubble":
        return (
            pd.DataFrame(
                {
                    "Term": ["P1", "P2"],
                    "Adjusted P-value": [0.001, 0.03],
                    "Overlap": ["5/30", "2/10"],
                }
            ),
            {},
        )
    if name == "chem.similarity_heatmap":
        return [[1, 0.4], [0.4, 1]], {"compound_ids": ["mol-z", "mol-a"]}
    if name == "generic.heatmap":
        return [[1, 2], [3, 4]], {"x_labels": ["g1", "g2"], "y_labels": ["s1", "s2"]}
    table = pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [4, 3, 2, 1],
            "group": ["b", "a", "b", "a"],
            "id": ["e4", "e3", "e2", "e1"],
        }
    )
    if name == "generic.correlation":
        return table, {"columns": ["x", "y"]}
    if name == "generic.histogram":
        return table, {"x": "x", "color": "group"}
    return table, {"x": "x", "y": "y", "color": "group"}


FIXTURE_NAMES = {
    "generic.scatter",
    "generic.line",
    "generic.bar",
    "generic.histogram",
    "generic.box",
    "generic.heatmap",
    "generic.correlation",
    "proteomics.volcano",
    "proteomics.pathway_bubble",
    "chem.similarity_heatmap",
    "sc.umap",
    "sc.violin",
    "sc.feature",
    "sc.dot",
    "sc.heatmap",
    "sc.elbow",
    "sc.cluster_composition",
    "sc.qc",
}
# Optional domains qualify when installed. Third-party providers supply their own
# fixtures; merely installing one must not make our fixture selector fail.
TEMPLATES = sorted(FIXTURE_NAMES & {item["name"] for item in list_templates()})


@pytest.mark.parametrize("name", TEMPLATES)
def test_template_fixture_purity_style_and_recipe(name, cells):
    data, args = _fixture(name, cells)
    before = copy.deepcopy(data)
    function = get_template(name + "@1")
    figure = function(data, **args)
    assert isinstance(figure, go.Figure)
    assert len(figure.data) > 0
    assert figure.layout.template.layout.font.family == "Manrope, Arial, sans-serif"
    recipe = figure.layout.meta["lobster_template"]
    assert recipe["name"] == name and recipe["version"] == "1"
    assert "data" not in recipe["args"]
    assert all(recipe["args"][key] == value for key, value in args.items())
    assert len(json.dumps(recipe["args"]).encode()) < 16 * 1024
    ids = [trace.uid for trace in figure.data]
    assert all(ids) and len(ids) == len(set(ids))
    payload = pio.to_json(figure, validate=True, remove_uids=False)
    reopened = pio.from_json(payload)
    assert [trace.uid for trace in reopened.data] == ids
    # Returned figures are unrestricted Plotly objects, not tool-bound configs.
    figure.update_layout(title="Edited freely")
    figure.add_annotation(text="Custom annotation", x=0, y=0)
    assert figure.layout.title.text == "Edited freely"
    assert figure.layout.meta["lobster_template"] == recipe
    if isinstance(data, pd.DataFrame):
        pd.testing.assert_frame_equal(data, before)
    elif isinstance(data, ad.AnnData):
        pd.testing.assert_frame_equal(data.obs, before.obs)
        pd.testing.assert_frame_equal(data.var, before.var)
        np.testing.assert_array_equal(data.X, before.X)
        np.testing.assert_array_equal(data.obsm["X_umap"], before.obsm["X_umap"])
    else:
        assert data == before


def test_core_discovery_is_cheap_and_on_demand():
    code = "from lobster.figure_templates import list_templates; import sys; assert list_templates(); assert not {'numpy', 'pandas', 'plotly'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-c", code], check=True)
    assert {"generic.scatter", "generic.line", "generic.heatmap"} <= set(TEMPLATES)
    for item in list_templates():
        assert item["purpose"] and item["arguments"]
    with pytest.raises(KeyError, match="version"):
        get_template("generic.scatter@2")
    with pytest.raises(KeyError):
        get_template("unknown.figure")
    assert "def scatter" in show_template_source("generic.scatter")
    assert "def _xy" in show_template_source("generic.scatter")


def test_custom_figures_inherit_embedded_house_style():
    previous = pio.templates.default
    try:
        initialize_house_style()
        custom = go.Figure(go.Scatter(x=[1], y=[2]))
        assert custom.layout.template.layout.font.family == "Manrope, Arial, sans-serif"
    finally:
        pio.templates.default = previous


def test_house_style_values():
    from lobster.figure_templates import _style

    assert _style.HOUSE_COLORS == [
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
    stops = [i / 6 for i in range(7)]
    tail = ["#FFE6E1", "#F7A696", "#EE6853", "#CC2C18", "#8C1E10", "#431009"]
    assert _style.SEQUENTIAL == [list(s) for s in zip(stops, ["#FFFFFF", *tail])]
    assert _style.SEQUENTIAL_EXPRESSION == [
        list(s) for s in zip(stops, ["#E0E0E0", *tail])
    ]
    diverging = ["#005732", "#0BAA6B", "#D9F7E5", "#FFFFFF"]
    diverging += ["#FDE3EC", "#C91A6D", "#8E0049"]
    assert _style.DIVERGING == [list(s) for s in zip(stops, diverging)]
    assert _style.DIVERGING[3] == [0.5, "#FFFFFF"]
    roles = {"INK": "#121212", "TICK": "#636363", "GRID": "#EBEBEB"}
    roles |= {"AXIS": "#E0E0E0", "NO_CALL": "#E0E0E0", "UP": "#C91A6D"}
    roles |= {"DOWN": "#0BAA6B", "KEY": "#CC2C18"}
    assert {name: getattr(_style, name) for name in roles} == roles

    template = _style._house_template()
    layout = template.layout
    assert layout.font.family == "Manrope, Arial, sans-serif"
    assert layout.font.size == 14 and layout.font.color == _style.INK
    assert layout.paper_bgcolor == layout.plot_bgcolor == "#FFFFFF"
    assert list(layout.colorway) == _style.HOUSE_COLORS
    for axis in (layout.xaxis, layout.yaxis):
        assert axis.gridcolor == _style.GRID
        assert axis.linecolor == axis.zerolinecolor == axis.tickcolor == _style.AXIS
        assert axis.tickfont.color == _style.TICK
    assert layout.legend.bgcolor == "rgba(0,0,0,0)"
    assert layout.coloraxis.colorbar.outlinewidth == 0
    # Markers on a coloraxis start grey; heatmap-type defaults stay white-start.
    assert [list(stop) for stop in layout.coloraxis.colorscale] == (
        _style.SEQUENTIAL_EXPRESSION
    )
    assert layout.shapedefaults.line.color == _style.INK
    assert layout.shapedefaults.line.dash == "dash"
    scales = layout.colorscale

    def as_lists(scale):
        return [list(stop) for stop in scale]

    assert as_lists(scales.sequential) == _style.SEQUENTIAL_EXPRESSION
    assert as_lists(scales.sequentialminus) == _style.SEQUENTIAL_EXPRESSION
    assert as_lists(scales.diverging) == _style.DIVERGING
    for kind in ("heatmap", "contour", "histogram2d", "histogram2dcontour"):
        (trace,) = getattr(template.data, kind)
        assert as_lists(trace.colorscale) == _style.SEQUENTIAL
        assert trace.colorbar.outlinewidth == 0
    for kind in ("scatter", "scattergl"):
        for trace in getattr(template.data, kind):
            assert trace.marker.colorbar.outlinewidth == 0


def test_value_scale_decides_by_the_sign_of_finite_values():
    from lobster.figure_templates._style import (
        DIVERGING,
        SEQUENTIAL,
        SEQUENTIAL_EXPRESSION,
        value_scale,
    )

    assert value_scale([[-1.0, 3.0], [np.nan, 2.0]]) == (DIVERGING, 3.0)
    assert value_scale([-np.inf, 0.0, 2.0]) == (SEQUENTIAL, None)
    assert value_scale([0, 0, 0]) == (SEQUENTIAL, None)  # all-zero: unsigned
    assert value_scale([], unsigned=SEQUENTIAL_EXPRESSION) == (
        SEQUENTIAL_EXPRESSION,
        None,
    )


def test_scatter_cycle_never_separates_red_and_green_by_colour_alone():
    from lobster.figure_templates._style import (
        GL_LINE_DASHES,
        HOUSE_COLORS,
        _house_template,
    )

    template = _house_template()
    for kind in ("scatter", "scattergl"):
        cycle = getattr(template.data, kind)
        assert len(cycle) == len(template.layout.colorway) == len(HOUSE_COLORS)
        symbols = [trace.marker.symbol for trace in cycle]
        dashes = [trace.line.dash for trace in cycle]
        assert len(set(symbols)) == len(symbols)
        red, green = HOUSE_COLORS.index("#CC2C18"), HOUSE_COLORS.index("#0BAA6B")
        assert (red, green) == (1, 2)
        assert symbols[red] != symbols[green]
        assert dashes[red] != dashes[green]
    scatter_dashes = [trace.line.dash for trace in template.data.scatter]
    assert len(set(scatter_dashes)) == len(HOUSE_COLORS)
    # Scattergl accepts only the named dashes: they repeat after 6.
    assert {trace.line.dash for trace in template.data.scattergl} == set(GL_LINE_DASHES)

    # Plotly gives go trace n colorway n and template entry n together, so the
    # embedded template carries both cycles in step (no kaleido needed).
    figure = go.Figure(
        [go.Scatter(x=[0, 1], y=[0, n], mode="lines+markers") for n in range(3)],
        layout={"template": _house_template()},
    )
    embedded = figure.layout.template
    assert list(embedded.layout.colorway[:3]) == HOUSE_COLORS[:3]
    assert [t.marker.symbol for t in embedded.data.scatter[:3]] == [
        "circle",
        "square",
        "diamond",
    ]
    assert all(t.marker.symbol is None for t in figure.data)


def test_recipe_size_and_non_json_refuse_before_drawing():
    calls = []

    @figure_template("test.chart")
    def draw(data, *, title=None):
        calls.append(True)
        return go.Figure()

    with pytest.raises(ValueError, match="16 KiB"):
        draw([], title="x" * 17000)
    with pytest.raises(ValueError, match="JSON"):
        draw([], title=object())
    assert not calls


def test_volcano_explicit_comparison_boundaries_units_and_ids(cells):
    data, _ = _fixture("proteomics.volcano", cells)
    data = pd.concat(
        [
            data,
            pd.DataFrame(
                {
                    "protein": ["up"],
                    "comparison": ["other"],
                    "log2_fold_change": [9],
                    "p_adjusted": [0.0001],
                }
            ),
        ],
        ignore_index=True,
    )
    data.loc[0, "log2_fold_change"] = 1.0
    data.loc[0, "p_adjusted"] = 0.05
    before = data.copy(deep=True)
    draw = get_template("proteomics.volcano")
    strict = draw(data, comparison="A_vs_B", log2fc_min=1.0)
    inclusive = draw(data, comparison="A_vs_B", log2fc_min=1.0, boundary="inclusive")
    by_class = lambda fig: {
        trace.uid: [row[0] for row in trace.customdata] for trace in fig.data
    }
    assert by_class(strict)["class:up"] == []
    assert by_class(inclusive)["class:up"] == ["up"]
    assert by_class(strict)["class:down"] == ["down"]
    assert set(sum(by_class(strict).values(), [])) == {"up", "down", "ns"}
    for trace in strict.data:
        assert trace.meta["lobster_ids"] == {
            "entity_kind": "protein",
            "comparison": "A_vs_B",
            "customdata_index": 0,
        }
    with pytest.raises(ValueError, match="comparison"):
        draw(data, comparison="absent")
    with pytest.raises(ValueError, match="unique"):
        draw(pd.concat([data, data.iloc[[0]]]), comparison="A_vs_B")
    pd.testing.assert_frame_equal(data, before)


def test_cell_ids_and_aggregate_non_binding(cells):
    umap = get_template("sc.umap")(cells)
    observed = [row[0] for trace in umap.data for row in trace.customdata]
    assert set(observed) == set(cells.obs_names)
    assert all(isinstance(value, str) for value in observed)
    assert {trace.uid for trace in umap.data} == {"class:a", "class:b"}
    for name in ("sc.dot", "sc.heatmap", "sc.violin", "sc.cluster_composition"):
        data, args = _fixture(name, cells)
        figure = get_template(name)(data, **args)
        assert all(not (trace.meta or {}).get("lobster_ids") for trace in figure.data)
    del cells.obs["n_counts"]
    before = cells.obs.copy(deep=True)
    with pytest.raises(ValueError, match="assess_data_quality"):
        get_template("sc.qc")(cells)
    pd.testing.assert_frame_equal(cells.obs, before)


def test_dot_descriptive_transform_and_explicit_scaling(cells):
    figure = get_template("sc.dot")(cells, genes=["gene-1"], scale="none")
    np.testing.assert_allclose(figure.data[0].customdata[:, 0], [1, 2])
    np.testing.assert_allclose(figure.data[0].customdata[:, 1], [50, 50])
    scaled = get_template("sc.dot")(cells, genes=["gene-1"], scale="zscore")
    np.testing.assert_allclose(scaled.data[0].customdata[:, 0], [-1, 1])
    assert scaled.layout.meta["lobster_template"]["args"]["scale"] == "zscore"


def test_namespace_extension_and_duplicate_rejection(tmp_path, monkeypatch):
    import lobster.figure_template_providers as namespace

    provider = tmp_path / "extension_fixture.py"
    provider.write_text(
        "from lobster.figure_templates import figure_template\n"
        'TEMPLATE_MANIFEST = [{"name":"extension.example","version":"1","purpose":"Example","callable":"draw"}]\n'
        '@figure_template("extension.example")\n'
        'def draw(data, *, title="Extension"):\n'
        "    import plotly.graph_objects as go\n"
        '    return go.Figure(go.Scatter(x=data, uid="example:series"))\n'
    )
    monkeypatch.setattr(namespace, "__path__", [*namespace.__path__, str(tmp_path)])
    try:
        assert "extension.example" in {item["name"] for item in list_templates()}
        assert (
            get_template("extension.example")([1, 2]).layout.meta["lobster_template"][
                "name"
            ]
            == "extension.example"
        )
        (tmp_path / "duplicate_fixture.py").write_text(
            provider.read_text().replace("extension.example", "generic.scatter")
        )
        with pytest.raises(ValueError, match="Duplicate"):
            list_templates()
    finally:
        sys.modules.pop("lobster.figure_template_providers.extension_fixture", None)
        sys.modules.pop("lobster.figure_template_providers.duplicate_fixture", None)


def test_generic_explicit_ids_palette_and_aggregate_non_binding(cells):
    from lobster.figure_templates._style import HOUSE_COLORS

    table, _ = _fixture("generic.scatter", cells)
    figure = get_template("generic.scatter")(
        table, x="x", y="y", color="group", id_col="id"
    )
    assert {row[0] for trace in figure.data for row in trace.customdata} == set(
        table["id"]
    )
    assert all(trace.marker.color in HOUSE_COLORS for trace in figure.data)
    with pytest.raises(ValueError, match="unique"):
        get_template("generic.scatter")(
            pd.concat([table, table.iloc[[0]]]), x="x", y="y", id_col="id"
        )
    for name in ("generic.histogram", "generic.box", "generic.correlation"):
        data, args = _fixture(name, cells)
        assert all(
            not (trace.meta or {}).get("lobster_ids")
            for trace in get_template(name)(data, **args).data
        )


@pytest.mark.parametrize(
    "args",
    [
        {"fdr": 0},
        {"fdr": 2},
        {"log2fc_min": -1},
        {"boundary": "guess"},
        {"p_floor": 0},
        {"label_top": -1},
    ],
)
def test_volcano_rejects_invalid_thresholds(cells, args):
    data, _ = _fixture("proteomics.volcano", cells)
    with pytest.raises(ValueError):
        get_template("proteomics.volcano")(data, comparison="A_vs_B", **args)


def test_invalid_sc_inputs_do_not_trigger_computation(cells):
    before = cells.copy()
    del cells.obsm["X_umap"]
    with pytest.raises(ValueError, match="analysis tool"):
        get_template("sc.umap")(cells)
    assert "X_umap" not in cells.obsm
    with pytest.raises(ValueError, match="Unknown genes"):
        get_template("sc.dot")(cells, genes=["invented-gene"])
    with pytest.raises(ValueError, match="scale"):
        get_template("sc.dot")(cells, genes=["gene-1"], scale="fit")
    del cells.uns["pca"]
    with pytest.raises(ValueError, match="analysis tool"):
        get_template("sc.elbow")(cells)
    np.testing.assert_array_equal(cells.X, before.X)
    pd.testing.assert_frame_equal(cells.obs, before.obs)


def test_numeric_color_layer_and_composition_modes(cells):
    cells.obs["score"] = [0.0, 0.2, 0.4, 1.0]
    colored = get_template("sc.umap")(cells, color="score")
    assert colored.data[0].uid == "value:score"
    np.testing.assert_allclose(colored.data[0].marker.color, cells.obs["score"])
    cells.layers["other"] = cells.X * 2
    feature = get_template("sc.feature")(cells, genes="gene-1", layer="other")
    np.testing.assert_array_equal(feature.data[0].marker.color, cells.X[:, 0] * 2)
    assert len(get_template("sc.elbow")(cells, n_pcs=2).data[0].x) == 2
    for mode, total in (("fraction", 1.0), ("percent", 100.0)):
        figure = get_template("sc.cluster_composition")(cells, normalization=mode)
        np.testing.assert_allclose(
            sum(np.asarray(trace.y) for trace in figure.data), [total, total]
        )
    cells.obs.loc[cells.obs.index[0], "score"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        get_template("sc.umap")(cells, color="score")


@pytest.mark.skipif(
    find_spec("lobster.figure_template_providers.chem") is None,
    reason="Optional chemistry template provider is not installed",
)
@pytest.mark.parametrize(
    "matrix,labels",
    [
        ([[1, 0]], ["a", "b"]),
        ([[1, 0], [0, 1]], ["a", "a"]),
        ([[1, 2], [0, 1]], ["a", "b"]),
        ([[1, float("nan")], [0, 1]], ["a", "b"]),
    ],
)
def test_invalid_similarity_is_not_sampled_or_guessed(matrix, labels):
    with pytest.raises(ValueError):
        get_template("chem.similarity_heatmap")(matrix, compound_ids=labels)


def test_volcano_labels_and_invalid_source_rows(cells):
    data, _ = _fixture("proteomics.volcano", cells)
    figure = get_template("proteomics.volcano")(data, comparison="A_vs_B", label_top=1)
    assert figure.layout.annotations[0].text == "down"
    with pytest.raises(ValueError, match="fields"):
        get_template("proteomics.volcano")(
            data.drop(columns=["p_adjusted"]), comparison="A_vs_B"
        )
    data.loc[0, "p_adjusted"] = -0.5
    with pytest.raises(ValueError, match="p-values"):
        get_template("proteomics.volcano")(data, comparison="A_vs_B")


# --- Palette: hue band, colorbar outlines, volcano roles, signed matrices ---

_COLOR_LISTS = {"colorway", "colors"}
_SCALE_KEYS = {"colorscale", "sequential", "sequentialminus", "diverging"}


@pytest.fixture
def house_session():
    """What a DataManagerV2 session does; other tests keep their default."""
    from lobster.figure_templates._style import HOUSE_STYLE_NAME

    previous_default = pio.templates.default
    previous = (
        pio.templates[HOUSE_STYLE_NAME] if HOUSE_STYLE_NAME in pio.templates else None
    )
    try:
        pio.templates.default = "plotly"  # a session replaces only the stock default
        initialize_house_style()
        yield
    finally:
        del pio.templates[HOUSE_STYLE_NAME]
        if previous is not None:
            pio.templates[HOUSE_STYLE_NAME] = previous
        pio.templates.default = previous_default


def _colors(node, key=None):
    """Yield every colour value in figure JSON, template included."""
    if isinstance(node, dict):
        for child_key, value in node.items():
            if child_key != "meta":  # recipe args name columns, not colours
                yield from _colors(value, child_key)
    elif isinstance(node, (list, tuple)):
        if key in _SCALE_KEYS:
            for stop in node:
                yield stop[1]
        elif key in _COLOR_LISTS or (key or "").endswith("color"):
            yield from (value for value in node if isinstance(value, str))
        else:
            for value in node:
                yield from _colors(value, key)
    elif isinstance(node, str):
        if key in _SCALE_KEYS:
            yield f"named scale {node}"
        elif (key or "").endswith("color"):
            yield node


def _in_band(color):
    """Blue to violet (hue 165-300 degrees) with saturation of 0.15 or more."""
    import colorsys
    import re

    from matplotlib.colors import to_rgb

    if color.startswith("named scale"):
        return True
    match = re.fullmatch(r"rgba?\(([^)]*)\)", color.replace(" ", ""))
    rgb = (
        [float(part) / 255 for part in match.group(1).split(",")[:3]]
        if match
        else to_rgb(color)
    )
    hue, saturation, _ = colorsys.rgb_to_hsv(*rgb)
    return 165 <= hue * 360 <= 300 and saturation >= 0.15


def _outline_widths(figure):
    """Effective outline width of every colorbar the figure can show."""
    template = figure.layout.template
    for trace in figure.data:
        owner = None
        if trace.type in {"heatmap", "histogram2d", "contour"}:
            if trace.showscale is not False and not getattr(trace, "coloraxis", None):
                owner = trace
        elif getattr(trace, "marker", None) is not None:
            if getattr(trace.marker, "showscale", None):
                owner = trace.marker
        if owner is None:
            continue
        width = owner.colorbar.outlinewidth
        if width is None and trace.type in template.data:
            defaults = getattr(template.data, trace.type)
            width = defaults[0].colorbar.outlinewidth if defaults else None
        yield trace.type, width
    for key in figure.layout:
        if key.startswith("coloraxis"):
            width = figure.layout[key].colorbar.outlinewidth
            if width is None:
                width = template.layout.coloraxis.colorbar.outlinewidth
            yield key, width


def _plot_tool_figures():
    from lobster.tools.plot_tools import _create_figure

    table = pd.DataFrame(
        {
            "x": ["a", "a", "b", "b"],
            "y": ["r1", "r2", "r1", "r2"],
            "v": [-2.0, 1.0, 0.5, 3.0],
            "w": [1.0, 2.0, 2.5, 4.0],
            "g": ["p", "q", "r", "p"],
        }
    )
    return {
        "plot_tools.scatter": _create_figure(
            "scatter", table, "w", "v", "", "g", "", "Scatter"
        ),
        "plot_tools.pivot": _create_figure(
            "heatmap", table, "x", "y", "v", "", "", "Pivot"
        ),
        "plot_tools.correlation": _create_figure(
            "heatmap", table, "", "", "", "", "", "Correlation"
        ),
    }


def test_checker_catches_blues_in_any_notation():
    for color in ("#1f77b4", "steelblue", "rgb(65, 105, 225)", "named scale Viridis"):
        assert _in_band(color), color
    for color in ("#404040", "#CC2C18", "#0BAA6B", "#E0E0E0", "rgba(0,0,0,0)"):
        assert not _in_band(color), color


def test_providers_and_plot_tools_use_no_blue_and_no_colorbar_outline(
    cells, house_session
):
    figures = {}
    for name in TEMPLATES:
        data, args = _fixture(name, cells)
        figures[name] = get_template(name)(data, **args)
    zscored = get_template("sc.dot")(cells, genes=["gene-1", "gene-2"], scale="zscore")
    figures["sc.dot zscore"] = zscored
    figures["generic.heatmap signed"] = get_template("generic.heatmap")([[-1, 2]])
    figures.update(_plot_tool_figures())
    # One figure from each provider and from plot_tools at least.
    assert {name.split(".")[0] for name in figures} >= {"generic", "plot_tools"}

    blue = {
        name: [c for c in _colors(fig.to_plotly_json()) if _in_band(c)]
        for name, fig in figures.items()
    }
    assert {name: found for name, found in blue.items() if found} == {}
    outlines = {name: list(_outline_widths(fig)) for name, fig in figures.items()}
    assert any(outlines.values())
    assert {
        name: found
        for name, found in outlines.items()
        if any(width != 0 for _, width in found)
    } == {}


@pytest.mark.skipif(
    find_spec("lobster.figure_template_providers.proteomics") is None,
    reason="proteomics templates not installed",
)
def test_volcano_uses_palette_roles_and_template_thresholds(cells):
    from lobster.figure_templates._style import DOWN, KEY, NO_CALL, UP

    data, args = _fixture("proteomics.volcano", cells)
    figure = get_template("proteomics.volcano")(data, label_top=1, **args)
    colors = {trace.name: trace.marker.color for trace in figure.data}
    assert colors == {"ns": NO_CALL, "down": DOWN, "up": UP}
    assert {trace.marker.symbol for trace in figure.data} == {"circle"}
    assert len(figure.layout.shapes) == 3
    for shape in figure.layout.shapes:
        assert shape.line.color is None and shape.line.dash is None
    template_line = figure.layout.template.layout.shapedefaults.line
    assert (template_line.color, template_line.dash) == ("#121212", "dash")
    assert [item.arrowcolor for item in figure.layout.annotations] == [KEY]


def test_signed_matrices_diverge_around_zero_over_a_symmetric_range(cells):
    from lobster.figure_templates._style import DIVERGING, SEQUENTIAL

    def as_lists(scale):
        return [list(stop) for stop in scale]

    signed = get_template("generic.heatmap")([[-1.0, 3.0], [0.5, 2.0]]).data[0]
    assert as_lists(signed.colorscale) == DIVERGING
    assert (signed.zmid, signed.zmin, signed.zmax) == (0, -3.0, 3.0)
    unsigned = get_template("generic.heatmap")([[1, 2], [3, 4]]).data[0]
    assert as_lists(unsigned.colorscale) == SEQUENTIAL and unsigned.zmid is None

    data, args = _fixture("generic.correlation", cells)
    corr = get_template("generic.correlation")(data, **args).data[0]
    assert as_lists(corr.colorscale) == DIVERGING
    assert (corr.zmid, corr.zmin, corr.zmax) == (0, -1, 1)

    tools = _plot_tool_figures()
    pivot = tools["plot_tools.pivot"].data[0]
    assert as_lists(pivot.colorscale) == DIVERGING
    assert (pivot.zmid, pivot.zmin, pivot.zmax) == (0, -3.0, 3.0)
    coloraxis = tools["plot_tools.correlation"].layout.coloraxis
    assert as_lists(coloraxis.colorscale) == DIVERGING and coloraxis.cmid == 0

    if find_spec("lobster.figure_template_providers.sc") is not None:
        dot = get_template("sc.dot")(cells, genes=["gene-1", "gene-2"], scale="zscore")
        marker = dot.data[0].marker
        assert as_lists(marker.colorscale) == DIVERGING
        assert marker.cmid == 0 and marker.cmin == -marker.cmax


@pytest.mark.skipif(
    find_spec("lobster.figure_template_providers.sc") is None,
    reason="single-cell templates not installed",
)
def test_sc_embedding_labels_each_category_at_its_median(cells):
    three = cells.copy()
    three.obs["leiden"] = ["a", "b", "c", "c"]
    figure = get_template("sc.umap")(three)
    labels = {item.text: (item.x, item.y) for item in figure.layout.annotations}
    assert labels == {"a": (0.0, 1.0), "b": (1.0, 2.0), "c": (2.5, 3.5)}
    assert not get_template("sc.umap")(cells).layout.annotations  # two categories
    many = ad.AnnData(np.ones((31, 1)))
    many.obs["leiden"] = [f"c{i}" for i in range(31)]
    many.obsm["X_umap"] = np.column_stack([np.arange(31), np.arange(31)])
    assert not get_template("sc.umap")(many).layout.annotations  # above 30


def test_px_figures_cycle_symbols_and_dashes_with_the_colours(house_session):
    from lobster.tools.plot_tools import _create_figure

    table = pd.DataFrame(
        {
            "x": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "y": [1, 2, 3, 2, 3, 4, 3, 4, 5],
            "group": ["a"] * 3 + ["b"] * 3 + ["c"] * 3,
        }
    )
    args = {"x": "x", "y": "y", "color": "group"}
    figures = {
        ("generic.scatter", "symbol"): get_template("generic.scatter")(table, **args),
        ("generic.line", "dash"): get_template("generic.line")(table, **args),
        ("plot_tools.scatter", "symbol"): _create_figure(
            "scatter", table, "x", "y", "", "group", "", "S"
        ),
        ("plot_tools.line", "dash"): _create_figure(
            "line", table, "x", "y", "", "group", "", "L"
        ),
    }
    if find_spec("lobster.services.visualization.visualization_service") is not None:
        from lobster.services.visualization.visualization_service import (
            SingleCellVisualizationService,
        )

        data = ad.AnnData(np.ones((9, 1)), obs=pd.DataFrame(index=list("abcdefghi")))
        data.obs["group"] = pd.Categorical(table["group"].to_numpy())
        data.obsm["X_umap"] = table[["x", "y"]].to_numpy(dtype=float)
        figures[("visualization.umap", "symbol")] = (
            SingleCellVisualizationService().create_umap_plot(data, color_by="group")
        )
    for (name, mark), figure in figures.items():
        values = [
            trace.marker.symbol if mark == "symbol" else trace.line.dash
            for trace in figure.data
        ]
        assert values[1] != values[2], (name, values)
        assert [trace.name for trace in figure.data] == ["a", "b", "c"], name
        assert figure.layout.legend.title.text == "group", name
    # Continuous colour gets no symbol mapping; booleans are categories.
    numeric = table.assign(group=[1.0, 2, 3, 4, 5, 6, 7, 8, 9])
    figure = get_template("generic.scatter")(numeric, **args)
    assert len(figure.data) == 1 and figure.data[0].marker.symbol == "circle"
    flags = table.assign(group=[True] * 3 + [False] * 6)
    for figure in (
        get_template("generic.scatter")(flags, **args),
        _create_figure("scatter", flags, "x", "y", "", "group", "", "S"),
    ):
        symbols = [trace.marker.symbol for trace in figure.data]
        assert len(symbols) == 2 and symbols[0] != symbols[1]


def _marker_scales(figure):
    """(trace, first colour) of every colour scale a marker of the figure uses."""
    from lobster.figure_templates._style import SEQUENTIAL

    for trace in figure.data:
        marker = getattr(trace, "marker", None)
        if marker is None or not isinstance(marker.color, (tuple, list, np.ndarray)):
            continue
        if not all(isinstance(value, numbers.Number) for value in marker.color):
            continue  # per-point colour names, not a scale
        scale = marker.colorscale
        if marker.coloraxis:
            scale = figure.layout[marker.coloraxis].colorscale
        scale = scale or SEQUENTIAL  # the template's sequential default
        yield trace.type, scale[0][1]


def test_hand_written_figures_start_points_grey_and_heatmaps_white(house_session):
    import plotly.express as px

    scatter = px.scatter(x=[1, 2, 3], y=[3, 1, 2], color=[0.0, 1.0, 2.0])
    assert scatter.layout.coloraxis.colorscale[0][1] == "#E0E0E0"
    image = px.imshow([[1, 2], [3, 4]])
    assert image.layout.coloraxis.colorscale[0][1] == "#E0E0E0"
    z = [[1, 2], [3, 4]]
    for trace in (
        go.Heatmap(z=z),
        go.Contour(z=z),
        go.Histogram2d(x=[1, 2], y=[1, 2]),
        go.Histogram2dContour(x=[1, 2], y=[1, 2]),
    ):
        figure = go.Figure(trace)
        assert figure.data[0].colorscale is None  # the template data default
        (default,) = getattr(figure.layout.template.data, trace.type)
        assert default.colorscale[0][1] == "#FFFFFF", trace.type


def test_marker_colour_scales_never_start_at_the_background(cells, house_session):
    import plotly.express as px

    from lobster.tools.plot_tools import _create_figure

    cells.obs["score"] = [0.0, 0.2, 0.4, 1.0]
    table = pd.DataFrame({"x": [1, 2, 3], "y": [3, 1, 2], "v": [0.0, 1.0, 2.0]})
    figures = {
        "generic.scatter": get_template("generic.scatter")(
            table, x="x", y="y", color="v"
        ),
        "generic.bar": get_template("generic.bar")(table, x="x", y="y", color="v"),
        "plot_tools.scatter": _create_figure(
            "scatter", table, "x", "y", "", "v", "", "S"
        ),
        "plot_tools.bar": _create_figure("bar", table, "x", "y", "", "v", "", "B"),
        "hand-written px": px.scatter(table, x="x", y="y", color="v"),
    }
    for name in TEMPLATES:
        if name.startswith(("sc.", "proteomics.")):
            data, args = _fixture(name, cells)
            figures[name] = get_template(name)(data, **args)
    if "sc.umap" in TEMPLATES:
        figures["sc.umap numeric"] = get_template("sc.umap")(cells, color="score")
    scales = {name: list(_marker_scales(fig)) for name, fig in figures.items()}
    assert len([name for name, found in scales.items() if found]) >= 4
    background = figures["generic.scatter"].layout.template.layout.plot_bgcolor
    assert {
        name: found
        for name, found in scales.items()
        if any(first.upper() == background.upper() for _, first in found)
    } == {}


@pytest.mark.skipif(
    find_spec("lobster.figure_template_providers.sc") is None,
    reason="single-cell templates not installed",
)
def test_single_series_qc_panels_do_not_cycle_symbols(cells):
    figure = get_template("sc.qc")(cells)
    assert {t.marker.symbol for t in figure.data if t.type == "scattergl"} == {"circle"}
