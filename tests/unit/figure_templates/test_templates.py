"""Every shipped drawing template: fixture, recipe, style, identities and purity."""

import copy
import json
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
    assert layout.shapedefaults.line.color == _style.INK
    assert layout.shapedefaults.line.dash == "dash"
    scales = layout.colorscale

    def as_lists(scale):
        return [list(stop) for stop in scale]

    assert as_lists(scales.sequential) == _style.SEQUENTIAL
    assert as_lists(scales.sequentialminus) == _style.SEQUENTIAL
    assert as_lists(scales.diverging) == _style.DIVERGING
    for kind in ("heatmap", "contour"):
        (trace,) = getattr(template.data, kind)
        assert as_lists(trace.colorscale) == _style.SEQUENTIAL
        assert trace.colorbar.outlinewidth == 0


def test_scatter_cycle_never_separates_red_and_green_by_colour_alone():
    from lobster.figure_templates._style import HOUSE_COLORS, _house_template

    for kind in ("scatter", "scattergl"):
        cycle = getattr(_house_template().data, kind)
        assert len(cycle) == len(HOUSE_COLORS)
        symbols = [trace.marker.symbol for trace in cycle]
        dashes = [trace.line.dash for trace in cycle]
        assert len(set(symbols)) == len(symbols)
        red, green = HOUSE_COLORS.index("#CC2C18"), HOUSE_COLORS.index("#0BAA6B")
        assert symbols[red] != symbols[green]
        assert dashes[red] != dashes[green]

    # The cycle reaches figures: trace n takes colorway n and symbol n together.
    figure = go.Figure(
        [go.Scatter(x=[0, 1], y=[0, n], mode="lines+markers") for n in range(3)],
        layout={"template": _house_template()},
    )
    full = figure.full_figure_for_development(warn=False)
    assert [t.marker.color for t in full.data] == HOUSE_COLORS[:3]
    assert len({t.marker.symbol for t in full.data}) == 3
    assert len({t.line.dash for t in full.data}) == 3


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
