"""The eight visualization_expert figures use only the Omics-OS palette.

Every colour in each figure's JSON (template included) must fall outside the
blue-to-violet hue band 165-300 degrees or be nearly grey (saturation < 0.15),
and every colorbar has no outline.
"""

import colorsys
import re

import anndata as ad
import numpy as np
import pandas as pd
import plotly.io as pio
import pytest
from matplotlib.colors import to_rgb

from lobster.figure_templates import initialize_house_style
from lobster.figure_templates._style import HOUSE_STYLE_NAME
from lobster.services.visualization.visualization_service import (
    SingleCellVisualizationService,
)

COLOR_LISTS = {"colorway", "colors"}
SCALE_KEYS = {"colorscale", "sequential", "sequentialminus", "diverging"}
RGB = re.compile(r"rgba?\(([^)]*)\)")


@pytest.fixture(autouse=True)
def house_style_session():
    """What a DataManagerV2 session does; other tests keep their default."""
    previous_default = pio.templates.default
    previous = (
        pio.templates[HOUSE_STYLE_NAME] if HOUSE_STYLE_NAME in pio.templates else None
    )
    try:
        pio.templates.default = "plotly"  # a session replaces only the stock default
        initialize_house_style()
        yield
    finally:
        if HOUSE_STYLE_NAME in pio.templates:
            del pio.templates[HOUSE_STYLE_NAME]
        if previous is not None:
            pio.templates[HOUSE_STYLE_NAME] = previous
        pio.templates.default = previous_default


@pytest.fixture
def cells():
    rng = np.random.default_rng(0)
    n_cells = 60
    genes = ["MT-CO1", "RPS3", "HBB", "gene-a", "gene-b"]
    data = ad.AnnData(
        rng.poisson(3, size=(n_cells, len(genes))).astype(float),
        obs=pd.DataFrame(index=[f"cell-{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=genes),
    )
    data.obs["leiden"] = pd.Categorical(["0", "1", "2"] * (n_cells // 3))
    data.obs["sample"] = ["s1", "s2"] * (n_cells // 2)
    offsets = data.obs["leiden"].astype(int).to_numpy()[:, None] * 5
    data.obsm["X_umap"] = rng.normal(size=(n_cells, 2)) + offsets
    data.obsm["X_pca"] = rng.normal(size=(n_cells, 10))
    data.uns["pca"] = {"variance_ratio": np.linspace(0.3, 0.01, 10)}
    counts = data.X.sum(axis=1)
    detected = (data.X > 0).sum(axis=1)
    data.obs["n_counts"] = counts
    data.obs["n_genes"] = detected
    data.obs["gene_detection_rate"] = detected / len(genes) * 100
    data.obs["log10_counts"] = np.log10(counts + 1)
    data.obs["log10_genes"] = np.log10(detected + 1)
    for column, gene in (
        ("percent_mito", "MT-CO1"),
        ("percent_ribo", "RPS3"),
        ("percent_hb", "HBB"),
    ):
        data.obs[column] = data[:, gene].X.ravel() / np.maximum(counts, 1) * 100
    data.obs["doublet_score"] = rng.uniform(size=n_cells)
    return data


def _figures(cells):
    """Each producer the visualization_expert tools call, on every colour branch."""
    service = SingleCellVisualizationService()
    plain = cells.copy()
    del plain.obs["sample"], plain.obs["doublet_score"]
    genes = ["gene-a", "gene-b"]
    return {
        "umap_categories": service.create_umap_plot(cells, color_by="leiden"),
        "umap_expression": service.create_umap_plot(cells, color_by="gene-a"),
        "umap_continuous_obs": service.create_umap_plot(cells, color_by="n_counts"),
        "qc_batch_doublet": service.create_qc_plots(cells),
        "qc_plain": service.create_qc_plots(plain),
        "violin": service.create_violin_plot(cells, genes, use_raw=False),
        "feature": service.create_feature_plot(cells, genes, use_raw=False),
        "dot_scaled": service.create_dot_plot(cells, genes, use_raw=False),
        "dot_unscaled": service.create_dot_plot(
            cells, genes, use_raw=False, standard_scale=None
        ),
        "heatmap_scaled": service.create_heatmap(cells, genes, use_raw=False),
        "heatmap_unscaled": service.create_heatmap(
            cells, genes, use_raw=False, standard_scale=False
        ),
        "elbow": service.create_elbow_plot(cells),
        "composition_by_sample": service.create_cluster_composition_plot(cells),
        "composition_sizes": service.create_cluster_composition_plot(plain),
    }


def _hsv(color):
    match = RGB.fullmatch(color.replace(" ", ""))
    if match:
        rgb = [float(part) / 255 for part in match.group(1).split(",")[:3]]
    else:
        rgb = to_rgb(color)
    return colorsys.rgb_to_hsv(*rgb)


def _colors(node, key=None):
    """Yield (path key, colour string) for every colour value in figure JSON."""
    if isinstance(node, dict):
        for child_key, value in node.items():
            yield from _colors(value, child_key)
    elif isinstance(node, (list, tuple)):
        if key in SCALE_KEYS:
            for stop in node:
                yield key, stop[1]
        elif key in COLOR_LISTS or (key or "").endswith("color"):
            for value in node:
                if isinstance(value, str):
                    yield key, value
        else:
            for value in node:
                yield from _colors(value, key)
    elif isinstance(node, str):
        if key in SCALE_KEYS:
            yield key, f"named scale {node}"
        elif (key or "").endswith("color"):
            yield key, node


def _in_band(color):
    hue, saturation, _ = _hsv(color)
    return 165 <= hue * 360 <= 300 and saturation >= 0.15


def test_checker_catches_blues_in_any_notation():
    for color in ("#1f77b4", "steelblue", "rgb(65, 105, 225)", "rgba(0,128,128,0.5)"):
        assert _in_band(color), color
    for color in ("#404040", "#CC2C18", "#0BAA6B", "#E0E0E0", "rgba(0,0,0,0)"):
        assert not _in_band(color), color


def test_producers_use_no_blue_teal_or_violet(cells):
    offenders = {}
    for name, figure in _figures(cells).items():
        payload = figure.to_plotly_json()
        found = [
            (key, color)
            for key, color in _colors(payload)
            if color.startswith("named scale") or _in_band(color)
        ]
        if found:
            offenders[name] = found
        assert payload["layout"]["template"]["layout"]["font"]["family"].startswith(
            "Manrope"
        ), name
    assert offenders == {}


def _outline_widths(figure):
    """Effective outline width of every colorbar the figure can show."""
    template = figure.layout.template
    for trace in figure.data:
        colorbar_owner = None
        if trace.type in {"heatmap", "histogram2d", "contour"}:
            if trace.showscale is not False and not getattr(trace, "coloraxis", None):
                colorbar_owner = trace
        elif getattr(trace, "marker", None) is not None:
            if getattr(trace.marker, "showscale", None):
                colorbar_owner = trace.marker
        if colorbar_owner is None:
            continue
        width = colorbar_owner.colorbar.outlinewidth
        if width is None and trace.type in template.data:
            (default,) = getattr(template.data, trace.type)[:1] or (None,)
            width = default.colorbar.outlinewidth if default else None
        yield trace.type, width
    for key in figure.layout:
        if key.startswith("coloraxis"):
            width = figure.layout[key].colorbar.outlinewidth
            if width is None:
                width = template.layout.coloraxis.colorbar.outlinewidth
            yield key, width


def test_every_colorbar_has_no_outline(cells):
    widths = {
        name: list(_outline_widths(figure)) for name, figure in _figures(cells).items()
    }
    assert any(widths.values())
    assert {
        name: found
        for name, found in widths.items()
        if any(width != 0 for _, width in found)
    } == {}


def test_embedding_labels_each_category_at_its_median(cells):
    figure = SingleCellVisualizationService().create_umap_plot(cells, color_by="leiden")
    labels = {item.text: (item.x, item.y) for item in figure.layout.annotations}
    assert set(labels) == {"0", "1", "2"}
    for category, (x, y) in labels.items():
        points = cells.obsm["X_umap"][(cells.obs["leiden"] == category).to_numpy()]
        assert (x, y) == pytest.approx(tuple(np.median(points, axis=0)))

    two = cells[cells.obs["leiden"] != "2"].copy()
    figure = SingleCellVisualizationService().create_umap_plot(two, color_by="leiden")
    assert not figure.layout.annotations


def test_categories_follow_house_order_and_expression_starts_grey(cells):
    from lobster.figure_templates._style import HOUSE_COLORS

    figures = _figures(cells)
    umap = figures["umap_categories"]
    assert [trace.marker.color for trace in umap.data] == HOUSE_COLORS[:3]
    feature = figures["feature"].data[0].marker.colorscale
    assert feature[0][1] == "#E0E0E0"
    expression = figures["umap_expression"].layout.coloraxis.colorscale
    assert expression[0][1] == "#E0E0E0"

    heatmap = figures["heatmap_scaled"].data[0]
    assert heatmap.zmid == 0 and heatmap.zmin == -heatmap.zmax
    assert heatmap.colorscale[3][1] == "#FFFFFF"
    dot = figures["dot_scaled"].data[0].marker
    assert dot.cmid == 0 and dot.cmin == -dot.cmax

    many = cells.copy()
    many.obs["leiden"] = pd.Categorical([str(i % 15) for i in range(many.n_obs)])
    many.obs["sample"] = [f"s{i % 2}" for i in range(many.n_obs)]
    bars = SingleCellVisualizationService().create_cluster_composition_plot(many)
    colors = [trace.marker.color for trace in bars.data]
    assert colors[:14] == HOUSE_COLORS and colors[14] == HOUSE_COLORS[0]
    assert all(trace.texttemplate for trace in bars.data)


def test_marker_scales_start_grey_or_diverge(cells):
    """No marker colour scale starts at white: its lowest points would vanish."""
    figures = _figures(cells)
    # n_counts has under 50 values here, so it draws as categories
    figures["umap_score"] = SingleCellVisualizationService().create_umap_plot(
        cells, color_by="doublet_score"
    )
    starts = {}
    for name, figure in figures.items():
        for trace in figure.data:
            marker = getattr(trace, "marker", None)
            scale = getattr(marker, "colorscale", None)
            if getattr(marker, "coloraxis", None):
                scale = figure.layout[marker.coloraxis].colorscale
            if scale is None:
                continue
            starts.setdefault(name, set()).add(scale[0][1])
    assert {"qc_batch_doublet", "umap_score", "feature"} <= set(starts)
    assert {name: s for name, s in starts.items() if "#FFFFFF" in s} == {}

    scaled = cells.copy()
    scaled.X = scaled.X - scaled.X.mean(axis=0)  # scaled expression is signed
    service = SingleCellVisualizationService()
    marker = service.create_feature_plot(scaled, ["gene-a"], use_raw=False).data[0]
    assert marker.marker.colorscale[3][1] == "#FFFFFF"  # DIVERGING, white at 0
    assert marker.marker.cmin == -marker.marker.cmax


def test_single_series_panels_do_not_cycle_symbols_or_dashes(cells):
    figure = _figures(cells)["qc_batch_doublet"]
    gl = [trace for trace in figure.data if trace.type == "scattergl"]
    assert len(gl) >= 4
    assert {(trace.marker.symbol, trace.line.dash) for trace in gl} == {
        ("circle", "solid")
    }
    (kept,) = [shape for shape in figure.layout.shapes if shape.type == "rect"]
    assert kept.line.dash == "solid"


def test_centroid_labels_stop_above_30_categories(cells):
    many = cells.copy()
    many.obs["leiden"] = pd.Categorical([str(i % 31) for i in range(many.n_obs)])
    figure = SingleCellVisualizationService().create_umap_plot(many, color_by="leiden")
    assert len(figure.data) == 31 and not figure.layout.annotations


def test_violin_groups_keep_their_colour_in_every_gene_panel(cells):
    from lobster.figure_templates._style import HOUSE_COLORS

    figure = SingleCellVisualizationService().create_violin_plot(
        cells, ["gene-a", "gene-b"], use_raw=False
    )
    colors = {}
    for trace in figure.data:
        colors.setdefault(trace.name, set()).add(trace.marker.color)
    assert colors == {f"leiden {i}": {HOUSE_COLORS[i]} for i in range(3)}


def test_centroid_labels_skip_categories_without_finite_points(cells):
    missing = cells.copy()
    missing.obsm["X_umap"][(missing.obs["leiden"] == "2").to_numpy()] = np.nan
    figure = SingleCellVisualizationService().create_umap_plot(
        missing, color_by="leiden"
    )
    assert {item.text for item in figure.layout.annotations} == {"0", "1"}


def test_signed_feature_plot_keeps_white_at_zero_with_a_user_range(cells):
    scaled = cells.copy()
    scaled.X = scaled.X - scaled.X.mean(axis=0)
    figure = SingleCellVisualizationService().create_feature_plot(
        scaled, ["gene-a"], use_raw=False, vmin=-1.0, vmax=3.0
    )
    marker = figure.data[0].marker
    assert (marker.cmin, marker.cmax) == (-3.0, 3.0)
