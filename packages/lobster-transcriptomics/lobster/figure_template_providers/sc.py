"""Single-cell drawing templates owned by transcriptomics, not a plotting agent.

They read already-computed data, never fit embeddings or write AnnData fields.
Means, fractions expressing, scaling and counts are descriptive transforms.
"""

from lobster.figure_templates import figure_template, set_trace_identity
from lobster.figure_templates._style import (
    DIVERGING,
    INK,
    SEQUENTIAL,
    SEQUENTIAL_EXPRESSION,
)

TEMPLATE_MANIFEST = [
    {"name": f"sc.{name}", "version": "1", "purpose": purpose, "callable": name}
    for name, purpose in (
        ("umap", "Draw a stored embedding with stable cell labels."),
        ("violin", "Expression distributions for explicit genes/groups."),
        ("feature", "Gene expression on a stored embedding."),
        ("dot", "Group means and percent-expressing dots; explicit scaling."),
        ("heatmap", "Mean expression by explicit group and genes."),
        ("elbow", "Draw stored PCA explained variance; no PCA fitting."),
        ("cluster_composition", "Descriptive cluster counts or fractions by sample."),
        ("qc", "Draw QC metrics already computed by assess_data_quality."),
    )
]


def _check(data):
    if getattr(data, "isbacked", False):
        raise ValueError(
            "Templates require in-memory data; load it explicitly before drawing"
        )
    if not data.obs_names.is_unique or not data.var_names.is_unique:
        raise ValueError("Cell and gene IDs must be unique, not row positions")
    if data.n_obs == 0:
        raise ValueError("No cells to draw")


def _genes(data, genes, layer=None):
    import numpy as np
    from scipy.sparse import issparse

    _check(data)
    genes = [genes] if isinstance(genes, str) else list(genes)
    if not genes or len(set(genes)) != len(genes):
        raise ValueError("Select at least one unique gene")
    missing = sorted(set(genes).difference(data.var_names))
    if missing:
        raise ValueError("Unknown genes: " + ", ".join(missing))
    subset = data[:, genes]
    values = subset.X if layer is None else subset.layers[layer]
    values = values.toarray() if issparse(values) else np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("Expression values must be finite")
    return genes, values


def _embedding(data, basis):
    import numpy as np

    _check(data)
    if basis not in data.obsm:
        raise ValueError(f"Compute {basis} in an analysis tool before drawing")
    coordinates = np.asarray(data.obsm[basis])
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] < 2
        or not np.isfinite(coordinates[:, :2]).all()
    ):
        raise ValueError("Embedding must contain two finite coordinates per cell")
    return coordinates


def _signed_scale(values, key):
    """DIVERGING around 0 over a symmetric range when values go negative."""
    import numpy as np

    values = np.asarray(values, dtype=float)
    if not values.size or values.min() >= 0:
        return None
    limit = float(np.abs(values).max())
    return {
        "colorscale": DIVERGING,
        f"{key}mid": 0,
        f"{key}min": -limit,
        f"{key}max": limit,
    }


@figure_template("sc.umap")
def umap(data, *, color="leiden", basis="X_umap", title="UMAP", marker_size=5):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    coordinates = _embedding(data, basis)
    labels = data.obs[color]
    figure = go.Figure()
    if pd.api.types.is_numeric_dtype(labels):
        if not np.isfinite(labels.to_numpy(dtype=float)).all():
            raise ValueError("Numeric color values must be finite")
        trace = go.Scattergl(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            mode="markers",
            uid=f"value:{color}",
            marker={
                "size": marker_size,
                "color": labels.to_numpy(),
                "colorscale": SEQUENTIAL,
                "showscale": True,
                "colorbar": {"outlinewidth": 0},
            },
        )
        set_trace_identity(trace, data.obs_names.tolist(), entity_kind="cell")
        figure.add_trace(trace)
    else:
        if labels.isna().any():
            raise ValueError("Group labels must be present")
        for label in sorted(labels.astype(str).unique()):
            mask = (labels.astype(str) == label).to_numpy()
            trace = go.Scattergl(
                x=coordinates[mask, 0],
                y=coordinates[mask, 1],
                name=label,
                mode="markers",
                uid=f"class:{label}",
                marker={"size": marker_size},
            )
            set_trace_identity(trace, data.obs_names[mask].tolist(), entity_kind="cell")
            figure.add_trace(trace)
        if len(figure.data) >= 3:
            # Label each category at its median, so a split cluster's label
            # still sits inside its own points.
            for trace in figure.data:
                figure.add_annotation(
                    x=float(np.median(trace.x)),
                    y=float(np.median(trace.y)),
                    text=trace.name,
                    showarrow=False,
                    font={"color": INK, "size": 12},
                    bgcolor="rgba(255,255,255,0.7)",
                )
    figure.update_layout(
        title=title, xaxis_title=f"{basis} 1", yaxis_title=f"{basis} 2"
    )
    return figure


@figure_template("sc.feature")
def feature(
    data,
    *,
    genes,
    basis="X_umap",
    layer=None,
    title="Feature expression",
    marker_size=5,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    coordinates = _embedding(data, basis)
    gene_names, expression = _genes(data, genes, layer)
    figure = make_subplots(rows=1, cols=len(gene_names), subplot_titles=gene_names)
    for index, gene in enumerate(gene_names):
        trace = go.Scattergl(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            name=gene,
            uid=f"gene:{gene}",
            mode="markers",
            marker={
                "color": expression[:, index],
                "colorscale": SEQUENTIAL_EXPRESSION,
                "size": marker_size,
                # Panels are one gene each, not series: no symbol cycle
                "symbol": "circle",
                "showscale": True,
                "colorbar": {"outlinewidth": 0},
            },
        )
        set_trace_identity(trace, data.obs_names.tolist(), entity_kind="cell")
        figure.add_trace(trace, row=1, col=index + 1)
    figure.update_layout(title=title)
    return figure


@figure_template("sc.violin")
def violin(
    data, *, genes, groupby="leiden", layer=None, points=False, title="Gene expression"
):
    import plotly.graph_objects as go

    gene_names, expression = _genes(data, genes, layer)
    if data.obs[groupby].isna().any():
        raise ValueError("Group labels must be present")
    groups = data.obs[groupby].astype(str)
    figure = go.Figure()
    for index, gene in enumerate(gene_names):
        figure.add_trace(
            go.Violin(
                x=groups.tolist(),
                y=expression[:, index],
                name=gene,
                uid=f"gene:{gene}",
                points=points,
                box_visible=True,
                meanline_visible=True,
            )
        )
    figure.update_layout(
        title=title, xaxis_title=groupby, yaxis_title="Expression", violinmode="group"
    )
    return figure  # Distribution marks have no single-cell binding.


def _group_expression(data, genes, groupby, layer):
    import numpy as np

    gene_names, expression = _genes(data, genes, layer)
    labels = data.obs[groupby]
    if labels.isna().any():
        raise ValueError("Group labels must be present")
    groups = sorted(labels.astype(str).unique())
    masks = [(labels.astype(str) == group).to_numpy() for group in groups]
    means = np.stack([expression[mask].mean(axis=0) for mask in masks])
    fractions = np.stack([(expression[mask] > 0).mean(axis=0) * 100 for mask in masks])
    return gene_names, groups, means, fractions


@figure_template("sc.dot")
def dot(
    data,
    *,
    genes,
    groupby="leiden",
    layer=None,
    scale="none",
    title="Expression dot plot",
):
    import numpy as np
    import plotly.graph_objects as go

    gene_names, groups, means, fractions = _group_expression(
        data, genes, groupby, layer
    )
    if scale == "zscore":
        std = means.std(axis=0)
        means = np.divide(
            means - means.mean(axis=0), std, out=np.zeros_like(means), where=std != 0
        )
    elif scale != "none":
        raise ValueError("scale must be none or zscore (per gene, across group means)")
    figure = go.Figure(
        go.Scatter(
            x=np.tile(gene_names, len(groups)).tolist(),
            y=np.repeat(groups, len(gene_names)).tolist(),
            mode="markers",
            uid="aggregate:expression",
            marker={
                "color": means.ravel(),
                "size": fractions.ravel(),
                "sizemode": "area",
                "sizeref": 0.25,
                "symbol": "circle",
                "showscale": True,
                "colorbar": {"outlinewidth": 0},
                **(_signed_scale(means, "c") or {"colorscale": SEQUENTIAL_EXPRESSION}),
            },
            customdata=np.column_stack([means.ravel(), fractions.ravel()]),
            hovertemplate="%{x} / %{y}<br>mean=%{customdata[0]:.3f}<br>% expressing=%{customdata[1]:.1f}<extra></extra>",
        )
    )
    figure.update_layout(title=title, xaxis_title="Gene", yaxis_title=groupby)
    return figure  # Group means do not claim a single-cell selection.


@figure_template("sc.heatmap")
def heatmap(data, *, genes, groupby="leiden", layer=None, title="Mean gene expression"):
    import plotly.graph_objects as go

    gene_names, groups, means, _ = _group_expression(data, genes, groupby, layer)
    figure = go.Figure(
        go.Heatmap(
            z=means,
            x=gene_names,
            y=groups,
            uid="matrix:group_expression",
            **(_signed_scale(means, "z") or {"colorscale": SEQUENTIAL_EXPRESSION}),
        )
    )
    figure.update_layout(title=title, xaxis_title="Gene", yaxis_title=groupby)
    return figure


@figure_template("sc.elbow")
def elbow(data, *, pca_key="pca", n_pcs=None, title="PCA explained variance"):
    import numpy as np
    import plotly.graph_objects as go

    _check(data)
    if pca_key not in data.uns or "variance_ratio" not in data.uns[pca_key]:
        raise ValueError("Compute PCA in an analysis tool before drawing")
    values = np.asarray(data.uns[pca_key]["variance_ratio"], dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("PCA variance ratios must be a finite vector")
    if n_pcs is not None and (not isinstance(n_pcs, int) or n_pcs < 1):
        raise ValueError("n_pcs must be a positive integer")
    values = values[:n_pcs]
    figure = go.Figure(
        go.Scatter(
            x=list(range(1, len(values) + 1)),
            y=values,
            mode="lines+markers",
            uid="pca:variance",
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="Principal component",
        yaxis_title="Explained variance ratio",
    )
    return figure


@figure_template("sc.cluster_composition")
def cluster_composition(
    data,
    *,
    cluster_key="leiden",
    groupby="sample",
    normalization="count",
    title="Cluster composition",
):
    import pandas as pd
    import plotly.graph_objects as go

    _check(data)
    if data.obs[[cluster_key, groupby]].isna().any().any():
        raise ValueError("Cluster and group labels must be present")
    counts = pd.crosstab(
        data.obs[groupby].astype(str), data.obs[cluster_key].astype(str)
    )
    if normalization in {"fraction", "percent"}:
        counts = counts.div(counts.sum(axis=1), axis=0) * (
            100 if normalization == "percent" else 1
        )
    elif normalization != "count":
        raise ValueError("normalization must be count, fraction or percent")
    figure = go.Figure()
    for cluster in counts.columns:
        figure.add_trace(
            go.Bar(
                x=counts.index.tolist(),
                y=counts[cluster].to_numpy(),
                name=str(cluster),
                uid=f"class:{cluster}",
            )
        )
    figure.update_layout(
        title=title, barmode="stack", xaxis_title=groupby, yaxis_title=normalization
    )
    return figure  # Counts have no single-cell selection binding.


@figure_template("sc.qc")
def qc(data, *, nbins=30, title="Single-cell QC"):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _check(data)
    required = {
        "n_counts",
        "n_genes",
        "gene_detection_rate",
        "log10_counts",
        "log10_genes",
        "percent_mito",
        "percent_ribo",
        "percent_hb",
    }
    missing = sorted(required.difference(data.obs.columns))
    if missing:
        raise ValueError(
            "Run assess_data_quality before drawing QC; missing: " + ", ".join(missing)
        )
    if not isinstance(nbins, int) or nbins < 1:
        raise ValueError("nbins must be a positive integer")
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Counts / genes",
            "Mitochondrial percentage",
            "Count distribution",
            "Ribosomal percentage",
        ],
    )
    for row, col, y, uid in (
        (1, 1, "n_genes", "qc:complexity"),
        (1, 2, "percent_mito", "qc:mitochondrial"),
        (2, 2, "percent_ribo", "qc:ribosomal"),
    ):
        trace = go.Scattergl(
            x=data.obs["n_counts"].to_numpy(),
            y=data.obs[y].to_numpy(),
            mode="markers",
            uid=uid,
            name=y,
        )
        set_trace_identity(trace, data.obs_names.tolist(), entity_kind="cell")
        figure.add_trace(trace, row=row, col=col)
    figure.add_trace(
        go.Histogram(
            x=data.obs["n_counts"].to_numpy(),
            nbinsx=nbins,
            uid="qc:count_distribution",
            name="Counts",
        ),
        row=2,
        col=1,
    )
    figure.update_layout(title=title, showlegend=False)
    return figure
