"""Pure drawings of retained proteomics results; no statistical testing or I/O."""

from lobster.figure_templates import figure_template, set_trace_identity
from lobster.figure_templates._style import (
    DOWN,
    INK,
    KEY,
    NO_CALL,
    SEQUENTIAL_EXPRESSION,
    UP,
)

TEMPLATE_MANIFEST = [
    {
        "name": "proteomics.volcano",
        "version": "1",
        "purpose": "Classify retained DE rows at explicit log2FC/FDR cutoffs.",
        "callable": "volcano",
    },
    {
        "name": "proteomics.pathway_bubble",
        "version": "1",
        "purpose": "Bubble chart of supplied enrichment results, without rerunning enrichment.",
        "callable": "pathway_bubble",
    },
]


@figure_template("proteomics.volcano")
def volcano(
    data,
    *,
    comparison,
    fdr=0.05,
    log2fc_min=1.0,
    boundary="strict",
    p_field="p_adjusted",
    effect_field="log2_fold_change",
    entity_field="protein",
    comparison_field="comparison",
    p_floor=1e-300,
    label_top=0,
    title=None,
):
    """Draw one comparison. Thresholds are LOG2 units; strict means p<FDR, |FC|>min.

    Changing cutoffs reclassifies plotted rows only, never the stored DE results.
    Invalid values/ambiguous duplicate proteins are refused rather than omitted.
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    if not 0 < fdr <= 1 or log2fc_min < 0 or not np.isfinite(log2fc_min):
        raise ValueError("fdr must be in (0,1] and log2fc_min finite and nonnegative")
    if boundary not in {"strict", "inclusive"}:
        raise ValueError("boundary must be strict or inclusive")
    if not 0 < p_floor <= 1 or not isinstance(label_top, int) or label_top < 0:
        raise ValueError("p_floor must be in (0,1] and label_top a nonnegative integer")
    table = pd.DataFrame(data).copy(deep=True)
    required = {p_field, effect_field, entity_field, comparison_field}
    if not required <= set(table.columns):
        raise ValueError(
            "DE rows require explicit protein, comparison, effect and p fields"
        )
    table = table.loc[table[comparison_field].astype(str) == comparison].copy()
    if table.empty:
        raise ValueError(f"No retained rows for comparison: {comparison}")
    if (
        table[entity_field].isna().any()
        or table[entity_field].astype(str).duplicated().any()
    ):
        raise ValueError("Protein IDs must be unique within this comparison")
    effects = pd.to_numeric(table[effect_field], errors="raise").to_numpy(dtype=float)
    pvalues = pd.to_numeric(table[p_field], errors="raise").to_numpy(dtype=float)
    if (
        not np.isfinite(effects).all()
        or not np.isfinite(pvalues).all()
        or ((pvalues < 0) | (pvalues > 1)).any()
    ):
        raise ValueError(
            "DE effects and p-values must be finite; p-values must be in [0,1]"
        )
    significant = (
        (pvalues < fdr) & (np.abs(effects) > log2fc_min)
        if boundary == "strict"
        else (pvalues <= fdr) & (np.abs(effects) >= log2fc_min)
    )
    classes = np.where(
        significant & (effects > 0),
        "up",
        np.where(significant & (effects < 0), "down", "ns"),
    )
    y = -np.log10(np.maximum(pvalues, p_floor))
    figure = go.Figure()
    for category, color in (("ns", NO_CALL), ("down", DOWN), ("up", UP)):
        mask = classes == category
        trace = go.Scattergl(
            x=effects[mask],
            y=y[mask],
            name=category,
            uid=f"class:{category}",
            mode="markers",
            # Up and down already differ by side of 0: no symbol cycle
            marker={"color": color, "size": 7, "symbol": "circle"},
            hovertemplate="%{customdata[0]}<br>log2FC=%{x:.3f}<br>-log10(p)=%{y:.3f}<extra>%{fullData.name}</extra>",
        )
        set_trace_identity(
            trace,
            table.loc[mask, entity_field].astype(str).tolist(),
            entity_kind="protein",
            comparison=comparison,
        )
        figure.add_trace(trace)
    # Thresholds take the template's dashed Ink
    figure.add_hline(y=-np.log10(fdr))
    for cutoff in (-log2fc_min, log2fc_min):
        figure.add_vline(x=cutoff)
    if label_top:
        eligible = np.flatnonzero(classes != "ns")
        for index in eligible[np.argsort(pvalues[eligible], kind="stable")[:label_top]]:
            figure.add_annotation(
                x=float(effects[index]),
                y=float(y[index]),
                text=str(table.iloc[index][entity_field]),
                showarrow=True,
                arrowcolor=KEY,
                font={"color": INK},
            )
    figure.update_layout(
        title=title or f"Volcano — {comparison}",
        xaxis_title="log2 fold change",
        yaxis_title=f"−log10({p_field})",
    )
    return figure


@figure_template("proteomics.pathway_bubble")
def pathway_bubble(
    data,
    *,
    pathway_field="Term",
    p_field="Adjusted P-value",
    overlap_field="Overlap",
    top_n=20,
    p_floor=1e-300,
    title="Pathway enrichment",
):
    """Draw supplied p-values and overlap counts; top_n selection is explicit."""
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    if not isinstance(top_n, int) or top_n < 1 or not 0 < p_floor <= 1:
        raise ValueError("top_n must be positive and p_floor in (0,1]")
    table = pd.DataFrame(data).copy(deep=True)
    if not {pathway_field, p_field, overlap_field} <= set(table.columns):
        raise ValueError("Supply pathway, p-value and overlap columns explicitly")
    table[p_field] = pd.to_numeric(table[p_field], errors="raise")
    if not np.isfinite(table[p_field]).all() or not table[p_field].between(0, 1).all():
        raise ValueError("Pathway p-values must be finite and in [0,1]")
    table = table.sort_values(p_field, kind="stable").head(top_n)
    overlap = pd.to_numeric(
        table[overlap_field].astype(str).str.split("/").str[0], errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(overlap).all() or (overlap < 0).any():
        raise ValueError("Overlap counts must be finite and nonnegative")
    scores = -np.log10(np.maximum(table[p_field].to_numpy(dtype=float), p_floor))
    figure = go.Figure(
        go.Scatter(
            x=scores,
            y=table[pathway_field].astype(str).tolist(),
            mode="markers",
            uid="pathway:bubbles",
            marker={
                "size": np.maximum(overlap, 1) * 4,
                "sizemode": "area",
                "color": scores,
                "colorscale": SEQUENTIAL_EXPRESSION,  # grey start: visible on white
                "showscale": True,
                "colorbar": {"outlinewidth": 0},
            },
            customdata=overlap.reshape(-1, 1),
            hovertemplate="%{y}<br>-log10(p)=%{x:.3f}<br>overlap=%{customdata[0]}<extra></extra>",
        )
    )
    # Pathway marks aggregate genes: no single-gene selection binding.
    figure.update_layout(
        title=title, xaxis_title=f"−log10({p_field})", yaxis_title="Pathway"
    )
    return figure
