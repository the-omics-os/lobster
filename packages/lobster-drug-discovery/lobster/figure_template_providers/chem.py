"""Draw existing chemical-analysis results; never compute fingerprints here."""

from lobster.figure_templates import figure_template

TEMPLATE_MANIFEST = [
    {
        "name": "chem.similarity_heatmap",
        "version": "1",
        "purpose": "Heatmap of a precomputed Tanimoto similarity matrix.",
        "callable": "similarity_heatmap",
    }
]


@figure_template("chem.similarity_heatmap")
def similarity_heatmap(data, *, compound_ids, fingerprint="morgan", title=None):
    import numpy as np
    import plotly.graph_objects as go

    matrix = np.asarray(data, dtype=float)
    labels = [str(label) for label in compound_ids]
    if matrix.shape != (len(labels), len(labels)) or not labels:
        raise ValueError("Similarity matrix must be square and match compound_ids")
    if len(set(labels)) != len(labels):
        raise ValueError("compound_ids must be unique")
    if not np.isfinite(matrix).all() or ((matrix < 0) | (matrix > 1)).any():
        raise ValueError("Tanimoto similarities must be finite and in [0,1]")
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            zmin=0,
            zmax=1,
            uid="matrix:similarity",  # colour scale: the template's sequential
            hovertemplate="%{x} / %{y}<br>Similarity=%{z:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title or f"Tanimoto similarity ({fingerprint})",
        xaxis_title="Compound",
        yaxis_title="Compound",
    )
    return figure  # Pair marks have no single-compound selection binding.
