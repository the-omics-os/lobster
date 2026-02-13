# lobster-visualization

Publication-quality plots and interactive visualizations for bioinformatics data analysis.

## Installation

```bash
pip install lobster-visualization
```

## Agents

| Agent | Description |
|-------|-------------|
| `visualization_expert` | General-purpose visualization agent for creating publication-ready figures across all omics data types. |

## Services

| Service | Purpose |
|---------|---------|
| VisualizationService | Core plotting engine with Plotly-based interactive visualizations |

## Features

- UMAP plots with automatic point size scaling based on cell count
- PCA plots with variance explained annotations
- t-SNE visualizations for dimensionality reduction
- Violin plots for gene expression distribution by cluster
- Feature plots showing expression intensity on embeddings
- Dot plots for marker gene comparison across clusters
- Heatmaps with optional row and column standardization
- Elbow plots for PCA variance analysis and component selection
- Cluster composition stacked bar charts for sample contribution
- QC plots with scientific validation (skips inappropriate modalities)
- Interactive Plotly figures with zoom, pan, and hover tooltips
- Static PNG export for publication and presentation use
- Consistent color palettes across related visualizations
- Automatic figure sizing based on data dimensions

## Requirements

- Python 3.12+
- lobster-ai >= 1.0.0
- plotly

## Documentation

Full documentation: [docs.omics-os.com/docs/agents/visualization](https://docs.omics-os.com/docs/agents/visualization)

## License

AGPL-3.0-or-later
