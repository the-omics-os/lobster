# Visualization Guide

Creating publication-quality visualizations with Lobster AI.

## Overview

**Agent**: Visualization Expert (automatic routing)
**Output formats**: HTML (interactive), PNG (publication), PDF (vector)

## Quick Commands

```
/plots                    # List all generated visualizations
/open plot.html           # Open in browser
```

## Dimensionality Reduction Plots

### UMAP
```
"Create UMAP visualization"
"Show UMAP colored by cell type"
"Create UMAP with cluster labels"
```

### t-SNE
```
"Generate t-SNE plot"
```

### PCA
```
"Show PCA plot of the samples"
"Create PCA with variance explained"
```

### Multiple Colorings
```
"Show UMAP colored by cell type, sample, and n_genes"
```

## Expression Plots

### Feature Plots
```
"Show expression of CD4, CD8A, and FOXP3 on UMAP"
```

### Violin Plots
```
"Create violin plots of marker genes across cell types"
```

### Dot Plots
```
"Generate dot plot of top markers for each cluster"
```

### Ridge Plots
```
"Show ridge plots of QC metrics by sample"
```

## Differential Expression Plots

### Volcano Plot
```
"Create volcano plot of DE results"
"Show volcano plot highlighting genes with |log2FC| > 2"
```

### MA Plot
```
"Generate MA plot"
```

### Heatmap
```
"Create heatmap of top 50 DE genes"
"Show heatmap with sample clustering"
```

## Quality Control Plots

### QC Metrics
```
"Show violin plots of QC metrics"
"Create scatter plot of genes vs UMIs"
```

### Sample Correlation
```
"Show sample correlation heatmap"
```

### Batch Effects
```
"Visualize batch effects with PCA before and after correction"
```

## Clustering Visualizations

### Cluster Proportions
```
"Show bar plot of cell type proportions by sample"
```

### Dendrogram
```
"Create hierarchical clustering dendrogram of cell types"
```

### Sankey Diagram
```
"Show Sankey diagram of cluster assignments across resolutions"
```

## Pathway & Enrichment Plots

### Enrichment Bar Plot
```
"Create bar plot of top enriched GO terms"
```

### Enrichment Dot Plot
```
"Show dot plot of KEGG pathway enrichment"
```

### Network Plot
```
"Visualize pathway connections as a network"
```

## Customization

### Colors
```
"Use viridis colormap for the UMAP"
"Color clusters with custom palette: red, blue, green"
```

### Labels
```
"Add title 'Tumor Microenvironment' to the UMAP"
"Include gene names on the heatmap"
```

### Size & Style
```
"Create high-resolution UMAP for publication"
"Generate figure with larger font sizes"
```

### Multiple Panels
```
"Create a 2x2 panel with UMAP, violin, heatmap, and volcano"
```

## Export Options

### Interactive HTML
```
"Save as interactive HTML"
/plots
/open visualization.html
```

### Publication PNG
```
"Export as high-resolution PNG (300 DPI)"
```

### Vector PDF
```
"Save volcano plot as PDF for publication"
```

### Specific Size
```
"Create 8x6 inch figure at 300 DPI"
```

## Complete Visualization Session

```bash
> "Create comprehensive visualization of the single-cell analysis"

# Lobster generates:
# - UMAP with clusters
# - Cell type proportions
# - Top marker dot plot
# - QC violin plots

> /plots
# Lists all generated figures

> "Add UMAP colored by specific markers: CD3D, CD14, CD19"
# Gene expression overlay

> "Export all figures as publication-ready PNGs"
# High-resolution export

> /files
# Check exported files
```

## Tips

1. **Interactive first**: HTML plots are great for exploration
2. **Export for publication**: Switch to PNG/PDF for papers
3. **Consistent styling**: Ask for uniform color schemes
4. **Check file sizes**: Large datasets may need subsampling
5. **Label everything**: Titles, axes, legends for clarity
