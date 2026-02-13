# Single-Cell RNA-seq Workflow

Complete workflow for analyzing single-cell RNA-seq data with Lobster AI.

## Overview

**Goal**: Identify cell types, find markers, understand cellular heterogeneity
**Agent**: Transcriptomics Expert (automatic routing)
**Time**: 15-30 minutes for typical dataset (10K-50K cells)

## Step 1: Load Data

**From GEO**:
```
"Download GSE109564 from GEO"
```

**From local file**:
```
/workspace load my_data.h5ad
```
Or:
```
"Load the single-cell data from my_data.h5ad"
```

**From 10X Genomics**:
```
/archive filtered_feature_bc_matrix.tar.gz
```

**Verify data loaded**:
```
/data
```

## Step 2: Quality Assessment

```
"Assess quality of the single-cell data and show me the metrics"
```

**What Lobster checks**:
- Mitochondrial gene percentage (cell viability)
- Ribosomal gene percentage (translation activity)
- Total gene counts (library complexity)
- Total UMI counts (sequencing depth)
- Doublet detection (multi-cell artifacts)

**View QC plots**:
```
/plots
```

## Step 3: Filtering & Preprocessing

**Standard filtering**:
```
"Filter low-quality cells and normalize the data"
```

**Custom thresholds**:
```
"Filter cells with less than 200 genes and more than 25% mitochondrial content, then normalize"
```

**What happens**:
1. Cell filtering (removes low-quality)
2. Gene filtering (removes rarely expressed)
3. Normalization (library size + log1p)
4. Highly variable gene selection

## Step 4: Clustering

**Standard clustering**:
```
"Cluster the cells and visualize with UMAP"
```

**Full pipeline**:
```
"Run PCA, compute neighbors, cluster with Leiden, and create UMAP"
```

**Custom resolution**:
```
"Cluster with resolution 0.8"
```

## Step 5: Cell Type Annotation

**Automatic annotation**:
```
"Identify cell types in each cluster"
```

**Tissue-specific**:
```
"Annotate cell types using known liver cell markers"
```

**Find markers first**:
```
"Find top marker genes for each cluster, then identify cell types"
```

## Step 6: Differential Expression

**Between cell types**:
```
"Find differentially expressed genes between T cells and macrophages"
```

**Between conditions**:
```
"Compare treatment vs control in each cell type"
```

**Within cluster**:
```
"Find genes that distinguish cluster 3 from all other clusters"
```

## Step 7: Visualization

**UMAP plots**:
```
"Create UMAP colored by cell type"
"Show UMAP colored by expression of CD4 and CD8A"
```

**Marker visualization**:
```
"Create a dot plot showing top markers for each cell type"
"Generate a heatmap of the top 50 DE genes"
```

**Violin plots**:
```
"Show violin plots of key markers across cell types"
```

## Step 8: Export Results

**Export markers**:
```
"Export marker genes to CSV"
```

**Export as notebook**:
```
"Export the analysis pipeline as a Jupyter notebook"
```

**Save session**:
```
/save
```

## Complete Example Session

```bash
lobster chat --workspace ./tumor_analysis

> "Download GSE109564 - tumor-infiltrating immune cells"
# Downloads and loads data

> "Run quality control"
# Generates QC metrics and plots

> /plots
# View QC visualizations

> "Filter low-quality cells and normalize"
# Preprocessing

> "Cluster cells and create UMAP"
# Dimensionality reduction + clustering

> "Identify cell types using immune cell markers"
# Annotation

> "Find differentially expressed genes between CD8 T cells and Tregs"
# DE analysis

> "Create publication-ready UMAP and export DE results"
# Final outputs

> /files
# Check generated files

> /save
```

## Common Variations

### Trajectory Analysis
```
"Perform trajectory analysis to identify developmental paths"
```

### Pseudobulk
```
"Aggregate cells by type and perform bulk DE analysis"
```

### Batch Correction
```
"Correct for batch effects between samples"
```

### Integration
```
"Integrate these two datasets and find conserved markers"
```

## Tips

1. **Start with QC**: Always assess quality before filtering
2. **Check intermediate results**: Use `/data` and `/plots` frequently
3. **Be specific**: "resolution 0.5" not just "cluster"
4. **Use session continuity**: `--session-id latest` for follow-ups
5. **Save often**: `/save` preserves your progress
