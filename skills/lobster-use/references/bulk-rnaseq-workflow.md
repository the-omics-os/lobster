# Bulk RNA-seq Workflow

Complete workflow for differential expression analysis with bulk RNA-seq data.

## Overview

**Goal**: Identify differentially expressed genes between conditions
**Agent**: DE Analysis Expert (automatic routing)
**Time**: 10-20 minutes for typical experiment

## Step 1: Load Data

**From count matrix**:
```
"Load the count matrix from counts.csv"
```

**With metadata**:
```
"Load counts.csv with sample metadata from metadata.csv"
```

**From Kallisto/Salmon**:
```
/archive kallisto_results.tar.gz
```

**Verify**:
```
/data
```

## Step 2: Experimental Design

**Specify conditions**:
```
"The samples are grouped by treatment column: control vs treated"
```

**Complex designs**:
```
"Compare treatment vs control, accounting for batch as a covariate"
```

**Check design**:
```
"Show me the experimental design and sample groups"
```

## Step 3: Quality Assessment

**Sample QC**:
```
"Assess sample quality and show PCA of all samples"
```

**What Lobster checks**:
- Library sizes
- Gene detection rates
- Sample clustering (PCA)
- Outlier detection

## Step 4: Differential Expression

**Simple comparison**:
```
"Run differential expression: treatment vs control"
```

**With thresholds**:
```
"Find genes with FDR < 0.05 and |log2FC| > 1"
```

**Multiple comparisons**:
```
"Compare all treatment groups against control"
```

**Methods available**:
- DESeq2 (default)
- pyDESeq2
- limma-voom

## Step 5: Results Exploration

**Summary**:
```
"How many genes are differentially expressed?"
```

**Top genes**:
```
"Show me the top 20 upregulated and downregulated genes"
```

**Specific gene**:
```
"What's the expression pattern of TP53?"
```

## Step 6: Visualization

**Volcano plot**:
```
"Create a volcano plot of the DE results"
```

**MA plot**:
```
"Generate an MA plot"
```

**Heatmap**:
```
"Create a heatmap of the top 50 DE genes"
```

**Sample clustering**:
```
"Show hierarchical clustering of samples based on DE genes"
```

## Step 7: Pathway Analysis

**Gene set enrichment**:
```
"Perform pathway enrichment on the upregulated genes"
```

**GO analysis**:
```
"Run GO enrichment analysis on the DE genes"
```

**KEGG pathways**:
```
"What KEGG pathways are enriched?"
```

## Step 8: Export

**DE results**:
```
"Export all DE results to CSV"
```

**Significant only**:
```
"Export genes with FDR < 0.05 to Excel"
```

**For publication**:
```
"Export publication-ready volcano plot as PDF"
```

## Complete Example Session

```bash
lobster chat --workspace ./rnaseq_analysis

> "Load counts.csv - it has gene IDs as rows and samples as columns"
# Loads count matrix

> "The metadata.csv file has sample info with a 'condition' column"
# Adds experimental design

> "Run sample QC and show PCA"
# Quality assessment

> /plots
# View PCA and QC plots

> "Perform differential expression: treated vs control"
# DE analysis with DESeq2

> "Show volcano plot and top 20 DE genes"
# Visualize results

> "Run GO enrichment on upregulated genes"
# Pathway analysis

> "Export DE results and volcano plot"
# Save outputs

> /files
# Check generated files
```

## Complex Designs

### Multi-factor
```
"Compare treatment vs control, with patient as a blocking factor"
```

### Interaction
```
"Test for interaction between treatment and genotype"
```

### Time series
```
"Identify genes with significant changes over time in the treated group"
```

### Paired samples
```
"Run paired analysis - samples are matched by patient"
```

## Tips

1. **Check library sizes**: Large differences may need normalization
2. **Examine PCA first**: Identify batch effects or outliers
3. **Use appropriate thresholds**: FDR < 0.05, |log2FC| > 1 is common
4. **Report methods**: Note DESeq2/limma version for reproducibility
5. **Multiple testing**: Always use adjusted p-values (FDR)
