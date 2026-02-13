# Available Agents

Lobster AI uses specialist agents for different analysis domains. The Supervisor
automatically routes your queries to the right agent.

## Free Tier Agents

### Research Agent
**Package**: `lobster-research`

**Capabilities**:
- PubMed literature search
- GEO dataset discovery
- Paper metadata extraction
- Full-text analysis (when available)

**Example queries**:
```
"Search PubMed for CRISPR papers"
"Find GEO datasets for liver single-cell"
"Extract accession numbers from recent papers"
```

---

### Data Expert
**Package**: `lobster-research`

**Capabilities**:
- Load data files (H5AD, CSV, 10X)
- Download from GEO/SRA
- Format conversion
- Data validation

**Example queries**:
```
"Load my_data.h5ad"
"Download GSE109564"
"Convert counts.csv to AnnData format"
```

---

### Transcriptomics Expert
**Package**: `lobster-transcriptomics`

**Capabilities**:
- Single-cell RNA-seq analysis
- Quality control
- Clustering
- Marker gene identification

**Example queries**:
```
"Run quality control on the single-cell data"
"Cluster cells and find markers"
"Perform dimensionality reduction"
```

---

### DE Analysis Expert
**Package**: `lobster-transcriptomics`

**Capabilities**:
- Differential expression analysis
- Statistical testing (DESeq2, limma)
- Multiple comparison handling
- Complex experimental designs

**Example queries**:
```
"Run differential expression: treatment vs control"
"Compare cell types for DE genes"
"Find genes with FDR < 0.05"
```

---

### Annotation Expert
**Package**: `lobster-transcriptomics`

**Capabilities**:
- Cell type annotation
- Gene set enrichment
- Pathway analysis
- Functional annotation

**Example queries**:
```
"Identify cell types in each cluster"
"Run GO enrichment on upregulated genes"
"Annotate using known liver markers"
```

---

### Visualization Expert
**Package**: `lobster-visualization`

**Capabilities**:
- UMAP/t-SNE plots
- Heatmaps
- Volcano plots
- Publication-ready figures

**Example queries**:
```
"Create UMAP colored by cell type"
"Generate heatmap of top DE genes"
"Make publication-ready volcano plot"
```

---

## Premium Tier Agents

### Proteomics Expert
**Package**: `lobster-proteomics`

**Capabilities**:
- Mass spectrometry data analysis
- DDA/DIA workflows
- Protein quantification
- PTM analysis

**Example queries**:
```
"Analyze the MaxQuant output"
"Find differentially abundant proteins"
"Detect phosphorylation sites"
```

---

### Genomics Expert
**Package**: `lobster-genomics`

**Capabilities**:
- VCF/BCF parsing
- Variant annotation
- GWAS analysis
- Population genetics

**Example queries**:
```
"Analyze the VCF file for variants"
"Run GWAS with the phenotype data"
"Annotate variants with functional impact"
```

---

### ML Expert
**Package**: `lobster-ml`

**Capabilities**:
- Embedding generation
- Classification models
- Clustering optimization
- Feature selection

**Example queries**:
```
"Generate scVI embeddings"
"Train a classifier for cell types"
"Optimize clustering parameters"
```

---

## Checking Available Agents

```bash
# Command line
lobster status

# In chat
/status
```

Shows:
- Your subscription tier
- Installed agent packages
- Available agents for your tier
- Premium agents (upgrade required)

## Agent Routing

You don't need to specify which agent to use — the Supervisor routes automatically:

```
"Cluster the cells"           → Transcriptomics Expert
"Search PubMed"               → Research Agent
"Create a heatmap"            → Visualization Expert
"Run differential expression" → DE Analysis Expert
```

For ambiguous queries, Supervisor will ask for clarification.
