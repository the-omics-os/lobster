# Available Agents

## Contents
- [Agent hierarchy](#agent-hierarchy)
- [Core agents](#core-agents)
- [Domain agents](#domain-agents)
- [Sub-agents](#sub-agents)
- [Internal agents](#internal-agents)
- [Multi-agent coordination](#multi-agent-coordination)
- [Checking available agents](#checking-available-agents)

## Agent Hierarchy

```
Supervisor (routes all queries)
├── Research Agent          (online: literature, datasets)
├── Data Expert             (offline: downloads, file loading)
├── Transcriptomics Expert
│   ├── Annotation Expert       (sub-agent)
│   └── DE Analysis Expert      (sub-agent)
├── Visualization Expert
├── Proteomics Expert       [alpha]
├── Genomics Expert         [alpha]
├── ML Expert               [alpha]
│   ├── Feature Selection Expert    (sub-agent)
│   └── Survival Analysis Expert    (sub-agent)
├── Metadata Assistant      (internal)
└── Protein Structure Viz Expert
```

## Core Agents

### Supervisor
**Package:** `lobster-ai` (core)
Routes queries to the right specialist. You never interact with it directly.

---

### Research Agent
**Package:** `lobster-research` | **Access:** Online only (PubMed, GEO, PMC, SRA)

Searches literature, discovers datasets, extracts accession numbers from papers.
The only agent with internet access. Queues downloads for Data Expert.

**Example queries:**
```
"Search PubMed for CRISPR screens in cancer"
"Find GEO datasets for liver single-cell"
"Extract accession numbers from recent fibrosis papers"
```

**Docs:** [docs.omics-os.com/docs/agents/research](https://docs.omics-os.com/raw/docs/agents/research.md)

---

### Data Expert
**Package:** `lobster-research` | **Access:** Offline only (local files, queued downloads)

Executes downloads queued by Research Agent, loads files (H5AD, CSV, 10X, VCF, PLINK),
manages modalities. Has zero internet access -- only processes local data.

**Example queries:**
```
"Load my_data.h5ad"
"Download GSE109564"
"Convert counts.csv to AnnData format"
```

**Docs:** [docs.omics-os.com/docs/agents/research](https://docs.omics-os.com/raw/docs/agents/research.md)

---

### Visualization Expert
**Package:** `lobster-visualization`

Plotly-based visualizations: UMAP, heatmaps, volcano plots, violin plots, dot plots.
Outputs HTML (interactive) and PNG (publication).

**Example queries:**
```
"Create UMAP colored by cell type"
"Generate heatmap of top 50 DE genes"
"Make publication-ready volcano plot"
```

**Docs:** [docs.omics-os.com/docs/agents/visualization](https://docs.omics-os.com/raw/docs/agents/visualization.md)

## Domain Agents

### Transcriptomics Expert
**Package:** `lobster-transcriptomics`

Single-cell RNA-seq: QC, filtering, normalization, clustering, marker genes, trajectory.
Delegates to Annotation Expert and DE Analysis Expert for specialized tasks.

**Example queries:**
```
"Run quality control on the single-cell data"
"Cluster cells and find markers"
"Perform trajectory analysis"
```

**Docs:** [docs.omics-os.com/docs/agents/transcriptomics](https://docs.omics-os.com/raw/docs/agents/transcriptomics.md)

---

### Proteomics Expert [alpha]
**Package:** `lobster-proteomics`

Mass spectrometry (DDA/DIA) and affinity platform (Olink/SomaScan) analysis.
Auto-detects platform type from data characteristics. 10 tools: shared QC/normalization/DE
plus MS-specific peptide mapping and affinity-specific antibody validation and plate correction.

**Example queries:**
```
"Analyze the MaxQuant output"
"Find differentially abundant proteins"
"Run quality control on the Olink data"
```

**Docs:** [docs.omics-os.com/docs/agents/proteomics](https://docs.omics-os.com/raw/docs/agents/proteomics.md)

---

### Genomics Expert [alpha]
**Package:** `lobster-genomics`

VCF (WGS) and PLINK (SNP array) analysis. QC, sample/variant filtering, PCA,
GWAS, variant annotation. 10 tools. Enforces filtering order: samples first, then variants.

**Example queries:**
```
"Load the VCF file and run quality control"
"Run GWAS with the phenotype data"
"Annotate significant variants"
```

**Docs:** [docs.omics-os.com/docs/agents/genomics](https://docs.omics-os.com/raw/docs/agents/genomics.md)

---

### ML Expert [alpha]
**Package:** `lobster-ml`

ML data preparation, scVI embeddings, train/test splits, framework export (NumPy, PyTorch,
TensorFlow, CSV). Routes to Feature Selection Expert and Survival Analysis Expert for
specialized workflows. 7 direct tools.

**Example queries:**
```
"Prepare the data for machine learning"
"Generate scVI embeddings"
"Export features for PyTorch training"
```

**Docs:** [docs.omics-os.com/docs/agents/ml](https://docs.omics-os.com/raw/docs/agents/ml.md)

## Sub-Agents

These are not directly accessible -- their parent agent delegates to them automatically.

### Annotation Expert
**Package:** `lobster-transcriptomics` | **Parent:** Transcriptomics Expert

Cell type annotation using marker genes, reference datasets, or LLM-assisted identification.

**Example queries:**
```
"Identify cell types in each cluster"
"Annotate using known liver cell markers"
"Run GO enrichment on upregulated genes"
```

---

### DE Analysis Expert
**Package:** `lobster-transcriptomics` | **Parent:** Transcriptomics Expert

Differential expression with pyDESeq2. Handles single-cell pseudobulk and bulk RNA-seq.
Supports complex designs (multi-factor, interaction, time series).

**Example queries:**
```
"Run differential expression: treatment vs control"
"Compare cell types for DE genes"
"Find genes with FDR < 0.05 and |log2FC| > 1"
```

---

### Feature Selection Expert [alpha]
**Package:** `lobster-ml` | **Parent:** ML Expert

Biomarker discovery via stability selection, LASSO, and variance filtering.
Includes pathway enrichment on selected features (INDRA API). 6 tools.

**Example queries:**
```
"Find the top biomarkers that distinguish treatment from control"
"Run stability selection with 100 features"
"What pathways are enriched in the selected genes?"
```

---

### Survival Analysis Expert [alpha]
**Package:** `lobster-ml` | **Parent:** ML Expert

Cox proportional hazards, risk threshold optimization, Kaplan-Meier curves.
Requires time-to-event and binary event columns. 6 tools.

**Example queries:**
```
"Build a survival model using the clinical features"
"Optimize the risk threshold for patient stratification"
"Generate Kaplan-Meier curves by treatment arm"
```

## Internal Agents

### Metadata Assistant
**Package:** `lobster-metadata`

Sample ID mapping, metadata standardization, dataset validation, disease enrichment.
Primarily used in automated pipelines (publication queue processing). 9 tools.
Users rarely interact with it directly -- Supervisor routes metadata tasks automatically.

**Docs:** [docs.omics-os.com/docs/agents/metadata](https://docs.omics-os.com/raw/docs/agents/metadata.md)

---

### Protein Structure Visualization Expert
**Package:** `lobster-structural-viz`

Fetches structures from RCSB PDB, creates 3D visualizations with PyMOL, performs
structural analysis (RMSD, secondary structure), links structures to expression data.
Requires local PyMOL installation. 5 tools.

**Example queries:**
```
"Fetch the structure for PDB ID 1AKE"
"Visualize the protein with highlighted active site residues"
"Compare these two structures by RMSD"
```

**Docs:** [docs.omics-os.com/docs/agents/structural-viz](https://docs.omics-os.com/raw/docs/agents/structural-viz.md)

## Multi-Agent Coordination

You describe what you want; Lobster routes automatically. A typical multi-step session
uses several agents in sequence without you needing to specify which:

```
"Search PubMed for liver fibrosis scRNA-seq datasets"
  -> Research Agent (searches, finds GSE IDs, queues download)

"Download the best dataset"
  -> Data Expert (executes queued download, loads H5AD)

"Run QC, filter, normalize, and cluster"
  -> Transcriptomics Expert (full preprocessing pipeline)

"Find biomarkers for fibrotic vs healthy cells"
  -> ML Expert -> Feature Selection Expert (stability selection)
```

**Key constraint:** Research Agent is the only agent with internet access.
Data Expert only processes local files and queued downloads. All other agents
operate on loaded modalities in memory.

## Checking Available Agents

```bash
lobster status              # Check config, installed agents, tier
lobster agents list         # List installed agent packages
lobster config-test --json  # Verify configuration
```
