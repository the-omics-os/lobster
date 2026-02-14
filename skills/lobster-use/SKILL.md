---
name: lobster-use
description: |
  Runs bioinformatics analysis with Lobster AI -- single-cell RNA-seq, bulk RNA-seq,
  genomics (VCF/GWAS), proteomics (mass spec), machine learning (feature selection,
  survival analysis), literature search, dataset discovery, and visualization.
  Use when working with biological data, omics analysis, or bioinformatics tasks.
  Covers: H5AD, CSV, VCF, PLINK, 10X formats, GEO/SRA accessions.

  TRIGGER PHRASES: "analyze cells", "search PubMed", "download GEO", "run QC",
  "cluster", "find markers", "differential expression", "UMAP", "volcano plot",
  "single-cell", "RNA-seq", "VCF", "GWAS", "proteomics", "mass spec",
  "feature selection", "survival analysis", "biomarker", "bioinformatics"

  ASSUMES: Lobster is installed and configured. For setup issues, tell user to
  run `lobster config-test` and fix any errors before proceeding.
---

# Lobster AI Usage Guide

Lobster AI is a multi-agent bioinformatics platform. Users describe analyses in natural
language -- Lobster routes to specialist agents automatically.

## Installation

If Lobster is not installed, guide the user to the right command for their platform:

### macOS / Linux
```bash
curl -fsSL https://install.lobsterbio.com | bash
```

### Windows (PowerShell)
```powershell
irm https://install.lobsterbio.com/windows | iex
```

### Manual install (any platform)
```bash
uv tool install 'lobster-ai[full,anthropic]' && lobster init
# or: pip install 'lobster-ai[full]' && lobster init
```

After install, `lobster init` configures API keys and selects agent packages.

### Upgrading
- uv tool: `uv tool upgrade lobster-ai`
- pip: `pip install --upgrade lobster-ai`

### Adding Agents (uv tool installs)
Users with uv tool installs add agents via:
`uv tool install lobster-ai --with lobster-transcriptomics --with lobster-proteomics`
Running `lobster init` will guide this process and generate the command.

## Quick Reference

| Task | Reference |
|------|-----------|
| **All agents & hierarchy** | [references/agents.md](references/agents.md) |
| **CLI commands** | [references/cli-commands.md](references/cli-commands.md) |
| **Single-cell analysis** | [references/single-cell-workflow.md](references/single-cell-workflow.md) |
| **Bulk RNA-seq analysis** | [references/bulk-rnaseq-workflow.md](references/bulk-rnaseq-workflow.md) |
| **Literature & datasets** | [references/research-workflow.md](references/research-workflow.md) |
| **Visualization** | [references/visualization.md](references/visualization.md) |
| **Genomics (VCF/GWAS)** | [docs.omics-os.com/docs/agents/genomics](https://docs.omics-os.com/raw/docs/agents/genomics.md) |
| **Proteomics (MS/affinity)** | [docs.omics-os.com/docs/agents/proteomics](https://docs.omics-os.com/raw/docs/agents/proteomics.md) |
| **Machine learning** | [docs.omics-os.com/docs/agents/ml](https://docs.omics-os.com/raw/docs/agents/ml.md) |
| **Getting started** | [docs.omics-os.com/docs/getting-started](https://docs.omics-os.com/raw/docs/getting-started.md) |

## Interaction Modes

### Interactive Chat
```bash
lobster chat                          # Start interactive session
lobster chat --workspace ./myproject  # Custom workspace
lobster chat --reasoning              # Enable detailed reasoning
```

### Single Query
```bash
lobster query "Your request"
lobster query --session-id latest "Follow-up request"
```

## Core Patterns

### Natural Language (Primary)
Just describe what you want:
```
"Download GSE109564 and run quality control"
"Cluster the cells and find marker genes"
"Compare hepatocytes vs stellate cells"
```

### Slash Commands (System Operations)
```
/data                    # Show loaded data info
/files                   # List workspace files
/workspace list          # List available datasets
/workspace load 1        # Load dataset by index
/plots                   # Show generated visualizations
/save                    # Save current session
/status                  # Show system status
/help                    # All commands
```

### Session Continuity
```bash
# Start named session
lobster query --session-id "my_analysis" "Load GSE109564"

# Continue with context
lobster query --session-id latest "Now cluster the cells"
lobster query --session-id latest "Find markers for cluster 3"
```

## Agent System

Lobster routes to specialist agents automatically. 14 agents across 8 packages:

| Agent | Package | Handles |
|-------|---------|---------|
| **Supervisor** | lobster-ai | Routes queries, coordinates agents |
| **Research Agent** | lobster-research | PubMed search, GEO/SRA discovery, paper extraction |
| **Data Expert** | lobster-research | File loading, downloads, format conversion |
| **Transcriptomics Expert** | lobster-transcriptomics | scRNA-seq: QC, clustering, markers, trajectory |
| **Annotation Expert** | lobster-transcriptomics | Cell type annotation, gene set enrichment |
| **DE Analysis Expert** | lobster-transcriptomics | Differential expression, statistical testing |
| **Visualization Expert** | lobster-visualization | UMAP, heatmaps, volcano plots, dot plots |
| **Metadata Assistant** | lobster-metadata | ID mapping, metadata standardization |
| **Proteomics Expert** | lobster-proteomics | Mass spec & affinity platform analysis [alpha] |
| **Genomics Expert** | lobster-genomics | VCF, PLINK, GWAS, variant annotation [alpha] |
| **ML Expert** | lobster-ml | ML prep, scVI embeddings, data export [alpha] |
| **Feature Selection Expert** | lobster-ml | Stability, LASSO, variance filtering [alpha] |
| **Survival Analysis Expert** | lobster-ml | Cox models, Kaplan-Meier, risk stratification [alpha] |
| **Protein Structure Viz** | lobster-structural-viz | PDB fetch, PyMOL visualization, RMSD |

Details and hierarchy: [references/agents.md](references/agents.md)

## How Multi-Agent Coordination Works

You describe what you want; Lobster handles the routing. A typical multi-step analysis
uses several agents in sequence:

```
"Search PubMed for liver fibrosis scRNA-seq datasets"
  -> Research Agent searches, finds GSE IDs, queues download

"Download the top dataset"
  -> Data Expert executes queued download, loads data

"Run QC, filter, normalize, and cluster"
  -> Transcriptomics Expert runs full pipeline

"Find biomarkers for fibrotic vs healthy cells"
  -> ML Expert -> Feature Selection Expert

"Create UMAP and export results"
  -> Visualization Expert + file export
```

**Key constraint:** Research Agent is the only agent with internet access.
All other agents operate on data already loaded in memory.

## Workspace & Outputs

**Default workspace**: `.lobster_workspace/`

**Output files**:
| Extension | Content |
|-----------|---------|
| `.h5ad` | Processed AnnData objects |
| `.html` | Interactive visualizations |
| `.png` | Publication-ready plots |
| `.csv` | Exported tables |
| `.json` | Metadata, provenance |

**Managing outputs**:
```
/files              # List all outputs
/plots              # View visualizations
/open results.html  # Open in browser
/read summary.csv   # Preview file contents
```

## Typical Workflows

### Single-Cell RNA-seq
```
"Download GSE109564 from GEO"           # Research + Data Expert
"Run quality control"                    # Transcriptomics Expert
"Filter, normalize, and cluster"         # Transcriptomics Expert
"Identify cell types"                    # Annotation Expert
"Find DE genes between T cells and macrophages"  # DE Analysis Expert
"Create UMAP colored by cell type"       # Visualization Expert
```
Details: [references/single-cell-workflow.md](references/single-cell-workflow.md)

### Bulk RNA-seq
```
"Load counts.csv with metadata from metadata.csv"
"Run differential expression: treatment vs control"
"Show volcano plot and top DE genes"
"Run GO enrichment on upregulated genes"
```
Details: [references/bulk-rnaseq-workflow.md](references/bulk-rnaseq-workflow.md)

### Genomics [alpha]
```
"Load the VCF file and assess quality"
"Filter samples, then filter variants"       # Order enforced
"Run GWAS with phenotype column 'disease'"
"Annotate significant variants"
```
Details: [docs.omics-os.com/docs/agents/genomics](https://docs.omics-os.com/raw/docs/agents/genomics.md)

### Proteomics [alpha]
```
"Load the MaxQuant proteinGroups.txt"        # Auto-detects MS platform
"Run quality control"
"Filter and normalize"
"Find differentially abundant proteins: treatment vs control"
```
Details: [docs.omics-os.com/docs/agents/proteomics](https://docs.omics-os.com/raw/docs/agents/proteomics.md)

### Machine Learning [alpha]
```
"Prepare the scRNA-seq data for ML"
"Find the top 100 biomarkers with stability selection"
"Build a Cox survival model"
"Export features for PyTorch"
```
Details: [docs.omics-os.com/docs/agents/ml](https://docs.omics-os.com/raw/docs/agents/ml.md)

## Troubleshooting Quick Reference

| Issue | Check |
|-------|-------|
| Lobster not responding | `lobster config-test` |
| No data loaded | `/data` to verify, `/workspace list` to see available |
| Analysis fails | Try with `--reasoning` flag |
| Missing outputs | Check `/files` and workspace directory |
| Agent not available | `lobster agents list` to check installed packages |

## Documentation

Online docs: **docs.omics-os.com**

Key sections:
- Getting Started -> Installation & Configuration
- Guides -> CLI Commands, Data Formats
- Tutorials -> Single-Cell, Bulk RNA-seq, Proteomics
- Agents -> Per-agent documentation (all 14 agents)
