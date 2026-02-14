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

## How to Invoke Lobster

All interaction happens via `lobster query`. Describe what you want in natural language.
Use `--session-id` to maintain context across multiple queries (loaded data persists).

```bash
# Start a session with a workspace
lobster query --workspace ./my_analysis --session-id "proj1" "Download GSE109564"

# Continue in the same session (data and context carry over)
lobster query --session-id "proj1" "Run quality control"
lobster query --session-id "proj1" "Cluster the cells and find marker genes"

# Use 'latest' to continue the most recent session
lobster query --session-id latest "Compare hepatocytes vs stellate cells"
```

### Key Flags

| Flag | Purpose |
|------|---------|
| `--session-id <id>` | Session continuity (required for multi-step analysis) |
| `--session-id latest` | Continue the most recent session |
| `--workspace <path>` | Set workspace directory (default: `.lobster_workspace/`) |
| `--json` | Machine-readable JSON output on stdout |
| `--reasoning` | Enable detailed agent reasoning |
| `--output <file>` | Save response to file |

### JSON Output (for parsing results)

```bash
lobster query --session-id latest --json "What data is loaded?" | jq .response
lobster query --session-id latest --json "List workspace files" | jq .response
```

### System Commands (no session needed)

```bash
lobster status              # Check config, installed agents, tier
lobster agents list         # List installed agent packages
lobster config-test --json  # Verify configuration
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

```bash
lobster query --workspace ./liver_study --session-id "liver" \
  "Search PubMed for liver fibrosis scRNA-seq datasets"
  # -> Research Agent searches, finds GSE IDs, queues download

lobster query --session-id "liver" "Download the top dataset"
  # -> Data Expert executes queued download, loads data

lobster query --session-id "liver" "Run QC, filter, normalize, and cluster"
  # -> Transcriptomics Expert runs full pipeline

lobster query --session-id "liver" "Find biomarkers for fibrotic vs healthy cells"
  # -> ML Expert -> Feature Selection Expert

lobster query --session-id "liver" "Create UMAP and export marker genes to CSV"
  # -> Visualization Expert + file export
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

**Checking outputs**:
```bash
lobster query --session-id latest "What data is currently loaded?"
lobster query --session-id latest "List all files in the workspace"
lobster query --session-id latest "Show me the generated plots"
ls .lobster_workspace/        # Direct filesystem inspection
```

## Typical Workflows

All queries below assume an active session (`--session-id`). Shown as just the
query string for readability.

### Single-Cell RNA-seq
```bash
lobster query -w ./scrna --session-id "sc" "Download GSE109564 from GEO"
lobster query --session-id "sc" "Run quality control"
lobster query --session-id "sc" "Filter, normalize, and cluster"
lobster query --session-id "sc" "Identify cell types"
lobster query --session-id "sc" "Find DE genes between T cells and macrophages"
lobster query --session-id "sc" "Create UMAP colored by cell type"
lobster query --session-id "sc" "Export marker genes to CSV"
```
Details: [references/single-cell-workflow.md](references/single-cell-workflow.md)

### Bulk RNA-seq
```bash
lobster query -w ./rnaseq --session-id "bulk" "Load counts.csv with metadata from metadata.csv"
lobster query --session-id "bulk" "Run differential expression: treatment vs control"
lobster query --session-id "bulk" "Show volcano plot and top DE genes"
lobster query --session-id "bulk" "Run GO enrichment on upregulated genes"
```
Details: [references/bulk-rnaseq-workflow.md](references/bulk-rnaseq-workflow.md)

### Genomics [alpha]
```bash
lobster query -w ./gwas --session-id "gen" "Load the VCF file and assess quality"
lobster query --session-id "gen" "Filter samples, then filter variants"
lobster query --session-id "gen" "Run GWAS with phenotype column 'disease'"
lobster query --session-id "gen" "Annotate significant variants"
```
Details: [docs.omics-os.com/docs/agents/genomics](https://docs.omics-os.com/raw/docs/agents/genomics.md)

### Proteomics [alpha]
```bash
lobster query -w ./prot --session-id "prot" "Load the MaxQuant proteinGroups.txt"
lobster query --session-id "prot" "Run quality control"
lobster query --session-id "prot" "Filter and normalize"
lobster query --session-id "prot" "Find differentially abundant proteins: treatment vs control"
```
Details: [docs.omics-os.com/docs/agents/proteomics](https://docs.omics-os.com/raw/docs/agents/proteomics.md)

### Machine Learning [alpha]
```bash
lobster query --session-id latest "Prepare the scRNA-seq data for ML"
lobster query --session-id latest "Find the top 100 biomarkers with stability selection"
lobster query --session-id latest "Build a Cox survival model"
lobster query --session-id latest "Export features for PyTorch"
```
Details: [docs.omics-os.com/docs/agents/ml](https://docs.omics-os.com/raw/docs/agents/ml.md)

## Troubleshooting Quick Reference

| Issue | Check |
|-------|-------|
| Lobster not responding | `lobster config-test --json` |
| No data loaded | `lobster query --session-id latest "What data is loaded?"` |
| Analysis fails | Add `--reasoning` flag to the query |
| Missing outputs | `ls .lobster_workspace/` or ask "List workspace files" |
| Agent not available | `lobster agents list` |

## Documentation

Online docs: **docs.omics-os.com**

Key sections:
- Getting Started -> Installation & Configuration
- Guides -> CLI Commands, Data Formats
- Tutorials -> Single-Cell, Bulk RNA-seq, Proteomics
- Agents -> Per-agent documentation (all 14 agents)
