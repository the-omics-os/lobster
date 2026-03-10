---
name: lobster-use
description: |
  Runs bioinformatics analysis with Lobster AI -- single-cell RNA-seq, bulk RNA-seq,
  genomics (VCF/GWAS), proteomics (mass spec/affinity), metabolomics (LC-MS/GC-MS/NMR),
  machine learning (feature selection, survival analysis), drug discovery,
  literature search, dataset discovery, and visualization.
  Use when working with biological data, omics analysis, or bioinformatics tasks.
  Covers: H5AD, CSV, VCF, PLINK, 10X, mzML formats, GEO/SRA/PRIDE/MetaboLights accessions.

  TRIGGER PHRASES: "analyze cells", "search PubMed", "download GEO", "run QC",
  "cluster", "find markers", "differential expression", "UMAP", "volcano plot",
  "single-cell", "RNA-seq", "VCF", "GWAS", "proteomics", "mass spec",
  "metabolomics", "MetaboLights", "LC-MS", "metabolite",
  "feature selection", "survival analysis", "biomarker", "bioinformatics",
  "drug discovery", "pharmacogenomics", "variant annotation"

  ASSUMES: Lobster is installed and configured. For setup issues, tell user to
  run `lobster config-test` and fix any errors before proceeding.
required_binaries:
  - lobster
  - python3
primary_credential: LLM_PROVIDER_API_KEY
required_env_vars:
  - name: ANTHROPIC_API_KEY
    required: one_of_provider
    description: Anthropic Claude API key
  - name: GOOGLE_API_KEY
    required: one_of_provider
    description: Google Gemini API key
  - name: OPENAI_API_KEY
    required: one_of_provider
    description: OpenAI API key
  - name: OPENROUTER_API_KEY
    required: one_of_provider
    description: OpenRouter API key (600+ models)
  - name: AWS_ACCESS_KEY_ID
    required: one_of_provider
    description: AWS Bedrock access key (must be paired with AWS_SECRET_ACCESS_KEY)
  - name: AWS_SECRET_ACCESS_KEY
    required: one_of_provider
    description: AWS Bedrock secret key (must be paired with AWS_ACCESS_KEY_ID)
  - name: AZURE_AI_ENDPOINT
    required: one_of_provider
    description: Azure AI endpoint URL (must be paired with AZURE_AI_CREDENTIAL)
  - name: AZURE_AI_CREDENTIAL
    required: one_of_provider
    description: Azure AI API credential (must be paired with AZURE_AI_ENDPOINT)
  - name: NCBI_API_KEY
    required: false
    description: NCBI API key for faster PubMed/GEO access (recommended)
credential_note: |
  Exactly ONE LLM provider is required (not all). Choose one provider and set
  only that provider's env var(s). Paired credentials (AWS, Azure) must both be set.
declared_writes:
  - .lobster_workspace/                        # Workspace data, session state, outputs
  - .lobster_workspace/.env                    # Provider credential (workspace-scoped, mode 0600)
  - .lobster_workspace/provider_config.json    # Provider selection config
  - ~/.config/lobster/credentials.env          # ONLY if --global flag is used (not default)
  - ~/.config/lobster/providers.json           # ONLY if --global flag is used (not default)
network_access:
  - docs.omics-os.com                          # On-demand documentation fetches
  - LLM provider API endpoint                  # Whichever single provider is configured
  - eutils.ncbi.nlm.nih.gov                   # PubMed/GEO search (Research Agent only)
  - ftp.ncbi.nlm.nih.gov                      # GEO/SRA dataset downloads (Data Expert only)
  - www.ebi.ac.uk                              # PRIDE/MetaboLights (Research Agent only)
source:
  github: https://github.com/the-omics-os/lobster
  pypi: https://pypi.org/project/lobster-ai/
always: false
---

# Lobster AI Usage Guide

Lobster AI is a multi-agent bioinformatics platform. Users describe analyses in natural
language -- Lobster routes to 22 specialist agents across 10 packages automatically.

## Requirements

- **Binaries**: `lobster` CLI (`pip install lobster-ai`), Python 3.12+
- **Credential**: Exactly ONE LLM provider key as env var (not all — pick one):
  - `ANTHROPIC_API_KEY` | `GOOGLE_API_KEY` | `OPENAI_API_KEY` | `OPENROUTER_API_KEY`
  - `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (Bedrock — both required)
  - `AZURE_AI_ENDPOINT` + `AZURE_AI_CREDENTIAL` (Azure — both required)
  - Ollama: no key needed (local models)
- **Optional**: `NCBI_API_KEY` for faster PubMed/GEO
- **Writes**: `.lobster_workspace/` (data, credentials in `.env` mode 0600, outputs)
- **Global config** (`--global` flag, NOT default): `~/.config/lobster/` — avoid unless needed
- **Network**: LLM provider API + public bio databases (GEO, SRA, PRIDE, MetaboLights)

## Docs Discovery

The docs site at **docs.omics-os.com** exposes LLM-friendly raw markdown:

| Route | Use |
|-------|-----|
| `/llms.txt` | Index of all pages (title + URL + description) |
| `/llms-full.txt` | Full content dump of all free pages |
| `/raw/docs/{slug}.md` | Raw markdown for a specific page |

**Workflow**: Fetch `/llms.txt` first to discover slugs, then fetch individual pages via `/raw/docs/{slug}.md`.

Example: `https://docs.omics-os.com/raw/docs/tutorials/single-cell-rnaseq.md`

## Two Modes

This skill supports coding agents in two modes:

**Orchestrator** -- The agent calls `lobster query --json --session-id` programmatically,
parses structured output, and chains multi-step analyses. See [agent-patterns.md](references/agent-patterns.md).

**Guide** -- The agent teaches a human user what to type in `lobster chat` or `lobster query`.
See the routing table below for which docs page to fetch.

## Quick Start

```bash
# Install (PyPI -- preferred)
pip install 'lobster-ai[full]'
# or: uv tool install 'lobster-ai[full]'

# Configure (uses env var -- never pass raw keys on command line)
lobster init --non-interactive --anthropic-key "$ANTHROPIC_API_KEY" --profile production

# Run analysis (always pass -w and --session-id together)
lobster query -w ./my_analysis --session-id "proj" --json "Download GSE109564 and run QC"

# Inspect workspace (no tokens burned, ~300ms)
lobster command data --json -w ./my_analysis
```

**Source**: [github.com/the-omics-os/lobster](https://github.com/the-omics-os/lobster) |
**PyPI**: [pypi.org/project/lobster-ai](https://pypi.org/project/lobster-ai/)

## Routing Table

| You want to... | Docs slug | Skill reference |
|---|---|---|
| **Install & configure** | `getting-started/installation` | -- |
| **Configuration options** | `getting-started/configuration` | -- |
| **Use the CLI** | `guides/cli-commands` | [cli-reference.md](references/cli-reference.md) |
| **Orchestrate programmatically** | -- | [agent-patterns.md](references/agent-patterns.md) |
| **Analyze scRNA-seq** | `tutorials/single-cell-rnaseq` | -- |
| **Analyze bulk RNA-seq** | `tutorials/bulk-rnaseq` | -- |
| **Analyze proteomics** | `tutorials/proteomics` | -- |
| **Understand data formats** | `guides/data-formats` | -- |
| **Search literature / datasets** | `agents/research` | -- |
| **Analyze genomics** | `agents/genomics` | -- |
| **Analyze metabolomics** | `case-studies/metabolomics` | -- |
| **ML / feature selection** | `agents/ml` | -- |
| **Drug discovery** | `agents/drug-discovery` | -- |
| **Visualize results** | `agents/visualization` | -- |
| **Troubleshoot** | `support/troubleshooting` | -- |
| **See case studies** | `case-studies/{domain}` | -- |
| **All agent capabilities** | `agents` | -- |
| **Extend Lobster (dev)** | -- | Use `lobster-dev` skill |

To fetch a docs page: `https://docs.omics-os.com/raw/docs/{slug}.md`

## Hard Rules

1. **Always use `--session-id`** for multi-step analyses -- loaded data persists across queries
2. **Use `lobster command --json`** for workspace inspection (no tokens burned, ~300ms)
3. **Research Agent is the ONLY agent with internet access** -- all others operate on loaded data
4. **Never skip QC** before analysis -- always assess quality first
5. **Use `--json` flag** when parsing output programmatically
6. **Check data is loaded** before running analysis steps (`lobster command data --json`)
7. **Default workspace**: `.lobster_workspace/` -- override with `-w <path>`
8. **Fetch docs on demand** from `docs.omics-os.com/raw/docs/{slug}.md` -- don't guess workflows

## Agent Overview

22 agents across 10 packages. Supervisor routes automatically based on natural language.

| Agent | Package | Handles |
|---|---|---|
| Supervisor | lobster-ai | Routes queries, coordinates agents |
| Research Agent | lobster-research | PubMed, GEO, SRA, PRIDE, MetaboLights search (online) |
| Data Expert | lobster-research | File loading, downloads, format conversion (offline) |
| Transcriptomics Expert | lobster-transcriptomics | scRNA-seq + bulk RNA-seq: QC, clustering, trajectory |
| Annotation Expert | lobster-transcriptomics | Cell type annotation, gene set enrichment (child) |
| DE Analysis Expert | lobster-transcriptomics | Differential expression, pseudobulk, GSEA (child) |
| Proteomics Expert | lobster-proteomics | MS + affinity import, QC, normalization, batch correction |
| Proteomics DE Expert | lobster-proteomics | Protein DE, pathway enrichment, KSEA, STRING PPI (child) |
| Biomarker Discovery | lobster-proteomics | Panel selection, nested CV, hub proteins (child) |
| Metabolomics Expert | lobster-metabolomics | LC-MS/GC-MS/NMR: QC, normalization, PCA/PLS-DA, annotation |
| Genomics Expert | lobster-genomics | VCF/PLINK: QC, GWAS, variant annotation |
| Variant Analysis Expert | lobster-genomics | VEP annotation, ClinVar, clinical prioritization (child) |
| ML Expert | lobster-ml | ML prep, scVI embeddings, data export |
| Feature Selection Expert | lobster-ml | Stability selection, LASSO, variance filtering (child) |
| Survival Analysis Expert | lobster-ml | Cox models, Kaplan-Meier, risk stratification (child) |
| Drug Discovery Expert | lobster-drug-discovery | Drug target validation, compound profiling |
| Cheminformatics Expert | lobster-drug-discovery | Molecular descriptors, fingerprints, similarity (child) |
| Clinical Dev Expert | lobster-drug-discovery | Trial design, endpoint analysis, safety signals (child) |
| Pharmacogenomics Expert | lobster-drug-discovery | PGx variants, drug-gene interactions (child) |
| Visualization Expert | lobster-visualization | UMAP, heatmaps, volcano plots, dot plots (Plotly) |
| Metadata Assistant | lobster-metadata | ID mapping, metadata standardization (internal) |
| Protein Structure Viz | lobster-structural-viz | PDB fetch, PyMOL visualization, RMSD |

Per-agent docs: `https://docs.omics-os.com/raw/docs/agents/{domain}.md`
