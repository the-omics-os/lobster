---
name: lobster-use
description: |
  Runs bioinformatics analysis with Lobster AI -- single-cell RNA-seq, bulk RNA-seq,
  genomics (VCF/GWAS), proteomics (mass spec/affinity), metabolomics (LC-MS/GC-MS/NMR),
  machine learning (feature selection, survival analysis), drug discovery,
  literature search, dataset discovery, and visualization.
  Use when working with biological data, omics analysis, or bioinformatics tasks.
  Covers: H5AD, CSV, VCF, PLINK, 10X, mzML formats, GEO/SRA/PRIDE/MetaboLights accessions.

  Lobster AI works in two deployment modes:
  - LOCAL: agents run on-device, you provide an LLM API key
  - CLOUD: agents run on Omics-OS Cloud (ECS Fargate), managed Bedrock, per-user billing

  TRIGGER PHRASES: "analyze cells", "search PubMed", "download GEO", "run QC",
  "cluster", "find markers", "differential expression", "UMAP", "volcano plot",
  "single-cell", "RNA-seq", "VCF", "GWAS", "proteomics", "mass spec",
  "metabolomics", "MetaboLights", "LC-MS", "metabolite",
  "feature selection", "survival analysis", "biomarker", "bioinformatics",
  "drug discovery", "pharmacogenomics", "variant annotation",
  "cloud chat", "omics-os cloud", "lobster cloud",
  "cloud login", "cloud session"

  ASSUMES: Lobster is installed and configured. For setup issues, tell user to
  run `lobster config-test` (local) or `lobster cloud status` (cloud) and fix any errors.
required_binaries:
  - lobster
  - python3
primary_credential: LLM_PROVIDER_API_KEY
required_env_vars:
  - name: ANTHROPIC_API_KEY
    required: one_of_provider
    description: Anthropic Claude API key (local mode)
  - name: GOOGLE_API_KEY
    required: one_of_provider
    description: Google Gemini API key (local mode)
  - name: OPENAI_API_KEY
    required: one_of_provider
    description: OpenAI API key (local mode)
  - name: OPENROUTER_API_KEY
    required: one_of_provider
    description: OpenRouter API key (local mode, 600+ models)
  - name: AWS_ACCESS_KEY_ID
    required: one_of_provider
    description: AWS Bedrock access key (local mode, must pair with SECRET)
  - name: AWS_SECRET_ACCESS_KEY
    required: one_of_provider
    description: AWS Bedrock secret key (local mode, must pair with ACCESS_KEY)
  - name: AZURE_AI_ENDPOINT
    required: one_of_provider
    description: Azure AI endpoint URL (local mode, must pair with CREDENTIAL)
  - name: AZURE_AI_CREDENTIAL
    required: one_of_provider
    description: Azure AI API credential (local mode, must pair with ENDPOINT)
  - name: NCBI_API_KEY
    required: false
    description: NCBI API key for faster PubMed/GEO access (recommended)
credential_note: |
  LOCAL MODE: Exactly ONE LLM provider is required. Choose one and set only that
  provider's env var(s). Paired credentials (AWS, Azure) must both be set.
  CLOUD MODE: No LLM keys needed. Run `lobster cloud login` to authenticate via
  browser OAuth or `lobster cloud login --api-key "$OMICS_OS_API_KEY"` for headless environments.
  Credentials stored at ~/.config/omics-os/credentials.json.
declared_writes:
  - .lobster_workspace/                        # Workspace data, session state, outputs
  - .lobster_workspace/.env                    # Provider credential (workspace-scoped, mode 0600)
  - .lobster_workspace/provider_config.json    # Provider selection config
  - ~/.config/lobster/credentials.env          # ONLY if --global flag is used (not default)
  - ~/.config/lobster/providers.json           # ONLY if --global flag is used (not default)
  - ~/.config/omics-os/credentials.json        # Cloud OAuth/API key credentials
network_access:
  - docs.omics-os.com                          # On-demand documentation fetches
  - app.omics-os.com                           # Omics-OS Cloud REST API (cloud mode)
  - stream.omics-os.com                        # Omics-OS Cloud streaming (cloud mode)
  - LLM provider API endpoint                  # Whichever single provider is configured (local mode)
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
- **Local mode** (one of):
  - `ANTHROPIC_API_KEY` | `GOOGLE_API_KEY` | `OPENAI_API_KEY` | `OPENROUTER_API_KEY`
  - `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (Bedrock)
  - `AZURE_AI_ENDPOINT` + `AZURE_AI_CREDENTIAL` (Azure)
  - Ollama: no key needed (local models)
- **Cloud mode**: `lobster cloud login` (browser OAuth or `--api-key "$KEY"`). No LLM keys needed.
  - Env var: `OMICS_OS_API_KEY` (cloud auth alternative to stored credentials)
- **Optional**: `NCBI_API_KEY` for faster PubMed/GEO
- **Writes**: `.lobster_workspace/` (local), `~/.config/omics-os/credentials.json` (cloud)
- **Network**: LLM provider (local) or `app.omics-os.com` + `stream.omics-os.com` (cloud)

## Docs Discovery

The docs site at **docs.omics-os.com** exposes LLM-friendly raw markdown:

| Route | Use |
|-------|-----|
| `/llms.txt` | Index of all pages (title + URL + description) |
| `/llms-full.txt` | Full content dump of all free pages |
| `/raw/docs/{slug}.md` | Raw markdown for a specific page |

**Workflow**: Fetch `/llms.txt` first to discover slugs, then fetch individual pages via `/raw/docs/{slug}.md`.

Example: `https://docs.omics-os.com/raw/docs/tutorials/single-cell-rnaseq.md`

## Three Modes

### Local Mode
Agents run on your machine. You provide LLM API key. Data stays local.
```bash
lobster init                              # Configure LLM provider
lobster chat                              # Interactive (Ink or Go TUI)
lobster query "Analyze my data" --json    # Single-turn
```

### Cloud Mode (Omics-OS Cloud)
Agents run on ECS Fargate. Managed Bedrock. Per-user billing. No LLM keys.
```bash
lobster cloud login                       # Browser OAuth (one-time)
lobster cloud chat                        # Interactive (launches npm TUI)
lobster cloud status                      # Check tier, usage, budget
lobster cloud logout                      # Clear stored credentials
```

### Orchestrator Mode
Coding agents call `lobster query --json` programmatically, parse structured output,
and chain multi-step analyses. Cloud mode uses `lobster cloud chat` (interactive npm TUI).
See [agent-patterns.md](references/agent-patterns.md).

## Quick Start

```bash
# Install
pip install 'lobster-ai[full]'
# or: uv tool install 'lobster-ai[full]'

# === Local mode ===
lobster init --non-interactive --anthropic-key "$ANTHROPIC_API_KEY" --profile production
lobster query -w ./my_analysis --session-id "proj" --json "Download GSE109564 and run QC"

# === Cloud mode ===
lobster cloud login                       # One-time browser OAuth
lobster cloud chat                        # Interactive cloud chat (npm TUI)

# Inspect workspace (no tokens, ~300ms, local mode only)
lobster command data --json -w ./my_analysis
```

**Source**: [github.com/the-omics-os/lobster](https://github.com/the-omics-os/lobster) |
**PyPI**: [pypi.org/project/lobster-ai](https://pypi.org/project/lobster-ai/)

## Routing Table

| You want to... | Docs slug | Skill reference |
|---|---|---|
| **Install & configure** | `getting-started/installation` | -- |
| **Configuration options** | `getting-started/configuration` | -- |
| **Use the CLI (local + cloud)** | `guides/cli-commands` | [cli-reference.md](references/cli-reference.md) |
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

1. **Always use `--session-id`** for multi-step local analyses -- loaded data persists across queries
2. **Use `lobster command --json`** for workspace inspection (no tokens, ~300ms, local mode only)
3. **Research Agent is the ONLY agent with internet access** -- all others operate on loaded data
4. **Never skip QC** before analysis -- always assess quality first
5. **Use `--json` flag** when parsing output programmatically (both local and cloud)
6. **Cloud mode**: run `lobster cloud login` before `lobster cloud chat`
7. **Cloud sessions persist server-side** -- managed within the npm TUI
9. **Default workspace**: `.lobster_workspace/` (local only) -- override with `-w <path>`
10. **Fetch docs on demand** from `docs.omics-os.com/raw/docs/{slug}.md` -- don't guess workflows

## Local vs Cloud Decision

| Factor | Local | Cloud |
|--------|-------|-------|
| LLM keys | You provide | Managed (Bedrock) |
| Agent execution | Your machine | ECS Fargate |
| Data storage | Local `.lobster_workspace/` | Cloud workspace |
| Session persistence | Disk (workspace) | Server-side (UUID) |
| Billing | Your LLM provider | Omics-OS usage-based |
| Offline | Yes | No |
| Multi-device | No | Yes (web + CLI continuity) |
| Setup | `lobster init` | `lobster cloud login` |

**Rule**: If user has Omics-OS Cloud account, prefer cloud mode. Otherwise local.

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
