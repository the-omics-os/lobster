# Lobster AI

[![PyPI version](https://img.shields.io/pypi/v/lobster-ai.svg)](https://pypi.org/project/lobster-ai/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Documentation](https://img.shields.io/badge/docs-omics--os.com-green.svg)](https://docs.omics-os.com)

> Open-source multi-agent bioinformatics engine. Describe your analysis in natural language.

[Documentation](https://docs.omics-os.com) · [PyPI](https://pypi.org/project/lobster-ai/) · [Omics-OS Cloud](https://app.omics-os.com)

## Quick Start

```bash
#installs only core lobster
curl -fsSL https://install.lobsterbio.com | bash
```

Then:

```bash
# Define LLM provider and domain expert installation
lobster init
```

Finaly:

```bash
lobster chat 
# or non-interactive
lobster query "Hi, what can you do?"
```

<details>
<summary>Windows, pip, and other install methods</summary>

**Windows** (PowerShell):
```powershell
irm https://install.lobsterbio.com/windows | iex
```

**uv** (recommended manual install):
```bash
uv tool install 'lobster-ai[full,anthropic]'
lobster init
```

**pip**:
```bash
pip install 'lobster-ai[full]'
lobster init
```

**Upgrade**:
```bash
uv tool upgrade lobster-ai    # uv
pip install -U lobster-ai      # pip
```

</details>

## Why Lobster AI

**Your machine, your data.** No uploads, no third-party cloud. Patient data and unpublished results stay on your hardware.

**Tool calls, not token dreams.** Agents execute real bioinformatics tools: Scanpy, PyDESeq2, AnnData. Every result is traceable to a function call with validated inputs and outputs.

**Reproducible by design.** W3C-PROV provenance tracking, Jupyter notebook export, and parameter schemas for every analysis. The same question produces the same pipeline.

## Agents

18 specialist agents across 9 packages. Install exactly what you need.

| Package | Domain | Status |
|---------|--------|--------|
| **lobster-ai** | Core engine, supervisor, infrastructure | Stable |
| **lobster-transcriptomics** | Single-cell & bulk RNA-seq, DE analysis, cell type annotation | Stable |
| **lobster-research** | PubMed, GEO, Pride, and more.. dataset discovery | Stable |
| **lobster-visualization** | Publication-quality Plotly plots | Stable |
| **lobster-metadata** | ID mapping, sample filtering, metadata validation | Stable |
| **lobster-genomics** | GWAS pipeline, clinical variant analysis | Beta |
| **lobster-proteomics** | Mass spec & affinity proteomics, biomarker discovery | Beta |
| **lobster-metabolomics** | LC-MS, GC-MS, NMR analysis | Beta |
| **lobster-structural-viz** | Protein structure visualization | Beta |
| **lobster-ml** | Feature selection, survival analysis, MOFA integration | Beta |

<details>
<summary>Install individual agents</summary>

```bash
# Pick agents interactively
lobster init --force

# Or install specific packages
uv tool install lobster-ai --with lobster-proteomics --with lobster-genomics

# Install everything
uv tool install 'lobster-ai[full,anthropic]'
```

</details>

<details>
<summary>Build your own agent</summary>

Create custom agents for any domain. Agents plug in via Python entry points — discovered automatically, no core changes needed.

Install the **lobster-dev** skill to teach your coding agent the full architecture:

```bash
curl -fsSL https://skills.lobsterbio.com | bash
```

Then ask your coding agent: *"Create a Lobster agent for [your domain]"* — it knows the package structure, AGENT_CONFIG pattern, factory function, tool design, testing, and the 28-step checklist.

</details>

## Usage

```bash
# Interactive chat
lobster chat

# Single-turn queries
lobster query "Download GSE109564 and cluster cells"
lobster query "Search PubMed for CRISPR studies in 2024"

# Session continuity
lobster query --session-id project1 "Search for Alzheimer's scRNA-seq datasets"
lobster query --session-id project1 "Download the top 3 results"
lobster query --session-id latest "Cluster the first dataset"
```

<details>
<summary>Pipeline export and slash commands</summary>

```bash
lobster chat
> /pipeline export         # Export reproducible Jupyter notebook
> /pipeline list           # List exported pipelines
> /pipeline run analysis.ipynb geo_gse109564
> /data                    # Show loaded datasets
> /status                  # Session info
> /help                    # All commands
```

</details>

<details>
<summary>Capabilities by domain</summary>

**Transcriptomics**
- Single-cell RNA-seq: QC, doublet detection (Scrublet), batch integration (Harmony/scVI), clustering, cell type annotation, trajectory inference (DPT/PAGA)
- Bulk RNA-seq: Salmon/kallisto/featureCounts import, sample QC, batch detection, normalization (DESeq2/VST/CPM), DE with PyDESeq2, GSEA, publication-ready export

**Genomics**
- GWAS: VCF/PLINK import, LD pruning, kinship, association testing, result clumping
- Clinical: variant annotation (VEP), gnomAD frequencies, ClinVar pathogenicity, variant prioritization

**Proteomics**
- Mass spec: MaxQuant/DIA-NN/Spectronaut import, PTM analysis, peptide-to-protein rollup, batch correction
- Affinity: Olink NPX/SomaScan ADAT/Luminex MFI import, LOD quality, bridge normalization
- Downstream: GO/Reactome/KEGG enrichment, kinase enrichment (KSEA), STRING PPI, biomarker panel selection

**Metabolomics**
- LC-MS, GC-MS, NMR with auto-detection
- QC (RSD, TIC), filtering, imputation, normalization (PQN/TIC/IS)
- PCA, PLS-DA, OPLS-DA, m/z annotation (HMDB/KEGG), lipid class analysis

**Machine Learning**
- Feature selection (stability selection, LASSO, variance filter)
- Survival analysis (Cox models, Kaplan-Meier, risk stratification)
- Cross-validation, SHAP interpretability, multi-omics integration (MOFA)

**Research & Metadata**
- Literature discovery (PubMed, PMC, GEO, PRIDE, MetaboLights)
- Dataset download orchestration, metadata harmonization, sample filtering

</details>

<details>
<summary>LLM providers</summary>

Lobster supports 5 LLM providers. Configure via `lobster init` or environment variables.

| Provider | Type | Setup | Use Case |
|----------|------|-------|----------|
| **Ollama** | Local | `ollama pull gpt-oss:20b` | Privacy, zero cost, offline |
| **Anthropic** | Cloud | API key | Fastest, best quality |
| **AWS Bedrock** | Cloud | AWS credentials | Enterprise, compliance |
| **Google Gemini** | Cloud | Google API key | Multimodal, long context |
| **Azure AI** | Cloud | Endpoint + credential | Enterprise Azure |

</details>

<details>
<summary>Coding agent skills</summary>

Teach your coding agents (Claude Code, Codex, Gemini CLI, OpenClaw) to use Lobster:

```bash
curl -fsSL https://skills.lobsterbio.com | bash
```

Installs two skills:
- **lobster-use** — End-user workflows (search PubMed, analyze cells, RNA-seq)
- **lobster-dev** — Developer guide (create agents, extend services, testing)

</details>

## Development

```bash
git clone https://github.com/the-omics-os/lobster.git
cd lobster
make dev-install    # editable install with dev deps
make test           # run all tests
make format         # black + isort
```

<details>
<summary>Environment variables</summary>

Set during `lobster init` or manually:

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Anthropic Direct |
| `AWS_BEDROCK_ACCESS_KEY` | AWS Bedrock |
| `AWS_BEDROCK_SECRET_ACCESS_KEY` | AWS Bedrock |
| `GOOGLE_API_KEY` | Google Gemini |
| `OLLAMA_BASE_URL` | Ollama server (default: `http://localhost:11434`) |
| `LOBSTER_LLM_PROVIDER` | Explicit provider selection |
| `LOBSTER_WORKSPACE` | Workspace directory |

</details>

## Ecosystem

| Project | Description | Link |
|---------|-------------|------|
| **Lobster AI** | Open-source multi-agent engine | [lobsterbio.com](https://lobsterbio.com) |
| **Omics-OS Cloud** | Managed cloud platform | [app.omics-os.com](https://app.omics-os.com) |
| **Documentation** | Guides, API reference, tutorials | [docs.omics-os.com](https://docs.omics-os.com) |

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

**Code:** AGPL-3.0-or-later · **Docs:** CC BY 4.0 · See [LICENSE](LICENSE)

---

<p align="center">
  Built by <a href="https://omics-os.com">Omics-OS</a> to accelerate multi-omics research
  <br>
  <a href="https://omics-os.com">Omics-OS</a> · <a href="https://lobsterbio.com">Lobster AI</a> · <a href="https://docs.omics-os.com">Docs</a>
</p>
