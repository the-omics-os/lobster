# 🦞 Lobster AI

[![PyPI version](https://img.shields.io/pypi/v/lobster-ai.svg)](https://pypi.org/project/lobster-ai/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Documentation](https://img.shields.io/badge/docs-omics--os.com-green.svg)](https://docs.omics-os.com)

> Multi-agent bioinformatics engine. Analyze multi-omics data in natural language.

[Documentation](https://docs.omics-os.com) · [PyPI](https://pypi.org/project/lobster-ai/) · [Omics-OS Cloud](https://app.omics-os.com) · [Agent Template](https://github.com/the-omics-os/lobster-agent-template)

---

## What is Lobster AI?

Lobster AI is an open-source multi-agent platform for bioinformatics. Describe your analysis in natural language — Lobster routes it to specialized AI agents that handle single-cell RNA-seq, bulk RNA-seq, proteomics, genomics, literature mining, and machine learning. Results are reproducible (W3C-PROV), exportable as Jupyter notebooks, and backed by the same open-source tools researchers already trust (Scanpy, PyDESeq2, AnnData).

Built for bioinformaticians, computational biologists, and research teams who need intelligent workflows without writing code.

---

## Quick Start

**macOS / Linux** (recommended):
```bash
curl -fsSL https://install.lobsterbio.com | bash
```

**Windows** (PowerShell):
```powershell
irm https://install.lobsterbio.com/windows | iex
```

**Manual** (any platform):
```bash
uv tool install --python 3.13 'lobster-ai[full,anthropic]'
lobster init
lobster chat
```

> Lobster AI requires Python 3.12 or 3.13. The `--python` flag ensures uv uses a compatible version.

Or with pip (inside a Python 3.12/3.13 virtualenv):
```bash
pip install 'lobster-ai[full]'
lobster init
lobster chat
```

---

## Agent Packages

Lobster AI uses a modular architecture. Install exactly the agents you need.

| Package | Agents | Status |
|---------|--------|--------|
| **lobster-ai** | Supervisor, Core infrastructure | Beta |
| **lobster-transcriptomics** | Transcriptomics, Annotation, DE Analysis | Beta |
| **lobster-research** | Research, Data Expert | Beta |
| **lobster-visualization** | Visualization Expert | Beta |
| **lobster-metadata** | Metadata Assistant | Beta |
| **lobster-structural-viz** | Protein Structure Visualization | Beta |
| **lobster-genomics** | Genomics Expert | Alpha |
| **lobster-proteomics** | Proteomics Expert | Alpha |
| **lobster-ml** | ML, Feature Selection, Survival Analysis | Alpha |

**Add individual agents** (uv tool installs):
```bash
uv tool install --python 3.13 lobster-ai --with lobster-proteomics --with lobster-genomics
```
Or run `lobster init --force` to interactively select agents — it generates the command for you.

**Install everything:**
```bash
uv tool install --python 3.13 'lobster-ai[full,anthropic]'   # uv tool (recommended)
pip install 'lobster-ai[full]'                  # pip
```

**Upgrade:**
```bash
uv tool upgrade lobster-ai                      # uv tool
pip install --upgrade lobster-ai                 # pip
```

**Build your own agent:**
Use the [lobster-agent-template](https://github.com/the-omics-os/lobster-agent-template) to create custom analysis agents.

---

## Coding Agent Skills

Teach your coding agents (Claude Code, Codex, Gemini CLI, OpenClaw) to use Lobster:

```bash
curl -fsSL https://skills.lobsterbio.com | bash
```

Installs two skills:
- **lobster-use** — End-user workflows (search PubMed, analyze cells, RNA-seq)
- **lobster-dev** — Developer guide (create agents, extend services, testing)

---

## LLM Providers

Lobster supports 5 LLM providers. Choose based on your needs:

| Provider | Type | Setup | Use Case |
|----------|------|-------|----------|
| **Ollama** | Local | `ollama pull gpt-oss:20b` | Privacy, zero cost, offline |
| **Anthropic** | Cloud | API key | Fastest, best quality |
| **AWS Bedrock** | Cloud | AWS credentials | Enterprise, compliance |
| **Google Gemini** | Cloud | Google API key | Multimodal, long context |
| **Azure AI** | Cloud | Endpoint + credential | Enterprise Azure |

Configure via `lobster init` or set environment variables manually.

---

## Features

**Multi-Omics Analysis**
- Single-cell RNA-seq (QC, clustering, cell type annotation, trajectory)
- Bulk RNA-seq (Kallisto/Salmon import, DE with PyDESeq2)
- Proteomics (DDA/DIA, missing values, normalization)
- Genomics (VCF/PLINK, GWAS, PCA, variant annotation)
- Multi-omics integration (MOFA, pathway enrichment)

**Machine Learning**
- Feature selection (stability selection, LASSO, variance filter)
- Survival analysis (Cox models, Kaplan-Meier, risk stratification)
- Cross-validation (stratified k-fold, nested CV)
- Interpretability (SHAP, feature importance)

**Research & Metadata**
- Literature discovery (PubMed, PMC, GEO)
- Dataset downloads (GEO, SRA, PRIDE)
- Metadata harmonization and validation
- Sample grouping and filtering

**Reproducibility**
- W3C-PROV provenance tracking
- Jupyter notebook export
- Parameter schemas for all analyses

---

## Usage Examples

**Interactive chat:**
```bash
lobster chat
```

**Single-turn queries:**
```bash
lobster query "Search PubMed for CRISPR studies in 2024"
lobster query "Download GSE109564 and cluster cells"
```

**Session continuity:**
```bash
lobster query --session-id project1 "Search for Alzheimer's scRNA-seq datasets"
lobster query --session-id project1 "Download the top 3 results"
lobster query --session-id latest "Cluster the first dataset"
```

**Export pipelines:**
```bash
lobster chat
> /pipeline export
> /pipeline list
> /pipeline run analysis.ipynb geo_gse109564
```

---

## Development

```bash
# Clone repository
git clone https://github.com/the-omics-os/lobster.git
cd lobster

# Install with dev dependencies (editable)
make dev-install

# Run tests
make test

# Format and lint
make format && make lint

# Activate environment
source .venv/bin/activate

# Test as end user (uv tool install)
uv tool install --python 3.13 'lobster-ai[full,anthropic]'
```

**Environment variables (will be created during init):**
- `ANTHROPIC_API_KEY` — Anthropic Direct
- `AWS_BEDROCK_ACCESS_KEY`, `AWS_BEDROCK_SECRET_ACCESS_KEY` — AWS Bedrock
- `GOOGLE_API_KEY` — Google Gemini
- `OLLAMA_BASE_URL` — Ollama server (default: http://localhost:11434)
- `LOBSTER_LLM_PROVIDER` — Explicit provider selection
- `LOBSTER_WORKSPACE` — Workspace directory

---

## Documentation

For detailed documentation, see [README_FULL.md](README_FULL.md) or visit [docs.omics-os.com](https://docs.omics-os.com).

**Key topics:**
- Platform-specific installation (macOS, Linux, Windows)
- One-line installers (`install.lobsterbio.com`)
- uv tool install and upgrade workflows
- Docker deployment
- LLM provider configuration
- Premium features
- Troubleshooting

---

## Ecosystem

| Project | Description | Link |
|---------|-------------|------|
| **Lobster AI** | Open-source multi-agent bioinformatics engine | [lobsterbio.com](https://lobsterbio.com) |
| Omics-OS Cloud | Managed cloud platform | [app.omics-os.com](https://app.omics-os.com) |
| Documentation | Guides, API reference, tutorials | [docs.omics-os.com](https://docs.omics-os.com) |
| Agent Template | Create your own Lobster agent | [GitHub](https://github.com/the-omics-os/lobster-agent-template) |

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Ways to contribute:**
- Report bugs and request features via GitHub Issues
- Submit pull requests for bug fixes or new features
- Improve documentation
- Create custom agent packages

---

## License

**Code:** AGPL-3.0-or-later
**Documentation:** CC BY 4.0

See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built by <a href="https://omics-os.com">Omics-OS</a> to accelerate multi-omics research
  <br>
  <a href="https://omics-os.com">Omics-OS</a> · <a href="https://lobsterbio.com">Lobster AI</a> · <a href="https://docs.omics-os.com">Docs</a>
</p>
