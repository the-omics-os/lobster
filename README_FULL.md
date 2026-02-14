# 🦞 Lobster AI - Complete Reference

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL%203.0--or--later-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Documentation: CC BY 4.0](https://img.shields.io/badge/Documentation-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

> **This is the complete reference documentation.** For a quick overview, see [README.md](README.md).

**Bioinformatics co-pilot to automate redundant tasks so you can focus on science**

---

## Table of Contents

- [What is Lobster AI?](#what-is-lobster-ai)
- [Installation Methods](#installation-methods)
  - [PyPI Installation (Recommended)](#pypi-installation-recommended)
  - [Platform-Specific Native Installation](#platform-specific-native-installation)
  - [Docker Deployment](#docker-deployment)
- [LLM Provider Configuration](#llm-provider-configuration)
  - [Ollama (Local)](#ollama-local)
  - [Anthropic (Cloud)](#anthropic-cloud)
  - [AWS Bedrock (Enterprise)](#aws-bedrock-enterprise)
  - [Provider Auto-Detection](#provider-auto-detection)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Features](#features)
- [API Rate Limits](#api-rate-limits)
- [Cell Type Annotation - Development Status](#cell-type-annotation---development-status)
- [Premium Features](#premium-features)
- [Roadmap](#roadmap)
- [For Developers](#for-developers)
- [Uninstalling](#uninstalling)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## What is Lobster AI?

Lobster AI is a bioinformatics platform that combines specialized AI agents with open-source tools to analyze complex multi-omics data, discover relevant literature, and manage metadata across datasets. Simply describe your analysis needs in natural language - no coding required.

**Perfect for:**
- Bioinformatics researchers analyzing RNA-seq data
- Computational biologists seeking intelligent analysis workflows
- Life science teams requiring reproducible, publication-ready results
- Students learning modern bioinformatics approaches

---

## Installation Methods

### PyPI Installation (Recommended)

```bash
# Install uv if not already installed
# macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install Lobster globally
uv tool install lobster-ai

# Configure API keys
lobster init

# Start using Lobster
lobster chat
```

**Alternative: Local virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install lobster-ai  # or: pip install lobster-ai
lobster init
lobster chat
```

### Platform-Specific Native Installation

#### macOS (Native)

```bash
git clone https://github.com/the-omics-os/lobster.git
cd lobster
make install
source .venv/bin/activate
lobster init
lobster chat

# Optional: Install globally
make install-global
```

#### Ubuntu/Debian (Native)

```bash
# Install system dependencies (REQUIRED)
sudo apt update
sudo apt install -y \
    build-essential \
    python3.12-dev \
    python3.12-venv \
    libhdf5-dev \
    libxml2-dev \
    libxslt-dev \
    libffi-dev \
    libssl-dev \
    libblas-dev \
    liblapack-dev

git clone https://github.com/the-omics-os/lobster.git
cd lobster
make install
source .venv/bin/activate
lobster chat
```

#### Windows

**Option 1: Docker Desktop (Recommended)**
```powershell
# Install Docker Desktop: https://www.docker.com/products/docker-desktop/
git clone https://github.com/the-omics-os/lobster.git
cd lobster
copy .env.example .env
notepad .env  # Add your API key
docker-compose run --rm lobster-cli
```

**Option 2: Native (Experimental)**
```powershell
# Install Python 3.12 from python.org
git clone https://github.com/the-omics-os/lobster.git
cd lobster
.\install.ps1
notepad .env
.\.venv\Scripts\Activate.ps1
lobster chat
```

### Docker Deployment

```bash
git clone https://github.com/the-omics-os/lobster.git
cd lobster
cp .env.example .env
# Edit .env with your API keys

# Build and run
make docker-build
make docker-run-cli

# Or as web service
make docker-run-server
curl http://localhost:8000/health
```

See [Docker Deployment Guide](wiki/43-docker-deployment-guide.md) for AWS ECS, Kubernetes, and production configurations.

---

## LLM Provider Configuration

Lobster supports three LLM providers. Choose based on your needs:

| Provider | Type | Cost | Best For |
|----------|------|------|----------|
| 🦙 **Ollama** | Local | Free | Privacy, offline, no API costs |
| 🔷 **Anthropic** | Cloud | Pay-per-use | Quick start, best quality |
| ☁️ **AWS Bedrock** | Cloud | Enterprise | Production, high rate limits |

### Ollama (Local)

Run Lobster **completely locally** with no cloud dependencies.

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull a model
ollama pull llama3:8b-instruct

# 3. Set provider and run
export LOBSTER_LLM_PROVIDER=ollama
lobster chat
```

**Model recommendations:**
| Model | RAM Required | Use Case |
|-------|--------------|----------|
| `llama3:8b-instruct` | 8-16GB | Testing, light analysis |
| `mixtral:8x7b-instruct` | 24-32GB | Production workflows |
| `llama3:70b-instruct` | 48GB VRAM | Maximum quality (GPU required) |

**Advanced configuration:**
```bash
export OLLAMA_DEFAULT_MODEL=mixtral:8x7b-instruct
export OLLAMA_BASE_URL=http://localhost:11434
```

### Anthropic (Cloud)

Quick setup for Claude API access.

```bash
lobster init
# Or manually:
export ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```

Get API key: https://console.anthropic.com/

### AWS Bedrock (Enterprise)

Production-grade with enterprise rate limits.

```bash
lobster init
# Or manually:
export AWS_BEDROCK_ACCESS_KEY=your-access-key
export AWS_BEDROCK_SECRET_ACCESS_KEY=your-secret-key
```

Setup guide: https://aws.amazon.com/bedrock/

### Provider Auto-Detection

Lobster automatically detects your provider based on:
1. `LOBSTER_LLM_PROVIDER` environment variable (explicit)
2. Ollama server running locally
3. `ANTHROPIC_API_KEY` present
4. AWS Bedrock credentials present

**Running multiple providers simultaneously:**
```bash
# Terminal 1: Local with Ollama
export LOBSTER_LLM_PROVIDER=ollama
lobster chat

# Terminal 2: Cloud with Claude
export LOBSTER_LLM_PROVIDER=anthropic
lobster chat
```

---

## Configuration

### Interactive Setup (Recommended)

```bash
lobster init        # Launch configuration wizard
lobster config test # Test API connectivity
lobster config show # Display current configuration
```

### Manual Configuration (.env file)

```bash
# Choose ONE LLM provider:

# Option 1: Ollama (Local)
LOBSTER_LLM_PROVIDER=ollama
OLLAMA_DEFAULT_MODEL=llama3:8b-instruct

# Option 2: Claude API
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# Option 3: AWS Bedrock
AWS_BEDROCK_ACCESS_KEY=your-access-key
AWS_BEDROCK_SECRET_ACCESS_KEY=your-secret-key

# Optional: Enhanced literature search
NCBI_API_KEY=your-ncbi-api-key
NCBI_EMAIL=your.email@example.com

# Optional: Performance tuning
LOBSTER_PROFILE=production
LOBSTER_MAX_FILE_SIZE_MB=500
LOBSTER_LOG_LEVEL=WARNING
```

### CI/CD Non-Interactive Mode

```bash
lobster init --non-interactive --anthropic-key=sk-ant-xxx
lobster init --non-interactive --bedrock-access-key=xxx --bedrock-secret-key=yyy
```

---

## Usage Examples

### Interactive Chat Mode

```bash
lobster chat

🦞 You: Download GSE109564 and perform single-cell clustering analysis

🦞 Lobster: I'll download and analyze this single-cell dataset for you...
✓ Downloaded 5,000 cells × 20,000 genes
✓ Quality control: filtered to 4,477 high-quality cells
✓ Identified 12 distinct cell clusters
✓ Generated UMAP visualization and marker gene analysis
```

### Single Query Mode

```bash
lobster query "download GSE109564 and perform quality control"
lobster query --workspace ~/my_analysis "cluster the loaded dataset"
lobster query --reasoning "differential expression between conditions"
```

### Dashboard Mode

```bash
lobster dashboard
# Or from chat: /dashboard
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/files` | List workspace files |
| `/read <file>` | Load a dataset |
| `/data` | Show current dataset info |
| `/plots` | List visualizations |
| `/workspace` | Show workspace info |
| `/pipeline export` | Export reproducible notebook |

---

## Features

### Single-Cell RNA-seq
- Quality control and filtering
- Normalization and scaling
- Clustering and UMAP visualization
- Cell type annotation
- Marker gene identification
- Pseudobulk aggregation

### Bulk RNA-seq
- Differential expression with pyDESeq2
- R-style formula-based statistics
- Complex experimental designs
- Batch effect correction

### Data Management
- Support for CSV, Excel, H5AD, 10X formats
- Multi-source dataset discovery (GEO, SRA, PRIDE, ENA)
- Literature mining and full-text retrieval
- Cross-dataset metadata harmonization
- Sample ID mapping and validation

### Optional ML Features (PREMIUM)

```bash
pip install lobster-ai[ml]  # Adds PyTorch + scVI-tools (~500MB)
```

Includes: scVI integration, GPU acceleration, advanced embeddings.

---

## API Rate Limits

| Use Case | Provider | Notes |
|----------|----------|-------|
| Quick Testing | Claude API | May encounter rate limits |
| Development | Claude API + Rate Increase | Request higher limits |
| Production | AWS Bedrock | Enterprise-grade limits |
| Privacy/Offline | Ollama | No limits, free |

If you encounter rate limit errors:
1. Wait and retry (limits reset after ~60 seconds)
2. [Request rate increase from Anthropic](https://docs.anthropic.com/en/api/rate-limits)
3. Switch to AWS Bedrock for production
4. Use Ollama for unlimited local usage

---

## Cell Type Annotation - Development Status

**IMPORTANT: Built-in marker gene lists are preliminary and not scientifically validated.**

Current limitations:
- No evidence scoring (AUC, logFC, specificity metrics)
- No validation against reference atlases
- No tissue/context-specific optimization
- SASP/Senescence detection not reliable with RNA-seq alone
- Tumor cell detection should use CNV inference (inferCNV/CopyKAT)

**Recommended for production:**
1. Provide custom validated markers specific to your tissue/context
2. Use reference-based tools: [Azimuth](https://azimuth.hubmapconsortium.org/), [CellTypist](https://www.celltypist.org/), [scANVI](https://docs.scvi-tools.org/)
3. Validate annotations manually with known markers

See [Manual Annotation Guide](wiki/35-manual-annotation-service.md) for details.

---

## Premium Features

Unlock advanced capabilities with a Lobster Cloud subscription:

- **Proteomics analysis** (DDA/DIA workflows, missing value handling)
- **Metadata assistant** for cross-dataset harmonization
- Priority support and cloud compute options
- Custom agent packages for enterprise customers

```bash
lobster activate <your-cloud-key>
lobster status  # Check tier and features
```

Contact: [info@omics-os.com](mailto:info@omics-os.com) | [Pricing](https://omics-os.com)

---

## Roadmap

**Current (v1.0+):**
- ✅ Single-cell & bulk RNA-seq analysis
- ✅ Literature mining & dataset discovery
- ✅ Protein structure visualization
- ✅ Machine learning & survival analysis

**2026 Development:**
- Knowledge graph integration for multi-dataset analysis
- Lobster Cloud compute infrastructure
- Enhanced multi-omics workflows (MuData integration)
- Community-contributed agent marketplace

**Submit feature ideas:** [GitHub Discussions](https://github.com/the-omics-os/lobster/discussions)

---

## For Developers

Lobster follows an **open-core model** with runtime tier gating:

```
lobster/config/subscription_tiers.py  (defines FREE vs PREMIUM tiers)
                ↓
    ComponentRegistry runtime checks
                ↓
     TierRestrictedError for premium features
                ↓
         lobster-ai on PyPI (all code public)
```

**Key files:**
- `subscription_tiers.py` - Defines which agents/features are FREE vs PREMIUM
- `core/component_registry.py` - Discovers agents via entry points
- `core/license_manager.py` - Validates entitlements at runtime
- `generate_allowlist.py --write` - Regenerates the sync allowlist
- `CLAUDE.md` - Complete developer guide with architecture details

CI enforces that `public_allowlist.txt` stays in sync with `subscription_tiers.py`.

---

## Uninstalling

### Remove Package

```bash
# Global installation (uv tool)
uv tool uninstall lobster-ai

# Virtual environment
pip uninstall lobster-ai
rm -rf .venv

# Development (make)
make uninstall-global
make uninstall
```

### Remove User Data (Optional)

```bash
rm -rf ~/.lobster
rm -rf ~/.lobster_workspace
rm .env  # In project directory
```

### Verify Removal

```bash
which lobster  # Should output nothing
uv tool list | grep lobster  # Should output nothing
```

---

## Troubleshooting

See [Troubleshooting Guide](https://github.com/the-omics-os/lobster/wiki/28-troubleshooting) for common issues.

**Quick fixes:**
- **Import errors**: `make clean-install`
- **Rate limits**: Switch to Bedrock or Ollama
- **Ollama not detected**: Ensure `ollama serve` is running

---

## License

Lobster AI is open source under **GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)**. This license ensures all users receive the freedoms to use, study, share, and modify the software.

Documentation is licensed **CC-BY-4.0**. Contributions accepted under CLA.

For commercial licensing: [info@omics-os.com](mailto:info@omics-os.com)

---

<div align="center">

**Transform Your Bioinformatics Research Today**

[Get Started](README.md) | [Documentation](https://github.com/the-omics-os/lobster/wiki)

*Made with ❤️ by [Omics-OS](https://omics-os.com)*

</div>
