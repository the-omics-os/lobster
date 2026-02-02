# Tooling & Environment Reference

Setup, testing, and deployment documentation. Load on-demand when configuring environment.

---

## Technology Stack

| Area | Tech |
|------|------|
| Agent framework | LangGraph |
| Models | AWS Bedrock (Claude), Anthropic Direct, Ollama (local) |
| Language | Python 3.11+ (typing, async/await) |
| Data structures | AnnData, MuData |
| Bioinformatics | Scanpy, PyDESeq2 |
| Genomics | cyvcf2 (VCF), bed-reader (PLINK), sgkit (GWAS/PCA), pygenebe |
| ML (optional) | PyTorch, scVI-tools (`pip install lobster-ai[ml]`) |
| CLI | Typer, Rich, prompt_toolkit |
| Visualization | Plotly |
| Storage | H5AD, HDF5, JSONL, S3 backends |

---

## Environment Setup

```bash
make dev-install     # full dev setup
make install         # minimal install
make clean-install   # fresh env
source .venv/bin/activate
```

---

## Testing

```bash
make test            # all tests
make test-fast       # parallel subset
make format          # black + isort
make lint            # flake8/pylint/bandit
make type-check      # mypy

pytest tests/unit/
pytest tests/integration/
pytest tests/integration/ -m real_api
pytest tests/integration/ -m "real_api and slow"
```

**Markers**:
- `@pytest.mark.real_api` – requires network + keys
- `@pytest.mark.slow` – >30s tests
- `@pytest.mark.integration` – multi-component

**Environment Variables**:

| Variable | Required | Purpose |
|----------|----------|---------|
| `AWS_BEDROCK_ACCESS_KEY` | Conditional | AWS Bedrock access key |
| `AWS_BEDROCK_SECRET_ACCESS_KEY` | Conditional | AWS Bedrock secret key |
| `ANTHROPIC_API_KEY` | Conditional | Anthropic Direct API key |
| `LOBSTER_LLM_PROVIDER` | No | Explicit provider: `ollama`, `bedrock`, `anthropic` |
| `OLLAMA_BASE_URL` | No | Ollama server (default: `http://localhost:11434`) |
| `OLLAMA_DEFAULT_MODEL` | No | Model name (default: `gpt-oss:20b`) |
| `NCBI_API_KEY` | No | PubMed (higher rate limit) |
| `LOBSTER_CLOUD_KEY` | No | enables cloud client mode |

**Note**: At least one LLM provider is required (Anthropic OR Bedrock OR Ollama).

---

## Running the App

```bash
lobster chat                    # interactive, multi-turn
lobster query "your request"    # single-turn automation
lobster --help                  # CLI help
```

**Chat vs Query**:
- `chat`: follow-ups, clarify, exploratory work
- `query`: single-shot, script/CI-friendly

**Session Continuity**:
```bash
lobster query --session-id "project_1" "Search PubMed"
lobster query --session-id "project_1" "Download the first dataset"
lobster query --session-id latest "Cluster that dataset"
```

**Workspace Configuration**:
```bash
lobster chat --workspace /path/to/workspace
lobster query "analyze data" -w ~/my_workspace
export LOBSTER_WORKSPACE=/shared/workspace
# Resolution: --workspace flag > LOBSTER_WORKSPACE env > cwd/.lobster_workspace
```

**CLI Commands**:

| Category | Commands |
|----------|----------|
| Help | `/help`, `/modes` |
| Session | `/session`, `/status` |
| Data | `/data`, `/files`, `/read <file>` |
| Workspace | `/workspace`, `/workspace list`, `/workspace load <name>` |
| Plots | `/plots` |
| Pipelines | `/pipeline export`, `/pipeline list`, `/pipeline run <nb> <modality>` |

---

## Package Publishing (PyPI)

**Package**: `lobster-ai` (PyPI) / `lobster` (import)
**Version**: `lobster/version.py` (single source of truth)
**Publishing**: GitHub Actions on git tags (`v*.*.*`)

**Critical Rule**: Only `lobster-local` (public) publishes to PyPI.

**Quick Release**:
```bash
# 1. Update lobster/version.py
__version__ = "0.2.0"

# 2. Tag and push
git add lobster/version.py
git commit -m "chore: bump version to 0.2.0"
git push origin main
git tag -a v0.2.0 -m "Release 0.2.0"
git push origin v0.2.0  # Triggers 7-stage pipeline
```

**Pipeline**: Sync → Build → TestPyPI → Manual Approval → PyPI → Release → Summary

**Details**: `docs/PYPI_SETUP_GUIDE.md`, `.github/workflows/publish-pypi.yml`

---

## Claude Code Integration

**Installation**:
```bash
# Automated
curl -fsSL https://raw.githubusercontent.com/the-omics-os/lobster-local/main/claude-skill/install.sh | bash

# Manual
mkdir -p ~/.claude/skills/
curl -o ~/.claude/skills/lobster-bioinformatics.md \
  https://raw.githubusercontent.com/the-omics-os/lobster-local/main/claude-skill/SKILL.md
```

**Key Files**:
- `claude-skill/SKILL.md` – Skill definition
- `claude-skill/install.sh` – Automated installation
- `wiki/50-claude-code-integration.md` – Complete documentation

**Usage Pattern**:
```
User (IDE) → Claude Code: "Download GSE109564 and cluster cells"
     ↓
Claude Code → Lobster skill
     ↓
lobster query --session-id latest "Download GSE109564 and cluster cells"
     ↓
research_agent → data_expert → singlecell_expert
     ↓
Results + file paths (.lobster_workspace/)
```

**Trigger Keywords**: H5AD, CSV, GEO/SRA accessions, QC, clustering, differential expression, PubMed, single-cell, bulk RNA-seq, proteomics

---

## Troubleshooting

- **Install issues**: Python 3.11+ required, try `make clean-install`
- **CLI quirks**: check `PROMPT_TOOLKIT_AVAILABLE`, verify `LobsterClientAdapter`
- **Profiling**: `--profile-timings` or `LOBSTER_PROFILE_TIMINGS=1`
- **Cloud mode**: ensure `LOBSTER_CLOUD_KEY` set, check network + timeouts
