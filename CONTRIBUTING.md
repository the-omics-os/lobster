# Contributing to Lobster AI

Thank you for your interest in contributing to Lobster AI -- the open-source multi-agent bioinformatics engine behind Omics-OS.

## Quick Start

```bash
# 1. Fork and clone
git clone https://github.com/YOUR-USERNAME/lobster.git
cd lobster

# 2. Install (uses uv internally -- never bare pip)
make dev-install

# 3. Activate and configure
source .venv/bin/activate
lobster init          # API keys, agent selection
lobster config test   # Verify connectivity

# 4. Verify
lobster chat          # Interactive mode

# 5. Create your branch
git checkout -b feature/your-feature-name
```

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.12+ | Required |
| Git | Required |
| `uv` | **Mandatory** -- all Python tooling uses `uv`. Never bare `pip`. |
| LLM API key | Anthropic, AWS Bedrock, or other supported provider |

## Architecture Overview

Lobster AI v1.0.0 uses a **modular package architecture**:

```
lobster/                     # Core SDK (supervisor, services, tools, infrastructure)
packages/
  lobster-transcriptomics/   # 3 agents (transcriptomics, annotation, DE analysis)
  lobster-research/          # 2 agents (research, data expert)
  lobster-visualization/     # 1 agent
  lobster-proteomics/        # 3 agents (proteomics, proteomics DE, biomarker)
  lobster-genomics/          # 2 agents (genomics, variant analysis)
  lobster-metabolomics/      # 1 agent
  lobster-metadata/          # 1 agent
  lobster-structural-viz/    # 1 agent
  lobster-ml/                # 3 agents (ML, feature selection, survival)
  lobster-drug-discovery/    # 4 agents (drug discovery, cheminformatics, clinical, PGx)
```

**Key concepts:**

- **Entry point discovery**: Agents register via `[project.entry-points."lobster.agents"]` in their `pyproject.toml`. `ComponentRegistry` discovers them at runtime -- no hardcoded registries.
- **AQUADIF tool taxonomy**: 10 categories (IMPORT, QUALITY, FILTER, PREPROCESS, ANALYZE, ANNOTATE, DELEGATE, SYNTHESIZE, UTILITY, CODE_EXEC). Every tool declares its category and provenance requirement.
- **Service pattern**: All services return `(AnnData, Dict, AnalysisStep)`. Tools wrap services and pass `ir=` to `log_tool_usage()`.
- **PEP 420 namespace packages**: No `lobster/__init__.py` in agent packages.

## Adding a New Agent

The recommended workflow:

```bash
# 1. Scaffold a new agent package
lobster scaffold agent --name my_expert --display-name "My Expert" \
  --description "My domain analysis"

# 2. Implement tools and services in the generated package
#    See skills/lobster-dev/ for detailed patterns

# 3. Validate the plugin structure (7 checks)
lobster validate-plugin ./lobster-mydomain/

# 4. Register entry points in your package's pyproject.toml
#    (scaffold generates this for you)

# 5. Run contract tests
pytest -m contract
```

For detailed implementation patterns (services, tools, AQUADIF metadata, prompts), see `skills/lobster-dev/references/`.

## Service Pattern

All analysis services return a 3-tuple:

```python
(processed_adata, stats_dict, ir)  # ir = AnalysisStep (provenance)
```

Tools must pass the `ir` to `data_manager.log_tool_usage(..., ir=ir)` for reproducibility. See `skills/lobster-dev/references/creating-services.md` for the full pattern.

## AQUADIF Tool Metadata

Every `@tool` function must declare its category:

```python
my_tool.metadata = {"categories": ["ANALYZE"], "provenance": True}
my_tool.tags = ["ANALYZE"]
```

Use string literals (not enum imports). Max 3 categories per tool. Validate with `pytest -m contract`.

## Testing

```bash
make test             # Run all tests
make format           # black + isort (run before committing)
pytest -m contract    # AQUADIF contract validation (all agents)
pytest tests/ -v      # Verbose output
```

## Ways to Contribute

- **Bug reports**: [GitHub Issues](https://github.com/the-omics-os/lobster/issues) with "bug" label. Include steps to reproduce.
- **Feature requests**: Open an issue with "enhancement" label. Describe the use case.
- **New analysis methods**: New agents, services, or tool implementations.
- **Documentation**: Fix typos, improve clarity, add examples.
- **Test coverage**: Edge cases, integration tests, contract tests.

## Pull Request Process

### Before Submitting

- [ ] All tests pass (`make test`)
- [ ] Code is formatted (`make format`)
- [ ] Contract tests pass (`pytest -m contract`) if touching agents/tools
- [ ] Documentation updated (docstrings, skills, or docs-site if applicable)
- [ ] New features include tests

### PR Description Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature (agent, service, tool)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Breaking change

## Testing
- [ ] Tests added/updated
- [ ] Contract tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style (PEP 8, type hints)
- [ ] AQUADIF metadata assigned to new tools
- [ ] Provenance (ir=) passed in log_tool_usage for new tools
- [ ] Self-review completed
```

## Code of Conduct

We follow a simple principle: **be kind, be constructive, be scientific.**

- Respect all contributors regardless of experience level
- Focus on facts and scientific accuracy
- Help newcomers learn and contribute
- Give constructive feedback on code and ideas

## Getting Help

- **[Documentation](https://docs.omics-os.com)** -- comprehensive guides
- **[GitHub Issues](https://github.com/the-omics-os/lobster/issues)** -- bugs and feature requests
- **[Email](mailto:kevin.yar@omics-os.com)** -- direct help

## License

By contributing to Lobster AI, you agree that your contributions will be licensed under the **AGPL-3.0-or-later** License.

---

Ready to contribute? We look forward to your work.
