# @omicsos/lobster

Terminal client for [Omics-OS Cloud](https://omics-os.com) — The Operating System for Biology.

Powered by [Lobster AI](https://github.com/the-omics-os/lobster), a multi-agent bioinformatics platform for single-cell RNA-seq, bulk RNA-seq, proteomics, genomics, metabolomics, and more.

## Install

```bash
npm install -g @omicsos/lobster
```

Requires Node.js >= 22.

## Quick Start

```bash
# Cloud mode (no Python required)
lobster --cloud
lobster cloud login
lobster cloud chat

# Local mode (requires Python + lobster-ai)
pip install 'lobster-ai[full]'
lobster
```

## Cloud Mode

Connect directly to Omics-OS Cloud — agents run on managed infrastructure, no local setup needed.

```bash
lobster --cloud                              # Start cloud chat
lobster --cloud --session-id latest          # Resume last session
lobster --cloud --project-id <UUID>          # Associate with project
```

## Local Mode

Run agents on your machine with your own LLM API keys. Requires `lobster-ai` Python package.

```bash
pip install 'lobster-ai[full]'
lobster init
lobster
```

## Links

- [Omics-OS Cloud](https://app.omics-os.com) — Web workspace
- [Documentation](https://docs.omics-os.com) — Guides and tutorials
- [GitHub](https://github.com/the-omics-os/lobster) — Source code
- [PyPI](https://pypi.org/project/lobster-ai/) — Python package

## License

AGPL-3.0-or-later
