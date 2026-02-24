# CLI Reference

Lobster AI command-line interface reference.

## Basic Usage

```bash
# Interactive chat mode
lobster chat

# Single query (non-interactive)
lobster query "Your request here"

# With session continuity
lobster query --session-id "my_project" "First request"
lobster query --session-id latest "Continue the previous work"
```

## Commands

### `lobster chat`

Interactive chat session with the multi-agent system.

```bash
lobster chat [OPTIONS]

Options:
  --workspace, -w PATH    Set workspace directory
  --model, -m MODEL       LLM model to use (default: from config)
  --verbose, -v           Enable verbose output
  --debug                 Enable debug mode
```

**In-chat commands:**
- `/help` — Show available commands
- `/data` — Show loaded data
- `/files` — List workspace files
- `/status` — Show session status
- `/pipeline export` — Export as Jupyter notebook
- `/quit`, `/exit` — End session

### `lobster query`

Single-turn query execution.

```bash
lobster query [OPTIONS] "QUERY"

Options:
  --session-id, -s ID     Session ID for continuity
  --workspace, -w PATH    Workspace directory
  --model, -m MODEL       LLM model
  --output, -o FORMAT     Output format (text, json)
  --verbose, -v           Verbose output
```

### `lobster init`

Interactive setup wizard for LLM provider, API keys, agent selection, and optional packages.

```bash
lobster init [OPTIONS]

Options:
  --global, -g              Save config globally (~/.config/lobster/) for all projects
  --force, -f               Overwrite existing configuration
  --non-interactive         Non-interactive mode (for CI/CD and coding agents)
  --anthropic-key KEY       Claude API key (non-interactive)
  --gemini-key KEY          Google Gemini API key (non-interactive)
  --openai-key KEY          OpenAI API key (non-interactive)
  --bedrock-access-key KEY  AWS Bedrock access key (non-interactive)
  --bedrock-secret-key KEY  AWS Bedrock secret key (non-interactive)
  --use-ollama              Use Ollama local LLM (non-interactive)
  --ollama-model MODEL      Ollama model name (default: llama3:8b-instruct)
  --profile PROFILE         Agent profile: development, production, performance, max
  --ncbi-key KEY            NCBI API key (optional)
  --cloud-key KEY           Omics-OS Cloud API key (optional)
  --agents LIST             Comma-separated agent names to enable
  --preset NAME             Agent preset: scrna-basic, scrna-full, multiomics-full
  --auto-agents             LLM-powered agent suggestion (requires --agents-description)
  --skip-docling            Skip docling install prompt
  --install-docling         Install docling for PDF intelligence
  --skip-extras             Skip all optional package prompts
  --skip-ssl-test           Skip SSL connectivity test
```

**NOTE:** The default (interactive) mode is a terminal wizard that requires user keyboard
input. Coding agents should use `--non-interactive` with the appropriate provider flags,
or ask the user to run `lobster init` in a separate terminal.

### `lobster config`

Manage configuration.

```bash
lobster config show           # Show current config
lobster config set KEY VALUE  # Set config value
lobster config list-models    # List available models
```

### `lobster agents`

List and manage agents.

```bash
lobster agents list           # List all agents
lobster agents info AGENT     # Show agent details
lobster agents enable AGENT   # Enable agent
lobster agents disable AGENT  # Disable agent
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic (Claude) API key | — |
| `GOOGLE_API_KEY` | Google Gemini API key | — |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `AWS_BEDROCK_ACCESS_KEY` | AWS Bedrock access key | — |
| `AWS_BEDROCK_SECRET_ACCESS_KEY` | AWS Bedrock secret key | — |
| `LOBSTER_PROFILE` | Agent profile (development/production/performance/max) | production |
| `NCBI_API_KEY` | NCBI API key for PubMed/GEO | — |
| `LOBSTER_CLOUD_KEY` | Omics-OS Cloud API key | — |
| `LOBSTER_SSL_VERIFY` | Enable/disable SSL verification | true |
| `LOBSTER_SSL_CERT_PATH` | Custom CA certificate bundle path | — |

## Configuration Files

Created by `lobster init`:

| Mode | Config | Credentials |
|------|--------|-------------|
| Workspace (default) | `.lobster_workspace/provider_config.json` | `.env` |
| Global (`--global`) | `~/.config/lobster/providers.json` | `~/.config/lobster/credentials.env` (0600) |

Agent selection stored in `.lobster_workspace/agents.json`.

## Session Management

Sessions persist conversation history and workspace state.

```bash
# Start named session
lobster query -s "my_analysis" "Load GEO dataset GSE12345"

# Continue latest session
lobster query -s latest "Run QC on the data"

# List sessions
lobster sessions list

# Resume specific session
lobster sessions resume "my_analysis"
```

## Workspace Structure

```
workspace/
├── data/                     # Loaded data files
│   ├── transcriptomics/
│   └── proteomics/
├── results/                  # Analysis outputs
├── notebooks/                # Exported Jupyter notebooks
├── cache/                    # Downloaded file cache
└── .lobster/
    ├── session.json          # Session state
    └── provenance.json       # W3C-PROV records
```

## Common Workflows

### Data Analysis

```bash
# Load and analyze
lobster query "Load GSE12345 and run QC"

# Multi-step with session
lobster query -s analysis "Load the GSE12345 dataset"
lobster query -s latest "Run quality control"
lobster query -s latest "Cluster the cells"
lobster query -s latest "Export as notebook"
```

### Literature Search

```bash
lobster query "Search PubMed for CRISPR cancer papers from 2024"
```

### Data Export

```bash
# Export current analysis as Jupyter notebook
lobster chat
> /pipeline export

# Or via query
lobster query -s latest "Export the analysis pipeline"
```

## Debugging

```bash
# Verbose mode
lobster query -v "Your request"

# Debug mode (full traces)
lobster --debug query "Your request"

# Check agent routing
lobster query -v "Why did you route to agent X?"
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Configuration error |
| 4 | Network error |
| 5 | Agent error |

## Help

```bash
lobster --help              # General help
lobster chat --help         # Chat command help
lobster query --help        # Query command help
```

For full CLI implementation, see `lobster/cli.py`.
