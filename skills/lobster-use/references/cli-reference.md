# CLI Reference

Complete reference for the Lobster AI command-line interface.

**Source**: [github.com/the-omics-os/lobster](https://github.com/the-omics-os/lobster) |
**PyPI**: [pypi.org/project/lobster-ai](https://pypi.org/project/lobster-ai/) |
**Docs**: `https://docs.omics-os.com/raw/docs/guides/cli-commands.md`

## Installation

| Method | Command |
|--------|---------|
| pip (recommended) | `pip install 'lobster-ai[full]' && lobster init` |
| uv tool (isolated) | `uv tool install 'lobster-ai[full]' && lobster init` |
| Targeted domain | `pip install 'lobster-ai[proteomics]'` (lighter) |

**Targeted extras**: `[proteomics]`, `[genomics]`, `[transcriptomics]` — each includes research + viz agents.

**Upgrade**: `pip install --upgrade lobster-ai` or `uv tool upgrade lobster-ai`

**Add agents (uv tool)**: `uv tool install lobster-ai --with lobster-proteomics --with lobster-genomics`

## Initialization (`lobster init`)

`lobster init` is an **interactive wizard** -- coding agents cannot run it directly.

**Non-interactive init** (agents can run this):

**Credential safety**: Always pass API keys via environment variables, never as raw strings.
Keys are written to workspace `.env` (mode 0600) or `~/.config/lobster/credentials.env`.

```bash
# Anthropic (most common)
lobster init --non-interactive --anthropic-key "$ANTHROPIC_API_KEY" --profile production

# Google Gemini
lobster init --non-interactive --gemini-key "$GOOGLE_API_KEY"

# OpenAI
lobster init --non-interactive --openai-key "$OPENAI_API_KEY"

# AWS Bedrock
lobster init --non-interactive --bedrock-access-key "$AWS_ACCESS_KEY_ID" --bedrock-secret-key "$AWS_SECRET_ACCESS_KEY"

# Ollama (local, no API key)
lobster init --non-interactive --use-ollama --ollama-model "llama3:8b-instruct"

# OpenRouter (600+ models)
lobster init --non-interactive --openrouter-key "$OPENROUTER_API_KEY"

# Azure AI
lobster init --non-interactive --azure-endpoint "$AZURE_AI_ENDPOINT" --azure-credential "$AZURE_AI_CREDENTIAL"
```

**Non-interactive flags**:

| Flag | Purpose |
|------|---------|
| `--profile <name>` | `development`, `production`, `performance`, `max` (Anthropic/Bedrock) |
| `--ncbi-key <key>` | NCBI API key for faster PubMed/GEO (via env var) |
| `--agents <list>` | Comma-separated agent names |
| `--preset <name>` | `scrna-basic`, `scrna-full`, `multiomics-full` |
| `--skip-extras` | Skip all optional packages |
| `--skip-ssl-test` | Skip SSL connectivity test |
| `--skip-docling` | Skip PDF intelligence install |
| `--install-docling` | Install docling |

**Advanced flags** (use with caution):

| Flag | Purpose |
|------|---------|
| `--global` | Save to `~/.config/lobster/` instead of workspace — writes global config |
| `--force` | Overwrite existing config (creates timestamped backup first) |
| `--cloud-key <key>` | Omics-OS Cloud API key (premium tier) |

**Check if already configured**:
```bash
lobster config-test --json    # Structured status
lobster status                # Human-readable
```

**Config file locations**:

| Mode | Config | Credentials |
|------|--------|-------------|
| Workspace (default) | `.lobster_workspace/provider_config.json` | `.env` |
| Global (`--global`) | `~/.config/lobster/providers.json` | `~/.config/lobster/credentials.env` |

**Priority**: workspace `.env` > global `credentials.env` > environment variables

## Top-Level Commands

| Command | Description |
|---------|-------------|
| `lobster chat` | Interactive chat (Go TUI by default) |
| `lobster query "..."` | Single-turn query |
| `lobster command <cmd>` | Execute slash command without LLM (~300ms) |
| `lobster init` | Configuration wizard |
| `lobster status` | Tier, packages, agents |
| `lobster config-test` | Test API connectivity |
| `lobster agents list` | List installed agent packages |
| `lobster agents info <name>` | Agent package details |
| `lobster serve --port 8080` | Start API server |
| `lobster dashboard` | Visual monitoring UI |
| `lobster purge` | Remove Lobster files (`--dry-run`, `--force`) |

## `lobster chat`

Interactive mode. **Go TUI** (Charm stack) is the default terminal experience.

```bash
lobster chat                        # Default (Go TUI auto-detected)
lobster chat --ui classic           # Force legacy Rich/Textual mode
lobster chat --classic              # Shorthand for --ui classic
lobster chat --no-intro             # Skip welcome animation
lobster chat -w ./my_analysis       # Set workspace
lobster chat -s "my_session"        # Continue named session
lobster chat --reasoning            # Show agent reasoning
```

**Go TUI features**:
- Tab completion for all 28+ slash commands and subcommands
- Dynamic completion for file paths and modality names
- Streaming markdown rendering (Glamour)

| Flag | Default | Description |
|------|---------|-------------|
| `-w, --workspace <path>` | auto | Workspace directory |
| `-s, --session-id <id>` | (none) | Named session to continue |
| `--ui <mode>` | auto | `go`, `classic`, or `auto` |
| `--classic` | -- | Shorthand for `--ui classic` |
| `--no-intro` | off | Skip intro animation |
| `--reasoning` | off | Show agent reasoning |
| `--stream/--no-stream` | on | Streaming (on by default in chat) |
| `-v, --verbose` | off | Debug output |
| `-p, --provider <name>` | config | Override LLM provider |
| `-m, --model <name>` | config | Override model |

## `lobster query`

Single-turn queries. Returns complete result by default (no streaming).

```bash
lobster query "Search PubMed for CRISPR in cancer"
lobster query --session-id "proj" "Download GSE109564"
lobster query --session-id "proj" "Run QC and cluster"
lobster query --session-id latest "Continue analysis"
lobster query --json "What data is loaded?" | jq .response
lobster query --output results.md "Generate report"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--session-id <id>` | (none) | Session continuity (required for multi-step) |
| `--session-id latest` | -- | Continue most recent session |
| `-w, --workspace <path>` | auto | Workspace directory |
| `-o, --output <file>` | (none) | Write response to file |
| `-j, --json` | off | Structured JSON on stdout |
| `--stream/--no-stream` | off | Streaming (off by default in query) |
| `--reasoning` | off | Show agent reasoning |
| `-v, --verbose` | off | Debug output |
| `-p, --provider <name>` | config | Override LLM provider |
| `-m, --model <name>` | config | Override model |

## `lobster command` (Programmatic Access)

Execute slash commands **without starting an LLM session**. No API keys needed. ~300ms.

```bash
lobster command data --json                          # Current dataset info
lobster command "workspace list" --json              # List datasets
lobster command files --json                         # List workspace files
lobster command "pipeline export" --session-id proj  # Export notebook
lobster command modalities --json                    # Modality details
```

Leading `/` is stripped automatically (`lobster command /data` = `lobster command data`).

**JSON output schema**:
```json
{
  "success": true,
  "command": "workspace list",
  "data": {
    "tables": [{"title": "...", "columns": ["..."], "rows": [["..."]]}],
    "messages": [{"text": "...", "style": "info"}]
  },
  "summary": "Listed 3 available datasets"
}
```

## Slash Commands (Interactive & `lobster command`)

### Data & Files

| Command | Description |
|---------|-------------|
| `/data` | Current dataset info (shape, columns, stats) |
| `/files` | List workspace files by category |
| `/tree` | Directory tree view |
| `/read <file>` | View file contents (pattern support: `*.json`) |
| `/open <file>` | Open in system default app |
| `/plots` | List generated visualizations |
| `/modalities` | Detailed modality information |
| `/describe <name>` | Statistical summary of a modality |

### Workspace Management

| Command | Description |
|---------|-------------|
| `/workspace` | Workspace status |
| `/workspace list` | List available datasets with index numbers |
| `/workspace info <sel>` | Dataset details by index or name |
| `/workspace load <sel>` | Load dataset by index, pattern, or file path |
| `/workspace remove <sel>` | Remove modality |
| `/workspace save` | Save modalities to workspace |
| `/restore` | Restore recent datasets from session |
| `/restore all` | Restore all available datasets |

### Queue

| Command | Description |
|---------|-------------|
| `/queue` | Download and publication queue status |
| `/queue list [type]` | List queued items (`publication`, `download`) |
| `/queue clear [type]` | Clear queue entries |
| `/queue export` | Export queue to CSV |
| `/queue load <file>` | Load file into queue (supports .ris) |

### Metadata

| Command | Description |
|---------|-------------|
| `/metadata` | Smart metadata overview |
| `/metadata publications` | Publication queue breakdown |
| `/metadata samples` | Sample statistics and disease coverage |
| `/metadata workspace` | File inventory across storage locations |
| `/metadata exports` | Export files with categories |
| `/metadata list` | Detailed metadata list |
| `/metadata clear` | Clear metadata entries |

### Pipeline

| Command | Description |
|---------|-------------|
| `/pipeline list` | List available notebooks |
| `/pipeline info` | Notebook details |
| `/pipeline export` | Export reproducible Jupyter notebook (needs `--session-id`) |
| `/pipeline run` | Run exported notebook (needs `--session-id`) |

### Session & State

| Command | Description |
|---------|-------------|
| `/session` | Current session info (ID, messages, data) |
| `/save [--force]` | Save all modalities to workspace |
| `/export [--no-png]` | Export session data + plots to `exports/` |
| `/clear` | Clear conversation history |
| `/reset` | Reset conversation (retains loaded data) |
| `/status` | Subscription tier, packages, agents |
| `/tokens` | Token usage and costs |

### Configuration (Runtime)

| Command | Description |
|---------|-------------|
| `/config` | Show current configuration |
| `/config show` | Show provider, model, config files |
| `/config provider list` | List available LLM providers |
| `/config provider <name>` | Switch provider at runtime |
| `/config provider <name> --save` | Switch and persist to config |
| `/config model list` | List available models |
| `/config model <name>` | Switch model at runtime |
| `/config model <name> --save` | Switch and persist |

### Config CLI Subcommands

```bash
lobster config show           # Display current config
lobster config test           # Test LLM connectivity
lobster config list-models    # List model presets
lobster config list-profiles  # List testing profiles
lobster config show-config    # Full runtime config
lobster config create-custom  # Interactive custom config
lobster config models         # Per-agent model config
lobster config generate-env   # Generate .env template
```

## Session Management

```bash
# Named session (persisted to disk)
lobster query --session-id "cancer_project" "Load data"
lobster query --session-id "cancer_project" "Run QC"

# Continue most recent session
lobster query --session-id latest "Next step"

# Cross-session pipeline export
lobster command "pipeline export" --session-id cancer_project
```

**Workspace isolation**: different `-w` paths = separate sessions.

## LLM Providers

7 providers supported:

| Provider | Init flag | Env var |
|----------|-----------|---------|
| Anthropic | `--anthropic-key` | `ANTHROPIC_API_KEY` |
| AWS Bedrock | `--bedrock-access-key` + `--bedrock-secret-key` | `AWS_BEDROCK_ACCESS_KEY` |
| Google Gemini | `--gemini-key` | `GOOGLE_API_KEY` |
| OpenAI | `--openai-key` | `OPENAI_API_KEY` |
| Ollama (local) | `--use-ollama` | -- |
| OpenRouter | `--openrouter-key` | `OPENROUTER_API_KEY` |
| Azure AI | `--azure-endpoint` + `--azure-key` | `AZURE_ENDPOINT` |

## Keyboard Shortcuts (Interactive)

| Shortcut | Action |
|----------|--------|
| Ctrl+C | Cancel current operation |
| Ctrl+D | Exit |
| Ctrl+L | Clear screen |
| Ctrl+R | Search command history |
| Tab | Autocomplete commands, files, datasets |
