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
| One-line installer | `curl -fsSL https://install.lobsterbio.com \| bash` |
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
lobster config-test --json    # Structured status (local)
lobster status                # Human-readable
lobster cloud status          # Cloud tier, usage, budget
```

**Config file locations**:

| Mode | Config | Credentials |
|------|--------|-------------|
| Workspace (default) | `.lobster_workspace/provider_config.json` | `.env` |
| Global (`--global`) | `~/.config/lobster/providers.json` | `~/.config/lobster/credentials.env` |
| Cloud | — | `~/.config/omics-os/credentials.json` |

**Priority**: workspace `.env` > global `credentials.env` > environment variables

## Top-Level Commands

| Command | Description |
|---------|-------------|
| `lobster chat` | Interactive chat (Ink TUI or Go TUI) |
| `lobster query "..."` | Single-turn query (local agents) |
| `lobster cloud <cmd>` | Omics-OS Cloud commands (see below) |
| `lobster command <cmd>` | Execute slash command without LLM (~300ms) |
| `lobster init` | Configuration wizard |
| `lobster status` | Tier, packages, agents |
| `lobster config-test` | Test API connectivity |
| `lobster agents list` | List installed agent packages |
| `lobster agents info <name>` | Agent package details |
| `lobster serve --port 8080` | Start API server |
| `lobster dashboard` | Visual monitoring UI |
| `lobster purge` | Remove Lobster files (`--dry-run`, `--force`) |

## `lobster cloud` — Omics-OS Cloud

Cloud commands connect to Omics-OS Cloud at `app.omics-os.com`. Agents run on ECS Fargate with managed Bedrock. No local LLM keys needed.

### Authentication

```bash
lobster cloud login                            # Browser OAuth (opens browser, stores credentials)
lobster cloud login --api-key "$OMICS_OS_API_KEY"  # Headless/SSH/CI (pass value, not flag alone)
lobster cloud logout                           # Clear stored credentials
```

Credentials stored at `~/.config/omics-os/credentials.json`. OAuth tokens auto-refresh.

**Cloud auth env vars** (alternative to stored credentials):
- `OMICS_OS_API_KEY` — used by `lobster cloud query` (Python CLI)
- `LOBSTER_TOKEN` — used by Ink TUI (`lobster cloud chat`), highest priority
- `--token` flag — explicit override (warning: visible in `ps` output)

**Note**: Agents should always use `lobster cloud login` or `lobster cloud query` — never invoke the `lobster-chat` binary directly (it's an internal implementation detail).

### Account & Usage

```bash
lobster cloud status             # Tier, token usage, budget remaining
lobster cloud account            # Email, tier, user ID, auth mode, endpoint
lobster cloud keys               # Manage API keys (web link)
```

### Projects

```bash
lobster cloud projects           # List projects (table with full UUIDs)
lobster cloud projects --json    # Machine-readable JSON
```

Use project IDs from this output with `--project-id` on `query` and `chat`.

### `lobster cloud query`

Single-turn cloud query. Agents run on ECS Fargate. Returns when complete.

```bash
lobster cloud query "Search PubMed for CRISPR"
lobster cloud query "Analyze my data" --json
lobster cloud query "Continue analysis" --session-id latest --json
lobster cloud query "Run QC" --project-id <UUID> --json
lobster cloud query "Quick question" --stream   # Stream text as it arrives
```

| Flag | Default | Description |
|------|---------|-------------|
| `--session-id, -s <id>` | (new session) | Resume session (UUID or `latest`) |
| `--project-id, -p <UUID>` | (none) | Associate with project (new sessions only — ignored on resume) |
| `--json, -j` | off | Structured JSON on stdout (`--stream` has no effect with `--json`) |
| `--stream / --no-stream` | off | Stream text as it arrives |
| `--token` | stored | Override auth token (visible in ps — prefer env var) |
| `--endpoint` | app.omics-os.com | Custom REST origin (allowlisted hosts only) |
| `--stream-endpoint` | stream.omics-os.com | Custom stream origin (allowlisted hosts only) |

**JSON output schema**:
```json
{
  "success": true,
  "response": "Analysis complete...",
  "session_id": "dfcf2a08-adcb-4a8f-8d94-43c9dc494190",
  "active_agent": "transcriptomics_expert",
  "token_usage": { "total_cost_usd": 0.0159 },
  "session_title": "CRISPR Analysis",
  "finish_reason": null,
  "workspace_files": [{"name": "results.csv", "size": 1024}]
}
```

**Error JSON**: `{"success": false, "error": "...", "session_id": null}`

### `lobster cloud chat`

Interactive cloud chat via Ink TUI. Direct connection to Omics-OS Cloud.

```bash
lobster cloud chat                              # New session
lobster cloud chat --session-id <UUID>          # Resume session
lobster cloud chat --session-id latest          # Resume most recent
lobster cloud chat --project-id <UUID>          # Associate with project
```

| Flag | Default | Description |
|------|---------|-------------|
| `--session-id, -s <id>` | (new session) | Resume existing session |
| `--project-id, -p <UUID>` | (none) | Associate with a cloud project |
| `--token` | stored | Override auth token |
| `--endpoint` | app.omics-os.com/api/v1 | Custom API endpoint |

**Inside cloud chat**: `/sessions` lists sessions, `/cloud account` shows account info.

**Ctrl+C**: Cancels both client stream and server-side run (POSTs `/chat/cancel`).

**Session continuity**: Same sessions accessible from both CLI and web app at `app.omics-os.com`.

## `lobster chat` (Local)

Interactive mode. **Ink TUI** (React Ink) is the primary surface; Go TUI is the fallback.

```bash
lobster chat                        # Default (Ink TUI auto-detected)
lobster chat --ui ink               # Force React Ink TUI
lobster chat --ui go                # Force Go/Charm TUI
lobster chat --ui classic           # Force legacy Rich/Textual mode
lobster chat --classic              # Shorthand for --ui classic
lobster chat --no-intro             # Skip welcome animation
lobster chat -w ./my_analysis       # Set workspace
lobster chat -s "my_session"        # Continue named session
lobster chat --reasoning            # Show agent reasoning
```

| Flag | Default | Description |
|------|---------|-------------|
| `-w, --workspace <path>` | auto | Workspace directory |
| `-s, --session-id <id>` | (none) | Named session to continue |
| `--ui <mode>` | auto | `ink`, `go`, `classic`, or `auto` |
| `--classic` | -- | Shorthand for `--ui classic` |
| `--no-intro` | off | Skip intro animation |
| `--reasoning` | off | Show agent reasoning |
| `--stream/--no-stream` | on | Streaming (on by default in chat) |
| `-v, --verbose` | off | Debug output |
| `-p, --provider <name>` | config | Override LLM provider |
| `-m, --model <name>` | config | Override model |

## `lobster query` (Local)

Single-turn queries against local agents. Returns complete result by default.

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

Execute slash commands **without starting an LLM session**. No API keys needed. ~300ms. **Local mode only.**

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
| `/sessions` | List cloud sessions (cloud chat only) |
| `/cloud account` | Cloud account info (cloud chat only) |
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

### Local Sessions

```bash
lobster query --session-id "cancer_project" "Load data"
lobster query --session-id "cancer_project" "Run QC"
lobster query --session-id latest "Next step"
lobster command "pipeline export" --session-id cancer_project
```

**Workspace isolation**: different `-w` paths = separate sessions.

### Cloud Sessions

```bash
lobster cloud query "Start analysis" --json
# Returns session_id: "dfcf2a08-..."

lobster cloud query "Continue" --session-id latest --json
# Resolves to most recent session

lobster cloud query "With project" --project-id <UUID> --json
# Associates session with a cloud project

lobster cloud chat --session-id <UUID>
# Resume in interactive mode
```

Cloud sessions persist server-side. Same session accessible from CLI and web app.

## LLM Providers (Local Mode)

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
| Ctrl+C | Cancel current operation (cloud: also cancels server-side run) |
| Ctrl+D | Exit |
| Ctrl+L | Clear screen |
| Ctrl+R | Search command history |
| Tab | Autocomplete commands, files, datasets |
