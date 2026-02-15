# CLI Commands Reference

Complete reference for Lobster AI command-line interface.

> **Agent note:** For workspace inspection, prefer `lobster command <cmd> --json` — it's fast
> (~300ms), needs no LLM/API keys, and returns structured JSON. Use `lobster query` only for
> analysis tasks that require the agent system.
> See [Programmatic Command Access](#programmatic-command-access) below.

## Installation & Setup

### Install (new users)

| Platform | Command |
|----------|---------|
| macOS / Linux | `curl -fsSL https://install.lobsterbio.com \| bash` |
| Windows (PowerShell) | `irm https://install.lobsterbio.com/windows \| iex` |
| Manual (any) | `uv tool install 'lobster-ai[full,anthropic]' && lobster init` |
| pip (any) | `pip install 'lobster-ai[full]' && lobster init` |

### Upgrade

| Method | Command |
|--------|---------|
| uv tool | `uv tool upgrade lobster-ai` |
| pip | `pip install --upgrade lobster-ai` |

### Add agent packages (uv tool installs)

```bash
uv tool install lobster-ai --with lobster-proteomics --with lobster-genomics
```

Or run `lobster init` to interactively select agents and generate the command.

### Configure / reconfigure

```bash
lobster init                    # Workspace setup (interactive)
lobster init --global           # Global defaults
lobster init --force            # Overwrite existing config
```

## Starting Lobster

```bash
# Interactive chat (primary mode)
lobster chat
lobster chat --workspace /path/to/project
lobster chat --reasoning              # Detailed agent reasoning
lobster chat --verbose                # Debug output

# Single query
lobster query "Your request"
lobster query --session-id latest "Follow-up"
lobster query --output results.md "Generate report"

# Dashboard (visual monitoring)
lobster dashboard

# API server (for web interfaces)
lobster serve --port 8080
```

## System Commands

### Information

| Command | Description |
|---------|-------------|
| `/help` | All available commands |
| `/status` | Installation status, subscription tier, available agents |
| `/session` | Current session info (ID, messages, data loaded) |
| `/input-features` | Show input capabilities (tab completion, history) |

### Workspace Management

| Command | Description |
|---------|-------------|
| `/workspace` | Show workspace info and loaded modalities |
| `/workspace list` | List all datasets with index numbers |
| `/workspace info <#>` | Detailed info for dataset by index |
| `/workspace load <#>` | Load dataset by index or name |
| `/restore` | Restore recent datasets from session |
| `/restore all` | Restore all available datasets |

**Index-based loading** (fast):
```
/workspace list              # Shows: #1, #2, #3...
/workspace load 1            # Load first dataset
/workspace info 3            # Details for third dataset
```

### File Operations

| Command | Description |
|---------|-------------|
| `/files` | List all files organized by category |
| `/tree` | Directory tree view |
| `/read <file>` | View file contents (inspection only) |
| `/open <file>` | Open in system default app |
| `/archive <file>` | Load from compressed archive |

**Pattern support**:
```
/read *.json                 # All JSON files
/read results_*              # Files starting with results_
```

### Data Commands

| Command | Description |
|---------|-------------|
| `/data` | Current dataset info (shape, columns, stats) |
| `/plots` | List generated visualizations |
| `/save` | Save current session state |
| `/clear` | Clear conversation history |

### Analysis Commands

| Command | Description |
|---------|-------------|
| `/describe` | Statistical summary of current data |
| `/compare <g1> <g2>` | Quick comparison between groups |

## Session Management

### Session IDs

```bash
# Start named session (provenance persisted to disk)
lobster query --session-id "cancer_project" "Load data"

# Continue with latest session
lobster query --session-id latest "Next step"

# Or use specific session
lobster query --session-id "cancer_project" "Continue analysis"
```

### Cross-Session Pipeline Export (v1.0.7+)

Provenance persists to disk when using `--session-id`. Export notebooks from any terminal:

```bash
# Day 1: Run analysis
lobster query --session-id liver_study "Download GSE109564 and run QC"
lobster query --session-id liver_study "Cluster and find markers"

# Day 2 (new terminal): Export the pipeline as a notebook
lobster command "pipeline export" --session-id liver_study
```

**When to use `--session-id`:**
- After closing terminal, to continue or export a previous analysis
- For scripted/CI pipelines that export notebooks
- When sharing reproducible workflows between sessions

**Without `--session-id` after restart:**
Pipeline export will show a guidance message — use `--session-id` to load provenance.

### Workspace Isolation

Different workspaces = isolated sessions:
```bash
lobster chat --workspace ./project-a    # Session A
lobster chat --workspace ./project-b    # Session B (separate)
```

## Enhanced Input Features

**Requires**: `pip install prompt-toolkit`

| Feature | Keys |
|---------|------|
| Navigate text | ← → |
| Command history | ↑ ↓ |
| Search history | Ctrl+R |
| Tab completion | Tab |
| Jump to start/end | Home/End |

**Tab completion works for**:
- Commands: `/` + Tab
- Files: `/read` + Tab
- Datasets: `/workspace load` + Tab

## Output Formats

```bash
# Default (interactive Rich output)
lobster query "Analyze data"

# Save to file
lobster query --output results.md "Generate report"

# JSON output (for programmatic/agent consumption)
lobster query --json "Get statistics"
# Returns: {"success": true, "response": "...", "session_id": "...", ...}
# All Rich UI output goes to stderr; only JSON on stdout
# Pipe-friendly: lobster query --json "..." | jq .response
```

## Configuration Commands

```bash
# Test configuration
lobster config-test --json

# Show current config
lobster config-show

# Check available agents
lobster status
```

## Keyboard Shortcuts (Interactive Mode)

| Shortcut | Action |
|----------|--------|
| Ctrl+C | Cancel current operation |
| Ctrl+D | Exit (same as /quit) |
| Ctrl+L | Clear screen |
| Ctrl+R | Search command history |

## Programmatic Command Access

`lobster command` executes workspace slash commands **without starting an LLM session**.
No API keys needed. Returns in ~300ms.

```bash
# Console mode (Rich tables, same output as /data in chat)
lobster command data
lobster command "workspace list"
lobster command files

# JSON mode (structured, for agents and scripts)
lobster command data --json
lobster command "workspace list" --json | jq '.data.tables[0].rows'
lobster command "metadata publications" --json -w ./my_project

# With session loading (for pipeline export across sessions)
lobster command "pipeline export" --session-id my_experiment
lobster command "pipeline export" --session-id latest

# Leading / is stripped automatically
lobster command /data --json
```

**`--session-id` flag** (v1.0.7+): Loads provenance from a previous session, enabling
pipeline export and run from `lobster command`. Without it, pipeline commands require
a live `lobster chat` session.

### Available Commands

| Command | Description |
|---------|-------------|
| **Data & Files** | |
| `data` | Current dataset info (shape, columns, stats) |
| `files` | List workspace files organized by category |
| `read <file>` | View file contents (text, code, data preview) |
| `plots` | List generated visualizations |
| `modalities` | Show detailed modality information |
| `describe <name>` | Statistical summary of a modality |
| **Workspace** | |
| `workspace` | Show workspace status |
| `workspace list` | List all datasets with status |
| `workspace info <sel>` | Detailed info for a dataset by index or name |
| `workspace load <sel>` | Load a dataset by index, pattern, or file path |
| `workspace remove <sel>` | Remove a dataset from workspace |
| **Queue** | |
| `queue` | Download and publication queue status |
| `queue list [type]` | List queued items (default: publication) |
| `queue clear [type]` | Clear queue entries |
| `queue export [type]` | Export queue to CSV |
| **Metadata** | |
| `metadata` | Smart metadata overview |
| `metadata publications` | Publication queue breakdown |
| `metadata samples` | Sample statistics and disease coverage |
| `metadata workspace` | File inventory across storage locations |
| `metadata exports` | Export files with categories |
| `metadata list` | Detailed metadata list |
| `metadata clear` | Clear metadata entries |
| **Pipeline** | |
| `pipeline list` | List available notebooks |
| `pipeline info` | Notebook details |
| `pipeline export` | Export reproducible Jupyter notebook (requires `--session-id`) |
| `pipeline run` | Run exported notebook (requires `--session-id`) |
| **State** | |
| `save [--force]` | Save all modalities to workspace |
| `restore [pattern]` | Restore datasets from disk (default: recent) |
| `export [--no-png] [--force]` | Export session data + plots to workspace/exports/ |
| **Config** | |
| `config` | Current configuration |
| `config provider list` | List available LLM providers |
| `config model list` | List available models |

### JSON Output Schema

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

### Agent Best Practice

```bash
# Check what data is loaded (fast, no tokens burned)
lobster command data --json -w .lobster_workspace | jq '.data'

# List workspace files
lobster command files --json -w .lobster_workspace | jq '.data.tables'

# Then use lobster query only for actual analysis
lobster query --session-id latest "Cluster the cells and find markers"
```

## Examples

**Complete workflow session**:
```bash
lobster chat --workspace ./my_analysis

> /workspace list
# Shows available datasets

> /workspace load 1
# Loads first dataset

> "Run quality control"
# Natural language analysis

> /plots
# View generated plots

> "Export DE genes to CSV"
# Export results

> /save
# Save session

> /quit
```
