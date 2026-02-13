# CLI Commands Reference

Complete reference for Lobster AI command-line interface.

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
# Start named session
lobster query --session-id "cancer_project" "Load data"

# Continue with latest
lobster query --session-id latest "Next step"

# Or use specific session
lobster query --session-id "cancer_project" "Continue analysis"
```

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
# Default (interactive output)
lobster query "Analyze data"

# Save to file
lobster query --output results.md "Generate report"

# JSON output (for scripting)
lobster query --format json "Get statistics"
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
