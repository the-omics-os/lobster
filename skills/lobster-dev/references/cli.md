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

Initialize a new workspace.

```bash
lobster init [OPTIONS] [PATH]

Options:
  --template TEMPLATE     Workspace template
```

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
| `LOBSTER_WORKSPACE` | Default workspace path | `./workspace` |
| `LOBSTER_MODEL` | Default LLM model | `gpt-4-turbo` |
| `LOBSTER_PROVIDER` | LLM provider | `openai` |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `AWS_PROFILE` | AWS profile for Bedrock | — |

## Configuration File

Located at `~/.lobster/config.yaml` or `./lobster.yaml`:

```yaml
model:
  provider: openai
  name: gpt-4-turbo
  
workspace:
  default_path: ./workspace
  
logging:
  level: INFO
  
agents:
  disabled:
    - premium_agent_name
```

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
