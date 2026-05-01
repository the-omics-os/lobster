# Agent Orchestration Patterns

How coding agents (Claude Code, Gemini CLI, Codex, OpenClaw) use Lobster AI programmatically
to offload bioinformatics analysis.

## Local vs Cloud

| Factor | Local (`lobster query`) | Cloud (`lobster cloud chat`) |
|--------|------------------------|-------------------------------|
| Setup | `lobster init` + LLM key | `lobster cloud login` (one-time) |
| Sessions | `--session-id <name>` (disk) | Interactive TUI (server-side sessions) |
| Workspace | `-w <path>` (local files) | Cloud workspace (server-side) |
| Inspection | `lobster command --json` | Inside TUI |
| Offline | Yes | No |

**Decision rule**: If user has Omics-OS Cloud credentials at `~/.config/omics-os/credentials.json`,
prefer cloud. Otherwise use local.

**Note**: Cloud mode is interactive only (`lobster cloud chat`). For programmatic cloud
queries, use the `@omicsos/lobster` npm CLI directly.

## Core Pattern — Local

The agent calls `lobster query --json --session-id` and parses structured output:

```bash
RESULT=$(lobster query --session-id "proj" --json "Your analysis request")
echo "$RESULT" | jq -r '.response'
```

**Key flags for agents**:
- `--json` -- structured JSON on stdout (Rich UI goes to stderr)
- `--session-id <id>` -- data persists across queries (REQUIRED for multi-step)
- `-w <path>` -- explicit workspace path
- `--no-stream` -- default for `query`, returns complete result

## JSON Output Schema

### Local Success

```json
{
  "success": true,
  "response": "Analysis complete. Found 15,234 cells...",
  "session_id": "proj",
  "last_agent": "transcriptomics_expert",
  "token_usage": { ... }
}
```

### Error

```json
{
  "success": false,
  "error": "Not authenticated. Run 'lobster cloud login' first.",
  "session_id": null
}
```

**Parse with**: `jq -r 'if .success then .response else (.error // .response // "Unknown error") end'`

**Note**: Not all failures include `.error` — some set `.success: false` with the error in `.response`. Always check `.success` first, then fall through `.error // .response // .finish_reason`.

## Workspace Inspection

### Local (No Tokens, ~300ms)

Use `lobster command --json` for fast inspection:

```bash
lobster command data --json -w .lobster_workspace | jq '.data'
lobster command files --json -w .lobster_workspace | jq '.data.tables'
lobster command "workspace list" --json | jq '.data.tables[0].rows'
lobster command modalities --json | jq '.data'
lobster config-test --json
```

**Rule**: Always use `lobster command` for inspection. Only use `lobster query` for
analysis that requires the agent system.

### Cloud

Cloud workspace files are managed within the interactive TUI (`lobster cloud chat`).
```bash
lobster cloud status          # Tier, budget, usage
```

## Multi-Step Analysis — Local

```bash
SESSION="liver_study"
WORKSPACE="./liver_analysis"

# Step 1: Search (Research Agent -- online)
lobster query -w "$WORKSPACE" --session-id "$SESSION" --json \
  "Search PubMed for liver fibrosis scRNA-seq datasets from 2023-2024"

# Step 2: Download (Data Expert)
lobster query -w "$WORKSPACE" --session-id "$SESSION" --json \
  "Download the top dataset"

# Step 3: Verify data loaded
DATA=$(lobster command data --json -w "$WORKSPACE")
if ! echo "$DATA" | jq -e '.success' > /dev/null 2>&1; then
  echo "No data loaded yet"
  lobster command "queue list" --json -w "$WORKSPACE"
  exit 1
fi

# Step 4: Analyze
lobster query -w "$WORKSPACE" --session-id "$SESSION" --json \
  "Run QC, filter low-quality cells, normalize, and cluster"

# Step 5: Annotate
lobster query -w "$WORKSPACE" --session-id "$SESSION" --json \
  "Identify cell types in each cluster"

# Step 6: DE
lobster query -w "$WORKSPACE" --session-id "$SESSION" --json \
  "Find DE genes between hepatocytes and stellate cells"

# Step 7: Visualize
lobster query -w "$WORKSPACE" --session-id "$SESSION" --json \
  "Create UMAP colored by cell type and export markers to CSV"

# Step 8: Check outputs
lobster command files --json -w "$WORKSPACE"
```

## Non-Interactive Setup

### Local

```bash
if ! lobster config-test --json 2>/dev/null | jq -e '.success' > /dev/null 2>&1; then
  lobster init --non-interactive --anthropic-key "$ANTHROPIC_API_KEY" --profile production
fi
```

### Cloud

```bash
# Check if already authenticated
if ! lobster cloud status 2>/dev/null | grep -q "Tier"; then
  echo "Not authenticated. User must run: lobster cloud login"
  exit 1
fi
```

Cloud login requires a browser (OAuth flow) or interactive API key paste.
Coding agents cannot perform initial cloud login — tell the user to run it manually.

## Output File Discovery

### Local

```bash
lobster command files --json -w "$WORKSPACE"
lobster command plots --json -w "$WORKSPACE"

WS_DIR="${WORKSPACE}/.lobster_workspace"
find "$WS_DIR" -name "*.h5ad"   # Processed AnnData objects
find "$WS_DIR" -name "*.html"   # Interactive Plotly visualizations
find "$WS_DIR" -name "*.png"    # Publication-ready plots
find "$WS_DIR" -name "*.csv"    # Exported tables
find "$WS_DIR" -name "*.ipynb"  # Reproducible notebooks
```

## Pipeline Export (Local Only)

```bash
lobster command "pipeline export" --session-id "$SESSION"
ls "$WORKSPACE"/*.ipynb
```

## Error Handling

Check the `success` field. **Do NOT redirect stderr** — it contains diagnostics.

```bash
RESULT=$(lobster query -w "$WORKSPACE" --session-id "$SESSION" --json "Run clustering" 2>lobster_stderr.log)

SUCCESS=$(echo "$RESULT" | jq -r '.success // false')
if [ "$SUCCESS" != "true" ]; then
  echo "Failed"
  echo "$RESULT" | jq -r '.error // .response // "Unknown error"'
  cat lobster_stderr.log
  exit 1
fi
```

**If `jq` is not available**:
```bash
echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('response',''))"
```

Common failure patterns:

| Error | Mode | Cause | Fix |
|-------|------|-------|-----|
| No data loaded | Local | Analysis before loading | `lobster command data --json` to verify |
| Session not found | Local | Missing `--session-id` | Always pass `--session-id` for multi-step |
| Workspace mismatch | Local | Omitting `-w` on follow-up | Always pass `-w` |
| Not authenticated | Cloud | No stored credentials | `lobster cloud login` |
| Cloud TUI not installed | Cloud | npm binary missing | `npm install -g @omicsos/lobster` |
| Agent not available | Both | Package not installed | `lobster agents list` |
| Config error | Local | Provider misconfigured | `lobster config-test --json` |

## Agent Routing

Describe the task in natural language — Lobster routes automatically:

- **Literature/dataset search** -> Research Agent (ONLY agent with internet)
- **File loading/downloading** -> Data Expert (offline only)
- **scRNA-seq / bulk RNA-seq** -> Transcriptomics Expert
- **Proteomics (any platform)** -> Proteomics Expert
- **Metabolomics** -> Metabolomics Expert
- **VCF / PLINK / GWAS** -> Genomics Expert
- **Plots and figures** -> Visualization Expert
- **ML, feature selection** -> ML Expert
- **Drug targets, compounds** -> Drug Discovery Expert

## Session Continuity Best Practices

### Local
1. **Always name sessions**: `--session-id "descriptive_name"` not random IDs
2. **One session per analysis**: Don't mix unrelated analyses
3. **Check before continuing**: `lobster command data --json` after resuming
4. **Workspace + session together**: `-w ./project --session-id "run_01"`

### Cloud
1. **Use `lobster cloud chat`** for interactive sessions — managed in the npm TUI
2. **Sessions are cross-device**: Same session accessible from CLI and web app
