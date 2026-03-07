# Agent Orchestration Patterns

How coding agents (Claude Code, Gemini CLI, Codex, OpenClaw) use Lobster AI programmatically
to offload bioinformatics analysis.

## Core Pattern

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

**Success response**:
```json
{
  "success": true,
  "response": "Analysis complete. Found 15,234 cells...",
  "session_id": "proj",
  "last_agent": "transcriptomics_expert",
  "token_usage": { ... }
}
```

**Error response**:
```json
{
  "success": false,
  "response": "",
  "session_id": "proj",
  "error": "No modality loaded. Load data first."
}
```

**Fields**: `success` and `response` always present. `last_agent` and `token_usage` only on success.
`workspace` is NOT included in the JSON output — track it yourself from `-w`.

Parse with: `jq -r '.response'`

## Workspace Inspection (No Tokens)

Use `lobster command --json` for fast inspection (~300ms, no LLM, no API keys):

```bash
# What data is loaded?
lobster command data --json -w .lobster_workspace | jq '.data'

# List workspace files
lobster command files --json -w .lobster_workspace | jq '.data.tables'

# List available datasets
lobster command "workspace list" --json | jq '.data.tables[0].rows'

# Modality details
lobster command modalities --json | jq '.data'

# Check system status
lobster config-test --json
```

**Rule**: Always use `lobster command` for inspection. Only use `lobster query` for
analysis that requires the agent system.

## Multi-Step Analysis Pattern

A typical orchestrated pipeline:

```bash
SESSION="liver_study"
WORKSPACE="./liver_analysis"

# IMPORTANT: Always pass BOTH -w and --session-id on EVERY call.
# Omitting -w causes workspace to default to .lobster_workspace/ and lose context.

# Step 1: Search (Research Agent -- online)
lobster query -w "$WORKSPACE" --session-id "$SESSION" --json \
  "Search PubMed for liver fibrosis scRNA-seq datasets from 2023-2024"

# Step 2: Download (Data Expert -- executes queued download)
lobster query -w "$WORKSPACE" --session-id "$SESSION" --json \
  "Download the top dataset"

# Step 3: Verify data loaded before proceeding
DATA=$(lobster command data --json -w "$WORKSPACE")
if ! echo "$DATA" | jq -e '.success' > /dev/null 2>&1; then
  echo "No data loaded yet -- check download status"
  lobster command "queue list" --json -w "$WORKSPACE"
  exit 1
fi

# Step 4: Analyze (Transcriptomics Expert)
lobster query -w "$WORKSPACE" --session-id "$SESSION" --json \
  "Run QC, filter low-quality cells, normalize, and cluster"

# Step 5: Annotate (Annotation Expert -- child of Transcriptomics)
lobster query -w "$WORKSPACE" --session-id "$SESSION" --json \
  "Identify cell types in each cluster"

# Step 6: DE (DE Analysis Expert -- child of Transcriptomics)
lobster query -w "$WORKSPACE" --session-id "$SESSION" --json \
  "Find differentially expressed genes between hepatocytes and stellate cells"

# Step 7: Visualize (Visualization Expert)
lobster query -w "$WORKSPACE" --session-id "$SESSION" --json \
  "Create UMAP colored by cell type and export marker genes to CSV"

# Step 8: Check outputs
lobster command files --json -w "$WORKSPACE"
```

## Non-Interactive Setup

Before first use, initialize Lobster without user interaction.
**Always pass API keys via environment variables** — never hardcode secrets.

```bash
# Check if already configured
if ! lobster config-test --json 2>/dev/null | jq -e '.success' > /dev/null 2>&1; then
  # Requires ANTHROPIC_API_KEY set in environment
  lobster init --non-interactive --anthropic-key "$ANTHROPIC_API_KEY" --profile production
fi
```

For all provider options, see [cli-reference.md](cli-reference.md) "Initialization" section.

## Output File Discovery

Lobster writes outputs to the workspace. When you pass `-w ./my_analysis`, the workspace
root is `./my_analysis/.lobster_workspace/`. When you omit `-w`, the default is
`.lobster_workspace/` in the current directory.

```bash
# Use lobster command to discover outputs (preferred -- no tokens)
lobster command files --json -w "$WORKSPACE"
lobster command plots --json -w "$WORKSPACE"

# Or find files directly
WS_DIR="${WORKSPACE}/.lobster_workspace"
find "$WS_DIR" -name "*.h5ad"   # Processed AnnData objects
find "$WS_DIR" -name "*.html"   # Interactive Plotly visualizations
find "$WS_DIR" -name "*.png"    # Publication-ready plots
find "$WS_DIR" -name "*.csv"    # Exported tables
find "$WS_DIR" -name "*.ipynb"  # Reproducible notebooks
```

## Pipeline Export

Export a completed analysis as a reproducible Jupyter notebook:

```bash
# Export from a named session
lobster command "pipeline export" --session-id "$SESSION"

# The notebook appears in the workspace
ls "$WORKSPACE"/*.ipynb
```

## Error Handling

Check the `success` field in JSON output. **Do NOT redirect stderr** — it contains
diagnostic information needed for debugging.

```bash
# Capture stdout (JSON) and stderr (diagnostics) separately
RESULT=$(lobster query -w "$WORKSPACE" --session-id "$SESSION" --json "Run clustering" 2>lobster_stderr.log)
SUCCESS=$(echo "$RESULT" | jq -r '.success // false')

if [ "$SUCCESS" != "true" ]; then
  echo "Analysis failed"
  echo "$RESULT" | jq -r '.error // .response // "Unknown error"'
  cat lobster_stderr.log  # Check for Python tracebacks or startup errors
  exit 1
fi
```

**If `jq` is not available**, use Python as fallback:
```bash
echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('response',''))"
```

Common failure patterns:
- **No data loaded**: Agent ran analysis before loading data. Use `lobster command data --json -w "$WORKSPACE"` to verify.
- **Session not found**: Missing `--session-id` on a follow-up query. Data from previous queries is lost.
- **Workspace mismatch**: Omitting `-w` on a follow-up causes default workspace, losing context. Always pass `-w`.
- **Agent not available**: Package not installed. Check with `lobster agents list`.
- **Config error**: Run `lobster config-test --json` to diagnose.
- **Download pending**: Data queued but not yet downloaded. Check `lobster command "queue list" --json -w "$WORKSPACE"`.

## Combining Lobster with Other Tools

Lobster handles the bioinformatics. The coding agent handles everything else:

```bash
# Extract specific values from Lobster's response
CELL_COUNT=$(lobster query --session-id "$SESSION" --json \
  "How many cells passed QC?" | jq -r '.response' | grep -oP '\d+(?= cells)')

# Use Lobster output files in downstream scripts
lobster query --session-id "$SESSION" --json "Export DE results to CSV"
RESULTS_CSV=$(find .lobster_workspace -name "*de_results*" -name "*.csv" | head -1)
python my_custom_analysis.py "$RESULTS_CSV"

# Feed external data into Lobster
lobster query -w ./analysis --session-id "ext" --json \
  "Load the data from /path/to/external_data.h5ad"
```

## Agent Routing

You don't need to specify which agent to use. Describe the task in natural language
and Lobster routes automatically. Key routing rules:

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

1. **Always name sessions**: `--session-id "descriptive_name"` not random IDs
2. **One session per analysis**: Don't mix unrelated analyses in one session
3. **Check before continuing**: `lobster command data --json` after resuming
4. **Use `latest` sparingly**: Prefer explicit session names for reproducibility
5. **Workspace + session together**: `-w ./project --session-id "run_01"`
