# Agent Orchestration Patterns

How coding agents (Claude Code, Gemini CLI, Codex, OpenClaw) use Lobster AI programmatically
to offload bioinformatics analysis.

## Local vs Cloud

| Factor | Local (`lobster query`) | Cloud (`lobster cloud query`) |
|--------|------------------------|-------------------------------|
| Setup | `lobster init` + LLM key | `lobster cloud login` (one-time) |
| Sessions | `--session-id <name>` (disk) | `--session-id <UUID>` (server) |
| Workspace | `-w <path>` (local files) | Cloud workspace (server-side) |
| Inspection | `lobster command --json` | `workspace_files` in JSON response |
| Projects | N/A | `--project-id <UUID>` |
| Offline | Yes | No |

**Decision rule**: If user has Omics-OS Cloud credentials at `~/.config/omics-os/credentials.json`,
prefer cloud. Otherwise use local.

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

## Core Pattern — Cloud

```bash
RESULT=$(lobster cloud query "Your analysis request" --json)
echo "$RESULT" | jq -r '.response'

# With session continuity
RESULT=$(lobster cloud query "Continue" --session-id latest --json)

# With project context
RESULT=$(lobster cloud query "Analyze" --project-id "$PROJECT_UUID" --json)
```

**Key flags for cloud agents**:
- `--json` -- structured JSON on stdout (one final object, no streaming deltas)
- `--session-id <UUID>` or `latest` -- session continuity (server-side)
- `--project-id <UUID>` -- associate with project (first query only — ignored on resume)

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

### Cloud Success

```json
{
  "success": true,
  "response": "Analysis complete...",
  "session_id": "dfcf2a08-adcb-4a8f-8d94-43c9dc494190",
  "active_agent": "transcriptomics_expert",
  "token_usage": { "total_cost_usd": 0.0159, "input_tokens": 5270, "output_tokens": 7 },
  "session_title": "CRISPR Analysis",
  "finish_reason": null,
  "workspace_files": [{"name": "results.csv", "size": 1024}]
}
```

### Error (both modes)

```json
{
  "success": false,
  "error": "Not authenticated. Run 'lobster cloud login' first.",
  "session_id": null
}
```

**Parse with**: `jq -r 'if .success then .response else (.error // .response // "Unknown error") end'`

**Cloud-specific fields**: `active_agent`, `session_title`, `finish_reason`, `workspace_files`

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

Cloud workspace files appear in the `workspace_files` field of JSON responses
when agents invoke tools during the query. **Caveats**:
- Best-effort only — returns `[]` on any fetch error (not a definitive "no files")
- Only populated when agents used tools (simple Q&A queries skip the fetch)
- Not equivalent to local `lobster command files` — it's a hint, not a listing
- For large outputs, ask Lobster to write files rather than relying on `.response` (10 MB cap)

```bash
# Check if agents produced files
RESULT=$(lobster cloud query "Run analysis" --json)
echo "$RESULT" | jq '.workspace_files'
```

Cloud account/usage inspection:
```bash
lobster cloud status          # Tier, budget, usage
lobster cloud projects --json # List projects with UUIDs
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

## Multi-Step Analysis — Cloud

```bash
# Step 1: Get project UUID (optional)
PROJECT=$(lobster cloud projects --json | jq -r '.projects[0].project_id')

# Step 2: Start analysis
RESULT=$(lobster cloud query "Search PubMed for CRISPR cancer datasets" \
  --project-id "$PROJECT" --json)
SID=$(echo "$RESULT" | jq -r '.session_id')
echo "Session: $SID"

# Step 3: Continue in same session
lobster cloud query "Download the top dataset" --session-id "$SID" --json

# Step 4: Analyze (session context preserved server-side)
lobster cloud query "Run QC and cluster" --session-id "$SID" --json

# Step 5: Continue with 'latest' shorthand
lobster cloud query "Identify cell types" --session-id latest --json

# Step 6: Check workspace files
RESULT=$(lobster cloud query "Export DE results to CSV" --session-id latest --json)
echo "$RESULT" | jq '.workspace_files'
```

**Key differences from local**:
- No `-w` flag — workspace is server-side
- Session IDs are UUIDs, not friendly names
- `--session-id latest` resolves to most recent session by last_activity
- `workspace_files` in JSON response shows agent output files
- No `lobster command` equivalent — use the JSON response fields

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

### Cloud

Cloud workspace files are listed in `workspace_files` of the JSON response.
Files are stored server-side. Future: `lobster cloud files download <session-id> <path>`.

## Pipeline Export (Local Only)

```bash
lobster command "pipeline export" --session-id "$SESSION"
ls "$WORKSPACE"/*.ipynb
```

## Error Handling

Check the `success` field. **Do NOT redirect stderr** — it contains diagnostics.

```bash
# Local
RESULT=$(lobster query -w "$WORKSPACE" --session-id "$SESSION" --json "Run clustering" 2>lobster_stderr.log)

# Cloud
RESULT=$(lobster cloud query "Run clustering" --session-id "$SID" --json 2>lobster_stderr.log)

# Both
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
| Budget exhausted (402) | Cloud | Monthly limit reached | `lobster cloud status` to check |
| Rate limited (429) | Cloud | Too many requests | Wait 10-15 seconds |
| Session rate limit (5/min) | Cloud | Creating sessions too fast | Reuse existing session_id |
| Invalid session ID | Cloud | Non-UUID string | Use UUID from `--json` output or `latest` |
| Invalid project ID | Cloud | Non-UUID or empty | Get UUIDs from `lobster cloud projects --json` |
| No sessions for `latest` | Cloud | No prior sessions | Omit `--session-id` to create new |
| Session not found (404) | Cloud | Deleted or expired | Start new session |
| Empty question | Cloud | Blank query string | Provide non-empty question |
| Endpoint not in allowlist | Cloud | Custom `--endpoint` rejected | Use default or `--unsafe-endpoint` |
| Network failure | Cloud | Cannot reach cloud | Check internet, retry |
| Stream timeout (600s) | Cloud | Backend still processing | Retry or break into smaller queries |
| Empty stream response | Cloud | No valid DataStream parts | Retry — may be transient |
| Cancelled by user | Cloud | Ctrl+C during query | Exit 130 + JSON error in `--json` mode |
| Projects disabled (403/404) | Cloud | Feature not enabled | Contact support or upgrade tier |
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
1. **Capture session_id from first query**: Store the UUID for follow-ups
2. **Use `latest` for sequential chains**: Only when you're the sole user
3. **Use `--project-id` for organization**: Group related sessions under projects
4. **Sessions are cross-device**: Same session accessible from CLI and web app
