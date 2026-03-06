# Manual qwen3 + Flow Cytometry Plugin Test Guide

Quick reference for manually verifying whether qwen3 models can route to and execute flow cytometry tools.

---

## 1. Install the Plugin

The best plugin (Claude WITH trial 1) lives at:
```
~/Omics-OS/lobster/.testing/skill-eval/results/fc-full-01/claude-with-flow-cytometry-t1/workspace/packages/lobster-flow_cytometry/
```

Install into the lobster dev venv:
```bash
cd ~/Omics-OS/lobster
source .venv/bin/activate
uv pip install -e .testing/skill-eval/results/fc-full-01/claude-with-flow-cytometry-t1/workspace/packages/lobster-flow_cytometry/
```

Verify:
```bash
python3 -c "from lobster.agents.flow_cytometry import AGENT_CONFIG; print(AGENT_CONFIG['name'])"
# Expected: flow_cytometry_expert
```

## 2. Reference Data

FCS file (OMIP-030 PBMC, 1.66M events, 22 params):
```
~/Omics-OS/lobster/.testing/skill-eval/reference-data/flow-cytometry/PBMC_control 1-4_cct.fcs
```

Ground truth: 1,664,607 events, 22 parameters (17 fluorescent + 4 scatter + 1 time), 17x17 spillover matrix embedded.

## 3. Run with qwen3

Ensure ollama is running (`ollama ps`). Then:

```bash
cd ~/Omics-OS/lobster

# qwen3:8b
lobster query --provider ollama --model qwen3:8b --workspace . \
  "Load the FCS file at $HOME/Omics-OS/lobster/.testing/skill-eval/reference-data/flow-cytometry/PBMC_control\ 1-4_cct.fcs and show me what is in it"

# qwen3:14b
lobster query --provider ollama --model qwen3:14b --workspace . \
  "Load the FCS file at $HOME/Omics-OS/lobster/.testing/skill-eval/reference-data/flow-cytometry/PBMC_control\ 1-4_cct.fcs and show me what is in it"
```

Use `--session-id test-8b` / `--session-id test-14b` for follow-up queries in the same session.

## 4. The 10 Test Queries

Run these sequentially in the same session. Each targets a specific AQUADIF category.

| # | Query | Expected Tool | Category |
|---|-------|--------------|----------|
| q01 | `Load the FCS file at {path} and show me what's in it` | `import_fcs_data` | IMPORT |
| q02 | `How many events are in the loaded dataset?` | `fcs_status` or similar | UTILITY |
| q03 | `What markers and channels are available in this data?` | `panel_info` or similar | UTILITY |
| q04 | `Assess the quality of this flow cytometry data` | `assess_fcs_quality` | QUALITY |
| q05 | `Apply compensation using the embedded spillover matrix` | `apply_compensation` | PREPROCESS |
| q06 | `Transform the marker intensities using arcsinh transformation` | `transform_fluorescence` | PREPROCESS |
| q07 | `Gate the live singlet cells from this dataset` | `gate_events` | FILTER |
| q08 | `Cluster the cell populations in this data` | `cluster_cells` | ANALYZE |
| q09 | `Run dimensionality reduction on the data` | `run_umap` / `run_pca` | ANALYZE |
| q10 | `Annotate the cell types based on marker expression patterns` | `annotate_cell_types` | ANNOTATE |

## 5. What to Look For

**PASS indicators:**
- Response mentions `◀ Flow Cytometry Expert` (routed to domain agent, not CODE_EXEC)
- Provenance JSONL shows tool calls matching the expected pattern
- For q01: reports 1,664,607 events, 22 parameters
- For q05: mentions 17x17 spillover/compensation matrix

**FAIL indicators:**
- Routes to `code_execution_expert` and writes raw Python instead of using tools
- No provenance JSONL created (no domain tools called)
- Produces generic text without calling any flow cytometry tools

**Check provenance after a session:**
```bash
cat ~/Omics-OS/lobster/.lobster/sessions/{session-id}/provenance.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    d = json.loads(line)
    if isinstance(d, dict):
        print(f\"tool={d.get('type','?'):35s} cats={d.get('metadata',{}).get('categories',[])}\")"
```

## 6. Control: Bedrock Sonnet

To confirm the plugin works (not a code bug), test with a frontier model:

```bash
lobster query --provider bedrock --workspace . \
  "Load the FCS file at {path} and show me what is in it"
```

Bedrock Sonnet routes correctly and calls 6 tools in a multi-step pipeline (confirmed 2026-03-03).

## 7. Available qwen3 Models on This Machine

| Model | Size | Ollama tag |
|-------|------|-----------|
| qwen3:8b | 5.2 GB | `qwen3:8b` |
| qwen3:14b | 9.3 GB | `qwen3:14b` |
| qwen3:30b-a3b | 18 GB | `qwen3:30b-a3b` |
| qwen3.5:27b | 17 GB | `qwen3.5:27b` |

The 30b-a3b and qwen3.5:27b have not been tested yet. They may cross the capability cliff.
