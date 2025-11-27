# Bug Fix Summary: Missing Handoff Tools for metadata_assistant

## Issue
**Error:** `ValueError: When providing custom handoff tools, you must provide them for all subagents. Missing handoff tools for agents '{'metadata_assistant'}'.`

**Root Cause:**
The `langgraph_supervisor.create_supervisor()` function requires that when custom handoff tools are provided, ALL agents passed to it must have corresponding handoff tools. However, `metadata_assistant` is a **child agent** (delegated to by both `data_expert_agent` and `research_agent`) and should NOT be directly accessible by the supervisor. Child agents are accessed through parent agent delegation tools, not supervisor handoff tools.

## Solution
Modified `/Users/tyo/GITHUB/omics-os/lobster/lobster/agents/graph.py` to:

1. **Track supervisor-accessible agents separately**: Created a new list `supervisor_accessible_agents` that only includes agents that should be directly accessible by the supervisor.

2. **Filter child agents**: Agents that appear in any parent's `child_agents` list are excluded from `supervisor_accessible_agents` (unless explicitly overridden via `supervisor_accessible=True` in the registry).

3. **Pass only supervisor-accessible agents to supervisor**: Changed the `create_supervisor()` call to use `supervisor_accessible_agents` instead of all agents.

## Changes Made

### Key Code Changes:

1. **Line 92**: Added `supervisor_accessible_agents = []` to track which agents should be passed to the supervisor.

2. **Lines 126-136**: Modified handoff tool creation logic:
   - When an agent is supervisor-accessible, add it to `supervisor_accessible_agents`
   - When an agent is a child agent, skip adding it to `supervisor_accessible_agents`

3. **Lines 174-177**: Updated Phase 2 (parent agent re-creation) to also update the `supervisor_accessible_agents` list when parent agents are re-created with delegation tools.

4. **Lines 185-186**: Added debug logging to show which agents are supervisor-accessible.

5. **Line 201**: Changed `create_supervisor()` call to pass `supervisor_accessible_agents` instead of `agents`.

## Verification

### Before Fix:
- Error occurred when trying to run any lobster command
- `metadata_assistant` was included in the agents list but had no handoff tool (because it's a child agent)

### After Fix:
- Graph creates successfully without errors
- Total agents created: 7
- Supervisor-accessible agents: 6 (excludes `metadata_assistant`)
- `metadata_assistant` is accessible only through `data_expert_agent` and `research_agent` delegation tools

### Graph Structure:
```
Supervisor (has handoff tools for):
├── data_expert_agent (has delegation tool for)
│   └── metadata_assistant
├── research_agent (has delegation tool for)
│   └── metadata_assistant
├── singlecell_expert_agent
├── machine_learning_expert_agent
├── visualization_expert_agent
└── protein_structure_visualization_expert_agent
```

## Architecture Notes

The fix maintains the intended hierarchical agent structure:
- **Supervisor** can directly handoff to top-level agents
- **Top-level agents** can delegate to their child agents via delegation tools
- **Child agents** (like `metadata_assistant`) are NOT directly accessible by the supervisor
- This ensures proper separation of concerns and prevents the supervisor from bypassing the delegation hierarchy

## Testing

Verified with:
1. Direct graph creation test (no errors)
2. CLI query test (successful execution)
3. Node structure verification (metadata_assistant not in graph nodes)
4. All expected supervisor-accessible agents present in graph

## Related Files
- `/Users/tyo/GITHUB/omics-os/lobster/lobster/agents/graph.py` (modified)
- `/Users/tyo/GITHUB/omics-os/lobster/lobster/config/agent_registry.py` (reference - shows child_agents configuration)
- `/Users/tyo/GITHUB/omics-os/lobster/lobster/agents/langgraph_supervisor/supervisor.py` (reference - shows validation logic)
