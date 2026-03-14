#!/bin/bash
# Ink TUI Migration — Agent loop
# Run from worktree: cd .claude/worktrees/react-ink-cli && ./run_loop.sh

unset CLAUDECODE

WORKTREE="$(cd "$(dirname "$0")" && pwd)"
INTERVAL=100  # 15 minutes between iterations

cd "$WORKTREE" || { echo "Worktree not found: $WORKTREE"; exit 1; }

echo "=== Ink TUI Migration Loop Started at $(date) ==="
echo "=== Worktree: $WORKTREE ==="
echo "=== Interval: ${INTERVAL}s ==="

PROMPT="You are building a react-ink terminal UI for Lobster AI. Read .planning/IMPLEMENTATION_STATE.md for your current step and phase progress. Read .planning/assistant-ui-ink_migration_plan.md for the full plan — it has detailed instructions per step including validation criteria and commit messages. Read /Users/tyo/Omics-OS/lobster-cloud/.planning/cross_surface_protocol.md for the authoritative stream/API contract (state patch shapes, schema versioning §1.3, reconnection §5, shared APIs §4). For TypeScript: validate with cd lobster-tui-ink && bun run typecheck. For Python: validate with make format. If the step passes validation, commit with the message specified in the plan, then update .planning/IMPLEMENTATION_STATE.md: change 'pending' to 'done', add notes, advance Current Step. If blocked, write the blocker in Notes and set Status to BLOCKED. Do NOT push. Focus only on your current step. You may complete MULTIPLE small steps in one iteration if they are sequential and independent."

while true; do
  echo ""
  echo "--- Iteration starting at $(date) ---"

  # Count progress
  PENDING=$(grep -c '| pending |' "$WORKTREE/.planning/IMPLEMENTATION_STATE.md" 2>/dev/null || echo "0")
  DONE=$(grep -c '| done |' "$WORKTREE/.planning/IMPLEMENTATION_STATE.md" 2>/dev/null || echo "0")
  echo "Progress: $DONE done, $PENDING pending"

  claude -p \
    --dangerously-skip-permissions \
    --verbose \
    --output-format stream-json \
    "$PROMPT" 2>&1 | python3 -u -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except:
        continue
    t = d.get('type','')
    if t == 'assistant':
        for block in d.get('message',{}).get('content',[]):
            if block.get('type') == 'text':
                print(block['text'], flush=True)
            elif block.get('type') == 'tool_use':
                print(f\"  -> {block.get('name','?')}  \", flush=True)
    elif t == 'result':
        print(f\"\n=== Done ({d.get('duration_ms',0)//1000}s, \${d.get('total_cost_usd',0):.4f}) ===\", flush=True)
"

  echo "--- Iteration complete at $(date) ---"

  # Auto-stop on completion
  if grep -q "ALL STEPS COMPLETE" "$WORKTREE/.planning/IMPLEMENTATION_STATE.md" 2>/dev/null; then
    echo "=== Ink TUI Migration Complete! ==="
    break
  fi

  # Auto-stop on blocker (match exact status field, not UNBLOCKED)
  if grep -q '| BLOCKED |' "$WORKTREE/.planning/IMPLEMENTATION_STATE.md" 2>/dev/null; then
    echo "=== Agent BLOCKED — check .planning/IMPLEMENTATION_STATE.md Notes ==="
    break
  fi

  # Auto-stop when all steps done
  if ! grep -q '| pending |' "$WORKTREE/.planning/IMPLEMENTATION_STATE.md" 2>/dev/null; then
    echo "=== All steps complete! ==="
    break
  fi

  sleep $INTERVAL
done
