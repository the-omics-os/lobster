# Human-in-the-Loop Component Architecture

**Status:** Design approved, implementation pending
**Date:** 2026-03-05
**Prerequisite:** Phase 5 parity baseline stable (no blocked commands)
**Reference implementation:** DeepAgents (`~/GITHUB/deepagents`)

---

## Problem

Lobster AI agents are fully autonomous. When a sub-agent encounters ambiguity (e.g., clusters with unclear markers, multiple valid thresholds), it either guesses or returns a generic result. There is no mechanism for an agent to ask the user for structured input during execution.

## Design Principles

1. **Sub-agents never talk to the user** — only the supervisor does.
2. **Sub-agents signal ambiguity** by returning incomplete/ambiguous results to the supervisor.
3. **Supervisor calls `ask_user` tool** — a single, universal entry point for all user interaction.
4. **A second LLM call maps the question to a UI component** — the supervisor doesn't know about components.
5. **The user interacts with a typed component** (Go TUI overlay, or prompt_toolkit in classic mode).
6. **The answer flows back through the supervisor** to the requesting sub-agent.

## Data Flow

```
Sub-agent (e.g., annotation_expert)
  | returns ambiguous result to supervisor
  v
Supervisor LLM decides: "I need user input"
  | calls ask_user(question="...", context={...})
  v
ask_user tool (lobster/tools/user_interaction.py)
  |
  +-- 1. Calls ComponentMapper LLM (structured output)
  |     Input:  question + context + COMPONENT_SCHEMAS registry
  |     Output: ComponentSelection{component, mode, data, fallback_prompt}
  |
  +-- 2. Calls interrupt(selection.model_dump())
  |     LangGraph checkpoints via InMemorySaver (already configured)
  |     Stream ends. Control returns to caller.
  |
  v
client._stream_query() detects __interrupt__ in stream updates
  | yields {"type": "interrupt", ...}
  v
+-----------------------------------------------+
| Go TUI path:                                  |
|   _handle_user_query() detects interrupt       |
|   -> bridge.send("component_render", {...})    |
|   -> Go renders BioCharm component overlay     |
|   -> User interacts, submits                   |
|   -> Go sends component_response to Python     |
|   -> client.resume_from_interrupt(response)    |
|   -> graph.stream(Command(resume=...), config) |
|   -> streaming continues                       |
+-----------------------------------------------+
| Classic CLI path:                              |
|   _handle_interrupt_classic()                  |
|   -> prompt_toolkit select/input/confirm       |
|   -> client.resume_from_interrupt(response)    |
|   -> continues in same loop                    |
+-----------------------------------------------+
  |
  v
ask_user tool receives resume value, returns it as string
  |
  v
Supervisor LLM has the user's answer
  | delegates back to sub-agent with answer
  v
Sub-agent completes work with user's choices
```

## Comparison with DeepAgents

| Aspect | DeepAgents | Lobster |
|--------|-----------|---------|
| Trigger | Static per-tool config: `interrupt_on={"write_file": True}` | Dynamic: supervisor LLM decides at runtime |
| Purpose | Safety gate (approve/reject tool calls) | Collaboration (ask user for structured input) |
| Component selection | Fixed registry: `tool_name -> renderer` | LLM mapper: `question + context -> component` |
| UI framework | Textual widgets (Python, in-process) | Go TUI components via IPC protocol |
| Resume primitive | `Command(resume=...)` | `Command(resume=...)` (same) |
| Interrupt detection | `__interrupt__` in stream updates | `__interrupt__` in stream updates (same) |

Key validated patterns adopted from DeepAgents:
- `Command(resume=...)` as the resume primitive
- `__interrupt__` detection in stream `updates` events
- `graph.get_state(config).interrupts` for pending interrupt check
- Resume loop pattern (stream -> interrupt -> render -> collect -> resume -> stream)

Key difference: DeepAgents interrupts are **pre-execution gates** (approve before tool runs). Lobster interrupts are **mid-workflow questions** (agent needs user input to decide).

## Component Architecture

### The `ask_user` Tool (Supervisor-Only)

```python
@tool
def ask_user(question: str, context: dict = None) -> str:
    """Ask the user a clarification question. Rendered as an interactive
    component (selector, slider, confirmation, text input, etc.).

    Args:
        question: Natural language question for the user
        context: Relevant data (cluster info, threshold ranges, option lists, etc.)
    """
```

From the supervisor's perspective, this is just "ask question, get answer." All component logic is encapsulated.

### Component Mapper (Second LLM)

Uses the session's configured LLM provider. Calls `ChatModel.with_structured_output()`:

```python
class ComponentSelection(BaseModel):
    component: str          # Registry key
    mode: str               # "inline" | "overlay" | "fullscreen"
    data: dict              # Component-specific typed payload
    fallback_prompt: str    # Always generated (for classic CLI or unknown component)
```

The mapper sees the question, context, and the full component schema registry. If no specific component fits, it defaults to `text_input` or `select`.

**Idempotency note:** The mapper LLM call happens before `interrupt()`. When the graph resumes, the tool restarts from the top (LangGraph rule). The mapper call is repeated but is idempotent (same inputs -> same output). This adds ~1-2s overhead on resume. Acceptable for now; cacheable later if needed.

### Component Schema Registry

```python
COMPONENT_SCHEMAS = {
    "confirm": {
        "description": "Yes/no confirmation dialog",
        "input_schema": {"question": str, "default": bool},
        "output_schema": {"confirmed": bool},
    },
    "select": {
        "description": "Single choice from a list of options",
        "input_schema": {"question": str, "options": list[str]},
        "output_schema": {"selected": str, "index": int},
    },
    "text_input": {
        "description": "Free-text answer (fallback for any question)",
        "input_schema": {"question": str, "placeholder": str},
        "output_schema": {"answer": str},
    },
    "cell_type_selector": {
        "description": "Assign cell type labels to single-cell clusters with marker gene context",
        "input_schema": {"clusters": [{"id": int, "size": int, "markers": list[str]}]},
        "output_schema": {"assignments": dict[str, str]},
    },
    "threshold_slider": {
        "description": "Adjust a numeric threshold with live preview of affected items",
        "input_schema": {"label": str, "min": float, "max": float, "default": float, "unit": str},
        "output_schema": {"value": float},
    },
}
```

### Interrupt Detection (Client)

In `_stream_query()`, after processing each `updates` event:

```python
if event_type == "updates" and isinstance(chunk, dict):
    if "__interrupt__" in chunk:
        for interrupt_obj in chunk["__interrupt__"]:
            yield {
                "type": "interrupt",
                "data": interrupt_obj.value,
                "interrupt_id": getattr(interrupt_obj, 'id', None),
            }
        return  # Stream ends at interrupt
```

After stream completes without interrupt, check for lingering interrupts:

```python
state = self.graph.get_state(config)
if hasattr(state, 'tasks') and state.tasks:
    for task in state.tasks:
        if hasattr(task, 'interrupts') and task.interrupts:
            yield {"type": "interrupt", "data": task.interrupts[0].value}
            return
```

### Resume Method (Client)

```python
def resume_from_interrupt(self, response: dict) -> Generator:
    """Resume graph execution with user's component response."""
    from langgraph.types import Command
    stream_input = Command(resume=response)
    config = {"configurable": {"thread_id": self.session_id}, "callbacks": self.callbacks}
    yield from self._stream_query(stream_input, config)
```

### Go TUI Event Loop (Resume Loop)

```python
def _handle_user_query(bridge, client, text):
    stream = client.query(text, stream=True)
    _process_stream(bridge, client, stream)

def _process_stream(bridge, client, stream):
    for event in stream:
        if event["type"] == "interrupt":
            _handle_interrupt(bridge, client, event)
            return  # Resume continues in _handle_interrupt
        _forward_event(bridge, event)

def _handle_interrupt(bridge, client, event):
    data = event["data"]
    # Send component to Go TUI
    msg_id = str(uuid4())
    bridge.send("component_render", data, id=msg_id)
    # Wait for user response
    while True:
        resp = bridge.recv_event(timeout=300)
        if resp and resp.get("type") == "component_response":
            break
    # Resume graph and continue processing
    resumed_stream = client.resume_from_interrupt(resp["payload"]["data"])
    _process_stream(bridge, client, resumed_stream)
```

### Classic CLI Fallback

```python
def _handle_interrupt_classic(client, interrupt_data):
    """Render interrupt as prompt_toolkit interaction in classic mode."""
    component = interrupt_data.get("component", "text_input")
    fallback = interrupt_data.get("fallback_prompt", "Please provide input:")
    data = interrupt_data.get("data", {})

    if component == "confirm":
        from prompt_toolkit import prompt
        answer = prompt(f"{fallback} [y/N]: ")
        return {"confirmed": answer.lower() in ("y", "yes")}
    elif component == "select":
        options = data.get("options", [])
        # Use prompt_toolkit's select or numbered list
        ...
    else:
        # Generic text fallback
        from prompt_toolkit import prompt
        answer = prompt(f"{fallback}: ")
        return {"answer": answer}
```

## File Map

| Layer | File | New/Modify | Purpose |
|-------|------|------------|---------|
| Schemas | `lobster/services/interaction/component_schemas.py` | NEW | Component registry (types, descriptions, input/output schemas) |
| Mapper | `lobster/services/interaction/component_mapper.py` | NEW | LLM with structured output: question+context -> ComponentSelection |
| Tool | `lobster/tools/user_interaction.py` | NEW | `create_ask_user_tool()` factory; calls mapper then `interrupt()` |
| Graph | `lobster/agents/graph.py` | MODIFY | Wire `ask_user` to supervisor tool list |
| Client | `lobster/core/client.py` | MODIFY | `__interrupt__` detection in `_stream_query()`, new `resume_from_interrupt()` |
| Go bridge | `lobster/cli_internal/go_tui_launcher.py` | MODIFY | Interrupt detection, `component_render` send, response wait, resume loop |
| Classic fallback | `lobster/cli_internal/classic_interaction.py` | NEW | prompt_toolkit renderers for each component type |
| Chat REPL | `lobster/cli_internal/commands/heavy/chat_commands.py` | MODIFY | Classic mode interrupt handling in REPL loop |
| Protocol (Go) | `lobster-tui/internal/protocol/types.go` | MODIFY | `TypeComponentRender`, `TypeComponentClose`, `TypeComponentResponse` |
| Protocol (Py) | `lobster/ui/bridge/protocol.py` | MODIFY | Matching Python constants |
| Go registry | `lobster-tui/internal/biocomp/registry.go` | NEW (Phase C) | `BioComponent` interface + factory map |
| Go components | `lobster-tui/internal/biocomp/*/` | NEW (Phase C) | Individual component packages |

## Execution Phases

### Phase 6A: Python Interrupt Infrastructure (No Go changes)

Build and validate entirely with classic CLI. All testable without Go binary.

1. `component_schemas.py` — registry with 5 base components (confirm, select, text_input, cell_type_selector, threshold_slider)
2. `component_mapper.py` — LLM mapper with `ChatModel.with_structured_output(ComponentSelection)`
3. `user_interaction.py` — `ask_user` tool with `interrupt()`
4. `client.py` — `__interrupt__` detection after stream updates, `resume_from_interrupt()` method
5. `graph.py` — wire `ask_user` to supervisor's tool list
6. `classic_interaction.py` — prompt_toolkit fallback renderers
7. `chat_commands.py` — interrupt handling in classic REPL loop

**Validation:**
- Unit: mapper produces valid ComponentSelection for known question patterns
- Unit: `ask_user` tool calls interrupt with correct payload
- Integration: end-to-end in classic mode (force supervisor to call `ask_user` via test prompt -> prompt_toolkit renders -> user responds -> graph resumes -> agent completes)

### Phase 6B: Protocol Extension (Python bridge + Go message types)

1. Add `component_render`, `component_close`, `component_response` to Go `types.go` and Python `protocol.py`
2. Modify `go_tui_launcher._handle_user_query()` — detect interrupt, send `component_render`, wait for `component_response`, resume
3. Go-side: route `component_render` to existing `pendingConfirm`/`pendingSelect` for basic types (confirm, select)
4. Protocol smoke tests for interrupt -> render -> response -> resume cycle

**Validation:**
- Protocol smoke: mock interrupt -> Go renders confirm -> Go sends response -> Python resumes
- Integration: same end-to-end as 6A but through Go TUI protocol

### Phase 6C: BioCharm Go Components (Incremental)

Each component is independent. Build in order of complexity:

1. `registry.go` — `BioComponent` interface + factory map
2. `confirm` component (simplest — validates full lifecycle)
3. `select` component (validates overlay rendering pattern)
4. `cell_type_selector` (first domain component — cluster annotation)
5. `threshold_slider` (first streaming component — `change` events from user, `component_render` updates from Python)
6. Remaining: `qc_dashboard`, `ontology_browser`, `sequence_input`, `dna_animation`

**Validation:** Per-component Go unit test + integration test via protocol

## Supervisor Prompt Changes

The supervisor prompt needs a small addition to its orchestration principles:

```
When a sub-agent returns ambiguous or incomplete results where user judgment
would improve the outcome, use the ask_user tool to present the user with
a structured question. Examples:
- Cluster annotation with unclear markers -> ask user to assign cell types
- Multiple valid thresholds -> ask user to choose cutoff
- Ambiguous dataset selection -> ask user to pick from options
Do not use ask_user for routine confirmations or progress updates.
```

This is additive — no existing behavior changes.

## Risks

1. **Double LLM call on resume** — mapper runs twice (once before interrupt, once on resume since node restarts). ~1-2s overhead. Mitigate later with state-level caching if needed.
2. **Component mapper hallucination** — LLM picks wrong component. `fallback_prompt` always generated as safety net. Classic CLI always has a text-based path.
3. **Go TUI timeout** — user doesn't respond within 300s. Need graceful timeout that cancels the interrupt and lets the supervisor continue with best guess.
4. **InMemorySaver doesn't survive process restart** — interrupt state lost if Python crashes. Acceptable for CLI; for cloud, upgrade to persistent checkpointer (separate concern).
5. **First-call latency** — heavy imports for mapper service on first interrupt (~1-2s). Show spinner via existing protocol pattern.
