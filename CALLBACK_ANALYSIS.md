# LangChain/LangGraph Callback Analysis for Multi-Agent Systems

## Executive Summary

**Problem**: TextualCallbackHandler is only capturing events from the first agent (supervisor), but subsequent agent handoffs are not triggering callbacks in the Textual UI dashboard.

**Root Cause**: Multiple issues with callback configuration and propagation in LangGraph's supervisor pattern.

---

## Complete LangChain BaseCallbackHandler Reference

### Core Callback Methods (All Available)

#### LLM Callbacks
```python
# For traditional LLMs (NOT chat models)
on_llm_start(serialized, prompts, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs)

# For chat models (use this instead of on_llm_start for chat models)
on_chat_model_start(serialized, messages, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs)

# Streaming support (only fires when streaming is enabled)
on_llm_new_token(token, *, chunk=None, run_id, parent_run_id=None, tags=None, **kwargs)

# Completion/Error
on_llm_end(response, *, run_id, parent_run_id=None, tags=None, **kwargs)
on_llm_error(error, *, run_id, parent_run_id=None, tags=None, **kwargs)
```

#### Chain Callbacks
```python
on_chain_start(serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs)
on_chain_end(outputs, *, run_id, parent_run_id=None, tags=None, **kwargs)
on_chain_error(error, *, run_id, parent_run_id=None, tags=None, **kwargs)
```

#### Tool Callbacks
```python
on_tool_start(serialized, input_str, *, run_id, parent_run_id=None, tags=None, metadata=None, inputs=None, **kwargs)
on_tool_end(output, *, run_id, parent_run_id=None, tags=None, **kwargs)
on_tool_error(error, *, run_id, parent_run_id=None, tags=None, **kwargs)
```

#### Agent Callbacks
```python
on_agent_action(action, *, run_id, parent_run_id=None, tags=None, **kwargs)
on_agent_finish(finish, *, run_id, parent_run_id=None, tags=None, **kwargs)
```

#### Retriever Callbacks
```python
on_retriever_start(serialized, query, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs)
on_retriever_end(documents, *, run_id, parent_run_id=None, tags=None, **kwargs)
on_retriever_error(error, *, run_id, parent_run_id=None, tags=None, **kwargs)
```

#### Other Callbacks
```python
on_text(text, *, run_id, parent_run_id=None, tags=None, **kwargs)
on_retry(retry_state, *, run_id, parent_run_id=None, **kwargs)
on_custom_event(name, data, *, run_id, tags=None, metadata=None, **kwargs)
```

### Run ID Hierarchy System

Every callback receives:
- `run_id: UUID` - Unique ID for the current operation
- `parent_run_id: UUID | None` - ID of the parent operation (for nested calls)

This enables tracking agent-within-agent execution flows.

### Ignore Flags (Granular Control)

```python
class BaseCallbackHandler:
    ignore_llm: bool = False           # Skip LLM events
    ignore_chain: bool = False         # Skip chain events
    ignore_agent: bool = False         # Skip agent events
    ignore_retriever: bool = False     # Skip retriever events
    ignore_chat_model: bool = False    # Skip chat model events
    ignore_retry: bool = False         # Skip retry events
    ignore_custom_event: bool = False  # Skip custom events
```

---

## Critical Findings: What We're Missing

### 1. **Missing Callback Method: `on_chat_model_start`**

**Current Implementation** (TextualCallbackHandler):
```python
def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
    """Called when an LLM starts."""
```

**Problem**:
- `on_llm_start` is for traditional LLMs (completion models)
- Claude/GPT are **chat models** → should use `on_chat_model_start` instead
- LangGraph with chat models may not fire `on_llm_start` at all

**Solution**: Add this method:
```python
def on_chat_model_start(
    self,
    serialized: Dict[str, Any],
    messages: List[List[BaseMessage]],
    *,
    run_id,
    parent_run_id=None,
    **kwargs
) -> None:
    """Called when a chat model starts (e.g., Claude, GPT)."""
    if serialized is None:
        serialized = {}
    agent_name = kwargs.get("name") or serialized.get("name", "unknown")

    # Track start time
    self.start_times[agent_name] = datetime.now()

    # Update current agent
    if agent_name != self.current_agent:
        self.current_agent = agent_name
        self.agent_stack.append(agent_name)

    # Post activity event
    self._post_activity(
        event_type=EventType.AGENT_START,
        agent_name=agent_name,
    )
```

---

### 2. **Callback List Nesting Bug**

**Location**: `lobster/agents/graph.py:117-118`

**Current Code**:
```python
# In client.py line 127
callback_handler=self.callbacks,  # This is a LIST: [TokenTracker, TextualCallback]

# In graph.py line 117-118
if callback_handler and hasattr(supervisor_model, "with_config"):
    supervisor_model = supervisor_model.with_config(callbacks=[callback_handler])
    # BUG: [callback_handler] wraps the list → [[TokenTracker, TextualCallback]]
```

**Problem**: Creates nested list `[[callback1, callback2]]` instead of flat `[callback1, callback2]`

**Solution**:
```python
if callback_handler:
    # Handle both list and single callback
    callbacks = callback_handler if isinstance(callback_handler, list) else [callback_handler]
    if hasattr(supervisor_model, "with_config"):
        supervisor_model = supervisor_model.with_config(callbacks=callbacks)
```

---

### 3. **Missing Callback Propagation to Worker Agents**

**Problem**: The supervisor pattern in LangGraph creates isolated execution contexts for each worker agent. Callbacks configured at the graph level don't automatically propagate to nested `create_react_agent` calls.

**Current Flow**:
```
1. Graph invoked with config={"callbacks": [callbacks]}
2. Supervisor LLM gets callbacks via .with_config()
3. Supervisor delegates to worker agent (e.g., research_agent)
4. Worker agent creates NEW execution context
5. ❌ Callbacks NOT inherited by worker agent's LLM calls
```

**Architecture Issue**:
```python
# graph.py creates agents like this:
agent = factory_function(
    data_manager=data_manager,
    callback_handler=callback_handler,  # Passed here
    agent_name=agent_config.name
)

# But agent factories do this:
def data_expert(data_manager, callback_handler=None, ...):
    llm = create_llm(...)
    if callback_handler and hasattr(llm, "with_config"):
        llm = llm.with_config(callbacks=[callback_handler])  # Same nesting bug!

    return create_react_agent(llm, tools)  # ⚠️ create_react_agent doesn't inherit callbacks
```

**Solution**: Pass callbacks explicitly when creating the agent executor:

```python
# In each agent factory (e.g., data_expert.py):
def data_expert(data_manager, callback_handler=None, ...):
    llm = create_llm(...)

    # Fix list nesting
    callbacks = callback_handler if isinstance(callback_handler, list) else [callback_handler] if callback_handler else []

    if callbacks and hasattr(llm, "with_config"):
        llm = llm.with_config(callbacks=callbacks)

    # IMPORTANT: Also pass to agent config
    agent = create_react_agent(
        llm,
        tools,
        state_schema=DataExpertState,  # If using custom state
        # ⚠️ LangGraph 0.2+ syntax - check your version
        # May need to pass via config parameter or different method
    )

    return agent
```

---

### 4. **LangGraph Invocation Config Not Propagating**

**Current Code** (`client.py:157-161`):
```python
config = {
    "configurable": {"thread_id": self.session_id},
    "callbacks": self.callbacks,  # ✅ Correct
    "recursion_limit": 100,
}

for event in self.graph.stream(input=graph_input, config=config, stream_mode="updates"):
    # Process events
```

**Problem**: LangGraph's `create_supervisor` may not properly propagate callbacks from the config to individual node executions, especially when using the supervisor pattern with tool-wrapped agents.

**Verification Needed**: Check if `langgraph_supervisor.create_supervisor` supports callback propagation. This is likely a limitation of the supervisor library.

---

### 5. **Missing `run_id` and `parent_run_id` Tracking**

**Current Implementation**: TextualCallbackHandler doesn't use `run_id` or `parent_run_id` parameters.

**Why This Matters**:
- Can't distinguish between nested agent calls (supervisor → worker → sub-worker)
- Can't track which events belong to which agent execution
- Loses hierarchy information critical for multi-agent debugging

**Solution**: Track run IDs:
```python
class TextualCallbackHandler(BaseCallbackHandler):
    def __init__(self, app=None, ...):
        self.run_stack: List[UUID] = []  # Track execution hierarchy
        self.run_agent_map: Dict[UUID, str] = {}  # Map run_id → agent_name

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        """Track hierarchical agent execution."""
        agent_name = kwargs.get("name") or serialized.get("name", "unknown")

        # Track run ID
        self.run_agent_map[run_id] = agent_name
        if parent_run_id:
            # This is a child execution
            parent_agent = self.run_agent_map.get(parent_run_id, "unknown")
            self._post_activity(
                event_type=EventType.HANDOFF,
                agent_name="system",
                content="",
                metadata={"from": parent_agent, "to": agent_name, "run_id": str(run_id)}
            )

        self.run_stack.append(run_id)
        self.current_agent = agent_name
        # ... rest of logic
```

---

## Recommended Fixes (Priority Order)

### Fix 1: Add `on_chat_model_start` (CRITICAL)
**File**: `lobster/ui/callbacks/textual_callback.py`
**Lines**: Add after line 177

```python
def on_chat_model_start(
    self,
    serialized: Dict[str, Any],
    messages: List[List[BaseMessage]],
    *,
    run_id,
    parent_run_id=None,
    tags=None,
    metadata=None,
    **kwargs
) -> None:
    """Called when a chat model starts (Claude, GPT, etc)."""
    if serialized is None:
        serialized = {}
    agent_name = kwargs.get("name") or serialized.get("name", "unknown")

    # Track start time
    self.start_times[agent_name] = datetime.now()

    # Update current agent
    if agent_name != self.current_agent:
        self.current_agent = agent_name
        self.agent_stack.append(agent_name)

    # Post activity event
    self._post_activity(
        event_type=EventType.AGENT_START,
        agent_name=agent_name,
    )
```

### Fix 2: Unwrap Callback List (HIGH)
**File**: `lobster/agents/graph.py`
**Line**: 117-118

```python
# OLD (BUGGY):
if callback_handler and hasattr(supervisor_model, "with_config"):
    supervisor_model = supervisor_model.with_config(callbacks=[callback_handler])

# NEW (FIXED):
if callback_handler:
    callbacks = callback_handler if isinstance(callback_handler, list) else [callback_handler]
    if hasattr(supervisor_model, "with_config"):
        supervisor_model = supervisor_model.with_config(callbacks=callbacks)
```

**Repeat fix in all agent factories** (data_expert.py, research_agent.py, etc.):
```python
# Pattern to find: grep -n "llm.with_config(callbacks=\[callback_handler\])" lobster/agents/*.py

# Replace with:
callbacks = callback_handler if isinstance(callback_handler, list) else [callback_handler] if callback_handler else []
if callbacks and hasattr(llm, "with_config"):
    llm = llm.with_config(callbacks=callbacks)
```

### Fix 3: Add `run_id` Tracking (MEDIUM)
**File**: `lobster/ui/callbacks/textual_callback.py`
**Lines**: Update `__init__` and all callback methods

```python
class TextualCallbackHandler(BaseCallbackHandler):
    def __init__(self, app=None, ...):
        # ... existing code ...
        self.run_stack: List[UUID] = []
        self.run_agent_map: Dict[UUID, str] = {}

    def on_chat_model_start(self, ..., *, run_id, parent_run_id=None, **kwargs):
        agent_name = kwargs.get("name") or serialized.get("name", "unknown")
        self.run_agent_map[run_id] = agent_name

        # Detect handoff via parent_run_id
        if parent_run_id and parent_run_id in self.run_agent_map:
            parent_agent = self.run_agent_map[parent_run_id]
            if parent_agent != agent_name:
                self._post_activity(
                    event_type=EventType.HANDOFF,
                    agent_name="system",
                    metadata={"from": parent_agent, "to": agent_name}
                )

        # ... rest of logic ...
```

### Fix 4: Debug LangGraph Supervisor Callbacks (INVESTIGATION)
**Action**: Test if `langgraph_supervisor.create_supervisor` supports callback propagation

**Test Script**:
```python
# Create minimal test case
from langgraph_supervisor import create_supervisor
from langchain_core.callbacks import BaseCallbackHandler

class DebugCallback(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        print(f"[CALLBACK] Chat model start: {kwargs.get('name')}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"[CALLBACK] Tool start: {serialized.get('name')}")

# Test with supervisor pattern
graph = create_supervisor(agents=[...], model=llm, ...)
config = {"callbacks": [DebugCallback()]}
result = graph.invoke(input, config=config)

# Check if callbacks fire for worker agents
```

---

## Alternative Approaches

### Option A: Patch `langgraph_supervisor` (if callbacks don't propagate)

If the supervisor library doesn't propagate callbacks, you can manually inject them at the graph node level:

```python
# In graph.py after line 306
from langgraph.pregel import Pregel

# Access the compiled graph's nodes and inject callbacks
if isinstance(graph, Pregel):
    for node_name, node in graph.nodes.items():
        if hasattr(node, "runnable") and hasattr(node.runnable, "with_config"):
            callbacks = callback_handler if isinstance(callback_handler, list) else [callback_handler]
            node.runnable = node.runnable.with_config(callbacks=callbacks)
```

**Warning**: This is a hack and may break with library updates.

### Option B: Custom Supervisor Implementation

Replace `langgraph_supervisor.create_supervisor` with a custom implementation that guarantees callback propagation:

```python
# Custom supervisor that explicitly passes callbacks
def create_callback_aware_supervisor(agents, model, callbacks, ...):
    # Ensure all agents have callbacks configured
    for agent in agents:
        if hasattr(agent, "model"):
            agent.model = agent.model.with_config(callbacks=callbacks)

    # Build graph with explicit callback passing
    # ... implementation ...
```

### Option C: Event-Based Monitoring (Workaround)

Instead of relying on callbacks, monitor the graph execution via LangGraph's event stream:

```python
# In client.py
for event in self.graph.stream(graph_input, config, stream_mode="values"):
    # Parse event to detect agent switches
    if "messages" in event:
        last_message = event["messages"][-1]
        # Detect agent from message metadata
        agent_name = extract_agent_from_message(last_message)
        # Post to UI
        ui_callback._post_activity(EventType.AGENT_START, agent_name)
```

**Pros**: Independent of callback system
**Cons**: Less precise timing, no tool-level events

---

## Testing Checklist

```bash
# 1. Test basic callback firing
lobster chat "test message"
# ✅ Check: Does on_chat_model_start fire?
# ✅ Check: Does on_llm_start fire? (should NOT for chat models)

# 2. Test multi-agent handoff
lobster query "Search for cancer genomics papers, then download GSE12345"
# ✅ Check: Supervisor → research_agent handoff visible in UI?
# ✅ Check: research_agent → data_expert handoff visible?
# ✅ Check: Activity log shows all agents?

# 3. Test tool execution
lobster query "List available modalities"
# ✅ Check: on_tool_start fires for list_modalities?
# ✅ Check: on_tool_end fires with duration?

# 4. Test nested delegation (if you have parent-child agents)
# ✅ Check: parent_run_id correctly links child to parent
# ✅ Check: UI shows hierarchy (supervisor → parent → child)
```

---

## Code Search Commands (for implementation)

```bash
# Find all callback attachment points
grep -rn "with_config(callbacks=" lobster/agents/

# Find all agent factories
grep -rn "def .*_expert(" lobster/agents/

# Find where graph is invoked
grep -rn "graph.stream\|graph.invoke" lobster/

# Check LangGraph supervisor usage
grep -rn "create_supervisor" lobster/
```

---

## Expected Behavior After Fixes

**Before**:
```
Activity Log:
● supervisor started
▶ supervisor thinking...
✓ supervisor done
(no other agents visible)
```

**After**:
```
Activity Log:
● supervisor started
▶ handoff: supervisor → research_agent
● research_agent started
  ├─ tool: search_pubmed
  └─ tool: extract_geo_ids
✓ research_agent done
▶ handoff: research_agent → data_expert
● data_expert started
  ├─ tool: execute_download_from_queue
  └─ tool: list_modalities
✓ data_expert done
▶ handoff: data_expert → supervisor
✓ supervisor done
```

---

## References

- LangChain Callbacks API: https://python.langchain.com/docs/modules/callbacks/
- BaseCallbackHandler Source: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/callbacks/base.py
- LangGraph Config Propagation: https://langchain-ai.github.io/langgraph/how-tos/configuration/
- Run ID Tracking: https://python.langchain.com/docs/modules/callbacks/#using-an-existing-handler

---

## Next Steps

1. ✅ **Implement Fix 1** (`on_chat_model_start`) - 5 min
2. ✅ **Implement Fix 2** (unwrap callback lists) - 15 min
3. ⚠️ **Test with real query** - validate callbacks fire
4. 🔬 **Implement Fix 3** (`run_id` tracking) if handoffs still missing - 30 min
5. 🔬 **Investigate Fix 4** (LangGraph supervisor) if still broken - 2 hours
6. 📊 **Consider Option C** (event-based monitoring) as backup - 1 hour

**Estimated Time**: 1-4 hours depending on supervisor library behavior.
