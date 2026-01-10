# Agent Coordination Quick Reference

## Key Files

### Core Coordination
- `/Users/tyo/GITHUB/omics-os/lobster/lobster/agents/graph.py` - Delegation tool creation
- `/Users/tyo/GITHUB/omics-os/lobster/lobster/agents/supervisor.py` - Supervisor coordination logic
- `/Users/tyo/GITHUB/omics-os/lobster/lobster/core/schemas/publication_queue.py` - Queue status definitions

### Agent Implementations
- `/Users/tyo/GITHUB/omics-os/lobster/lobster/agents/research_agent.py` - Creates HANDOFF_READY entries
- `/Users/tyo/GITHUB/omics-os/lobster/lobster/agents/metadata_assistant.py` - Processes queue, requests permissions

### Data Sharing
- `/Users/tyo/GITHUB/omics-os/lobster/lobster/tools/workspace_tool.py` - Unified workspace access
- `/Users/tyo/GITHUB/omics-os/lobster/lobster/core/queue_storage.py` - File locking for concurrent safety
- `/Users/tyo/GITHUB/omics-os/lobster/lobster/services/data_access/workspace_content_service.py` - Workspace service

## Key Code Patterns

### 1. Creating Delegation Tool (graph.py:39-89)
```python
def _create_agent_tool(agent_name: str, agent, tool_name: str, description: str):
    @tool(tool_name, description=description)
    def invoke_agent(task_description: str) -> str:
        config = {"run_name": agent_name, "tags": [agent_name]}
        result = agent.invoke(
            {"messages": [{"role": "user", "content": task_description}]},
            config=config
        )
        return result.get("messages", [])[-1].content
    return invoke_agent
```

### 2. Auto-Status Detection (publication_processing_service.py)
```python
is_ready_for_handoff = (
    bool(extracted_identifiers) and
    bool(dataset_ids) and
    bool(workspace_metadata_keys)
)
# → Status transitions to HANDOFF_READY
```

### 3. Permission Request (metadata_assistant.py:2693-2699)
```python
permission_request = f"""
Disease validation failed: {coverage:.1f}% (required: 50%)

I can attempt automatic enrichment to extract disease from publication abstracts.
- Estimated time: 20 seconds
- Cost: ~$0.03 in LLM tokens

Should I proceed with automatic enrichment?"""
```

### 4. Workspace File Writing (workspace_tool.py + workspace_content_service.py)
```python
# Write to workspace
content = MetadataContent(
    identifier="sra_PRJNA123_samples",
    data={"samples": sample_list},
    source="research_agent"
)
workspace_service.write_content(content, ContentType.METADATA)
```

### 5. Queue Status Update (publication_queue.py)
```python
queue.update_status(
    entry_id,
    PublicationStatus.HANDOFF_READY,
    handoff_status=HandoffStatus.READY_FOR_METADATA,
    workspace_metadata_keys=["sra_PRJNA123_samples"]
)
```

### 6. File Locking (queue_storage.py)
```python
with queue_file_lock(thread_lock, lock_path):
    atomic_write_jsonl(queue_path, entries, serializer)
```

## Status Flow

```
PENDING → EXTRACTING → METADATA_EXTRACTED → METADATA_ENRICHED → HANDOFF_READY → COMPLETED
                                                                          ↓
                                                                      FAILED
```

## Agent Handoff Matrix

| From | To | Trigger | Method |
|------|-----|---------|--------|
| Supervisor | research_agent | User request | `handoff_to_research_agent(task)` |
| research_agent | Supervisor | HANDOFF_READY status | Return message with status |
| Supervisor | metadata_assistant | HANDOFF_READY detected | `handoff_to_metadata_assistant(task)` |
| metadata_assistant | Supervisor | Permission needed | Return with options |
| metadata_assistant | Supervisor | Complete | Return with results |

## Data Sharing Patterns

| Pattern | Location | Persistence | Use Case |
|---------|----------|-------------|----------|
| **Workspace Files** | `workspace/metadata/*.json` | ✓ Permanent | Cross-agent data |
| **metadata_store** | `data_manager.metadata_store[key]` | ✗ Session | Within-agent cache |
| **Publication Queue** | `workspace/publication_queue.jsonl` | ✓ Permanent | Workflow state |

## Error Recovery Examples

### Missing Workspace File
```python
if workspace_key not in metadata_store:
    # Try workspace
    ws_data = workspace_service.read_content(workspace_key)
    if not ws_data:
        return "❌ Not found. Use get_content_from_workspace() to list available files"
```

### Low Disease Coverage
```python
if disease_coverage < min_disease_coverage:
    if strict_disease_validation:
        return f"❌ Failed: {coverage:.1f}% (need {min_disease_coverage*100:.1f}%)"
    else:
        logger.warning(f"Low coverage: {coverage:.1f}%")
        # Continue with warning
```

## Key Constants

### PublicationStatus (publication_queue.py:16-27)
- `PENDING` - Initial state
- `EXTRACTING` - Processing
- `HANDOFF_READY` - Ready for metadata_assistant
- `COMPLETED` - Successfully processed
- `FAILED` - Error occurred

### HandoffStatus (publication_queue.py:73-83)
- `NOT_READY` - Missing requirements
- `READY_FOR_METADATA` - Can process
- `METADATA_COMPLETE` - Successfully filtered
- `METADATA_FAILED` - Processing error

## Admin Override

```python
# In supervisor.py:120-130
if "ADMIN SUPERUSER" in user_message:
    # Bypass ALL confirmations
    # Execute immediately
    # No permission checks
```