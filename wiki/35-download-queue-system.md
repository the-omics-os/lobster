# 25. Download Queue System

## Overview

The Download Queue System is a structured handoff contract between `research_agent` and `data_expert`, enabling coordinated dataset downloads with pre-validation, error recovery, and supervisor visibility.

**Key Benefits**:
- **Pre-download validation**: Validate metadata before downloading (saves 5-10 minutes per invalid dataset)
- **Layered enrichment**: research_agent prepares everything once, data_expert executes
- **Supervisor coordination**: Centralized queue visibility for multi-agent workflows
- **Error recovery**: Failed downloads tracked with error logs and retry capability
- **Concurrent protection**: Prevents duplicate downloads via status management

**Introduced**: Phase 2 (November 2024)

---

## Architecture Overview

### System Components

```mermaid
graph LR
    subgraph "research_agent"
        V[validate_dataset_metadata]
        GEO[GEOProvider.get_download_urls]
    end

    subgraph "Download Queue"
        Q[(download_queue.jsonl)]
        QM[DownloadQueue Manager]
    end

    subgraph "supervisor"
        WS[get_content_from_workspace]
    end

    subgraph "data_expert"
        DL[execute_download_from_queue]
    end

    V --> GEO
    GEO --> QM
    QM --> Q
    WS -.reads.-> Q
    WS --> |entry_id| DL
    DL --> |downloads| QM

    classDef agent fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef queue fill:#fff3e0,stroke:#f57c00,stroke-width:2px

    class V,GEO,WS,DL agent
    class Q,QM queue
```

### Queue Lifecycle

```mermaid
stateDiagram-v2
    [*] --> PENDING: research_agent<br/>creates entry

    PENDING --> IN_PROGRESS: data_expert<br/>starts download

    IN_PROGRESS --> COMPLETED: Download<br/>succeeds
    IN_PROGRESS --> FAILED: Download<br/>fails

    COMPLETED --> [*]: Modality<br/>available
    FAILED --> PENDING: Retry<br/>(manual)
    FAILED --> [*]: Abandon

    note right of PENDING
        Queue entry created with:
        - Metadata from validation
        - URLs from GEOProvider
        - Status: PENDING
    end note

    note right of IN_PROGRESS
        Prevents concurrent downloads
        - Downloaded_by: agent name
        - Updated_at: timestamp
    end note

    note right of COMPLETED
        Success metadata:
        - Modality_name
        - Final status
        - Completion timestamp
    end note

    note right of FAILED
        Error tracking:
        - Error_log: [errors]
        - Retry count
        - Failure reason
    end note
```

---

## Queue Entry Structure

### DownloadQueueEntry Schema

```python
from lobster.core.schemas.download_queue import DownloadQueueEntry, DownloadStatus

entry = DownloadQueueEntry(
    # Unique identifier
    entry_id="queue_GSE180759_5c1fb112",

    # Dataset identification
    dataset_id="GSE180759",
    database="geo",  # "geo", "sra", "pride", "metabolomics"

    # Queue management
    priority=5,  # 1-10 (10 = highest)
    status=DownloadStatus.PENDING,  # PENDING, IN_PROGRESS, COMPLETED, FAILED

    # Prepared by research_agent
    metadata={
        "title": "Multiple sclerosis single-cell RNA-seq",
        "n_samples": 20,
        "platform_id": "GPL24676",
        "organism": "Homo sapiens"
    },
    validation_result={
        "recommendation": "proceed",
        "confidence_score": 0.95,
        "issues": []
    },

    # URLs from GEOProvider
    matrix_url="ftp://ftp.ncbi.nlm.nih.gov/.../GSE180759_series_matrix.txt.gz",
    raw_urls=[],
    supplementary_urls=[
        "ftp://ftp.ncbi.nlm.nih.gov/.../GSE180759_annotation.txt.gz",
        "ftp://ftp.ncbi.nlm.nih.gov/.../GSE180759_expression.csv.gz"
    ],
    h5_url=None,

    # Timestamps
    created_at=datetime(2024, 11, 14, 19, 30, 0),
    updated_at=datetime(2024, 11, 14, 19, 30, 0),

    # Execution metadata (filled by data_expert)
    recommended_strategy=None,  # Filled by data_expert_assistant (optional)
    downloaded_by=None,  # Agent name
    modality_name=None,  # Final modality name in DataManagerV2
    error_log=[]  # Error messages if FAILED
)
```

### Field Descriptions

| Field | Type | Description | Set By | When |
|-------|------|-------------|--------|------|
| `entry_id` | str | Unique identifier (`queue_{dataset_id}_{uuid}`) | research_agent | Creation |
| `dataset_id` | str | GEO accession (GSE12345, etc.) | research_agent | Creation |
| `database` | str | Database type ("geo", "sra", etc.) | research_agent | Creation |
| `priority` | int | Download priority (1-10, 10=highest) | research_agent | Creation |
| `status` | DownloadStatus | Queue status (PENDING/IN_PROGRESS/COMPLETED/FAILED) | research_agent → data_expert | Lifecycle |
| `metadata` | dict | Full GEO metadata from validation | research_agent | Creation |
| `validation_result` | dict | Validation report with recommendation | research_agent | Creation |
| `matrix_url` | str | Matrix file URL | research_agent | Creation |
| `raw_urls` | List[str] | Raw data file URLs | research_agent | Creation |
| `supplementary_urls` | List[str] | Supplementary file URLs | research_agent | Creation |
| `h5_url` | str | H5AD file URL (if available) | research_agent | Creation |
| `created_at` | datetime | Queue entry creation timestamp | research_agent | Creation |
| `updated_at` | datetime | Last update timestamp | research_agent → data_expert | Updates |
| `recommended_strategy` | StrategyConfig | Download strategy recommendation | data_expert_assistant | Optional |
| `downloaded_by` | str | Agent name that executed download | data_expert | Download start |
| `modality_name` | str | Final modality name in DataManagerV2 | data_expert | Completion |
| `error_log` | List[str] | Error messages if download failed | data_expert | Failure |

---

## Workflow Patterns

### Pattern 1: Standard Download Workflow

**Scenario**: User wants to download a validated GEO dataset

```python
# Step 1: research_agent validates and queues
result = research_agent.validate_dataset_metadata(
    dataset_id="GSE180759",
    add_to_queue=True  # Default
)
# Output: "✅ Dataset added to download queue: queue_GSE180759_5c1fb112"

# Step 2: supervisor queries queue
result = supervisor.get_content_from_workspace(
    workspace="download_queue",
    level="summary"
)
# Output: "1 pending entry: queue_GSE180759_5c1fb112 (GSE180759, priority 5)"

# Step 3: data_expert downloads from queue
result = data_expert.execute_download_from_queue(
    entry_id="queue_GSE180759_5c1fb112"
)
# Output: "✅ Download complete: geo_gse180759_transcriptomics (20 samples × 15000 features)"
```

### Pattern 2: Pre-Download Validation Workflow

**Scenario**: Validate dataset content BEFORE downloading (saves time/bandwidth)

```python
# Step 1: research_agent validates and queues
result = research_agent.validate_dataset_metadata(
    dataset_id="GSE180759",
    add_to_queue=True
)

# Step 2: metadata_assistant validates from cached metadata (no download!)
result = metadata_assistant.validate_dataset_content(
    modality_name="geo_gse180759",  # Not yet downloaded
    source_type="metadata_store",   # Use cached metadata
    required_fields="cell_type,condition,replicate",
    threshold=0.8
)
# Output: "✅ Validation passed: 85% of required fields present"

# Step 3: If validation passes, data_expert downloads
if "validation passed" in result:
    result = data_expert.execute_download_from_queue(
        entry_id="queue_GSE180759_5c1fb112"
    )
```

### Pattern 3: Multi-Dataset Download Workflow

**Scenario**: Download multiple datasets in batch

```python
# Step 1: research_agent queues multiple datasets
datasets = ["GSE180759", "GSE109564", "GSE126906"]

for dataset_id in datasets:
    research_agent.validate_dataset_metadata(
        dataset_id=dataset_id,
        add_to_queue=True
    )

# Step 2: supervisor reviews all pending
result = supervisor.get_content_from_workspace(
    workspace="download_queue",
    status_filter="PENDING"
)
# Output: "3 pending entries"

# Step 3: data_expert downloads all
pending_entries = data_manager.download_queue.list_entries(status=DownloadStatus.PENDING)
for entry in pending_entries:
    data_expert.execute_download_from_queue(entry_id=entry.entry_id)
```

### Pattern 4: Priority-Based Download

**Scenario**: User requests high-priority download for urgent analysis

```python
# Step 1: research_agent validates with high priority
result = research_agent.validate_dataset_metadata(
    dataset_id="GSE180759",
    add_to_queue=True,
    priority=10  # Highest priority
)

# Step 2: supervisor queries by priority (implicitly sorts)
result = supervisor.get_content_from_workspace(
    workspace="download_queue",
    status_filter="PENDING"
)
# Output: "queue_GSE180759_5c1fb112 (priority 10) appears first"

# Step 3: data_expert downloads highest priority first
pending = data_manager.download_queue.list_entries(status=DownloadStatus.PENDING)
highest_priority = max(pending, key=lambda e: e.priority)
data_expert.execute_download_from_queue(entry_id=highest_priority.entry_id)
```

---

## Error Handling

### Error Recovery Pattern

```python
# Query failed downloads
failed_entries = data_manager.download_queue.list_entries(status=DownloadStatus.FAILED)

for entry in failed_entries:
    print(f"Failed: {entry.dataset_id}")
    print(f"Errors: {entry.error_log}")

    # Option 1: Retry (reset to PENDING)
    data_manager.download_queue.update_status(
        entry_id=entry.entry_id,
        status=DownloadStatus.PENDING
    )

    # Option 2: Remove from queue (abandon)
    data_manager.download_queue.remove_entry(entry.entry_id)
```

### Common Errors & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `EntryNotFoundError` | Invalid entry_id | Query download_queue to get valid entry_ids |
| `Network timeout` | GEO FTP unavailable | Retry later (queue persists across sessions) |
| `File not found (404)` | Dataset not public/deleted | Remove entry, try alternative dataset |
| `Validation failed` | Metadata doesn't meet criteria | Adjust validation parameters or skip |
| `Already completed` | Duplicate download attempt | Entry already has modality_name set |
| `Permission denied` | Dataset restricted/embargoed | Wait for public release or request access |
| `Corrupt download` | Partial file transfer | Retry with network stability check |

### Error Log Structure

```python
# Example error log after failed download
entry.error_log = [
    "2024-11-14 19:35:00: Network timeout after 60s",
    "2024-11-14 19:40:00: Retry attempt 1 failed: Connection reset",
    "2024-11-14 19:45:00: Retry attempt 2 failed: FTP server unavailable"
]
```

---

## Performance Considerations

### Queue Operations Performance

| Operation | Target | Actual | Notes |
|-----------|--------|--------|-------|
| `add_entry()` | <100ms | ~50ms | Atomic write to JSONL |
| `get_entry()` | <50ms | ~20ms | Read from in-memory cache |
| `update_status()` | <100ms | ~60ms | Update + backup |
| `list_entries()` | <200ms | ~100ms | Filter 1000 entries |
| `remove_entry()` | <100ms | ~70ms | Rewrite JSONL file |
| `clear_queue()` | <500ms | ~200ms | Delete + recreate file |

### Download Workflow Performance

**Before (Synchronous Pattern)**:
- Metadata fetch: 2-3 seconds
- Duplicate fetch: 2-3 seconds (research + data_expert both fetch)
- Total overhead: **4-6 seconds**

**After (Queue Pattern)**:
- Metadata fetch: 2-3 seconds (once by research_agent)
- Queue operations: <100ms
- Total overhead: **2-3 seconds** (50% faster)

### Memory Usage

| Queue Size | Memory Usage | Notes |
|------------|--------------|-------|
| 10 entries | ~50 KB | Minimal impact |
| 100 entries | ~500 KB | In-memory cache |
| 1000 entries | ~5 MB | Recommended max |

**Best Practice**: Clear completed entries periodically to maintain performance.

---

## API Reference

### DataManagerV2.download_queue

```python
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.schemas.download_queue import DownloadStatus

data_manager = DataManagerV2(workspace_path="./workspace")

# Access download queue
queue = data_manager.download_queue

# Add entry
queue.add_entry(entry)

# Get entry
entry = queue.get_entry("queue_GSE180759_abc123")

# Update status
queue.update_status(
    entry_id="queue_GSE180759_abc123",
    status=DownloadStatus.COMPLETED,
    modality_name="geo_gse180759_transcriptomics"
)

# List entries (with optional filter)
pending = queue.list_entries(status=DownloadStatus.PENDING)
in_progress = queue.list_entries(status=DownloadStatus.IN_PROGRESS)
completed = queue.list_entries(status=DownloadStatus.COMPLETED)
failed = queue.list_entries(status=DownloadStatus.FAILED)
all_entries = queue.list_entries()  # All statuses

# Remove entry
queue.remove_entry("queue_GSE180759_abc123")

# Clear all entries
queue.clear_queue()
```

### get_content_from_workspace (Supervisor/Research Agent)

```python
# Query all entries
result = get_content_from_workspace(workspace="download_queue")

# Query specific entry
result = get_content_from_workspace(
    identifier="queue_GSE180759_abc123",
    workspace="download_queue",
    level="metadata"  # summary | metadata | validation | strategy
)

# Filter by status
result = get_content_from_workspace(
    workspace="download_queue",
    status_filter="PENDING"  # PENDING | IN_PROGRESS | COMPLETED | FAILED
)

# Get detailed entry information
result = get_content_from_workspace(
    identifier="queue_GSE180759_abc123",
    workspace="download_queue",
    level="full"  # All fields including URLs and error logs
)
```

### execute_download_from_queue (Data Expert)

```python
# Execute download from queue entry
result = data_expert.execute_download_from_queue(
    entry_id="queue_GSE180759_abc123"
)
# Returns: "✅ Download complete: geo_gse180759_transcriptomics (20 samples × 15000 features)"

# Status automatically updated:
# - PENDING → IN_PROGRESS (at start)
# - IN_PROGRESS → COMPLETED (on success)
# - IN_PROGRESS → FAILED (on error)

# On completion, entry populated with:
# - downloaded_by: "data_expert"
# - modality_name: "geo_gse180759_transcriptomics"
# - updated_at: current timestamp

# On failure, entry populated with:
# - error_log: ["Error message 1", "Error message 2"]
# - status: FAILED
```

---

## Integration with Other Systems

### Workspace Persistence

The download queue persists across sessions via `download_queue.jsonl` in the workspace directory.

```python
# Session 1: research_agent queues datasets
research_agent.validate_dataset_metadata("GSE180759", add_to_queue=True)
research_agent.validate_dataset_metadata("GSE109564", add_to_queue=True)

# Exit session

# Session 2: data_expert resumes downloads
data_manager = DataManagerV2(workspace_path="./workspace")
pending = data_manager.download_queue.list_entries(status=DownloadStatus.PENDING)
# Output: 2 pending entries (persisted from Session 1)

for entry in pending:
    data_expert.execute_download_from_queue(entry_id=entry.entry_id)
```

### Provenance Tracking

All queue operations are logged in the W3C-PROV provenance system.

```python
# Queue entry creation logged as:
activity = prov.Activity(
    identifier="validate_dataset_metadata_GSE180759",
    attributes={
        "tool": "validate_dataset_metadata",
        "dataset_id": "GSE180759",
        "queue_entry_id": "queue_GSE180759_abc123",
        "status": "PENDING"
    }
)

# Download execution logged as:
activity = prov.Activity(
    identifier="execute_download_from_queue_GSE180759",
    attributes={
        "tool": "execute_download_from_queue",
        "entry_id": "queue_GSE180759_abc123",
        "modality_name": "geo_gse180759_transcriptomics",
        "status": "COMPLETED"
    }
)
```

### Metadata Store Integration

Queue entries reference metadata stored in `DataManagerV2.metadata_store`:

```python
# research_agent stores metadata
data_manager.metadata_store["GSE180759"] = {
    "title": "...",
    "organism": "Homo sapiens",
    "n_samples": 20,
    "platform_id": "GPL24676"
}

# Queue entry references metadata
entry.metadata = data_manager.metadata_store["GSE180759"]

# metadata_assistant can validate without downloading
metadata_assistant.validate_dataset_content(
    modality_name="geo_gse180759",
    source_type="metadata_store"  # Uses cached metadata
)
```

---

## Testing

### Unit Tests

**File**: `tests/unit/core/test_download_queue.py` (25 tests)

**Coverage**:
- Queue initialization and file creation
- Entry addition, retrieval, update, removal
- Status transitions (PENDING → IN_PROGRESS → COMPLETED/FAILED)
- Error log management
- Priority sorting
- Concurrent operation safety

**Key Test Cases**:
```python
def test_add_entry_creates_queue_file()
def test_get_entry_returns_correct_entry()
def test_update_status_transitions_correctly()
def test_list_entries_filters_by_status()
def test_remove_entry_deletes_correctly()
def test_clear_queue_removes_all_entries()
def test_concurrent_updates_maintain_consistency()
```

### Integration Tests

**File**: `tests/integration/test_download_queue_workspace.py` (15 tests)

**Coverage**:
- Complete workflow (research → supervisor → data_expert)
- Multi-dataset queue management
- Error recovery and retry logic
- Workspace persistence across sessions
- Provenance tracking integration
- Multi-agent coordination

**Key Test Scenarios**:
```python
def test_complete_download_workflow()
def test_multi_dataset_batch_download()
def test_failed_download_error_recovery()
def test_workspace_persistence_across_sessions()
def test_supervisor_queue_visibility()
def test_pre_download_validation_workflow()
```

### Performance Benchmarks

**File**: `tests/performance/test_queue_performance.py`

**Benchmarks**:
- Queue operation latency (add/get/update/list)
- Scalability with 100/1000/10000 entries
- Memory usage tracking
- Concurrent access stress testing

---

## Troubleshooting

### Issue: Queue entry not found

**Symptoms**: `EntryNotFoundError: Entry 'queue_GSE180759_abc123' not found`

**Causes**:
- Entry ID typo
- Entry already removed
- Queue file corruption

**Solutions**:
```python
# List all entries to verify ID
all_entries = data_manager.download_queue.list_entries()
for entry in all_entries:
    print(entry.entry_id)

# Check for completed/failed entries
completed = data_manager.download_queue.list_entries(status=DownloadStatus.COMPLETED)
failed = data_manager.download_queue.list_entries(status=DownloadStatus.FAILED)
```

### Issue: Download stuck in IN_PROGRESS

**Symptoms**: Entry status remains IN_PROGRESS for extended period

**Causes**:
- Agent crash during download
- Network interruption
- Process termination

**Solutions**:
```python
# Query IN_PROGRESS entries
in_progress = data_manager.download_queue.list_entries(status=DownloadStatus.IN_PROGRESS)

# Reset to PENDING for retry
for entry in in_progress:
    if (datetime.now() - entry.updated_at).seconds > 600:  # 10 minutes
        data_manager.download_queue.update_status(
            entry_id=entry.entry_id,
            status=DownloadStatus.PENDING,
            downloaded_by=None
        )
```

### Issue: Duplicate downloads

**Symptoms**: Same dataset downloaded multiple times

**Causes**:
- Multiple queue entries for same dataset
- Status not updated correctly

**Prevention**:
```python
# Check if dataset already queued
existing_entries = data_manager.download_queue.list_entries()
dataset_ids = [e.dataset_id for e in existing_entries]

if "GSE180759" in dataset_ids:
    print("Dataset already in queue")
else:
    # Safe to add
    research_agent.validate_dataset_metadata("GSE180759", add_to_queue=True)
```

### Issue: Queue file corruption

**Symptoms**: `JSONDecodeError` when accessing queue

**Causes**:
- Interrupted write operation
- Disk full
- File permission issues

**Recovery**:
```bash
# Backup corrupted file
cp workspace/download_queue.jsonl workspace/download_queue.jsonl.backup

# Attempt manual repair (remove malformed lines)
# Or delete and recreate
rm workspace/download_queue.jsonl

# Queue will be recreated on next operation
```

---

## Best Practices

### 1. Always Validate Before Queueing

```python
# Good: Validation before queueing
result = research_agent.validate_dataset_metadata(
    dataset_id="GSE180759",
    add_to_queue=True  # Only adds if validation passes
)

# Bad: Blind queueing without validation
# (Not supported - validate_dataset_metadata handles both)
```

### 2. Use Priority for Time-Sensitive Downloads

```python
# High priority for urgent analysis
research_agent.validate_dataset_metadata(
    dataset_id="GSE180759",
    add_to_queue=True,
    priority=10  # Download first
)

# Normal priority for exploratory work
research_agent.validate_dataset_metadata(
    dataset_id="GSE109564",
    add_to_queue=True,
    priority=5  # Default
)
```

### 3. Check Queue Before Downloading

```python
# Check queue status first
result = supervisor.get_content_from_workspace(
    workspace="download_queue",
    status_filter="PENDING"
)

# Then download in priority order
pending = data_manager.download_queue.list_entries(status=DownloadStatus.PENDING)
pending_sorted = sorted(pending, key=lambda e: e.priority, reverse=True)

for entry in pending_sorted:
    data_expert.execute_download_from_queue(entry_id=entry.entry_id)
```

### 4. Clean Up Completed Entries

```python
# Periodically remove old completed entries
completed = data_manager.download_queue.list_entries(status=DownloadStatus.COMPLETED)

for entry in completed:
    # Keep entries for 7 days
    if (datetime.now() - entry.updated_at).days > 7:
        data_manager.download_queue.remove_entry(entry.entry_id)
```

### 5. Monitor Failed Downloads

```python
# Regular monitoring of failures
failed = data_manager.download_queue.list_entries(status=DownloadStatus.FAILED)

if len(failed) > 0:
    print(f"⚠️ {len(failed)} failed downloads")
    for entry in failed:
        print(f"  - {entry.dataset_id}: {entry.error_log[-1]}")  # Last error
```

---

## Future Enhancements

### Planned Features

1. **Automatic Retry Logic**
   - Exponential backoff for network failures
   - Configurable retry limits
   - Smart retry scheduling

2. **Batch Download Optimization**
   - Parallel downloads for multiple datasets
   - Bandwidth throttling
   - Download progress tracking

3. **Database-Specific Queues**
   - Separate queues for GEO, SRA, PRIDE
   - Database-specific validation rules
   - Custom URL builders per database

4. **Download Strategy Recommendations**
   - Automatic strategy detection by data_expert_assistant
   - Optimal file format selection
   - Size-based download optimization

5. **Queue Analytics**
   - Download success rates
   - Average download times
   - Failure pattern analysis

### Extension Points

```python
# Custom priority calculation
class CustomPriorityQueue(DownloadQueue):
    def calculate_priority(self, metadata: dict) -> int:
        """Calculate priority based on dataset characteristics."""
        priority = 5  # Default

        # Boost priority for large sample counts
        if metadata.get("n_samples", 0) > 100:
            priority += 2

        # Boost for specific organisms
        if metadata.get("organism") == "Homo sapiens":
            priority += 1

        return min(priority, 10)  # Cap at 10

# Custom validation
class CustomValidationQueue(DownloadQueue):
    def add_entry_with_validation(self, entry: DownloadQueueEntry) -> None:
        """Add entry with custom validation logic."""
        # Custom validation rules
        if entry.metadata.get("n_samples", 0) < 3:
            raise ValueError("Dataset must have at least 3 samples")

        super().add_entry(entry)
```

---

## Migration Notes

### From Synchronous Pattern (Pre-Phase 2)

**Before**:
```python
# research_agent directly passes metadata to data_expert
result = research_agent.validate_dataset_metadata("GSE180759")
# ... user manually passes info to data_expert ...
data_expert.download_geo_dataset("GSE180759")
```

**After**:
```python
# research_agent queues dataset
research_agent.validate_dataset_metadata("GSE180759", add_to_queue=True)

# supervisor/user checks queue
supervisor.get_content_from_workspace(workspace="download_queue")

# data_expert downloads from queue
data_expert.execute_download_from_queue(entry_id="queue_GSE180759_abc123")
```

### For New Database Support

**Example: Adding SRA database**:

```python
# 1. Extend schema
class SRADownloadQueueEntry(DownloadQueueEntry):
    database: Literal["sra"] = "sra"
    sra_run_ids: List[str] = Field(..., description="SRA run accessions")
    layout: str = Field(..., description="SINGLE or PAIRED")

# 2. Create SRA-specific URLs
sra_provider = SRAProvider()
urls = sra_provider.get_download_urls("SRR123456")

# 3. Add to queue
entry = SRADownloadQueueEntry(
    entry_id="queue_SRR123456_abc",
    dataset_id="SRR123456",
    database="sra",
    sra_run_ids=["SRR123456", "SRR123457"],
    raw_urls=urls,
    # ... other fields ...
)
data_manager.download_queue.add_entry(entry)
```

---

## See Also

### Wiki Pages
- [Architecture Overview (Wiki 18)](18-architecture-overview.md) - System-wide architecture
- [Two-Tier Caching Architecture (Wiki 39)](39-two-tier-caching-architecture.md) - Metadata caching strategy
- [Data Management (Wiki 20)](20-data-management.md) - Data management patterns

### Developer Documentation
- Data Expert Agent - See agent implementation in `lobster/agents/data_expert.py`
- Research Agent - See agent implementation in `lobster/agents/research_agent.py`
- Pydantic Schemas - See schema definitions in `lobster/core/schemas/`

### Code References
- `lobster/core/download_queue.py` - Queue implementation (342 lines)
- `lobster/core/schemas/download_queue.py` - Schema definitions
- `lobster/agents/data_expert.py` - Queue consumer
- `lobster/agents/research_agent.py` - Queue producer
- `lobster/tools/workspace_content_service.py` - Queue access tool

### Test Files
- `tests/unit/core/test_download_queue.py` - Unit tests
- `tests/integration/test_download_queue_workspace.py` - Integration tests
- `tests/unit/tools/test_workspace_content_service.py` - Tool tests

---

**Last Updated**: 2024-11-14 (Phase 2 completion)
**Authors**: Lobster AI Development Team
**Version**: 1.0.0
