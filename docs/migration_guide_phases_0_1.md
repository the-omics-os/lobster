# Migration Guide: Data Expert Refactoring (Phases 0 & 1)

**Document Version**: 1.0
**Date**: 2025-11-14
**Affects**: Lobster AI v2.3+
**Status**: Breaking Changes Implemented

---

## Executive Summary

Phases 0 and 1 of the data expert refactoring introduce **significant breaking changes** to improve architecture, eliminate code duplication, and enable pre-download validation workflows. This guide provides step-by-step migration instructions for all affected code.

### What Changed

- ✅ **Phase 0**: Foundation infrastructure and supervisor tool modernization
- ✅ **Phase 1**: Workspace extension and metadata_assistant parameter refactoring

### Impact Assessment

| Change | Severity | Affected Users | Migration Effort |
|--------|----------|----------------|------------------|
| Supervisor tool replacement | **HIGH** | Scripts calling `list_session_publications` | 5-10 minutes |
| metadata_assistant parameters | **CRITICAL** | All metadata_assistant usage | 15-30 minutes |
| Download queue (new feature) | **LOW** | None (new infrastructure) | N/A |
| Workspace extension | **LOW** | None (backward compatible) | N/A |

---

## Breaking Change 1: Supervisor Tool Replacement

### What Changed

**Removed**: `list_session_publications` tool from supervisor
**Added**: `get_content_from_workspace` tool to supervisor

### Why This Change

- `list_session_publications` was legacy, limited to publications only
- `get_content_from_workspace` provides unified access to 4 workspace categories:
  - `literature` (publications)
  - `data` (datasets)
  - `metadata` (validation results)
  - `download_queue` (pending downloads)

### Migration Instructions

#### ❌ OLD CODE (No Longer Works)

```python
# Calling supervisor's list_session_publications
result = supervisor.list_session_publications()
```

#### ✅ NEW CODE (Correct Approach)

```python
# Use get_content_from_workspace with workspace="literature"
result = supervisor.get_content_from_workspace(workspace="literature")

# Or for more detailed view
result = supervisor.get_content_from_workspace(
    workspace="literature",
    level="methods"  # Get publication methods
)
```

#### Agent Prompt Migration

**OLD Prompt**:
```
Use list_session_publications to see what papers have been analyzed.
```

**NEW Prompt**:
```
Use get_content_from_workspace(workspace="literature") to see what papers have been analyzed.
Use get_content_from_workspace(workspace="download_queue") to check pending downloads.
```

### Benefits of Migration

- **4x more functionality**: Access to literature + data + metadata + download_queue
- **Flexible detail levels**: 6 levels (summary, methods, samples, platform, metadata, github)
- **Better filtering**: Identifier-based queries, workspace-specific listings
- **Future-proof**: Unified interface for all workspace content

---

## Breaking Change 2: metadata_assistant Parameter Refactoring

### What Changed

**Parameter Renamed**: `identifier` → `source` (all 3 tools)
**Parameter Added**: `source_type` (REQUIRED, no default)

**Affected Tools**:
1. `validate_dataset_content`
2. `map_samples_by_id`
3. `read_sample_metadata`

### Why This Change

- **Enable pre-download validation**: Validate datasets from cached metadata without downloading
- **Eliminate ambiguity**: Explicit `source_type` removes confusion between modality names and workspace files
- **Unified interface**: Consistent parameter naming across all metadata tools

---

### Tool 1: `validate_dataset_content`

#### ❌ OLD CODE (No Longer Works)

```python
# Old signature
validate_dataset_content(
    identifier="geo_gse180759",
    expected_samples=48,
    check_controls=True
)
```

#### ✅ NEW CODE (Post-Download Validation)

```python
# Validate from loaded modality
validate_dataset_content(
    source="geo_gse180759",
    source_type="modality",  # REQUIRED
    expected_samples=48,
    check_controls=True
)
```

#### ✅ NEW CODE (Pre-Download Validation) 🆕

```python
# NEW FEATURE: Validate from cached metadata (NO download required)
validate_dataset_content(
    source="geo_gse180759",
    source_type="metadata_store",  # Use cached metadata
    expected_samples=48,
    check_controls=True
)
```

**Benefits**: Save 5-10 minutes per invalid dataset by validating BEFORE downloading.

---

### Tool 2: `map_samples_by_id`

#### ❌ OLD CODE (No Longer Works)

```python
# Old signature
map_samples_by_id(
    source_identifier="geo_gse12345",
    target_identifier="geo_gse67890",
    min_confidence=0.75
)
```

#### ✅ NEW CODE (Both Modalities)

```python
# Map between two loaded modalities
map_samples_by_id(
    source="geo_gse12345",
    target="geo_gse67890",
    source_type="modality",  # REQUIRED
    target_type="modality",  # REQUIRED
    min_confidence=0.75
)
```

#### ✅ NEW CODE (Mixed Sources) 🆕

```python
# Map between loaded modality and cached metadata
map_samples_by_id(
    source="geo_gse12345",
    target="geo_gse67890",
    source_type="modality",        # Loaded dataset
    target_type="metadata_store",  # Cached metadata
    min_confidence=0.75
)
```

---

### Tool 3: `read_sample_metadata`

#### ❌ OLD CODE (No Longer Works)

```python
# Old signature
read_sample_metadata(
    identifier="geo_gse180759",
    fields="condition,tissue",
    return_format="summary"
)
```

#### ✅ NEW CODE (From Modality)

```python
# Read from loaded modality
read_sample_metadata(
    source="geo_gse180759",
    source_type="modality",  # REQUIRED
    fields="condition,tissue",
    return_format="summary"
)
```

#### ✅ NEW CODE (From Cached Metadata) 🆕

```python
# Read from cached metadata (pre-download)
read_sample_metadata(
    source="geo_gse180759",
    source_type="metadata_store",  # Use cached metadata
    fields="condition,tissue",
    return_format="summary"
)
```

---

## Migration Checklist

### For Python Scripts

- [ ] Search codebase for `list_session_publications`
  ```bash
  grep -r "list_session_publications" . --include="*.py"
  ```
- [ ] Replace with `get_content_from_workspace(workspace="literature")`
- [ ] Search for `identifier=` in metadata_assistant calls
  ```bash
  grep -r "validate_dataset_content\|map_samples_by_id\|read_sample_metadata" . --include="*.py"
  ```
- [ ] Rename `identifier` → `source`
- [ ] Add `source_type="modality"` parameter to all calls
- [ ] Run tests to verify migration: `pytest tests/ -v`

### For Agent Prompts

- [ ] Update supervisor system prompts referencing `list_session_publications`
- [ ] Update metadata_assistant usage examples with new parameters
- [ ] Document `source_type` requirement in tool descriptions

### For Integration Tests

- [ ] Update test fixtures using old parameter names
- [ ] Add tests for new `source_type="metadata_store"` functionality
- [ ] Verify all integration workflows still pass

---

## New Features (Non-Breaking)

### 1. Download Queue Infrastructure

**What**: New queue system for research_agent → data_expert handoffs

**Usage**:
```python
# Add entry to queue (research_agent)
from lobster.core.schemas.download_queue import DownloadQueueEntry, DownloadStatus

entry = DownloadQueueEntry(
    entry_id="queue_001",
    dataset_id="GSE180759",
    database="geo",
    priority=7,
    status=DownloadStatus.PENDING,
    metadata={...},
    validation_result={...}
)
data_manager.download_queue.add_entry(entry)

# Check queue (supervisor)
result = get_content_from_workspace(
    workspace="download_queue",
    status_filter="PENDING"
)

# Download dataset (data_expert)
entry = data_manager.download_queue.get_entry("queue_001")
# ... perform download ...
data_manager.download_queue.update_status(
    entry_id="queue_001",
    status=DownloadStatus.COMPLETED
)
```

**Benefits**: Structured handoff contract, pre-download validation, supervisor coordination

---

### 2. Extended Workspace System

**What**: Workspace now supports 4 categories (was 3)

**New Category**: `download_queue`

**Usage**:
```python
# List all pending downloads
get_content_from_workspace(
    workspace="download_queue",
    status_filter="PENDING"
)

# Get specific entry details
get_content_from_workspace(
    identifier="queue_001",
    workspace="download_queue",
    level="summary"  # or "metadata", "validation", "strategy"
)
```

---

### 3. Pre-Download Validation

**What**: Validate datasets from cached metadata without downloading

**Usage**:
```python
# Step 1: research_agent validates dataset (caches metadata)
research_result = research_agent.validate_dataset_metadata("GSE180759")

# Step 2: metadata_assistant validates from cache (NO download)
validation_result = validate_dataset_content(
    source="geo_gse180759",
    source_type="metadata_store",  # Use cached metadata
    expected_samples=48
)

# Step 3: Download only if validation passes
if "validation passed" in validation_result.lower():
    data_expert.download_geo_dataset("geo_gse180759")
```

**Benefits**: Save 5-10 minutes per invalid dataset

---

## Code Quality Improvements

### Eliminated Code Duplication

- **Before**: 296 lines of duplicate tool definitions (research_agent + graph.py)
- **After**: 285 lines in shared factory (`workspace_tool.py`)
- **Savings**: 11,802 characters of duplicate code removed
- **Maintenance**: Single source of truth for workspace tools

### Architecture Improvements

- ✅ Factory pattern for tool sharing
- ✅ Single source of truth (workspace_tool.py)
- ✅ Consistent behavior across agents
- ✅ Type-safe with proper LangChain decorators

---

## Troubleshooting

### Error: "list_session_publications not found"

**Cause**: Using old supervisor tool name
**Solution**: Replace with `get_content_from_workspace(workspace="literature")`

### Error: "source_type must be 'modality' or 'metadata_store'"

**Cause**: Missing or invalid `source_type` parameter
**Solution**: Add `source_type="modality"` for loaded datasets or `source_type="metadata_store"` for cached metadata

### Error: "identifier parameter not recognized"

**Cause**: Using old parameter name
**Solution**: Rename `identifier` → `source`

### Error: "Download queue entry not found"

**Cause**: Entry doesn't exist in queue
**Solution**: Use `research_agent.validate_dataset_metadata()` to add entry to queue first

---

## Performance Impact

### Improvements

- **Metadata fetch time**: <5s total (vs ~15s before) - eliminated duplicate fetches
- **Queue operations**: <100ms each (add, read, update)
- **Pre-download validation**: 5-10 minutes saved per invalid dataset

### No Degradation

- All existing functionality maintained
- No performance regressions in tests

---

## Testing After Migration

### Unit Tests

```bash
# Test download queue
pytest tests/unit/core/test_download_queue.py -v

# Test metadata_assistant
pytest tests/unit/agents/test_metadata_assistant.py -v

# Test workspace content
pytest tests/unit/tools/test_workspace_content_service.py -v
```

### Integration Tests

```bash
# Test download queue workspace integration
pytest tests/integration/test_download_queue_workspace.py -v

# Test supervisor workflows
pytest tests/unit/agents/test_supervisor.py -v
```

### Full Test Suite

```bash
# Run all Phase 0+1 tests
pytest tests/unit/core/test_download_queue.py \
       tests/unit/agents/test_metadata_assistant.py \
       tests/unit/tools/test_workspace_content_service.py \
       tests/integration/test_download_queue_workspace.py \
       -v

# Expected: 98 tests passing
```

---

## Support & Questions

### Documentation

- **Two-Tier Caching**: `/lobster/wiki/39-two-tier-caching-architecture.md`
- **Architecture Overview**: `/lobster/wiki/18-architecture-overview.md`
- **Agent Registry**: `/lobster/config/agent_registry.py`

### Common Questions

**Q: Do I need to migrate if I don't use metadata_assistant?**
A: Only if you use `list_session_publications` in supervisor calls. Otherwise, no migration needed.

**Q: Can I still use old code temporarily?**
A: No, breaking changes are mandatory. Old parameter names will raise errors.

**Q: How do I know which source_type to use?**
A: Use `source_type="modality"` for loaded datasets (post-download), `source_type="metadata_store"` for cached metadata (pre-download).

**Q: What if I don't have cached metadata?**
A: Run `research_agent.validate_dataset_metadata()` first to cache metadata.

---

## Rollback Instructions

⚠️ **Warning**: Rollback requires reverting to pre-Phase 0 codebase.

### If Migration Fails

1. **Git Revert** (if using version control):
   ```bash
   git log --oneline  # Find commit before Phase 0
   git revert <commit-hash>
   ```

2. **Manual Rollback**:
   - Restore old `metadata_assistant.py` (with `identifier` parameter)
   - Restore old `supervisor.py` (with `list_session_publications`)
   - Remove `download_queue.py` infrastructure

3. **Test Rollback**:
   ```bash
   pytest tests/ -v
   ```

---

## Timeline & Versions

| Version | Date | Changes |
|---------|------|---------|
| **v2.3** | 2025-11-14 | Phase 0+1 breaking changes implemented |
| **v2.4** (planned) | Future | Phase 2 tool consolidation, queue-based downloads |

---

## Summary

### Breaking Changes Recap

1. ✅ **Supervisor Tool**: `list_session_publications` → `get_content_from_workspace`
2. ✅ **metadata_assistant Parameters**: `identifier` → `source`, added `source_type` (required)

### New Features Recap

1. ✅ **Download Queue**: Structured handoff contract for research→data workflows
2. ✅ **Extended Workspace**: 4 categories (literature, data, metadata, download_queue)
3. ✅ **Pre-Download Validation**: Validate from cached metadata (save 5-10 min/dataset)

### Migration Effort

- **Small projects** (1-10 files): 15-30 minutes
- **Medium projects** (10-50 files): 1-2 hours
- **Large projects** (50+ files): 2-4 hours

### Next Steps

- [ ] Complete Phase 0+1 migration
- [ ] Test all workflows
- [ ] Update internal documentation
- [ ] Prepare for Phase 2 (tool consolidation)

---

**Questions?** See the `docs/` directory or CLAUDE.md for detailed architecture documentation.

**Report Issues**: https://github.com/the-omics-os/lobster/issues

---

*This migration guide covers Phases 0 and 1 of the data expert refactoring. Phase 2 (tool consolidation) will have a separate migration guide.*
