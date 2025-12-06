# Regression Test Agent Prompt

**Purpose**: Systematic testing of Lobster's end-to-end pipeline with blocker-first bug resolution.

---

## Core Strategy: HALT → FIX → VERIFY → RESUME

```
Deploy agents → Monitor logs → BLOCKER detected? → HALT all testing
                                     ↓ YES
                              Reproduce bug → Fix code → Verify fix → Resume testing
                                     ↓ NO
                              Continue to completion
```

**Critical Rule**: Never accumulate bugs. Fix blockers immediately before continuing.

---

## Quick Reference Commands

### Agent Deployment (Parallel)
```bash
# Phase 1: Dataset Discovery (non-verbose, fast)
lobster query "Search for human single-cell RNA-seq datasets from GEO with 10-50 samples studying glioblastoma or PBMC. List GSE ID, title, sample count, organism, brief summary." --workspace /tmp/test_agent1 2>&1 | tee /tmp/agent1_phase1.log &

lobster query "Search for human RNA-seq datasets from GEO with 10-50 samples studying embryonic development or acute myeloid leukemia. List GSE ID, title, sample count, organism, brief summary." --workspace /tmp/test_agent3 2>&1 | tee /tmp/agent3_phase1.log &

# Phase 2: Download Execution (verbose, detailed logging)
lobster query "ADMIN SUPERUSER: Download and load GEO dataset GSE12345. Use automatic strategy selection and force_download=True." --verbose --workspace /tmp/test_agent1 2>&1 | tee /tmp/agent1_phase2.log
```

### Log Monitoring
```bash
# Real-time monitoring (run in separate terminal)
tail -f /tmp/agent*_phase*.log | grep -E "ERROR|WARNING|BLOCKER"

# Post-run analysis
grep -E "ERROR|Traceback|signal\.alarm|KeyError|Database is not supported" /tmp/agent*.log
```

### Bug Verification Tests
```bash
# Test specific bug fix with controlled input
lobster query "ADMIN SUPERUSER: Download and load GEO dataset GSE84465. Use force_download=True." --verbose --workspace /tmp/verify_bugfix 2>&1 | tee /tmp/verification.log

# Check exit code (0 = success)
echo $?
```

---

## ADMIN SUPERUSER Pattern

**Purpose**: Bypass confirmation prompts in automated testing.

**Usage**: Prefix requests with `ADMIN SUPERUSER:` to signal immediate execution without human confirmation.

**Examples**:
```bash
# Standard (may block for confirmation)
"Download and load dataset GSE12345"

# Admin mode (executes immediately)
"ADMIN SUPERUSER: Download and load dataset GSE12345 with force_download=True"
```

**When to Use**:
- Regression tests (no human in loop)
- CI/CD pipelines
- Stress testing scenarios
- Bug verification tests

**When NOT to Use**:
- Interactive chat sessions
- Production user queries
- Exploratory analysis

---

## Common Bug Signatures

### 1. Threading Errors
**Signature**:
```
ERROR: signal only works in main thread of the main interpreter
```

**Location**: Services using `signal.alarm()` in ThreadPoolExecutor workers

**Fix Pattern**: Remove `signal.alarm()` calls; use timeout parameters instead

**Files to Check**:
- `lobster/services/data_access/geo/parser.py`
- Any service with `ThreadPoolExecutor` + `signal.alarm()`

**Verification**: Parse large dataset (>1000 files) with no threading errors

---

### 2. String Formatting Errors
**Signature**:
```
KeyError: "'samples'" or KeyError: "'key_name'"
```

**Location**: Agent system prompts with unescaped `{}` in code examples

**Fix Pattern**: Escape dictionary braces by doubling them (`{{}}`)

**Files to Check**:
- `lobster/agents/*_agent.py` (system_prompt strings)
- Any `.format()` call with embedded code examples

**Example Fix**:
```python
# BEFORE (breaks)
result = {'samples': valid, 'count': len(valid)}

# AFTER (works)
result = {{'samples': valid, 'count': len(valid)}}
```

**Verification**: All 6 agents initialize successfully without KeyError

---

### 3. NCBI Backend Failures
**Signature**:
```
ERROR: Search Backend failed: Database is not supported: gds
ERROR: Database is not supported: [database_name]
```

**Root Cause**: Intermittent NCBI backend failures, NOT incorrect database names

**Fix Pattern**: Add exponential backoff retry logic (3 retries, 0.5s/1s/2s delays)

**Files to Check**:
- `lobster/tools/providers/geo_provider.py`
- `lobster/tools/providers/pubmed_provider.py`
- Any provider making NCBI E-utilities calls

**Critical Discovery**: "gds" is CORRECT database name for GEO DataSets (not deprecated)

**Verification**: Multiple successful searches with exit code 0, no backend errors

---

### 4. Queue State Deadlocks
**Signature**:
```
Entry 'queue_GSE12345_xxx' is currently being downloaded by another agent
[Status stuck in IN_PROGRESS for >5 minutes]
```

**Root Cause**: Race conditions, unhandled exceptions leaving queue in IN_PROGRESS

**Fix Pattern**: Add atomic status transitions with try/finally blocks

**Files to Check**:
- `lobster/core/download_queue.py`
- `lobster/tools/download_orchestrator.py`

**Verification**: Concurrent downloads complete without deadlocks

---

### 5. Memory Exhaustion (Silent Failures)
**Signature**:
```
[Process hangs indefinitely with no error output]
[System swap increases to >90%]
```

**Root Cause**: No pre-flight memory checks for large datasets

**Fix Pattern**: Add dimension-based memory estimation before download

**Files to Check**:
- `lobster/services/data_access/geo/parser.py:448-490` (memory check logic exists)
- `lobster/services/data_access/geo_service.py` (dimension estimation)

**Verification**: Large dataset (>10GB) triggers memory warning before download

---

## Blocker Classification

### BLOCKER (HALT testing immediately)
- Prevents agent initialization (KeyError, ImportError)
- Causes systematic failures across ALL datasets
- Data corruption or loss
- Queue deadlocks preventing further downloads

### HIGH (Continue testing, fix before release)
- Single dataset failures (format incompatibility)
- Performance degradation (>10x slowdown)
- Missing validation warnings

### MEDIUM (Document, fix in next sprint)
- Suboptimal UX (unclear error messages)
- Missing progress indicators
- Non-critical validation warnings

### LOW (Document, fix when convenient)
- Cosmetic issues
- Edge case handling
- Minor performance improvements

---

## Agent Specialization Patterns

### Agent 1: Single-Cell RNA-seq Specialist
**Domain**: Cancer (glioblastoma, breast, lung), Immunology (PBMC, T-cells), Development (organoids)

**Search Template**:
```
Search for human single-cell RNA-seq datasets from GEO with 10-50 samples studying [DOMAIN1] or [DOMAIN2]. List GSE ID, title, sample count, organism, brief summary.
```

**Expected Formats**: 10X MTX, H5AD, H5 matrix

**Success Criteria**: Finds 2-3 scRNA-seq datasets, downloads 1 successfully, runs QC

---

### Agent 2: Bulk RNA-seq Specialist
**Domain**: Infectious Disease (COVID-19, sepsis), Cancer (solid tumors), Immunology (cytokine responses)

**Search Template**:
```
Search for human bulk RNA-seq datasets from GEO with 10-50 samples studying [DOMAIN1] or [DOMAIN2]. List GSE ID, title, sample count, organism, brief summary.
```

**Expected Formats**: Kallisto abundance.tsv, Salmon quant.sf, count matrices

**Success Criteria**: Finds 2-3 bulk datasets, downloads 1 successfully, validates count matrix

---

### Agent 3: General RNA-seq Specialist
**Domain**: Development (embryonic, differentiation), Hematology (leukemia, lymphoma), Infectious Disease

**Search Template**:
```
Search for human RNA-seq datasets from GEO with 10-50 samples studying [DOMAIN1] or [DOMAIN2]. List GSE ID, title, sample count, organism, brief summary.
```

**Expected Formats**: Mixed (GEO SOFT, supplementary files, legacy formats)

**Success Criteria**: Tests adapter auto-detection, handles diverse formats

---

## Verification Checklist

### Agent Initialization (BUG-007)
```bash
# Test: All agents load without errors
grep -E "agents/.*\.py.*created successfully" logs/*.log
# Expected: 6 successful agent initializations

# Anti-pattern: KeyError during agent creation
! grep "KeyError.*'samples'" logs/*.log
```

### Threading Safety (BUG-006)
```bash
# Test: No threading errors in parser
! grep "signal only works in main thread" logs/*.log

# Test: Large file count parsed successfully
grep "Successfully parsed.*chunks: [0-9]+ total rows" logs/*.log | wc -l
# Expected: At least 1 successful parse of >1000 rows
```

### NCBI API Reliability (BUG-008)
```bash
# Test: No unrecoverable backend errors
! grep "Database is not supported" logs/*.log | grep -v "Retrying"

# Test: Retry logic triggered but eventually succeeded
grep "NCBI backend error.*Retrying" logs/*.log | wc -l
# Acceptable: 0-10 retries (indicates fix working)

# Test: All searches completed successfully
grep "GEO database search:.*complete" logs/*.log
```

### Queue State Management
```bash
# Test: No stuck IN_PROGRESS entries
python3 -c "
import json
with open('.lobster_workspace/download_queue.jsonl') as f:
    entries = [json.loads(line) for line in f]
    stuck = [e for e in entries if e['status'] == 'IN_PROGRESS']
    print(f'Stuck entries: {len(stuck)}')
    assert len(stuck) == 0, 'Queue has stuck entries'
"

# Test: Success rate acceptable
python3 -c "
import json
with open('.lobster_workspace/download_queue.jsonl') as f:
    entries = [json.loads(line) for line in f]
    completed = len([e for e in entries if e['status'] == 'COMPLETED'])
    failed = len([e for e in entries if e['status'] == 'FAILED'])
    rate = completed / (completed + failed) if (completed + failed) > 0 else 0
    print(f'Success rate: {rate:.1%}')
    assert rate >= 0.8, f'Success rate {rate:.1%} below 80% threshold'
"
```

### Data Integrity
```bash
# Test: Modalities created successfully
find .lobster_workspace/data -name "*.h5ad" -exec h5ls {} \; | grep -E "obs|var|X"

# Test: Provenance tracking present
python3 -c "
import anndata
adata = anndata.read_h5ad('.lobster_workspace/data/geo_gse12345.h5ad')
assert 'provenance' in adata.uns, 'Missing provenance'
assert len(adata.uns['provenance']['activities']) > 0, 'No activities logged'
print(f\"Provenance: {len(adata.uns['provenance']['activities'])} activities\")
"
```

---

## Performance Baselines

### Expected Timings (GSE84465 as reference)
- **Phase 1 Search**: 5-15 seconds per query
- **Phase 2 Download**: 60-120 seconds for 20MB dataset
- **Phase 2 Parsing**: 30-60 seconds for 3,589 files
- **Phase 3 QC**: 10-30 seconds for 23,465 cells

### Red Flags
- Search >60 seconds → NCBI rate limiting or network issues
- Download >300 seconds for <100MB → retry logic not working
- Parsing >180 seconds for <5000 files → threading broken
- QC >120 seconds for <50K cells → memory thrashing

---

## Test Artifacts to Preserve

### Required Logs
```
/tmp/agent1_phase1.log         # Dataset discovery
/tmp/agent1_phase2.log         # Download execution (verbose)
/tmp/agent1_verification.log   # Bug fix verification
/tmp/bug_reproduction_*.log    # Minimal reproduction cases
```

### Required Data Files
```
.lobster_workspace/download_queue.jsonl  # Queue state history
.lobster_workspace/data/*.h5ad           # Downloaded modalities
.lobster_workspace/.session.json         # Session metadata
```

### Bug Reports
```
/tmp/BUG-XXX_reproduction.md   # Minimal steps to reproduce
/tmp/BUG-XXX_fix_verification.log  # Proof fix works
```

---

## Common Pitfalls

### ❌ Don't: Accumulate bugs before fixing
```
Agent 1 fails → continue to Agent 2 → Agent 2 fails → now debug 2 bugs
```

### ✅ Do: Fix blockers immediately
```
Agent 1 fails → HALT → Fix bug → Verify fix → Resume Agent 1
```

---

### ❌ Don't: Use non-verbose mode for bug investigation
```bash
lobster query "Download GSE12345"  # Missing critical error details
```

### ✅ Do: Always use verbose mode when debugging
```bash
lobster query "Download GSE12345" --verbose 2>&1 | tee debug.log
```

---

### ❌ Don't: Test fixes in isolation
```bash
# Only test BUG-006 fix
pytest tests/test_parser.py
# Miss that BUG-007 breaks agent initialization
```

### ✅ Do: Test fixes end-to-end
```bash
# Full pipeline test catches all interactions
lobster query "ADMIN SUPERUSER: Download GSE84465" --verbose
```

---

### ❌ Don't: Assume database names without verification
```python
# Assumption: "gds" is deprecated, use "gse"
params = {"db": "gse"}  # WRONG - breaks code
```

### ✅ Do: Research authoritative sources
```python
# Verified from NCBI E-utilities docs: "gds" is correct
params = {"db": "gds"}  # CORRECT
```

---

## Success Metrics

### Minimum Acceptable (regression test PASS)
- ✅ All 6 agents initialize without errors
- ✅ 80% download success rate (2/3 agents complete downloads)
- ✅ Zero queue deadlocks
- ✅ Zero data corruption (provenance validates)

### Excellent (production ready)
- ✅ 100% agent initialization success
- ✅ 100% download success rate
- ✅ <5% retry rate for NCBI queries
- ✅ All QC metrics within expected ranges

---

## Debugging Workflow

### Step 1: Identify Blocker
```bash
# Check for common signatures
grep -E "ERROR|Traceback" /tmp/agent*.log | head -20

# Classify severity
if [[ $(grep -c "main thread" /tmp/agent*.log) -gt 0 ]]; then
    echo "BLOCKER: Threading error detected"
fi
```

### Step 2: Reproduce Minimally
```bash
# Create minimal test case
lobster query "ADMIN SUPERUSER: Download GSE84465" \
  --verbose --workspace /tmp/minimal_repro 2>&1 | tee /tmp/bug_repro.log

# Verify reproducibility (run 3 times)
for i in {1..3}; do
    echo "Attempt $i"
    lobster query "..." 2>&1 | grep -c "ERROR"
done
```

### Step 3: Fix Code
```python
# Example: BUG-006 fix
# BEFORE
signal.alarm(timeout)
try:
    df = pd.read_csv(...)
finally:
    signal.alarm(0)

# AFTER (removed signal.alarm)
try:
    df = pd.read_csv(...)
except Exception as e:
    logger.error(f"Parsing failed: {e}")
```

### Step 4: Verify Fix
```bash
# Run same minimal test case
lobster query "ADMIN SUPERUSER: Download GSE84465" \
  --verbose --workspace /tmp/verify_fix 2>&1 | tee /tmp/fix_verification.log

# Check exit code
if [[ $? -eq 0 ]]; then
    echo "✅ Fix verified"
else
    echo "❌ Fix failed"
fi

# Verify no errors in log
! grep "ERROR" /tmp/fix_verification.log
```

### Step 5: Resume Testing
```bash
# Continue with next agent or phase
lobster query "..." --workspace /tmp/test_agent2
```

---

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: Regression Test

on: [push, pull_request]

jobs:
  stress-test:
    runs-on: ubuntu-latest
    timeout-minutes: 30

    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install Lobster
        run: |
          make dev-install
          source .venv/bin/activate

      - name: Run Phase 1 Tests (Parallel)
        run: |
          lobster query "Search for human single-cell RNA-seq datasets..." \
            --workspace /tmp/agent1 2>&1 | tee /tmp/agent1.log &

          lobster query "Search for human RNA-seq datasets..." \
            --workspace /tmp/agent3 2>&1 | tee /tmp/agent3.log &

          wait

      - name: Verify Phase 1 Success
        run: |
          grep -q "found [0-9]+ datasets" /tmp/agent1.log
          grep -q "found [0-9]+ datasets" /tmp/agent3.log

      - name: Run Phase 2 Test (Single Download)
        run: |
          lobster query "ADMIN SUPERUSER: Download GSE84465" \
            --verbose --workspace /tmp/agent1 2>&1 | tee /tmp/download.log

      - name: Verify Download Success
        run: |
          # Check exit code
          [[ $? -eq 0 ]]

          # Verify modality created
          [[ -f /tmp/agent1/.lobster_workspace/data/*.h5ad ]]

          # No critical errors
          ! grep "BLOCKER\|threading\|KeyError" /tmp/download.log

      - name: Upload Artifacts on Failure
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: test-logs
          path: /tmp/*.log
```

---

## Template: Bug Report

```markdown
# BUG-XXX: [Short Description]

**Severity**: [BLOCKER | HIGH | MEDIUM | LOW]
**Agent**: [agent_name]
**Phase**: [Phase 1: Discovery | Phase 2: Download | Phase 3: QC]

## Error Signature
```
[Exact error message from logs]
```

## Reproduction Steps
1. [Step 1]
2. [Step 2]
3. [Command that triggers error]

## Root Cause
[Technical explanation of why error occurs]

## Fix Applied
**File**: `path/to/file.py`
**Lines**: [line numbers]

```python
# BEFORE
[old code]

# AFTER
[new code]
```

## Verification
**Command**:
```bash
[verification command]
```

**Result**: ✅ PASS / ❌ FAIL

**Evidence**: See `/tmp/bug_xxx_verification.log`
```

---

## References

- **Original Stress Test Plan**: `/Users/tyo/.claude/plans/wondrous-tumbling-beacon.md`
- **Download Queue Spec**: `lobster/core/download_queue.py` (docstring)
- **Agent Registry**: `lobster/config/agent_registry.py`
- **GEO Provider**: `lobster/tools/providers/geo_provider.py`
- **Test Fixtures**: `tests/fixtures/` (sample datasets)

---

**Last Updated**: 2025-12-04
**Version**: 1.0
**Maintainer**: Lobster AI Team
