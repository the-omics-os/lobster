# Workspace Pollution Test Suite

**Test Category:** Data Integrity and Persistence Attacks
**Tester:** Agent 7 (Workspace Pollution Specialist)
**Date:** 2025-11-30

## Overview

This test suite examines workspace corruption and provenance tampering vulnerabilities in CustomCodeExecutionService. The tests demonstrate that user code has **unrestricted write access to the entire workspace**, enabling:

- File deletion (queues, session data, H5AD files)
- File modification (corruption, credential theft)
- Provenance tampering (breaking reproducibility)
- IR injection (malicious code in exported notebooks)
- Cache poisoning (fake literature)
- Workspace destruction (DoS attacks)

## Test Files

### 1. `test_workspace_corruption.py`
Tests for file system attacks on workspace structure.

**Test Classes:**
- `TestFileDelection` - Delete critical files
- `TestFileModification` - Corrupt data structures
- `TestLockFileManipulation` - Bypass concurrency control
- `TestCachePoisoning` - Inject fake data
- `TestWorkspaceStructureDestruction` - Catastrophic damage

**Total Tests:** 12

### 2. `test_provenance_tampering.py`
Tests for provenance integrity and reproducibility attacks.

**Test Classes:**
- `TestProvenanceInjection` - Inject fake analysis steps
- `TestProvenanceModification` - Modify existing records
- `TestProvenanceDeletion` - Delete audit trail
- `TestIRTampering` - Inject malicious code into IR
- `TestSessionHistoryManipulation` - Alter command history

**Total Tests:** 10

### 3. `WORKSPACE_POLLUTION_REPORT.md`
Comprehensive vulnerability report with:
- Attack vector analysis
- Root cause analysis
- Real-world attack scenarios
- Remediation plan (4 phases)
- Business impact assessment
- Compliance implications

## Running Tests

### Run All Tests
```bash
cd /Users/tyo/GITHUB/omics-os/lobster
pytest tests/manual/custom_code_execution/07_workspace_pollution/ -v -s
```

### Run Specific Test Files
```bash
# Workspace corruption tests
pytest tests/manual/custom_code_execution/07_workspace_pollution/test_workspace_corruption.py -v -s

# Provenance tampering tests
pytest tests/manual/custom_code_execution/07_workspace_pollution/test_provenance_tampering.py -v -s
```

### Run Specific Test
```bash
pytest tests/manual/custom_code_execution/07_workspace_pollution/test_workspace_corruption.py::TestFileDelection::test_delete_download_queue_EXPECT_SUCCESS -v -s
```

## Expected Results

### BEFORE Mitigations
All tests should **PASS** (indicating vulnerabilities exist):

```
✅ test_delete_download_queue_EXPECT_SUCCESS - Queue deleted
✅ test_delete_session_file_EXPECT_SUCCESS - Session deleted
✅ test_delete_all_h5ad_files_EXPECT_SUCCESS - Data destroyed
✅ test_corrupt_queue_file_EXPECT_SUCCESS - Queue corrupted
✅ test_modify_session_credentials_EXPECT_SUCCESS - Credentials hijacked
✅ test_inject_fake_analysis_step_EXPECT_SUCCESS - Provenance forged
✅ test_modify_code_templates_EXPECT_SUCCESS - IR injected with malicious code
... (22 total tests pass)
```

### AFTER Mitigations
Tests should **FAIL** with security errors:

```
❌ test_delete_download_queue_EXPECT_SUCCESS
    CodeExecutionError: Workspace integrity violated
❌ test_modify_session_credentials_EXPECT_SUCCESS
    PermissionError: Cannot write to protected file: .session.json
❌ test_inject_fake_analysis_step_EXPECT_SUCCESS
    PermissionError: Cannot write to protected file: .lobster/provenance/analysis_log.jsonl
... (tests fail as expected)
```

## Key Findings

### Severity: CRITICAL

**Vulnerability Summary:**
- ✅ Complete workspace write access (no restrictions)
- ✅ No file integrity checks
- ✅ No provenance immutability
- ✅ No isolation mechanisms
- ✅ No backup/recovery system

**Impact:**
1. **Data Integrity:** Files can be deleted or corrupted
2. **Reproducibility:** Provenance can be forged or deleted
3. **Security:** Credentials can be stolen from .session.json
4. **Scientific Integrity:** Analysis history can be falsified
5. **Supply Chain:** Exported notebooks can contain malicious code

**Business Impact:**
- Cannot meet HIPAA/GDPR/SOC2 requirements
- Enterprise sales blocked (70% of target market)
- Research integrity concerns (academic adoption risk)
- Revenue targets unachievable ($810K ARR → $0)

## Detailed Analysis

See `WORKSPACE_POLLUTION_REPORT.md` for:
- Complete attack vector analysis (25+ vectors)
- Root cause analysis
- Real-world attack scenarios (4 scenarios)
- Comprehensive remediation plan (4 phases)
- Compliance implications
- Cost-benefit analysis

## Remediation Roadmap

### Phase 1: Immediate (Weeks 1-2)
- File integrity checks
- Protected file list
- Provenance signing
- Pre-execution backups

### Phase 2: Comprehensive (Weeks 3-4)
- Docker-based isolation
- Database-backed provenance
- File monitoring

### Phase 3: Enterprise (Weeks 5-8)
- Workspace versioning
- Disk quotas
- Security alerts

### Phase 4: Compliance (Weeks 9-12)
- HIPAA enhancements
- Audit reporting
- Certifications (SOC2, GDPR)

**Total Effort:** ~160 engineering hours (3 months for 1 senior engineer)

**ROI:** Unblocks $810K ARR + enables enterprise deals ($18K-$30K ACV)

## Test Design Philosophy

These tests follow the **"EXPECT_SUCCESS"** naming convention to indicate that vulnerabilities are **expected to exist**. When tests PASS, it means:

1. The attack succeeded (vulnerability confirmed)
2. User code had insufficient restrictions
3. Security controls are missing

This is the OPPOSITE of normal testing where passing = good. Here:
- **PASS = Vulnerability exists** ⚠️
- **FAIL = Protection working** ✅

## Related Security Testing

- [Agent 1] Import Restrictions
- [Agent 2] API Abuse
- [Agent 3] Resource Exhaustion
- [Agent 4] Code Injection
- [Agent 5] Side Channels
- [Agent 6] Execution Isolation
- **[Agent 7] Workspace Pollution** ← YOU ARE HERE

## Contact

Questions about these tests? See:
- `WORKSPACE_POLLUTION_REPORT.md` for detailed analysis
- `tests/manual/custom_code_execution/README.md` for overall test strategy
- Project CLAUDE.md for security architecture

---

**Last Updated:** 2025-11-30
**Status:** Tests complete, vulnerabilities documented, remediation plan provided
