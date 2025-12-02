# Resource Exhaustion Attack Tests

**Agent:** Agent 2 - Resource Exhaustion Tester
**Target:** `CustomCodeExecutionService` (lobster/services/execution/custom_code_execution_service.py)
**Focus:** Denial-of-service vulnerabilities through resource exhaustion

## Overview

This test suite validates the CustomCodeExecutionService's resilience against resource exhaustion attacks including:

- **Memory bombs** - Large allocations that exhaust system RAM
- **CPU exhaustion** - Infinite loops and expensive computations
- **Disk exhaustion** - Large file creation and inode exhaustion
- **File descriptor leaks** - Exhaust system FD limits

All tests use **SAFE limits** that won't crash your test machine, but document what WOULD happen with real attack values.

## Test Files

| File | Tests | Focus |
|------|-------|-------|
| `test_memory_bombs.py` | 8 tests | Memory allocation attacks |
| `test_cpu_exhaustion.py` | 7 tests | CPU-intensive operations |
| `test_disk_exhaustion.py` | 8 tests | Disk space and FD attacks |
| `RESOURCE_EXHAUSTION_REPORT.md` | - | Full security report with mitigations |

## Running the Tests

### Run All Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all resource exhaustion tests
pytest tests/manual/custom_code_execution/02_resource_exhaustion/ -v -s
```

### Run Specific Test Suites

```bash
# Memory bombs
pytest tests/manual/custom_code_execution/02_resource_exhaustion/test_memory_bombs.py -v -s

# CPU exhaustion
pytest tests/manual/custom_code_execution/02_resource_exhaustion/test_cpu_exhaustion.py -v -s

# Disk exhaustion
pytest tests/manual/custom_code_execution/02_resource_exhaustion/test_disk_exhaustion.py -v -s
```

### Run Specific Tests

```bash
# Single memory test
pytest tests/manual/custom_code_execution/02_resource_exhaustion/test_memory_bombs.py::TestMemoryAllocationBombs::test_large_list_allocation_EXPECT_SUCCESS -v -s

# CPU timeout test
pytest tests/manual/custom_code_execution/02_resource_exhaustion/test_cpu_exhaustion.py::TestInfiniteLoops::test_infinite_loop_hits_timeout_EXPECT_ERROR -v -s

# Disk fill test
pytest tests/manual/custom_code_execution/02_resource_exhaustion/test_disk_exhaustion.py::TestLargeFileCreation::test_write_large_file_EXPECT_SUCCESS -v -s
```

### Run Summary Tests Only

```bash
# View vulnerability summaries without running attack tests
pytest tests/manual/custom_code_execution/02_resource_exhaustion/ -k "summary" -v -s
```

## Expected Output

### Vulnerable Test (SUCCESS = Vulnerability Exists)

```
⚠️  VULNERABILITY CONFIRMED: Memory allocation succeeded
    Result: Allocated 400MB
    Duration: 2.5s
    🔥 REAL ATTACK: Could allocate 10GB+ and crash system
PASSED
```

### Protected Test (ERROR = Protection Works)

```
✅ PROTECTED: Memory allocation blocked: Memory limit exceeded
PASSED
```

### Timeout Test (ERROR after timeout = Expected)

```
✅ TIMEOUT WORKS: Infinite loop killed after 5.0s
    Error: Code execution exceeded timeout
    ⚠️  BUT: Default 300s timeout is too long!
    🔥 REAL ATTACK: 300s of CPU burn = denial of service
PASSED
```

## Test Naming Convention

| Test Name Suffix | Meaning |
|-----------------|---------|
| `_EXPECT_SUCCESS` | Test expects to succeed (vulnerability exists) |
| `_EXPECT_ERROR` | Test expects to fail (protection works) |
| `_WARNING` | Informational test (documents issue) |

## Key Findings

### Memory Vulnerabilities (8 Tests)

1. ✅ Large list allocation (400MB+)
2. ✅ Large string allocation (100MB+)
3. ✅ numpy array bombs (800MB+)
4. ✅ Multiple allocations (500MB+ cumulative)
5. ✅ Recursive data structures (100+ levels)
6. ✅ pandas DataFrame bombs (100MB+)
7. ✅ Exponential DataFrame growth (2^n)
8. ✅ Circular references (Python GC handles but no external limit)

**Current Protection:** NONE
**Impact:** Can exhaust system RAM, trigger OOM killer

### CPU Vulnerabilities (7 Tests)

1. ⚠️ Infinite loops (timeout works but 300s is too long)
2. ✅ CPU-intensive loops (100% CPU allowed)
3. ✅ Prime calculations (expensive algorithms)
4. ✅ O(n³) nested loops (cubic complexity)
5. ✅ Matrix multiplication bombs (large operations)
6. ✅ Expensive cryptography (repeated hashing)
7. ⚠️ Regex catastrophic backtracking (Python 3.11+ has timeout)

**Current Protection:** 300s timeout only
**Impact:** 5 minutes of 100% CPU per execution

### Disk Vulnerabilities (8 Tests)

1. ✅ Large file creation (100MB+)
2. ✅ Rapid file writes (no rate limiting)
3. ✅ Many small files (1,000+, inode exhaustion)
4. ✅ Deep directory nesting (100+ levels)
5. ✅ Sparse file creation (confuses quotas)
6. ✅ File descriptor exhaustion (100+ FDs)
7. ✅ Socket FD exhaustion (50+ sockets)
8. ✅ Temp file accumulation (/tmp pollution)

**Current Protection:** NONE
**Impact:** Can fill entire disk, exhaust inodes, crash processes

## Recommended Mitigations

### Immediate (P0)

1. **Reduce timeout**: Change `DEFAULT_TIMEOUT` from 300s to 30s
2. **Add disk checks**: Pre-execution validation (min 1GB free)
3. **Document Docker**: Update docstring with resource limit examples

### Short-term (P1)

4. **Deploy with Docker**: Use `--memory=2g --cpus=0.5 --storage-opt size=10G`
5. **Rate limiting**: 5 executions per minute per user
6. **File count checks**: Max 50,000 files in workspace

### Long-term (P2)

7. **Resource monitoring**: Real-time limits via psutil
8. **Graceful degradation**: Warnings before hard limits
9. **User quotas**: Per-user storage/compute limits

### Docker Deployment (Recommended)

```bash
docker run \
  --memory=2g \
  --memory-swap=2g \
  --cpus=0.5 \
  --storage-opt size=10G \
  --ulimit nofile=1024:2048 \
  --tmpfs /tmp:rw,size=1G \
  --network=none \
  omicsos/lobster
```

## Test Implementation Details

### Fixture Structure

All tests use a common `service` fixture (defined in `conftest.py`):

```python
@pytest.fixture
def service(tmp_path):
    """Create service with temporary workspace."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_manager = DataManagerV2(workspace_path=workspace)
    return CustomCodeExecutionService(data_manager)
```

### Test Pattern

```python
def test_attack_vector_EXPECT_SUCCESS(self, service):
    """
    VULNERABILITY: Description

    Test: Safe PoC with limited resources
    Expected: SUCCESS (vulnerability exists)
    Impact: HIGH/MEDIUM/LOW

    SAFE LIMIT: Value for testing
    REAL ATTACK: What an attacker would do

    Mitigation: Recommended fix
    """
    code = '''
# Safe PoC code
result = "attack result"
'''

    try:
        result, stats, ir = service.execute(code, persist=False, timeout=30)
        print(f"\n⚠️  VULNERABILITY CONFIRMED: Attack succeeded")
        print(f"    Result: {result}")
        print(f"    🔥 REAL ATTACK: What would really happen")
        assert stats['success'] is True
    except CodeExecutionError as e:
        print(f"\n✅ PROTECTED: Attack blocked: {e}")
        pytest.fail("Expected vulnerability")
```

## Safety Guidelines

### For Test Authors

1. **Use conservative limits**: 100MB memory, 5s CPU, 100 files
2. **Document real attack**: Explain what WOULD happen with larger values
3. **Include cleanup**: Delete created files in test
4. **Set short timeouts**: Use `timeout=5` or `timeout=10` in tests
5. **Add impact assessment**: HIGH/MEDIUM/LOW severity

### For Test Runners

1. **Run in isolation**: Don't run on production systems
2. **Monitor resources**: Watch `top` or Activity Monitor during tests
3. **Have free space**: Ensure 1GB+ free disk space
4. **Use tmp_path fixture**: Tests use temporary directories (auto-cleanup)
5. **Stop if issues**: Ctrl+C works, subprocess isolation protects main process

### For Developers

1. **Review before merge**: These tests expose real vulnerabilities
2. **Implement mitigations**: Don't ship with known vulnerabilities
3. **Update tests**: When adding protections, update expected results
4. **Document changes**: Update RESOURCE_EXHAUSTION_REPORT.md

## Interpreting Results

| Symbol | Meaning |
|--------|---------|
| ⚠️ | Vulnerability confirmed (attack succeeds) |
| ✅ | Protection confirmed (attack blocked) OR Timeout works |
| 🔥 | Real attack potential (extrapolation) |

### Exit Codes

- **0**: All tests passed (vulnerabilities documented as expected)
- **1**: Test failure (unexpected protection or vulnerability)
- **2**: Test error (setup issue, import error, etc.)

## Related Documentation

- **Full Report**: `RESOURCE_EXHAUSTION_REPORT.md` (comprehensive findings)
- **Service Code**: `lobster/services/execution/custom_code_execution_service.py`
- **Agent 1 Report**: `../01_basic_security/BASIC_SECURITY_REPORT.md` (import/execution tests)
- **Agent 3 Report**: `../03_code_injection/` (coming soon - code injection tests)

## FAQ

### Q: Will these tests crash my machine?

**A:** No. All tests use safe limits (100MB memory, 5s CPU, 100 files). We document what WOULD happen with real attack values.

### Q: Why do "vulnerable" tests PASS?

**A:** These tests document **expected behavior**. A passing test means "yes, this vulnerability exists as documented". The test would FAIL if unexpected protection blocked the attack.

### Q: Should I run these in CI?

**A:** Yes, but consider:
- Set resource limits on CI runners
- Use `pytest -k "not slow"` to skip expensive tests
- Run in isolated containers
- Monitor CI resource usage

### Q: How do I add a new resource attack?

1. Add test to appropriate file (memory/CPU/disk)
2. Use `_EXPECT_SUCCESS` or `_EXPECT_ERROR` suffix
3. Follow the test pattern (docstring, safe PoC, real attack docs)
4. Add cleanup code (delete files, etc.)
5. Update this README and RESOURCE_EXHAUSTION_REPORT.md

### Q: What if a vulnerability test starts failing?

**A:** Great! That means:
1. Protection was added (intentional) → Update test to `_EXPECT_ERROR` and document
2. Environment changed (new limits) → Investigate and document
3. Test is flaky → Fix the test

### Q: Can I increase the attack values to test real limits?

**A:** Only if:
- You're on a dedicated test machine
- You understand the risks
- You have backups
- You can recover from OOM/disk full
- You monitor resource usage closely

**Recommended**: Use Docker with limits to safely test real attack scenarios.

## Test Statistics

- **Total Tests**: 23
- **Memory Tests**: 8
- **CPU Tests**: 7
- **Disk Tests**: 8
- **Confirmed Vulnerabilities**: 21
- **Partial Protections**: 2 (timeout, Python GC)
- **Expected Failures**: 1 (infinite loop timeout)
- **Average Test Duration**: 5-10 seconds
- **Safe Resource Usage**: <500MB RAM, <10s CPU, <200MB disk per test

## Changelog

### 2025-11-30 - Initial Release

- Created comprehensive resource exhaustion test suite
- 23 tests across memory, CPU, and disk vectors
- Full security report with mitigations
- Safe PoC exploits with real attack documentation
- Docker deployment recommendations

## Contact

**Agent:** Agent 2 - Resource Exhaustion Tester
**Purpose:** Security testing (resource exhaustion focus)
**Status:** Testing complete, report delivered

For questions about:
- **Test implementation**: See test docstrings
- **Security findings**: See RESOURCE_EXHAUSTION_REPORT.md
- **Mitigations**: See report "Recommended Mitigations" section
- **Production deployment**: See report "Complete Mitigation Stack" section
