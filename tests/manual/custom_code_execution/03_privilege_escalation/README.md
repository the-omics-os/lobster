# Privilege Escalation Security Tests

**Agent 3: Privilege Escalation Tester**

This directory contains comprehensive security tests for subprocess breakout and privilege escalation vulnerabilities in the CustomCodeExecutionService.

---

## Quick Start

```bash
# Run all privilege escalation tests
pytest tests/manual/custom_code_execution/03_privilege_escalation/ -v -s

# Run specific test files
pytest tests/manual/custom_code_execution/03_privilege_escalation/test_subprocess_breakout.py -v -s
pytest tests/manual/custom_code_execution/03_privilege_escalation/test_signal_manipulation.py -v -s
pytest tests/manual/custom_code_execution/03_privilege_escalation/test_process_injection.py -v -s

# Run specific vulnerability tests
pytest tests/manual/custom_code_execution/03_privilege_escalation/ -v -s -k "SIGKILL"
pytest tests/manual/custom_code_execution/03_privilege_escalation/ -v -s -k "exec_builtin"
```

---

## Test Files

### 1. test_subprocess_breakout.py (18KB, 10 tests)

**Purpose:** Test process creation, import bypass, and resource exhaustion.

**Test Classes:**
- `TestSubprocessBreakout` - Process creation and import bypass (7 tests)
- `TestProcessResourceExhaustion` - Resource exhaustion attacks (3 tests)

**Critical Vulnerabilities Tested:**
- ⚠️ Fork bomb (Linux only)
- ⚠️ Multiprocessing process spawning
- ⚠️ Threading bomb
- ⚠️ Subprocess import (should be blocked)
- ⚠️ os.system() via getattr() bypass
- ⚠️ exec() builtin bypass
- ⚠️ __import__() builtin bypass
- ⚠️ Memory exhaustion
- ⚠️ Infinite loop timeout
- ⚠️ File descriptor exhaustion

**Expected Results:**
- Fork: BLOCKED on macOS, SUCCESS on Linux
- Multiprocessing: SUCCESS (vulnerability)
- Threading: SUCCESS (vulnerability)
- Subprocess import: BLOCKED (protected)
- exec/eval bypass: SUCCESS (CRITICAL vulnerability)
- Resource limits: NONE (vulnerability)

---

### 2. test_signal_manipulation.py (21KB, 10 tests)

**Purpose:** Test parent process discovery and signal-based attacks.

**Test Classes:**
- `TestParentProcessDiscovery` - Parent PID discovery (2 tests)
- `TestSignalAttacks` - Signal injection attacks (4 tests)
- `TestAdvancedProcessAttacks` - Advanced process manipulation (4 tests)

**Critical Vulnerabilities Tested:**
- ⚠️ Parent PID discovery (os.getppid)
- ⚠️ Parent cmdline reading (/proc, Linux only)
- ⚠️ SIGTERM attack capability
- ⚠️ SIGKILL attack capability (CRITICAL)
- ⚠️ SIGSTOP attack capability
- ⚠️ Signal handler hijacking (subprocess only)
- ⚠️ Process tree enumeration
- ⚠️ Orphan process creation
- ⚠️ Process priority manipulation
- ⚠️ Environment variable exfiltration

**Expected Results:**
- Parent PID: SUCCESS (vulnerability)
- Parent cmdline: BLOCKED on macOS, SUCCESS on Linux
- Signal attacks: ALL AVAILABLE (CRITICAL)
- Environment access: SUCCESS (API key exfiltration possible)

---

### 3. test_process_injection.py (21KB, 10 tests)

**Purpose:** Test memory injection and inter-process communication attacks.

**Test Classes:**
- `TestMemoryInjection` - Direct memory access (4 tests)
- `TestIPCMechanisms` - IPC channel creation (3 tests)
- `TestDockerEscapes` - Docker-specific attacks (3 tests)

**Critical Vulnerabilities Tested:**
- ⚠️ /proc/PID/mem access (Linux only)
- ⚠️ ptrace() system call
- ⚠️ Shared memory (mmap)
- ⚠️ ctypes memory manipulation
- ⚠️ Unix domain sockets
- ⚠️ Named pipes (FIFO)
- ⚠️ System V IPC
- ⚠️ Docker socket access
- ⚠️ Container environment detection
- ⚠️ Linux capabilities check

**Expected Results:**
- /proc/PID/mem: BLOCKED (permissions)
- ptrace: BLOCKED (macOS SIP, Linux permissions)
- IPC channels: SUCCESS (vulnerability)
- Docker socket: VARIES (depends on mounting)

---

## Vulnerability Summary

| Category | Total Tests | Critical | High | Medium | Protected |
|----------|------------|----------|------|--------|-----------|
| **Subprocess Breakout** | 10 | 3 | 2 | 0 | 5 |
| **Signal Manipulation** | 10 | 3 | 3 | 4 | 0 |
| **Process Injection** | 10 | 1 | 0 | 5 | 4 |
| **TOTAL** | 30 | 7 | 5 | 9 | 9 |

---

## Critical Findings

### 🔴 CRITICAL Vulnerabilities (Fix Immediately)

1. **SIGKILL Attack** - User code can kill parent Lobster process
   - Test: `test_signal_manipulation.py::test_sigkill_capability_EXPECT_SUCCESS`
   - Fix: Requires PID namespaces (Docker)

2. **exec() Bypass** - Static import checks bypassed
   - Test: `test_subprocess_breakout.py::test_exec_builtin_EXPECT_SUCCESS`
   - Fix: Disable __builtins__['exec']

3. **__import__() Bypass** - FORBIDDEN_MODULES bypassed
   - Test: `test_subprocess_breakout.py::test_import_builtin_EXPECT_SUCCESS`
   - Fix: Disable __builtins__['__import__']

4. **os.system() via getattr** - Runtime import bypass
   - Test: `test_subprocess_breakout.py::test_os_system_via_getattr_EXPECT_SUCCESS`
   - Fix: Disable getattr or use Docker

---

## Detailed Report

See: **PRIVILEGE_ESCALATION_REPORT.md** for comprehensive analysis including:
- Security model explanation
- Threat scenarios
- Mitigation strategies (short/medium/long-term)
- Docker hardening guide
- gVisor deployment instructions

---

## Safety Notes

⚠️ **ALL TESTS ARE DETECTION-ONLY**

- Tests check if attack is POSSIBLE, not execute it
- No signals are actually sent to parent process
- No memory is actually corrupted
- No resources are actually exhausted
- Safe to run in production environments (detection only)

**Example:**
```python
# Test checks capability, doesn't execute
if hasattr(signal, 'SIGKILL'):
    result = "VULNERABILITY: Can send SIGKILL"
    # ⚠️ NOT EXECUTED - test only
```

---

## Integration with Security Test Suite

This directory is part of the comprehensive CustomCodeExecutionService security assessment:

```
tests/manual/custom_code_execution/
├── 01_filesystem_access/      # Agent 1: Filesystem tests
├── 02_network_access/         # Agent 2: Network tests
├── 03_privilege_escalation/   # Agent 3: Privilege escalation (THIS)
├── 04_resource_limits/        # Agent 4: Resource limits
└── 05_data_exfiltration/      # Agent 5: Data exfiltration
```

---

## Expected Test Execution Time

- **test_subprocess_breakout.py**: ~30 seconds
- **test_signal_manipulation.py**: ~40 seconds
- **test_process_injection.py**: ~35 seconds
- **Total**: ~105 seconds (< 2 minutes)

All tests respect timeout limits and are designed for fast feedback.

---

## Platform-Specific Behavior

| Test | macOS | Linux | Windows |
|------|-------|-------|---------|
| Fork bomb | BLOCKED | SUCCESS | N/A |
| /proc access | N/A | SUCCESS | N/A |
| Signal attacks | SUCCESS | SUCCESS | VARIES |
| IPC channels | SUCCESS | SUCCESS | VARIES |

Tests automatically detect platform and adjust expectations.

---

## Questions?

See: **PRIVILEGE_ESCALATION_REPORT.md** for:
- Detailed vulnerability analysis
- Risk assessment
- Mitigation roadmap
- Docker deployment guide
