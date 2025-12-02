# Supply Chain Attack Test Suite

**Test Category:** Supply Chain Vulnerabilities (Python Import System)
**Target Service:** `lobster/services/execution/custom_code_execution_service.py`
**Severity:** CRITICAL
**Agent:** Agent 4 - Supply Chain Attack Tester

---

## Overview

This test suite validates vulnerabilities in Python's import system related to the CustomCodeExecutionService. The service adds the workspace directory to `sys.path[0]`, allowing malicious modules in the workspace to shadow ANY standard library or installed package.

**Critical Finding:** Line 340 in `custom_code_execution_service.py`:
```python
sys.path.insert(0, str(WORKSPACE))  # ⚠️ CRITICAL VULNERABILITY
```

---

## Test Files

### 1. `test_malicious_imports.py`
**Focus:** Module shadowing and package poisoning attacks

**Attack Vectors:**
- Standard library shadowing (json, os, sys)
- Scientific package shadowing (numpy, pandas, scanpy)
- Import-time code execution (`__init__.py`)
- Credential theft at import
- Typosquatting attacks (reqeusts vs requests)
- Lobster internal module shadowing (pathlib)

**Test Count:** 10 test cases
**Severity:** CRITICAL

**Key Findings:**
- ✅ Can shadow json to steal API keys from config files
- ✅ Can shadow numpy to manipulate scientific data
- ✅ Can shadow os to bypass security checks
- ✅ Code executes at import time (before user code runs)
- ✅ Can shadow pathlib (used by Lobster core) to break system

**Example Vulnerability:**
```python
# workspace/json.py (malicious)
import json as _real_json

def load(fp):
    data = _real_json.load(fp)
    # Exfiltrate credentials
    with open('.stolen.json', 'w') as f:
        _real_json.dump(data, f)
    return data
```

### 2. `test_syspath_manipulation.py`
**Focus:** Runtime sys.path manipulation and dynamic imports

**Attack Vectors:**
- Runtime `sys.path.insert(0, malicious_dir)`
- Runtime `sys.path.append(malicious_dir)`
- Parent directory escape (`../`)
- /tmp directory exploitation
- sys.modules cache poisoning
- Dynamic import bypass (`__import__`, `importlib`)
- Obfuscated imports via `getattr(__builtins__, '__import__')`

**Test Count:** 9 test cases
**Severity:** CRITICAL

**Key Findings:**
- ✅ AST validation only checks STATIC imports
- ✅ Cannot prevent RUNTIME sys.path modifications
- ✅ Can inject external malicious directories
- ✅ Can poison sys.modules cache (no file system traces)
- ✅ Can bypass import blocking via dynamic imports

**Example Vulnerability:**
```python
import sys
# Inject external malicious directory
sys.path.insert(0, '/tmp/malicious_modules')
import backdoor  # Loaded from /tmp
```

### 3. `test_package_shadowing.py`
**Focus:** Standard library and security-critical module shadowing

**Attack Vectors:**
- subprocess shadowing (command logging)
- sys module shadowing (stdout interception)
- json shadowing (credential theft)
- pickle shadowing (object injection)
- io module shadowing (file operation logging)
- tempfile shadowing (predictable paths)
- pathlib shadowing (Lobster internal compromise)

**Test Count:** 11 test cases
**Severity:** CRITICAL

**Key Findings:**
- ✅ Can shadow subprocess to log all system commands
- ✅ Can shadow sys to intercept all stdout
- ✅ Can shadow json to steal credentials from configs
- ✅ Can shadow pickle to inject backdoors into objects
- ✅ Can shadow io to log all file operations
- ✅ Can shadow pathlib to break Lobster core functionality

**Example Vulnerability:**
```python
# workspace/sys.py (malicious)
class _StdoutInterceptor:
    def write(self, text):
        # Log all output to hidden file
        with open('.stdout_intercept.txt', 'a') as f:
            f.write(text)

stdout = _StdoutInterceptor()
```

---

## Running the Tests

### Run All Supply Chain Tests
```bash
cd /Users/tyo/GITHUB/omics-os/lobster
pytest tests/manual/custom_code_execution/04_supply_chain/ -v
```

### Run Specific Test File
```bash
# Malicious imports
pytest tests/manual/custom_code_execution/04_supply_chain/test_malicious_imports.py -v

# sys.path manipulation
pytest tests/manual/custom_code_execution/04_supply_chain/test_syspath_manipulation.py -v

# Package shadowing
pytest tests/manual/custom_code_execution/04_supply_chain/test_package_shadowing.py -v
```

### Run with Detailed Output
```bash
pytest tests/manual/custom_code_execution/04_supply_chain/ -v -s
```

### Run Specific Test
```bash
pytest tests/manual/custom_code_execution/04_supply_chain/test_malicious_imports.py::TestMaliciousStandardLibraryShadowing::test_json_shadowing_EXPECT_SUCCESS -v
```

---

## Test Naming Convention

**Pattern:** `test_<vulnerability>_EXPECT_<result>`

**Examples:**
- `test_json_shadowing_EXPECT_SUCCESS` - Expects vulnerability to succeed (confirming it exists)
- `test_subprocess_command_logging_EXPECT_SUCCESS` - Expects malicious code to execute
- `test_builtin_import_bypass_EXPECT_WARNING` - Expects warning but execution continues

**Interpretation:**
- **PASS** = Vulnerability confirmed (test shows exploit works)
- **FAIL** = Vulnerability fixed (exploit prevented)

**Note:** These are "negative tests" - they test that bad things CAN happen.

---

## Understanding Test Results

### Expected Results (Current State)

```
tests/manual/custom_code_execution/04_supply_chain/test_malicious_imports.py
  ✅ test_json_shadowing_EXPECT_SUCCESS PASSED
  ✅ test_os_shadowing_bypasses_security_EXPECT_SUCCESS PASSED
  ✅ test_numpy_data_manipulation_EXPECT_SUCCESS PASSED
  ... (all tests PASS)
```

**Interpretation:** All vulnerabilities confirmed. System is vulnerable.

### After Remediation (Desired State)

```
tests/manual/custom_code_execution/04_supply_chain/test_malicious_imports.py
  ❌ test_json_shadowing_EXPECT_SUCCESS FAILED
  ❌ test_os_shadowing_bypasses_security_EXPECT_SUCCESS FAILED
  ❌ test_numpy_data_manipulation_EXPECT_SUCCESS FAILED
  ... (all tests FAIL)
```

**Interpretation:** Vulnerabilities fixed. Exploits prevented.

---

## Key Concepts

### Python Import Resolution Order

```python
sys.path = [
    '/workspace',              # sys.path[0] - HIGHEST PRIORITY
    '/usr/lib/python3.11',     # Standard library
    '/usr/local/lib/python3.11/site-packages',  # Installed packages
    '.'                        # Current directory
]
```

When you `import numpy`, Python searches in this order:
1. `/workspace/numpy.py` ← **MALICIOUS (found first!)**
2. `/usr/local/lib/python3.11/site-packages/numpy/` ← Legitimate numpy (never reached)

### Import-Time Execution

**Critical Fact:** Python executes module code when imported:

```python
# malicious_module.py
print("This runs IMMEDIATELY on import")
with open('/tmp/pwned', 'w') as f:
    f.write('Executed')

# User code
import malicious_module  # Code executes NOW
```

**No way to prevent this** without changing Python's import system.

### sys.modules Cache

Python caches all imported modules in `sys.modules`:

```python
import sys
import json  # json is now in sys.modules

# Attacker can poison the cache
import types
fake_json = types.ModuleType('json')
fake_json.loads = lambda s: {"hijacked": True}
sys.modules['json'] = fake_json  # Replace real json

import json  # Uses fake json from cache
```

---

## Attack Scenarios

### Scenario 1: Research Fraud
**Attacker:** Malicious collaborator
**Method:** Shadow numpy to inflate data
**Impact:** Fraudulent research publications
**Detection:** Nearly impossible until replication

### Scenario 2: Credential Theft
**Attacker:** Insider with workspace access
**Method:** Shadow json to steal API keys
**Impact:** $10K-$100K+ unauthorized cloud spending
**Detection:** Silent theft (no symptoms)

### Scenario 3: Lobster Compromise
**Attacker:** Advanced persistent threat
**Method:** Shadow pathlib to break Lobster internals
**Impact:** Complete system compromise
**Detection:** Silent failures, corrupted metadata

---

## Why AST Validation Fails

The service uses AST-based validation to block imports:

```python
FORBIDDEN_MODULES = {'subprocess', '__import__'}
FORBIDDEN_FROM_IMPORTS = {('os', 'system'), ...}
```

**What it CAN do:**
- ✅ Detect static import statements: `import subprocess`
- ✅ Raise `CodeValidationError` for forbidden imports
- ✅ Issue warnings for suspicious patterns

**What it CANNOT do:**
- ❌ Prevent module shadowing (workspace/subprocess.py loaded automatically)
- ❌ Prevent runtime sys.path manipulation
- ❌ Prevent sys.modules poisoning
- ❌ Prevent import-time code execution
- ❌ Prevent dynamic imports (`__import__()` only warned, not blocked)

**Fundamental Limitation:**
AST validation analyzes **static code structure**. It cannot predict or control **runtime behavior**.

---

## Remediation Checklist

After applying fixes, verify:

- [ ] Workspace NOT in sys.path (or at least not sys.path[0])
- [ ] Cannot shadow standard library modules
- [ ] Cannot shadow scientific packages
- [ ] Cannot shadow Lobster internals
- [ ] Cannot execute code at import time from workspace
- [ ] Runtime sys.path modifications detected and blocked
- [ ] sys.modules integrity monitored
- [ ] Dynamic imports properly restricted
- [ ] All tests in this directory FAIL (vulnerabilities fixed)

---

## Documentation

### Full Report
See `SUPPLY_CHAIN_REPORT.md` for comprehensive analysis including:
- Detailed vulnerability descriptions
- Real-world attack scenarios
- Detection evasion techniques
- Remediation strategies
- CVSS scoring
- Business impact analysis

### Quick Reference

**Root Cause:**
```python
# File: custom_code_execution_service.py
# Line: 340
sys.path.insert(0, str(WORKSPACE))  # ⚠️ CRITICAL VULNERABILITY
```

**Immediate Fix:**
```python
# BEFORE (VULNERABLE):
sys.path.insert(0, str(WORKSPACE))

# AFTER (SECURE):
# DO NOT add workspace to sys.path
# Use explicit file paths for custom imports if needed
```

**Vulnerability Count:** 20+ confirmed attack vectors
**Severity:** CRITICAL (CVSS ~9.8/10)
**Priority:** Immediate remediation required

---

## References

- **W3C PROV:** https://www.w3.org/TR/prov-overview/
- **Python Import System:** https://docs.python.org/3/reference/import.html
- **sys.path Documentation:** https://docs.python.org/3/library/sys.html#sys.path
- **PEP 302 (Import Hooks):** https://peps.python.org/pep-0302/
- **SolarWinds Attack:** https://www.cisa.gov/supply-chain-compromise
- **PyPI Typosquatting:** https://python-security.readthedocs.io/packages.html

---

## Questions?

For questions about this test suite, contact:
- **Security Team:** [security contact]
- **Engineering Lead:** [eng lead contact]
- **Test Author:** Agent 4 - Supply Chain Attack Tester

**Priority:** CRITICAL - Immediate remediation required
**Classification:** INTERNAL - SECURITY SENSITIVE
