# AST Bypass Security Tests

This directory contains security tests demonstrating vulnerabilities in AST-based code validation for CustomCodeExecutionService.

## Purpose

**Goal**: Prove that AST (Abstract Syntax Tree) validation alone is insufficient for securing arbitrary code execution.

**Scope**: Tests 15+ bypass techniques that allow execution of forbidden operations (subprocess, system commands) despite AST validation.

## Test Files

### 1. `test_dynamic_imports.py`
**Focus**: Runtime import mechanisms that bypass static analysis

**Techniques Tested**:
- `__import__()` builtin function (AST sees Call, not Import)
- `importlib.import_module()` dynamic imports
- `getattr(__builtins__, '__import__')` reflection
- `exec("import subprocess")` string-based imports
- Lambda obfuscation
- String concatenation to build module names
- Character code building (`chr()`)
- Format string imports

**Expected Results**: Most tests should SUCCEED (proving vulnerabilities exist)

### 2. `test_obfuscation_techniques.py`
**Focus**: Encoding and obfuscation that hides malicious code

**Techniques Tested**:
- Base64 encoding (critical - can hide entire payloads)
- ROT13 encoding via codecs
- Hex encoding
- Pickle deserialization RCE
- Marshal code object loading
- Multi-layer encoding (Base64 + ROT13)
- Dynamic string building + exec

**Expected Results**: Base64 and pickle tests are CRITICAL vulnerabilities

### 3. `test_encoding_tricks.py`
**Focus**: Unicode attacks and character encoding exploits

**Techniques Tested**:
- Homograph attacks (Cyrillic 'е' vs Latin 'e')
- Mixed script identifiers
- Zero-width characters
- Bidirectional override (Trojan Source CVE-2021-42574)
- Unicode normalization bypasses
- Ligature confusion

**Expected Results**: Demonstrates attacks that can deceive human code reviewers

## Running Tests

### Run All Bypass Tests
```bash
cd /Users/tyo/GITHUB/omics-os/lobster
pytest tests/manual/custom_code_execution/05_ast_bypass/ -v -s
```

### Run Specific Test File
```bash
# Dynamic import bypasses
pytest tests/manual/custom_code_execution/05_ast_bypass/test_dynamic_imports.py -v -s

# Encoding obfuscation
pytest tests/manual/custom_code_execution/05_ast_bypass/test_obfuscation_techniques.py -v -s

# Unicode tricks
pytest tests/manual/custom_code_execution/05_ast_bypass/test_encoding_tricks.py -v -s
```

### Run Specific Test
```bash
pytest tests/manual/custom_code_execution/05_ast_bypass/test_dynamic_imports.py::TestDynamicImportBypass::test_dunder_import_bypass_EXPECT_SUCCESS -v -s
```

## Understanding Test Results

### BEFORE Security Fixes

**Expected behavior**: Tests should SUCCEED (proving vulnerabilities)

```
test_dunder_import_bypass_EXPECT_SUCCESS PASSED
  ⚠️ CRITICAL VULNERABILITY: __import__() bypasses AST validation
```

If a test marked `_EXPECT_SUCCESS` **passes**, it means:
- ✅ The test correctly demonstrated the vulnerability
- ❌ The vulnerability exists in the service
- 🔴 Security fix is needed

### AFTER Security Fixes

**Expected behavior**: Tests should FAIL with CodeValidationError

```
test_dunder_import_bypass_EXPECT_SUCCESS FAILED
  ✅ PROTECTED: CodeValidationError: '__import__' is forbidden
```

If a test marked `_EXPECT_SUCCESS` **fails** with validation error:
- ✅ The security fix is working
- ✅ Vulnerability is patched

## Test Output Format

Each test prints detailed analysis:

```
================================================================================
CRITICAL VULNERABILITY CONFIRMED: __import__() Bypass
================================================================================
Result: subprocess module imported: subprocess
Success: True
Warnings: ["Use of '__import__()' detected. This may bypass safety checks."]

🚨 SECURITY BREACH:
   - Attacker can import ANY module, including subprocess
   - Can then execute arbitrary system commands
   - Example: subprocess.run(['rm', '-rf', '/'])

📋 Mitigation Required:
   - Remove __import__ from allowed builtins
   - Use RestrictedPython's safe_builtins
```

## Key Vulnerabilities Identified

### Critical (RCE Possible)

1. **`__import__()` bypass** - Only warned, not blocked
2. **`exec()`/`eval()` bypass** - Only warned, not blocked
3. **Base64 encoding** - Complete payload obfuscation
4. **Pickle deserialization** - Classic Python RCE vector
5. **String manipulation** - Build forbidden names at runtime

### High (Security Bypass)

6. **importlib.import_module()** - Trivial bypass
7. **ROT13/Hex encoding** - Simple obfuscation
8. **getattr(__builtins__)** - Reflection-based bypass

### Medium (Code Review Deception)

9. **Homograph attacks** - Visual deception
10. **Trojan Source (RTL override)** - CVE-2021-42574
11. **Unicode normalization** - Different bytes, same appearance

## Documentation

**Complete analysis**: See `AST_BYPASS_REPORT.md` for:
- Detailed vulnerability analysis
- Root cause explanation
- Mitigation recommendations (RestrictedPython, Docker isolation)
- Industry comparison
- CVSS scoring
- Proof-of-concept exploits

## Mitigation Priority

**Immediate Actions** (1-2 days):
1. Block `exec`/`eval`/`compile` (raise error, not warn)
2. Implement RestrictedPython for proper sandboxing
3. Add Unicode normalization and bidirectional text checks

**Short-Term** (1 week):
4. Restrict `__builtins__` namespace (remove `__import__`)
5. Block encoding modules (`base64`, `codecs`)
6. Block serialization modules (`pickle`, `marshal`)

**Medium-Term** (2-4 weeks):
7. Add Docker/container isolation
8. Implement seccomp-bpf syscall filtering

## References

- **RestrictedPython**: https://restrictedpython.readthedocs.io/
- **Trojan Source**: https://trojansource.codes/
- **OWASP Code Injection**: https://owasp.org/www-community/attacks/Code_Injection
- **Python Pickle Security**: https://docs.python.org/3/library/pickle.html

## Test Philosophy

These tests follow **security research ethics**:

✅ **Responsible**:
- Tests run in isolated subprocess (no harm to host)
- No actual malicious payloads executed
- Clear documentation of risks

✅ **Educational**:
- Demonstrates WHY AST validation alone fails
- Provides concrete examples for team review
- Includes mitigation recommendations

✅ **Actionable**:
- Each test includes specific fix recommendation
- Links to proven libraries (RestrictedPython)
- Prioritized by severity

## Notes

- All tests use `persist=False` to avoid polluting provenance
- Tests run in temporary workspace (pytest fixture)
- No actual system damage (subprocess isolation)
- Tests are deterministic and repeatable

## Contributing

When adding new bypass techniques:

1. Follow naming: `test_<technique>_EXPECT_SUCCESS`
2. Include detailed docstring explaining why AST fails
3. Print clear vulnerability/mitigation messages
4. Add entry to AST_BYPASS_REPORT.md
5. Update this README

## Questions?

See `AST_BYPASS_REPORT.md` for comprehensive analysis.

---

**Status**: 🚨 **VULNERABILITIES CONFIRMED**
**Action Required**: Implement RestrictedPython sandboxing
**Timeline**: High priority (1-2 days for initial fixes)
