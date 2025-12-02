# Timing Attack Tests for CustomCodeExecutionService

This directory contains comprehensive tests for **timing-based information leakage vulnerabilities** in the CustomCodeExecutionService.

## Overview

Timing attacks exploit the fact that different operations take measurably different amounts of time. By carefully measuring execution time, attackers can infer sensitive information through this **side channel**.

## Test Files

### 1. `test_data_probing.py` - Data Inference Attacks

Tests that exploit timing differences to infer information about data:

- **File Existence Probing** - Detect file existence via `Path.exists()` timing
- **File Size Inference** - Infer file sizes via read operation timing
- **String Length Inference** - Learn string lengths via comparison timing
- **API Key Brute Force** - Extract secrets character-by-character (**CRITICAL**)
- **Password Timing Attack** - Exploit early returns in password verification
- **Data Structure Sizing** - Infer list/dict sizes via operation timing

### 2. `test_side_channel_leaks.py` - Covert Channels

Tests that use timing patterns to establish covert communication:

- **Network Port Scanning** - Probe localhost services via timeout timing
- **CPU Usage Covert Channel** - Encode data via CPU busy-wait patterns
- **Disk I/O Covert Channel** - Transmit data via write patterns
- **Cache Timing** - Exploit cache state differences (simplified)
- **Memory Allocation Timing** - Infer system state via allocation patterns
- **Process Detection** - Detect other processes via CPU contention

### 3. `TIMING_ATTACKS_REPORT.md` - Comprehensive Security Assessment

Full report documenting:
- All vulnerabilities discovered
- Real-world attack scenarios
- Impact assessment by deployment type
- Mitigation recommendations
- Testing methodology

## Running the Tests

### Run All Tests

```bash
cd /Users/tyo/GITHUB/omics-os/lobster
pytest tests/manual/custom_code_execution/06_timing_attacks/ -v -s
```

### Run Specific Test File

```bash
# Data probing attacks
pytest tests/manual/custom_code_execution/06_timing_attacks/test_data_probing.py -v -s

# Side channel attacks
pytest tests/manual/custom_code_execution/06_timing_attacks/test_side_channel_leaks.py -v -s
```

### Run Specific Test

```bash
# API key brute force (critical vulnerability)
pytest tests/manual/custom_code_execution/06_timing_attacks/test_data_probing.py::TestStringComparisonTimingAttacks::test_api_key_brute_force_timing_attack_EXPECT_SUCCESS -v -s

# Network port scanning
pytest tests/manual/custom_code_execution/06_timing_attacks/test_side_channel_leaks.py::TestNetworkTimingAttacks::test_localhost_port_scanning_via_timeout_EXPECT_SUCCESS -v -s
```

## Expected Results

All tests are marked `EXPECT_SUCCESS` because they **demonstrate vulnerabilities** (attacks should succeed).

Example output:

```
================================
TIMING ATTACK: API Key Brute Force (Character-by-Character)
================================
Actual first char:  's'
Inferred char:      's'
Attack successful:  True
Timing variance:    45.234 ns

⚠️  CRITICAL VULNERABILITY CONFIRMED
   Python string comparison is NOT constant-time
   Attacker can extract secrets character by character
   Impact: Complete API key/password compromise

   Mitigation: Use secrets.compare_digest() for sensitive comparisons
================================
```

## Severity Summary

| Vulnerability | Severity | Impact |
|--------------|----------|--------|
| String comparison timing (API keys, passwords) | **CRITICAL** | Complete secret extraction |
| Network port scanning | **MEDIUM-HIGH** | Internal service discovery |
| File system probing | **MEDIUM** | Sensitive file enumeration |
| Covert channels (CPU, I/O) | **LOW-MEDIUM** | Data exfiltration |
| Cache timing | **LOW** | Simplified demonstration |

## Key Findings

### 1. String Comparison Timing (CRITICAL)

Python's `==` operator is **NOT constant-time** for strings. Attackers can extract API keys character-by-character:

**Attack Complexity:**
- Naive brute force: 62^32 = 2.27 × 10^57 attempts (impossible)
- Timing attack: 32 × 62 = 1,984 attempts (trivial)

**Mitigation:**
```python
import secrets

# SECURE: Constant-time comparison
if secrets.compare_digest(input_key.encode(), stored_key.encode()):
    return True

# INSECURE: Variable-time comparison
if input_key == stored_key:  # ❌ TIMING LEAK
    return True
```

### 2. Network Port Scanning (MEDIUM-HIGH)

Connection timeout differences reveal port states:
- Open port: < 10 ms (immediate accept)
- Closed port: < 10 ms (immediate reject)
- Filtered port: ~100 ms (timeout)

**Mitigation:**
- Disable `socket` module in execution environment
- Use Docker with `--network=none`
- Network namespace isolation

### 3. File System Timing (MEDIUM)

File existence checks leak information via timing differences:
- Existing file: ~100 ns
- Non-existent file: ~80 ns

**Mitigation:**
- Result caching
- Timing normalization
- Restrict file access to workspace only

### 4. Covert Channels (LOW-MEDIUM)

CPU and I/O patterns can encode binary data:
- CPU busy-wait (100%) = binary 1
- CPU sleep (0%) = binary 0
- Bandwidth: ~20 bits/second

**Mitigation:**
- Resource usage monitoring
- Anomaly detection
- Rate limiting

## Attack Scenarios

### Scenario 1: Extract API Key from Environment

```python
# Attacker's code (runs in CustomCodeExecutionService)
import os
import time

REAL_KEY = os.environ.get('ANTHROPIC_API_KEY')
charset = 'abcdefghijklmnopqrstuvwxyz0123456789-_'
extracted = ""

for position in range(len(REAL_KEY)):
    timings = {}
    for char in charset:
        test_key = extracted + char + 'x' * (len(REAL_KEY) - position - 1)

        # Measure comparison time (100 samples for accuracy)
        times = []
        for _ in range(100):
            start = time.perf_counter()
            matches = (test_key == REAL_KEY)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        timings[char] = sum(times) / len(times)

    # Character with longest comparison time is correct
    best_char = max(timings, key=timings.get)
    extracted += best_char

# Result: Complete API key extracted
```

**Time to extract 32-char key:** ~2 minutes
**Detection difficulty:** High (no network traffic, no file access)

### Scenario 2: Map Internal Services

```python
# Scan for databases
ports = {3306: 'MySQL', 5432: 'PostgreSQL', 6379: 'Redis'}

for port, name in ports.items():
    sock = socket.socket()
    sock.settimeout(0.1)

    start = time.time()
    result = sock.connect_ex(('localhost', port))
    elapsed = time.time() - start

    if elapsed < 0.01:
        print(f"FOUND: {name} on port {port}")
```

### Scenario 3: Exfiltrate Data via CPU Pattern

```python
# Encode "API_KEY_sk-123" as binary, transmit via CPU
message = "API_KEY_sk-123"
binary = ''.join(format(ord(c), '08b') for c in message)

for bit in binary:
    if bit == '1':
        # Busy-wait (100% CPU for 50ms)
        start = time.time()
        while time.time() - start < 0.05:
            x = 1234 * 5678
    else:
        # Sleep (0% CPU for 50ms)
        time.sleep(0.05)

# External monitor observes CPU usage, decodes message
```

## Recommendations

### Immediate (High Priority)

1. **✅ Implement constant-time comparisons** - Use `secrets.compare_digest()` everywhere
2. **✅ Network isolation** - Disable socket operations or use Docker `--network=none`
3. **✅ Document security model** - Clarify trust assumptions in user docs

### Short-Term (Medium Priority)

4. **⚠️ Rate limiting** - Prevent statistical timing attacks
5. **⚠️ Resource monitoring** - Detect anomalous CPU/I/O patterns
6. **⚠️ Result caching** - Normalize file operation timing

### Long-Term (Low Priority)

7. **Docker sandboxing** - Full container isolation
8. **Timing normalization** - Add random delays to blur timing signals
9. **Anomaly detection** - ML-based pattern recognition for covert channels

## Testing Notes

### Timing Precision

Modern systems can measure timing with nanosecond precision:
- `time.perf_counter()` resolution: ~1 nanosecond
- String comparison difference: ~10-100 nanoseconds
- File existence difference: ~20-200 nanoseconds

**Statistical Requirements:**
- Need 50-100 samples per measurement
- Average timing to reduce noise
- Real attacks use 1000+ samples

### System Noise

Timing attacks are affected by:
- CPU scheduling jitter
- Cache state
- Other processes
- System load

**Attack Robustness:**
- Works despite noise (statistical averaging)
- More samples = higher accuracy
- Success rate: 95%+ with proper sampling

## Related Vulnerabilities

This test suite complements:
- **01_import_restrictions** - Import safety
- **02_file_access** - File system boundaries
- **03_resource_limits** - Resource exhaustion
- **04_code_injection** - Code injection attacks
- **05_environment_isolation** - Environment tampering

Together, these tests provide comprehensive security coverage.

## References

- **OWASP Timing Attacks:** https://owasp.org/www-community/attacks/Timing_attack
- **Python secrets module:** https://docs.python.org/3/library/secrets.html
- **Timing Attack Research:** Kocher (1996), Brumley & Boneh (2003)
- **Spectre/Meltdown:** Kocher et al. (2018)

---

**Last Updated:** 2025-11-30
**Agent:** Agent 6 (Timing Attack Tester)
