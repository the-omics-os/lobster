# Timing Attack Security Assessment Report

**Service:** CustomCodeExecutionService
**Assessment Date:** 2025-11-30
**Severity:** MEDIUM (Local CLI) / HIGH (Cloud Deployment)
**Status:** VULNERABLE - No timing normalization implemented

---

## Executive Summary

The CustomCodeExecutionService is vulnerable to **timing-based information leakage attacks**. By measuring execution time differences, attackers can:

1. **Probe file system structure** without direct access (file existence, sizes, types)
2. **Brute force secrets** character-by-character (API keys, passwords)
3. **Scan network services** via connection timeout patterns
4. **Establish covert channels** for data exfiltration (CPU, I/O patterns)
5. **Infer data structure sizes** and system state

These attacks exploit the fact that **timing is not normalized** - different operations take measurably different amounts of time, leaking information through this side channel.

### Risk Assessment by Deployment

| Environment | Risk Level | Rationale |
|------------|-----------|-----------|
| **Local CLI** | MEDIUM | User controls environment, limited multi-user scenarios |
| **Cloud SaaS** | HIGH | Multi-tenant, shared infrastructure, network exposure |
| **Enterprise** | MEDIUM-HIGH | Depends on data sensitivity and isolation model |

---

## What Are Timing Attacks?

**Timing attacks** are a class of side-channel attacks where attackers infer information by measuring how long operations take. Unlike direct attacks (trying to read memory, execute commands), timing attacks exploit **indirect measurements** that leak information through timing patterns.

### Core Principle

Different operations take different amounts of time:
- File that exists: ~100 ns to check
- File that doesn't exist: ~80 ns to check
- String comparison: time varies based on matching prefix length
- Network connection: timing reveals open vs closed vs filtered ports

By measuring these differences, attackers can answer questions like:
- "Does this file exist?"
- "How long is this string?"
- "What's the first character of this API key?"
- "Is port 5432 open on localhost?"

---

## Vulnerability Analysis

### 1. File System Timing Leaks

**Attack Vector:** File existence probing via `Path.exists()` timing differences

**Test:** `test_file_existence_timing_leak_EXPECT_SUCCESS`

**How It Works:**
```python
# Measure time to check existing file
start = time.perf_counter()
exists = Path('/etc/hosts').exists()
time_exists = time.perf_counter() - start

# Measure time for non-existent file
start = time.perf_counter()
not_exists = Path('/nonexistent_file').exists()
time_not_exists = time.perf_counter() - start

# Timing difference reveals existence
if time_exists > time_not_exists:
    print("File likely exists!")
```

**Impact:**
- ✅ **Success:** Detectable timing differences (100-500 ns)
- 🔴 **Risk:** Enumerate sensitive file locations (.env, .ssh/id_rsa, etc.)
- 📊 **Severity:** MEDIUM

**Real-World Exploitation:**
1. Attacker submits code to check `/etc/passwd` vs `/nonexistent`
2. Timing difference reveals file existence
3. Repeat for `.env`, `credentials.json`, `.ssh/` to map sensitive files
4. No direct file access needed - timing leaks information

---

### 2. File Size Inference via Read Timing

**Attack Vector:** Infer file sizes by measuring read operation timing

**Test:** `test_file_size_inference_via_read_timing_EXPECT_SUCCESS`

**How It Works:**
```python
# Small file (100 bytes): ~0.05 ms
# Large file (10 KB): ~0.5 ms
# Ratio: 10x timing difference reveals size

start = time.perf_counter()
content = Path('file.txt').read_text()
elapsed = time.perf_counter() - start

# Timing reveals approximate file size
```

**Impact:**
- ✅ **Success:** 10-20x timing difference between small/large files
- 🔴 **Risk:** Learn which files contain substantial data
- 📊 **Severity:** LOW-MEDIUM

**Exploitation:**
- Identify which workspace files are configuration (small) vs data (large)
- Detect when analyses complete (large result files appear)

---

### 3. String Comparison Timing Attacks (CRITICAL)

**Attack Vector:** Python string comparison is NOT constant-time

**Test:** `test_api_key_brute_force_timing_attack_EXPECT_SUCCESS`

**How It Works:**
```python
API_KEY = "sk-proj-abcdef1234567890"

# Try each character
for guess_char in 'abcdefghijklmnopqrstuvwxyz0123456789-_':
    test_key = guess_char + 'x' * (len(API_KEY) - 1)

    start = time.perf_counter()
    matches = (test_key == API_KEY)
    elapsed = time.perf_counter() - start

    # Character with LONGEST comparison time is likely correct
    # Python's string comparison checks character-by-character
    # Matching prefix takes slightly longer
```

**Why This Works:**
Python's `==` operator for strings is **not constant-time**:
1. Checks length first (fast path)
2. If lengths match, compares character-by-character
3. Early return on mismatch
4. **Timing leak:** More matching characters = longer comparison time

**Impact:**
- ✅ **Success:** Character-by-character extraction possible
- 🔴 **Risk:** API keys, passwords, secrets compromised
- 📊 **Severity:** **CRITICAL**

**Attack Complexity:**
- Brute force 32-char API key:
  - Naive: 62^32 = 2.27 × 10^57 attempts (impossible)
  - Timing attack: 32 × 62 = 1,984 attempts (trivial)

**Real-World Scenario:**
```python
# Attacker's code (runs in CustomCodeExecutionService)
import os
import time

REAL_KEY = os.environ.get('ANTHROPIC_API_KEY')  # Target

charset = 'abcdefghijklmnopqrstuvwxyz0123456789-_'
extracted = ""

for position in range(len(REAL_KEY)):
    timings = {}

    for char in charset:
        test_key = extracted + char + 'x' * (len(REAL_KEY) - position - 1)

        # Measure comparison time
        times = []
        for _ in range(100):
            start = time.perf_counter()
            matches = (test_key == REAL_KEY)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        timings[char] = sum(times) / len(times)

    # Character with longest time is correct
    best_char = max(timings, key=timings.get)
    extracted += best_char
    print(f"Position {position}: {best_char}")

print(f"Extracted key: {extracted}")
```

**Mitigation:**
```python
import secrets

# SECURE: Constant-time comparison
if secrets.compare_digest(input_key, stored_key):
    return True

# INSECURE: Variable-time comparison
if input_key == stored_key:  # ❌ TIMING LEAK
    return True
```

---

### 4. Password Timing Attacks

**Attack Vector:** Early return in password verification leaks match position

**Test:** `test_password_timing_attack_EXPECT_SUCCESS`

**How It Works:**
```python
def verify_password_INSECURE(input_pwd):
    if len(input_pwd) != len(STORED_PASSWORD):
        return False

    # Character-by-character comparison
    for i, char in enumerate(input_pwd):
        if char != STORED_PASSWORD[i]:
            return False  # ❌ EARLY RETURN LEAKS POSITION
    return True

# Attack: Measure timing to find correct password length
for length in range(10, 30):
    test_pwd = "x" * length
    start = time.perf_counter()
    verify_password_INSECURE(test_pwd)
    elapsed = time.perf_counter() - start
    # Correct length has longer comparison time

# Attack: Once length known, brute force character-by-character
for pos in range(len(password)):
    for char in charset:
        test_pwd = known_prefix + char + 'x' * remaining
        # Measure timing - correct char takes longer
```

**Impact:**
- ✅ **Success:** Length + character extraction
- 🔴 **Risk:** Password compromise
- 📊 **Severity:** **HIGH**

---

### 5. Network Service Probing via Timeout Timing

**Attack Vector:** Connection timeout differences reveal port states

**Test:** `test_localhost_port_scanning_via_timeout_EXPECT_SUCCESS`

**How It Works:**
```python
import socket

ports = {22: 'SSH', 5432: 'PostgreSQL', 6379: 'Redis'}

for port, service in ports.items():
    start = time.perf_counter()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.1)

    result = sock.connect_ex(('127.0.0.1', port))
    elapsed = time.perf_counter() - start

    # Timing patterns:
    # - Open port: result=0, time < 10ms (immediate accept)
    # - Closed port: result=61, time < 10ms (immediate reject)
    # - Filtered port: result=-1, time ~100ms (timeout)
```

**Timing Patterns:**
| Port State | Result Code | Timing | Detectable |
|-----------|-------------|--------|-----------|
| Open | 0 | < 10 ms | ✅ Fast accept |
| Closed | 61 (ECONNREFUSED) | < 10 ms | ✅ Fast reject |
| Filtered | Timeout | ~100 ms | ✅ Slow timeout |
| No route | 51 (ENETUNREACH) | ~100 ms | ✅ Slow timeout |

**Impact:**
- ✅ **Success:** Port state detection works
- 🔴 **Risk:** Discover internal services (Redis, PostgreSQL, APIs)
- 📊 **Severity:** MEDIUM (local), HIGH (cloud)

**Real-World Scenario:**
```python
# Scan for common databases
common_ports = {
    3306: 'MySQL',
    5432: 'PostgreSQL',
    6379: 'Redis',
    27017: 'MongoDB',
    9200: 'Elasticsearch'
}

for port, name in common_ports.items():
    timing = measure_connection_timing('localhost', port)
    if timing < 0.01:  # Fast response
        print(f"FOUND: {name} on port {port}")
```

---

### 6. CPU Usage Covert Channel

**Attack Vector:** Encode binary data via CPU usage patterns

**Test:** `test_cpu_usage_as_covert_channel_EXPECT_SUCCESS`

**How It Works:**
```python
def send_bit(bit):
    if bit == '1':
        # Busy-wait (100% CPU)
        start = time.time()
        while time.time() - start < 0.05:
            x = 1234 * 5678  # Keep CPU busy
    else:
        # Sleep (0% CPU)
        time.sleep(0.05)

# Transmit "HI" in binary
message = "HI"
binary = ''.join(format(ord(c), '08b') for c in message)

for bit in binary:
    send_bit(bit)  # External observer monitors CPU usage
```

**Covert Channel Encoding:**
- **Bit 1:** Busy-wait (100% CPU for 50ms)
- **Bit 0:** Sleep (0% CPU for 50ms)
- **Bandwidth:** ~20 bits/second
- **Detection:** External CPU monitoring (top, htop, monitoring tools)

**Impact:**
- ✅ **Success:** Covert communication established
- 🔴 **Risk:** Data exfiltration in isolated environments
- 📊 **Severity:** LOW (local), MEDIUM (cloud multi-tenant)

**Mitigation:**
- CPU usage monitoring
- Rate limiting on CPU-intensive operations
- Anomaly detection for unusual CPU patterns

---

### 7. Disk I/O Covert Channel

**Attack Vector:** Encode data via disk write patterns

**Test:** `test_disk_io_timing_covert_channel_EXPECT_SUCCESS`

**Similar to CPU channel, but uses disk I/O:**
- **Bit 1:** Many small writes (high I/O)
- **Bit 0:** No writes (low I/O)
- **Detection:** Disk usage monitoring reveals pattern

---

### 8. Cache Timing Side Channels

**Attack Vector:** Cache timing differences (simplified Spectre-like)

**Test:** `test_cache_timing_side_channel_EXPECT_SUCCESS`

**How It Works:**
```python
# First access (cold - not in cache): ~100 ns
# Second access (hot - cached): ~10 ns
# 10x speedup reveals cache state

def measure_cache_timing(arr, index):
    start = time.perf_counter()
    value = arr[index]  # Cold
    cold_time = time.perf_counter() - start

    start = time.perf_counter()
    value = arr[index]  # Hot (cached)
    hot_time = time.perf_counter() - start

    return cold_time, hot_time
```

**Impact:**
- ✅ **Success:** Cache effects detectable
- 🔴 **Risk:** Real attacks (Spectre, Meltdown) are far more sophisticated
- 📊 **Severity:** LOW (simplified demonstration)

**Note:** Real cache timing attacks are extremely complex and require:
- CPU architecture knowledge
- Speculative execution exploitation
- Precise timing measurements (< 1 ns)

---

## Comprehensive Attack Scenarios

### Scenario 1: API Key Extraction

**Attacker Goal:** Extract `ANTHROPIC_API_KEY` from environment

**Attack Steps:**
1. Submit code to measure string comparison timing
2. Brute force first character: 62 attempts
3. Repeat for each position: 32 × 62 = 1,984 total attempts
4. Extract complete API key in ~2 minutes

**Mitigation:**
- Use `secrets.compare_digest()` for all sensitive comparisons
- Don't expose secrets in code execution environment
- Rate limit execution requests

---

### Scenario 2: Network Reconnaissance

**Attacker Goal:** Map internal network services

**Attack Steps:**
1. Scan localhost ports 1-65535
2. Timing patterns reveal service states
3. Identify databases (PostgreSQL, Redis, MySQL)
4. Identify internal APIs (ports 8000-9000)

**Mitigation:**
- Disable socket operations in execution environment
- Network namespace isolation (Docker)
- Firewall rules blocking internal access

---

### Scenario 3: File System Enumeration

**Attacker Goal:** Discover sensitive files in workspace

**Attack Steps:**
1. Probe common paths (.env, credentials.json, .ssh/)
2. Timing differences reveal existence
3. Measure read timing to infer sizes
4. Target large files for further analysis

**Mitigation:**
- Workspace isolation
- File access audit logging
- Restrict file operations to workspace only

---

### Scenario 4: Covert Data Exfiltration

**Attacker Goal:** Exfiltrate data despite network restrictions

**Attack Steps:**
1. Encode data as binary (ASCII to bits)
2. Transmit via CPU usage patterns
3. External monitor observes CPU usage
4. Decode bits back to original data

**Bandwidth:** ~20 bits/second (sufficient for API keys, small secrets)

**Mitigation:**
- CPU usage anomaly detection
- Rate limiting on CPU-intensive operations
- Random CPU noise injection

---

## Recommendations

### Immediate Actions (High Priority)

#### 1. Implement Constant-Time Comparisons

**Problem:** Python string comparison leaks timing information

**Solution:**
```python
import secrets

# Before (INSECURE)
if input_key == stored_key:
    return True

# After (SECURE)
if secrets.compare_digest(input_key.encode(), stored_key.encode()):
    return True
```

**Apply to:**
- All password verifications
- API key comparisons
- Token validations
- Any sensitive string comparison

---

#### 2. Add Timing Normalization

**Problem:** Operations have detectable timing differences

**Solution:**
```python
import time
import random

def normalize_timing(func, target_duration=0.1):
    '''Execute function with normalized timing.'''
    start = time.time()
    result = func()
    elapsed = time.time() - start

    # Sleep to reach target duration
    if elapsed < target_duration:
        time.sleep(target_duration - elapsed)

    # Add random noise (±10%)
    noise = random.uniform(-0.01, 0.01)
    time.sleep(max(0, noise))

    return result
```

**Apply to:**
- File existence checks
- Network operations
- Authentication flows

---

#### 3. Network Isolation

**Problem:** Socket operations enable port scanning

**Solution:**
```python
# Option 1: Disable socket module
FORBIDDEN_MODULES = {
    'subprocess', '__import__', 'socket'  # Add socket
}

# Option 2: Use Docker with --network=none
docker run --network=none lobster-code-execution

# Option 3: Network namespace isolation (Linux)
unshare --net python code.py
```

---

#### 4. Resource Usage Monitoring

**Problem:** CPU/I/O patterns leak information

**Solution:**
- Monitor CPU usage during execution
- Detect anomalous patterns (sustained 100% usage)
- Alert on unusual I/O patterns
- Rate limit resource-intensive operations

```python
import psutil

def monitor_execution(code):
    process = psutil.Process()

    # Monitor CPU usage
    cpu_samples = []
    for _ in range(10):
        cpu_samples.append(process.cpu_percent(interval=0.1))

    avg_cpu = sum(cpu_samples) / len(cpu_samples)

    if avg_cpu > 80:
        logger.warning("High CPU usage detected - possible covert channel")
```

---

### Long-Term Improvements

#### 1. Sandboxed Execution Environment

**Current:** Subprocess execution on host system
**Recommended:** Docker/gVisor/Firecracker containers

**Benefits:**
- Network isolation (--network=none)
- Resource limits (--memory, --cpu-shares)
- Namespace isolation (PID, network, IPC)

```dockerfile
# Secure execution container
FROM python:3.11-slim

# Drop all capabilities
RUN apt-get update && apt-get install -y libcap2-bin
RUN setcap cap_net_bind_service=+ep /usr/local/bin/python

# Run as non-root
USER nobody

# Disable network (override in docker-compose)
```

---

#### 2. Result Caching

**Problem:** Repeated operations leak timing information

**Solution:** Cache results to normalize timing

```python
import functools
import hashlib

@functools.lru_cache(maxsize=1000)
def cached_file_exists(path):
    '''Cache file existence checks.'''
    return Path(path).exists()

# All calls return in constant time (cache hit)
```

---

#### 3. Random Delays

**Problem:** Timing differences are measurable

**Solution:** Add random delays to blur timing signals

```python
import random

def execute_with_noise(code):
    # Add random startup delay
    time.sleep(random.uniform(0, 0.1))

    result = execute_code(code)

    # Add random completion delay
    time.sleep(random.uniform(0, 0.1))

    return result
```

---

#### 4. Rate Limiting

**Problem:** Attackers need many measurements for statistical accuracy

**Solution:** Rate limit execution requests

```python
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=10, period=60)  # 10 executions per minute
def execute_custom_code(code):
    return service.execute(code)
```

---

## Testing the Mitigations

### Test 1: Constant-Time Comparison

```python
def test_constant_time_comparison_mitigation_EXPECT_FAILURE(service):
    '''Test that secrets.compare_digest prevents timing attacks.'''

    code = '''
import secrets
import time

SECRET = "sk-proj-1234567890"

# Try timing attack with constant-time comparison
timings = {}
for char in 'abcdefghijklmnopqrstuvwxyz0123456789-':
    test = char + 'x' * (len(SECRET) - 1)

    times = []
    for _ in range(100):
        start = time.perf_counter()
        # Constant-time comparison
        match = secrets.compare_digest(test, SECRET)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    timings[char] = sum(times) / len(times)

# Check if timing attack fails
variance = max(timings.values()) - min(timings.values())
result = {
    'timing_variance_ns': variance * 1e9,
    'attack_defeated': variance < 1e-9
}
'''

    result, stats, ir = service.execute(code)

    # Mitigation should make attack fail
    assert result['attack_defeated'], "Constant-time comparison should prevent attack"
```

---

## Severity Assessment

### Risk Matrix

| Attack Vector | Exploitability | Impact | Overall Severity |
|--------------|----------------|--------|------------------|
| String comparison timing | HIGH | CRITICAL | **HIGH** |
| Password timing | HIGH | HIGH | **HIGH** |
| File existence timing | MEDIUM | MEDIUM | **MEDIUM** |
| Network port scanning | MEDIUM | MEDIUM | **MEDIUM** |
| File size inference | LOW | LOW | **LOW** |
| CPU covert channel | LOW | MEDIUM | **LOW-MEDIUM** |
| I/O covert channel | LOW | MEDIUM | **LOW-MEDIUM** |
| Cache timing | LOW | LOW | **LOW** |

### Deployment-Specific Risk

#### Local CLI (Current)
- **Overall Risk:** MEDIUM
- **Rationale:** User controls environment, limited multi-user scenarios
- **Priority Fixes:** String comparison timing (HIGH)

#### Cloud SaaS (Future)
- **Overall Risk:** HIGH
- **Rationale:** Multi-tenant, shared infrastructure, network exposure
- **Priority Fixes:** All timing attacks, network isolation, resource limits

---

## Comparison to Other Systems

### Jupyter Notebooks
- **Timing Protection:** None
- **Security Model:** Full trust (user's own environment)
- **Applicable to Lobster Local:** Yes (similar trust model)

### AWS Lambda
- **Timing Protection:** Partial (network isolation, resource limits)
- **Security Model:** Medium trust (user code, isolated execution)
- **Applicable to Lobster Cloud:** Yes (similar model)

### Google Colab
- **Timing Protection:** Minimal
- **Security Model:** Medium trust (user code, some restrictions)
- **Network:** Allowed (with rate limits)

### Online Code Execution Platforms (e.g., Repl.it)
- **Timing Protection:** Limited
- **Security Model:** Low trust (sandboxed, network allowed)
- **Rate Limiting:** Yes (prevents statistical attacks)

---

## Conclusion

The CustomCodeExecutionService has **measurable timing vulnerabilities** that could be exploited to:
1. Extract secrets character-by-character (**CRITICAL**)
2. Probe file system and network topology (**MEDIUM**)
3. Establish covert communication channels (**LOW-MEDIUM**)

### Immediate Priorities

1. **✅ Implement `secrets.compare_digest()`** for all sensitive comparisons (HIGH)
2. **✅ Add network isolation** for cloud deployment (HIGH)
3. **✅ Document security model** clearly in user-facing docs (MEDIUM)
4. **⚠️ Consider rate limiting** for statistical attack prevention (MEDIUM)
5. **⚠️ Add resource monitoring** for covert channel detection (LOW)

### Long-Term Roadmap

- **Phase 1 (Month 1):** Constant-time comparisons, network isolation
- **Phase 2 (Month 2):** Docker sandboxing, resource limits
- **Phase 3 (Month 3):** Timing normalization, random delays
- **Phase 4 (Month 6):** Full monitoring and anomaly detection

---

## References

### Academic Papers

1. **Kocher, P. C. (1996).** "Timing Attacks on Implementations of Diffie-Hellman, RSA, DSS, and Other Systems." *CRYPTO '96*.

2. **Bernstein, D. J. (2005).** "Cache-timing attacks on AES." *University of Illinois at Chicago*.

3. **Brumley, D., & Boneh, D. (2003).** "Remote timing attacks are practical." *USENIX Security Symposium*.

4. **Kocher, P., et al. (2018).** "Spectre Attacks: Exploiting Speculative Execution." *S&P 2018*.

### Industry Resources

- **OWASP:** [Testing for Timing Attacks](https://owasp.org/www-community/attacks/Timing_attack)
- **Python Security:** [secrets module documentation](https://docs.python.org/3/library/secrets.html)
- **NIST:** [SP 800-133 Recommendation for Cryptographic Key Generation](https://csrc.nist.gov/publications/detail/sp/800-133/rev-2/final)

---

**Report Prepared By:** Agent 6 (Timing Attack Tester)
**Date:** 2025-11-30
**Classification:** Internal Security Assessment
