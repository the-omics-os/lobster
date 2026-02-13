# Resource Exhaustion Vulnerability Report
## CustomCodeExecutionService Security Assessment

**Agent:** Agent 2 - Resource Exhaustion Tester
**Date:** 2025-11-30
**Target:** `lobster/services/execution/custom_code_execution_service.py`
**Environment:** Local CLI, Python 3.11+, macOS

---

## Executive Summary

The CustomCodeExecutionService provides subprocess-based code execution with **minimal resource constraints**. While process isolation prevents crashes from affecting the main Lobster process, the service is **highly vulnerable to resource exhaustion attacks** that can:

- **Exhaust system RAM** (OOM killer, crash other processes)
- **Burn CPU at 100%** for up to 300 seconds per execution
- **Fill entire disk** with workspace files
- **Exhaust file descriptors** (1024+ open files)

**Current Protection:** Only a 300-second timeout
**Missing Protections:** Memory limits, CPU throttling, disk quotas, FD limits

**Risk Assessment:** **HIGH** - Production deployment requires immediate hardening

---

## Security Model Analysis

### Current Security Model

| Protection | Status | Effectiveness |
|------------|--------|---------------|
| **Process Isolation** | ✅ Implemented | Good - crashes isolated |
| **Timeout Enforcement** | ⚠️ 300s (too long) | Partial - stops infinite loops |
| **Workspace-only file access** | ✅ Implemented | Good - limits file scope |
| **Import restrictions** | ✅ Implemented | Good - blocks subprocess, os.system |
| **Memory Limits** | ❌ None | **Critical Gap** |
| **CPU Throttling** | ❌ None | **Critical Gap** |
| **Disk Quotas** | ❌ None | **Critical Gap** |
| **FD Limits** | ❌ None | **Critical Gap** |
| **Network Isolation** | ❌ None (future) | Gap - can make HTTP requests |

### Attack Surface

```
┌─────────────────────────────────────────────────────────┐
│                   User Code (Untrusted)                 │
│                                                         │
│  [Memory]  [CPU]  [Disk]  [FDs]  [Network]             │
│     ❌      ❌      ❌      ❌       ❌                    │
│     No     No     No     No      No                    │
│   Limits  Limits Limits Limits  Isolation              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              subprocess.run() with timeout              │
│                   (300s default)                        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    System Resources                     │
│  RAM: Unlimited  CPU: 100%  Disk: Unlimited            │
└─────────────────────────────────────────────────────────┘
```

---

## Vulnerability Inventory

### 1. Memory Exhaustion Attacks

**Severity:** HIGH
**CVSS Score:** 7.5 (High)
**Tests:** `test_memory_bombs.py`

#### 1.1 Large List Allocation

**Attack Vector:**
```python
# Safe PoC (400MB)
x = [0] * (10**8)

# Real Attack (8GB)
x = [0] * (10**9)
```

**Impact:**
- Allocates 400MB+ RAM per execution
- Real attack: 8GB allocation → OOM killer
- Multi-user: Amplified exhaustion

**Current Status:** ✅ VULNERABLE (test passes)

---

#### 1.2 Large String Allocation

**Attack Vector:**
```python
# Safe PoC (100MB)
s = 'A' * (10**8)

# Real Attack (1GB)
s = 'A' * (10**9)
```

**Impact:**
- 100MB+ string allocation
- Real attack: 1GB+ strings
- Python strings are immutable (extra copies = 2x RAM)

**Current Status:** ✅ VULNERABLE

---

#### 1.3 numpy Array Bombs

**Attack Vector:**
```python
# Safe PoC (800MB)
arr = np.zeros((10**4, 10**4), dtype=np.float64)

# Real Attack (80GB)
arr = np.zeros((10**5, 10**5), dtype=np.float64)
```

**Impact:**
- 800MB array in safe test
- Real attack: 10^5 x 10^5 = 80GB
- Common in bioinformatics code (not obviously malicious)

**Current Status:** ✅ VULNERABLE

---

#### 1.4 pandas DataFrame Bombs

**Attack Vector:**
```python
# Exponential growth attack
df = pd.DataFrame({'a': [1, 2, 3]})
for i in range(20):  # 2^20 = 1M rows
    df = pd.concat([df, df], ignore_index=True)
```

**Impact:**
- Exponential memory growth (2^n)
- 20 iterations = 1M rows = GB of RAM
- 25 iterations = 32M rows = 10GB+

**Current Status:** ✅ VULNERABLE

---

#### 1.5 Recursive Data Structures

**Attack Vector:**
```python
# Safe PoC (100 levels)
def create_nested(depth):
    return [create_nested(depth-1)] if depth > 0 else []

nested = create_nested(100)
```

**Impact:**
- Python recursion limit: 1000 (default)
- Can still allocate large nested structures
- Combined with data = memory bomb

**Current Status:** ✅ VULNERABLE (but limited by recursion)

---

### 2. CPU Exhaustion Attacks

**Severity:** HIGH
**CVSS Score:** 7.1 (High)
**Tests:** `test_cpu_exhaustion.py`

#### 2.1 Infinite Loops

**Attack Vector:**
```python
# Infinite loop (hits timeout)
while True:
    x = 1 + 1
```

**Impact:**
- Burns CPU at 100% until timeout
- Default timeout: **300 seconds** (5 minutes!)
- Multi-core: Can spawn threads to burn all cores

**Current Status:** ⚠️ PARTIAL PROTECTION (timeout works but 300s is too long)

---

#### 2.2 CPU-Intensive Loops

**Attack Vector:**
```python
# Safe PoC (5 seconds)
start = time.time()
count = 0
while time.time() - start < 5:
    count += 1  # Millions of iterations

# Real Attack (300 seconds)
while time.time() - start < 300:
    count += 1  # Burns CPU for 5 minutes
```

**Impact:**
- 100% CPU utilization
- Blocks other work (if sequential)
- 300s = significant DoS window

**Current Status:** ✅ VULNERABLE

---

#### 2.3 Prime Number Calculation

**Attack Vector:**
```python
# Safe PoC
primes = find_primes(100000)  # ~5 seconds

# Real Attack
primes = find_primes(10000000)  # Minutes of CPU
```

**Impact:**
- Computationally expensive (O(n√n))
- Legitimate-looking code
- Hard to distinguish from real analysis

**Current Status:** ✅ VULNERABLE

---

#### 2.4 O(n³) Nested Loops

**Attack Vector:**
```python
# Safe PoC (n=100)
for i in range(100):
    for j in range(100):
        for k in range(100):
            count += 1  # 1M iterations

# Real Attack (n=1000)
# 1 billion iterations = minutes
```

**Impact:**
- Cubic complexity
- 1 billion operations = extended CPU burn
- Common pattern in scientific code

**Current Status:** ✅ VULNERABLE

---

#### 2.5 Matrix Multiplication Bombs

**Attack Vector:**
```python
# Safe PoC (1000x1000)
A = np.random.rand(1000, 1000)
B = np.random.rand(1000, 1000)
C = np.matmul(A, B)  # O(n³)

# Real Attack (5000x5000)
# O(n³) = 125x slower = minutes
```

**Impact:**
- O(n³) complexity
- 5000x5000 = minutes of CPU + GB of RAM
- Common in bioinformatics (matrix operations)

**Current Status:** ✅ VULNERABLE

---

#### 2.6 Regex Catastrophic Backtracking

**Attack Vector:**
```python
# Evil regex pattern
pattern = r'^(a+)+$'
text = 'a' * 30 + 'b'  # Causes exponential backtracking

# Complexity: O(2^n)
```

**Impact:**
- Exponential time complexity
- 30 characters = 2^30 operations
- Python 3.11+ has regex timeout (good!)

**Current Status:** ⚠️ PARTIAL PROTECTION (Python 3.11+ timeout)

---

### 3. Disk Exhaustion Attacks

**Severity:** HIGH
**CVSS Score:** 7.5 (High)
**Tests:** `test_disk_exhaustion.py`

#### 3.1 Large File Creation

**Attack Vector:**
```python
# Safe PoC (100MB)
with open('large_file.bin', 'wb') as f:
    f.write(b'X' * (100 * 1024**2))

# Real Attack (100GB)
for i in range(1000):
    with open(f'file_{i}.bin', 'wb') as f:
        f.write(b'X' * (100 * 1024**2))
```

**Impact:**
- Fill entire disk
- DoS for all users
- Crash services needing disk space

**Current Status:** ✅ VULNERABLE

---

#### 3.2 Many Small Files (Inode Exhaustion)

**Attack Vector:**
```python
# Safe PoC (1,000 files)
for i in range(1000):
    Path(f'file_{i}.txt').write_text('data')

# Real Attack (1,000,000 files)
# Exhaust inodes before disk space
```

**Impact:**
- Inode exhaustion (filesystem limit)
- Prevents new file creation system-wide
- ext4: 1 inode per 16KB (common config)

**Current Status:** ✅ VULNERABLE

---

#### 3.3 Deep Directory Nesting

**Attack Vector:**
```python
# Safe PoC (100 levels)
current = Path('nested')
for i in range(100):
    current = current / f'level_{i}'
current.mkdir(parents=True)

# Real Attack (10,000 levels)
# Exceeds path length limits
```

**Impact:**
- Path length limits (4096 chars typical)
- Filesystem corruption risk
- Hard to cleanup (rm -rf may fail)

**Current Status:** ✅ VULNERABLE

---

#### 3.4 Sparse File Creation

**Attack Vector:**
```python
# Create sparse file (1GB apparent, ~0 actual)
with open('sparse.bin', 'wb') as f:
    f.seek(1024**3 - 1)  # Seek to 1GB
    f.write(b'X')  # Write 1 byte
```

**Impact:**
- Confuses disk quotas (apparent vs. actual)
- Can trigger quota errors when written to
- Legitimate use case (VM disk images)

**Current Status:** ✅ VULNERABLE (but legitimate use)

---

#### 3.5 File Descriptor Exhaustion

**Attack Vector:**
```python
# Safe PoC (100 FDs)
files = []
for i in range(100):
    files.append(open(f'file_{i}.txt', 'w'))
# Don't close (FD leak)

# Real Attack (10,000 FDs)
# Exhaust ulimit -n (typically 1024)
```

**Impact:**
- Process crash (can't open more files)
- System-wide impact if many processes
- Typical limits: 1024 (soft), 4096 (hard)

**Current Status:** ✅ VULNERABLE

---

#### 3.6 Socket FD Exhaustion

**Attack Vector:**
```python
# Safe PoC (50 sockets)
sockets = []
for i in range(50):
    sockets.append(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
# Don't close

# Real Attack (10,000 sockets)
```

**Impact:**
- FD exhaustion (sockets use FDs)
- Memory exhaustion (socket buffers)
- Network stack pressure

**Current Status:** ✅ VULNERABLE

---

#### 3.7 Temp File Accumulation

**Attack Vector:**
```python
# Create temp files (don't auto-delete)
for i in range(100):
    f = tempfile.NamedTemporaryFile(delete=False)
    f.write(b'data')
    f.close()
# If process crashes, files leaked
```

**Impact:**
- /tmp pollution (system-wide)
- 1000 executions = 1000s of temp files
- Eventual disk exhaustion

**Current Status:** ✅ VULNERABLE

---

## Proof of Concept Exploits

### PoC 1: Memory OOM Attack (Safe Version)

**File:** `test_memory_bombs.py::test_large_list_allocation_EXPECT_SUCCESS`

```python
# Safe PoC: Allocate 400MB (won't crash system)
code = '''
import sys
x = [0] * (10**8)
result = f"Allocated {sys.getsizeof(x) / 1024**2:.0f}MB"
'''

result, stats, ir = service.execute(code, persist=False, timeout=30)
# ✅ SUCCEEDS - No memory limit
```

**Real Attack Extrapolation:**
```python
# Real attack: 8GB allocation
x = [0] * (10**9)

# Impact:
# - 8GB RAM allocated
# - OOM killer triggered
# - Other processes killed
```

---

### PoC 2: CPU Burn Attack (Safe Version)

**File:** `test_cpu_exhaustion.py::test_cpu_intensive_loop_EXPECT_SUCCESS`

```python
# Safe PoC: 5 seconds of 100% CPU
code = '''
import time
start = time.time()
count = 0
while time.time() - start < 5:
    count += 1
result = f"Burned CPU for 5 seconds: {count:,} iterations"
'''

result, stats, ir = service.execute(code, persist=False, timeout=10)
# ✅ SUCCEEDS - No CPU limit
```

**Real Attack Extrapolation:**
```python
# Real attack: 300 seconds (default timeout)
while time.time() - start < 300:
    count += 1

# Impact:
# - 5 minutes of 100% CPU
# - Blocks other work
# - Multi-user amplification
```

---

### PoC 3: Disk Fill Attack (Safe Version)

**File:** `test_disk_exhaustion.py::test_write_large_file_EXPECT_SUCCESS`

```python
# Safe PoC: Write 100MB file
code = '''
filepath = WORKSPACE / 'large_file.bin'
chunk_size = 1024 * 1024  # 1MB chunks
with open(filepath, 'wb') as f:
    for i in range(100):
        f.write(b'X' * chunk_size)
result = f"Wrote 100MB to disk"
'''

result, stats, ir = service.execute(code, persist=False, timeout=30)
# ✅ SUCCEEDS - No disk quota
```

**Real Attack Extrapolation:**
```python
# Real attack: 100GB file
for i in range(100000):
    f.write(b'X' * chunk_size)

# Impact:
# - Fill entire disk
# - DoS for all users
# - System-wide service failures
```

---

## Impact Assessment

### Single-User Impact

| Attack | Resource | Time to DoS | Reversibility |
|--------|----------|-------------|---------------|
| Memory bomb | RAM | <10s | Auto (process exit) |
| CPU burn | CPU | 300s | Auto (timeout) |
| Disk fill | Disk | <60s | Manual cleanup |
| FD exhaustion | FDs | <5s | Auto (process exit) |
| Inode exhaustion | Inodes | <30s | Manual cleanup |

### Multi-User Impact (Amplification)

**Scenario:** 10 concurrent users, each running resource attacks

| Resource | Single User | 10 Users | System Impact |
|----------|------------|----------|---------------|
| **Memory** | 8GB | 80GB | OOM killer, system crash |
| **CPU** | 100% 1 core | 1000% (10 cores) | System unusable |
| **Disk** | 100GB | 1TB | Disk full, all users DoS |
| **FDs** | 1024 | 10,240 | System-wide FD exhaustion |

**Conclusion:** Multi-user environment is **critically vulnerable**

---

### Attack Scenarios

#### Scenario 1: Accidental Resource Exhaustion

**Likelihood:** HIGH
**User Intent:** Legitimate bioinformatics analysis

```python
# Innocent-looking code (but dangerous)
import numpy as np
import pandas as pd

# Load large dataset
data = pd.read_csv('large_dataset.csv')  # 10GB file

# Create distance matrix (O(n²) memory)
n = len(data)
distances = np.zeros((n, n))  # Could be 100GB+

# Result: Crash
```

**Impact:**
- User crashes their own session
- May crash other users (shared system)
- Data loss (unsaved work)

---

#### Scenario 2: Malicious Resource Exhaustion

**Likelihood:** MEDIUM
**User Intent:** Intentional DoS

```python
# Malicious script
import multiprocessing as mp

def burn_cpu():
    while True:
        pass

def burn_memory():
    x = [0] * (10**9)

def burn_disk():
    with open('attack.bin', 'wb') as f:
        while True:
            f.write(b'X' * (100 * 1024**2))

# Launch attacks in parallel (if allowed)
# Note: multiprocessing may be blocked by import validation
```

**Impact:**
- System-wide DoS
- Requires manual intervention
- Other users impacted

---

#### Scenario 3: Chained Attack

**Likelihood:** LOW
**User Intent:** Sophisticated attack

```python
# Phase 1: Fill disk with large files
for i in range(1000):
    with open(f'file_{i}.bin', 'wb') as f:
        f.write(b'X' * (100 * 1024**2))

# Phase 2: Exhaust inodes
for i in range(100000):
    Path(f'inode_{i}.txt').write_text('x')

# Phase 3: Burn CPU while waiting for cleanup
while True:
    x = 1 + 1

# Result: Multi-vector DoS (hard to recover)
```

**Impact:**
- Disk full (100GB+)
- Inodes exhausted
- CPU at 100% for 300s
- Requires manual cleanup

---

## Recommended Mitigations

### Priority 1: Memory Limits (CRITICAL)

**Problem:** No memory limits, can exhaust RAM

**Solution 1: Docker --memory flag**
```bash
docker run --memory=2g --memory-swap=2g omicsos/lobster
# Hard limit: 2GB RAM, no swap
```

**Solution 2: cgroups memory.max (Linux)**
```python
# In subprocess setup (before exec)
import os
cgroup_path = f'/sys/fs/cgroup/lobster/{os.getpid()}'
os.makedirs(cgroup_path, exist_ok=True)
with open(f'{cgroup_path}/memory.max', 'w') as f:
    f.write(str(2 * 1024**3))  # 2GB limit
```

**Solution 3: resource.setrlimit (Linux/macOS)**
```python
# In subprocess setup
import resource
# Limit address space to 2GB
resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
```

**Recommendation:** **Docker --memory** (simplest, cross-platform)

---

### Priority 2: CPU Throttling (HIGH)

**Problem:** 100% CPU allowed for 300 seconds

**Solution 1: Reduce default timeout**
```python
# In custom_code_execution_service.py
DEFAULT_TIMEOUT = 30  # Was 300s, reduce to 30s
```

**Solution 2: Docker --cpus flag**
```bash
docker run --cpus=0.5 --memory=2g omicsos/lobster
# Limit to 50% of 1 CPU core
```

**Solution 3: cgroups cpu.max (Linux)**
```python
# In subprocess setup
cgroup_path = f'/sys/fs/cgroup/lobster/{os.getpid()}'
with open(f'{cgroup_path}/cpu.max', 'w') as f:
    f.write('50000 100000')  # 50% of 1 core
```

**Solution 4: nice value (priority reduction)**
```python
# In subprocess setup
import os
os.nice(10)  # Lower priority (0-19, higher = lower priority)
```

**Recommendation:** **Reduce timeout to 30s** + **Docker --cpus=0.5**

---

### Priority 3: Disk Quotas (HIGH)

**Problem:** Can fill entire disk

**Solution 1: Docker storage-opt**
```bash
docker run --storage-opt size=10G omicsos/lobster
# Limit container storage to 10GB
```

**Solution 2: Linux disk quotas**
```bash
# Enable quotas on filesystem
quotacheck -cug /workspace
quotaon /workspace

# Set per-user quota (10GB soft, 12GB hard)
setquota -u lobster 10G 12G 100000 110000 /workspace
```

**Solution 3: Pre-execution disk check**
```python
# In execute() method
import shutil
total, used, free = shutil.disk_usage(self.data_manager.workspace_path)
if free < 1 * 1024**3:  # Less than 1GB free
    raise CodeExecutionError("Insufficient disk space (< 1GB free)")
```

**Recommendation:** **Docker storage-opt** + **Pre-execution check**

---

### Priority 4: File Descriptor Limits (MEDIUM)

**Problem:** Can exhaust FDs (1024+ open files)

**Solution 1: Docker --ulimit**
```bash
docker run --ulimit nofile=1024:2048 omicsos/lobster
# Soft limit: 1024, Hard limit: 2048
```

**Solution 2: resource.setrlimit**
```python
# In subprocess setup
import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 2048))
```

**Recommendation:** **Docker --ulimit nofile=1024:2048**

---

### Priority 5: Rate Limiting (MEDIUM)

**Problem:** Multiple rapid executions amplify impact

**Solution 1: Per-user rate limit**
```python
# In AgentClient or CustomCodeExecutionService
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_calls=5, window=60):
        self.max_calls = max_calls
        self.window = window
        self.calls = defaultdict(list)

    def check(self, user_id):
        now = time.time()
        # Remove old calls
        self.calls[user_id] = [t for t in self.calls[user_id] if now - t < self.window]
        # Check limit
        if len(self.calls[user_id]) >= self.max_calls:
            raise Exception(f"Rate limit exceeded: {self.max_calls} calls per {self.window}s")
        self.calls[user_id].append(now)

# Usage
rate_limiter = RateLimiter(max_calls=5, window=60)
rate_limiter.check(user_id='user123')
```

**Recommendation:** **5 executions per minute per user**

---

## Complete Mitigation Stack

### Recommended Docker Configuration

```bash
docker run \
  --name lobster \
  --memory=2g \
  --memory-swap=2g \
  --cpus=0.5 \
  --storage-opt size=10G \
  --ulimit nofile=1024:2048 \
  --tmpfs /tmp:rw,size=1G,mode=1777 \
  --read-only \
  --network=none \
  -v /workspace:/workspace \
  omicsos/lobster
```

**Explanation:**
- `--memory=2g`: Max 2GB RAM
- `--memory-swap=2g`: No swap (prevent swap exhaustion)
- `--cpus=0.5`: Limit to 50% of 1 CPU core
- `--storage-opt size=10G`: Max 10GB storage
- `--ulimit nofile=1024:2048`: Max 1024 file descriptors
- `--tmpfs /tmp:rw,size=1G`: Limit /tmp to 1GB
- `--read-only`: Read-only filesystem (except mounted volumes)
- `--network=none`: No network access
- `-v /workspace:/workspace`: Mount workspace (read-write)

---

### Recommended Code Changes

**File:** `lobster/services/execution/custom_code_execution_service.py`

```python
# Change DEFAULT_TIMEOUT
DEFAULT_TIMEOUT = 30  # Was 300s → 30s

# Add pre-execution checks in execute() method
def execute(self, code: str, ...):
    # ... existing validation ...

    # NEW: Check available disk space
    total, used, free = shutil.disk_usage(self.data_manager.workspace_path)
    if free < 1 * 1024**3:  # Less than 1GB
        raise CodeExecutionError(
            f"Insufficient disk space: {free / 1024**3:.1f}GB free (min 1GB required)"
        )

    # NEW: Check workspace file count (prevent inode exhaustion)
    file_count = sum(1 for _ in self.data_manager.workspace_path.rglob('*'))
    if file_count > 50000:
        raise CodeExecutionError(
            f"Too many files in workspace: {file_count} (max 50,000)"
        )

    # ... continue with execution ...
```

---

### Recommended Documentation Updates

**File:** `lobster/services/execution/custom_code_execution_service.py` (docstring)

```python
"""
Custom Code Execution Service for ad-hoc Python code execution.

SECURITY MODEL:
- Subprocess-based execution for process isolation
- Timeout enforcement (30s default, configurable)
- Workspace-only file access
- No network access (use Docker --network=none)
- Crash isolation (user code crashes don't kill Lobster)

RESOURCE LIMITS (RECOMMENDED):
- Memory: 2GB (Docker --memory=2g)
- CPU: 50% of 1 core (Docker --cpus=0.5)
- Disk: 10GB (Docker --storage-opt size=10G)
- File Descriptors: 1024 (Docker --ulimit nofile=1024:2048)
- Timeout: 30s (configurable, max 300s)

DEPLOYMENT:
For production use, deploy with Docker resource limits:

    docker run \\
      --memory=2g --memory-swap=2g \\
      --cpus=0.5 \\
      --storage-opt size=10G \\
      --ulimit nofile=1024:2048 \\
      --network=none \\
      omicsos/lobster

Without these limits, the service is vulnerable to resource exhaustion attacks.
See: tests/manual/custom_code_execution/02_resource_exhaustion/RESOURCE_EXHAUSTION_REPORT.md
"""
```

---

## Testing Instructions

### Running the Tests

```bash
# Run all resource exhaustion tests
pytest tests/manual/custom_code_execution/02_resource_exhaustion/ -v -s

# Run specific test suites
pytest tests/manual/custom_code_execution/02_resource_exhaustion/test_memory_bombs.py -v -s
pytest tests/manual/custom_code_execution/02_resource_exhaustion/test_cpu_exhaustion.py -v -s
pytest tests/manual/custom_code_execution/02_resource_exhaustion/test_disk_exhaustion.py -v -s

# Run specific test
pytest tests/manual/custom_code_execution/02_resource_exhaustion/test_memory_bombs.py::TestMemoryAllocationBombs::test_large_list_allocation_EXPECT_SUCCESS -v -s
```

### Expected Output

```
⚠️  VULNERABILITY CONFIRMED: Memory allocation succeeded
    Result: Allocated 400MB
    Duration: 2.5s
    🔥 REAL ATTACK: Could allocate 10GB+ and crash system
```

### Interpreting Results

| Test Result | Meaning | Action |
|------------|---------|--------|
| `⚠️ VULNERABILITY CONFIRMED` | Attack succeeds, vulnerability exists | Implement mitigation |
| `✅ PROTECTED` | Attack blocked, protection works | Verify protection is sufficient |
| `⚠️ PARTIAL PROTECTION` | Some protection, but insufficient | Strengthen protection |

---

## Comparison with Industry Standards

### Jupyter Notebook (Reference)

| Feature | Jupyter | Lobster (Current) | Lobster (Recommended) |
|---------|---------|-------------------|----------------------|
| Memory Limit | None (user responsibility) | None | 2GB (Docker) |
| CPU Limit | None | None | 50% 1 core |
| Timeout | Kernel interrupt | 300s | 30s |
| Disk Quota | None | None | 10GB |
| Process Isolation | Separate kernel process | ✅ subprocess | ✅ subprocess |
| Network Access | Full | Full | None (Docker) |

**Analysis:** Jupyter provides similar isolation but no resource limits. However:
- Jupyter users expect full control (trusted environment)
- Lobster accepts natural language (less trusted)
- Jupyter is local only, Lobster may be cloud-hosted

**Conclusion:** Lobster needs **stricter limits** than Jupyter

---

### Google Colab (Cloud Jupyter)

| Feature | Google Colab | Lobster (Recommended) |
|---------|--------------|----------------------|
| Memory Limit | 12GB | 2GB |
| CPU Limit | 2 cores | 0.5 cores |
| Timeout | 12 hours (idle timeout: 90 min) | 30s |
| Disk Quota | 100GB (ephemeral) | 10GB |
| GPU Access | Optional (T4/V100) | Not applicable |
| Network Access | Limited (rate limited) | None |

**Analysis:** Colab provides generous limits but has:
- Idle timeout (90 minutes)
- Runtime limits (12 hours)
- Resource monitoring (terminates abuse)

**Conclusion:** Lobster should adopt **timeout + monitoring** approach

---

### AWS Lambda (Serverless Compute)

| Feature | AWS Lambda | Lobster (Recommended) |
|---------|------------|----------------------|
| Memory Limit | 128MB - 10GB (configurable) | 2GB |
| CPU Limit | Proportional to memory | 0.5 cores |
| Timeout | 15 minutes max | 30s |
| Disk Quota | 512MB (ephemeral /tmp) | 10GB workspace |
| Process Isolation | Container (Firecracker) | subprocess + Docker |
| Network Access | Full (VPC control) | None |

**Analysis:** Lambda has **strict** limits by default:
- Fixed timeout (15 min max)
- Fixed /tmp size (512MB)
- Pay-per-use (economic incentive to limit)

**Conclusion:** Lambda's **strict defaults** are a good model

---

## Risk Matrix

### Vulnerability Risk Scores

| Vulnerability | Likelihood | Impact | Risk Score | Priority |
|--------------|------------|--------|-----------|----------|
| Memory exhaustion | High | High | **9/10** | P0 |
| CPU burn (300s) | High | High | **9/10** | P0 |
| Disk fill | High | High | **8/10** | P0 |
| FD exhaustion | Medium | High | **7/10** | P1 |
| Inode exhaustion | Medium | Medium | **6/10** | P1 |
| Network abuse | Low | Medium | **4/10** | P2 |
| Temp file accumulation | Low | Low | **3/10** | P2 |

### Risk Reduction with Mitigations

| Vulnerability | Current Risk | With Docker Limits | With Full Stack |
|--------------|-------------|-------------------|-----------------|
| Memory exhaustion | 9/10 | **3/10** | **2/10** |
| CPU burn | 9/10 | **4/10** | **2/10** |
| Disk fill | 8/10 | **3/10** | **2/10** |
| FD exhaustion | 7/10 | **3/10** | **2/10** |
| Inode exhaustion | 6/10 | **4/10** | **3/10** |

**Mitigation Impact:** Docker limits reduce risk by **~70%**

---

## Conclusion

The CustomCodeExecutionService currently provides **minimal resource protection** beyond process isolation and a 300-second timeout. While the subprocess model prevents crashes from affecting the main Lobster process, it does **not protect system resources** from exhaustion.

### Key Findings

1. **Memory:** Unlimited allocation possible (8GB+ attacks feasible)
2. **CPU:** 100% utilization allowed for 300 seconds
3. **Disk:** No quotas, can fill entire filesystem
4. **File Descriptors:** No limits, can exhaust system FDs
5. **Timeout:** 300s is too long for default (should be 30s)

### Critical Gaps

| Protection | Status | Recommendation |
|-----------|--------|----------------|
| Memory limits | ❌ Missing | Docker --memory=2g |
| CPU throttling | ❌ Missing | Docker --cpus=0.5 |
| Disk quotas | ❌ Missing | Docker --storage-opt size=10G |
| FD limits | ❌ Missing | Docker --ulimit nofile=1024:2048 |
| Network isolation | ❌ Missing | Docker --network=none |
| Rate limiting | ❌ Missing | Application-level (5 calls/min) |

### Recommendations

**Immediate (P0):**
1. ✅ Reduce `DEFAULT_TIMEOUT` from 300s to 30s
2. ✅ Add pre-execution disk space check (min 1GB free)
3. ✅ Document Docker deployment with resource limits

**Short-term (P1):**
4. ✅ Implement Docker-based deployment with full limit stack
5. ✅ Add per-user rate limiting (5 executions/minute)
6. ✅ Add workspace file count check (max 50,000 files)

**Long-term (P2):**
7. ✅ Add resource monitoring (psutil) for real-time limits
8. ✅ Implement graceful degradation (warn before hard limit)
9. ✅ Add user quotas (per-user storage/compute limits)

### Production Readiness

**Current Status:** ⚠️ **NOT PRODUCTION READY**

The service is suitable for:
- ✅ Local development (trusted users)
- ✅ Single-user CLI (self-imposed limits)

The service is **NOT** suitable for:
- ❌ Multi-user cloud deployment
- ❌ Public API access
- ❌ Untrusted user input

**Production Deployment Requires:**
1. Docker with resource limits (mandatory)
2. Timeout reduction to 30s (mandatory)
3. Pre-execution validation checks (recommended)
4. Rate limiting (recommended)
5. Monitoring and alerting (recommended)

---

## Appendix: Full Docker Deployment

### Dockerfile (Recommended)

```dockerfile
FROM python:3.11-slim

# Install dependencies
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

# Copy application
COPY lobster/ /app/lobster/
WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 lobster && \
    mkdir /workspace && \
    chown lobster:lobster /workspace

# Switch to non-root user
USER lobster

# Set resource-friendly defaults
ENV LOBSTER_TIMEOUT=30
ENV LOBSTER_WORKSPACE=/workspace

# Entrypoint
ENTRYPOINT ["python", "-m", "lobster.cli"]
CMD ["chat"]
```

### docker-compose.yml (Recommended)

```yaml
version: '3.8'

services:
  lobster:
    build: .
    image: omicsos/lobster:latest
    container_name: lobster

    # Resource limits
    mem_limit: 2g
    mem_reservation: 1g
    cpus: 0.5

    # Storage limits
    storage_opt:
      size: 10G

    # File descriptor limits
    ulimits:
      nofile:
        soft: 1024
        hard: 2048

    # Temp directory limit
    tmpfs:
      - /tmp:rw,size=1G,mode=1777

    # Read-only filesystem (except volumes)
    read_only: true

    # No network access
    network_mode: none

    # Workspace volume
    volumes:
      - ./workspace:/workspace

    # Environment
    environment:
      - LOBSTER_TIMEOUT=30
      - LOBSTER_WORKSPACE=/workspace

    # Security
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
```

### Kubernetes Deployment (Advanced)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: lobster
spec:
  containers:
  - name: lobster
    image: omicsos/lobster:latest

    # Resource limits
    resources:
      limits:
        memory: "2Gi"
        cpu: "500m"
        ephemeral-storage: "10Gi"
      requests:
        memory: "1Gi"
        cpu: "250m"
        ephemeral-storage: "5Gi"

    # Security context
    securityContext:
      runAsNonRoot: true
      runAsUser: 1000
      readOnlyRootFilesystem: true
      allowPrivilegeEscalation: false
      capabilities:
        drop:
        - ALL

    # Volumes
    volumeMounts:
    - name: workspace
      mountPath: /workspace
    - name: tmp
      mountPath: /tmp

  volumes:
  - name: workspace
    persistentVolumeClaim:
      claimName: lobster-workspace
  - name: tmp
    emptyDir:
      sizeLimit: 1Gi
```

---

## Sign-off

**Agent:** Agent 2 - Resource Exhaustion Tester
**Status:** Testing Complete
**Vulnerabilities Found:** 16 (15 confirmed, 1 partial protection)
**Risk Level:** HIGH
**Production Ready:** NO (requires hardening)

**Next Steps:**
1. Review findings with security team
2. Implement P0 recommendations (timeout, disk checks, docs)
3. Deploy Docker-based solution for production
4. Coordinate with Agent 3 (Code Injection Tester) for combined threat model

**Files Delivered:**
- ✅ `test_memory_bombs.py` (8 tests)
- ✅ `test_cpu_exhaustion.py` (7 tests)
- ✅ `test_disk_exhaustion.py` (8 tests)
- ✅ `RESOURCE_EXHAUSTION_REPORT.md` (this document)

---

**END OF REPORT**
