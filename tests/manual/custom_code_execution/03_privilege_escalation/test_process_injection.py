"""
Process Injection and Memory Manipulation Tests for CustomCodeExecutionService

Tests whether user code can inject into or manipulate parent process memory:
- /proc/PID/mem access (Linux)
- ptrace debugging
- Shared memory attacks
- Memory manipulation via ctypes/cffi

SECURITY MODEL: subprocess.run() provides process isolation, but doesn't
prevent all forms of inter-process manipulation.

⚠️ SAFETY: Tests only CHECK for capability, DO NOT perform actual injection.
"""

import os
import tempfile
from pathlib import Path

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.execution.custom_code_execution_service import (
    CodeExecutionError,
    CodeValidationError,
    CustomCodeExecutionService,
)


class TestMemoryInjection:
    """Test memory manipulation and injection attacks."""

    @pytest.fixture
    def workspace(self, tmp_path):
        """Create temporary workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        return workspace

    @pytest.fixture
    def service(self, workspace):
        """Create service instance."""
        data_manager = DataManagerV2(workspace_path=workspace)
        return CustomCodeExecutionService(data_manager)

    # =========================================================================
    # Test 1: /proc/PID/mem Access (Linux Only)
    # =========================================================================

    def test_proc_mem_access_EXPECT_BLOCKED_ON_MACOS(self, service):
        """
        Test: Access parent process memory via /proc/PID/mem
        Expected: BLOCKED on macOS (no /proc), VARIES on Linux (permissions)
        Impact: CRITICAL - Direct memory manipulation

        Attack: Read/write parent process memory
        """
        code = """
import os
from pathlib import Path

try:
    parent_pid = os.getppid()
    mem_path = Path(f"/proc/{parent_pid}/mem")

    if mem_path.exists():
        # Try to open (will likely fail due to permissions)
        try:
            with open(mem_path, 'rb') as f:
                result = "VULNERABILITY: Can open /proc/PID/mem"
                print(f"⚠️  CRITICAL: /proc/{parent_pid}/mem accessible")
                print(f"   Could read/write parent process memory")
                print(f"   Attack: Inject shellcode, modify data structures")
        except PermissionError:
            result = "PROTECTED: /proc/PID/mem exists but permission denied"
            print(f"✅ Permission denied for /proc/{parent_pid}/mem")
            print(f"   Linux kernel prevents cross-process memory access")
    else:
        result = "PROTECTED: /proc filesystem not available (macOS expected)"
        print(f"✅ /proc filesystem not available")
        print(f"   macOS does not expose process memory via filesystem")
except Exception as e:
    result = f"Error: {e}"
"""

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: /proc/PID/mem Memory Injection")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "VULNERABILITY" in str(result):
            print("\n⚠️  CRITICAL VULNERABILITY")
            print("   Attack Vector: /proc/PID/mem accessible")
            print("   Impact: Direct memory manipulation")
            print("   Mitigation: PID namespaces")
        else:
            print("\n✅ PROTECTED")
            print("   Either /proc unavailable (macOS) or permission denied (Linux)")

    # =========================================================================
    # Test 2: ptrace System Call
    # =========================================================================

    def test_ptrace_capability_EXPECT_BLOCKED(self, service):
        """
        Test: Attach debugger to parent process via ptrace
        Expected: BLOCKED (requires CAP_SYS_PTRACE or same UID)
        Impact: CRITICAL - Full process control

        Attack: Use ptrace to debug and control parent
        """
        code = """
import os
import ctypes
import ctypes.util

try:
    # Try to load libc and access ptrace
    libc_name = ctypes.util.find_library('c')
    if libc_name:
        libc = ctypes.CDLL(libc_name)

        # Check if ptrace is available
        if hasattr(libc, 'ptrace'):
            parent_pid = os.getppid()

            result = "VULNERABILITY: ptrace available (likely permission denied)"
            print(f"⚠️ ptrace system call available via ctypes")
            print(f"   Target PID: {parent_pid}")
            print(f"   Attack: libc.ptrace(PTRACE_ATTACH, parent_pid, 0, 0)")
            print(f"   Impact: Full debugger control over parent")
            print(f"   Note: Likely blocked by permissions/macOS SIP")
            print(f"   ⚠️ NOT EXECUTED - would fail with EPERM")
        else:
            result = "PROTECTED: ptrace not found in libc"
    else:
        result = "PROTECTED: libc not loadable"
except Exception as e:
    result = f"Error: {e}"
"""

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: ptrace Debugging Attack")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "VULNERABILITY" in str(result):
            print("\n⚠️  VULNERABILITY DETECTED (LIKELY MITIGATED)")
            print("   Attack Vector: ptrace system call available")
            print("   Impact: Could control parent process")
            print("   Mitigation: Kernel permissions (EPERM), macOS SIP")
            print("   Note: Usually blocked by OS, but shouldn't be accessible")

    # =========================================================================
    # Test 3: Shared Memory via mmap
    # =========================================================================

    def test_shared_memory_creation_EXPECT_SUCCESS(self, service):
        """
        Test: Create shared memory segments
        Expected: SUCCESS (vulnerability - mmap available)
        Impact: MEDIUM - IPC channel, not direct parent access

        Attack: Create shared memory for data exfiltration
        """
        code = """
import mmap
import os

try:
    # Create anonymous shared memory
    shm = mmap.mmap(-1, 1024, flags=mmap.MAP_SHARED)

    result = "VULNERABILITY: Can create shared memory (mmap)"
    print(f"⚠️ mmap.MAP_SHARED available")
    print(f"   Created: 1KB shared memory segment")
    print(f"   Impact: IPC channel, not direct parent access")
    print(f"   Attack: If parent cooperates, could exfiltrate data")
    print(f"   Note: Doesn't grant access to parent's existing memory")

    shm.close()
except Exception as e:
    result = f"Error: {e}"
"""

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: Shared Memory Creation")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "VULNERABILITY" in str(result):
            print("\n⚠️  LOW-IMPACT VULNERABILITY")
            print("   Attack Vector: mmap shared memory")
            print("   Impact: Can create IPC channels")
            print("   Limitation: Doesn't access parent memory")
            print("   Note: Standard IPC mechanism")

    # =========================================================================
    # Test 4: ctypes Memory Manipulation
    # =========================================================================

    def test_ctypes_memory_access_EXPECT_SUCCESS(self, service):
        """
        Test: Access arbitrary memory addresses via ctypes
        Expected: SUCCESS (vulnerability - ctypes available)
        Impact: HIGH - Can crash subprocess, not parent

        Attack: Read/write arbitrary memory in subprocess
        """
        code = """
import ctypes

try:
    # Can access subprocess's own memory
    value = ctypes.c_int(42)
    ptr = ctypes.pointer(value)

    result = "VULNERABILITY: ctypes allows arbitrary memory access"
    print(f"⚠️ ctypes available for memory manipulation")
    print(f"   Created pointer: {ptr}")
    print(f"   Can access: ctypes.cast(address, ctypes.POINTER(ctypes.c_char))")
    print(f"   Impact: Full memory access within subprocess")
    print(f"   Limitation: Cannot access parent process memory")
    print(f"   Could crash: Segfault subprocess by accessing invalid addresses")
except Exception as e:
    result = f"Error: {e}"
"""

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: ctypes Memory Manipulation")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "VULNERABILITY" in str(result):
            print("\n⚠️  VULNERABILITY DETECTED")
            print("   Attack Vector: ctypes memory manipulation")
            print("   Impact: Can crash subprocess")
            print("   Limitation: Only subprocess memory, not parent")
            print("   Note: Standard Python capability")


class TestIPCMechanisms:
    """Test inter-process communication mechanisms."""

    @pytest.fixture
    def workspace(self, tmp_path):
        """Create temporary workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        return workspace

    @pytest.fixture
    def service(self, workspace):
        """Create service instance."""
        data_manager = DataManagerV2(workspace_path=workspace)
        return CustomCodeExecutionService(data_manager)

    # =========================================================================
    # Test 5: Unix Domain Sockets
    # =========================================================================

    def test_unix_socket_creation_EXPECT_SUCCESS(self, service):
        """
        Test: Create Unix domain sockets for IPC
        Expected: SUCCESS (vulnerability - socket available)
        Impact: MEDIUM - IPC channel

        Attack: Create socket for command/control
        """
        code = """
import socket
import tempfile
from pathlib import Path

try:
    # Create Unix domain socket
    socket_path = Path(tempfile.gettempdir()) / "evil_socket.sock"

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    # Don't actually bind (would require cleanup)

    result = "VULNERABILITY: Can create Unix domain sockets"
    print(f"⚠️ socket module available")
    print(f"   Can create: AF_UNIX sockets")
    print(f"   Impact: IPC channel for C&C")
    print(f"   Attack: sock.bind('/tmp/evil.sock'); listen for commands")
    print(f"   Note: Subprocess isolated, but socket persists")

    sock.close()
except Exception as e:
    result = f"Error: {e}"
"""

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: Unix Domain Socket Creation")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "VULNERABILITY" in str(result):
            print("\n⚠️  VULNERABILITY DETECTED")
            print("   Attack Vector: Unix domain sockets")
            print("   Impact: IPC channel survives subprocess")
            print("   Real Attack: C&C server in /tmp")
            print("   Mitigation: Network namespaces")

    # =========================================================================
    # Test 6: Named Pipes (FIFO)
    # =========================================================================

    def test_named_pipe_creation_EXPECT_SUCCESS(self, service):
        """
        Test: Create named pipes for IPC
        Expected: SUCCESS (vulnerability - os.mkfifo available)
        Impact: MEDIUM - IPC channel

        Attack: Create persistent IPC channel
        """
        code = """
import os
import tempfile
from pathlib import Path

try:
    # Test if mkfifo is available
    if hasattr(os, 'mkfifo'):
        result = "VULNERABILITY: Can create named pipes (FIFO)"
        print(f"⚠️ os.mkfifo() available")
        print(f"   Can create: Named pipes in filesystem")
        print(f"   Impact: Persistent IPC channel")
        print(f"   Attack: os.mkfifo('/tmp/evil_pipe'); exfiltrate data")
    else:
        result = "PROTECTED: mkfifo not available (Windows expected)"
except Exception as e:
    result = f"Error: {e}"
"""

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: Named Pipe (FIFO) Creation")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "VULNERABILITY" in str(result):
            print("\n⚠️  VULNERABILITY DETECTED")
            print("   Attack Vector: Named pipes (FIFO)")
            print("   Impact: Persistent IPC channel")
            print("   Real Attack: Create pipe, reader in background")
            print("   Mitigation: Filesystem restrictions")

    # =========================================================================
    # Test 7: System V IPC (shmget, semget, msgget)
    # =========================================================================

    def test_sysv_ipc_EXPECT_PARTIAL(self, service):
        """
        Test: Use System V IPC (shared memory, semaphores, message queues)
        Expected: PARTIAL (sysv_ipc module likely not installed)
        Impact: MEDIUM - IPC mechanisms

        Attack: Use kernel IPC primitives
        """
        code = """
try:
    import sysv_ipc

    result = "VULNERABILITY: sysv_ipc module available"
    print(f"⚠️ sysv_ipc module installed")
    print(f"   Can create: Shared memory, semaphores, message queues")
    print(f"   Impact: Kernel-level IPC")
except ImportError:
    result = "PROTECTED: sysv_ipc not installed"
    print(f"✅ sysv_ipc not available")
    print(f"   Cannot use System V IPC primitives")
"""

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: System V IPC Mechanisms")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "VULNERABILITY" in str(result):
            print("\n⚠️  VULNERABILITY DETECTED")
            print("   Attack Vector: System V IPC")
            print("   Impact: Kernel IPC mechanisms")
            print("   Mitigation: Remove sysv_ipc module")


class TestDockerEscapes:
    """Test Docker-specific escape mechanisms."""

    @pytest.fixture
    def workspace(self, tmp_path):
        """Create temporary workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        return workspace

    @pytest.fixture
    def service(self, workspace):
        """Create service instance."""
        data_manager = DataManagerV2(workspace_path=workspace)
        return CustomCodeExecutionService(data_manager)

    # =========================================================================
    # Test 8: Docker Socket Access
    # =========================================================================

    def test_docker_socket_access_EXPECT_VARIES(self, service):
        """
        Test: Access Docker socket to control containers
        Expected: VARIES (depends on Docker socket mounting)
        Impact: CRITICAL - Container escape

        Attack: Control Docker daemon
        """
        code = """
from pathlib import Path

try:
    docker_sock = Path('/var/run/docker.sock')

    if docker_sock.exists():
        result = "VULNERABILITY: Docker socket accessible"
        print(f"⚠️  CRITICAL: /var/run/docker.sock exists")
        print(f"   Impact: Can control Docker daemon")
        print(f"   Attack: Spawn privileged container, escape to host")
        print(f"   Real attack: docker run --privileged -v /:/host ...")
    else:
        result = "PROTECTED: Docker socket not accessible"
        print(f"✅ Docker socket not mounted")
        print(f"   Not running in Docker or socket not exposed")
except Exception as e:
    result = f"Error: {e}"
"""

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: Docker Socket Access")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "VULNERABILITY" in str(result):
            print("\n⚠️  CRITICAL VULNERABILITY")
            print("   Attack Vector: Docker socket mounted")
            print("   Impact: Complete container escape")
            print("   Mitigation: Never mount /var/run/docker.sock")

    # =========================================================================
    # Test 9: /proc Filesystem Docker Escape
    # =========================================================================

    def test_proc_cgroup_analysis_EXPECT_PARTIAL(self, service):
        """
        Test: Analyze cgroups to detect container environment
        Expected: PARTIAL (varies by platform)
        Impact: LOW - Information disclosure

        Attack: Fingerprint container environment
        """
        code = """
from pathlib import Path

try:
    cgroup_path = Path('/proc/self/cgroup')

    if cgroup_path.exists():
        cgroup_data = cgroup_path.read_text()

        in_docker = 'docker' in cgroup_data
        in_k8s = 'kube' in cgroup_data or 'k8s' in cgroup_data

        if in_docker or in_k8s:
            result = f"INFO: Running in container (docker={in_docker}, k8s={in_k8s})"
            print(f"ℹ️  Container environment detected")
            print(f"   Docker: {in_docker}")
            print(f"   Kubernetes: {in_k8s}")
            print(f"   Impact: Information disclosure only")
        else:
            result = "INFO: Not in container (cgroup exists but not containerized)"
            print(f"ℹ️  /proc/self/cgroup exists but not containerized")
    else:
        result = "INFO: /proc not available (macOS expected)"
        print(f"ℹ️  /proc filesystem not available")
except Exception as e:
    result = f"Error: {e}"
"""

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: Container Environment Detection")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        print("\nℹ️  INFORMATION DISCLOSURE")
        print("   Can detect if running in container")
        print("   Low impact - just fingerprinting")

    # =========================================================================
    # Test 10: Privileged Syscalls (capsh)
    # =========================================================================

    def test_capability_detection_EXPECT_VARIES(self, service):
        """
        Test: Check for elevated Linux capabilities
        Expected: VARIES (depends on container config)
        Impact: HIGH - Indicates weak isolation

        Attack: Detect privileged mode
        """
        code = """
import os
from pathlib import Path

try:
    # Check for capabilities on Linux
    status_path = Path('/proc/self/status')

    if status_path.exists():
        status = status_path.read_text()

        # Look for CapEff (effective capabilities)
        cap_lines = [line for line in status.split('\\n') if 'Cap' in line]

        result = f"INFO: Capability check (Linux)"
        print(f"ℹ️  Process capabilities (Linux):")
        for line in cap_lines[:5]:
            print(f"   {line}")

        # Check if running as root
        if os.getuid() == 0:
            print(f"⚠️  Running as root (UID=0)")
            result = "VULNERABILITY: Running as root"
    else:
        result = "INFO: /proc not available (macOS expected)"
        print(f"ℹ️  Not on Linux")
        print(f"   UID: {os.getuid()}")
except Exception as e:
    result = f"Error: {e}"
"""

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: Linux Capabilities Detection")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "root" in str(result).lower():
            print("\n⚠️  VULNERABILITY DETECTED")
            print("   Running as root (UID=0)")
            print("   Impact: Elevated privileges")


if __name__ == "__main__":
    print("=" * 70)
    print("PROCESS INJECTION & MEMORY MANIPULATION TESTS")
    print("=" * 70)
    print("Run with: pytest test_process_injection.py -v -s")
    print("")
    print("These tests check if user code can:")
    print("  1. Access parent process memory (/proc/PID/mem)")
    print("  2. Use ptrace to debug parent")
    print("  3. Create IPC channels (sockets, pipes, shared memory)")
    print("  4. Escape Docker containers")
    print("")
    print("⚠️  EXPECTED RESULTS:")
    print("  - /proc/PID/mem: BLOCKED (permissions or macOS)")
    print("  - ptrace: BLOCKED (permissions/SIP)")
    print("  - IPC channels: SUCCESS (Unix sockets, pipes, mmap)")
    print("  - Docker socket: VARIES (depends on mounting)")
    print("")
    print("⚠️  SAFETY: Tests only check capability, no actual attacks")
    print("=" * 70)
