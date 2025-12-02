"""
Signal Manipulation Tests for CustomCodeExecutionService

Tests whether user code can manipulate signals to affect the parent Lobster process:
- Find parent PID
- Send signals to parent (SIGKILL, SIGTERM, SIGSTOP)
- Hijack signal handlers
- Interrupt parent process

SECURITY MODEL: subprocess.run() creates isolated process with separate PID.
User code CAN discover parent PID and attempt signal-based attacks.

⚠️ SAFETY: Tests only CHECK for capability, DO NOT send actual kill signals.
"""

import os
import signal
import tempfile
from pathlib import Path

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.execution.custom_code_execution_service import (
    CustomCodeExecutionService,
    CodeValidationError,
    CodeExecutionError
)


class TestParentProcessDiscovery:
    """Test if user code can discover and interact with parent process."""

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
    # Test 1: Find Parent PID
    # =========================================================================

    def test_discover_parent_pid_EXPECT_SUCCESS(self, service):
        """
        Test: Discover parent process PID
        Expected: SUCCESS (vulnerability - os.getppid() available)
        Impact: HIGH - Prerequisite for signal attacks

        Attack: Find Lobster's main process PID
        """
        code = '''
import os

try:
    current_pid = os.getpid()
    parent_pid = os.getppid()

    result = f"VULNERABILITY: Parent PID={parent_pid} (current={current_pid})"
    print(f"⚠️ Parent process discovery successful")
    print(f"   Current subprocess PID: {current_pid}")
    print(f"   Parent Lobster PID: {parent_pid}")
    print(f"   Can now target parent with signals")
except Exception as e:
    result = f"PROTECTED: {e}"
'''

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: Parent Process Discovery")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "VULNERABILITY" in str(result):
            print("\n⚠️  VULNERABILITY DETECTED")
            print("   Attack Vector: os.getppid() reveals parent PID")
            print("   Impact: Enables targeted signal attacks")
            print("   Next Attack: os.kill(parent_pid, signal.SIGKILL)")
            print("   Mitigation: Use PID namespaces (Docker/Kubernetes)")

    # =========================================================================
    # Test 2: Check Parent Process Details
    # =========================================================================

    def test_read_parent_cmdline_EXPECT_PARTIAL(self, service):
        """
        Test: Read parent process command line
        Expected: PARTIAL (works on Linux /proc, fails on macOS)
        Impact: MEDIUM - Information disclosure

        Attack: Read parent's command line arguments
        """
        code = '''
import os
import sys
from pathlib import Path

try:
    parent_pid = os.getppid()
    cmdline_path = Path(f"/proc/{parent_pid}/cmdline")

    if cmdline_path.exists():
        # Linux only
        cmdline = cmdline_path.read_text().replace('\\x00', ' ')
        result = f"VULNERABILITY: Parent cmdline={cmdline[:100]}"
        print(f"⚠️ Parent process command line readable")
        print(f"   {cmdline}")
    else:
        result = f"PROTECTED: /proc not available (macOS expected)"
        print(f"✅ /proc filesystem not available (macOS)")
        print(f"   On Linux, could read: /proc/{parent_pid}/cmdline")
        print(f"   Would reveal: Python interpreter, script path, arguments")
except Exception as e:
    result = f"Error: {e}"
'''

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: Parent Process Command Line")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "VULNERABILITY" in str(result):
            print("\n⚠️  VULNERABILITY DETECTED")
            print("   Attack Vector: /proc filesystem access")
            print("   Impact: Information disclosure (command line args)")
            print("   Mitigation: Use PID namespaces")
        else:
            print("\n✅ PLATFORM-SPECIFIC PROTECTION")
            print("   macOS doesn't expose /proc filesystem")
            print("   Linux would be vulnerable")


class TestSignalAttacks:
    """Test signal-based attacks on parent process."""

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
    # Test 3: SIGTERM Attack (Graceful Shutdown)
    # =========================================================================

    def test_sigterm_capability_EXPECT_SUCCESS(self, service):
        """
        Test: Check if we can send SIGTERM to parent
        Expected: SUCCESS (vulnerability - os.kill available)
        Impact: CRITICAL - Can terminate Lobster

        Attack: os.kill(parent_pid, signal.SIGTERM)

        ⚠️ SAFETY: Only tests capability, does NOT send signal
        """
        code = '''
import os
import signal

try:
    parent_pid = os.getppid()

    # CHECK only - DO NOT actually send signal
    if hasattr(os, 'kill') and hasattr(signal, 'SIGTERM'):
        result = f"VULNERABILITY: Can send SIGTERM to parent (PID={parent_pid})"
        print(f"⚠️ os.kill() and signal.SIGTERM available")
        print(f"   Target PID: {parent_pid}")
        print(f"   Attack command: os.kill({parent_pid}, signal.SIGTERM)")
        print(f"   Impact: Gracefully terminates Lobster process")
        print(f"   ⚠️ NOT EXECUTED - test only")
    else:
        result = "PROTECTED: os.kill or SIGTERM not available"
except Exception as e:
    result = f"Error: {e}"
'''

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: SIGTERM Attack Capability")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "VULNERABILITY" in str(result):
            print("\n⚠️  CRITICAL VULNERABILITY")
            print("   Attack Vector: os.kill() with SIGTERM")
            print("   Impact: Can gracefully terminate Lobster")
            print("   Real Attack: os.kill(os.getppid(), signal.SIGTERM)")
            print("   Mitigation: Disable os.kill or use PID namespaces")

    # =========================================================================
    # Test 4: SIGKILL Attack (Forced Kill)
    # =========================================================================

    def test_sigkill_capability_EXPECT_SUCCESS(self, service):
        """
        Test: Check if we can send SIGKILL to parent
        Expected: SUCCESS (vulnerability - SIGKILL available)
        Impact: CRITICAL - Can force-kill Lobster

        Attack: os.kill(parent_pid, signal.SIGKILL)

        ⚠️ SAFETY: Only tests capability, does NOT send signal
        """
        code = '''
import os
import signal

try:
    parent_pid = os.getppid()

    # CHECK only - DO NOT actually send signal
    if hasattr(signal, 'SIGKILL'):
        result = f"VULNERABILITY: Can send SIGKILL to parent (PID={parent_pid})"
        print(f"⚠️ signal.SIGKILL available")
        print(f"   Target PID: {parent_pid}")
        print(f"   Attack command: os.kill({parent_pid}, signal.SIGKILL)")
        print(f"   Impact: Immediately kills Lobster (no cleanup)")
        print(f"   Cannot be caught or ignored by parent")
        print(f"   ⚠️ NOT EXECUTED - test only")
    else:
        result = "PROTECTED: SIGKILL not available"
except Exception as e:
    result = f"Error: {e}"
'''

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: SIGKILL Attack Capability")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "VULNERABILITY" in str(result):
            print("\n⚠️  CRITICAL VULNERABILITY")
            print("   Attack Vector: os.kill() with SIGKILL")
            print("   Impact: Immediately kills Lobster process")
            print("   Cannot be defended against by signal handlers")
            print("   Real Attack: os.kill(os.getppid(), 9)")
            print("   Mitigation: PID namespaces or disable os.kill")

    # =========================================================================
    # Test 5: SIGSTOP Attack (Freeze Process)
    # =========================================================================

    def test_sigstop_capability_EXPECT_SUCCESS(self, service):
        """
        Test: Check if we can send SIGSTOP to parent
        Expected: SUCCESS (vulnerability - SIGSTOP available)
        Impact: HIGH - Can freeze Lobster

        Attack: os.kill(parent_pid, signal.SIGSTOP)

        ⚠️ SAFETY: Only tests capability, does NOT send signal
        """
        code = '''
import os
import signal

try:
    parent_pid = os.getppid()

    # CHECK only - DO NOT actually send signal
    if hasattr(signal, 'SIGSTOP'):
        result = f"VULNERABILITY: Can send SIGSTOP to parent (PID={parent_pid})"
        print(f"⚠️ signal.SIGSTOP available")
        print(f"   Target PID: {parent_pid}")
        print(f"   Attack command: os.kill({parent_pid}, signal.SIGSTOP)")
        print(f"   Impact: Freezes Lobster (requires SIGCONT to resume)")
        print(f"   DoS attack: User must manually kill -CONT")
        print(f"   ⚠️ NOT EXECUTED - test only")
    else:
        result = "PROTECTED: SIGSTOP not available"
except Exception as e:
    result = f"Error: {e}"
'''

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: SIGSTOP Attack Capability")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "VULNERABILITY" in str(result):
            print("\n⚠️  VULNERABILITY DETECTED")
            print("   Attack Vector: os.kill() with SIGSTOP")
            print("   Impact: Denial of Service (freezes parent)")
            print("   Real Attack: os.kill(os.getppid(), signal.SIGSTOP)")
            print("   Mitigation: PID namespaces")

    # =========================================================================
    # Test 6: Signal Handler Hijacking (Self)
    # =========================================================================

    def test_hijack_own_signal_handlers_EXPECT_SUCCESS(self, service):
        """
        Test: Hijack subprocess's own signal handlers
        Expected: SUCCESS (vulnerability - signal.signal available)
        Impact: LOW - Only affects subprocess, not parent

        Attack: Override signal handlers to ignore signals
        """
        code = '''
import signal

try:
    # Override SIGTERM handler in subprocess (not parent)
    def ignore_signal(signum, frame):
        print(f"Signal {signum} ignored!")

    signal.signal(signal.SIGTERM, ignore_signal)
    signal.signal(signal.SIGINT, ignore_signal)

    result = "VULNERABILITY: Can override signal handlers (subprocess only)"
    print(f"⚠️ signal.signal() available")
    print(f"   Overrode: SIGTERM, SIGINT handlers")
    print(f"   Impact: Subprocess can ignore ctrl-C")
    print(f"   Note: Only affects subprocess, not parent Lobster")
except Exception as e:
    result = f"Error: {e}"
'''

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: Signal Handler Hijacking (Subprocess)")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "VULNERABILITY" in str(result):
            print("\n⚠️  LOW-IMPACT VULNERABILITY")
            print("   Attack Vector: signal.signal() available")
            print("   Impact: Subprocess can ignore signals")
            print("   Limitation: Doesn't affect parent process")
            print("   Note: Timeout still enforced by parent")


class TestAdvancedProcessAttacks:
    """Test advanced process manipulation attacks."""

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
    # Test 7: Process Tree Discovery
    # =========================================================================

    def test_enumerate_process_tree_EXPECT_PARTIAL(self, service):
        """
        Test: Enumerate entire process tree
        Expected: PARTIAL (psutil if installed, otherwise limited)
        Impact: MEDIUM - Information disclosure

        Attack: Discover all running processes
        """
        code = '''
import os
import sys

try:
    # Try psutil (may not be installed)
    try:
        import psutil
        current_process = psutil.Process(os.getpid())
        parent_process = current_process.parent()

        result = f"VULNERABILITY: Full process tree accessible via psutil"
        print(f"⚠️ psutil available - can enumerate all processes")
        print(f"   Current: {current_process.name()} (PID {current_process.pid})")
        print(f"   Parent: {parent_process.name()} (PID {parent_process.pid})")
        print(f"   Can enumerate: All system processes")
    except ImportError:
        result = "PARTIAL: psutil not installed (basic enumeration only)"
        print(f"✅ psutil not available")
        print(f"   Basic info: PID={os.getpid()}, PPID={os.getppid()}")
        print(f"   Limited enumeration capability")
except Exception as e:
    result = f"Error: {e}"
'''

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: Process Tree Enumeration")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "psutil" in str(result):
            print("\n⚠️  VULNERABILITY DETECTED")
            print("   Attack Vector: psutil module available")
            print("   Impact: Can enumerate all system processes")
            print("   Mitigation: Remove psutil or use PID namespaces")

    # =========================================================================
    # Test 8: Orphan Process Creation
    # =========================================================================

    def test_orphan_process_creation_EXPECT_SUCCESS(self, service):
        """
        Test: Create orphan processes that outlive subprocess
        Expected: SUCCESS (vulnerability - double fork possible)
        Impact: MEDIUM - Processes survive after execution

        Attack: Double fork to create daemon
        """
        code = '''
import os
import sys

try:
    # Test if double fork is possible (don't actually do it)
    if hasattr(os, 'fork'):
        result = "VULNERABILITY: Can create orphan processes via double fork"
        print(f"⚠️ os.fork() available - orphan creation possible")
        print(f"   Attack: Double fork to create daemon")
        print(f"   Impact: Process survives after subprocess exits")
        print(f"   Real attack: Background process exfiltrates data")
    else:
        result = "PROTECTED: fork not available (macOS spawn context)"
        print(f"✅ fork not available on this platform")
except Exception as e:
    result = f"Error: {e}"
'''

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: Orphan Process Creation")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "VULNERABILITY" in str(result):
            print("\n⚠️  VULNERABILITY DETECTED")
            print("   Attack Vector: Double fork technique")
            print("   Impact: Daemon processes survive execution")
            print("   Real Attack: Fork, exit parent, continue in child")
            print("   Mitigation: PID namespaces or cgroups")

    # =========================================================================
    # Test 9: Process Priority Manipulation
    # =========================================================================

    def test_process_priority_manipulation_EXPECT_SUCCESS(self, service):
        """
        Test: Change process priority (nice value)
        Expected: SUCCESS (vulnerability - os.nice available)
        Impact: LOW - Can affect scheduling

        Attack: Set high priority to hog CPU
        """
        code = '''
import os

try:
    current_nice = os.nice(0)  # Get current nice value

    # Test if we can change priority
    result = f"VULNERABILITY: Can change process priority (nice={current_nice})"
    print(f"⚠️ os.nice() available")
    print(f"   Current nice value: {current_nice}")
    print(f"   Can set: os.nice(-20) for highest priority")
    print(f"   Impact: CPU scheduling manipulation")
    print(f"   Note: Requires root for negative values")
except Exception as e:
    result = f"Error: {e}"
'''

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: Process Priority Manipulation")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "VULNERABILITY" in str(result):
            print("\n⚠️  LOW-IMPACT VULNERABILITY")
            print("   Attack Vector: os.nice() available")
            print("   Impact: Can change scheduling priority")
            print("   Limitation: Only affects subprocess")
            print("   Real Attack: os.nice(-20) to max out priority")

    # =========================================================================
    # Test 10: Environment Variable Manipulation
    # =========================================================================

    def test_environment_manipulation_EXPECT_SUCCESS(self, service):
        """
        Test: Read and modify environment variables
        Expected: SUCCESS (vulnerability - os.environ accessible)
        Impact: MEDIUM - Information disclosure

        Attack: Read sensitive environment variables (API keys, etc.)
        """
        code = '''
import os

try:
    # Read environment variables
    env_vars = list(os.environ.keys())
    sensitive_keys = [k for k in env_vars if any(
        keyword in k.upper()
        for keyword in ['KEY', 'SECRET', 'TOKEN', 'PASSWORD', 'AWS', 'ANTHROPIC']
    )]

    result = f"VULNERABILITY: {len(sensitive_keys)} sensitive env vars accessible"
    print(f"⚠️ os.environ accessible")
    print(f"   Total environment variables: {len(env_vars)}")
    print(f"   Sensitive keys found: {len(sensitive_keys)}")

    if sensitive_keys:
        print(f"   Examples: {sensitive_keys[:5]}")
        print(f"   Impact: Can exfiltrate API keys, secrets")
    else:
        print(f"   No sensitive keys in current environment")
except Exception as e:
    result = f"Error: {e}"
'''

        result, stats, ir = service.execute(code, persist=False, timeout=5)

        print(f"\n{'='*70}")
        print("TEST: Environment Variable Access")
        print(f"{'='*70}")
        print(f"Result: {result}")
        print(f"Stdout: {stats['stdout_preview']}")

        if "VULNERABILITY" in str(result):
            print("\n⚠️  VULNERABILITY DETECTED")
            print("   Attack Vector: os.environ accessible")
            print("   Impact: Can read API keys, secrets from environment")
            print("   Real Attack: Exfiltrate AWS_SECRET_ACCESS_KEY")
            print("   Mitigation: Use --env-file with Docker, scrub environment")


if __name__ == "__main__":
    print("="*70)
    print("SIGNAL MANIPULATION SECURITY TESTS")
    print("="*70)
    print("Run with: pytest test_signal_manipulation.py -v -s")
    print("")
    print("These tests check if user code can:")
    print("  1. Discover parent process PID")
    print("  2. Send signals to kill/freeze parent")
    print("  3. Hijack signal handlers")
    print("  4. Manipulate process properties")
    print("")
    print("⚠️  EXPECTED VULNERABILITIES:")
    print("  - Parent PID discovery (os.getppid)")
    print("  - Signal attacks possible (SIGTERM, SIGKILL, SIGSTOP)")
    print("  - Environment variable access")
    print("  - Process priority manipulation")
    print("")
    print("⚠️  SAFETY: Tests only check capability, no signals actually sent")
    print("="*70)
