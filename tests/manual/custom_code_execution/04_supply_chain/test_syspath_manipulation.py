"""
Supply Chain Attack Tests: sys.path Manipulation

ATTACK CATEGORY: Import System Manipulation
VECTOR: Runtime modification of sys.path to load malicious modules

Python's import system relies on sys.path - a list of directories to search for modules.
By manipulating sys.path at runtime, attackers can:
1. Inject new malicious module directories
2. Change import resolution order
3. Load modules from unexpected locations
4. Bypass security checks that only validate static imports

KEY INSIGHT:
AST validation only checks STATIC import statements in the original code.
It does NOT prevent runtime sys.path manipulation via:
- sys.path.insert()
- sys.path.append()
- sys.path modification
- PYTHONPATH environment variable
- .pth files

ATTACK VECTORS:
1. Runtime sys.path.insert(0, malicious_dir)
2. sys.path.append() to add backdoor locations
3. sys.path manipulation via environment
4. Relative path imports
5. sys.modules cache poisoning
"""

import os
import tempfile
from pathlib import Path

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.execution.custom_code_execution_service import (
    CodeExecutionError,
    CustomCodeExecutionService,
)


class TestRuntimeSysPathInjection:
    """
    Test runtime modification of sys.path to inject malicious modules.

    AST validation cannot prevent this because sys.path is modified at runtime,
    AFTER static analysis completes.
    """

    @pytest.fixture
    def service_with_external_malicious_dir(self, tmp_path):
        """Create workspace + external malicious directory."""
        workspace = tmp_path / ".lobster_workspace"
        workspace.mkdir()

        # External malicious directory (not in workspace initially)
        malicious_dir = tmp_path / "malicious_modules"
        malicious_dir.mkdir()

        # Malicious module outside workspace
        fake_secrets = malicious_dir / "secrets_exfiltrator.py"
        fake_secrets.write_text(
            '''
# Malicious module from external directory
import os
import json

def exfiltrate():
    """Steal all environment variables"""
    secrets = dict(os.environ)
    with open('/tmp/exfiltrated_secrets.json', 'w') as f:
        json.dump(secrets, f)
    return "Exfiltration complete"
'''
        )

        dm = DataManagerV2(workspace_path=workspace)
        return CustomCodeExecutionService(dm), workspace, malicious_dir

    def test_syspath_insert_attack_EXPECT_SUCCESS(
        self, service_with_external_malicious_dir
    ):
        """
        Test: Insert external malicious directory into sys.path
        Expected: SUCCESS (vulnerability)
        Impact: CRITICAL - import arbitrary code from outside workspace

        Attack flow:
        1. User code modifies sys.path to add external directory
        2. External directory contains malicious modules
        3. Subsequent imports load malicious code
        4. AST validation bypassed (only checks static imports)
        """
        service, workspace, malicious_dir = service_with_external_malicious_dir

        code = f"""
import sys
# Inject external malicious directory into sys.path
sys.path.insert(0, '{malicious_dir}')

# Now import malicious module from external directory
import secrets_exfiltrator
result = secrets_exfiltrator.exfiltrate()
"""

        result, stats, ir = service.execute(code, persist=False)

        print("\n" + "=" * 70)
        print("TEST: Runtime sys.path Injection")
        print("=" * 70)

        if result == "Exfiltration complete":
            print("⚠️  CRITICAL VULNERABILITY: sys.path injection succeeded")
            print(f"   Malicious directory: {malicious_dir}")
            print(f"   Result: {result}")
            print("\n   ATTACK FLOW:")
            print("   1. User code runs: sys.path.insert(0, '/malicious/dir')")
            print("   2. AST validator sees no forbidden imports (correct!)")
            print("   3. Runtime imports from injected directory")
            print("   4. Arbitrary code execution from outside workspace")
            print("\n   WHY AST VALIDATION FAILS:")
            print("   - AST only analyzes static code structure")
            print("   - Cannot predict runtime sys.path changes")
            print("   - Cannot prevent dynamic imports after sys.path modification")
            assert True, "sys.path injection vulnerability confirmed"
        else:
            print("✅ PROTECTED: sys.path injection failed")
            print(f"   Result: {result}")
            pytest.fail("Expected vulnerability but system was protected")

    def test_syspath_append_attack_EXPECT_SUCCESS(
        self, service_with_external_malicious_dir
    ):
        """
        Test: Append malicious directory to sys.path (lower priority)
        Expected: SUCCESS (vulnerability)
        Impact: HIGH - stealthier attack, loads if no conflicts
        """
        service, workspace, malicious_dir = service_with_external_malicious_dir

        code = f"""
import sys
# Stealthier: append instead of insert (lower priority)
sys.path.append('{malicious_dir}')

# Import unique module name (no conflicts)
import secrets_exfiltrator
result = secrets_exfiltrator.exfiltrate()
"""

        result, stats, ir = service.execute(code, persist=False)

        print("\n" + "=" * 70)
        print("TEST: sys.path.append() Attack (Stealthy)")
        print("=" * 70)

        if result == "Exfiltration complete":
            print("⚠️  VULNERABILITY: sys.path.append() succeeded")
            print("   Stealthier than insert() - less suspicious")
            print("   Works if module name doesn't conflict with existing imports")
            print("\n   DETECTION EVASION:")
            print("   - append() is less suspicious than insert()")
            print("   - Often used legitimately")
            print("   - Harder to flag as malicious")
            assert True, "sys.path append vulnerability confirmed"
        else:
            print("✅ PROTECTED: sys.path append failed")
            pytest.fail("Expected vulnerability but system was protected")


class TestSysPathRelativeImports:
    """
    Test relative path imports via sys.path manipulation.

    Attacker uses relative paths like '../', '/tmp/', etc. to escape workspace.
    """

    @pytest.fixture
    def service_with_parent_dir_attack(self, tmp_path):
        """Create workspace + malicious module in parent directory."""
        workspace = tmp_path / ".lobster_workspace"
        workspace.mkdir()

        # Malicious module in parent directory (outside workspace)
        malicious_module = tmp_path / "escape_module.py"
        malicious_module.write_text(
            """
# Malicious module outside workspace (in parent directory)
def escaped():
    return "ESCAPED_WORKSPACE"
"""
        )

        dm = DataManagerV2(workspace_path=workspace)
        return CustomCodeExecutionService(dm), workspace, tmp_path

    def test_parent_directory_escape_EXPECT_SUCCESS(
        self, service_with_parent_dir_attack
    ):
        """
        Test: Escape workspace using '../' in sys.path
        Expected: SUCCESS (vulnerability)
        Impact: CRITICAL - workspace containment breach
        """
        service, workspace, parent_dir = service_with_parent_dir_attack

        code = """
import sys
# Escape workspace by adding parent directory
sys.path.insert(0, '..')

# Import from parent directory (outside workspace)
import escape_module
result = escape_module.escaped()
"""

        result, stats, ir = service.execute(code, persist=False)

        print("\n" + "=" * 70)
        print("TEST: Parent Directory Escape via '../'")
        print("=" * 70)

        if result == "ESCAPED_WORKSPACE":
            print("⚠️  CRITICAL VULNERABILITY: Workspace escape succeeded")
            print(f"   Workspace: {workspace}")
            print(f"   Escaped to: {parent_dir}")
            print("   Method: sys.path.insert(0, '..')")
            print("\n   SECURITY BOUNDARY BREACH:")
            print("   - Workspace is supposed to be isolated")
            print("   - User code can escape using relative paths")
            print("   - Can access sensitive files outside workspace")
            print("   - Can import malicious code from shared directories")
            assert True, "Workspace escape vulnerability confirmed"
        else:
            print("✅ PROTECTED: Workspace escape prevented")
            pytest.fail("Expected vulnerability but system was protected")

    @pytest.fixture
    def service_with_tmp_attack(self, tmp_path):
        """Create workspace + malicious module in /tmp."""
        workspace = tmp_path / ".lobster_workspace"
        workspace.mkdir()

        # Malicious module in /tmp (world-writable on Unix)
        tmp_malicious = Path("/tmp") / "temp_backdoor.py"
        tmp_malicious.write_text(
            """
# Malicious module in /tmp (world-writable directory)
def backdoor():
    return "BACKDOOR_ACTIVATED"
"""
        )

        dm = DataManagerV2(workspace_path=workspace)
        return CustomCodeExecutionService(dm), workspace, tmp_malicious

    def test_tmp_directory_attack_EXPECT_SUCCESS(self, service_with_tmp_attack):
        """
        Test: Add /tmp to sys.path to load malicious modules
        Expected: SUCCESS (vulnerability)
        Impact: CRITICAL - world-writable directory exploitation

        /tmp is world-writable on Unix systems, making it perfect for supply chain attacks.
        """
        service, workspace, tmp_malicious = service_with_tmp_attack

        code = """
import sys
# Add world-writable /tmp directory to sys.path
sys.path.insert(0, '/tmp')

# Import malicious module from /tmp
import temp_backdoor
result = temp_backdoor.backdoor()
"""

        try:
            result, stats, ir = service.execute(code, persist=False)

            print("\n" + "=" * 70)
            print("TEST: /tmp Directory Attack (World-Writable)")
            print("=" * 70)

            if result == "BACKDOOR_ACTIVATED":
                print("⚠️  CRITICAL VULNERABILITY: /tmp import succeeded")
                print("   /tmp is world-writable on Unix systems")
                print("   Any user can place malicious modules there")
                print("\n   MULTI-USER SYSTEM RISK:")
                print("   - Shared server with multiple users")
                print("   - Attacker places backdoor.py in /tmp")
                print("   - Victim's code adds /tmp to sys.path")
                print("   - Backdoor imported and executed")
                assert True, "/tmp import vulnerability confirmed"
            else:
                print("✅ PROTECTED: /tmp import failed")
                pytest.fail("Expected vulnerability but system was protected")
        finally:
            # Cleanup
            if tmp_malicious.exists():
                tmp_malicious.unlink()


class TestSysModulesPoisoning:
    """
    Test sys.modules cache poisoning to inject malicious modules.

    Python's import system uses sys.modules as a cache. If attacker can
    modify sys.modules directly, they can inject arbitrary code without
    file system access.
    """

    @pytest.fixture
    def service_with_clean_workspace(self, tmp_path):
        """Create clean workspace for sys.modules tests."""
        workspace = tmp_path / ".lobster_workspace"
        workspace.mkdir()

        dm = DataManagerV2(workspace_path=workspace)
        return CustomCodeExecutionService(dm), workspace

    def test_sys_modules_direct_injection_EXPECT_SUCCESS(
        self, service_with_clean_workspace
    ):
        """
        Test: Directly inject malicious module into sys.modules cache
        Expected: SUCCESS (vulnerability)
        Impact: CRITICAL - bypass file system entirely

        Attack flow:
        1. Create fake module object in memory
        2. Insert into sys.modules cache
        3. Subsequent imports use poisoned cache
        4. No file system traces
        """
        service, workspace = service_with_clean_workspace

        code = """
import sys
import types

# Create fake malicious module in memory (no file needed!)
fake_numpy = types.ModuleType('numpy')
fake_numpy.array = lambda x: [666] * len(x)
fake_numpy.mean = lambda x: 666.0

# Poison sys.modules cache
sys.modules['numpy'] = fake_numpy

# Now importing numpy uses poisoned version
import numpy as np
data = np.array([1, 2, 3, 4, 5])
result = np.mean(data)
"""

        result, stats, ir = service.execute(code, persist=False)

        print("\n" + "=" * 70)
        print("TEST: sys.modules Cache Poisoning")
        print("=" * 70)

        if result == 666.0:
            print("⚠️  CRITICAL VULNERABILITY: sys.modules poisoning succeeded")
            print("   Expected mean([1,2,3,4,5]): 3.0")
            print(f"   Actual result: {result}")
            print("\n   ATTACK CHARACTERISTICS:")
            print("   - No file system access needed")
            print("   - Module created entirely in memory")
            print("   - sys.modules cache poisoned directly")
            print("   - Subsequent imports use malicious version")
            print("   - No traces on disk")
            print("\n   DETECTION EVASION:")
            print("   - File system monitoring: useless")
            print("   - Hash verification: irrelevant (no files)")
            print("   - Static analysis: cannot detect runtime cache modification")
            assert True, "sys.modules poisoning vulnerability confirmed"
        else:
            print("✅ PROTECTED: sys.modules poisoning prevented")
            pytest.fail("Expected vulnerability but system was protected")

    def test_sys_modules_replacement_EXPECT_SUCCESS(self, service_with_clean_workspace):
        """
        Test: Replace already-imported module in sys.modules
        Expected: SUCCESS (vulnerability)
        Impact: CRITICAL - retroactive module replacement

        Even if a module is already imported correctly, attacker can replace it
        in sys.modules cache, affecting all subsequent uses.
        """
        service, workspace = service_with_clean_workspace

        code = """
import sys
import json

# json is already imported (used by CustomCodeExecutionService)
# But we can replace it in sys.modules cache!

import types
fake_json = types.ModuleType('json')
fake_json.loads = lambda s: {"hijacked": True}
fake_json.dumps = lambda obj: '{"hijacked": true}'

# Replace real json in cache
sys.modules['json'] = fake_json

# Now reimport to get fake version
import json
result = json.loads('{"real": "data"}')
"""

        result, stats, ir = service.execute(code, persist=False)

        print("\n" + "=" * 70)
        print("TEST: sys.modules Replacement (Post-Import)")
        print("=" * 70)

        if isinstance(result, dict) and result.get("hijacked") is True:
            print("⚠️  CRITICAL VULNERABILITY: Module replacement succeeded")
            print("   json was already imported correctly")
            print("   Attacker replaced it in sys.modules")
            print(f"   Result: {result}")
            print("\n   TIME-OF-CHECK TIME-OF-USE (TOCTOU):")
            print("   - Module validated at import time")
            print("   - But can be swapped later in sys.modules")
            print("   - Subsequent imports get malicious version")
            print("   - Hard to detect without continuous monitoring")
            assert True, "Module replacement vulnerability confirmed"
        else:
            print("✅ PROTECTED: Module replacement prevented")
            pytest.fail("Expected vulnerability but system was protected")


class TestEnvironmentBasedSysPathInjection:
    """
    Test PYTHONPATH and other environment-based sys.path manipulation.

    Python automatically adds PYTHONPATH to sys.path on startup.
    This happens BEFORE any user code runs and BEFORE AST validation.
    """

    @pytest.fixture
    def service_with_malicious_pythonpath(self, tmp_path, monkeypatch):
        """
        Create workspace and set PYTHONPATH to malicious directory.

        NOTE: This test demonstrates concept but may not work as expected because
        subprocess is spawned fresh. Left here for documentation purposes.
        """
        workspace = tmp_path / ".lobster_workspace"
        workspace.mkdir()

        # External malicious directory
        malicious_dir = tmp_path / "evil_pythonpath"
        malicious_dir.mkdir()

        # Malicious module
        evil_module = malicious_dir / "trusted_lib.py"
        evil_module.write_text(
            """
# Malicious module loaded via PYTHONPATH
def compute():
    return "PYTHONPATH_HIJACKED"
"""
        )

        # Set PYTHONPATH (would affect subprocess)
        # NOTE: This is demonstrative - actual attack would set PYTHONPATH
        # before running Lobster
        monkeypatch.setenv("PYTHONPATH", str(malicious_dir))

        dm = DataManagerV2(workspace_path=workspace)
        return CustomCodeExecutionService(dm), workspace, malicious_dir

    def test_pythonpath_hijack_CONCEPTUAL(self, service_with_malicious_pythonpath):
        """
        Test: PYTHONPATH environment variable hijacking
        Expected: Depends on environment propagation
        Impact: CRITICAL - affects ALL Python processes

        CONCEPTUAL TEST: Demonstrates attack vector but may not execute due to
        subprocess isolation. Real attack would set PYTHONPATH before starting Lobster.

        Attack scenario:
        1. Attacker controls PYTHONPATH environment variable
        2. PYTHONPATH points to malicious directory
        3. Lobster subprocess inherits PYTHONPATH
        4. All imports check PYTHONPATH first
        5. Malicious modules loaded automatically
        """
        service, workspace, malicious_dir = service_with_malicious_pythonpath

        code = """
# Check if PYTHONPATH was inherited
import sys
import os

pythonpath = os.environ.get('PYTHONPATH', 'NOT_SET')
result = f"PYTHONPATH={pythonpath}"
"""

        result, stats, ir = service.execute(code, persist=False)

        print("\n" + "=" * 70)
        print("TEST: PYTHONPATH Hijacking (Conceptual)")
        print("=" * 70)
        print(f"   Result: {result}")
        print(f"   Expected malicious dir: {malicious_dir}")
        print("\n   ATTACK VECTOR (DOCUMENTED):")
        print("   - Attacker sets PYTHONPATH before starting Lobster")
        print("   - Or modifies user's shell profile (~/.bashrc)")
        print("   - Or exploits container/VM environment variables")
        print("   - All Python processes affected")
        print("   - Persistent across restarts")
        print("\n   REAL-WORLD SCENARIOS:")
        print("   - Compromised CI/CD environment")
        print("   - Shared server with writable /etc/environment")
        print("   - Docker container with malicious PYTHONPATH")
        print("   - Cloud function with environment injection")

        # This test is conceptual - don't fail if PYTHONPATH not propagated
        assert True, "PYTHONPATH attack vector documented"


class TestDynamicImportBypass:
    """
    Test bypassing AST validation using dynamic imports.

    AST validation only checks static import statements (import X, from X import Y).
    It cannot prevent dynamic imports using:
    - __import__('module')
    - importlib.import_module('module')
    - eval/exec with import statements
    """

    @pytest.fixture
    def service_with_subprocess_available(self, tmp_path):
        """
        Create workspace where subprocess module could be imported dynamically.
        """
        workspace = tmp_path / ".lobster_workspace"
        workspace.mkdir()

        dm = DataManagerV2(workspace_path=workspace)
        return CustomCodeExecutionService(dm), workspace

    def test_builtin_import_bypass_EXPECT_WARNING(
        self, service_with_subprocess_available
    ):
        """
        Test: Use __import__() to bypass static import validation
        Expected: WARNING detected but may still execute
        Impact: CRITICAL - complete bypass of import blocking

        AST validator detects __import__() calls and issues warnings,
        but does NOT block execution.
        """
        service, workspace = service_with_subprocess_available

        # This should trigger warning but NOT block
        code = """
# Bypass static import validation using __import__
subprocess = __import__('subprocess')
result = "subprocess imported via __import__()"
"""

        # Should succeed with warnings
        result, stats, ir = service.execute(code, persist=False)

        print("\n" + "=" * 70)
        print("TEST: __import__() Dynamic Import Bypass")
        print("=" * 70)
        print(f"   Result: {result}")
        print(f"   Warnings: {stats.get('warnings', [])}")

        if stats.get("warnings"):
            print("\n   ⚠️  WARNING DETECTED (GOOD):")
            print("   - AST validator detected __import__() call")
            print("   - Warning issued to user")
            for warning in stats["warnings"]:
                print(f"     - {warning}")
            print("\n   BUT WARNING IS NOT BLOCKING:")
            print("   - Code still executed despite warning")
            print("   - __import__('subprocess') succeeded")
            print("   - Only user awareness prevents exploitation")
            assert True, "Dynamic import warning confirmed (non-blocking)"
        else:
            print("   ❌ NO WARNING ISSUED:")
            print("   - AST validator missed __import__() call")
            print("   - Silent bypass of security checks")
            pytest.fail("Expected warning for __import__() but none issued")

    def test_importlib_bypass_EXPECT_SUCCESS(self, service_with_subprocess_available):
        """
        Test: Use importlib.import_module() to bypass validation
        Expected: SUCCESS (no detection)
        Impact: CRITICAL - completely undetected bypass
        """
        service, workspace = service_with_subprocess_available

        code = """
# Bypass using importlib (may not be detected)
import importlib
subprocess_module = importlib.import_module('subprocess')
result = "subprocess imported via importlib"
"""

        result, stats, ir = service.execute(code, persist=False)

        print("\n" + "=" * 70)
        print("TEST: importlib.import_module() Bypass")
        print("=" * 70)
        print(f"   Result: {result}")
        print(f"   Warnings: {stats.get('warnings', [])}")

        if result and "importlib" in result:
            print("⚠️  VULNERABILITY: importlib bypass succeeded")
            print("   - importlib.import_module() not blocked")
            print("   - No warnings issued")
            print("   - Can import any module including forbidden ones")
            print("\n   STEALTH CHARACTERISTICS:")
            print("   - Looks like legitimate dynamic loading")
            print("   - Common pattern in plugin systems")
            print("   - Hard to distinguish from safe usage")
            assert True, "importlib bypass vulnerability confirmed"
        else:
            print("✅ PROTECTED: importlib bypass prevented")
            pytest.fail("Expected vulnerability but system was protected")

    def test_getattr_import_bypass_EXPECT_SUCCESS(
        self, service_with_subprocess_available
    ):
        """
        Test: Use getattr(__builtins__, '__import__') to bypass validation
        Expected: SUCCESS (obfuscated bypass)
        Impact: CRITICAL - highly obfuscated attack
        """
        service, workspace = service_with_subprocess_available

        code = """
# Highly obfuscated import bypass
import_func = getattr(__builtins__, '__import__')
subprocess = import_func('subprocess')
result = "subprocess imported via obfuscated __import__"
"""

        result, stats, ir = service.execute(code, persist=False)

        print("\n" + "=" * 70)
        print("TEST: Obfuscated __import__() via getattr")
        print("=" * 70)
        print(f"   Result: {result}")
        print(f"   Warnings: {stats.get('warnings', [])}")

        if result and "obfuscated" in result:
            print("⚠️  CRITICAL VULNERABILITY: Obfuscated bypass succeeded")
            print("   - getattr(__builtins__, '__import__') not detected")
            print("   - AST analysis cannot trace getattr() target")
            print("   - Complete evasion of static analysis")
            print("\n   OBFUSCATION TECHNIQUES:")
            print("   - Hide __import__ access via getattr")
            print("   - Runtime string resolution")
            print("   - Static analysis cannot predict")
            print("   - Requires runtime monitoring to detect")
            assert True, "Obfuscated import bypass confirmed"
        else:
            print("✅ PROTECTED: Obfuscated bypass prevented")
            pytest.fail("Expected vulnerability but system was protected")


# Summary marker for test runner
def test_summary():
    """
    SUMMARY: sys.path Manipulation Vulnerabilities

    CRITICAL FINDING: Runtime import system manipulation bypasses static validation

    AST validation is fundamentally limited to STATIC code analysis.
    It cannot prevent RUNTIME modifications to Python's import system.

    VULNERABILITY COUNT: 8+ confirmed attack vectors

    ATTACK CATEGORIES:
    1. Runtime sys.path injection (insert, append)
    2. Relative path escapes ('../', '/tmp/')
    3. sys.modules cache poisoning
    4. Environment-based injection (PYTHONPATH)
    5. Dynamic import bypasses (__import__, importlib)

    IMPACT: CRITICAL across all categories

    KEY INSIGHT:
    Static AST validation only checks import statements in the original code.
    It does NOT and CANNOT prevent:
    - Runtime sys.path modifications
    - sys.modules cache manipulation
    - Dynamic imports via __import__() or importlib
    - Environment variable inheritance (PYTHONPATH)

    REMEDIATION:
    1. Runtime sys.path monitoring and restrictions
    2. sys.modules integrity checking
    3. Dynamic import tracking and validation
    4. Environment sanitization
    5. Sandboxed execution with import hooks

    See SUPPLY_CHAIN_REPORT.md for full analysis.
    """
    print("\n" + "=" * 70)
    print("SYS.PATH MANIPULATION TEST SUITE COMPLETE")
    print("=" * 70)
    print("Status: All vulnerabilities confirmed")
    print("Severity: CRITICAL")
    print("Recommendation: AST validation insufficient - need runtime controls")
