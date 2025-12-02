"""
AST Bypass Testing: Dynamic Import Techniques

Tests runtime import methods that bypass AST validation.
AST only validates static Import/ImportFrom nodes, not runtime function calls.

CRITICAL: These tests demonstrate that __import__(), importlib, and eval/exec
can completely bypass the FORBIDDEN_MODULES check.
"""

import pytest
from pathlib import Path
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.execution.custom_code_execution_service import (
    CustomCodeExecutionService,
    CodeExecutionError,
    CodeValidationError
)


@pytest.fixture
def service(tmp_path):
    """Create service with temporary workspace."""
    data_manager = DataManagerV2(workspace_path=tmp_path)
    return CustomCodeExecutionService(data_manager)


class TestDynamicImportBypass:
    """
    Test Category: Runtime Import Bypasses

    Vulnerability: AST validation only checks ast.Import and ast.ImportFrom nodes.
    Runtime import functions (__import__, importlib.import_module, etc.) are not blocked.
    """

    def test_dunder_import_bypass_EXPECT_SUCCESS(self, service):
        """
        Test: Use __import__() to bypass AST import blocking
        Expected: SUCCESS (vulnerability - __import__ not blocked, only warned)
        Impact: CRITICAL - Can import ANY module including subprocess

        Explanation:
        - AST validation checks: isinstance(node, ast.Import)
        - But __import__() is ast.Call node, not ast.Import
        - FORBIDDEN_MODULES = {'subprocess', '__import__'} checks import statements, not builtins
        - Only generates warning, does not block execution
        """
        code = '''
# AST validation only checks 'import subprocess' statements
# But __import__() is a function call (ast.Call), not an import statement (ast.Import)
sp = __import__('subprocess')
result = f"subprocess module imported: {sp.__name__}"
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "="*80)
            print("CRITICAL VULNERABILITY CONFIRMED: __import__() Bypass")
            print("="*80)
            print(f"Result: {result}")
            print(f"Success: {stats['success']}")
            print(f"Warnings: {stats['warnings']}")
            print(f"\nResult contains 'subprocess': {'subprocess' in str(result)}")

            if stats['success'] and 'subprocess' in str(result):
                print("\n🚨 SECURITY BREACH:")
                print("   - Attacker can import subprocess via __import__('subprocess')")
                print("   - Can then execute arbitrary system commands")
                print("   - Example: subprocess.run(['rm', '-rf', '/'])")
                print("\n📋 Mitigation Required:")
                print("   - Remove __import__ from allowed builtins")
                print("   - Or use RestrictedPython's safe_builtins")

        except CodeExecutionError as e:
            print(f"✅ PROTECTED (unexpected): {e}")
            pytest.fail("Expected vulnerability to succeed, but code execution failed")
        except CodeValidationError as e:
            print(f"✅ PROTECTED (good): {e}")
            pytest.fail("Expected warning only, got validation error (fix may have been applied)")

    def test_importlib_bypass_EXPECT_SUCCESS(self, service):
        """
        Test: Use importlib.import_module() to bypass AST blocking
        Expected: SUCCESS (vulnerability)
        Impact: CRITICAL

        Explanation:
        - importlib is in Python standard library (allowed)
        - importlib.import_module() is a runtime function call
        - AST cannot detect what string is passed to it
        """
        code = '''
import importlib
# Import subprocess dynamically
sp = importlib.import_module('subprocess')
result = f"subprocess loaded via importlib: {sp.__name__}"
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "="*80)
            print("CRITICAL VULNERABILITY CONFIRMED: importlib.import_module() Bypass")
            print("="*80)
            print(f"Result: {result}")
            print(f"Success: {stats['success']}")

            if stats['success'] and 'subprocess' in str(result):
                print("\n🚨 SECURITY BREACH:")
                print("   - importlib.import_module() bypasses all import checks")
                print("   - Works for ANY forbidden module")
                print("\n📋 Mitigation:")
                print("   - Block importlib module entirely")
                print("   - Or use RestrictedPython with custom import hooks")

        except (CodeExecutionError, CodeValidationError) as e:
            print(f"✅ PROTECTED: {e}")
            pytest.fail("Expected vulnerability to succeed")

    def test_getattr_builtins_import_EXPECT_SUCCESS(self, service):
        """
        Test: Access __import__ via getattr(__builtins__, '__import__')
        Expected: SUCCESS (vulnerability)
        Impact: CRITICAL

        Explanation:
        - Even if __import__ name is blocked in validation
        - Can access it via getattr() on __builtins__
        - AST sees: Call(func=Name(id='getattr'), ...)
        """
        code = '''
# Access __import__ indirectly via getattr
import_func = getattr(__builtins__, '__import__')
sp = import_func('subprocess')
result = f"subprocess via getattr: {sp.__name__}"
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "="*80)
            print("CRITICAL VULNERABILITY CONFIRMED: getattr(__builtins__) Bypass")
            print("="*80)
            print(f"Result: {result}")

            if stats['success'] and 'subprocess' in str(result):
                print("\n🚨 SECURITY BREACH:")
                print("   - getattr() can access any builtin, including __import__")
                print("   - Cannot be blocked by AST validation")
                print("\n📋 Mitigation:")
                print("   - Use RestrictedPython with safe_builtins")
                print("   - Or execute in restricted namespace without __builtins__")

        except (CodeExecutionError, CodeValidationError) as e:
            print(f"✅ PROTECTED: {e}")
            pytest.fail("Expected vulnerability to succeed")

    def test_exec_import_bypass_EXPECT_SUCCESS(self, service):
        """
        Test: Use exec() to execute import statement as string
        Expected: SUCCESS (exec only generates warning, not blocked)
        Impact: CRITICAL

        Explanation:
        - _validate_code_safety() warns about exec(), but allows it (line 304-308)
        - exec("import subprocess") generates warning but executes
        - AST cannot analyze strings passed to exec()
        """
        code = '''
# exec() only generates warning, does not block execution
exec("import subprocess")
# subprocess is now in globals()
result = f"subprocess in globals: {'subprocess' in globals()}"
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "="*80)
            print("CRITICAL VULNERABILITY CONFIRMED: exec() Bypass")
            print("="*80)
            print(f"Result: {result}")
            print(f"Warnings: {stats['warnings']}")
            print(f"Success: {stats['success']}")

            if stats['success']:
                print("\n🚨 SECURITY BREACH:")
                print("   - exec() is only warned about, not blocked")
                print("   - Can execute ANY Python code as string")
                print("   - Completely bypasses AST validation")
                print("\n📋 Current Code (line 304-308):")
                print("   if node.func.id in ['eval', 'exec', 'compile', '__import__']:")
                print("       warnings.append(...)")
                print("   # ⚠️ Only warns, does not raise exception!")
                print("\n📋 Mitigation Required:")
                print("   - Change warning to CodeValidationError")
                print("   - Or remove exec/eval from allowed builtins")

        except CodeValidationError as e:
            print(f"✅ PROTECTED: {e}")
            pytest.fail("Expected warning only, got validation error")
        except CodeExecutionError as e:
            print(f"⚠️ Execution failed (may be subprocess isolation): {e}")


class TestNestedImportBypass:
    """
    Test Category: Nested and Obfuscated Import Patterns

    These test combinations of techniques that make detection even harder.
    """

    def test_lambda_import_EXPECT_SUCCESS(self, service):
        """
        Test: Use lambda with __import__ to obfuscate
        Expected: SUCCESS (vulnerability)
        Impact: HIGH
        """
        code = '''
# Lambda obfuscates the __import__ call
import_subprocess = (lambda: __import__('subprocess'))()
result = f"subprocess via lambda: {import_subprocess.__name__}"
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "="*80)
            print("VULNERABILITY CONFIRMED: Lambda Import Obfuscation")
            print("="*80)
            print(f"Result: {result}")

            if stats['success'] and 'subprocess' in str(result):
                print("\n🚨 Lambda obfuscation successful")
                print("📋 Mitigation: Block __import__ in builtins")

        except (CodeExecutionError, CodeValidationError) as e:
            print(f"✅ PROTECTED: {e}")

    def test_nested_exec_EXPECT_SUCCESS(self, service):
        """
        Test: Nest exec() calls to evade detection
        Expected: SUCCESS (vulnerability)
        Impact: HIGH
        """
        code = '''
# Double exec() to evade simple pattern matching
exec("exec('import subprocess')")
result = "Double exec() completed"
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "="*80)
            print("VULNERABILITY CONFIRMED: Nested exec() Bypass")
            print("="*80)
            print(f"Result: {result}")
            print(f"Warnings: {stats['warnings']}")

            if stats['success']:
                print("\n🚨 Nested exec() successful")
                print("📋 AST can detect outer exec(), but not inner string content")

        except (CodeExecutionError, CodeValidationError) as e:
            print(f"✅ PROTECTED: {e}")

    def test_eval_import_expression_EXPECT_SUCCESS(self, service):
        """
        Test: Use eval() with import expression
        Expected: SUCCESS with warning (vulnerability)
        Impact: CRITICAL
        """
        code = '''
# eval() can evaluate import expressions
result = eval("__import__('subprocess').__name__")
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "="*80)
            print("VULNERABILITY CONFIRMED: eval() Import Expression")
            print("="*80)
            print(f"Result: {result}")
            print(f"Warnings: {stats['warnings']}")

            if stats['success'] and result == 'subprocess':
                print("\n🚨 eval() successfully imported subprocess")
                print("📋 eval() is only warned, not blocked")

        except CodeValidationError as e:
            print(f"✅ PROTECTED: {e}")
            pytest.fail("Expected warning only")
        except CodeExecutionError as e:
            print(f"⚠️ Execution error: {e}")


class TestRuntimeStringManipulation:
    """
    Test Category: Runtime String Manipulation

    Build forbidden module names at runtime, completely invisible to AST.
    """

    def test_string_concatenation_import_EXPECT_SUCCESS(self, service):
        """
        Test: Build module name via string concatenation
        Expected: SUCCESS (vulnerability)
        Impact: CRITICAL

        Explanation:
        - AST cannot evaluate runtime string operations
        - module_name = 'sub' + 'process' is invisible to static analysis
        """
        code = '''
# Build module name at runtime
module_name = 'sub' + 'process'
sp = __import__(module_name)
result = f"Imported {module_name}: {sp.__name__}"
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "="*80)
            print("VULNERABILITY CONFIRMED: String Concatenation Bypass")
            print("="*80)
            print(f"Result: {result}")

            if stats['success'] and 'subprocess' in str(result):
                print("\n🚨 String concatenation completely bypasses AST")
                print("📋 AST sees: BinOp(op=Add, left=Constant('sub'), right=Constant('process'))")
                print("    But cannot evaluate to 'subprocess' at validation time")
                print("\n📋 Mitigation:")
                print("   - Block __import__ builtin entirely")
                print("   - Use RestrictedPython")

        except (CodeExecutionError, CodeValidationError) as e:
            print(f"✅ PROTECTED: {e}")

    def test_chr_join_import_EXPECT_SUCCESS(self, service):
        """
        Test: Build module name using chr() and join()
        Expected: SUCCESS (vulnerability)
        Impact: HIGH
        """
        code = '''
# Build 'subprocess' from character codes
chars = [115, 117, 98, 112, 114, 111, 99, 101, 115, 115]  # 'subprocess'
module_name = ''.join(chr(c) for c in chars)
sp = __import__(module_name)
result = f"Imported via chr(): {sp.__name__}"
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "="*80)
            print("VULNERABILITY CONFIRMED: chr() Join Bypass")
            print("="*80)
            print(f"Result: {result}")

            if stats['success'] and 'subprocess' in str(result):
                print("\n🚨 Character code obfuscation successful")
                print("📋 Completely invisible to AST validation")

        except (CodeExecutionError, CodeValidationError) as e:
            print(f"✅ PROTECTED: {e}")

    def test_format_string_import_EXPECT_SUCCESS(self, service):
        """
        Test: Use f-string or format() to build module name
        Expected: SUCCESS (vulnerability)
        Impact: HIGH
        """
        code = '''
# Use format string
prefix = 'sub'
suffix = 'process'
module_name = f'{prefix}{suffix}'
sp = __import__(module_name)
result = f"Imported {module_name}"
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "="*80)
            print("VULNERABILITY CONFIRMED: Format String Bypass")
            print("="*80)
            print(f"Result: {result}")

            if stats['success'] and 'subprocess' in str(result):
                print("\n🚨 Format string bypass successful")

        except (CodeExecutionError, CodeValidationError) as e:
            print(f"✅ PROTECTED: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
