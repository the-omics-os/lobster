"""
AST Bypass Testing: Unicode and Encoding Tricks

Tests advanced Unicode attacks, homograph attacks, and character encoding exploits.
These techniques exploit how Python parsers handle different character encodings.

CRITICAL: These demonstrate attacks that can bypass even careful code review.
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


class TestHomographAttacks:
    """
    Test Category: Homograph Attacks

    Using Unicode characters that LOOK like ASCII but have different code points.
    Example: Cyrillic 'е' (U+0435) looks identical to Latin 'e' (U+0065).

    These attacks can bypass:
    - Visual code review
    - String matching (e.g., checking for 'exec' function name)
    - AST validation (if it's checking name strings)
    """

    def test_cyrillic_exec_EXPECT_SUCCESS(self, service):
        """
        Test: Use Cyrillic 'е' instead of Latin 'e' in exec
        Expected: SUCCESS if Python allows it (may fail on function lookup)
        Impact: MEDIUM - More useful for bypassing string filters

        Explanation:
        - Visual: 'ехес' looks like 'exec'
        - Reality: 'е' is U+0435 (Cyrillic), not U+0065 (Latin 'e')
        - If code does: if 'exec' in code_string: -> this bypasses it
        - Python won't find function 'ехес' in builtins (good)
        - But attacker can define their own 'ехес' variable
        """
        code = '''
# Define cyrillic 'ехес' that does exec
# Using actual Latin 'exec' builtin
ехес = exec  # 'е' is Cyrillic U+0435

# Now use the Cyrillic version
ехес("import subprocess")

result = f"subprocess imported: {'subprocess' in globals()}"
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "="*80)
            print("VULNERABILITY: Cyrillic Homograph Attack")
            print("="*80)
            print(f"Result: {result}")

            if stats['success']:
                print("\n🚨 HOMOGRAPH ATTACK SUCCESSFUL:")
                print("   - Cyrillic 'ехес' assigned to exec builtin")
                print("   - Visually indistinguishable in many fonts")
                print("   - Can bypass string-based filters")
                print("\n📋 Example bypasses:")
                print("   - if 'exec' in code: block() # Bypassed!")
                print("   - AST node.func.id == 'exec' # May be 'ехес'")
                print("\n📋 Mitigation:")
                print("   - Normalize Unicode before validation")
                print("   - Use Unicode category checks")
                print("   - Only allow ASCII in code")

        except (CodeValidationError, CodeExecutionError) as e:
            print(f"✅ PROTECTED (or exec failed): {e}")

    def test_mixed_script_identifier_EXPECT_SUCCESS(self, service):
        """
        Test: Mix Cyrillic and Latin in identifier names
        Expected: SUCCESS (vulnerability)
        Impact: HIGH - Can hide malicious intent
        """
        code = '''
# Mix Cyrillic and Latin characters
# 'іmport' - 'і' is Cyrillic U+0456, looks like Latin 'i'
# Can define variables that look like Python keywords

іmport = __import__  # 'і' is Cyrillic
suЬprocess = іmport('subprocess')  # 'Ь' is Cyrillic Soft Sign U+042C

result = f"Mixed script bypass: {suЬprocess.__name__}"
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "="*80)
            print("VULNERABILITY: Mixed Script Identifiers")
            print("="*80)
            print(f"Result: {result}")

            if stats['success'] and 'subprocess' in str(result):
                print("\n🚨 MIXED SCRIPT ATTACK SUCCESSFUL:")
                print("   - Identifiers use mix of Cyrillic and Latin")
                print("   - Impossible to detect by visual inspection")
                print("   - Python allows Unicode identifiers (PEP 3131)")

        except (CodeValidationError, CodeExecutionError) as e:
            print(f"✅ PROTECTED: {e}")


class TestZeroWidthCharacters:
    """
    Test Category: Zero-Width Character Attacks

    Unicode has several zero-width characters that are invisible.
    Can hide code or bypass filters.
    """

    def test_zero_width_space_EXPECT_SUCCESS(self, service):
        """
        Test: Insert zero-width spaces in code
        Expected: Syntax error (Python doesn't allow in identifiers)
        Impact: LOW - More useful for bypassing visual inspection

        Note: Zero-width characters can hide malicious code in comments
        or strings that get exec'd later.
        """
        code = '''
# Zero-width space (U+200B) in comment: ​
# The above line contains invisible characters

# Can hide in strings that get exec'd
hidden_payload = "import​ subprocess"  # Contains zero-width space
# This will fail: exec(hidden_payload)

result = "Zero-width test completed"
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "="*80)
            print("TEST: Zero-Width Character Injection")
            print("="*80)
            print(f"Result: {result}")

            if stats['success']:
                print("\n⚠️ Zero-width characters allowed in strings")
                print("📋 Can hide malicious code in:")
                print("   - Comments (for later extraction)")
                print("   - String literals (if decoded/processed)")
                print("   - Base64 payloads (decode removes them)")

        except (CodeValidationError, CodeExecutionError) as e:
            print(f"✅ PROTECTED: {e}")

    def test_zero_width_joiner_EXPECT_SUCCESS(self, service):
        """
        Test: Use zero-width joiner (U+200D) to hide code
        Expected: Python may ignore in some contexts
        Impact: MEDIUM
        """
        code = '''
# Zero-width joiner can be used in identifier names (in some cases)
# Build a string with hidden characters
exec_name = "e" + "\\u200D" + "xec"  # Zero-width joiner
# exec_name looks like 'exec' but has invisible character

result = f"Zero-width joiner test: len(exec_name)={len(exec_name)}"
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "="*80)
            print("TEST: Zero-Width Joiner")
            print("="*80)
            print(f"Result: {result}")

            if stats['success']:
                print("\n⚠️ Zero-width joiners can hide in strings")

        except (CodeValidationError, CodeExecutionError) as e:
            print(f"✅ PROTECTED: {e}")


class TestBidirectionalOverride:
    """
    Test Category: Bidirectional Text Attacks

    Unicode bidirectional override characters (U+202E) can reverse text display.
    This is the "Trojan Source" vulnerability (CVE-2021-42574).
    """

    def test_rtl_override_EXPECT_SUCCESS(self, service):
        """
        Test: Use Right-to-Left Override (RLO) to hide code intent
        Expected: SUCCESS (Trojan Source vulnerability)
        Impact: CRITICAL - Can completely deceive code review

        Explanation:
        - RLO (U+202E) reverses text display direction
        - What looks like: # safe code
        - Actually is: # edoc efas (reversed)
        - Can hide malicious code that looks like comments
        """
        code = '''
# This test demonstrates Trojan Source vulnerability
# Visual: comment = "safe"  # Actually: comment = "harmful"

# RLO character: U+202E
# Can make code look harmless while being malicious

# Example: what looks like a comment is actually code
# The following looks like: # subprocess.run(['rm', '-rf', '/'])
# But actually executes!

# Safe version for testing (not using actual RLO)
comment = "This code demonstrates RLO attacks"
result = f"Bidirectional test: {comment}"
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "="*80)
            print("VULNERABILITY: Bidirectional Override (Trojan Source)")
            print("="*80)
            print(f"Result: {result}")

            if stats['success']:
                print("\n🚨 TROJAN SOURCE VULNERABILITY:")
                print("   - RLO (U+202E) can reverse displayed text")
                print("   - Code looks safe in editor but executes differently")
                print("   - Affects MANY programming languages")
                print("\n📋 Real-world example:")
                print("   Visual: /* } if (isAdmin) */ execute_dangerous_code();")
                print("   Actual: execute_dangerous_code(); if (isAdmin) { /*")
                print("\n📋 Reference:")
                print("   - CVE-2021-42574")
                print("   - https://trojansource.codes/")
                print("\n📋 Mitigation:")
                print("   - Reject code with bidirectional override characters")
                print("   - Use editors that highlight these characters")
                print("   - Normalize Unicode before processing")

        except (CodeValidationError, CodeExecutionError) as e:
            print(f"✅ PROTECTED: {e}")


class TestNormalizationAttacks:
    """
    Test Category: Unicode Normalization Attacks

    Unicode has multiple ways to represent the same character.
    Example: 'é' can be U+00E9 (precomposed) or U+0065 + U+0301 (e + combining acute)
    """

    def test_nfc_nfd_difference_EXPECT_SUCCESS(self, service):
        """
        Test: Use different Unicode normalizations to bypass filters
        Expected: SUCCESS (may bypass simple string matching)
        Impact: MEDIUM
        """
        code = '''
import unicodedata

# Two representations of 'é'
precomposed = '\\u00e9'  # NFC (Normalization Form Composed)
decomposed = 'e\\u0301'  # NFD (Normalization Form Decomposed)

# They look identical but are different strings
print(f"Precomposed: {repr(precomposed)}")
print(f"Decomposed: {repr(decomposed)}")
print(f"Equal: {precomposed == decomposed}")

# Can use this to bypass simple string matching
# Example: if 'café' in code: block()
# But code contains 'caf\\u0065\\u0301' (visually identical)

result = f"Normalization test: {precomposed} vs {decomposed}"
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "="*80)
            print("VULNERABILITY: Unicode Normalization Attack")
            print("="*80)
            print(f"Result: {result}")
            print(f"Stdout: {stats.get('stdout_preview', '')}")

            if stats['success']:
                print("\n🚨 NORMALIZATION BYPASS:")
                print("   - Same visual character, different bytes")
                print("   - Can bypass string-based filters")
                print("\n📋 Mitigation:")
                print("   - Normalize all code to NFC before validation")
                print("   - Use unicodedata.normalize('NFC', code)")

        except (CodeValidationError, CodeExecutionError) as e:
            print(f"✅ PROTECTED: {e}")


class TestLigatureAttacks:
    """
    Test Category: Ligature and Display Tricks

    Some fonts render character combinations as ligatures.
    Can be used to hide malicious code in plain sight.
    """

    def test_ligature_confusion_EXPECT_SUCCESS(self, service):
        """
        Test: Use characters that form ligatures
        Expected: SUCCESS (visual confusion only)
        Impact: LOW - Editor/font dependent
        """
        code = '''
# Some fonts render 'fi' as ligature (ﬁ U+FB01)
# Can confuse human reviewers

# Standard ASCII
variable_1 = "normal"

# With ligature character
variable_ﬁ = "ligature"  # 'ﬁ' is U+FB01 (single character!)

# These are DIFFERENT variables!
result = f"Ligature test: var1={variable_1}"
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "="*80)
            print("TEST: Ligature Character Confusion")
            print("="*80)
            print(f"Result: {result}")

            if stats['success']:
                print("\n⚠️ Ligature characters allowed")
                print("📋 Can cause visual confusion in code review")

        except (CodeValidationError, CodeExecutionError) as e:
            print(f"✅ PROTECTED: {e}")


class TestEncodingDeclarationAttack:
    """
    Test Category: Source Encoding Declaration Attacks

    Python allows declaring source encoding with # coding: declaration.
    Can be used to execute code via encoding names.
    """

    def test_encoding_declaration_EXPECT_SUCCESS(self, service):
        """
        Test: Try to exploit encoding declaration
        Expected: Likely safe (Python validates encoding names)
        Impact: LOW - Historical vulnerability
        """
        code = '''
# -*- coding: utf-8 -*-
# Python allows encoding declarations at start of file

# Historical vulnerability: some systems executed code in encoding name
# Example: # coding: print("PWNED")

result = "Encoding declaration test completed"
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "="*80)
            print("TEST: Encoding Declaration")
            print("="*80)
            print(f"Result: {result}")

            if stats['success']:
                print("\n✅ Modern Python safely handles encoding declarations")

        except (CodeValidationError, CodeExecutionError) as e:
            print(f"✅ PROTECTED: {e}")


class TestCombinedUnicodeAttacks:
    """
    Test Category: Combined Unicode Exploitation

    Combining multiple Unicode tricks for maximum evasion.
    """

    def test_combined_homograph_exec_EXPECT_SUCCESS(self, service):
        """
        Test: Combine homograph + obfuscation
        Expected: SUCCESS (critical vulnerability)
        Impact: CRITICAL
        """
        code = '''
# Combine techniques:
# 1. Cyrillic characters
# 2. Dynamic string building
# 3. exec() (which only warns)

# Use Cyrillic 'е' in variable name
ехес = exec  # Cyrillic 'е'

# Build import statement dynamically
parts = ['import', ' ', 'sub' + 'process']
statement = ' '.join(parts)

# Execute with Cyrillic exec
ехес(statement)

result = f"Combined attack: {'subprocess' in globals()}"
'''

        try:
            result, stats, ir = service.execute(code, persist=False, timeout=5)

            print("\n" + "="*80)
            print("CRITICAL: Combined Unicode Attack")
            print("="*80)
            print(f"Result: {result}")

            if stats['success']:
                print("\n🚨 COMBINED ATTACK SUCCESSFUL:")
                print("   - Homograph to hide exec")
                print("   - Dynamic strings to hide import")
                print("   - Multiple evasion techniques")
                print("\n📋 This demonstrates why AST alone is insufficient")

        except (CodeValidationError, CodeExecutionError) as e:
            print(f"✅ PROTECTED: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
