#!/usr/bin/env python3
"""
Manual integration test for unified identifier parameter architecture.

Tests the renamed parameters by inspecting the tool function signatures directly.

Usage:
    python tests/manual/test_unified_identifier_params_v2.py
"""

import inspect
import sys
from pathlib import Path

# Add lobster to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_parameter_signatures():
    """Test parameter signatures by inspecting the source code directly."""
    print("=" * 80)
    print("UNIFIED IDENTIFIER PARAMETER - SIGNATURE INSPECTION")
    print("=" * 80)

    # Import the module
    from lobster.agents import research_agent as ra_module

    # Find the research_agent factory function
    research_agent_factory = ra_module.research_agent

    # Get the source code
    source = inspect.getsource(research_agent_factory)

    print("\n" + "=" * 80)
    print("TEST 1: validate_dataset_metadata signature")
    print("=" * 80)

    # Check validate_dataset_metadata
    if "def validate_dataset_metadata(" in source:
        # Extract the function signature
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'def validate_dataset_metadata(' in line:
                # Get function signature (may span multiple lines)
                sig_lines = []
                j = i
                while j < len(lines) and ('-> str:' not in lines[j]):
                    sig_lines.append(lines[j])
                    j += 1
                sig_lines.append(lines[j])  # Include the return type line

                signature = '\n'.join(sig_lines)
                print(f"\nFound signature:\n{signature}\n")

                # Check for identifier parameter
                if 'identifier: str' in signature:
                    print("✓ PASSED: Uses 'identifier: str' parameter")
                    test1_pass = True
                else:
                    print("✗ FAILED: Does NOT use 'identifier' parameter")
                    test1_pass = False

                # Check for old accession parameter
                if 'accession: str' in signature:
                    print("✗ FAILED: Old 'accession' parameter still exists!")
                    test1_pass = False
                else:
                    print("✓ PASSED: Old 'accession' parameter removed")

                break
        else:
            print("✗ FAILED: validate_dataset_metadata function not found")
            test1_pass = False
    else:
        print("✗ FAILED: validate_dataset_metadata function not found")
        test1_pass = False

    print("\n" + "=" * 80)
    print("TEST 2: extract_methods signature")
    print("=" * 80)

    # Check extract_methods
    if "def extract_methods(" in source:
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'def extract_methods(' in line:
                # Get function signature
                sig_lines = []
                j = i
                while j < len(lines) and ('-> str:' not in lines[j]):
                    sig_lines.append(lines[j])
                    j += 1
                sig_lines.append(lines[j])

                signature = '\n'.join(sig_lines)
                print(f"\nFound signature:\n{signature}\n")

                # Check for identifier parameter
                if 'identifier: str' in signature:
                    print("✓ PASSED: Uses 'identifier: str' parameter")
                    test2_pass = True
                else:
                    print("✗ FAILED: Does NOT use 'identifier' parameter")
                    test2_pass = False

                # Check for old url_or_pmid parameter
                if 'url_or_pmid: str' in signature:
                    print("✗ FAILED: Old 'url_or_pmid' parameter still exists!")
                    test2_pass = False
                else:
                    print("✓ PASSED: Old 'url_or_pmid' parameter removed")

                break
        else:
            print("✗ FAILED: extract_methods function not found")
            test2_pass = False
    else:
        print("✗ FAILED: extract_methods function not found")
        test2_pass = False

    print("\n" + "=" * 80)
    print("TEST 3: System prompt includes parameter naming convention")
    print("=" * 80)

    # Check system prompt includes parameter naming convention
    if '<parameter naming convention>' in source:
        print("\n✓ PASSED: System prompt includes '<parameter naming convention>' section")
        test3_pass = True

        # Check for specific content
        if 'External identifiers' in source and 'Internal queue IDs' in source:
            print("✓ PASSED: Convention includes both external and internal ID guidance")
        else:
            print("✗ WARNING: Convention section may be incomplete")

        if 'identifier` parameter' in source:
            print("✓ PASSED: Mentions 'identifier' parameter")
        else:
            print("✗ WARNING: Doesn't explicitly mention 'identifier' parameter")

        if 'entry_id` parameter' in source:
            print("✓ PASSED: Mentions 'entry_id' parameter")
        else:
            print("✗ WARNING: Doesn't explicitly mention 'entry_id' parameter")

    else:
        print("✗ FAILED: System prompt does NOT include parameter naming convention")
        test3_pass = False

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    results = [
        ("validate_dataset_metadata uses 'identifier'", test1_pass),
        ("extract_methods uses 'identifier'", test2_pass),
        ("System prompt includes naming convention", test3_pass),
    ]

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed ({100 * passed / total:.1f}%)")

    if passed == total:
        print("\n✅ ALL TESTS PASSED - Unified identifier architecture verified!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(test_parameter_signatures())
