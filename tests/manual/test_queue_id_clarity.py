#!/usr/bin/env python3
"""
Test to verify system prompt clearly distinguishes publication queue vs download queue IDs.

This test checks that the research_agent system prompt:
1. Explains the difference between pub_queue_* and queue_* IDs
2. Makes it clear research_agent cannot handle download queue IDs
3. Provides examples of what NOT to do

Usage:
    python tests/manual/test_queue_id_clarity.py
"""

import inspect
import sys
from pathlib import Path

# Add lobster to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lobster.agents import research_agent as ra_module


def test_system_prompt_queue_clarity():
    """Test that system prompt clearly distinguishes queue types."""
    print("=" * 80)
    print("QUEUE ID CLARITY TEST - System Prompt Verification")
    print("=" * 80)

    # Get research_agent factory source
    source = inspect.getsource(ra_module.research_agent)

    print("\nChecking for queue type distinction in system prompt...")

    checks = []

    # Check 1: Mentions publication queue IDs
    if "pub_queue_" in source:
        print("✓ PASS: Mentions 'pub_queue_' format")
        checks.append(True)
    else:
        print("✗ FAIL: Does NOT mention 'pub_queue_' format")
        checks.append(False)

    # Check 2: Mentions download queue IDs
    if "queue_GSE" in source or "queue_SRP" in source or "Download queue IDs" in source:
        print("✓ PASS: Mentions download queue ID format")
        checks.append(True)
    else:
        print("✗ FAIL: Does NOT mention download queue ID format")
        checks.append(False)

    # Check 3: Clarifies research_agent doesn't handle download queue
    if "YOU DO NOT HAVE TOOLS for download queue" in source or "NOT accessible to research_agent" in source:
        print("✓ PASS: Explicitly states research_agent cannot handle download queue IDs")
        checks.append(True)
    else:
        print("✗ FAIL: Doesn't clarify research_agent limitations on download queue")
        checks.append(False)

    # Check 4: Mentions data_expert handles download queue
    if "data_expert" in source and ("download" in source or "execute_download_from_queue" in source):
        print("✓ PASS: Explains data_expert handles download queue")
        checks.append(True)
    else:
        print("✗ FAIL: Doesn't explain data_expert role")
        checks.append(False)

    # Check 5: Provides negative example (what NOT to do)
    if "WRONG:" in source and "queue_GSE" in source:
        print("✓ PASS: Provides explicit 'WRONG' example with download queue ID")
        checks.append(True)
    else:
        print("✗ FAIL: Missing explicit 'WRONG' example for download queue ID")
        checks.append(False)

    # Check 6: Distinguishes between pub_queue and queue prefixes
    if "pub_queue_doi_" in source or "pub_queue_pmid_" in source:
        print("✓ PASS: Shows specific publication queue ID format examples")
        checks.append(True)
    else:
        print("✗ FAIL: Missing specific publication queue ID format examples")
        checks.append(False)

    # Check 7: Parameter naming convention section exists
    if "<parameter naming convention>" in source:
        print("✓ PASS: Has '<parameter naming convention>' section")
        checks.append(True)
    else:
        print("✗ FAIL: Missing '<parameter naming convention>' section")
        checks.append(False)

    # Summary
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)

    passed = sum(checks)
    total = len(checks)

    print(f"\nPassed: {passed}/{total} checks ({100 * passed / total:.1f}%)")

    if passed == total:
        print("\n✅ ALL CHECKS PASSED - System prompt clearly distinguishes queue types")
        print("\nThe research_agent should now understand:")
        print("  - pub_queue_* → process_publication_entry (its own tool)")
        print("  - queue_* → hand off to data_expert (not its responsibility)")
        return 0
    else:
        print("\n❌ SOME CHECKS FAILED - System prompt needs improvement")
        return 1


if __name__ == "__main__":
    sys.exit(test_system_prompt_queue_clarity())
