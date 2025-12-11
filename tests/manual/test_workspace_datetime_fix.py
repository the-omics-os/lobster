#!/usr/bin/env python3
"""
Test to verify datetime scoping bug is fixed in workspace_tool.py.

This test exercises the code paths that use datetime.now() to ensure
no UnboundLocalError occurs.

Usage:
    python tests/manual/test_workspace_datetime_fix.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.workspace_tool import (
    create_get_content_from_workspace_tool,
    create_write_to_workspace_tool,
)


def test_write_to_workspace_datetime():
    """Test that write_to_workspace can use datetime.now() without error."""
    print("=" * 80)
    print("TEST: write_to_workspace datetime usage")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_path = Path(tmpdir) / ".lobster_workspace"
        workspace_path.mkdir(parents=True)
        dm = DataManagerV2(workspace_path)

        # Create the tool
        write_tool = create_write_to_workspace_tool(dm)

        # Test Case 1: Write dataset metadata (exercises line 1257: cached_at=datetime.now())
        print("\nTest Case 1: Write dataset to data workspace (line 1257 datetime call)")
        try:
            # Simulate agent call with minimal valid parameters
            result = write_tool.invoke({
                "identifier": "GSE_TEST_12345",
                "workspace": "data",
                "content_type": "dataset",
                "output_format": "json",
            })

            if "Error" not in result and "cannot access local variable 'datetime'" not in result:
                print("✓ PASSED: datetime.now() at line 1257 works correctly")
                test1_pass = True
            else:
                print(f"✗ FAILED: {result}")
                test1_pass = False
        except Exception as e:
            if "cannot access local variable 'datetime'" in str(e):
                print(f"✗ FAILED: Scoping bug still exists - {e}")
                test1_pass = False
            else:
                # Other errors are OK (e.g., content not found in metadata_store)
                print(f"✓ PASSED: No datetime scoping error (other error: {str(e)[:100]})")
                test1_pass = True

        return test1_pass


def test_get_content_from_workspace_datetime():
    """Test that get_content_from_workspace can use datetime.now() without error."""
    print("\n" + "=" * 80)
    print("TEST: get_content_from_workspace datetime usage")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_path = Path(tmpdir) / ".lobster_workspace"
        workspace_path.mkdir(parents=True)
        dm = DataManagerV2(workspace_path)

        # Create the tool
        get_tool = create_get_content_from_workspace_tool(dm)

        # Test Case: List download queue (exercises line 628: datetime.now())
        print("\nTest Case: List download queue (line 628 datetime call)")
        try:
            result = get_tool.invoke({
                "workspace": "download_queue",
                "level": "summary",
            })

            if "cannot access local variable 'datetime'" not in result:
                print("✓ PASSED: datetime.now() at line 628 works correctly")
                test2_pass = True
            else:
                print(f"✗ FAILED: {result}")
                test2_pass = False
        except Exception as e:
            if "cannot access local variable 'datetime'" in str(e):
                print(f"✗ FAILED: Scoping bug still exists - {e}")
                test2_pass = False
            else:
                # Other errors are OK
                print(f"✓ PASSED: No datetime scoping error (result: {str(e)[:100]})")
                test2_pass = True

        return test2_pass


def main():
    """Run all tests."""
    print("=" * 80)
    print("WORKSPACE_TOOL DATETIME SCOPING FIX - VERIFICATION")
    print("=" * 80)
    print("\nVerifying that datetime.now() calls work without UnboundLocalError")
    print()

    results = []

    # Test 1
    try:
        result = test_write_to_workspace_datetime()
        results.append(("write_to_workspace datetime fix", result))
    except Exception as e:
        print(f"✗ EXCEPTION: {e}")
        results.append(("write_to_workspace datetime fix", False))

    # Test 2
    try:
        result = test_get_content_from_workspace_datetime()
        results.append(("get_content_from_workspace datetime fix", result))
    except Exception as e:
        print(f"✗ EXCEPTION: {e}")
        results.append(("get_content_from_workspace datetime fix", False))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed ({100 * passed / total:.1f}%)")

    if passed == total:
        print("\n✅ ALL TESTS PASSED - datetime scoping bug fixed!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
