#!/usr/bin/env python3
"""
Test to reproduce "Identifier not found" bug when data NOT in metadata_store.

Usage:
    python tests/manual/test_identifier_not_found_bug.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.workspace_tool import create_write_to_workspace_tool


def test_identifier_not_in_metadata_store():
    """Test write_to_workspace with identifier NOT in metadata_store."""
    print("=" * 80)
    print("TEST: Identifier NOT in metadata_store")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_path = Path(tmpdir) / ".lobster_workspace"
        workspace_path.mkdir(parents=True)
        dm = DataManagerV2(workspace_path)

        # DO NOT populate metadata_store - this is the bug scenario
        print("\nmetadata_store is empty (no data)")
        print(f"metadata_store keys: {list(dm.metadata_store.keys())}")

        # Create the tool
        write_tool = create_write_to_workspace_tool(dm)

        print("\nAttempting to write non-existent identifier to workspace...")
        print("Calling: write_to_workspace(identifier='GSE_NOT_EXIST', workspace='data')")

        try:
            result = write_tool.invoke({
                "identifier": "GSE_NOT_EXIST",
                "workspace": "data",
                "content_type": "dataset",
                "output_format": "json",
            })

            print(f"\nResult: {result}")

            # Check what error message we get
            if "Identifier not found" in result:
                print("\n✓ Expected error message: 'Identifier not found'")
                return True
            elif "not found" in result.lower() or "does not exist" in result.lower():
                print(f"\n✓ Got 'not found' error (different wording): {result[:200]}")
                return True
            elif "Error" in result:
                print(f"\n⚠ Got error but different message: {result[:200]}")
                return False
            else:
                print(f"\n✗ No error message! Unexpected success: {result[:200]}")
                return False

        except Exception as e:
            print(f"\n✗ EXCEPTION raised: {type(e).__name__}: {e}")
            return False


def test_valid_identifier_for_comparison():
    """Test with valid identifier in metadata_store for comparison."""
    print("\n" + "=" * 80)
    print("TEST: Valid identifier IN metadata_store (for comparison)")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_path = Path(tmpdir) / ".lobster_workspace"
        workspace_path.mkdir(parents=True)
        dm = DataManagerV2(workspace_path)

        # Populate metadata_store with test data
        dm.metadata_store["GSE_VALID"] = {
            "metadata": {
                "title": "Valid Test Dataset",
                "n_samples": 10,
                "organism": "human",
            },
            "stored_by": "test",
        }

        print(f"\nmetadata_store has 1 entry: {list(dm.metadata_store.keys())}")

        # Create the tool
        write_tool = create_write_to_workspace_tool(dm)

        print("\nAttempting to write VALID identifier to workspace...")
        print("Calling: write_to_workspace(identifier='GSE_VALID', workspace='data')")

        try:
            result = write_tool.invoke({
                "identifier": "GSE_VALID",
                "workspace": "data",
                "content_type": "dataset",
                "output_format": "json",
            })

            print(f"\nResult: {result[:200]}...")

            if "Error" in result:
                print(f"\n⚠ Got error even with valid ID: {result[:200]}")
                return False
            else:
                print(f"\n✓ No error with valid identifier")
                return True

        except Exception as e:
            print(f"\n✗ EXCEPTION with valid ID: {type(e).__name__}: {e}")
            return False


def main():
    """Run tests to reproduce the bug."""
    print("\n" + "=" * 80)
    print("REPRODUCE: 'Identifier not found' Bug")
    print("=" * 80)
    print("\nScenario: Calling write_to_workspace with identifier NOT in metadata_store")
    print("Expected: Clear error message like 'Identifier not found'")
    print("Bug: Error message may be unclear or missing\n")

    results = []

    # Test 1: Identifier not in metadata_store (bug scenario)
    result1 = test_identifier_not_in_metadata_store()
    results.append(("Identifier NOT in metadata_store", result1))

    # Test 2: Valid identifier (for comparison)
    result2 = test_valid_identifier_for_comparison()
    results.append(("Valid identifier IN metadata_store", result2))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} passed")

    if results[0][1]:
        print("\n✅ Error message is clear and helpful")
        return 0
    else:
        print("\n❌ Error message needs improvement")
        return 1


if __name__ == "__main__":
    sys.exit(main())
