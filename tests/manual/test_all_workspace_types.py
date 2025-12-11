#!/usr/bin/env python3
"""
Comprehensive test for all workspace types after model mismatch bug fix.

Tests that write_to_workspace works correctly for:
- workspace='literature' (PublicationContent)
- workspace='data' (DatasetContent)
- workspace='metadata' (MetadataContent)

Usage:
    python tests/manual/test_all_workspace_types.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.workspace_tool import create_write_to_workspace_tool


def test_workspace_literature():
    """Test workspace='literature' with publication data."""
    print("=" * 80)
    print("TEST 1: workspace='literature' (PublicationContent)")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_path = Path(tmpdir) / ".lobster_workspace"
        workspace_path.mkdir(parents=True)
        dm = DataManagerV2(workspace_path)

        # Populate metadata_store with publication data
        dm.metadata_store["PMID_12345678"] = {
            "title": "Test Publication on RNA-seq",
            "authors": ["Smith J", "Jones A"],
            "journal": "Nature Biotechnology",
            "year": 2024,
            "abstract": "We performed single-cell RNA-seq analysis...",
            "pmid": "12345678",
            "doi": "10.1038/nbt.1234",
        }

        write_tool = create_write_to_workspace_tool(dm)

        print("\nAttempting to write publication to literature workspace...")
        try:
            result = write_tool.invoke({
                "identifier": "PMID_12345678",
                "workspace": "literature",
                "output_format": "json",
            })

            if "Error" in result and "requires PublicationContent" in result:
                print(f"✗ FAILED: Model mismatch error still exists!")
                print(f"   {result[:200]}")
                return False
            elif "Content Cached Successfully" in result:
                print(f"✓ PASSED: Successfully cached to literature workspace")
                print(f"   {result[:150]}")
                return True
            else:
                print(f"⚠ UNCLEAR: {result[:200]}")
                return False

        except Exception as e:
            print(f"✗ EXCEPTION: {e}")
            return False


def test_workspace_data():
    """Test workspace='data' with dataset metadata."""
    print("\n" + "=" * 80)
    print("TEST 2: workspace='data' (DatasetContent)")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_path = Path(tmpdir) / ".lobster_workspace"
        workspace_path.mkdir(parents=True)
        dm = DataManagerV2(workspace_path)

        # Populate metadata_store with dataset metadata (nested structure)
        dm.metadata_store["GSE123456"] = {
            "metadata": {
                "title": "Single-cell RNA-seq of mouse brain",
                "organism": "Mus musculus",
                "n_samples": 24,
                "platform": "Illumina NovaSeq",
                "database": "GEO",
            },
            "stored_by": "research_agent",
        }

        write_tool = create_write_to_workspace_tool(dm)

        print("\nAttempting to write dataset to data workspace...")
        try:
            result = write_tool.invoke({
                "identifier": "GSE123456",
                "workspace": "data",
                "output_format": "json",
            })

            if "Error" in result and "requires DatasetContent" in result:
                print(f"✗ FAILED: Model mismatch error still exists!")
                print(f"   {result[:200]}")
                return False
            elif "Content Cached Successfully" in result:
                print(f"✓ PASSED: Successfully cached to data workspace")
                print(f"   {result[:150]}")
                return True
            else:
                print(f"⚠ UNCLEAR: {result[:200]}")
                return False

        except Exception as e:
            print(f"✗ EXCEPTION: {e}")
            return False


def test_workspace_metadata():
    """Test workspace='metadata' (existing behavior - should still work)."""
    print("\n" + "=" * 80)
    print("TEST 3: workspace='metadata' (MetadataContent)")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_path = Path(tmpdir) / ".lobster_workspace"
        workspace_path.mkdir(parents=True)
        dm = DataManagerV2(workspace_path)

        # Populate metadata_store with arbitrary metadata
        dm.metadata_store["custom_mapping"] = {
            "sample_ids": ["S1", "S2", "S3"],
            "mapping_type": "cross_dataset",
            "validation_score": 0.95,
        }

        write_tool = create_write_to_workspace_tool(dm)

        print("\nAttempting to write metadata to metadata workspace...")
        try:
            result = write_tool.invoke({
                "identifier": "custom_mapping",
                "workspace": "metadata",
                "content_type": "metadata",  # Fixed: must be 'metadata' not 'sample_mapping'
                "output_format": "json",
            })

            if "Error" in result:
                print(f"✗ FAILED: {result[:200]}")
                return False
            elif "Content Cached Successfully" in result:
                print(f"✓ PASSED: Successfully cached to metadata workspace")
                print(f"   {result[:150]}")
                return True
            else:
                print(f"⚠ UNCLEAR: {result[:200]}")
                return False

        except Exception as e:
            print(f"✗ EXCEPTION: {e}")
            return False


def test_workspace_data_flat_structure():
    """Test workspace='data' with flat metadata structure (edge case)."""
    print("\n" + "=" * 80)
    print("TEST 4: workspace='data' with flat structure (DatasetContent)")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_path = Path(tmpdir) / ".lobster_workspace"
        workspace_path.mkdir(parents=True)
        dm = DataManagerV2(workspace_path)

        # Populate with FLAT structure (no nested "metadata" key)
        dm.metadata_store["GSE999999"] = {
            "title": "Flat structure dataset",
            "organism": "Homo sapiens",
            "n_samples": 10,
            "database": "GEO",
        }

        write_tool = create_write_to_workspace_tool(dm)

        print("\nAttempting to write flat-structure dataset...")
        try:
            result = write_tool.invoke({
                "identifier": "GSE999999",
                "workspace": "data",
                "output_format": "json",
            })

            if "Error" in result:
                print(f"✗ FAILED: {result[:200]}")
                return False
            elif "Content Cached Successfully" in result:
                print(f"✓ PASSED: Flat structure handled correctly")
                print(f"   {result[:150]}")
                return True
            else:
                print(f"⚠ UNCLEAR: {result[:200]}")
                return False

        except Exception as e:
            print(f"✗ EXCEPTION: {e}")
            return False


def main():
    """Run all workspace type tests."""
    print("\n" + "=" * 80)
    print("WORKSPACE MODEL MISMATCH BUG - COMPREHENSIVE FIX VERIFICATION")
    print("=" * 80)
    print("\nTesting all 3 workspace types after fix...")
    print()

    results = []

    # Test 1: literature
    result1 = test_workspace_literature()
    results.append(("workspace='literature' (PublicationContent)", result1))

    # Test 2: data (nested structure)
    result2 = test_workspace_data()
    results.append(("workspace='data' nested (DatasetContent)", result2))

    # Test 3: metadata (existing behavior)
    result3 = test_workspace_metadata()
    results.append(("workspace='metadata' (MetadataContent)", result3))

    # Test 4: data (flat structure)
    result4 = test_workspace_data_flat_structure()
    results.append(("workspace='data' flat (DatasetContent)", result4))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} passed ({100 * passed / total:.1f}%)")

    if passed == total:
        print("\n✅ ALL WORKSPACE TYPES WORK - Bug is completely fixed!")
        print("\nFix summary:")
        print("  - Added PublicationContent and DatasetContent imports")
        print("  - Implemented model selection based on workspace parameter")
        print("  - Used .get() with defaults for flexible field handling")
        print("  - Preserved existing behavior for workspace='metadata'")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Review implementation")
        return 1


if __name__ == "__main__":
    sys.exit(main())
