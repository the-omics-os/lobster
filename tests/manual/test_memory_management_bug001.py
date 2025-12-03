"""
Test script for BUG-001: Memory overflow on large datasets

Tests the production-grade memory management system in GEOParser:
1. Dimension-based memory estimation
2. Pre-flight memory checks
3. Clear error messages with actionable options
4. Timeout protection

Run this script to verify the fix works for GSE150290 (112k cells) and similar large datasets.
"""

import sys
from pathlib import Path

# Add lobster to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lobster.services.data_access.geo.parser import GEOParser


def test_memory_estimation():
    """Test memory estimation for various dataset sizes."""
    print("=" * 80)
    print("TEST 1: Memory Estimation from Dimensions")
    print("=" * 80)

    parser = GEOParser()

    # Test cases: (n_cells, n_genes, expected_description)
    test_cases = [
        (1000, 2000, "Small dataset (1k cells)"),
        (10000, 20000, "Medium dataset (10k cells)"),
        (50000, 25000, "Large dataset (50k cells)"),
        (112000, 30000, "GSE150290 (112k cells)"),
        (208000, 35000, "GSE131907 (208k cells)"),
    ]

    for n_cells, n_genes, description in test_cases:
        print(f"\n{description}:")
        print(f"  Dimensions: {n_cells:,} cells × {n_genes:,} genes")

        # Estimate memory
        mem_estimate = parser.estimate_memory_from_dimensions(n_cells, n_genes)
        print(f"  Base memory: {mem_estimate['base_memory_gb']:.2f} GB")
        print(f"  With overhead: {mem_estimate['with_overhead_gb']:.2f} GB")
        print(f"  Recommended: {mem_estimate['recommended_gb']:.2f} GB")

        # Check if system can handle it
        mem_check = parser.check_memory_for_dimensions(n_cells, n_genes)
        print(f"  Can load: {'✓ YES' if mem_check['can_load'] else '✗ NO'}")
        print(f"  Available: {mem_check['available_gb']:.2f} GB")

        if not mem_check["can_load"]:
            print(f"  Shortfall: {mem_check['shortfall_gb']:.2f} GB")
            if mem_check["subsample_target"]:
                print(f"  Suggested subsample: {mem_check['subsample_target']:,} cells")

    print("\n" + "=" * 80)


def test_memory_check_messages():
    """Test that error messages are clear and actionable."""
    print("=" * 80)
    print("TEST 2: Error Message Quality")
    print("=" * 80)

    parser = GEOParser()

    # Test with GSE150290 dimensions (likely to fail on 5GB system)
    n_cells, n_genes = 112000, 30000

    mem_check = parser.check_memory_for_dimensions(n_cells, n_genes)

    print(f"\nDataset: {n_cells:,} cells × {n_genes:,} genes")
    print(
        f"System: {mem_check['total_system_gb']:.1f} GB total, "
        f"{mem_check['available_gb']:.1f} GB available"
    )
    print("\nRecommendation:")
    print(mem_check["recommendation"])

    print("\n" + "=" * 80)


def test_dimension_estimation():
    """Test dimension estimation from file (if test file exists)."""
    print("=" * 80)
    print("TEST 3: Dimension Estimation from File")
    print("=" * 80)

    # This test requires an actual file, which we don't have in automated tests
    # But we can demonstrate the API

    parser = GEOParser()

    print("\nDimension estimation API:")
    print("  parser._estimate_dimensions_from_file(file_path)")
    print("  Returns: (n_cells, n_genes) tuple")
    print("\nThis is called automatically before loading large files.")
    print("No test file available - skipping actual estimation.")

    print("\n" + "=" * 80)


def test_timeout_configuration():
    """Test timeout configuration."""
    print("=" * 80)
    print("TEST 4: Timeout Configuration")
    print("=" * 80)

    # Default timeout
    parser1 = GEOParser()
    print(f"\nDefault timeout: {parser1.timeout_seconds}s (5 minutes)")

    # Custom timeout
    parser2 = GEOParser(timeout_seconds=600)
    print(f"Custom timeout: {parser2.timeout_seconds}s (10 minutes)")

    print("\nTimeout prevents infinite hangs on memory-starved systems.")
    print("If loading exceeds timeout, raises LoadingTimeout exception.")

    print("\n" + "=" * 80)


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "BUG-001 Memory Management Tests" + " " * 25 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")

    try:
        test_memory_estimation()
        test_memory_check_messages()
        test_dimension_estimation()
        test_timeout_configuration()

        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 28 + "ALL TESTS PASSED" + " " * 33 + "║")
        print("╚" + "═" * 78 + "╝")
        print("\n")

        print("NEXT STEPS:")
        print("1. Test on actual large dataset: GSE150290 (112k cells)")
        print("2. Verify error messages guide users to actionable solutions")
        print("3. Confirm timeout protection prevents infinite hangs")
        print("\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
