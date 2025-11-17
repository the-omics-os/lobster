"""
Manual functional test for Phase 2.1 LLM-based modality detection.

Tests the two-tier validation system (GPL registry + LLM modality detection)
with real GEO datasets covering supported and unsupported modalities.

Test Cases:
1. GSE156793 - Multiome (GEX+ATAC) - should be REJECTED
2. GSE123814 - CITE-seq (RNA+protein) - should be REJECTED
   NOTE: This is a SuperSeries (container) with empty metadata, not a valid test case.
   SuperSeries have no experimental data - only references to SubSeries.
   Phase 2.1 cannot detect modalities from empty SuperSeries metadata.
   This test is kept for documentation purposes but expected to fail.
3. GSE147507 - Bulk RNA-seq - should be ACCEPTED
4. GSE132044 - Single-cell RNA-seq - should be ACCEPTED

Expected Results: 3/4 tests pass (GSE123814 is a known limitation)
"""

import sys
from pathlib import Path

# Add lobster to path
lobster_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(lobster_root))

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.exceptions import FeatureNotImplementedError, UnsupportedPlatformError
from lobster.tools.geo_service import GEOService


def test_dataset(geo_id: str, expected_outcome: str, description: str):
    """
    Test a single GEO dataset.

    Args:
        geo_id: GEO series identifier
        expected_outcome: "ACCEPT" or "REJECT"
        description: Human-readable dataset description
    """
    print(f"\n{'='*70}")
    print(f"Testing: {geo_id} - {description}")
    print(f"Expected: {expected_outcome}")
    print(f"{'='*70}")

    try:
        # Initialize services
        data_manager = DataManagerV2()
        geo_service = GEOService(data_manager)

        # Attempt to fetch metadata (triggers validation)
        print(f"\nFetching metadata for {geo_id}...")
        result = geo_service.fetch_metadata_only(geo_id)

        # If we get here, validation passed
        if expected_outcome == "ACCEPT":
            print(f"✅ PASS: Dataset {geo_id} was ACCEPTED as expected")

            # Check modality detection details
            if geo_id in data_manager.metadata_store:
                modality_info = data_manager.metadata_store[geo_id].get(
                    "modality_detection"
                )
                if modality_info:
                    print(f"   Detected Modality: {modality_info['modality']}")
                    print(f"   Confidence: {modality_info['confidence']:.2%}")
                    print(f"   Signals: {modality_info['detected_signals'][:3]}")
            return True
        else:
            print(
                f"❌ FAIL: Dataset {geo_id} was ACCEPTED but should have been REJECTED"
            )
            return False

    except FeatureNotImplementedError as e:
        # Dataset rejected due to unsupported modality
        if expected_outcome == "REJECT":
            print(f"✅ PASS: Dataset {geo_id} was REJECTED as expected")
            print(f"   Reason: {e.message}")
            print(f"   Modality: {e.details.get('modality', 'N/A')}")
            print(f"   Confidence: {e.details.get('confidence', 0):.2%}")
            print(f"   Suggestions: {len(e.details.get('suggestions', []))} provided")
            return True
        else:
            print(
                f"❌ FAIL: Dataset {geo_id} was REJECTED but should have been ACCEPTED"
            )
            print(f"   Error: {e.message}")
            return False

    except UnsupportedPlatformError as e:
        # Dataset rejected due to microarray platform (Phase 2, Tier 1)
        if expected_outcome == "REJECT":
            print(f"✅ PASS: Dataset {geo_id} was REJECTED at Tier 1 (GPL registry)")
            print(f"   Reason: {e.message}")
            return True
        else:
            print(
                f"❌ FAIL: Dataset {geo_id} was REJECTED at Tier 1 but should have been ACCEPTED"
            )
            print(f"   Error: {e.message}")
            return False

    except Exception as e:
        print(f"❌ ERROR: Unexpected exception for {geo_id}")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        return False


def main():
    """Run all test cases."""
    print("\n" + "=" * 70)
    print("Phase 2.1 Functional Test: LLM-Based Modality Detection")
    print("=" * 70)

    test_cases = [
        # Unsupported modalities (should be rejected)
        ("GSE156793", "REJECT", "Multiome (GEX+ATAC) - 10X Single Cell Multiome"),
        ("GSE123814", "SKIP", "CITE-seq (RNA+protein) - SuperSeries (invalid test)"),
        # Supported modalities (should be accepted)
        ("GSE147507", "ACCEPT", "Bulk RNA-seq - COVID-19 patient samples"),
        ("GSE132044", "ACCEPT", "Single-cell RNA-seq - 10X Chromium"),
    ]

    results = []
    for geo_id, expected, description in test_cases:
        if expected == "SKIP":
            print(f"\n{'='*70}")
            print(f"Testing: {geo_id} - {description}")
            print(f"Expected: {expected}")
            print(f"{'='*70}")
            print(
                f"⚠️  SKIP: {geo_id} is a SuperSeries with empty metadata (known limitation)"
            )
            results.append((geo_id, None))  # None = skipped
        else:
            passed = test_dataset(geo_id, expected, description)
            results.append((geo_id, passed))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for _, passed in results if passed is True)
    failed_count = sum(1 for _, passed in results if passed is False)
    skipped_count = sum(1 for _, passed in results if passed is None)
    total_count = len(results)

    for geo_id, passed in results:
        if passed is True:
            status = "✅ PASS"
        elif passed is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  SKIP"
        print(f"{status}: {geo_id}")

    print(
        f"\nResults: {passed_count} passed, {failed_count} failed, {skipped_count} skipped (of {total_count} total)"
    )

    if failed_count == 0:
        print(
            "\n🎉 All valid tests passed! Phase 2.1 integration is working correctly."
        )
        print(f"   (Note: {skipped_count} test(s) skipped due to known limitations)")
        return 0
    else:
        print(f"\n⚠️  {failed_count} test(s) failed. Review logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
