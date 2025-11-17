#!/usr/bin/env python3
"""
Inspect GEO metadata to understand what the LLM sees during modality detection.

This script examines the actual supplementary files and metadata for test datasets
to diagnose why the LLM is misclassifying Multiome and CITE-seq datasets.
"""

import sys
from pathlib import Path

# Add lobster to path
lobster_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(lobster_root))

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.geo_service import GEOService


def inspect_dataset(geo_id: str, expected_modality: str):
    """
    Inspect GEO dataset metadata and file list.

    Args:
        geo_id: GEO series identifier
        expected_modality: Expected modality classification
    """
    print(f"\n{'='*80}")
    print(f"Inspecting: {geo_id}")
    print(f"Expected Modality: {expected_modality}")
    print(f"{'='*80}\n")

    # Initialize services
    data_manager = DataManagerV2()
    geo_service = GEOService(data_manager)

    try:
        # Fetch metadata only (no file download)
        print("Fetching metadata from GEO...")
        result = geo_service.fetch_metadata_only(geo_id)

        # Extract metadata
        if geo_id in data_manager.metadata_store:
            metadata = data_manager.metadata_store[geo_id]

            # Display title
            title = metadata.get("title", "N/A")
            print(f"\n📋 Title: {title}\n")

            # Display summary
            summary = metadata.get("summary", "N/A")
            print(f"📝 Summary:\n{summary[:300]}...\n")

            # Display overall design
            overall_design = metadata.get("overall_design", "N/A")
            print(f"🔬 Overall Design:\n{overall_design[:300]}...\n")

            # Display supplementary files
            supplementary_files = metadata.get("supplementary_file", [])
            print(f"📦 Supplementary Files ({len(supplementary_files)} total):\n")

            # Show ALL files (not just first 20)
            for i, file_name in enumerate(supplementary_files[:50], 1):
                # Highlight multiome/ATAC/CITE-seq related patterns
                file_lower = file_name.lower()
                flag = ""
                if any(
                    pattern in file_lower
                    for pattern in ["atac", "fragment", "multiome"]
                ):
                    flag = " ⚠️  [ATAC/MULTIOME PATTERN]"
                elif any(
                    pattern in file_lower
                    for pattern in ["cite", "adt", "antibody", "protein"]
                ):
                    flag = " ⚠️  [CITE-SEQ PATTERN]"
                elif any(
                    pattern in file_lower
                    for pattern in ["gex", "rna", "matrix", "barcode", "features"]
                ):
                    flag = " ✓ [RNA PATTERN]"

                print(f"  {i}. {file_name}{flag}")

            if len(supplementary_files) > 50:
                print(f"\n  ... and {len(supplementary_files) - 50} more files")

            # Display platform information
            platforms = metadata.get("platforms", {})
            print(f"\n🔧 Platforms:")
            for gpl_id, platform_data in platforms.items():
                platform_title = platform_data.get("title", "Unknown")
                print(f"  - {gpl_id}: {platform_title}")

            # Display modality detection result if present
            if "modality_detection" in metadata:
                modality_info = metadata["modality_detection"]
                print(f"\n🤖 LLM Modality Detection Result:")
                print(f"  - Detected Modality: {modality_info['modality']}")
                print(f"  - Confidence: {modality_info['confidence']:.2%}")
                print(
                    f"  - Detected Signals: {modality_info.get('detected_signals', [])}"
                )
            else:
                print("\n⚠️  No modality detection result in metadata")

        else:
            print(f"❌ No metadata found for {geo_id} in metadata_store")

    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()


def main():
    """Inspect test datasets."""
    print("\n" + "=" * 80)
    print("GEO Metadata Inspection for Phase 2.1 Modality Detection")
    print("=" * 80)

    # Test cases from functional test
    test_cases = [
        ("GSE156793", "multiome_gex_atac"),  # Multiome (should be rejected)
        ("GSE123814", "cite_seq"),  # CITE-seq (should be rejected)
        ("GSE147507", "bulk_rna"),  # Bulk RNA-seq (should be accepted)
        ("GSE132044", "scrna_10x"),  # Single-cell (should be accepted)
    ]

    for geo_id, expected_modality in test_cases:
        inspect_dataset(geo_id, expected_modality)
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
