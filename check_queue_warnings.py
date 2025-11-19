"""
Helper script to check download queue for entries with validation warnings.
This helps identify datasets to test the warning display feature.
"""

import os
import json

# Set up environment
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', '')
os.environ['AWS_BEDROCK_ACCESS_KEY'] = os.getenv('AWS_BEDROCK_ACCESS_KEY', '')
os.environ['AWS_BEDROCK_SECRET_ACCESS_KEY'] = os.getenv('AWS_BEDROCK_SECRET_ACCESS_KEY', '')

from lobster.core.client import AgentClient

def main():
    print("Initializing AgentClient...")
    client = AgentClient()

    print("\n" + "="*80)
    print("DOWNLOAD QUEUE ANALYSIS")
    print("="*80)

    # Get all queue entries
    all_entries = client.data_manager.download_queue.list_entries()

    print(f"\nTotal queue entries: {len(all_entries)}")

    # Categorize by validation status
    by_status = {
        'validated_clean': [],
        'validated_warnings': [],
        'failed': [],
        'unknown': []
    }

    for entry in all_entries:
        if hasattr(entry, 'validation_status'):
            status = str(entry.validation_status)
            if status in by_status:
                by_status[status].append(entry)
            else:
                by_status['unknown'].append(entry)
        else:
            by_status['unknown'].append(entry)

    print("\n" + "-"*80)
    print("ENTRIES BY VALIDATION STATUS")
    print("-"*80)

    for status, entries in by_status.items():
        print(f"\n{status.upper()}: {len(entries)} entries")
        for entry in entries:
            print(f"  • {entry.dataset_id} (entry_id: {entry.entry_id})")

            # Show warnings if present
            if hasattr(entry, 'validation_result') and entry.validation_result:
                if 'warnings' in entry.validation_result:
                    warnings = entry.validation_result['warnings']
                    if warnings:
                        print(f"    ⚠️  {len(warnings)} warnings:")
                        for w in warnings[:3]:
                            print(f"      - {w}")
                        if len(warnings) > 3:
                            print(f"      - ... and {len(warnings) - 3} more")

    # Find best candidate for warning testing
    print("\n" + "="*80)
    print("RECOMMENDATION FOR WARNING DISPLAY TESTING")
    print("="*80)

    if by_status['validated_warnings']:
        print(f"\n✅ Found {len(by_status['validated_warnings'])} dataset(s) with warnings:")
        for entry in by_status['validated_warnings']:
            print(f"\n  Dataset: {entry.dataset_id}")
            print(f"  Entry ID: {entry.entry_id}")
            print(f"  Status: {entry.status}")

            if hasattr(entry, 'validation_result') and entry.validation_result:
                if 'warnings' in entry.validation_result:
                    warnings = entry.validation_result['warnings']
                    print(f"  Warnings: {len(warnings)}")
                    for i, w in enumerate(warnings[:5], 1):
                        print(f"    {i}. {w}")

            print(f"\n  Test command:")
            print(f"  python test_warning_display.py {entry.entry_id}")
    else:
        print("\n⚠️  No datasets with warnings found in queue")
        print("\n  To test warning display, try adding datasets that might trigger warnings:")
        print("  - Datasets with incomplete metadata")
        print("  - Datasets with unsupported platforms")
        print("  - Datasets with ambiguous modality detection")
        print("\n  Example datasets to try:")
        print("  - GSE60424 (potential metadata issues)")
        print("  - GSE10000 (older dataset, may have compatibility warnings)")

    # Show queue stats
    print("\n" + "="*80)
    print("QUEUE STATISTICS")
    print("="*80)

    by_queue_status = {}
    for entry in all_entries:
        status = entry.status
        by_queue_status[status] = by_queue_status.get(status, 0) + 1

    print("\nBy Queue Status:")
    for status, count in sorted(by_queue_status.items()):
        print(f"  {status}: {count}")

    print("\nBy Validation Status:")
    for status, entries in sorted(by_status.items()):
        print(f"  {status}: {len(entries)}")

if __name__ == "__main__":
    main()
