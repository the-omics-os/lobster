"""
Supplementary Test: Warning Display Feature
Tests data_expert warning display with datasets that have validation warnings.
"""

import os
import json
import sys

# Set up environment
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', '')
os.environ['AWS_BEDROCK_ACCESS_KEY'] = os.getenv('AWS_BEDROCK_ACCESS_KEY', '')
os.environ['AWS_BEDROCK_SECRET_ACCESS_KEY'] = os.getenv('AWS_BEDROCK_SECRET_ACCESS_KEY', '')

from lobster.core.client import AgentClient

def test_existing_entry_with_warnings(client, entry_id):
    """Test warning display for an existing queue entry with warnings."""
    print("\n" + "="*80)
    print(f"TEST: Warning Display for Entry {entry_id}")
    print("="*80)

    # Get entry details
    entry = None
    for e in client.data_manager.download_queue.list_entries():
        if e.entry_id == entry_id:
            entry = e
            break

    if not entry:
        print(f"❌ Entry {entry_id} not found in queue")
        return None

    print(f"\nEntry Details:")
    print(f"  Dataset: {entry.dataset_id}")
    print(f"  Status: {entry.status}")
    print(f"  Validation Status: {getattr(entry, 'validation_status', 'NOT SET')}")

    # Check for warnings
    has_warnings = False
    if hasattr(entry, 'validation_result') and entry.validation_result:
        if 'warnings' in entry.validation_result:
            warnings = entry.validation_result['warnings']
            if warnings:
                has_warnings = True
                print(f"\n⚠️  Warnings Found: {len(warnings)}")
                for i, w in enumerate(warnings, 1):
                    print(f"  {i}. {w}")

    if not has_warnings:
        print("\n⚠️  No warnings found for this entry")
        return None

    # Test 1: Try download WITHOUT force_download
    print("\n" + "-"*80)
    print("TEST 1: Download WITHOUT force_download (should show warnings)")
    print("-"*80)

    query1 = f"Execute download from queue using entry_id {entry_id}"
    print(f"\nQuery: {query1}")

    try:
        response1 = client.query(query1, stream=False)
        response_text = response1.get('response', '') if isinstance(response1, dict) else str(response1)

        print("\nResponse:")
        print(response_text)

        # Check for warning indicators
        test_results = {
            'warning_emoji_present': '⚠️' in response_text,
            'warning_keyword_present': 'warning' in response_text.lower(),
            'force_download_mentioned': 'force_download' in response_text.lower(),
            'response_preview': response_text[:300]
        }

        print("\n" + "-"*40)
        print("Response Analysis:")
        print("-"*40)
        for key, value in test_results.items():
            if key != 'response_preview':
                status = "✅" if value else "❌"
                print(f"  {status} {key}: {value}")

        return test_results

    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        return {'error': str(e)}

def test_fresh_dataset_with_warnings(client):
    """Test with a dataset known to have validation issues."""
    print("\n" + "="*80)
    print("TEST: Fresh Dataset Validation (Looking for Warnings)")
    print("="*80)

    # Try GSE60424 - known to have potential metadata issues
    test_datasets = ["GSE60424", "GSE10000"]

    for dataset_id in test_datasets:
        print(f"\n\nTrying {dataset_id}...")
        print("-"*80)

        query = f"Validate and add {dataset_id} to the download queue"
        print(f"Query: {query}")

        try:
            response = client.query(query, stream=False)
            response_text = response.get('response', '') if isinstance(response, dict) else str(response)

            print("\nResponse:")
            print(response_text[:500])

            # Check if warnings were reported
            if 'warning' in response_text.lower() or '⚠️' in response_text:
                print(f"\n✅ {dataset_id} triggered warnings - good candidate for testing!")

                # Find the entry
                for entry in client.data_manager.download_queue.list_entries():
                    if dataset_id in entry.dataset_id and entry.status == "pending":
                        print(f"\nEntry ID: {entry.entry_id}")
                        if hasattr(entry, 'validation_status'):
                            print(f"Validation Status: {entry.validation_status}")

                        # Now test warning display
                        return test_existing_entry_with_warnings(client, entry.entry_id)
            else:
                print(f"\n⚠️  {dataset_id} did not trigger warnings")

        except Exception as e:
            print(f"\n❌ Error with {dataset_id}: {e}")
            continue

    return None

def main():
    print("Initializing AgentClient...")
    client = AgentClient()

    test_report = {
        'test_type': 'warning_display',
        'tests_run': [],
        'summary': {}
    }

    # Test 1: Check existing entry GSE139555 (has warnings in validation_result)
    print("\n" + "="*80)
    print("SCENARIO 1: Test Existing Entry with Warnings (GSE139555)")
    print("="*80)

    gse139555_results = test_existing_entry_with_warnings(client, "queue_GSE139555_4bcbc050")
    if gse139555_results:
        test_report['tests_run'].append({
            'scenario': 'existing_entry_gse139555',
            'results': gse139555_results
        })

    # Test 2: Try fresh datasets that might trigger warnings
    print("\n\n" + "="*80)
    print("SCENARIO 2: Fresh Dataset Validation (Looking for Warnings)")
    print("="*80)

    fresh_results = test_fresh_dataset_with_warnings(client)
    if fresh_results:
        test_report['tests_run'].append({
            'scenario': 'fresh_dataset_validation',
            'results': fresh_results
        })

    # Summary
    print("\n\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    print(json.dumps(test_report, indent=2))

    # Determine success
    success = len(test_report['tests_run']) > 0
    if success:
        print("\n✅ Warning display feature tested")
    else:
        print("\n⚠️  Could not fully test warning display (no warnings triggered)")

    return test_report

if __name__ == "__main__":
    test_report = main()
    sys.exit(0 if test_report['tests_run'] else 1)
