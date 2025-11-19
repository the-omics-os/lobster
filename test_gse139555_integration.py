"""
Integration Test: GSE139555 Validation + Strategy Recommendation
Tests tiered validation system and strategy recommendation for a real GEO dataset.
"""

import os
import json
import sys

# Set environment variables
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', '')
os.environ['AWS_BEDROCK_ACCESS_KEY'] = os.getenv('AWS_BEDROCK_ACCESS_KEY', '')
os.environ['AWS_BEDROCK_SECRET_ACCESS_KEY'] = os.getenv('AWS_BEDROCK_SECRET_ACCESS_KEY', '')

from lobster.core.client import AgentClient

def main():
    print("\n" + "="*80)
    print("INTEGRATION TEST: GSE139555")
    print("Testing: Tiered Validation + Strategy Recommendation System")
    print("="*80 + "\n")

    # Initialize client
    print("Initializing AgentClient...")
    client = AgentClient()
    print("✅ Client initialized\n")

    # =========================================================================
    # PHASE 1: VALIDATION + QUEUE CREATION
    # =========================================================================
    print("="*80)
    print("PHASE 1: VALIDATION + QUEUE CREATION - GSE139555")
    print("="*80)

    query = "Validate and add GSE139555 to the download queue"
    print(f"Query: {query}\n")

    try:
        response = client.query(query, stream=False)
        print("Response:")
        print(response)
        print("\n✅ Phase 1 completed successfully\n")
    except Exception as e:
        print(f"❌ Phase 1 failed with error: {e}")
        sys.exit(1)

    # =========================================================================
    # PHASE 2: QUEUE ENTRY VERIFICATION
    # =========================================================================
    print("="*80)
    print("PHASE 2: QUEUE ENTRY VERIFICATION")
    print("="*80 + "\n")

    # Get queue status
    queue_status = client.data_manager.download_queue.list_entries(status="pending")
    print(f"Total pending entries in queue: {len(queue_status)}\n")

    # Find GSE139555 entry
    gse139555_entry = None
    for entry in queue_status:
        if "GSE139555" in entry.dataset_id:
            gse139555_entry = entry
            break

    # Initialize test results
    test_results = {
        'queue_entry_created': False,
        'validation_status_exists': False,
        'recommended_strategy_exists': False,
        'confidence_score_valid': False,
        'urls_extracted': False,
        'all_checks_passed': False
    }

    if not gse139555_entry:
        print("❌ ERROR: GSE139555 not found in queue")
        print("\nAvailable entries:")
        for entry in queue_status:
            print(f"  - {entry.dataset_id} (status: {entry.status})")

        # Print final report
        print_test_report(test_results)
        sys.exit(1)

    print(f"✅ Queue entry found: {gse139555_entry.entry_id}\n")
    test_results['queue_entry_created'] = True

    print("Entry Details:")
    print(f"  Dataset ID: {gse139555_entry.dataset_id}")
    print(f"  Status: {gse139555_entry.status}")
    print(f"  Database: {gse139555_entry.database}")

    # =========================================================================
    # CHECK 1: validation_status
    # =========================================================================
    print("\n" + "-"*80)
    print("CHECK 1: validation_status")
    print("-"*80)
    if hasattr(gse139555_entry, 'validation_status'):
        print(f"  ✅ validation_status exists: {gse139555_entry.validation_status}")
        test_results['validation_status_exists'] = True
    else:
        print(f"  ❌ validation_status: MISSING")

    # =========================================================================
    # CHECK 2: recommended_strategy
    # =========================================================================
    print("\n" + "-"*80)
    print("CHECK 2: recommended_strategy")
    print("-"*80)
    if hasattr(gse139555_entry, 'recommended_strategy') and gse139555_entry.recommended_strategy:
        strategy = gse139555_entry.recommended_strategy
        print(f"  ✅ recommended_strategy exists:")
        print(f"     - strategy_name: {strategy.strategy_name}")
        print(f"     - confidence: {strategy.confidence:.2f}")
        print(f"     - rationale: {strategy.rationale[:150]}...")
        print(f"     - concatenation_strategy: {strategy.concatenation_strategy}")
        test_results['recommended_strategy_exists'] = True

        # Check confidence score
        if 0.50 <= strategy.confidence <= 0.95:
            print(f"     ✅ Confidence score in valid range (0.50-0.95)")
            test_results['confidence_score_valid'] = True
        else:
            print(f"     ⚠️  Confidence score outside expected range: {strategy.confidence:.2f}")
    else:
        print(f"  ❌ recommended_strategy: MISSING or None")

    # =========================================================================
    # CHECK 3: URLs populated
    # =========================================================================
    print("\n" + "-"*80)
    print("CHECK 3: URLs Extraction")
    print("-"*80)
    print(f"  URLs:")

    urls_extracted = False
    if gse139555_entry.matrix_url:
        print(f"     ✅ matrix_url: Present")
        urls_extracted = True
    else:
        print(f"     - matrix_url: Not available")

    if gse139555_entry.h5_url:
        print(f"     ✅ h5_url: Present")
        urls_extracted = True
    else:
        print(f"     - h5_url: Not available")

    if gse139555_entry.raw_urls and len(gse139555_entry.raw_urls) > 0:
        print(f"     ✅ raw_urls: {len(gse139555_entry.raw_urls)} file(s)")
        urls_extracted = True
    else:
        print(f"     - raw_urls: None")

    if gse139555_entry.supplementary_urls and len(gse139555_entry.supplementary_urls) > 0:
        print(f"     ✅ supplementary_urls: {len(gse139555_entry.supplementary_urls)} file(s)")
    else:
        print(f"     - supplementary_urls: None")

    test_results['urls_extracted'] = urls_extracted

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY - GSE139555")
    print("="*80)

    summary = {
        'dataset_id': gse139555_entry.dataset_id,
        'validation_status': str(gse139555_entry.validation_status) if hasattr(gse139555_entry, 'validation_status') else 'MISSING',
        'recommended_strategy': gse139555_entry.recommended_strategy.strategy_name if gse139555_entry.recommended_strategy else 'None',
        'confidence': gse139555_entry.recommended_strategy.confidence if gse139555_entry.recommended_strategy else 'None',
        'concatenation_strategy': gse139555_entry.recommended_strategy.concatenation_strategy if gse139555_entry.recommended_strategy else 'None',
        'urls_available': {
            'matrix': bool(gse139555_entry.matrix_url),
            'h5': bool(gse139555_entry.h5_url),
            'raw': len(gse139555_entry.raw_urls) if gse139555_entry.raw_urls else 0,
            'supplementary': len(gse139555_entry.supplementary_urls) if gse139555_entry.supplementary_urls else 0
        }
    }

    print(json.dumps(summary, indent=2))

    # =========================================================================
    # FINAL TEST REPORT
    # =========================================================================
    test_results['all_checks_passed'] = all([
        test_results['queue_entry_created'],
        test_results['validation_status_exists'],
        test_results['recommended_strategy_exists'],
        test_results['confidence_score_valid'],
        test_results['urls_extracted']
    ])

    print_test_report(test_results)

    # Exit with appropriate code
    if test_results['all_checks_passed']:
        sys.exit(0)
    else:
        sys.exit(1)


def print_test_report(results):
    """Print formatted test report."""
    print("\n" + "="*80)
    print("TEST REPORT")
    print("="*80)

    checks = [
        ('Queue Entry Created', results['queue_entry_created']),
        ('Validation Status Exists', results['validation_status_exists']),
        ('Recommended Strategy Exists', results['recommended_strategy_exists']),
        ('Confidence Score Valid (0.50-0.95)', results['confidence_score_valid']),
        ('URLs Extracted', results['urls_extracted']),
    ]

    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {check_name}")

    print("\n" + "-"*80)
    if results['all_checks_passed']:
        print("  🎉 ALL TESTS PASSED")
    else:
        passed_count = sum(1 for _, passed in checks if passed)
        total_count = len(checks)
        print(f"  ⚠️  {passed_count}/{total_count} TESTS PASSED")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
