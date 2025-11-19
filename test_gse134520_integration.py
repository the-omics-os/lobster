"""
Integration Test: GSE134520 Tiered Validation + Strategy Recommendation
Tests the complete validation and queue creation flow for GSE134520.
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
    """Run integration test for GSE134520."""

    print("="*80)
    print("INTEGRATION TEST: GSE134520 Tiered Validation")
    print("="*80)

    # Initialize client
    print("\nInitializing AgentClient...")
    client = AgentClient()

    # Phase 1: Validation + Queue Creation
    print("\n" + "="*80)
    print("PHASE 1: VALIDATION + QUEUE CREATION - GSE134520")
    print("="*80)

    query = "Validate and add GSE134520 to the download queue"
    print(f"\nQuery: {query}")
    print("\nExecuting query (this may take 30-60 seconds)...")

    try:
        response = client.query(query, stream=False)
        print("\nResponse:")
        print(response)
    except Exception as e:
        print(f"\n❌ ERROR during Phase 1: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Queue Entry Verification
    print("\n" + "="*80)
    print("PHASE 2: QUEUE ENTRY VERIFICATION")
    print("="*80)

    try:
        queue_status = client.data_manager.download_queue.list_entries(status="pending")
        print(f"\nTotal pending entries: {len(queue_status)}")

        # Find GSE134520 entry
        gse134520_entry = None
        for entry in queue_status:
            if "GSE134520" in entry.dataset_id:
                gse134520_entry = entry
                break

        if not gse134520_entry:
            print("\n❌ ERROR: GSE134520 not found in queue")
            print("\nAvailable entries:")
            for entry in queue_status[:5]:  # Show first 5
                print(f"  - {entry.dataset_id} ({entry.status})")
            sys.exit(1)

        print(f"\n✅ Queue entry found: {gse134520_entry.entry_id}")
        print(f"\nEntry Details:")
        print(f"  Dataset ID: {gse134520_entry.dataset_id}")
        print(f"  Status: {gse134520_entry.status}")

        # Initialize test results
        checks = {
            'validation_status_exists': False,
            'recommended_strategy_exists': False,
            'confidence_valid': False,
            'strategy_name_valid': False
        }

        # CHECK 1: validation_status
        if hasattr(gse134520_entry, 'validation_status'):
            print(f"  ✅ validation_status: {gse134520_entry.validation_status}")
            checks['validation_status_exists'] = True
        else:
            print(f"  ❌ validation_status: MISSING")

        # CHECK 2: recommended_strategy
        if hasattr(gse134520_entry, 'recommended_strategy') and gse134520_entry.recommended_strategy:
            strategy = gse134520_entry.recommended_strategy
            checks['recommended_strategy_exists'] = True

            print(f"  ✅ recommended_strategy:")
            print(f"     - strategy_name: {strategy.strategy_name}")
            print(f"     - confidence: {strategy.confidence:.2f}")
            print(f"     - rationale: {strategy.rationale[:100]}...")

            # Verify strategy makes sense
            if strategy.confidence >= 0.50 and strategy.confidence <= 1.0:
                print(f"     ✅ Confidence score in valid range (0.50-1.0)")
                checks['confidence_valid'] = True
            else:
                print(f"     ❌ Confidence score out of range: {strategy.confidence}")

            valid_strategies = ['H5_FIRST', 'MATRIX_FIRST', 'SAMPLES_FIRST',
                              'SUPPLEMENTARY_FIRST', 'RAW_FIRST', 'AUTO']
            if strategy.strategy_name in valid_strategies:
                print(f"     ✅ Valid strategy name")
                checks['strategy_name_valid'] = True
            else:
                print(f"     ❌ Invalid strategy name: {strategy.strategy_name}")
        else:
            print(f"  ❌ recommended_strategy: MISSING or None")

        # Final Summary
        print(f"\n" + "="*80)
        print("SUMMARY - GSE134520")
        print("="*80)

        test_passed = all(checks.values())

        summary = {
            'dataset_id': gse134520_entry.dataset_id,
            'validation_status': str(gse134520_entry.validation_status) if hasattr(gse134520_entry, 'validation_status') else 'MISSING',
            'recommended_strategy': gse134520_entry.recommended_strategy.strategy_name if gse134520_entry.recommended_strategy else 'None',
            'confidence': float(gse134520_entry.recommended_strategy.confidence) if gse134520_entry.recommended_strategy else None,
            'checks': checks,
            'test_passed': test_passed
        }

        print(json.dumps(summary, indent=2))

        if test_passed:
            print("\n" + "="*80)
            print("✅ ALL TESTS PASSED")
            print("="*80)
            sys.exit(0)
        else:
            print("\n" + "="*80)
            print("❌ TESTS FAILED")
            print("="*80)
            failed_checks = [k for k, v in checks.items() if not v]
            print(f"Failed checks: {', '.join(failed_checks)}")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ ERROR during Phase 2: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
