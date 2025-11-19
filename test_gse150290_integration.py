"""
Integration Test: GSE150290 Validation + Strategy Recommendation System
Tests tiered validation, strategy recommendation, and warning display.
"""

import os
import json
import sys

# Set up environment
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', '')
os.environ['AWS_BEDROCK_ACCESS_KEY'] = os.getenv('AWS_BEDROCK_ACCESS_KEY', '')
os.environ['AWS_BEDROCK_SECRET_ACCESS_KEY'] = os.getenv('AWS_BEDROCK_SECRET_ACCESS_KEY', '')

from lobster.core.client import AgentClient

def main():
    test_report = {
        'dataset_id': 'GSE150290',
        'phase_1_queue_creation': {},
        'phase_2_queue_verification': {},
        'phase_3_warning_display': {}
    }

    # Initialize client
    print("Initializing AgentClient...")
    client = AgentClient()

    # ========================================================================
    # PHASE 1: VALIDATION + QUEUE CREATION
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 1: VALIDATION + QUEUE CREATION - GSE150290")
    print("="*80)

    query = "Validate and add GSE150290 to the download queue"
    print(f"\nQuery: {query}")

    try:
        response = client.query(query, stream=False)
        print("\nResponse:")
        if isinstance(response, dict):
            print(json.dumps(response, indent=2))
            response_text = response.get('response', str(response))
            test_report['phase_1_queue_creation']['response_preview'] = response_text[:200] if isinstance(response_text, str) else str(response_text)[:200]
        else:
            print(response)
            test_report['phase_1_queue_creation']['response_preview'] = str(response)[:200]
        test_report['phase_1_queue_creation']['success'] = True
    except Exception as e:
        print(f"\n❌ ERROR in Phase 1: {e}")
        test_report['phase_1_queue_creation']['success'] = False
        test_report['phase_1_queue_creation']['error'] = str(e)
        print(json.dumps(test_report, indent=2))
        return test_report

    # ========================================================================
    # PHASE 2: QUEUE ENTRY VERIFICATION
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 2: QUEUE ENTRY VERIFICATION")
    print("="*80)

    # Get queue status
    queue_status = client.data_manager.download_queue.list_entries(status="pending")

    # Find GSE150290 entry
    gse150290_entry = None
    for entry in queue_status:
        if "GSE150290" in entry.dataset_id:
            gse150290_entry = entry
            break

    if gse150290_entry:
        entry_id = gse150290_entry.entry_id
        print(f"\n✅ Queue entry found: {entry_id}")

        # Verification checks
        test_report['phase_2_queue_verification']['entry_found'] = True
        test_report['phase_2_queue_verification']['entry_id'] = entry_id

        # CHECK: validation_status
        has_validation_status = hasattr(gse150290_entry, 'validation_status')
        test_report['phase_2_queue_verification']['validation_status'] = {
            'exists': has_validation_status,
            'value': str(gse150290_entry.validation_status) if has_validation_status else None
        }
        print(f"  validation_status: {gse150290_entry.validation_status if has_validation_status else 'MISSING'}")

        # CHECK: recommended_strategy
        has_strategy = hasattr(gse150290_entry, 'recommended_strategy') and gse150290_entry.recommended_strategy
        if has_strategy:
            strategy = gse150290_entry.recommended_strategy
            test_report['phase_2_queue_verification']['recommended_strategy'] = {
                'exists': True,
                'strategy_name': strategy.strategy_name,
                'confidence': float(strategy.confidence),
                'rationale': strategy.rationale[:100]
            }
            print(f"  recommended_strategy: {strategy.strategy_name} ({strategy.confidence:.2f})")
            print(f"  rationale: {strategy.rationale[:80]}...")
        else:
            test_report['phase_2_queue_verification']['recommended_strategy'] = {'exists': False}
            print(f"  recommended_strategy: MISSING or None")

        # CHECK: Warnings
        if gse150290_entry.validation_result and 'warnings' in gse150290_entry.validation_result:
            warnings = gse150290_entry.validation_result['warnings']
            test_report['phase_2_queue_verification']['warnings'] = {
                'count': len(warnings),
                'examples': warnings[:3]
            }
            print(f"\n  Validation warnings ({len(warnings)}):")
            for w in warnings[:3]:
                print(f"    • {w}")
        else:
            test_report['phase_2_queue_verification']['warnings'] = {'count': 0}
            print("\n  No validation warnings")
    else:
        test_report['phase_2_queue_verification']['entry_found'] = False
        print("\n❌ ERROR: GSE150290 not found in queue")
        print(json.dumps(test_report, indent=2))
        return test_report

    # ========================================================================
    # PHASE 3: WARNING DISPLAY TEST
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 3: WARNING DISPLAY TEST")
    print("="*80)

    if gse150290_entry and hasattr(gse150290_entry, 'validation_status'):
        if str(gse150290_entry.validation_status) == 'validated_warnings':
            print("\n✅ Dataset has validation_status = validated_warnings")
            print("Testing data_expert warning display...")

            # Test download WITHOUT force_download (should show warnings)
            try:
                warning_query = f"Execute download from queue using entry_id {gse150290_entry.entry_id}"
                print(f"\nQuery: {warning_query}")
                warning_response = client.query(warning_query, stream=False)

                print("\ndata_expert Response (without force_download):")
                print(warning_response)

                # Check if warning message was displayed
                if '⚠️' in warning_response or 'warning' in warning_response.lower():
                    print("\n✅ data_expert correctly displayed warnings")
                    test_report['phase_3_warning_display']['warning_displayed'] = True

                    if 'force_download=True' in warning_response or 'force_download' in warning_response:
                        print("✅ force_download=True option provided")
                        test_report['phase_3_warning_display']['override_option_shown'] = True
                    else:
                        print("⚠️ force_download option not explicitly shown")
                        test_report['phase_3_warning_display']['override_option_shown'] = False
                else:
                    print("\n❌ Warning display not working as expected")
                    test_report['phase_3_warning_display']['warning_displayed'] = False
            except Exception as e:
                print(f"\n❌ Error testing warning display: {e}")
                test_report['phase_3_warning_display']['error'] = str(e)
        else:
            validation_status = str(gse150290_entry.validation_status)
            print(f"\nDataset has validation_status = {validation_status}")
            print("No warnings to display (clean validation)")
            test_report['phase_3_warning_display']['status'] = 'clean_validation'
            test_report['phase_3_warning_display']['validation_status'] = validation_status

    # ========================================================================
    # FINAL REPORT
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL TEST REPORT - GSE150290")
    print("="*80)

    # Calculate overall success
    test_report['all_phases_passed'] = all([
        test_report['phase_2_queue_verification'].get('entry_found', False),
        test_report['phase_2_queue_verification'].get('validation_status', {}).get('exists', False),
        test_report['phase_2_queue_verification'].get('recommended_strategy', {}).get('exists', False)
    ])

    print(json.dumps(test_report, indent=2))

    if test_report['all_phases_passed']:
        print("\n✅ ALL CORE CHECKS PASSED")
    else:
        print("\n❌ SOME CHECKS FAILED")

    return test_report

if __name__ == "__main__":
    test_report = main()
    sys.exit(0 if test_report.get('all_phases_passed', False) else 1)
