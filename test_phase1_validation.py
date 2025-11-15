"""
Phase 1 SRA Provider Validation Test Suite
Tests all critical paths with proper delays to avoid rate limiting
"""
import time
import sys
from pathlib import Path

# Add lobster to path
sys.path.insert(0, str(Path(__file__).parent))

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.sra_provider import SRAProvider, SRAProviderConfig


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def print_test(test_name, status, details=""):
    """Print test result"""
    status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    print(f"{status_icon} {test_name}: {status}")
    if details:
        print(f"   {details}")


def main():
    """Run comprehensive Phase 1 validation tests"""
    print_section("Phase 1 SRA Provider Validation - Comprehensive Test Suite")

    # Initialize provider
    print("Initializing SRA Provider...")
    dm = DataManagerV2()
    config = SRAProviderConfig(max_results=20)
    provider = SRAProvider(dm, config)
    print("✅ Provider initialized successfully\n")

    test_results = {
        "passed": 0,
        "failed": 0,
        "skipped": 0
    }

    # ============================================================================
    # TEST 1: Known GEUVADIS Study (SRP033351)
    # ============================================================================
    print_section("TEST 1: Known GEUVADIS Study Accession (SRP033351)")
    try:
        result = provider.search_publications("SRP033351", max_results=3)

        # Validate required fields
        checks = {
            "Contains 'SRA Database Search Results' header": "## 🧬 SRA Database Search Results" in result,
            "Shows query": "SRP033351" in result,
            "Has study_accession": "study_accession" in result.lower() or "SRP" in result,
            "Has organism": "organism" in result.lower() or "sapiens" in result.lower(),
            "Has library_strategy": "strategy" in result.lower() or "RNA-Seq" in result or "RNA" in result,
            "Has clickable NCBI link": "https://www.ncbi.nlm.nih.gov/sra/" in result,
            "No pandas NA errors": "<NA>" not in result,
            "No 'Unknown' accessions": result.count("Unknown") == 0,
        }

        all_passed = all(checks.values())
        print_test("Test 1: GEUVADIS SRP033351", "PASS" if all_passed else "FAIL")

        if not all_passed:
            for check_name, passed in checks.items():
                if not passed:
                    print(f"   ❌ {check_name}")

        print(f"\n📄 Sample output:\n{result[:500]}...\n")

        if all_passed:
            test_results["passed"] += 1
        else:
            test_results["failed"] += 1

    except Exception as e:
        print_test("Test 1: GEUVADIS SRP033351", "FAIL", str(e))
        test_results["failed"] += 1

    time.sleep(2)  # Rate limit protection

    # ============================================================================
    # TEST 2: Simple Keyword Search
    # ============================================================================
    print_section("TEST 2: Simple Keyword Search (cancer)")
    try:
        result = provider.search_publications("cancer", max_results=3)

        checks = {
            "No crash": True,
            "Returns results": len(result) > 100,  # Should have some content
            "Has header": "SRA Database Search Results" in result,
            "Has accessions": "SRP" in result or "SRR" in result or "SRX" in result,
        }

        all_passed = all(checks.values())
        print_test("Test 2: Simple keyword 'cancer'", "PASS" if all_passed else "FAIL")

        print(f"\n📄 Result length: {len(result)} characters\n")
        print(f"Sample output:\n{result[:300]}...\n")

        if all_passed:
            test_results["passed"] += 1
        else:
            test_results["failed"] += 1

    except Exception as e:
        print_test("Test 2: Simple keyword", "FAIL", str(e))
        test_results["failed"] += 1

    time.sleep(2)

    # ============================================================================
    # TEST 3: Keyword + Organism Filter
    # ============================================================================
    print_section("TEST 3: Keyword + Organism Filter (microbiome + Homo sapiens)")
    try:
        result = provider.search_publications(
            "microbiome",
            max_results=3,
            filters={"organism": "Homo sapiens"}
        )

        checks = {
            "No crash": True,
            "Has results": len(result) > 100,
            "Shows filters applied": "Filters:" in result or "organism" in result.lower(),
            "Has organism field": "Organism" in result or "sapiens" in result,
        }

        all_passed = all(checks.values())
        print_test("Test 3: Keyword + organism filter", "PASS" if all_passed else "FAIL")

        print(f"\n📄 Sample output:\n{result[:400]}...\n")

        if all_passed:
            test_results["passed"] += 1
        else:
            test_results["failed"] += 1

    except Exception as e:
        print_test("Test 3: Keyword + filter", "FAIL", str(e))
        test_results["failed"] += 1

    time.sleep(2)

    # ============================================================================
    # TEST 4: Invalid Accession Handling
    # ============================================================================
    print_section("TEST 4: Invalid Accession Graceful Handling (SRP999999999)")
    try:
        result = provider.search_publications("SRP999999999", max_results=3)

        checks = {
            "No crash": True,
            "Has 'No results' message": "No" in result and ("Results Found" in result or "metadata found" in result),
            "Not empty": len(result) > 50,
        }

        all_passed = all(checks.values())
        print_test("Test 4: Invalid accession handling", "PASS" if all_passed else "FAIL")

        print(f"\n📄 Result:\n{result}\n")

        if all_passed:
            test_results["passed"] += 1
        else:
            test_results["failed"] += 1

    except Exception as e:
        # Should NOT raise exception, should return graceful message
        print_test("Test 4: Invalid accession", "FAIL", f"Should not raise exception: {e}")
        test_results["failed"] += 1

    time.sleep(2)

    # ============================================================================
    # TEST 5: Agent OR Query (Critical for PubMed pattern)
    # ============================================================================
    print_section("TEST 5: Agent OR Query (microbiome OR metagenome)")
    try:
        result = provider.search_publications("microbiome OR metagenome", max_results=3)

        checks = {
            "No crash": True,
            "OR preserved in query display": "microbiome OR metagenome" in result or "OR" in result[:200],
            "Returns results": len(result) > 100,
        }

        all_passed = all(checks.values())
        print_test("Test 5: OR query", "PASS" if all_passed else "FAIL")

        print(f"\n📄 Sample output:\n{result[:400]}...\n")

        if all_passed:
            test_results["passed"] += 1
        else:
            test_results["failed"] += 1

    except Exception as e:
        print_test("Test 5: OR query", "FAIL", str(e))
        test_results["failed"] += 1

    time.sleep(2)

    # ============================================================================
    # TEST 6: Multiple Filters
    # ============================================================================
    print_section("TEST 6: Multiple Filters (gut microbiome + organism + strategy)")
    try:
        result = provider.search_publications(
            "gut microbiome",
            max_results=3,
            filters={"organism": "Homo sapiens", "strategy": "AMPLICON"}
        )

        checks = {
            "No crash": True,
            "Shows both filters": "organism" in result.lower() and "strategy" in result.lower(),
            "Has results or graceful no results": len(result) > 50,
        }

        all_passed = all(checks.values())
        print_test("Test 6: Multiple filters", "PASS" if all_passed else "FAIL")

        print(f"\n📄 Sample output:\n{result[:400]}...\n")

        if all_passed:
            test_results["passed"] += 1
        else:
            test_results["failed"] += 1

    except Exception as e:
        print_test("Test 6: Multiple filters", "FAIL", str(e))
        test_results["failed"] += 1

    time.sleep(2)

    # ============================================================================
    # TEST 7: Empty Results Query
    # ============================================================================
    print_section("TEST 7: Empty Results Query (zzz_nonexistent_12345)")
    try:
        result = provider.search_publications("zzz_nonexistent_12345", max_results=3)

        checks = {
            "No crash": True,
            "Has 'No results' message": "No" in result and "Results Found" in result or "No datasets found" in result,
            "Suggests alternatives": "Try:" in result or "Broadening" in result or "try" in result.lower(),
        }

        all_passed = all(checks.values())
        print_test("Test 7: Empty results", "PASS" if all_passed else "FAIL")

        print(f"\n📄 Result:\n{result[:500]}...\n")

        if all_passed:
            test_results["passed"] += 1
        else:
            test_results["failed"] += 1

    except Exception as e:
        print_test("Test 7: Empty results", "FAIL", f"Should not crash: {e}")
        test_results["failed"] += 1

    time.sleep(2)

    # ============================================================================
    # TEST 8: ENA Accession (European Nucleotide Archive)
    # ============================================================================
    print_section("TEST 8: ENA Accession (ERP000171)")
    try:
        result = provider.search_publications("ERP000171", max_results=3)

        checks = {
            "No crash": True,
            "Has results or graceful message": len(result) > 50,
            "No Python errors": "Error" not in result and "Exception" not in result,
        }

        all_passed = all(checks.values())
        print_test("Test 8: ENA accession", "PASS" if all_passed else "FAIL")

        print(f"\n📄 Sample output:\n{result[:400]}...\n")

        if all_passed:
            test_results["passed"] += 1
        else:
            test_results["failed"] += 1

    except Exception as e:
        print_test("Test 8: ENA accession", "FAIL", str(e))
        test_results["failed"] += 1

    time.sleep(2)

    # ============================================================================
    # TEST 9: Performance - Accession Search
    # ============================================================================
    print_section("TEST 9: Performance - Accession Search Speed")
    try:
        start = time.time()
        result = provider.search_publications("SRP033351", max_results=5)
        duration = time.time() - start

        target = 2.0
        acceptable = 3.0  # Allow some buffer for network

        if duration < acceptable:
            print_test("Test 9: Accession search performance", "PASS",
                      f"{duration:.2f}s (target: <{target}s)")
            test_results["passed"] += 1
        else:
            print_test("Test 9: Accession search performance", "⚠️ SLOW",
                      f"{duration:.2f}s (target: <{target}s, acceptable: <{acceptable}s)")
            if duration < 5.0:  # Still functional, just slow
                test_results["passed"] += 1
            else:
                test_results["failed"] += 1

    except Exception as e:
        print_test("Test 9: Performance", "FAIL", str(e))
        test_results["failed"] += 1

    time.sleep(2)

    # ============================================================================
    # TEST 10: Output Formatting Validation
    # ============================================================================
    print_section("TEST 10: Comprehensive Output Formatting")
    try:
        result = provider.search_publications("SRP033351", max_results=3)

        checks = {
            "Has SRA header": "🧬 SRA Database Search Results" in result,
            "Query shown": "Query" in result and "SRP033351" in result,
            "Has accession with link": "[SRP" in result and "](https://www.ncbi.nlm.nih.gov/sra/" in result,
            "Has organism metadata": "Organism" in result or "organism" in result.lower(),
            "Has strategy metadata": "Strategy" in result or "strategy" in result.lower(),
            "Has platform metadata": "Platform" in result or "platform" in result.lower(),
            "Has layout metadata": "Layout" in result or "layout" in result.lower(),
            "No pandas NA": "<NA>" not in result,
            "No @instrument_model": "@instrument_model" not in result,
            "No Unknown accessions": not ("**Accession**: Unknown" in result or "**Accession**: [Unknown]" in result),
        }

        all_passed = all(checks.values())
        print_test("Test 10: Output formatting", "PASS" if all_passed else "FAIL")

        if not all_passed:
            for check_name, passed in checks.items():
                if not passed:
                    print(f"   ❌ {check_name}")

        if all_passed:
            test_results["passed"] += 1
        else:
            test_results["failed"] += 1

    except Exception as e:
        print_test("Test 10: Output formatting", "FAIL", str(e))
        test_results["failed"] += 1

    # ============================================================================
    # FINAL REPORT
    # ============================================================================
    print_section("FINAL TEST REPORT")

    total_tests = test_results["passed"] + test_results["failed"] + test_results["skipped"]

    print(f"Total Tests Run: {total_tests}")
    print(f"✅ Passed: {test_results['passed']}")
    print(f"❌ Failed: {test_results['failed']}")
    print(f"⚠️  Skipped: {test_results['skipped']}")

    pass_rate = (test_results["passed"] / total_tests * 100) if total_tests > 0 else 0
    print(f"\nPass Rate: {pass_rate:.1f}%")

    # Production readiness assessment
    print_section("PRODUCTION READINESS ASSESSMENT")

    if test_results["failed"] == 0:
        print("✅ PRODUCTION READY")
        print("\nAll critical tests passed. Phase 1 implementation is functionally correct.")
        return 0
    elif test_results["failed"] <= 2 and pass_rate >= 80:
        print("⚠️  MOSTLY READY (Minor Issues)")
        print(f"\n{test_results['failed']} tests failed but pass rate is {pass_rate:.1f}%.")
        print("Review failed tests for minor issues.")
        return 1
    else:
        print("❌ NOT READY")
        print(f"\n{test_results['failed']} critical tests failed. Address issues before production.")
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
