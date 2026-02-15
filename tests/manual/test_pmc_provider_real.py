"""
Real-world testing of PMCProvider with cancer publications.

Tests PMC full text extraction with actual high-impact publications from different
publishers to evaluate:
- PMC availability across publishers
- Extraction quality (methods, tables, software tools)
- Performance (< 2 seconds per extraction)
- Error handling for non-PMC papers

Created: 2025-01-10
"""

import time
from typing import Dict, List, Optional

from lobster.tools.providers.pmc_provider import (
    PMCNotAvailableError,
    PMCProvider,
    PMCProviderError,
)

# Test publications: High-impact cancer research from different publishers
TEST_PUBLICATIONS = [
    {
        "pmid": "35042229",
        "doi": "10.1038/s41586-021-03852-1",
        "publisher": "Nature",
        "title": "Pan-cancer analysis (expected PMC-available)",
    },
    {
        "pmid": "33861949",
        "doi": "10.1016/j.cell.2021.03.007",
        "publisher": "Cell Press",
        "title": "Cancer cell plasticity (expected PMC-available after embargo)",
    },
    {
        "pmid": "35324292",
        "doi": "10.1126/science.abf3066",
        "publisher": "Science",
        "title": "Cancer immunotherapy (expected PMC-available)",
    },
    {
        "pmid": "33534773",
        "doi": "10.1371/journal.pone.0245970",
        "publisher": "PLOS",
        "title": "Cancer biomarkers (open access - PMC available)",
    },
    {
        "pmid": "33388025",
        "doi": "10.1186/s12885-020-07729-w",
        "publisher": "BMC",
        "title": "Cancer genomics (open access - PMC available)",
    },
]


class TestResult:
    """Store test results for a single publication."""

    def __init__(self, pmid: str, publisher: str, title: str):
        self.pmid = pmid
        self.publisher = publisher
        self.title = title

        # Availability
        self.pmc_available = False
        self.pmc_id: Optional[str] = None

        # Extraction results
        self.success = False
        self.error_message: Optional[str] = None
        self.extraction_time: Optional[float] = None

        # Quality metrics
        self.full_text_chars = 0
        self.methods_chars = 0
        self.results_chars = 0
        self.discussion_chars = 0
        self.table_count = 0
        self.software_count = 0
        self.github_count = 0

        # Software tools found
        self.software_tools: List[str] = []

    def to_dict(self) -> Dict:
        """Convert to dictionary for reporting."""
        return {
            "pmid": self.pmid,
            "publisher": self.publisher,
            "title": self.title,
            "pmc_available": self.pmc_available,
            "pmc_id": self.pmc_id,
            "success": self.success,
            "error_message": self.error_message,
            "extraction_time": self.extraction_time,
            "full_text_chars": self.full_text_chars,
            "methods_chars": self.methods_chars,
            "results_chars": self.results_chars,
            "discussion_chars": self.discussion_chars,
            "table_count": self.table_count,
            "software_count": self.software_count,
            "github_count": self.github_count,
            "software_tools": self.software_tools,
        }


def test_publication(provider: PMCProvider, pub: Dict) -> TestResult:
    """
    Test PMC extraction for a single publication.

    Args:
        provider: PMCProvider instance
        pub: Publication metadata dict

    Returns:
        TestResult object with extraction results
    """
    result = TestResult(pub["pmid"], pub["publisher"], pub["title"])

    print(f"\n{'='*80}")
    print(f"Testing: {pub['publisher']} - PMID:{pub['pmid']}")
    print(f"Title: {pub['title']}")
    print(f"{'='*80}")

    # Step 1: Check PMC availability
    print("\n[1/2] Checking PMC availability...")
    try:
        pmc_id = provider.get_pmc_id(pub["pmid"])
        if pmc_id:
            result.pmc_available = True
            result.pmc_id = pmc_id
            print(f"✅ PMC available: PMC{pmc_id}")
        else:
            result.pmc_available = False
            print(f"❌ PMC not available")
            return result

    except Exception as e:
        result.error_message = f"Error checking PMC availability: {str(e)}"
        print(f"❌ Error: {result.error_message}")
        return result

    # Step 2: Extract full text (if PMC available)
    print("\n[2/2] Extracting full text...")
    try:
        start_time = time.time()
        full_text = provider.extract_full_text(pub["pmid"])
        extraction_time = time.time() - start_time

        result.success = True
        result.extraction_time = extraction_time

        # Quality metrics
        result.full_text_chars = len(full_text.full_text)
        result.methods_chars = len(full_text.methods_section)
        result.results_chars = len(full_text.results_section)
        result.discussion_chars = len(full_text.discussion_section)
        result.table_count = len(full_text.tables)
        result.software_count = len(full_text.software_tools)
        result.github_count = len(full_text.github_repos)
        result.software_tools = full_text.software_tools

        # Report
        print(f"✅ Extraction successful in {extraction_time:.2f}s")
        print(f"\nQuality Metrics:")
        print(f"  - Full text: {result.full_text_chars:,} chars")
        print(f"  - Methods section: {result.methods_chars:,} chars")
        print(f"  - Results section: {result.results_chars:,} chars")
        print(f"  - Discussion section: {result.discussion_chars:,} chars")
        print(f"  - Tables: {result.table_count}")
        print(f"  - Software tools: {result.software_count}")
        print(f"  - GitHub repos: {result.github_count}")

        if result.software_tools:
            print(f"\nSoftware Tools Detected:")
            for tool in result.software_tools[:10]:  # Limit to first 10
                print(f"  - {tool}")

        # Performance check
        if extraction_time > 2.0:
            print(
                f"\n⚠️  WARNING: Extraction took {extraction_time:.2f}s (target: < 2.0s)"
            )
        else:
            print(f"\n✅ Performance: {extraction_time:.2f}s (< 2.0s target)")

    except PMCNotAvailableError as e:
        result.error_message = str(e)
        print(f"❌ PMC not available: {result.error_message}")

    except PMCProviderError as e:
        result.error_message = str(e)
        print(f"❌ PMC provider error: {result.error_message}")

    except Exception as e:
        result.error_message = f"Unexpected error: {str(e)}"
        print(f"❌ Error: {result.error_message}")

    return result


def generate_report(results: List[TestResult]) -> str:
    """
    Generate comprehensive test report.

    Args:
        results: List of TestResult objects

    Returns:
        Markdown-formatted report
    """
    report = []
    report.append("# PMC Provider Direct Testing Report")
    report.append("")
    report.append(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Publications Tested: {len(results)}")
    report.append("")

    # Categorize results
    pmc_available = [r for r in results if r.pmc_available]
    pmc_not_available = [r for r in results if not r.pmc_available]
    successful_extractions = [r for r in results if r.success]
    failed_extractions = [r for r in pmc_available if not r.success]

    # Summary statistics
    report.append("## Executive Summary")
    report.append("")
    report.append(
        f"- **PMC Available:** {len(pmc_available)}/{len(results)} publications ({len(pmc_available)/len(results)*100:.0f}%)"
    )
    report.append(
        f"- **Successful Extractions:** {len(successful_extractions)}/{len(pmc_available)} ({len(successful_extractions)/len(pmc_available)*100:.0f}% of available)"
    )
    report.append(f"- **Failed Extractions:** {len(failed_extractions)}")
    report.append("")

    # PMC-Available Papers
    report.append("## PMC-Available Papers")
    report.append("")

    if pmc_available:
        for result in pmc_available:
            status = "✅" if result.success else "❌"
            report.append(f"### {status} [{result.publisher}] PMID:{result.pmid}")
            report.append("")
            report.append(f"**PMC ID:** PMC{result.pmc_id}")
            report.append("")

            if result.success:
                report.append(f"**Extraction Time:** {result.extraction_time:.2f}s")
                perf_status = "✅" if result.extraction_time < 2.0 else "⚠️"
                report.append(
                    f"**Performance Status:** {perf_status} ({'PASS' if result.extraction_time < 2.0 else 'SLOW'})"
                )
                report.append("")

                report.append("**Quality Metrics:**")
                report.append(f"- Full text: {result.full_text_chars:,} chars")
                report.append(f"- Methods section: {result.methods_chars:,} chars")
                report.append(f"- Results section: {result.results_chars:,} chars")
                report.append(
                    f"- Discussion section: {result.discussion_chars:,} chars"
                )
                report.append(f"- Tables extracted: {result.table_count}")
                report.append(f"- Software tools detected: {result.software_count}")
                report.append(f"- GitHub repos found: {result.github_count}")
                report.append("")

                if result.software_tools:
                    report.append("**Software Tools:**")
                    for tool in result.software_tools[:10]:
                        report.append(f"- {tool}")
                    if len(result.software_tools) > 10:
                        report.append(
                            f"- ... and {len(result.software_tools) - 10} more"
                        )
                    report.append("")

                # Quality assessment
                quality_issues = []
                if result.methods_chars == 0:
                    quality_issues.append("No methods section extracted")
                if result.table_count == 0:
                    quality_issues.append("No tables extracted")
                if result.software_count == 0:
                    quality_issues.append("No software tools detected")

                if quality_issues:
                    report.append("**Quality Issues:**")
                    for issue in quality_issues:
                        report.append(f"- ⚠️ {issue}")
                    report.append("")

            else:
                report.append(f"**Error:** {result.error_message}")
                report.append("")

    else:
        report.append("No PMC-available papers found.")
        report.append("")

    # Non-PMC Papers
    report.append("## Non-PMC Papers")
    report.append("")

    if pmc_not_available:
        for result in pmc_not_available:
            report.append(f"### [{result.publisher}] PMID:{result.pmid}")
            report.append("")
            report.append(
                f"**Status:** ❌ PMC not available (expected PMCNotAvailableError)"
            )
            if result.error_message:
                report.append(f"**Details:** {result.error_message}")
            report.append("")
    else:
        report.append("All tested papers are available in PMC.")
        report.append("")

    # Performance Summary
    report.append("## Performance Summary")
    report.append("")

    if successful_extractions:
        extraction_times = [r.extraction_time for r in successful_extractions]
        avg_time = sum(extraction_times) / len(extraction_times)
        min_time = min(extraction_times)
        max_time = max(extraction_times)

        report.append(f"- **Average extraction time:** {avg_time:.2f}s")
        report.append(f"- **Min extraction time:** {min_time:.2f}s")
        report.append(f"- **Max extraction time:** {max_time:.2f}s")
        report.append(
            f"- **Target compliance:** {sum(1 for t in extraction_times if t < 2.0)}/{len(extraction_times)} extractions < 2.0s"
        )
        report.append("")

        # Publisher comparison
        report.append("### Performance by Publisher")
        report.append("")
        report.append("| Publisher | Extraction Time | Status |")
        report.append("|-----------|----------------|--------|")
        for result in successful_extractions:
            status = "✅ PASS" if result.extraction_time < 2.0 else "⚠️ SLOW"
            report.append(
                f"| {result.publisher} | {result.extraction_time:.2f}s | {status} |"
            )
        report.append("")

    # Quality Summary
    report.append("## Quality Summary")
    report.append("")

    if successful_extractions:
        total_methods_chars = sum(r.methods_chars for r in successful_extractions)
        total_tables = sum(r.table_count for r in successful_extractions)
        total_software = sum(r.software_count for r in successful_extractions)

        report.append(f"- **Total methods content:** {total_methods_chars:,} chars")
        report.append(f"- **Total tables extracted:** {total_tables}")
        report.append(f"- **Total software tools detected:** {total_software}")
        report.append(
            f"- **Average methods per paper:** {total_methods_chars/len(successful_extractions):,.0f} chars"
        )
        report.append(
            f"- **Average tables per paper:** {total_tables/len(successful_extractions):.1f}"
        )
        report.append(
            f"- **Average software tools per paper:** {total_software/len(successful_extractions):.1f}"
        )
        report.append("")

    # Recommendations
    report.append("## Recommendations")
    report.append("")

    if len(pmc_available) / len(results) < 0.5:
        report.append(
            "- ⚠️ **Low PMC coverage:** Consider implementing PDF fallback for non-PMC papers"
        )

    if failed_extractions:
        report.append(
            f"- ⚠️ **Failed extractions:** {len(failed_extractions)} papers had errors - investigate error handling"
        )

    slow_extractions = [r for r in successful_extractions if r.extraction_time > 2.0]
    if slow_extractions:
        report.append(
            f"- ⚠️ **Slow extractions:** {len(slow_extractions)} papers exceeded 2.0s target - consider caching"
        )

    if not any(r.table_count > 0 for r in successful_extractions):
        report.append(
            "- ⚠️ **No tables extracted:** Table parsing may need improvement"
        )

    if not any(r.software_count > 0 for r in successful_extractions):
        report.append(
            "- ⚠️ **No software detected:** Software detection patterns may need expansion"
        )

    if not report[-1].startswith("-"):
        report.append("- ✅ **All tests passed:** PMC provider performing as expected")

    report.append("")

    return "\n".join(report)


def main():
    """Run PMC provider tests and generate report."""
    print("=" * 80)
    print("PMC Provider Real-World Testing")
    print("=" * 80)
    print(
        f"\nTesting {len(TEST_PUBLICATIONS)} cancer publications from different publishers"
    )
    print("\nObjectives:")
    print("  1. Verify PMC availability across publishers")
    print("  2. Test extraction quality (methods, tables, software)")
    print("  3. Validate performance (< 2.0s per extraction)")
    print("  4. Check error handling for non-PMC papers")

    # Initialize provider
    print("\nInitializing PMC Provider...")
    provider = PMCProvider()

    # Test each publication
    results = []
    for pub in TEST_PUBLICATIONS:
        result = test_publication(provider, pub)
        results.append(result)

    # Generate report
    print("\n" + "=" * 80)
    print("Generating Test Report")
    print("=" * 80)

    report = generate_report(results)
    print("\n" + report)

    # Save report to file
    report_path = (
        "/Users/tyo/GITHUB/omics-os/lobster/tests/manual/PMC_PROVIDER_TEST_REPORT.md"
    )
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n✅ Report saved to: {report_path}")

    # Return exit code based on results
    successful = [r for r in results if r.success]
    if len(successful) == 0:
        print("\n❌ FAILED: No successful extractions")
        return 1
    elif len(successful) < len([r for r in results if r.pmc_available]):
        print("\n⚠️  WARNING: Some extractions failed")
        return 0  # Still pass if some succeeded
    else:
        print("\n✅ SUCCESS: All PMC-available papers extracted successfully")
        return 0


if __name__ == "__main__":
    exit(main())
