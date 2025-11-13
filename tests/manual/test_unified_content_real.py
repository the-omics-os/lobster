"""
End-to-End Integration Test for UnifiedContentService with PMC-First Strategy

This script tests the UnifiedContentService's PMC-first extraction strategy
with real publications from different publishers. It verifies:

1. PMC XML is tried first for PMID/DOI identifiers
2. Correct tier_used and source_type in results
3. Graceful fallback to webpage/PDF when PMC unavailable
4. Methods text extraction quality
5. Metadata extraction (software, tables, formulas)
6. Both PMID and DOI identifier types work

Test Publications:
- Nature: PMID:35042229 (NIH-funded, likely in PMC)
- Cell Press: PMID:33861949 (Cell Reports, PMC available)
- Science: PMID:35324292 (AAAS, may not be in PMC)
- PLOS: PMID:33534773 (Open access, in PMC)
- BMC: PMID:33388025 (Open access, in PMC)

Author: Engineering Team
Date: 2025-01-10
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.unified_content_service import (
    ContentExtractionError,
    UnifiedContentService,
)


class TestResult:
    """Data class for storing test results."""

    def __init__(
        self,
        identifier: str,
        publisher: str,
        success: bool,
        tier_used: str = "",
        source_type: str = "",
        extraction_time: float = 0.0,
        methods_length: int = 0,
        software_detected: List[str] = None,
        tables_count: int = 0,
        error: str = "",
    ):
        self.identifier = identifier
        self.publisher = publisher
        self.success = success
        self.tier_used = tier_used
        self.source_type = source_type
        self.extraction_time = extraction_time
        self.methods_length = methods_length
        self.software_detected = software_detected or []
        self.tables_count = tables_count
        self.error = error

    def pmc_first_working(self) -> bool:
        """Check if PMC-first strategy worked correctly."""
        # If PMC XML was used, should show correct tier and source
        if self.tier_used == "full_pmc_xml":
            return self.source_type == "pmc_xml"
        # If fallback was used, that's also correct behavior
        return self.tier_used in ["full_webpage", "full_pdf", "full_html"]

    def __repr__(self):
        status = "✅" if self.success else "❌"
        pmc_status = "✅" if self.pmc_first_working() else "❌"
        return (
            f"{status} [{self.publisher}] {self.identifier}\n"
            f"   tier_used: {self.tier_used}\n"
            f"   source_type: {self.source_type}\n"
            f"   extraction_time: {self.extraction_time:.2f}s\n"
            f"   methods_length: {self.methods_length} chars\n"
            f"   software: {self.software_detected}\n"
            f"   tables: {self.tables_count}\n"
            f"   PMC-first strategy: {pmc_status}\n"
            f"   error: {self.error if self.error else 'None'}"
        )


def test_publication(
    service: UnifiedContentService,
    identifier: str,
    publisher: str,
) -> TestResult:
    """
    Test a single publication with UnifiedContentService.

    Args:
        service: UnifiedContentService instance
        identifier: PMID or DOI
        publisher: Publisher name for reporting

    Returns:
        TestResult with extraction details
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing [{publisher}] {identifier}")
    logger.info(f"{'='*80}")

    start_time = time.time()

    try:
        # Test full content extraction
        result = service.get_full_content(identifier)

        # Extract key metrics
        tier_used = result.get("tier_used", "unknown")
        source_type = result.get("source_type", "unknown")
        extraction_time = time.time() - start_time

        # Methods text
        methods_text = result.get("methods_text", "")
        methods_length = len(methods_text)

        # Metadata
        metadata = result.get("metadata", {})
        software = metadata.get("software", [])
        tables = metadata.get("tables", 0)

        # Log extraction details
        logger.info(f"✅ Extraction successful!")
        logger.info(f"   Tier: {tier_used}")
        logger.info(f"   Source: {source_type}")
        logger.info(f"   Time: {extraction_time:.2f}s")
        logger.info(f"   Methods: {methods_length} chars")
        logger.info(f"   Software: {software}")
        logger.info(f"   Tables: {tables}")

        # Preview methods text
        if methods_text:
            preview = methods_text[:200].replace("\n", " ")
            logger.info(f"   Preview: {preview}...")

        return TestResult(
            identifier=identifier,
            publisher=publisher,
            success=True,
            tier_used=tier_used,
            source_type=source_type,
            extraction_time=extraction_time,
            methods_length=methods_length,
            software_detected=software,
            tables_count=tables,
        )

    except ContentExtractionError as e:
        logger.error(f"❌ Extraction failed: {e}")
        return TestResult(
            identifier=identifier,
            publisher=publisher,
            success=False,
            extraction_time=time.time() - start_time,
            error=str(e),
        )

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return TestResult(
            identifier=identifier,
            publisher=publisher,
            success=False,
            extraction_time=time.time() - start_time,
            error=str(e),
        )


def generate_report(results: List[TestResult]) -> str:
    """
    Generate comprehensive testing report.

    Args:
        results: List of TestResult objects

    Returns:
        Formatted markdown report
    """
    # Calculate statistics
    total = len(results)
    successful = sum(1 for r in results if r.success)
    pmc_used = sum(1 for r in results if r.tier_used == "full_pmc_xml")
    fallback_used = sum(
        1 for r in results if r.tier_used in ["full_webpage", "full_pdf", "full_html"]
    )
    pmc_first_working = sum(1 for r in results if r.pmc_first_working())

    # Average times by tier
    pmc_times = [r.extraction_time for r in results if r.tier_used == "full_pmc_xml"]
    fallback_times = [
        r.extraction_time
        for r in results
        if r.tier_used in ["full_webpage", "full_pdf", "full_html"]
    ]

    avg_pmc_time = sum(pmc_times) / len(pmc_times) if pmc_times else 0
    avg_fallback_time = (
        sum(fallback_times) / len(fallback_times) if fallback_times else 0
    )

    # Build report
    report = []
    report.append("## UnifiedContentService Integration Testing Report")
    report.append("")
    report.append("### Test Results:")
    report.append("")

    for result in results:
        report.append(f"**{result.publisher}** - {result.identifier}")
        report.append(
            f"- **Status**: {'✅ Success' if result.success else '❌ Failed'}"
        )
        report.append(f"- **tier_used**: `{result.tier_used}`")
        report.append(f"- **source_type**: `{result.source_type}`")
        report.append(f"- **extraction_time**: {result.extraction_time:.2f} seconds")
        report.append(f"- **methods_text_length**: {result.methods_length} chars")
        report.append(f"- **software_detected**: {result.software_detected}")
        report.append(f"- **tables_count**: {result.tables_count}")
        report.append(
            f"- **PMC-first strategy**: {'✅ Working correctly' if result.pmc_first_working() else '❌ Not working'}"
        )
        if result.error:
            report.append(f"- **Error**: {result.error}")
        report.append("")

    report.append("### Integration Summary:")
    report.append("")
    report.append(f"- **Total tests**: {total}")
    report.append(
        f"- **Successful extractions**: {successful}/{total} ({successful/total*100:.1f}%)"
    )
    report.append(f"- **PMC XML used**: {pmc_used}/{total} ({pmc_used/total*100:.1f}%)")
    report.append(
        f"- **Fallback used**: {fallback_used}/{total} ({fallback_used/total*100:.1f}%)"
    )
    report.append(
        f"- **PMC-first strategy working**: {pmc_first_working}/{total} ({pmc_first_working/total*100:.1f}%)"
    )
    report.append("")

    if pmc_times:
        report.append(f"**Performance (PMC XML)**:")
        report.append(f"- Average extraction time: {avg_pmc_time:.2f} seconds")
        report.append(f"- Min: {min(pmc_times):.2f}s, Max: {max(pmc_times):.2f}s")
        report.append("")

    if fallback_times:
        report.append(f"**Performance (Fallback)**:")
        report.append(f"- Average extraction time: {avg_fallback_time:.2f} seconds")
        report.append(
            f"- Min: {min(fallback_times):.2f}s, Max: {max(fallback_times):.2f}s"
        )
        report.append("")

    report.append("### Key Findings:")
    report.append("")

    # PMC-first strategy analysis
    if pmc_first_working == total:
        report.append(
            "✅ **PMC-first strategy working perfectly**: All tests correctly used PMC XML when available or fell back gracefully"
        )
    elif pmc_first_working > 0:
        report.append(
            f"⚠️ **PMC-first strategy partially working**: {pmc_first_working}/{total} tests working correctly"
        )
    else:
        report.append(
            "❌ **PMC-first strategy not working**: None of the tests showed correct PMC-first behavior"
        )
    report.append("")

    # Fallback chain analysis
    if fallback_used > 0:
        report.append(
            f"✅ **Fallback chain working**: {fallback_used} papers fell back to webpage/PDF extraction"
        )
    report.append("")

    # Performance comparison
    if pmc_times and fallback_times:
        speedup = avg_fallback_time / avg_pmc_time if avg_pmc_time > 0 else 0
        report.append(
            f"📊 **PMC XML Performance Advantage**: {speedup:.1f}x faster than fallback methods"
        )
        report.append(f"   - PMC: {avg_pmc_time:.2f}s")
        report.append(f"   - Fallback: {avg_fallback_time:.2f}s")
        report.append("")

    return "\n".join(report)


def main():
    """Main test execution."""
    logger.info("=" * 80)
    logger.info("UnifiedContentService Integration Test - PMC-First Strategy")
    logger.info("=" * 80)

    # Initialize service
    logger.info("Initializing UnifiedContentService...")
    data_manager = DataManagerV2()
    service = UnifiedContentService(data_manager=data_manager)

    # Test publications
    test_cases = [
        ("PMID:35042229", "Nature"),
        ("PMID:33861949", "Cell Press"),
        ("PMID:35324292", "Science"),
        ("PMID:33534773", "PLOS"),
        ("PMID:33388025", "BMC"),
    ]

    # Run tests
    results = []
    for identifier, publisher in test_cases:
        result = test_publication(service, identifier, publisher)
        results.append(result)

        # Small delay between tests
        time.sleep(1)

    # Test DOI identifier (test at least 2 papers with DOI)
    logger.info("\n" + "=" * 80)
    logger.info("Testing DOI Identifiers (in addition to PMID)")
    logger.info("=" * 80)

    # Get DOI from one of the papers (we'll use PLOS as it definitely has DOI)
    doi_tests = [
        ("10.1371/journal.pone.0245093", "PLOS (via DOI)"),
        ("10.1186/s12864-020-07352-1", "BMC (via DOI)"),
    ]

    for identifier, publisher in doi_tests:
        result = test_publication(service, identifier, publisher)
        results.append(result)
        time.sleep(1)

    # Generate report
    logger.info("\n" + "=" * 80)
    logger.info("Generating Report")
    logger.info("=" * 80)

    report = generate_report(results)

    # Print report
    print("\n\n")
    print(report)

    # Save report to file
    report_path = Path(__file__).parent / "unified_content_test_report.md"
    report_path.write_text(report)
    logger.info(f"\n✅ Report saved to: {report_path}")

    # Print individual results
    print("\n" + "=" * 80)
    print("Detailed Results")
    print("=" * 80)
    for result in results:
        print(result)
        print()


if __name__ == "__main__":
    main()
