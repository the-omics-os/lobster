"""
Fallback Chain Testing Script for UnifiedContentService

This script tests the PMC → Webpage → PDF fallback chain with real-world scenarios.

Test Scenarios:
1. PMC Available (should use PMC)
2. PMC Unavailable, Webpage Available (should use webpage)
3. Non-PMC PMID (should fall back)
4. DOI Resolution (should try PMC first)

Author: Engineering Team
Date: 2025-01-10
"""

import logging
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.unified_content_service import (
    ContentExtractionError,
    PaywalledError,
    UnifiedContentService,
)

# Configure logging to see fallback chain
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class FallbackChainTracer:
    """Traces the fallback chain execution for UnifiedContentService."""

    def __init__(self):
        """Initialize tracer with DataManager and service."""
        # Create DataManager with unique workspace to avoid cache conflicts
        self.data_manager = DataManagerV2(
            workspace_path=Path.home()
            / ".lobster"
            / "workspaces"
            / f"fallback_test_{int(time.time())}"
        )

        # Initialize service
        self.service = UnifiedContentService(data_manager=self.data_manager)

        self.results = []

    def trace_scenario(
        self, scenario_name: str, source: str, expected_path: str, **kwargs
    ) -> dict:
        """
        Trace a single fallback scenario.

        Args:
            scenario_name: Human-readable test name
            source: Publication source (PMID, DOI, or URL)
            expected_path: Expected fallback path (e.g., "PMC → None")
            **kwargs: Additional arguments for get_full_content()

        Returns:
            Dictionary with test results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"SCENARIO: {scenario_name}")
        logger.info(f"Input: {source}")
        logger.info(f"Expected Path: {expected_path}")
        logger.info(f"{'='*80}\n")

        start_time = time.time()
        result = {
            "scenario": scenario_name,
            "input": source,
            "expected_path": expected_path,
            "start_time": start_time,
        }

        try:
            # Execute extraction
            content = self.service.get_full_content(source, **kwargs)

            end_time = time.time()
            duration = end_time - start_time

            # Extract fallback path from result
            tier_used = content.get("tier_used", "unknown")
            source_type = content.get("source_type", "unknown")

            result.update(
                {
                    "success": True,
                    "tier_used": tier_used,
                    "source_type": source_type,
                    "duration": duration,
                    "methods_length": len(content.get("methods_text", "")),
                    "metadata": content.get("metadata", {}),
                    "error": None,
                }
            )

            logger.info(f"✅ SUCCESS")
            logger.info(f"   Tier Used: {tier_used}")
            logger.info(f"   Source Type: {source_type}")
            logger.info(f"   Duration: {duration:.2f}s")
            logger.info(f"   Methods Length: {result['methods_length']} chars")

        except PaywalledError as e:
            end_time = time.time()
            duration = end_time - start_time

            result.update(
                {
                    "success": False,
                    "tier_used": "paywalled",
                    "source_type": "paywalled",
                    "duration": duration,
                    "error": str(e),
                    "error_type": "PaywalledError",
                }
            )

            logger.warning(f"⚠️ PAYWALLED")
            logger.warning(f"   Error: {e}")
            logger.warning(f"   Duration: {duration:.2f}s")

        except ContentExtractionError as e:
            end_time = time.time()
            duration = end_time - start_time

            result.update(
                {
                    "success": False,
                    "tier_used": "failed",
                    "source_type": "failed",
                    "duration": duration,
                    "error": str(e),
                    "error_type": "ContentExtractionError",
                }
            )

            logger.error(f"❌ FAILED")
            logger.error(f"   Error: {e}")
            logger.error(f"   Duration: {duration:.2f}s")

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time

            result.update(
                {
                    "success": False,
                    "tier_used": "error",
                    "source_type": "error",
                    "duration": duration,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )

            logger.error(f"❌ UNEXPECTED ERROR")
            logger.error(f"   Error Type: {type(e).__name__}")
            logger.error(f"   Error: {e}")
            logger.error(f"   Duration: {duration:.2f}s")

        self.results.append(result)
        return result

    def generate_report(self) -> str:
        """Generate formatted report of all test results."""
        report = []
        report.append("\n" + "=" * 80)
        report.append("FALLBACK CHAIN TESTING REPORT")
        report.append("=" * 80 + "\n")

        for i, result in enumerate(self.results, 1):
            report.append(f"### Scenario {i}: {result['scenario']}")
            report.append(f"- **Input**: {result['input']}")
            report.append(f"- **Expected Path**: {result['expected_path']}")

            if result["success"]:
                report.append(f"- **Actual Path**: {result['tier_used']}")
                report.append(f"- **Source Type**: {result['source_type']}")
                report.append(f"- **Time**: {result['duration']:.2f} seconds")
                report.append(f"- **Methods Length**: {result['methods_length']} chars")
                report.append(f"- **Result**: ✅ SUCCESS")
            else:
                report.append(f"- **Tier Used**: {result['tier_used']}")
                report.append(f"- **Error Type**: {result['error_type']}")
                report.append(f"- **Time**: {result['duration']:.2f} seconds")
                report.append(f"- **Error**: {result['error']}")
                report.append(f"- **Result**: ❌ FAILED")

            report.append("")

        # Summary
        report.append("### Fallback Chain Summary:")

        success_count = sum(1 for r in self.results if r["success"])
        total_count = len(self.results)

        report.append(f"- **Total Tests**: {total_count}")
        report.append(f"- **Successful**: {success_count}")
        report.append(f"- **Failed**: {total_count - success_count}")

        # Analyze fallback paths
        pmc_used = sum(1 for r in self.results if "pmc" in r.get("tier_used", ""))
        webpage_used = sum(
            1 for r in self.results if "webpage" in r.get("tier_used", "")
        )
        pdf_used = sum(1 for r in self.results if "pdf" in r.get("tier_used", ""))

        report.append(f"- **PMC Used**: {pmc_used}")
        report.append(f"- **Webpage Used**: {webpage_used}")
        report.append(f"- **PDF Used**: {pdf_used}")

        # Performance analysis
        if self.results:
            durations = [r["duration"] for r in self.results if r["success"]]
            if durations:
                avg_duration = sum(durations) / len(durations)
                min_duration = min(durations)
                max_duration = max(durations)

                report.append(f"\n### Performance:")
                report.append(f"- **Average Time**: {avg_duration:.2f}s")
                report.append(f"- **Min Time**: {min_duration:.2f}s")
                report.append(f"- **Max Time**: {max_duration:.2f}s")

        # Assessment
        report.append(f"\n### Assessment:")

        # Check if PMC-first logic is working
        pmc_first_working = any(
            "pmc" in r.get("tier_used", "") for r in self.results if r["success"]
        )
        report.append(
            f"- **PMC-first logic working**: {'✅' if pmc_first_working else '❌'}"
        )

        # Check fallback transitions
        fallback_smooth = all(
            r["success"]
            or r["error_type"] in ["PaywalledError", "ContentExtractionError"]
            for r in self.results
        )
        report.append(
            f"- **Fallback transitions smooth**: {'✅' if fallback_smooth else '❌'}"
        )

        # Check error handling
        error_handling_graceful = all(
            r.get("error_type") != "Exception" for r in self.results if not r["success"]
        )
        report.append(
            f"- **Error handling graceful**: {'✅' if error_handling_graceful else '❌'}"
        )

        report.append("\n" + "=" * 80)

        return "\n".join(report)


def main():
    """Run all fallback chain test scenarios."""
    tracer = FallbackChainTracer()

    # Scenario 1: PMC Available (should use PMC)
    tracer.trace_scenario(
        scenario_name="PMC Available",
        source="PMID:35042229",
        expected_path="PMC ✅ → (no fallback needed)",
    )

    # Scenario 2: PMC Unavailable, Webpage Available (should use webpage)
    # Using a Nature article URL directly (bypasses PMC for URLs)
    tracer.trace_scenario(
        scenario_name="PMC Unavailable, Webpage Available",
        source="https://www.nature.com/articles/s41586-021-03852-1",
        expected_path="(PMC skipped for URL) → Webpage ✅",
    )

    # Scenario 3: Non-existent PMID (should error gracefully)
    tracer.trace_scenario(
        scenario_name="Non-existent PMID",
        source="PMID:99999999",
        expected_path="PMC ❌ → Fallback attempted → Error handling",
    )

    # Scenario 4: DOI Resolution (should try PMC first)
    tracer.trace_scenario(
        scenario_name="DOI with PMC",
        source="10.1038/s41586-021-03852-1",
        expected_path="PMC ✅ → (no fallback needed)",
    )

    # Scenario 5: bioRxiv preprint (should use PDF extraction)
    # Note: bioRxiv papers typically don't have PMC full text
    tracer.trace_scenario(
        scenario_name="bioRxiv Preprint (PDF)",
        source="https://www.biorxiv.org/content/10.1101/2024.01.23.576916v1",
        expected_path="(PMC skipped for URL) → Webpage → PDF ✅",
    )

    # Generate and print report
    report = tracer.generate_report()
    print(report)

    # Save report to file
    report_path = Path(__file__).parent / "fallback_chain_test_report.txt"
    report_path.write_text(report)
    logger.info(f"\n📄 Report saved to: {report_path}")


if __name__ == "__main__":
    main()
