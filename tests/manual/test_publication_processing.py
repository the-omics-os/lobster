#!/usr/bin/env python3
"""
Manual test suite for publication queue processing.

Tests content extraction from various publishers with smart request strategies
to identify optimal approaches for avoiding bot protection.

Usage:
    # Test with default RIS file
    python tests/manual/test_publication_processing.py

    # Test with custom RIS file
    python tests/manual/test_publication_processing.py --ris-file path/to/file.ris

    # Test with specific strategy
    python tests/manual/test_publication_processing.py --strategy stealth

    # Dry run (parse RIS and show queue, no requests)
    python tests/manual/test_publication_processing.py --dry-run

    # Test single entry by index
    python tests/manual/test_publication_processing.py --entry 0

Strategies:
    - default: No special handling
    - polite: Longer delays between requests (5-10s)
    - browser: Full browser headers + randomized User-Agent
    - stealth: Browser headers + random delays + referrer spoofing
    - auto: Use publisher-specific optimal strategies (default)
"""

import argparse
import random
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.ris_parser import RISParser
from lobster.core.schemas.publication_queue import PublicationQueueEntry
from lobster.services.publication_processing_service import PublicationProcessingService
from lobster.tools.content_access_service import ContentAccessService


# =============================================================================
# Request Strategy Definitions
# =============================================================================


@dataclass
class RequestStrategy:
    """Configuration for HTTP request behavior."""

    name: str
    min_delay: float = 0.0  # Minimum seconds between requests
    max_delay: float = 0.0  # Maximum seconds (random within range)
    user_agent: Optional[str] = None  # Custom User-Agent string
    headers: Dict[str, str] = field(default_factory=dict)  # Additional headers
    use_referrer: bool = False  # Add domain-based Referer header
    retry_on_403: bool = True  # Auto-retry with escalated strategy on 403
    max_retries: int = 2  # Maximum retry attempts


# Chrome on macOS User-Agent (common and less likely to be blocked)
CHROME_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Firefox User-Agent (alternative)
FIREFOX_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) "
    "Gecko/20100101 Firefox/121.0"
)

# Standard browser Accept headers
BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

STEALTH_HEADERS = {
    **BROWSER_HEADERS,
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}


STRATEGIES: Dict[str, RequestStrategy] = {
    "default": RequestStrategy(
        name="default",
        min_delay=0.5,
        max_delay=1.0,
    ),
    "polite": RequestStrategy(
        name="polite",
        min_delay=3.0,
        max_delay=7.0,
        user_agent=CHROME_USER_AGENT,
    ),
    "browser": RequestStrategy(
        name="browser",
        min_delay=1.0,
        max_delay=3.0,
        user_agent=CHROME_USER_AGENT,
        headers=BROWSER_HEADERS,
    ),
    "stealth": RequestStrategy(
        name="stealth",
        min_delay=5.0,
        max_delay=15.0,
        user_agent=CHROME_USER_AGENT,
        headers=STEALTH_HEADERS,
        use_referrer=True,
    ),
}


# =============================================================================
# Publisher-Specific Configurations
# =============================================================================


@dataclass
class PublisherConfig:
    """Configuration for a specific publisher."""

    domain: str
    strategy: str
    notes: str
    prefer_pdf: bool = False  # Prefer PDF URL over fulltext
    skip_fulltext: bool = False  # Skip fulltext attempt entirely


PUBLISHER_CONFIGS: Dict[str, PublisherConfig] = {
    "cell.com": PublisherConfig(
        domain="cell.com",
        strategy="stealth",
        notes="Aggressive bot detection, Cell Press uses Cloudflare",
    ),
    "frontiersin.org": PublisherConfig(
        domain="frontiersin.org",
        strategy="polite",
        notes="PDF links work well, open access friendly",
        prefer_pdf=True,
    ),
    "nature.com": PublisherConfig(
        domain="nature.com",
        strategy="browser",
        notes="Moderate protection, Nature/Springer",
    ),
    "wiley.com": PublisherConfig(
        domain="wiley.com",
        strategy="stealth",
        notes="TDM API unreliable, may require institutional access",
    ),
    "onlinelibrary.wiley.com": PublisherConfig(
        domain="onlinelibrary.wiley.com",
        strategy="stealth",
        notes="Wiley Online Library, same as wiley.com",
    ),
    "springer.com": PublisherConfig(
        domain="springer.com",
        strategy="polite",
        notes="Generally accessible with delays",
    ),
    "link.springer.com": PublisherConfig(
        domain="link.springer.com",
        strategy="polite",
        notes="Springer Link, same as springer.com",
    ),
    "sciencedirect.com": PublisherConfig(
        domain="sciencedirect.com",
        strategy="stealth",
        notes="Elsevier, aggressive bot detection",
    ),
    "pubmed.ncbi.nlm.nih.gov": PublisherConfig(
        domain="pubmed.ncbi.nlm.nih.gov",
        strategy="default",
        notes="PubMed/NCBI, use API key for rate limits",
    ),
    "ncbi.nlm.nih.gov": PublisherConfig(
        domain="ncbi.nlm.nih.gov",
        strategy="default",
        notes="NCBI APIs, well-documented rate limits",
    ),
    "pmc.ncbi.nlm.nih.gov": PublisherConfig(
        domain="pmc.ncbi.nlm.nih.gov",
        strategy="default",
        notes="PMC full text, use E-utils API",
    ),
    "tandfonline.com": PublisherConfig(
        domain="tandfonline.com",
        strategy="browser",
        notes="Taylor & Francis",
    ),
    "mdpi.com": PublisherConfig(
        domain="mdpi.com",
        strategy="polite",
        notes="MDPI open access, generally friendly",
    ),
    "plos.org": PublisherConfig(
        domain="plos.org",
        strategy="polite",
        notes="PLOS journals, open access",
    ),
    "journals.plos.org": PublisherConfig(
        domain="journals.plos.org",
        strategy="polite",
        notes="PLOS journals domain",
    ),
    "biorxiv.org": PublisherConfig(
        domain="biorxiv.org",
        strategy="polite",
        notes="bioRxiv preprints, open access",
    ),
    "medrxiv.org": PublisherConfig(
        domain="medrxiv.org",
        strategy="polite",
        notes="medRxiv preprints, open access",
    ),
}


# =============================================================================
# Test Result Tracking
# =============================================================================


@dataclass
class URLTestResult:
    """Result of testing a single URL."""

    url: str
    url_type: str  # fulltext, pdf, metadata, pubmed, doi
    success: bool
    http_status: Optional[int] = None
    content_length: int = 0
    error_message: Optional[str] = None
    request_time: float = 0.0
    cloudflare_blocked: bool = False
    captcha_detected: bool = False
    paywall_detected: bool = False


@dataclass
class PublicationTestResult:
    """Complete test result for a publication."""

    entry_id: str
    title: str
    publisher: str
    strategy_used: str

    # URL test results
    url_results: List[URLTestResult] = field(default_factory=list)

    # Best successful extraction
    best_url_type: Optional[str] = None
    content_extracted: bool = False
    content_length: int = 0
    methods_found: bool = False
    identifiers_found: Dict[str, List[str]] = field(default_factory=dict)

    # Performance
    total_time: float = 0.0
    retries: int = 0

    def get_successful_urls(self) -> List[URLTestResult]:
        """Get all successful URL results."""
        return [r for r in self.url_results if r.success]

    def get_blocked_urls(self) -> List[URLTestResult]:
        """Get URLs that were blocked."""
        return [
            r
            for r in self.url_results
            if r.cloudflare_blocked or r.captcha_detected or r.http_status == 403
        ]


# =============================================================================
# Utility Functions
# =============================================================================


def detect_publisher(entry: PublicationQueueEntry) -> str:
    """Detect publisher from entry URLs."""
    urls_to_check = [
        entry.fulltext_url,
        entry.metadata_url,
        entry.pdf_url,
        entry.pubmed_url,
    ]

    for url in urls_to_check:
        if not url:
            continue

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]

            # Check against known publishers
            for pub_domain in PUBLISHER_CONFIGS.keys():
                if pub_domain in domain:
                    return pub_domain

            # Return domain if no specific config
            return domain

        except Exception:
            continue

    # Fallback to DOI resolution domain guess
    if entry.doi:
        # Common DOI prefixes
        if entry.doi.startswith("10.1016/"):
            return "cell.com"  # Elsevier/Cell
        elif entry.doi.startswith("10.1038/"):
            return "nature.com"
        elif entry.doi.startswith("10.1002/"):
            return "wiley.com"
        elif entry.doi.startswith("10.3389/"):
            return "frontiersin.org"
        elif entry.doi.startswith("10.1007/"):
            return "springer.com"

    return "unknown"


def apply_strategy_delay(strategy: RequestStrategy) -> None:
    """Apply random delay based on strategy."""
    if strategy.max_delay > 0:
        delay = random.uniform(strategy.min_delay, strategy.max_delay)
        print(f"      Waiting {delay:.1f}s (strategy: {strategy.name})...")
        time.sleep(delay)


def detect_blocking_signals(content: str, http_status: Optional[int]) -> Tuple[bool, bool, bool]:
    """
    Detect signs of bot blocking.

    Returns:
        (cloudflare_blocked, captcha_detected, paywall_detected)
    """
    cloudflare_blocked = False
    captcha_detected = False
    paywall_detected = False

    content_lower = content.lower() if content else ""

    # Cloudflare detection
    cloudflare_signals = [
        "cloudflare",
        "cf-ray",
        "attention required",
        "checking your browser",
        "ray id",
        "performance & security by cloudflare",
    ]
    cloudflare_blocked = any(signal in content_lower for signal in cloudflare_signals)

    # CAPTCHA detection
    captcha_signals = [
        "captcha",
        "recaptcha",
        "hcaptcha",
        "robot",
        "verify you are human",
        "security check",
    ]
    captcha_detected = any(signal in content_lower for signal in captcha_signals)

    # Paywall detection
    paywall_signals = [
        "paywall",
        "subscribe to read",
        "purchase article",
        "institutional access",
        "sign in to access",
        "this content is available",
        "full text is not available",
    ]
    paywall_detected = any(signal in content_lower for signal in paywall_signals)

    # HTTP status based detection
    if http_status == 403:
        cloudflare_blocked = True
    elif http_status == 429:
        cloudflare_blocked = True  # Rate limited

    return cloudflare_blocked, captcha_detected, paywall_detected


def create_test_data_manager() -> DataManagerV2:
    """Create a temporary DataManagerV2 for testing."""
    temp_dir = tempfile.mkdtemp(prefix="lobster_test_")
    return DataManagerV2(workspace_path=temp_dir)


def extract_identifiers_from_content(content: str) -> Dict[str, List[str]]:
    """Extract dataset identifiers from content."""
    if not content:
        return {}

    identifiers = {
        "geo": list(set(re.findall(r"GSE\d+", content))),
        "sra": list(set(re.findall(r"SRP\d+|SRX\d+|SRR\d+|PRJNA\d+", content))),
        "bioproject": list(set(re.findall(r"PRJNA\d+", content))),
        "biosample": list(set(re.findall(r"SAMN\d+", content))),
        "ena": list(set(re.findall(r"E-[A-Z]+-\d+", content))),
    }

    # Remove empty lists
    return {k: v for k, v in identifiers.items() if v}


# =============================================================================
# Test Functions
# =============================================================================


def test_url_extraction(
    url: str,
    url_type: str,
    content_service: ContentAccessService,
    strategy: RequestStrategy,
) -> URLTestResult:
    """
    Test content extraction from a single URL.

    Note: Currently uses ContentAccessService which doesn't support custom headers.
    This function is structured to allow future enhancement with request customization.
    """
    start_time = time.time()

    result = URLTestResult(
        url=url,
        url_type=url_type,
        success=False,
    )

    try:
        # Log attempt
        print(f"      Trying {url_type}: {url[:80]}...")

        # Use ContentAccessService for extraction
        # NOTE: Future enhancement - pass strategy headers to the service
        content_result = content_service.get_full_content(
            source=url,
            prefer_webpage=True,
            keywords=["abstract", "introduction", "methods", "results"],
            max_paragraphs=100,
        )

        result.request_time = time.time() - start_time

        if content_result and not content_result.get("error"):
            content = content_result.get("content", "")
            result.content_length = len(content)

            # Check for blocking signals
            cloudflare, captcha, paywall = detect_blocking_signals(
                content, content_result.get("http_status")
            )
            result.cloudflare_blocked = cloudflare
            result.captcha_detected = captcha
            result.paywall_detected = paywall

            # Consider successful if we got substantial content without blocking
            if result.content_length > 500 and not (cloudflare or captcha):
                result.success = True
                print(f"        -> Success: {result.content_length:,} chars ({result.request_time:.1f}s)")
            elif paywall:
                result.error_message = "Paywall detected"
                print(f"        -> Paywall detected ({result.request_time:.1f}s)")
            elif cloudflare or captcha:
                result.error_message = "Bot protection triggered"
                print(f"        -> Blocked: {'Cloudflare' if cloudflare else 'CAPTCHA'} ({result.request_time:.1f}s)")
            else:
                result.error_message = f"Insufficient content ({result.content_length} chars)"
                print(f"        -> Insufficient: {result.content_length} chars ({result.request_time:.1f}s)")
        else:
            error = content_result.get("error", "Unknown error") if content_result else "No response"
            result.error_message = error
            print(f"        -> Error: {error[:50]}...")

    except Exception as e:
        result.request_time = time.time() - start_time
        result.error_message = str(e)
        print(f"        -> Exception: {str(e)[:50]}...")

    return result


def test_single_entry(
    entry: PublicationQueueEntry,
    strategy_name: str,
    data_manager: DataManagerV2,
) -> PublicationTestResult:
    """Test content extraction for a single publication entry."""

    publisher = detect_publisher(entry)
    strategy = STRATEGIES.get(strategy_name, STRATEGIES["default"])

    result = PublicationTestResult(
        entry_id=entry.entry_id,
        title=entry.title or "Unknown",
        publisher=publisher,
        strategy_used=strategy_name,
    )

    content_service = ContentAccessService(data_manager=data_manager)
    start_time = time.time()

    # Get publisher config
    pub_config = PUBLISHER_CONFIGS.get(publisher)

    # Define URL priority order
    urls_to_test = []

    # Priority 1: Fulltext URL (unless configured to skip)
    if entry.fulltext_url and not (pub_config and pub_config.skip_fulltext):
        urls_to_test.append((entry.fulltext_url, "fulltext"))

    # Priority 2: PDF URL (or first if prefer_pdf)
    if entry.pdf_url:
        if pub_config and pub_config.prefer_pdf:
            urls_to_test.insert(0, (entry.pdf_url, "pdf"))
        else:
            urls_to_test.append((entry.pdf_url, "pdf"))

    # Priority 3: Metadata URL
    if entry.metadata_url:
        urls_to_test.append((entry.metadata_url, "metadata"))

    # Priority 4: PubMed URL -> PMID
    if entry.pubmed_url:
        # Extract PMID from URL
        pmid_match = re.search(r"/pubmed/(\d+)", entry.pubmed_url.lower())
        if pmid_match:
            urls_to_test.append((f"PMID:{pmid_match.group(1)}", "pubmed"))

    # Priority 5: DOI
    if entry.doi:
        urls_to_test.append((entry.doi, "doi"))

    # Priority 6: PMID/PMC ID directly
    if entry.pmid and not any(u[1] == "pubmed" for u in urls_to_test):
        urls_to_test.append((f"PMID:{entry.pmid}", "pmid"))
    if entry.pmc_id:
        urls_to_test.append((entry.pmc_id, "pmc"))

    # Test each URL until we get success
    for url, url_type in urls_to_test:
        if not url:
            continue

        url_result = test_url_extraction(
            url=url,
            url_type=url_type,
            content_service=content_service,
            strategy=strategy,
        )
        result.url_results.append(url_result)

        if url_result.success:
            result.best_url_type = url_type
            result.content_extracted = True
            result.content_length = url_result.content_length

            # Try to get the actual content for identifier extraction
            try:
                content_data = content_service.get_full_content(source=url)
                if content_data and content_data.get("content"):
                    content = content_data["content"]
                    result.identifiers_found = extract_identifiers_from_content(content)
                    result.methods_found = "method" in content.lower() or "materials and methods" in content.lower()
            except Exception:
                pass

            break  # Stop on first success

        # Apply delay between attempts
        apply_strategy_delay(strategy)

    result.total_time = time.time() - start_time
    return result


# =============================================================================
# Reporting Functions
# =============================================================================


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_entry_result(idx: int, total: int, result: PublicationTestResult) -> None:
    """Print result for a single entry."""
    status_icon = "+" if result.content_extracted else "x"
    title_short = result.title[:50] + "..." if len(result.title) > 50 else result.title

    print(f"\n[{idx}/{total}] {status_icon} {title_short}")
    print(f"      Publisher: {result.publisher}")
    print(f"      Strategy: {result.strategy_used}")

    if result.content_extracted:
        print(f"      Best URL: {result.best_url_type}")
        print(f"      Content: {result.content_length:,} chars | Methods: {'Y' if result.methods_found else 'N'}")
        if result.identifiers_found:
            ids_str = ", ".join(
                f"{k.upper()}: {', '.join(v[:3])}" for k, v in result.identifiers_found.items()
            )
            print(f"      Identifiers: {ids_str}")
    else:
        blocked = result.get_blocked_urls()
        if blocked:
            print(f"      BLOCKED: {len(blocked)} URL(s) blocked by bot protection")
        else:
            print(f"      FAILED: No content extracted")

    print(f"      Time: {result.total_time:.1f}s")


def generate_summary_report(results: List[PublicationTestResult]) -> str:
    """Generate comprehensive summary report."""
    lines = []

    # Overall stats
    total = len(results)
    successful = sum(1 for r in results if r.content_extracted)
    blocked = sum(1 for r in results if r.get_blocked_urls())

    lines.append("\n")
    lines.append("=" * 70)
    lines.append("  SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Total Entries: {total}")
    lines.append(f"Successful:    {successful} ({successful/total*100:.0f}%)")
    lines.append(f"Blocked:       {blocked}")
    lines.append(f"Failed:        {total - successful}")

    # By publisher
    lines.append("\n--- By Publisher ---")
    publishers: Dict[str, List[PublicationTestResult]] = {}
    for r in results:
        publishers.setdefault(r.publisher, []).append(r)

    for pub, pub_results in sorted(publishers.items()):
        pub_success = sum(1 for r in pub_results if r.content_extracted)
        pub_blocked = sum(1 for r in pub_results if r.get_blocked_urls())
        rate = pub_success / len(pub_results) * 100 if pub_results else 0

        status = "OK" if rate >= 80 else "WARN" if rate >= 50 else "FAIL"
        blocked_str = f" [{pub_blocked} blocked]" if pub_blocked else ""
        lines.append(f"  {pub:30} {pub_success}/{len(pub_results)} ({rate:.0f}%) {status}{blocked_str}")

    # URL Priority Analysis
    lines.append("\n--- URL Priority Analysis ---")
    url_types = ["fulltext", "pdf", "metadata", "pubmed", "doi", "pmid", "pmc"]
    for url_type in url_types:
        type_results = []
        for r in results:
            for ur in r.url_results:
                if ur.url_type == url_type:
                    type_results.append(ur)

        if type_results:
            successes = sum(1 for ur in type_results if ur.success)
            blocked_count = sum(
                1 for ur in type_results if ur.cloudflare_blocked or ur.captcha_detected
            )
            lines.append(
                f"  {url_type:12} {successes}/{len(type_results)} successful"
                f"{f' ({blocked_count} blocked)' if blocked_count else ''}"
            )

    # Recommendations
    lines.append("\n--- Recommendations ---")
    for pub, pub_results in sorted(publishers.items()):
        pub_success = sum(1 for r in pub_results if r.content_extracted)
        pub_blocked = sum(1 for r in pub_results if r.get_blocked_urls())

        if pub_blocked > 0 and pub_success == 0:
            config = PUBLISHER_CONFIGS.get(pub)
            if config:
                lines.append(f"  - {pub}: Try '{config.strategy}' strategy. {config.notes}")
            else:
                lines.append(f"  - {pub}: Unknown publisher, may need manual handling")

    # Best working URLs
    lines.append("\n--- Working URL Types ---")
    successful_by_url_type: Dict[str, int] = {}
    for r in results:
        if r.content_extracted and r.best_url_type:
            successful_by_url_type[r.best_url_type] = (
                successful_by_url_type.get(r.best_url_type, 0) + 1
            )

    for url_type, count in sorted(successful_by_url_type.items(), key=lambda x: -x[1]):
        lines.append(f"  {url_type}: {count} successful extractions")

    return "\n".join(lines)


# =============================================================================
# Main Test Runner
# =============================================================================


def run_publication_tests(
    ris_file: Path,
    strategy: str = "auto",
    max_entries: Optional[int] = None,
    entry_index: Optional[int] = None,
    dry_run: bool = False,
) -> List[PublicationTestResult]:
    """
    Load RIS file and test publication queue processing.

    Args:
        ris_file: Path to RIS file
        strategy: Strategy name or "auto" for publisher-specific
        max_entries: Maximum entries to test
        entry_index: Test only this entry (by index)
        dry_run: Just load queue, don't make requests

    Returns:
        List of test results
    """
    print_header("Publication Queue Processing Test Suite")
    print(f"\nRIS File: {ris_file}")
    print(f"Strategy: {strategy}")
    print(f"Dry Run: {dry_run}")

    # Parse RIS file
    print("\n--- Loading RIS File ---")
    parser = RISParser()
    entries = parser.parse_file(ris_file)
    print(f"Entries loaded: {len(entries)}")
    print(f"Parser stats: {parser.get_statistics()}")

    # Filter entries if specified
    if entry_index is not None:
        if 0 <= entry_index < len(entries):
            entries = [entries[entry_index]]
            print(f"Testing single entry at index {entry_index}")
        else:
            print(f"Error: Entry index {entry_index} out of range (0-{len(entries)-1})")
            return []
    elif max_entries:
        entries = entries[:max_entries]
        print(f"Limited to first {max_entries} entries")

    # Show queue preview
    print("\n--- Queue Preview ---")
    for i, entry in enumerate(entries):
        publisher = detect_publisher(entry)
        title_short = (entry.title or "No title")[:60]
        urls = []
        if entry.fulltext_url:
            urls.append("fulltext")
        if entry.pdf_url:
            urls.append("pdf")
        if entry.metadata_url:
            urls.append("metadata")
        if entry.pubmed_url:
            urls.append("pubmed")
        if entry.doi:
            urls.append("doi")

        print(f"  [{i}] {title_short}")
        print(f"      Publisher: {publisher} | URLs: {', '.join(urls)}")

    if dry_run:
        print("\n--- Dry Run Complete ---")
        print("No requests made. Remove --dry-run to test extraction.")
        return []

    # Create temporary DataManager
    data_manager = create_test_data_manager()
    print(f"\nWorkspace: {data_manager.workspace_path}")

    # Process each entry
    print("\n--- Testing Entries ---")
    results = []

    for i, entry in enumerate(entries, 1):
        publisher = detect_publisher(entry)

        # Determine strategy
        if strategy == "auto":
            pub_config = PUBLISHER_CONFIGS.get(publisher)
            effective_strategy = pub_config.strategy if pub_config else "default"
        else:
            effective_strategy = strategy

        result = test_single_entry(entry, effective_strategy, data_manager)
        results.append(result)

        print_entry_result(i, len(entries), result)

        # Delay between entries
        if i < len(entries):
            strat = STRATEGIES.get(effective_strategy, STRATEGIES["default"])
            apply_strategy_delay(strat)

    # Generate report
    report = generate_summary_report(results)
    print(report)

    return results


# =============================================================================
# NCBI E-Link Enrichment Test
# =============================================================================


@dataclass
class EnrichmentTestResult:
    """Result of NCBI E-Link enrichment test for a publication."""

    entry_id: str
    title: str
    pmid: Optional[str] = None
    pmc_id: Optional[str] = None
    linked_datasets: Dict[str, List[str]] = field(default_factory=dict)
    success: bool = False
    error_message: Optional[str] = None
    request_time: float = 0.0


def test_ncbi_enrichment(
    ris_file: Path,
    max_entries: Optional[int] = None,
    entry_index: Optional[int] = None,
) -> List[EnrichmentTestResult]:
    """
    Test NCBI E-Link enrichment for publication entries.

    This tests the new ncbi_enrich task that retrieves linked datasets
    from NCBI E-Link API BEFORE attempting full publication extraction.

    Args:
        ris_file: Path to RIS file
        max_entries: Maximum entries to test
        entry_index: Test only this entry (by index)

    Returns:
        List of enrichment test results
    """
    print_header("NCBI E-Link Enrichment Test")
    print(f"\nRIS File: {ris_file}")

    # Parse RIS file
    print("\n--- Loading RIS File ---")
    parser = RISParser()
    entries = parser.parse_file(ris_file)
    print(f"Entries loaded: {len(entries)}")

    # Filter entries if specified
    if entry_index is not None:
        if 0 <= entry_index < len(entries):
            entries = [entries[entry_index]]
            print(f"Testing single entry at index {entry_index}")
        else:
            print(f"Error: Entry index {entry_index} out of range (0-{len(entries)-1})")
            return []
    elif max_entries:
        entries = entries[:max_entries]
        print(f"Limited to first {max_entries} entries")

    # Create temporary DataManager and service
    data_manager = create_test_data_manager()
    print(f"Workspace: {data_manager.workspace_path}")

    # Import PublicationProcessingService
    from lobster.services.publication_processing_service import PublicationProcessingService

    service = PublicationProcessingService(data_manager)

    # Test each entry
    print("\n--- Testing NCBI E-Link Enrichment ---")
    results: List[EnrichmentTestResult] = []

    for i, entry in enumerate(entries, 1):
        title_short = (entry.title or "No title")[:50]
        if len(entry.title or "") > 50:
            title_short += "..."

        print(f"\n[{i}/{len(entries)}] {title_short}")

        result = EnrichmentTestResult(
            entry_id=entry.entry_id,
            title=entry.title or "Unknown",
        )

        # Get PMID
        pmid = entry.pmid
        if not pmid and entry.pubmed_url:
            pmid = service._extract_pmid_from_url(entry.pubmed_url)

        if not pmid:
            result.error_message = "No PMID available"
            print(f"      ⚠ No PMID available for enrichment")
            results.append(result)
            continue

        result.pmid = pmid
        print(f"      PMID: {pmid}")

        # Test enrichment
        start_time = time.time()
        try:
            enrichment = service._enrich_from_ncbi(entry)
            result.request_time = time.time() - start_time

            if enrichment["success"]:
                result.success = True
                result.pmc_id = enrichment.get("pmc_id")
                result.linked_datasets = enrichment.get("linked_datasets", {})

                total_linked = sum(len(v) for v in result.linked_datasets.values())
                print(f"      ✓ Enrichment success ({result.request_time:.1f}s)")
                if result.pmc_id:
                    print(f"      PMC ID: {result.pmc_id}")
                if total_linked > 0:
                    print(f"      Linked datasets: {total_linked}")
                    for db, ids in result.linked_datasets.items():
                        if ids:
                            print(f"        - {db}: {', '.join(ids[:3])}")
                            if len(ids) > 3:
                                print(f"          (+{len(ids) - 3} more)")
                else:
                    print("      No linked datasets found in NCBI")
            else:
                result.error_message = enrichment.get("error", "Unknown error")
                print(f"      ✗ Enrichment failed: {result.error_message}")

        except Exception as e:
            result.request_time = time.time() - start_time
            result.error_message = str(e)
            print(f"      ✗ Exception: {str(e)[:50]}")

        results.append(result)

        # Small delay between NCBI requests (rate limiting)
        if i < len(entries):
            time.sleep(0.5)

    # Generate enrichment summary
    print("\n")
    print("=" * 70)
    print("  NCBI E-LINK ENRICHMENT SUMMARY")
    print("=" * 70)

    total = len(results)
    successful = sum(1 for r in results if r.success)
    with_pmc = sum(1 for r in results if r.pmc_id)
    with_datasets = sum(1 for r in results if any(r.linked_datasets.values()))
    no_pmid = sum(1 for r in results if r.error_message == "No PMID available")

    print(f"\nTotal Entries: {total}")
    print(f"Successful:    {successful} ({successful/total*100:.0f}%)")
    print(f"With PMC ID:   {with_pmc}")
    print(f"With Datasets: {with_datasets}")
    print(f"No PMID:       {no_pmid}")

    # Dataset summary
    all_datasets: Dict[str, List[str]] = {"GEO": [], "SRA": [], "BioProject": [], "BioSample": []}
    for r in results:
        for db, ids in r.linked_datasets.items():
            all_datasets.setdefault(db, []).extend(ids)

    print("\n--- Linked Datasets Found ---")
    for db, ids in all_datasets.items():
        unique_ids = list(set(ids))
        if unique_ids:
            print(f"  {db}: {len(unique_ids)} unique")
            for acc in unique_ids[:5]:
                print(f"    - {acc}")
            if len(unique_ids) > 5:
                print(f"    (+{len(unique_ids) - 5} more)")

    return results


# =============================================================================
# Identifier Resolution Test
# =============================================================================


@dataclass
class IdentifierResolutionResult:
    """Result of identifier resolution test for a publication."""

    entry_id: str
    title: str
    doi: Optional[str] = None
    resolved_pmid: Optional[str] = None
    resolved_pmc: Optional[str] = None
    success: bool = False
    skipped: bool = False
    error_message: Optional[str] = None
    request_time: float = 0.0


def test_identifier_resolution(
    ris_file: Path,
    max_entries: Optional[int] = None,
    entry_index: Optional[int] = None,
) -> List[IdentifierResolutionResult]:
    """
    Test identifier resolution (DOI → PMID) for publication entries.

    This tests the new resolve_identifiers task that uses NCBI ID Converter API
    to resolve DOIs to PMIDs BEFORE attempting E-Link enrichment.

    Args:
        ris_file: Path to RIS file
        max_entries: Maximum entries to test
        entry_index: Test only this entry (by index)

    Returns:
        List of identifier resolution test results
    """
    print_header("Identifier Resolution Test (DOI → PMID)")
    print(f"\nRIS File: {ris_file}")

    # Parse RIS file
    print("\n--- Loading RIS File ---")
    parser = RISParser()
    entries = parser.parse_file(ris_file)
    print(f"Entries loaded: {len(entries)}")

    # Filter entries if specified
    if entry_index is not None:
        if 0 <= entry_index < len(entries):
            entries = [entries[entry_index]]
            print(f"Testing single entry at index {entry_index}")
        else:
            print(f"Error: Entry index {entry_index} out of range (0-{len(entries)-1})")
            return []
    elif max_entries:
        entries = entries[:max_entries]
        print(f"Limited to first {max_entries} entries")

    # Create temporary DataManager and service
    data_manager = create_test_data_manager()
    print(f"Workspace: {data_manager.workspace_path}")

    # Import PublicationProcessingService
    from lobster.services.publication_processing_service import PublicationProcessingService

    service = PublicationProcessingService(data_manager)

    # Test each entry
    print("\n--- Testing Identifier Resolution ---")
    results: List[IdentifierResolutionResult] = []

    for i, entry in enumerate(entries, 1):
        title_short = (entry.title or "No title")[:50]
        if len(entry.title or "") > 50:
            title_short += "..."

        print(f"\n[{i}/{len(entries)}] {title_short}")

        result = IdentifierResolutionResult(
            entry_id=entry.entry_id,
            title=entry.title or "Unknown",
            doi=entry.doi,
        )

        if not entry.doi:
            result.error_message = "No DOI available"
            print(f"      ⚠ No DOI available for resolution")
            results.append(result)
            continue

        print(f"      DOI: {entry.doi}")

        # Check if PMID already present
        if entry.pmid or entry.pubmed_url:
            pmid = entry.pmid or service._extract_pmid_from_url(entry.pubmed_url)
            result.skipped = True
            result.resolved_pmid = pmid
            print(f"      ⚠ PMID already present: {pmid} (skipping resolution)")
            results.append(result)
            continue

        # Test resolution
        start_time = time.time()
        try:
            resolution = service._resolve_identifiers(entry)
            result.request_time = time.time() - start_time

            if resolution["success"]:
                result.success = True
                result.resolved_pmid = resolution["resolved_pmid"]
                result.resolved_pmc = resolution["resolved_pmc"]

                print(f"      ✓ Resolution success ({result.request_time:.1f}s)")
                print(f"      PMID: {result.resolved_pmid}")
                if result.resolved_pmc:
                    print(f"      PMC ID: {result.resolved_pmc}")
            elif resolution["skipped"]:
                result.skipped = True
                result.resolved_pmid = resolution["resolved_pmid"]
                print(f"      ⚠ Skipped: PMID already present ({result.resolved_pmid})")
            else:
                result.error_message = resolution.get("error", "Unknown error")
                print(f"      ⚠ Not resolved: {result.error_message}")

        except Exception as e:
            result.request_time = time.time() - start_time
            result.error_message = str(e)
            print(f"      ✗ Exception: {str(e)[:50]}")

        results.append(result)

        # Small delay between NCBI requests (rate limiting)
        if i < len(entries):
            time.sleep(0.5)

    # Generate resolution summary
    print("\n")
    print("=" * 70)
    print("  IDENTIFIER RESOLUTION SUMMARY")
    print("=" * 70)

    total = len(results)
    successful = sum(1 for r in results if r.success)
    skipped = sum(1 for r in results if r.skipped)
    no_doi = sum(1 for r in results if r.error_message == "No DOI available")
    not_in_pubmed = sum(
        1 for r in results
        if r.error_message and "not found in PubMed" in r.error_message
    )

    print(f"\nTotal Entries: {total}")
    print(f"Resolved:      {successful} ({successful/total*100:.0f}%)")
    print(f"Skipped:       {skipped} (PMID already present)")
    print(f"No DOI:        {no_doi}")
    print(f"Not in PubMed: {not_in_pubmed} (preprints or non-indexed)")

    # Show resolved identifiers
    if successful > 0:
        print("\n--- Resolved Identifiers ---")
        for r in results:
            if r.success:
                pmc_str = f", PMC:{r.resolved_pmc}" if r.resolved_pmc else ""
                print(f"  {r.doi} → PMID:{r.resolved_pmid}{pmc_str}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test publication queue processing with smart request strategies"
    )
    parser.add_argument(
        "--ris-file",
        type=Path,
        default=Path("kevin_notes/databiomix/CRC_microbiome.ris"),
        help="Path to RIS file (default: kevin_notes/databiomix/CRC_microbiome.ris)",
    )
    parser.add_argument(
        "--strategy",
        choices=["default", "polite", "browser", "stealth", "auto"],
        default="auto",
        help="Request strategy to use (default: auto)",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Maximum number of entries to test",
    )
    parser.add_argument(
        "--entry",
        type=int,
        default=None,
        help="Test only this entry (by index)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse RIS and show queue without making requests",
    )
    parser.add_argument(
        "--enrich-only",
        action="store_true",
        help="Test only NCBI E-Link enrichment (no full publication extraction)",
    )
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Test only identifier resolution (DOI → PMID via NCBI ID Converter)",
    )

    args = parser.parse_args()

    # Resolve RIS file path
    ris_path = args.ris_file
    if not ris_path.is_absolute():
        # Try relative to project root
        project_root = Path(__file__).parent.parent.parent
        ris_path = project_root / ris_path

    if not ris_path.exists():
        print(f"Error: RIS file not found: {ris_path}")
        print("\nAvailable RIS files:")
        project_root = Path(__file__).parent.parent.parent
        for ris in project_root.rglob("*.ris"):
            print(f"  {ris.relative_to(project_root)}")
        sys.exit(1)

    try:
        # Run identifier resolution test if flag is set
        if args.resolve_only:
            results = test_identifier_resolution(
                ris_file=ris_path,
                max_entries=args.max_entries,
                entry_index=args.entry,
            )

            # Exit code based on success rate
            if results:
                success_rate = sum(1 for r in results if r.success or r.skipped) / len(results)
                sys.exit(0 if success_rate >= 0.5 else 1)

        # Run enrichment-only test if flag is set
        elif args.enrich_only:
            results = test_ncbi_enrichment(
                ris_file=ris_path,
                max_entries=args.max_entries,
                entry_index=args.entry,
            )

            # Exit code based on success rate
            if results:
                success_rate = sum(1 for r in results if r.success) / len(results)
                sys.exit(0 if success_rate >= 0.5 else 1)
        else:
            # Full publication processing test
            results = run_publication_tests(
                ris_file=ris_path,
                strategy=args.strategy,
                max_entries=args.max_entries,
                entry_index=args.entry,
                dry_run=args.dry_run,
            )

            # Exit code based on success rate
            if results:
                success_rate = sum(1 for r in results if r.content_extracted) / len(results)
                sys.exit(0 if success_rate >= 0.5 else 1)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
