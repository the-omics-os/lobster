"""
Rate Limiter Arena - Comprehensive Publisher Access Strategy Testing Framework

This module provides a full-fledged testing arena for evaluating access strategies
across diverse publishers. Dynamically reads configuration from rate_limiter.py and
runs comparative tests to validate current strategies and discover optimal approaches.

Architecture:
- Strategy implementations (DEFAULT, POLITE, BROWSER, STEALTH, SESSION)
- Test URL registry (manually curated hard-to-reach publishers)
- Arena test runner (batch testing with statistical analysis)
- Report generation (comparative analysis like ASM tests)

Big Picture Goal:
Understand which strategies make most sense for each publisher category.

Usage:
    # Run full arena test
    python -m lobster.tools.rate_limiter_arena

    # Test specific publisher
    python -m lobster.tools.rate_limiter_arena --publisher nature

    # Test specific strategy
    python -m lobster.tools.rate_limiter_arena --strategy SESSION

    # Add new test URL
    python -m lobster.tools.rate_limiter_arena --add-url "https://..."

Created: 2025-12-01
Status: Production-ready
"""

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

# Import strategy configurations from rate_limiter
from lobster.tools.rate_limiter import (
    DOMAIN_CONFIG,
    HeaderStrategy,
    MultiDomainRateLimiter,
)

# Cloudscraper (optional but recommended for STEALTH testing)
try:
    import cloudscraper

    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False
    print("⚠️  cloudscraper not installed - STEALTH strategy will be skipped")
    print("   Install with: pip install cloudscraper")


# ============================================================================
# TEST URL REGISTRY (Manually Curated Hard-to-Reach Publishers)
# ============================================================================


@dataclass
class TestURL:
    """Test URL with metadata for arena testing."""

    url: str
    publisher: str
    domain: str
    expected_strategy: HeaderStrategy
    article_type: str = "research article"
    year: Optional[int] = None
    notes: str = ""

    def __post_init__(self):
        """Auto-extract domain if not provided."""
        if not self.domain:
            self.domain = urlparse(self.url).netloc


# Manual test URL registry - add challenging publishers here
TEST_URLS: List[TestURL] = [
    # ========== SESSION Strategy Publishers ==========
    TestURL(
        url="https://journals.asm.org/doi/10.1128/JCM.01893-20",
        publisher="ASM - Journal of Clinical Microbiology",
        domain="journals.asm.org",
        expected_strategy=HeaderStrategy.SESSION,
        year=2020,
        notes="Validated 93.3% success with SESSION strategy",
    ),
    TestURL(
        url="https://journals.asm.org/doi/10.1128/AAC.00483-20",
        publisher="ASM - Antimicrobial Agents and Chemotherapy",
        domain="journals.asm.org",
        expected_strategy=HeaderStrategy.SESSION,
        year=2020,
        notes="SESSION strategy confirmed 100% success",
    ),
    # ========== STEALTH Strategy Publishers ==========
    TestURL(
        url="https://www.cell.com/cell/fulltext/S0092-8674(20)30820-6",
        publisher="Cell Press",
        domain="cell.com",
        expected_strategy=HeaderStrategy.STEALTH,
        year=2020,
        notes="High Cloudflare protection - needs cloudscraper",
    ),
    TestURL(
        url="https://www.cell.com/cell/abstract/S0092-8674(24)00538-5",
        publisher="Cell",
        domain="cell.com",
        expected_strategy=HeaderStrategy.STEALTH,
        year=2024,
        notes="Microbiome research - Cell protection systems",
    ),
    TestURL(
        url="https://www.cell.com/cell-reports/abstract/S2211-1247(19)30405-X",
        publisher="Cell Reports",
        domain="cell.com",
        expected_strategy=HeaderStrategy.STEALTH,
        year=2019,
        notes="Pregnancy microbiome study",
    ),
    TestURL(
        url="https://www.science.org/doi/10.1126/science.abc1234",
        publisher="AAAS Science",
        domain="science.org",
        expected_strategy=HeaderStrategy.STEALTH,
        year=2021,
        notes="Aggressive bot detection - STEALTH recommended",
    ),
    TestURL(
        url="https://academic.oup.com/nar/article/49/D1/D1/6018440",
        publisher="Oxford University Press",
        domain="academic.oup.com",
        expected_strategy=HeaderStrategy.STEALTH,
        year=2021,
        notes="Cloudflare + TLS fingerprinting",
    ),
    TestURL(
        url="https://academic.oup.com/bioinformatics/article/36/22-23/5263/5939999",
        publisher="Oxford - Bioinformatics",
        domain="academic.oup.com",
        expected_strategy=HeaderStrategy.STEALTH,
        year=2021,
        notes="PCR primer design software review",
    ),
    TestURL(
        url="https://academic.oup.com/nar/article/50/D1/D912/6446532",
        publisher="Oxford - Nucleic Acids Research",
        domain="academic.oup.com",
        expected_strategy=HeaderStrategy.STEALTH,
        year=2022,
        notes="VFDB 2022 database article",
    ),
    TestURL(
        url="https://onlinelibrary.wiley.com/doi/abs/10.1111/2041-210X.12628",
        publisher="Wiley - Methods in Ecology and Evolution",
        domain="onlinelibrary.wiley.com",
        expected_strategy=HeaderStrategy.STEALTH,
        year=2017,
        notes="ggtree R package for phylogenetic tree visualization",
    ),
    TestURL(
        url="https://onlinelibrary.wiley.com/doi/10.1111/j.1439-0507.2008.01548.x",
        publisher="Wiley - Mycoses",
        domain="onlinelibrary.wiley.com",
        expected_strategy=HeaderStrategy.STEALTH,
        year=2009,
        notes="Clostridium tyrobutyricum typing techniques",
    ),
    TestURL(
        url="https://www.sciencedirect.com/science/article/pii/S135964462400134X",
        publisher="Elsevier - Drug Discovery Today",
        domain="sciencedirect.com",
        expected_strategy=HeaderStrategy.STEALTH,
        year=2024,
        notes="AI-discovered drugs in clinical trials",
    ),
    TestURL(
        url="https://www.sciencedirect.com/science/article/pii/S246812532400311X",
        publisher="Elsevier - The Lancet Gastroenterology",
        domain="sciencedirect.com",
        expected_strategy=HeaderStrategy.STEALTH,
        year=2025,
        notes="Microbiome testing in clinical practice consensus",
    ),
    TestURL(
        url="https://www.sciencedirect.com/science/article/pii/S221475351730181X",
        publisher="Elsevier - Biomolecular Detection",
        domain="sciencedirect.com",
        expected_strategy=HeaderStrategy.STEALTH,
        year=2017,
        notes="qPCR primer design methodology",
    ),
    TestURL(
        url="https://www.sciencedirect.com/science/article/pii/S0740002017304112",
        publisher="Elsevier - Food Microbiology",
        domain="sciencedirect.com",
        expected_strategy=HeaderStrategy.STEALTH,
        year=2018,
        notes="Listeria detection via filtration and qPCR",
    ),
    TestURL(
        url="https://www.journalofdairyscience.org/article/S0022-0302(17)30701-4/fulltext",
        publisher="Elsevier - Journal of Dairy Science",
        domain="journalofdairyscience.org",
        expected_strategy=HeaderStrategy.STEALTH,
        year=2017,
        notes="Staphylococcus aureus genotype B detection",
    ),
    # ========== BROWSER Strategy Publishers ==========
    TestURL(
        url="https://www.nature.com/articles/s41586-020-2649-2",
        publisher="Nature Publishing Group",
        domain="nature.com",
        expected_strategy=HeaderStrategy.BROWSER,
        year=2020,
        notes="Moderate protection - BROWSER headers sufficient",
    ),
    TestURL(
        url="https://www.nature.com/articles/s41467-024-51651-9",
        publisher="Nature Communications",
        domain="nature.com",
        expected_strategy=HeaderStrategy.BROWSER,
        year=2024,
        notes="Gut Microbiome Wellness Index 2 study",
    ),
    TestURL(
        url="https://www.nature.com/articles/s41467-024-49851-4",
        publisher="Nature Communications",
        domain="nature.com",
        expected_strategy=HeaderStrategy.BROWSER,
        year=2024,
        notes="Real-time genomics for antibiotic resistance",
    ),
    TestURL(
        url="https://www.nature.com/articles/s41587-024-02276-2",
        publisher="Nature Biotechnology",
        domain="nature.com",
        expected_strategy=HeaderStrategy.BROWSER,
        year=2024,
        notes="Strain tracking using synteny analysis",
    ),
    TestURL(
        url="https://www.nature.com/articles/s41591-024-03280-4",
        publisher="Nature Medicine",
        domain="nature.com",
        expected_strategy=HeaderStrategy.BROWSER,
        year=2024,
        notes="Microbiome-based IBD diagnosis",
    ),
    TestURL(
        url="https://link.springer.com/article/10.1007/s00253-020-10736-4",
        publisher="Springer",
        domain="link.springer.com",
        expected_strategy=HeaderStrategy.BROWSER,
        year=2020,
        notes="Standard bot detection",
    ),
    TestURL(
        url="https://bmcmicrobiol.biomedcentral.com/articles/10.1186/s12866-022-02451-y",
        publisher="BMC Microbiology",
        domain="biomedcentral.com",
        expected_strategy=HeaderStrategy.BROWSER,
        year=2022,
        notes="HT-qPCR and 16S rRNA for cheese microbiota",
    ),
    # ========== POLITE Strategy Publishers (Open Access) ==========
    TestURL(
        url="https://www.mdpi.com/2076-2607/9/1/1",
        publisher="MDPI",
        domain="mdpi.com",
        expected_strategy=HeaderStrategy.POLITE,
        year=2021,
        notes="Open access - bot-friendly",
    ),
    TestURL(
        url="https://www.mdpi.com/2076-2607/8/7/1057",
        publisher="MDPI - Microorganisms",
        domain="mdpi.com",
        expected_strategy=HeaderStrategy.POLITE,
        year=2020,
        notes="Clostridium tyrobutyricum characterization",
    ),
    TestURL(
        url="https://www.frontiersin.org/articles/10.3389/fmicb.2020.00001/full",
        publisher="Frontiers",
        domain="frontiersin.org",
        expected_strategy=HeaderStrategy.POLITE,
        year=2020,
        notes="Open access - explicitly allows scraping",
    ),
    TestURL(
        url="https://www.frontiersin.org/articles/10.3389/fmicb.2023.1183018/full",
        publisher="Frontiers in Microbiology",
        domain="frontiersin.org",
        expected_strategy=HeaderStrategy.POLITE,
        year=2023,
        notes="Bovine intramammary bacteriome and resistome",
    ),
    TestURL(
        url="https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2020.619166/full",
        publisher="Frontiers in Microbiology",
        domain="frontiersin.org",
        expected_strategy=HeaderStrategy.POLITE,
        year=2021,
        notes="HT-qPCR for bacteria in cheese",
    ),
    TestURL(
        url="https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2023.1154508/full",
        publisher="Frontiers in Microbiology",
        domain="frontiersin.org",
        expected_strategy=HeaderStrategy.POLITE,
        year=2023,
        notes="Raw milk microbiota enrichment for cheese",
    ),
    TestURL(
        url="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0250000",
        publisher="PLOS ONE",
        domain="journals.plos.org",
        expected_strategy=HeaderStrategy.POLITE,
        year=2021,
        notes="Open access - very permissive",
    ),
    TestURL(
        url="https://peerj.com/articles/8544",
        publisher="PeerJ",
        domain="peerj.com",
        expected_strategy=HeaderStrategy.POLITE,
        year=2020,
        notes="SpeciesPrimer bioinformatics pipeline",
    ),
    TestURL(
        url="https://peerj.com/articles/17673",
        publisher="PeerJ",
        domain="peerj.com",
        expected_strategy=HeaderStrategy.POLITE,
        year=2024,
        notes="WGS reporting in clinical microbiology",
    ),
    TestURL(
        url="https://www.microbiologyresearch.org/content/journal/mgen/10.1099/mgen.0.001254",
        publisher="Microbial Genomics",
        domain="microbiologyresearch.org",
        expected_strategy=HeaderStrategy.POLITE,
        year=2024,
        notes="Short-read polishing of ONT assemblies",
    ),
    TestURL(
        url="https://www.microbiologyresearch.org/content/journal/mgen/10.1099/mgen.0.000748",
        publisher="Microbial Genomics",
        domain="microbiologyresearch.org",
        expected_strategy=HeaderStrategy.POLITE,
        year=2022,
        notes="ResFinder - AMR gene identification",
    ),
    # ========== DEFAULT Strategy (NCBI) ==========
    TestURL(
        url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7500000/",
        publisher="PubMed Central",
        domain="pmc.ncbi.nlm.nih.gov",
        expected_strategy=HeaderStrategy.DEFAULT,
        year=2020,
        notes="Official API - minimal headers OK",
    ),
]


# ============================================================================
# STRATEGY IMPLEMENTATIONS
# ============================================================================


@dataclass
class StrategyResult:
    """Result of a single strategy test."""

    url: str
    strategy: HeaderStrategy
    success: bool
    status_code: Optional[int] = None
    response_time: Optional[float] = None
    content_length: Optional[int] = None
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class StrategyTester:
    """Test different access strategies against publisher URLs."""

    def __init__(self):
        self.rate_limiter = MultiDomainRateLimiter()

        # Chrome User-Agent for browser/stealth strategies
        self.chrome_ua = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )

    def test_default(self, url: str, timeout: int = 30) -> StrategyResult:
        """Test DEFAULT strategy (minimal headers)."""
        start = time.time()
        try:
            response = requests.get(url, timeout=timeout)
            elapsed = time.time() - start

            return StrategyResult(
                url=url,
                strategy=HeaderStrategy.DEFAULT,
                success=response.status_code == 200,
                status_code=response.status_code,
                response_time=elapsed,
                content_length=len(response.content),
            )
        except Exception as e:
            elapsed = time.time() - start
            return StrategyResult(
                url=url,
                strategy=HeaderStrategy.DEFAULT,
                success=False,
                response_time=elapsed,
                error_message=str(e)[:200],
            )

    def test_polite(self, url: str, timeout: int = 30) -> StrategyResult:
        """Test POLITE strategy (bot-friendly headers)."""
        headers = {
            "User-Agent": "LobsterAI/1.0 (Bioinformatics Research; +https://github.com/the-omics-os/lobster)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        start = time.time()
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            elapsed = time.time() - start

            return StrategyResult(
                url=url,
                strategy=HeaderStrategy.POLITE,
                success=response.status_code == 200,
                status_code=response.status_code,
                response_time=elapsed,
                content_length=len(response.content),
            )
        except Exception as e:
            elapsed = time.time() - start
            return StrategyResult(
                url=url,
                strategy=HeaderStrategy.POLITE,
                success=False,
                response_time=elapsed,
                error_message=str(e)[:200],
            )

    def test_browser(self, url: str, timeout: int = 30) -> StrategyResult:
        """Test BROWSER strategy (full browser headers)."""
        headers = {
            "User-Agent": self.chrome_ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        start = time.time()
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            elapsed = time.time() - start

            return StrategyResult(
                url=url,
                strategy=HeaderStrategy.BROWSER,
                success=response.status_code == 200,
                status_code=response.status_code,
                response_time=elapsed,
                content_length=len(response.content),
            )
        except Exception as e:
            elapsed = time.time() - start
            return StrategyResult(
                url=url,
                strategy=HeaderStrategy.BROWSER,
                success=False,
                response_time=elapsed,
                error_message=str(e)[:200],
            )

    def test_stealth(self, url: str, timeout: int = 30) -> StrategyResult:
        """Test STEALTH strategy (cloudscraper + Sec-Fetch-*)."""
        if not CLOUDSCRAPER_AVAILABLE:
            return StrategyResult(
                url=url,
                strategy=HeaderStrategy.STEALTH,
                success=False,
                error_message="cloudscraper not installed",
            )

        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
        }

        start = time.time()
        try:
            scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "darwin", "mobile": False},
            )
            response = scraper.get(url, headers=headers, timeout=timeout)
            elapsed = time.time() - start

            return StrategyResult(
                url=url,
                strategy=HeaderStrategy.STEALTH,
                success=response.status_code == 200,
                status_code=response.status_code,
                response_time=elapsed,
                content_length=len(response.content),
            )
        except Exception as e:
            elapsed = time.time() - start
            return StrategyResult(
                url=url,
                strategy=HeaderStrategy.STEALTH,
                success=False,
                response_time=elapsed,
                error_message=str(e)[:200],
            )

    def test_session(self, url: str, timeout: int = 30) -> StrategyResult:
        """Test SESSION strategy (homepage visit + session cookies + Sec-Fetch-*)."""
        # Extract domain for homepage visit
        parsed = urlparse(url)
        homepage = f"{parsed.scheme}://{parsed.netloc}/"

        headers = {
            "User-Agent": self.chrome_ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Sec-Ch-Ua": '"Chromium";v="120", "Not A Brand";v="99"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"macOS"',
        }

        start = time.time()
        try:
            # Step 1: Visit homepage to establish session
            session = requests.Session()
            session.get(homepage, headers=headers, timeout=timeout)

            # Step 2: Wait 2 seconds (mimic human behavior)
            time.sleep(2)

            # Step 3: Request article with session cookies
            response = session.get(url, headers=headers, timeout=timeout)
            elapsed = time.time() - start

            return StrategyResult(
                url=url,
                strategy=HeaderStrategy.SESSION,
                success=response.status_code == 200,
                status_code=response.status_code,
                response_time=elapsed,
                content_length=len(response.content),
            )
        except Exception as e:
            elapsed = time.time() - start
            return StrategyResult(
                url=url,
                strategy=HeaderStrategy.SESSION,
                success=False,
                response_time=elapsed,
                error_message=str(e)[:200],
            )

    def test_all_strategies(
        self, url: str, delay_between: float = 5.0
    ) -> List[StrategyResult]:
        """Test all strategies against a single URL."""
        results = []

        # Test each strategy with delays to respect rate limits
        for strategy in HeaderStrategy:
            if strategy == HeaderStrategy.DEFAULT:
                result = self.test_default(url)
            elif strategy == HeaderStrategy.POLITE:
                result = self.test_polite(url)
            elif strategy == HeaderStrategy.BROWSER:
                result = self.test_browser(url)
            elif strategy == HeaderStrategy.STEALTH:
                result = self.test_stealth(url)
            elif strategy == HeaderStrategy.SESSION:
                result = self.test_session(url)
            else:
                continue  # Unknown strategy

            results.append(result)
            print(
                f"  [{strategy.value:8s}] {'✓' if result.success else '✗'} "
                f"({result.status_code or 'ERR':3}) "
                f"{result.response_time:.2f}s"
                if result.response_time
                else ""
            )

            # Rate limiting delay
            if delay_between > 0:
                time.sleep(delay_between)

        return results


# ============================================================================
# ARENA TEST RUNNER
# ============================================================================


@dataclass
class ArenaReport:
    """Comprehensive arena test report."""

    test_date: str
    total_urls: int
    total_tests: int
    results: List[StrategyResult]

    # Aggregate statistics
    strategy_success_rates: Dict[str, float] = field(default_factory=dict)
    strategy_avg_latency: Dict[str, float] = field(default_factory=dict)
    publisher_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Recommendations
    strategy_recommendations: Dict[str, str] = field(default_factory=dict)


class ArenaRunner:
    """Main arena test runner."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.tester = StrategyTester()
        self.output_dir = (
            output_dir
            or Path(__file__).parent.parent.parent
            / "tests"
            / "manual"
            / "arena_results"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_full_arena(
        self, test_urls: Optional[List[TestURL]] = None, attempts_per_url: int = 3
    ) -> ArenaReport:
        """Run full arena test across all registered URLs."""
        if test_urls is None:
            test_urls = TEST_URLS

        print(f"\n{'=' * 80}")
        print("RATE LIMITER ARENA - Full Test Suite")
        print(f"{'=' * 80}")
        print(f"Test URLs: {len(test_urls)}")
        print(f"Attempts per URL: {attempts_per_url}")
        print(f"Total tests: {len(test_urls) * len(HeaderStrategy) * attempts_per_url}")
        print(
            f"Estimated time: ~{len(test_urls) * len(HeaderStrategy) * attempts_per_url * 5 / 60:.0f} minutes"
        )
        print(f"{'=' * 80}\n")

        all_results = []

        for idx, test_url in enumerate(test_urls, 1):
            print(f"\n[{idx}/{len(test_urls)}] Testing: {test_url.publisher}")
            print(f"URL: {test_url.url[:80]}...")
            print(f"Expected strategy: {test_url.expected_strategy.value}")

            for attempt in range(attempts_per_url):
                print(f"\n  Attempt {attempt + 1}/{attempts_per_url}:")
                results = self.tester.test_all_strategies(
                    test_url.url, delay_between=5.0
                )
                all_results.extend(results)

        # Generate report
        report = self._generate_report(all_results, test_urls)

        # Save report
        self._save_report(report)

        return report

    def run_single_publisher(
        self, publisher_name: str, attempts: int = 3
    ) -> ArenaReport:
        """Run arena test for a specific publisher."""
        test_urls = [
            url for url in TEST_URLS if publisher_name.lower() in url.publisher.lower()
        ]

        if not test_urls:
            print(f"❌ No test URLs found for publisher: {publisher_name}")
            return None

        return self.run_full_arena(test_urls, attempts_per_url=attempts)

    def run_single_strategy(
        self, strategy: HeaderStrategy, attempts: int = 3
    ) -> ArenaReport:
        """Run arena test for a specific strategy across all URLs."""
        print(f"\n{'=' * 80}")
        print(f"Testing Strategy: {strategy.value}")
        print(f"{'=' * 80}\n")

        all_results = []

        for idx, test_url in enumerate(TEST_URLS, 1):
            print(f"\n[{idx}/{len(TEST_URLS)}] {test_url.publisher}")

            for attempt in range(attempts):
                if strategy == HeaderStrategy.DEFAULT:
                    result = self.tester.test_default(test_url.url)
                elif strategy == HeaderStrategy.POLITE:
                    result = self.tester.test_polite(test_url.url)
                elif strategy == HeaderStrategy.BROWSER:
                    result = self.tester.test_browser(test_url.url)
                elif strategy == HeaderStrategy.STEALTH:
                    result = self.tester.test_stealth(test_url.url)
                elif strategy == HeaderStrategy.SESSION:
                    result = self.tester.test_session(test_url.url)

                all_results.append(result)
                print(
                    f"  Attempt {attempt + 1}: {'✓' if result.success else '✗'} "
                    f"({result.status_code or 'ERR'}) "
                    f"{result.response_time:.2f}s"
                    if result.response_time
                    else ""
                )

                time.sleep(5)  # Rate limiting

        report = self._generate_report(all_results, TEST_URLS)
        self._save_report(report)

        return report

    def _generate_report(
        self, results: List[StrategyResult], test_urls: List[TestURL]
    ) -> ArenaReport:
        """Generate comprehensive arena report from results."""
        # Aggregate by strategy
        strategy_results = defaultdict(list)
        for result in results:
            strategy_results[result.strategy.value].append(result)

        # Calculate success rates
        strategy_success_rates = {}
        strategy_avg_latency = {}

        for strategy_name, strat_results in strategy_results.items():
            successes = sum(1 for r in strat_results if r.success)
            total = len(strat_results)
            strategy_success_rates[strategy_name] = (
                (successes / total * 100) if total > 0 else 0
            )

            latencies = [r.response_time for r in strat_results if r.response_time]
            strategy_avg_latency[strategy_name] = (
                sum(latencies) / len(latencies) if latencies else 0
            )

        # Aggregate by publisher
        publisher_results = {}
        for test_url in test_urls:
            url_results = [r for r in results if r.url == test_url.url]

            # Group by strategy
            strategy_breakdown = {}
            for strategy in HeaderStrategy:
                strat_results = [r for r in url_results if r.strategy == strategy]
                successes = sum(1 for r in strat_results if r.success)
                total = len(strat_results)
                strategy_breakdown[strategy.value] = {
                    "success_rate": (successes / total * 100) if total > 0 else 0,
                    "attempts": total,
                }

            publisher_results[test_url.publisher] = {
                "url": test_url.url,
                "domain": test_url.domain,
                "expected_strategy": test_url.expected_strategy.value,
                "strategy_breakdown": strategy_breakdown,
            }

        # Generate recommendations
        strategy_recommendations = {}
        for publisher, data in publisher_results.items():
            best_strategy = max(
                data["strategy_breakdown"].items(), key=lambda x: x[1]["success_rate"]
            )[0]
            strategy_recommendations[publisher] = best_strategy

        return ArenaReport(
            test_date=datetime.now().isoformat(),
            total_urls=len(test_urls),
            total_tests=len(results),
            results=results,
            strategy_success_rates=strategy_success_rates,
            strategy_avg_latency=strategy_avg_latency,
            publisher_results=publisher_results,
            strategy_recommendations=strategy_recommendations,
        )

    def _save_report(self, report: ArenaReport):
        """Save arena report to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw data as JSON
        json_path = self.output_dir / f"arena_results_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "test_date": report.test_date,
                    "total_urls": report.total_urls,
                    "total_tests": report.total_tests,
                    "strategy_success_rates": report.strategy_success_rates,
                    "strategy_avg_latency": report.strategy_avg_latency,
                    "publisher_results": report.publisher_results,
                    "strategy_recommendations": report.strategy_recommendations,
                    "raw_results": [asdict(r) for r in report.results],
                },
                f,
                indent=2,
            )

        # Generate markdown report
        md_path = self.output_dir / f"arena_report_{timestamp}.md"
        self._generate_markdown_report(report, md_path)

        print(f"\n{'=' * 80}")
        print("✅ Arena Report Generated")
        print(f"{'=' * 80}")
        print(f"JSON: {json_path}")
        print(f"Markdown: {md_path}")
        print(f"{'=' * 80}\n")

    def _generate_markdown_report(self, report: ArenaReport, output_path: Path):
        """Generate markdown report."""
        with open(output_path, "w") as f:
            f.write("# Rate Limiter Arena Test Report\n\n")
            f.write(f"**Test Date**: {report.test_date}\n\n")
            f.write(f"**Total URLs Tested**: {report.total_urls}\n")
            f.write(f"**Total Tests Executed**: {report.total_tests}\n\n")

            f.write("## Strategy Performance Summary\n\n")
            f.write("| Strategy | Success Rate | Avg Latency |\n")
            f.write("|----------|--------------|-------------|\n")
            for strategy in HeaderStrategy:
                success_rate = report.strategy_success_rates.get(strategy.value, 0)
                latency = report.strategy_avg_latency.get(strategy.value, 0)
                f.write(
                    f"| {strategy.value} | {success_rate:.1f}% | {latency:.2f}s |\n"
                )

            f.write("\n## Publisher Results\n\n")
            for publisher, data in report.publisher_results.items():
                f.write(f"### {publisher}\n\n")
                f.write(f"- **URL**: {data['url']}\n")
                f.write(f"- **Domain**: {data['domain']}\n")
                f.write(f"- **Expected Strategy**: {data['expected_strategy']}\n")
                f.write(
                    f"- **Recommended Strategy**: {report.strategy_recommendations[publisher]}\n\n"
                )

                f.write("| Strategy | Success Rate | Attempts |\n")
                f.write("|----------|--------------|----------|\n")
                for strategy_name, breakdown in data["strategy_breakdown"].items():
                    f.write(
                        f"| {strategy_name} | {breakdown['success_rate']:.1f}% | {breakdown['attempts']} |\n"
                    )
                f.write("\n")

            f.write("## Recommendations\n\n")
            f.write(
                "Based on test results, the following strategy changes are recommended:\n\n"
            )

            for publisher, recommended in report.strategy_recommendations.items():
                data = report.publisher_results[publisher]
                if recommended != data["expected_strategy"]:
                    f.write(
                        f"- **{publisher}**: Change from `{data['expected_strategy']}` to `{recommended}`\n"
                    )


# ============================================================================
# CLI INTERFACE
# ============================================================================


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Rate Limiter Arena - Publisher Access Strategy Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full arena test (all publishers, all strategies)
  python -m lobster.tools.rate_limiter_arena

  # Test specific publisher
  python -m lobster.tools.rate_limiter_arena --publisher "Nature"

  # Test specific strategy across all publishers
  python -m lobster.tools.rate_limiter_arena --strategy SESSION

  # List available test URLs
  python -m lobster.tools.rate_limiter_arena --list-urls

  # Show current domain configurations
  python -m lobster.tools.rate_limiter_arena --show-config
        """,
    )

    parser.add_argument(
        "--publisher", "-p", help="Test specific publisher (partial name matching)"
    )
    parser.add_argument(
        "--strategy",
        "-s",
        choices=[s.value for s in HeaderStrategy],
        help="Test specific strategy across all publishers",
    )
    parser.add_argument(
        "--attempts",
        "-a",
        type=int,
        default=3,
        help="Number of attempts per test (default: 3)",
    )
    parser.add_argument(
        "--list-urls", "-l", action="store_true", help="List all registered test URLs"
    )
    parser.add_argument(
        "--show-config",
        "-c",
        action="store_true",
        help="Show current domain configurations from rate_limiter.py",
    )

    args = parser.parse_args()

    # List URLs
    if args.list_urls:
        print(f"\n{'=' * 80}")
        print(f"Registered Test URLs ({len(TEST_URLS)} total)")
        print(f"{'=' * 80}\n")
        for idx, test_url in enumerate(TEST_URLS, 1):
            print(f"{idx}. {test_url.publisher}")
            print(f"   URL: {test_url.url}")
            print(f"   Domain: {test_url.domain}")
            print(f"   Expected: {test_url.expected_strategy.value}")
            print(f"   Notes: {test_url.notes}\n")
        return

    # Show config
    if args.show_config:
        print(f"\n{'=' * 80}")
        print(f"Current Domain Configurations ({len(DOMAIN_CONFIG)} domains)")
        print(f"{'=' * 80}\n")
        for domain, config in DOMAIN_CONFIG.items():
            print(
                f"{domain:40s} {config.header_strategy.value:10s} {config.rate_limit:5.1f} req/s - {config.comment}"
            )
        return

    # Run arena tests
    runner = ArenaRunner()

    if args.publisher:
        runner.run_single_publisher(args.publisher, attempts=args.attempts)
    elif args.strategy:
        strategy = HeaderStrategy(args.strategy)
        runner.run_single_strategy(strategy, attempts=args.attempts)
    else:
        runner.run_full_arena(attempts_per_url=args.attempts)


if __name__ == "__main__":
    main()
