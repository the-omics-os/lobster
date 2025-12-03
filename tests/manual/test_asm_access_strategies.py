#!/usr/bin/env python3
"""
Test script to identify working strategy for ASM journal access.

Tests multiple approaches to bypass ASM's bot protection:
1. Cloudscraper with various configurations
2. Alternative libraries (playwright, selenium-stealth, curl_cffi)
3. Different header combinations
4. Alternative access routes
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single test attempt."""

    strategy: str
    success: bool
    status_code: Optional[int]
    response_size: Optional[int]
    content_type: Optional[str]
    execution_time: float
    error_message: Optional[str] = None
    dependencies: List[str] = None


class ASMAccessTester:
    """Test various strategies for accessing ASM journals."""

    TEST_URL = "https://journals.asm.org/doi/10.1128/JCM.01893-20"

    def __init__(self):
        self.results: List[TestResult] = []

    # ========== STRATEGY 1: Cloudscraper Variations ==========

    def test_cloudscraper_basic(self) -> TestResult:
        """Test basic cloudscraper (current failing approach)."""
        strategy = "cloudscraper_basic"
        start = time.time()

        try:
            import cloudscraper

            scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "darwin", "mobile": False},
            )
            response = scraper.get(self.TEST_URL, timeout=30)

            return TestResult(
                strategy=strategy,
                success=response.status_code == 200,
                status_code=response.status_code,
                response_size=len(response.content),
                content_type=response.headers.get("Content-Type"),
                execution_time=time.time() - start,
                dependencies=["cloudscraper"],
            )
        except Exception as e:
            return TestResult(
                strategy=strategy,
                success=False,
                status_code=None,
                response_size=None,
                content_type=None,
                execution_time=time.time() - start,
                error_message=str(e),
                dependencies=["cloudscraper"],
            )

    def test_cloudscraper_firefox(self) -> TestResult:
        """Test cloudscraper with Firefox profile."""
        strategy = "cloudscraper_firefox"
        start = time.time()

        try:
            import cloudscraper

            scraper = cloudscraper.create_scraper(
                browser={"browser": "firefox", "platform": "darwin", "mobile": False},
            )
            response = scraper.get(self.TEST_URL, timeout=30)

            return TestResult(
                strategy=strategy,
                success=response.status_code == 200,
                status_code=response.status_code,
                response_size=len(response.content),
                content_type=response.headers.get("Content-Type"),
                execution_time=time.time() - start,
                dependencies=["cloudscraper"],
            )
        except Exception as e:
            return TestResult(
                strategy=strategy,
                success=False,
                status_code=None,
                response_size=None,
                content_type=None,
                execution_time=time.time() - start,
                error_message=str(e),
                dependencies=["cloudscraper"],
            )

    def test_cloudscraper_enhanced_headers(self) -> TestResult:
        """Test cloudscraper with enhanced stealth headers."""
        strategy = "cloudscraper_enhanced_headers"
        start = time.time()

        try:
            import cloudscraper

            scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "darwin", "mobile": False},
            )

            headers = {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0",
                "Referer": "https://journals.asm.org/",
            }

            response = scraper.get(self.TEST_URL, headers=headers, timeout=30)

            return TestResult(
                strategy=strategy,
                success=response.status_code == 200,
                status_code=response.status_code,
                response_size=len(response.content),
                content_type=response.headers.get("Content-Type"),
                execution_time=time.time() - start,
                dependencies=["cloudscraper"],
            )
        except Exception as e:
            return TestResult(
                strategy=strategy,
                success=False,
                status_code=None,
                response_size=None,
                content_type=None,
                execution_time=time.time() - start,
                error_message=str(e),
                dependencies=["cloudscraper"],
            )

    def test_cloudscraper_with_delay(self) -> TestResult:
        """Test cloudscraper with pre-request delay and session persistence."""
        strategy = "cloudscraper_with_delay"
        start = time.time()

        try:
            import cloudscraper

            scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "darwin", "mobile": False},
            )

            # First, visit the homepage to establish session
            logger.info("Visiting homepage first...")
            scraper.get("https://journals.asm.org/", timeout=30)

            # Wait 3 seconds
            time.sleep(3)

            # Then try the article
            logger.info("Fetching article...")
            response = scraper.get(self.TEST_URL, timeout=30)

            return TestResult(
                strategy=strategy,
                success=response.status_code == 200,
                status_code=response.status_code,
                response_size=len(response.content),
                content_type=response.headers.get("Content-Type"),
                execution_time=time.time() - start,
                dependencies=["cloudscraper"],
            )
        except Exception as e:
            return TestResult(
                strategy=strategy,
                success=False,
                status_code=None,
                response_size=None,
                content_type=None,
                execution_time=time.time() - start,
                error_message=str(e),
                dependencies=["cloudscraper"],
            )

    # ========== STRATEGY 2: curl_cffi (TLS fingerprinting) ==========

    def test_curl_cffi(self) -> TestResult:
        """Test curl_cffi which mimics curl's TLS fingerprint."""
        strategy = "curl_cffi"
        start = time.time()

        try:
            from curl_cffi import requests

            response = requests.get(self.TEST_URL, impersonate="chrome110", timeout=30)

            return TestResult(
                strategy=strategy,
                success=response.status_code == 200,
                status_code=response.status_code,
                response_size=len(response.content),
                content_type=response.headers.get("Content-Type"),
                execution_time=time.time() - start,
                dependencies=["curl_cffi"],
            )
        except ImportError:
            return TestResult(
                strategy=strategy,
                success=False,
                status_code=None,
                response_size=None,
                content_type=None,
                execution_time=time.time() - start,
                error_message="curl_cffi not installed (pip install curl-cffi)",
                dependencies=["curl_cffi"],
            )
        except Exception as e:
            return TestResult(
                strategy=strategy,
                success=False,
                status_code=None,
                response_size=None,
                content_type=None,
                execution_time=time.time() - start,
                error_message=str(e),
                dependencies=["curl_cffi"],
            )

    # ========== STRATEGY 3: Playwright (real browser) ==========

    def test_playwright(self) -> TestResult:
        """Test playwright with real browser automation."""
        strategy = "playwright"
        start = time.time()

        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    viewport={"width": 1920, "height": 1080},
                )
                page = context.new_page()

                response = page.goto(
                    self.TEST_URL, wait_until="domcontentloaded", timeout=30000
                )
                content = page.content()

                browser.close()

                return TestResult(
                    strategy=strategy,
                    success=response.status == 200,
                    status_code=response.status,
                    response_size=len(content),
                    content_type=response.headers.get("content-type"),
                    execution_time=time.time() - start,
                    dependencies=["playwright"],
                )
        except ImportError:
            return TestResult(
                strategy=strategy,
                success=False,
                status_code=None,
                response_size=None,
                content_type=None,
                execution_time=time.time() - start,
                error_message="playwright not installed (pip install playwright && playwright install chromium)",
                dependencies=["playwright"],
            )
        except Exception as e:
            return TestResult(
                strategy=strategy,
                success=False,
                status_code=None,
                response_size=None,
                content_type=None,
                execution_time=time.time() - start,
                error_message=str(e),
                dependencies=["playwright"],
            )

    # ========== STRATEGY 4: Alternative routes ==========

    def test_pdf_direct_download(self) -> TestResult:
        """Test direct PDF download endpoint if available."""
        strategy = "pdf_direct_download"
        start = time.time()

        # ASM typically uses /doi/pdf/ endpoint
        pdf_url = self.TEST_URL.replace("/doi/", "/doi/pdf/")

        try:
            import cloudscraper

            scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "darwin", "mobile": False},
            )

            headers = {
                "Accept": "application/pdf,*/*",
                "Referer": self.TEST_URL,
            }

            response = scraper.get(pdf_url, headers=headers, timeout=30)

            return TestResult(
                strategy=strategy,
                success=response.status_code == 200,
                status_code=response.status_code,
                response_size=len(response.content),
                content_type=response.headers.get("Content-Type"),
                execution_time=time.time() - start,
                dependencies=["cloudscraper"],
            )
        except Exception as e:
            return TestResult(
                strategy=strategy,
                success=False,
                status_code=None,
                response_size=None,
                content_type=None,
                execution_time=time.time() - start,
                error_message=str(e),
                dependencies=["cloudscraper"],
            )

    def test_requests_with_spoofing(self) -> TestResult:
        """Test plain requests with aggressive header spoofing."""
        strategy = "requests_with_spoofing"
        start = time.time()

        try:
            import requests

            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"macOS"',
                "Cache-Control": "max-age=0",
                "Referer": "https://journals.asm.org/",
            }

            session = requests.Session()
            # Visit homepage first
            session.get("https://journals.asm.org/", headers=headers, timeout=30)
            time.sleep(2)

            # Then article
            response = session.get(self.TEST_URL, headers=headers, timeout=30)

            return TestResult(
                strategy=strategy,
                success=response.status_code == 200,
                status_code=response.status_code,
                response_size=len(response.content),
                content_type=response.headers.get("Content-Type"),
                execution_time=time.time() - start,
                dependencies=["requests"],
            )
        except Exception as e:
            return TestResult(
                strategy=strategy,
                success=False,
                status_code=None,
                response_size=None,
                content_type=None,
                execution_time=time.time() - start,
                error_message=str(e),
                dependencies=["requests"],
            )

    # ========== Test Runner ==========

    def run_all_tests(self):
        """Run all test strategies and collect results."""
        test_methods = [
            self.test_cloudscraper_basic,
            self.test_cloudscraper_firefox,
            self.test_cloudscraper_enhanced_headers,
            self.test_cloudscraper_with_delay,
            self.test_curl_cffi,
            self.test_playwright,
            self.test_pdf_direct_download,
            self.test_requests_with_spoofing,
        ]

        logger.info(
            f"Testing {len(test_methods)} strategies against: {self.TEST_URL}\n"
        )

        for test_method in test_methods:
            logger.info(f"Testing: {test_method.__name__}")
            result = test_method()
            self.results.append(result)

            if result.success:
                logger.info(
                    f"✅ SUCCESS - {result.strategy}: {result.status_code} ({result.response_size} bytes, {result.execution_time:.2f}s)"
                )
            else:
                logger.warning(
                    f"❌ FAILED - {result.strategy}: {result.error_message or f'Status {result.status_code}'}"
                )

            # Be polite - wait between tests
            time.sleep(2)

        logger.info("\n" + "=" * 80)
        self.print_summary()

    def print_summary(self):
        """Print formatted summary of all test results."""
        print("\n📊 TEST RESULTS SUMMARY")
        print("=" * 80)

        # Success summary
        successful = [r for r in self.results if r.success]
        print(f"\n✅ Successful strategies: {len(successful)}/{len(self.results)}")

        if successful:
            print("\n🎯 WORKING SOLUTIONS:")
            print(
                f"{'Strategy':<35} {'Status':<8} {'Size (KB)':<12} {'Time (s)':<10} {'Dependencies'}"
            )
            print("-" * 80)
            for r in successful:
                size_kb = r.response_size / 1024 if r.response_size else 0
                deps = ", ".join(r.dependencies) if r.dependencies else "N/A"
                print(
                    f"{r.strategy:<35} {r.status_code:<8} {size_kb:<12.1f} {r.execution_time:<10.2f} {deps}"
                )

        # Failure analysis
        failed = [r for r in self.results if not r.success]
        if failed:
            print(f"\n❌ Failed strategies: {len(failed)}/{len(self.results)}")
            print("\n🔍 FAILURE ANALYSIS:")
            print(f"{'Strategy':<35} {'Error/Status':<50}")
            print("-" * 80)
            for r in failed:
                error = r.error_message or f"HTTP {r.status_code}"
                error_short = error[:50] + "..." if len(error) > 50 else error
                print(f"{r.strategy:<35} {error_short}")

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if successful:
            # Prefer lighter solutions
            lightweight = [r for r in successful if "playwright" not in r.strategy]
            if lightweight:
                best = min(lightweight, key=lambda r: r.execution_time)
                print(f"   1️⃣  RECOMMENDED: {best.strategy}")
                print(
                    f"      - Fastest lightweight solution ({best.execution_time:.2f}s)"
                )
                print(f"      - Dependencies: {', '.join(best.dependencies)}")
                print(
                    f"      - Status: {best.status_code}, Size: {best.response_size/1024:.1f} KB"
                )

            # Fallback to heavy solution
            heavy = [r for r in successful if "playwright" in r.strategy]
            if heavy:
                print(f"   2️⃣  FALLBACK: {heavy[0].strategy}")
                print(
                    f"      - Most reliable but heavier ({heavy[0].execution_time:.2f}s)"
                )
                print(f"      - Use if lightweight solutions become unreliable")
        else:
            print("   ⚠️  NO WORKING SOLUTION FOUND")
            print("   - ASM may have very aggressive bot protection")
            print(
                "   - Consider: API access, institutional proxy, or contact ASM directly"
            )
            print("   - Check if papers available via PMC instead")


def main():
    """Main test execution."""
    tester = ASMAccessTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
