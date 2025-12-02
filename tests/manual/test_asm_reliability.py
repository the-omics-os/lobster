#!/usr/bin/env python3
"""
Reliability test for ASM journal access strategies.

Tests the most promising strategies multiple times to assess consistency.
"""

import time
import statistics
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def test_pdf_direct_download(n_attempts: int = 5) -> Dict:
    """Test PDF direct download reliability."""
    import cloudscraper

    test_url = "https://journals.asm.org/doi/10.1128/JCM.01893-20"
    pdf_url = test_url.replace("/doi/", "/doi/pdf/")

    results = []
    execution_times = []

    for i in range(n_attempts):
        start = time.time()
        try:
            scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "darwin", "mobile": False},
            )

            headers = {
                'Accept': 'application/pdf,*/*',
                'Referer': test_url,
            }

            response = scraper.get(pdf_url, headers=headers, timeout=30)
            elapsed = time.time() - start

            success = response.status_code == 200
            results.append(success)
            execution_times.append(elapsed)

            status = "✅" if success else "❌"
            logger.info(f"  Attempt {i+1}/{n_attempts}: {status} {response.status_code} ({len(response.content)} bytes, {elapsed:.2f}s)")

            # Wait between attempts
            if i < n_attempts - 1:
                time.sleep(2)

        except Exception as e:
            elapsed = time.time() - start
            results.append(False)
            execution_times.append(elapsed)
            logger.error(f"  Attempt {i+1}/{n_attempts}: ❌ Error: {str(e)[:50]}")
            time.sleep(2)

    success_rate = sum(results) / len(results) * 100
    avg_time = statistics.mean(execution_times) if execution_times else 0

    return {
        "strategy": "pdf_direct_download",
        "success_rate": success_rate,
        "attempts": n_attempts,
        "successes": sum(results),
        "avg_time": avg_time,
        "min_time": min(execution_times) if execution_times else 0,
        "max_time": max(execution_times) if execution_times else 0,
    }


def test_requests_with_spoofing(n_attempts: int = 5) -> Dict:
    """Test requests with aggressive header spoofing reliability."""
    import requests

    test_url = "https://journals.asm.org/doi/10.1128/JCM.01893-20"

    results = []
    execution_times = []

    for i in range(n_attempts):
        start = time.time()
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                'Sec-Ch-Ua-Mobile': '?0',
                'Sec-Ch-Ua-Platform': '"macOS"',
                'Cache-Control': 'max-age=0',
                'Referer': 'https://journals.asm.org/',
            }

            session = requests.Session()
            # Visit homepage first
            session.get("https://journals.asm.org/", headers=headers, timeout=30)
            time.sleep(2)

            # Then article
            response = session.get(test_url, headers=headers, timeout=30)
            elapsed = time.time() - start

            success = response.status_code == 200
            results.append(success)
            execution_times.append(elapsed)

            status = "✅" if success else "❌"
            logger.info(f"  Attempt {i+1}/{n_attempts}: {status} {response.status_code} ({len(response.content)} bytes, {elapsed:.2f}s)")

            # Wait between attempts
            if i < n_attempts - 1:
                time.sleep(2)

        except Exception as e:
            elapsed = time.time() - start
            results.append(False)
            execution_times.append(elapsed)
            logger.error(f"  Attempt {i+1}/{n_attempts}: ❌ Error: {str(e)[:50]}")
            time.sleep(2)

    success_rate = sum(results) / len(results) * 100
    avg_time = statistics.mean(execution_times) if execution_times else 0

    return {
        "strategy": "requests_with_spoofing",
        "success_rate": success_rate,
        "attempts": n_attempts,
        "successes": sum(results),
        "avg_time": avg_time,
        "min_time": min(execution_times) if execution_times else 0,
        "max_time": max(execution_times) if execution_times else 0,
    }


def test_cloudscraper_with_delay(n_attempts: int = 5) -> Dict:
    """Test cloudscraper with homepage visit + delay reliability."""
    import cloudscraper

    test_url = "https://journals.asm.org/doi/10.1128/JCM.01893-20"

    results = []
    execution_times = []

    for i in range(n_attempts):
        start = time.time()
        try:
            scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "darwin", "mobile": False},
            )

            # Visit homepage
            scraper.get("https://journals.asm.org/", timeout=30)
            time.sleep(3)

            # Then article
            response = scraper.get(test_url, timeout=30)
            elapsed = time.time() - start

            success = response.status_code == 200
            results.append(success)
            execution_times.append(elapsed)

            status = "✅" if success else "❌"
            logger.info(f"  Attempt {i+1}/{n_attempts}: {status} {response.status_code} ({len(response.content)} bytes, {elapsed:.2f}s)")

            # Wait between attempts
            if i < n_attempts - 1:
                time.sleep(2)

        except Exception as e:
            elapsed = time.time() - start
            results.append(False)
            execution_times.append(elapsed)
            logger.error(f"  Attempt {i+1}/{n_attempts}: ❌ Error: {str(e)[:50]}")
            time.sleep(2)

    success_rate = sum(results) / len(results) * 100
    avg_time = statistics.mean(execution_times) if execution_times else 0

    return {
        "strategy": "cloudscraper_with_delay",
        "success_rate": success_rate,
        "attempts": n_attempts,
        "successes": sum(results),
        "avg_time": avg_time,
        "min_time": min(execution_times) if execution_times else 0,
        "max_time": max(execution_times) if execution_times else 0,
    }


def main():
    """Run reliability tests."""
    n_attempts = 5

    print("="*80)
    print("🔬 ASM ACCESS RELIABILITY TEST")
    print(f"Testing each strategy {n_attempts} times to assess consistency")
    print("="*80)

    strategies = [
        test_pdf_direct_download,
        test_requests_with_spoofing,
        test_cloudscraper_with_delay,
    ]

    all_results = []

    for test_func in strategies:
        print(f"\n📊 Testing: {test_func.__name__}")
        print("-"*80)
        result = test_func(n_attempts)
        all_results.append(result)
        time.sleep(5)  # Longer wait between different strategies

    # Print summary
    print("\n" + "="*80)
    print("📈 RELIABILITY SUMMARY")
    print("="*80)
    print(f"{'Strategy':<35} {'Success Rate':<15} {'Avg Time (s)':<15} {'Details'}")
    print("-"*80)

    for r in all_results:
        success_pct = f"{r['success_rate']:.0f}%"
        details = f"{r['successes']}/{r['attempts']} ({r['min_time']:.2f}s-{r['max_time']:.2f}s)"
        print(f"{r['strategy']:<35} {success_pct:<15} {r['avg_time']:<15.2f} {details}")

    # Recommendation
    print("\n💡 FINAL RECOMMENDATION:")
    best = max(all_results, key=lambda x: (x['success_rate'], -x['avg_time']))

    if best['success_rate'] >= 80:
        print(f"   ✅ RECOMMENDED: {best['strategy']}")
        print(f"      - Success rate: {best['success_rate']:.0f}% ({best['successes']}/{best['attempts']})")
        print(f"      - Avg latency: {best['avg_time']:.2f}s (range: {best['min_time']:.2f}s-{best['max_time']:.2f}s)")
        print(f"      - Reliability: {'EXCELLENT' if best['success_rate'] == 100 else 'GOOD'}")
    else:
        print(f"   ⚠️  BEST AVAILABLE: {best['strategy']}")
        print(f"      - Success rate: {best['success_rate']:.0f}% (below 80% threshold)")
        print(f"      - Consider implementing retry logic or rate limiting")


if __name__ == "__main__":
    main()
