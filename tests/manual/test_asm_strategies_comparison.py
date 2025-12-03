#!/usr/bin/env python3
"""
Statistical Comparison of ASM Access Strategies

Tests TWO strategies on 10 diverse ASM journal URLs:
- Strategy A: Session-based with requests (100% success on initial testing)
- Strategy B: Cloudscraper (current implementation in docling_service.py)

Each strategy tested 3 times per URL (60 total tests).

Usage:
    python3 tests/manual/test_asm_strategies_comparison.py

Output:
    - Detailed results printed to console
    - Comprehensive markdown report: ASM_STRATEGY_COMPARISON.md
    - Raw JSON data: asm_strategy_comparison_data.json
"""

import json
import logging
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Import test URLs
from asm_test_urls import ASM_TEST_URLS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class StrategyResult:
    """Container for single test result."""

    def __init__(
        self,
        url: str,
        journal: str,
        doi: str,
        strategy: str,
        attempt: int,
        success: bool,
        status_code: Optional[int],
        response_time: float,
        content_length: int,
        error_message: Optional[str] = None,
    ):
        self.url = url
        self.journal = journal
        self.doi = doi
        self.strategy = strategy
        self.attempt = attempt
        self.success = success
        self.status_code = status_code
        self.response_time = response_time
        self.content_length = content_length
        self.error_message = error_message
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "url": self.url,
            "journal": self.journal,
            "doi": self.doi,
            "strategy": self.strategy,
            "attempt": self.attempt,
            "success": self.success,
            "status_code": self.status_code,
            "response_time": self.response_time,
            "content_length": self.content_length,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
        }


def test_strategy_a(url: str, timeout: int = 30) -> Dict:
    """
    Test Strategy A: Session-based requests with comprehensive headers.

    This is the strategy that achieved 100% success in initial testing.
    """
    import requests

    start_time = time.time()

    try:
        # Comprehensive browser-like headers
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

        # Create session
        session = requests.Session()

        # Step 1: Visit homepage to establish session
        session.get("https://journals.asm.org/", headers=headers, timeout=timeout)

        # Step 2: Wait before article request
        time.sleep(2)

        # Step 3: Fetch article
        response = session.get(url, headers=headers, timeout=timeout)

        elapsed = time.time() - start_time

        return {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "response_time": elapsed,
            "content_length": len(response.content),
            "error_message": (
                None if response.status_code == 200 else f"HTTP {response.status_code}"
            ),
        }

    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "status_code": None,
            "response_time": elapsed,
            "content_length": 0,
            "error_message": "Timeout",
        }

    except requests.exceptions.RequestException as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "status_code": None,
            "response_time": elapsed,
            "content_length": 0,
            "error_message": str(e)[:100],  # Truncate long error messages
        }

    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "status_code": None,
            "response_time": elapsed,
            "content_length": 0,
            "error_message": f"Unexpected error: {str(e)[:100]}",
        }


def test_strategy_b(url: str, timeout: int = 30) -> Dict:
    """
    Test Strategy B: Cloudscraper (current implementation).

    This is what's currently used in docling_service.py.
    """
    try:
        import cloudscraper
    except ImportError:
        return {
            "success": False,
            "status_code": None,
            "response_time": 0,
            "content_length": 0,
            "error_message": "cloudscraper not installed",
        }

    start_time = time.time()

    try:
        scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "darwin", "mobile": False},
        )

        response = scraper.get(url, timeout=timeout)

        elapsed = time.time() - start_time

        return {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "response_time": elapsed,
            "content_length": len(response.content),
            "error_message": (
                None if response.status_code == 200 else f"HTTP {response.status_code}"
            ),
        }

    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "status_code": None,
            "response_time": elapsed,
            "content_length": 0,
            "error_message": str(e)[:100],
        }


def run_comparison() -> List[StrategyResult]:
    """
    Run full comparison test suite.

    Tests both strategies on all 10 URLs, 3 attempts each.
    Total: 10 URLs × 2 strategies × 3 attempts = 60 tests
    """
    results = []
    total_tests = len(ASM_TEST_URLS) * 2 * 3  # 60 tests
    current_test = 0

    print("=" * 80)
    print("🔬 ASM STRATEGY COMPARISON - STATISTICAL VALIDATION")
    print("=" * 80)
    print(f"Total tests: {total_tests}")
    print(f"URLs: {len(ASM_TEST_URLS)}")
    print(f"Strategies: 2 (Session-based vs Cloudscraper)")
    print(f"Attempts per URL per strategy: 3")
    print(f"Estimated time: ~{total_tests * 7 / 60:.0f} minutes (with 5s delays)")
    print("=" * 80)
    print()

    for url_data in ASM_TEST_URLS:
        url = url_data["url"]
        journal = url_data["journal"]
        doi = url_data["doi"]

        print(f"\n📄 Testing: {journal}")
        print(f"   DOI: {doi}")
        print(f"   URL: {url}")
        print("-" * 80)

        # Test Strategy A (3 attempts)
        for attempt in range(1, 4):
            current_test += 1
            print(
                f"   [{current_test}/{total_tests}] Strategy A (Session-based) - Attempt {attempt}/3...",
                end=" ",
            )

            result_dict = test_strategy_a(url)

            result = StrategyResult(
                url=url,
                journal=journal,
                doi=doi,
                strategy="A_session",
                attempt=attempt,
                **result_dict,
            )
            results.append(result)

            status_icon = "✅" if result.success else "❌"
            print(
                f"{status_icon} {result.status_code or 'ERROR'} ({result.response_time:.2f}s)"
            )

            # Rate limiting delay
            time.sleep(5)

        # Test Strategy B (3 attempts)
        for attempt in range(1, 4):
            current_test += 1
            print(
                f"   [{current_test}/{total_tests}] Strategy B (Cloudscraper) - Attempt {attempt}/3...",
                end=" ",
            )

            result_dict = test_strategy_b(url)

            result = StrategyResult(
                url=url,
                journal=journal,
                doi=doi,
                strategy="B_cloudscraper",
                attempt=attempt,
                **result_dict,
            )
            results.append(result)

            status_icon = "✅" if result.success else "❌"
            print(
                f"{status_icon} {result.status_code or 'ERROR'} ({result.response_time:.2f}s)"
            )

            # Rate limiting delay
            time.sleep(5)

    print("\n" + "=" * 80)
    print("✅ Testing complete!")
    print("=" * 80)

    return results


def analyze_results(results: List[StrategyResult]) -> Dict:
    """Analyze test results and generate statistics."""

    # Separate by strategy
    strategy_a_results = [r for r in results if r.strategy == "A_session"]
    strategy_b_results = [r for r in results if r.strategy == "B_cloudscraper"]

    # Calculate overall statistics
    def calc_stats(strategy_results):
        total = len(strategy_results)
        successes = sum(1 for r in strategy_results if r.success)
        success_rate = (successes / total * 100) if total > 0 else 0

        success_times = [r.response_time for r in strategy_results if r.success]
        avg_time = statistics.mean(success_times) if success_times else 0
        std_time = statistics.stdev(success_times) if len(success_times) > 1 else 0

        # Failure modes
        failure_modes = {}
        for r in strategy_results:
            if not r.success:
                error = r.error_message or f"HTTP {r.status_code}"
                failure_modes[error] = failure_modes.get(error, 0) + 1

        return {
            "total": total,
            "successes": successes,
            "failures": total - successes,
            "success_rate": success_rate,
            "avg_response_time": avg_time,
            "std_response_time": std_time,
            "failure_modes": failure_modes,
        }

    # Per-URL statistics
    url_comparison = []
    for url_data in ASM_TEST_URLS:
        url = url_data["url"]
        journal = url_data["journal"]

        a_results = [r for r in strategy_a_results if r.url == url]
        b_results = [r for r in strategy_b_results if r.url == url]

        a_successes = sum(1 for r in a_results if r.success)
        b_successes = sum(1 for r in b_results if r.success)

        url_comparison.append(
            {
                "url": url,
                "journal": journal,
                "strategy_a_success": f"{a_successes}/3",
                "strategy_b_success": f"{b_successes}/3",
                "winner": (
                    "A"
                    if a_successes > b_successes
                    else ("B" if b_successes > a_successes else "Tie")
                ),
            }
        )

    return {
        "strategy_a": calc_stats(strategy_a_results),
        "strategy_b": calc_stats(strategy_b_results),
        "url_comparison": url_comparison,
    }


def generate_report(results: List[StrategyResult], analysis: Dict) -> str:
    """Generate comprehensive markdown report."""

    strategy_a = analysis["strategy_a"]
    strategy_b = analysis["strategy_b"]

    # Determine winner
    if strategy_a["success_rate"] > strategy_b["success_rate"]:
        winner = "Strategy A (Session-based)"
        confidence = (
            "HIGH"
            if strategy_a["success_rate"] - strategy_b["success_rate"] > 20
            else "MEDIUM"
        )
    elif strategy_b["success_rate"] > strategy_a["success_rate"]:
        winner = "Strategy B (Cloudscraper)"
        confidence = (
            "HIGH"
            if strategy_b["success_rate"] - strategy_a["success_rate"] > 20
            else "MEDIUM"
        )
    else:
        winner = "TIE"
        confidence = "LOW"

    report = f"""# ASM Access Strategy Comparison Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Test Scope:** 10 diverse ASM journal URLs
**Total Tests:** 60 (10 URLs × 2 strategies × 3 attempts)

---

## 1. Executive Summary

### Statistical Results

- **Strategy A (Session-based):** {strategy_a['successes']}/{strategy_a['total']} success rate ({strategy_a['success_rate']:.1f}%)
- **Strategy B (Cloudscraper):** {strategy_b['successes']}/{strategy_b['total']} success rate ({strategy_b['success_rate']:.1f}%)

### Recommendation

**PRIMARY STRATEGY:** {winner}
**STATISTICAL CONFIDENCE:** {confidence}

**Rationale:**
"""

    if winner == "Strategy A (Session-based)":
        report += f"""- {strategy_a['success_rate']:.1f}% success rate vs {strategy_b['success_rate']:.1f}% (Δ = {strategy_a['success_rate'] - strategy_b['success_rate']:.1f}%)
- Average latency: {strategy_a['avg_response_time']:.2f}s vs {strategy_b['avg_response_time']:.2f}s
- Consistency (std dev): {strategy_a['std_response_time']:.2f}s vs {strategy_b['std_response_time']:.2f}s
- Lightweight dependencies (requests only vs cloudscraper)
"""
    elif winner == "Strategy B (Cloudscraper)":
        report += f"""- {strategy_b['success_rate']:.1f}% success rate vs {strategy_a['success_rate']:.1f}% (Δ = {strategy_b['success_rate'] - strategy_a['success_rate']:.1f}%)
- Average latency: {strategy_b['avg_response_time']:.2f}s vs {strategy_a['avg_response_time']:.2f}s
- Consistency (std dev): {strategy_b['std_response_time']:.2f}s vs {strategy_a['std_response_time']:.2f}s
- Already integrated in docling_service.py
"""
    else:
        report += f"""- Both strategies achieved identical success rates: {strategy_a['success_rate']:.1f}%
- Consider using Strategy A (Session-based) for lighter dependencies
- Or keep Strategy B (Cloudscraper) since it's already integrated
"""

    # Integration recommendation
    report += f"""

### Integration Pattern

"""

    if (
        strategy_a["success_rate"] >= 80
        and strategy_a["success_rate"] >= strategy_b["success_rate"]
    ):
        report += """**Recommended:** Replace cloudscraper with session-based approach for ASM domains.

```python
# In docling_service.py
def _fetch_asm_article(self, url: str) -> str:
    \"\"\"ASM-specific access strategy.\"\"\"
    from asm_access_solution import fetch_asm_article_with_retry
    return fetch_asm_article_with_retry(url, max_retries=3)
```
"""
    elif strategy_b["success_rate"] >= 80:
        report += """**Recommended:** Keep cloudscraper as primary strategy (already working well).

Consider adding homepage visit + delay pattern to improve reliability further.
"""
    else:
        report += """**Recommended:** Implement hybrid approach with retry logic.

```python
# Try Strategy A first, fallback to Strategy B
try:
    return fetch_with_session(url)
except:
    return fetch_with_cloudscraper(url)
```
"""

    # Per-URL results table
    report += f"""

---

## 2. Per-URL Results

| URL | Journal | Strategy A | Strategy B | Winner |
|-----|---------|------------|------------|--------|
"""

    for comparison in analysis["url_comparison"]:
        journal_short = comparison["journal"].split("(")[0].strip()[:30]
        url_short = comparison["url"].split("/")[-1][:30]
        report += f"| {url_short} | {journal_short} | {comparison['strategy_a_success']} | {comparison['strategy_b_success']} | {comparison['winner']} |\n"

    # Aggregate statistics
    report += f"""

---

## 3. Aggregate Statistics

| Metric | Strategy A (Session) | Strategy B (Cloudscraper) | Winner |
|--------|----------------------|---------------------------|--------|
| **Success Rate** | {strategy_a['success_rate']:.1f}% ({strategy_a['successes']}/{strategy_a['total']}) | {strategy_b['success_rate']:.1f}% ({strategy_b['successes']}/{strategy_b['total']}) | {'A' if strategy_a['success_rate'] > strategy_b['success_rate'] else 'B' if strategy_b['success_rate'] > strategy_a['success_rate'] else 'Tie'} |
| **Avg Latency** | {strategy_a['avg_response_time']:.2f}s | {strategy_b['avg_response_time']:.2f}s | {'A' if strategy_a['avg_response_time'] < strategy_b['avg_response_time'] else 'B' if strategy_b['avg_response_time'] < strategy_a['avg_response_time'] else 'Tie'} |
| **Consistency (σ)** | {strategy_a['std_response_time']:.2f}s | {strategy_b['std_response_time']:.2f}s | {'A' if strategy_a['std_response_time'] < strategy_b['std_response_time'] else 'B' if strategy_b['std_response_time'] < strategy_a['std_response_time'] else 'Tie'} |
| **Failures** | {strategy_a['failures']} | {strategy_b['failures']} | {'A' if strategy_a['failures'] < strategy_b['failures'] else 'B' if strategy_b['failures'] < strategy_a['failures'] else 'Tie'} |
| **Complexity** | Low (requests only) | Medium (cloudscraper) | A |

"""

    # Failure analysis
    report += f"""

---

## 4. Failure Analysis

### Strategy A Failures ({strategy_a['failures']} total)

"""

    if strategy_a["failure_modes"]:
        for error, count in sorted(
            strategy_a["failure_modes"].items(), key=lambda x: x[1], reverse=True
        ):
            report += f"- **{error}:** {count} occurrences\n"
    else:
        report += "✅ No failures!\n"

    report += f"""

### Strategy B Failures ({strategy_b['failures']} total)

"""

    if strategy_b["failure_modes"]:
        for error, count in sorted(
            strategy_b["failure_modes"].items(), key=lambda x: x[1], reverse=True
        ):
            report += f"- **{error}:** {count} occurrences\n"
    else:
        report += "✅ No failures!\n"

    # Failure patterns analysis
    report += """

### Failure Pattern Analysis

"""

    # Analyze if failures are journal-specific
    url_failures_a = {}
    url_failures_b = {}

    for result in results:
        if not result.success:
            if result.strategy == "A_session":
                url_failures_a[result.journal] = (
                    url_failures_a.get(result.journal, 0) + 1
                )
            else:
                url_failures_b[result.journal] = (
                    url_failures_b.get(result.journal, 0) + 1
                )

    if url_failures_a or url_failures_b:
        report += "**Journal-specific failures detected:**\n\n"
        all_journals = set(list(url_failures_a.keys()) + list(url_failures_b.keys()))
        for journal in sorted(all_journals):
            a_count = url_failures_a.get(journal, 0)
            b_count = url_failures_b.get(journal, 0)
            report += (
                f"- **{journal}:** Strategy A: {a_count}/3, Strategy B: {b_count}/3\n"
            )
    else:
        report += "✅ No journal-specific patterns detected. Failures appear random/transient.\n"

    # Performance comparison
    report += f"""

---

## 5. Performance Comparison

### Latency Distribution

**Strategy A (Session-based):**
- Successful requests: {len([r for r in results if r.strategy == 'A_session' and r.success])}
- Average: {strategy_a['avg_response_time']:.2f}s
- Std Dev: {strategy_a['std_response_time']:.2f}s
- Range: {min([r.response_time for r in results if r.strategy == 'A_session' and r.success], default=0):.2f}s - {max([r.response_time for r in results if r.strategy == 'A_session' and r.success], default=0):.2f}s

**Strategy B (Cloudscraper):**
- Successful requests: {len([r for r in results if r.strategy == 'B_cloudscraper' and r.success])}
- Average: {strategy_b['avg_response_time']:.2f}s
- Std Dev: {strategy_b['std_response_time']:.2f}s
- Range: {min([r.response_time for r in results if r.strategy == 'B_cloudscraper' and r.success], default=0):.2f}s - {max([r.response_time for r in results if r.strategy == 'B_cloudscraper' and r.success], default=0):.2f}s

### Reliability Assessment

**Strategy A:** {'EXCELLENT' if strategy_a['success_rate'] >= 95 else 'GOOD' if strategy_a['success_rate'] >= 80 else 'FAIR' if strategy_a['success_rate'] >= 60 else 'POOR'}
**Strategy B:** {'EXCELLENT' if strategy_b['success_rate'] >= 95 else 'GOOD' if strategy_b['success_rate'] >= 80 else 'FAIR' if strategy_b['success_rate'] >= 60 else 'POOR'}

"""

    # Implementation recommendation
    report += f"""

---

## 6. Implementation Recommendation

### Production Deployment Strategy

"""

    if strategy_a["success_rate"] >= 80:
        report += """**Primary:** Strategy A (Session-based)
**Fallback:** Strategy B (Cloudscraper)
**Retry Logic:** 3 attempts with exponential backoff

### Integration Steps

1. **Move asm_access_solution.py to production location**
   ```bash
   mv tests/manual/asm_access_solution.py lobster/services/data_access/
   ```

2. **Update docling_service.py**
   ```python
   from lobster.services.data_access.asm_access_solution import is_asm_url, fetch_asm_article_with_retry

   def _fetch_with_domain_specific_strategy(self, url: str) -> str:
       if is_asm_url(url):
           return fetch_asm_article_with_retry(url, max_retries=3)
       return self._fetch_with_cloudscraper(url)
   ```

3. **Add monitoring**
   ```python
   # Track ASM access metrics
   self._log_access_metrics(domain="journals.asm.org", success=True, latency=4.2)
   ```

4. **Configure rate limiting**
   ```python
   # In rate_limiter.py
   "journals.asm.org": {"max_requests": 10, "time_window": 60, "delay": 2.0}
   ```
"""
    elif strategy_b["success_rate"] >= 80:
        report += """**Primary:** Strategy B (Cloudscraper - current)
**Enhancement:** Add homepage visit + delay pattern
**Retry Logic:** 3 attempts with exponential backoff

### Integration Steps

1. **Enhance cloudscraper implementation**
   ```python
   def _fetch_asm_with_cloudscraper(self, url: str) -> str:
       scraper = cloudscraper.create_scraper(...)
       # Visit homepage first
       scraper.get("https://journals.asm.org/", timeout=30)
       time.sleep(2)
       # Fetch article
       response = scraper.get(url, timeout=30)
       return response.text
   ```

2. **Add retry logic**
   ```python
   for attempt in range(3):
       try:
           return self._fetch_asm_with_cloudscraper(url)
       except Exception as e:
           if attempt < 2:
               time.sleep(5 * (2 ** attempt))
           else:
               raise
   ```
"""
    else:
        report += """**PRIMARY:** Hybrid approach (try both strategies)
**FALLBACK:** PMC-first strategy (most ASM papers in PMC)
**ALERT:** Both strategies show <80% success rate - needs investigation

### Recommended Actions

1. **Investigate failure patterns**
   - Are failures transient or permanent?
   - Is rate limiting triggering blocks?
   - Do failures cluster by journal?

2. **Implement hybrid strategy**
   ```python
   def _fetch_asm_hybrid(self, url: str) -> str:
       # Try Strategy A first (lighter weight)
       try:
           return fetch_with_session(url)
       except Exception:
           # Fallback to Strategy B
           return fetch_with_cloudscraper(url)
   ```

3. **Strengthen PMC-first fallback**
   - Most ASM articles available in PMC
   - Zero bot protection, official API
"""

    # Testing checklist
    report += """

---

## 7. Testing Checklist

Before production deployment:

"""

    checklist_items = [
        f"[{'x' if strategy_a['success_rate'] >= 80 or strategy_b['success_rate'] >= 80 else ' '}] At least one strategy achieves ≥80% success rate",
        "[ ] Integration test with docling_service.py",
        "[ ] Test with publication queue (10-100 articles)",
        "[ ] Verify error caching behavior (24h TTL)",
        "[ ] Test retry logic with simulated failures",
        "[ ] Confirm rate limiting prevents IP blocks",
        "[ ] Load testing (50+ requests over 5 minutes)",
        "[ ] Monitor ASM access metrics for 1 week",
        "[ ] Document fallback procedure (PMC-first)",
        "[ ] Add alerting for success rate drops below 80%",
    ]

    for item in checklist_items:
        report += f"- {item}\n"

    # Raw data reference
    report += f"""

---

## 8. Raw Data

**Full test results:** See `asm_strategy_comparison_data.json`
**Test script:** `test_asm_strategies_comparison.py`
**Test URLs:** `asm_test_urls.py`

### Data Summary

- Total tests run: {len(results)}
- Unique URLs tested: {len(set(r.url for r in results))}
- Journals covered: {len(set(r.journal for r in results))}
- Test duration: ~{len(results) * 5 / 60:.0f} minutes (with rate limiting)

---

## Conclusion

**Statistical confidence: {confidence}**

Based on {len(results)} tests across {len(ASM_TEST_URLS)} diverse ASM journal articles, we {'have high confidence' if confidence == 'HIGH' else 'have moderate confidence' if confidence == 'MEDIUM' else 'recommend further testing'} that **{winner}** is the optimal approach for production deployment.

"""

    if strategy_a["success_rate"] >= 80:
        report += """**Key Takeaways:**
- Session-based approach is reliable, lightweight, and maintainable
- No heavy dependencies (requests vs cloudscraper)
- Consistent performance across different ASM journals
- Ready for immediate integration

**Next Steps:**
1. Integrate session-based strategy into docling_service.py
2. Add monitoring and alerting
3. Document in ASM_QUICK_REFERENCE.md
4. Monitor production metrics for 1 week
"""
    elif strategy_b["success_rate"] >= 80:
        report += """**Key Takeaways:**
- Cloudscraper continues to work well for ASM
- Already integrated, minimal changes needed
- Can enhance with homepage visit pattern for even better reliability

**Next Steps:**
1. Enhance cloudscraper implementation with session establishment
2. Add retry logic with exponential backoff
3. Monitor production metrics for 1 week
"""
    else:
        report += """**Key Takeaways:**
- Both strategies show concerning failure rates
- Hybrid approach recommended
- PMC-first strategy critical as safety net

**Next Steps:**
1. Investigate root causes of failures
2. Implement hybrid strategy
3. Strengthen PMC fallback
4. Consider reaching out to ASM for API access
"""

    report += """

---

**Report generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Testing duration:** ~{len(results) * 5 / 60:.0f} minutes
**Test methodology:** Statistical validation with 3 attempts per URL per strategy
"""

    return report


def main():
    """Main execution function."""

    # Run comparison tests
    print("\n🚀 Starting ASM strategy comparison...")
    results = run_comparison()

    # Analyze results
    print("\n📊 Analyzing results...")
    analysis = analyze_results(results)

    # Generate report
    print("\n📝 Generating comprehensive report...")
    report = generate_report(results, analysis)

    # Save report
    report_path = Path(__file__).parent / "ASM_STRATEGY_COMPARISON.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"✅ Report saved to: {report_path}")

    # Save raw data
    data_path = Path(__file__).parent / "asm_strategy_comparison_data.json"
    with open(data_path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"✅ Raw data saved to: {data_path}")

    # Print summary to console
    print("\n" + "=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)
    print(
        f"Strategy A (Session-based): {analysis['strategy_a']['successes']}/{analysis['strategy_a']['total']} ({analysis['strategy_a']['success_rate']:.1f}%)"
    )
    print(
        f"Strategy B (Cloudscraper):  {analysis['strategy_b']['successes']}/{analysis['strategy_b']['total']} ({analysis['strategy_b']['success_rate']:.1f}%)"
    )
    print()
    print(f"Average latency:")
    print(
        f"  Strategy A: {analysis['strategy_a']['avg_response_time']:.2f}s (±{analysis['strategy_a']['std_response_time']:.2f}s)"
    )
    print(
        f"  Strategy B: {analysis['strategy_b']['avg_response_time']:.2f}s (±{analysis['strategy_b']['std_response_time']:.2f}s)"
    )
    print("=" * 80)
    print(f"\n✅ Full report: {report_path}")
    print(f"✅ Raw data: {data_path}")


if __name__ == "__main__":
    main()
