# Rate Limiter Arena - Testing Guide

## Overview

The **Rate Limiter Arena** is a comprehensive testing framework for evaluating publisher access strategies. It dynamically reads configurations from `rate_limiter.py` and runs comparative tests to validate current strategies and discover optimal approaches.

**Location**: `lobster/tools/rate_limiter_arena.py`

## Big Picture Goal

**Understand which strategies make most sense for each publisher category.**

The arena helps answer:
- Is our current strategy still working for Publisher X?
- Which strategy has the highest success rate for hard-to-reach publishers?
- Should we upgrade/downgrade strategy levels based on test results?
- Are there publishers that need SESSION strategy like ASM?

---

## Quick Start

### Run Full Arena Test (All Publishers, All Strategies)

```bash
cd /Users/tyo/GITHUB/omics-os/lobster
python -m lobster.tools.rate_limiter_arena
```

**Expected Duration**: ~10-15 minutes (depends on number of URLs × strategies × attempts)

### Test Specific Publisher

```bash
python -m lobster.tools.rate_limiter_arena --publisher "Nature"
python -m lobster.tools.rate_limiter_arena --publisher "ASM"
python -m lobster.tools.rate_limiter_arena -p "Cell"
```

### Test Specific Strategy Across All Publishers

```bash
python -m lobster.tools.rate_limiter_arena --strategy SESSION
python -m lobster.tools.rate_limiter_arena -s STEALTH
python -m lobster.tools.rate_limiter_arena -s BROWSER
```

### List Available Test URLs

```bash
python -m lobster.tools.rate_limiter_arena --list-urls
```

### Show Current Domain Configurations

```bash
python -m lobster.tools.rate_limiter_arena --show-config
```

---

## Strategy Types

The arena tests 5 different access strategies:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **DEFAULT** | Minimal headers (requests default) | NCBI, official APIs |
| **POLITE** | Bot-friendly headers with LobsterAI UA | Open access publishers |
| **BROWSER** | Full browser headers (Chrome UA) | Moderate bot detection |
| **STEALTH** | cloudscraper + Sec-Fetch-* headers | Cloudflare-protected sites |
| **SESSION** | Homepage visit + cookies + Sec-Fetch-* | Advanced bot detection (ASM) |

---

## Test URL Registry

The arena maintains a curated list of **hard-to-reach publishers** for testing:

**Current Publishers** (as of 2025-12-01):
- ASM Journals (SESSION strategy - validated 93.3% success)
- Cell Press (STEALTH)
- Nature (BROWSER)
- MDPI (POLITE)
- PubMed Central (DEFAULT)
- And more...

**To add new test URLs**, edit `rate_limiter_arena.py`:

```python
TEST_URLS.append(
    TestURL(
        url="https://publisher.com/article/12345",
        publisher="Publisher Name",
        domain="publisher.com",
        expected_strategy=HeaderStrategy.STEALTH,
        year=2024,
        notes="Why this URL is challenging"
    )
)
```

---

## Output Format

### Terminal Output

```
================================================================================
RATE LIMITER ARENA - Full Test Suite
================================================================================
Test URLs: 10
Attempts per URL: 3
Total tests: 150
Estimated time: ~13 minutes
================================================================================

[1/10] Testing: ASM - Journal of Clinical Microbiology
URL: https://journals.asm.org/doi/10.1128/JCM.01893-20...
Expected strategy: session

  Attempt 1/3:
  [default ] ✗ (403) 1.23s
  [polite  ] ✗ (403) 1.45s
  [browser ] ✗ (403) 1.67s
  [stealth ] ✗ (403) 2.34s
  [session ] ✓ (200) 4.56s

  Attempt 2/3:
  ...
```

### Generated Files

Arena generates two files per test run:

1. **JSON Data** (`arena_results_YYYYMMDD_HHMMSS.json`)
   - Raw test results
   - Complete metrics (status codes, latencies, errors)
   - Machine-readable for further analysis

2. **Markdown Report** (`arena_report_YYYYMMDD_HHMMSS.md`)
   - Executive summary
   - Strategy performance comparison table
   - Per-publisher results
   - Recommendations for strategy changes

**Output Location**: `tests/manual/arena_results/`

---

## Report Structure

### 1. Strategy Performance Summary

```markdown
| Strategy | Success Rate | Avg Latency |
|----------|--------------|-------------|
| default  | 20.0%        | 1.23s       |
| polite   | 40.0%        | 1.45s       |
| browser  | 60.0%        | 1.67s       |
| stealth  | 80.0%        | 2.34s       |
| session  | 93.3%        | 4.56s       |
```

### 2. Publisher Results

For each publisher:
- URL tested
- Domain
- Expected strategy (from DOMAIN_CONFIG)
- Recommended strategy (from test results)
- Success rates for all strategies

### 3. Recommendations

Actionable recommendations:
- "Publisher X: Change from BROWSER to SESSION (success rate: 33% → 93%)"
- "Publisher Y: STEALTH strategy working as expected (100% success)"

---

## Common Workflows

### Workflow 1: Validate Current Configurations

**Goal**: Ensure all publishers in DOMAIN_CONFIG are still accessible with their assigned strategies.

```bash
# Run full arena test
python -m lobster.tools.rate_limiter_arena

# Review report
cat tests/manual/arena_results/arena_report_*.md

# Look for mismatches between expected vs recommended strategies
```

### Workflow 2: Discover Strategy for New Publisher

**Goal**: Find optimal strategy for a new challenging publisher.

```bash
# 1. Add test URL to TEST_URLS in rate_limiter_arena.py
# 2. Run arena test
python -m lobster.tools.rate_limiter_arena --publisher "New Publisher"

# 3. Review which strategy had highest success rate
# 4. Update DOMAIN_CONFIG in rate_limiter.py with recommended strategy
```

### Workflow 3: Debug Publisher Access Issues

**Goal**: Troubleshoot why a publisher is returning 403 errors.

```bash
# Test all strategies against failing URL
python -m lobster.tools.rate_limiter_arena --publisher "Failing Publisher"

# Check report for:
# - Which strategies succeed (if any)
# - Error messages (timeout, 403, 502, etc.)
# - Latency patterns (slow = possible rate limiting)
```

### Workflow 4: Monthly Health Check

**Goal**: Proactive monitoring of publisher access patterns.

```bash
# Run full arena test monthly
python -m lobster.tools.rate_limiter_arena > monthly_check.log

# Compare success rates month-over-month
# Alert if any publisher drops below 80% success rate
```

---

## Best Practices

### Rate Limiting

The arena automatically adds **5-second delays** between tests to respect rate limits. For faster testing during development:

```python
# In rate_limiter_arena.py, adjust delay_between parameter
results = self.tester.test_all_strategies(test_url.url, delay_between=2.0)
```

### Statistical Significance

Default is **3 attempts per URL per strategy** for statistical validity:
- 1 attempt: Not statistically meaningful
- 3 attempts: Good for catching intermittent failures (recommended)
- 5 attempts: Better confidence but longer runtime

Adjust with `--attempts` flag:

```bash
python -m lobster.tools.rate_limiter_arena --attempts 5
```

### Test URL Quality

When adding new test URLs:

1. **Verify accessibility**: Test URL manually in browser first
2. **Check for paywall**: Ensure article is not behind institutional login
3. **Use recent articles**: Publishers update bot detection (prefer 2020+)
4. **Diverse journals**: Don't add 10 URLs from same journal

### Interpreting Results

**Success rate thresholds**:
- **90-100%**: Excellent - strategy working perfectly
- **80-89%**: Good - acceptable with retry logic
- **70-79%**: Marginal - consider upgrading strategy
- **<70%**: Poor - definitely needs strategy change

**Latency guidelines**:
- **<2s**: Fast (DEFAULT, POLITE, BROWSER)
- **2-5s**: Acceptable (SESSION with homepage visit)
- **5-10s**: Slow (STEALTH with cloudscraper)
- **>10s**: Too slow - may indicate rate limiting or timeout

---

## Extending the Arena

### Add New Strategy Type

1. **Update HeaderStrategy enum** in `rate_limiter.py`:
   ```python
   class HeaderStrategy(str, Enum):
       # ... existing ...
       PLAYWRIGHT = "playwright"  # Real browser automation
   ```

2. **Implement test method** in `StrategyTester`:
   ```python
   def test_playwright(self, url: str, timeout: int = 30) -> StrategyResult:
       """Test PLAYWRIGHT strategy (real browser)."""
       # Implementation here
   ```

3. **Add to test_all_strategies** method:
   ```python
   elif strategy == HeaderStrategy.PLAYWRIGHT:
       result = self.test_playwright(url)
   ```

### Add Custom Metrics

Extend `StrategyResult` dataclass:

```python
@dataclass
class StrategyResult:
    # ... existing fields ...
    content_type: Optional[str] = None  # Track if PDF or HTML
    redirect_count: Optional[int] = None  # Track redirects
    cookies_received: Optional[int] = None  # Track session cookies
```

### Integration with CI/CD

Run arena tests in GitHub Actions:

```yaml
name: Publisher Access Health Check

on:
  schedule:
    - cron: '0 0 1 * *'  # Monthly on 1st

jobs:
  arena-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Arena Test
        run: python -m lobster.tools.rate_limiter_arena
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: arena-results
          path: tests/manual/arena_results/
```

---

## Troubleshooting

### ImportError: cloudscraper not installed

**Solution**: Install cloudscraper for STEALTH strategy testing:
```bash
pip install cloudscraper
```

Or skip STEALTH tests - arena will automatically skip if not available.

### Timeout errors

**Symptoms**: Most tests fail with timeout errors

**Causes**:
- Network connectivity issues
- Publisher rate limiting (too many requests)
- Firewall blocking requests

**Solutions**:
- Increase timeout: Edit `timeout=30` to `timeout=60` in test methods
- Reduce concurrent tests: Increase `delay_between` parameter
- Use VPN if publisher blocks your IP range

### All strategies fail for a URL

**Symptoms**: 100% failure across all strategies

**Likely causes**:
1. URL is invalid/404 (check in browser)
2. Publisher requires authentication (institutional login)
3. Publisher is completely paywalled
4. IP-based blocking (need proxy/VPN)

**Action**: Remove from TEST_URLS or mark as "paywalled" in notes

### SESSION strategy slower than expected

**Normal**: SESSION strategy includes homepage visit + 2s delay = ~4-5s latency

This is expected and acceptable for publishers that require session establishment.

---

## Related Files

- `lobster/tools/rate_limiter.py` - Strategy configurations (DOMAIN_CONFIG)
- `tests/manual/asm_test_urls.py` - ASM-specific test collection
- `tests/manual/asm_access_solution.py` - ASM SESSION strategy implementation
- `tests/manual/ASM_STRATEGY_COMPARISON.md` - ASM validation report (93.3% success)

---

## Future Enhancements

Planned features:
- [ ] Web UI for interactive testing
- [ ] Real-time dashboard (Grafana/Prometheus)
- [ ] Automated strategy recommendations (ML-based)
- [ ] Proxy/VPN support for geo-restricted publishers
- [ ] Historical trend analysis (track success rates over time)
- [ ] Slack/email alerts for failing publishers

---

## Questions?

Contact: Lobster AI Development Team
Documentation: `lobster/wiki/rate-limiter-arena.md`
Issues: https://github.com/the-omics-os/lobster/issues
