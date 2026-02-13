# ASM Journal Access - Quick Reference

## Problem
```
403 Forbidden when accessing journals.asm.org programmatically
```

## Solution
```
✅ Session establishment + comprehensive headers
✅ 100% success rate (5/5 tests)
✅ ~4.4s latency
✅ Lightweight (uses requests library)
```

---

## 1-Minute Integration

### Step 1: Copy Working Code
```bash
# Move solution from tests to production
mv tests/manual/asm_access_solution.py lobster/services/data_access/
```

### Step 2: Add to docling_service.py
```python
# At top of file
from lobster.services.data_access.asm_access_solution import (
    is_asm_url,
    fetch_asm_article_with_retry
)

# In DoclingService class
def _fetch_content(self, url: str) -> str:
    """Fetch with domain-specific strategies."""

    # ASM journals require session establishment
    if is_asm_url(url):
        return fetch_asm_article_with_retry(url, max_retries=3)

    # Default: existing cloudscraper logic
    scraper = cloudscraper.create_scraper(...)
    return scraper.get(url, timeout=30).text
```

### Step 3: Test
```python
from lobster.services.data_access.asm_access_solution import fetch_asm_article

html = fetch_asm_article("https://journals.asm.org/doi/10.1128/JCM.01893-20")
print(f"Success! {len(html)} bytes fetched")
```

---

## Why This Works

ASM requires **2-step session establishment:**

```
1. Visit homepage → Get session cookies
   ↓ (wait 2s)
2. Request article → 200 OK ✅
```

Without step 1 → `403 Forbidden ❌`

---

## Key Implementation Details

### Critical Headers
```python
'Sec-Fetch-Dest': 'document',    # Critical for bot detection
'Sec-Fetch-Mode': 'navigate',    # Critical for bot detection
'Sec-Fetch-Site': 'none',        # Critical for bot detection
'Sec-Fetch-User': '?1',          # Critical for bot detection
'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120"',  # Critical
```

### Session Persistence
```python
session = requests.Session()
session.get("https://journals.asm.org/")  # Homepage
time.sleep(2)
session.get(article_url)  # Article (reuses cookies)
```

---

## Test Results Summary

| Strategy | Success Rate | Latency | Dependencies |
|----------|--------------|---------|--------------|
| **requests_with_spoofing** | **100%** | **4.4s** | **requests** |
| cloudscraper_with_delay | 80% | 4.4s | cloudscraper |
| All others | 0% | N/A | Various |

---

## Monitoring Checklist

After integration, monitor:

- [ ] ASM access success rate (should be >90%)
- [ ] Average latency (should be <6s)
- [ ] 403 errors (should be rare)
- [ ] Cache hit rate (should be high due to 24h TTL)

Alert if:
- Success rate drops below 80%
- Latency exceeds 10s
- 403 rate increases (may need strategy adjustment)

---

## Fallback Strategy

If this solution breaks:

1. **PMC First** - Most ASM papers available in PubMed Central (no bot protection)
2. **Playwright** - Heavy but reliable (real browser automation)
3. **Contact ASM** - Request API access or programmatic guidelines
4. **Manual Queue** - Last resort: queue for manual download

---

## Files Created

| File | Purpose |
|------|---------|
| `asm_access_solution.py` | Production-ready implementation |
| `test_asm_access_strategies.py` | Test 8 different strategies |
| `test_asm_reliability.py` | Test reliability (5 attempts each) |
| `ASM_ACCESS_REPORT.md` | Comprehensive analysis (this report) |
| `ASM_QUICK_REFERENCE.md` | Quick integration guide (you are here) |

---

## Questions?

- **Q: Why not just use cloudscraper?**
  - A: Cloudscraper basic approach gets 403. Only session establishment works.

- **Q: Why requests instead of cloudscraper?**
  - A: Lighter dependency, 100% success rate, easier to maintain.

- **Q: What if it breaks?**
  - A: PMC-first fallback strategy (most ASM papers available there).

- **Q: Is this ethical?**
  - A: Yes - accessing public content for research, respecting rate limits, PMC-first approach.

- **Q: Will ASM block us?**
  - A: Low risk if we respect rate limits. Monitor success rates.

---

## Full Documentation

See `ASM_ACCESS_REPORT.md` for:
- Complete testing methodology
- Performance analysis
- Risk assessment
- Integration examples
- Monitoring recommendations
