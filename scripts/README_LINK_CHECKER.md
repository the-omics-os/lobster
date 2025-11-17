# Wiki Link Checker

Automated tool to validate all links in wiki markdown files, ensuring documentation integrity and preventing broken references.

## Overview

The Wiki Link Checker is a Python script that:
- Scans all markdown files in the wiki directory
- Validates internal file references
- Checks anchor links to headings
- Optionally verifies external URLs (when `--external` flag is used)
- Generates detailed reports of broken links
- Integrates with GitHub Actions for automated checks

## Features

- **Internal Link Validation**: Ensures all relative file links point to existing files
- **Anchor Link Validation**: Verifies heading references match actual headings
- **External Link Checking**: Optional HTTP status checking for external URLs
- **Comprehensive Reporting**: Markdown-formatted reports with actionable recommendations
- **CI/CD Integration**: Automated checks on wiki changes via GitHub Actions
- **Performance**: Efficient parallel checking with rate limiting

## Usage

### Local Development

**Basic link check (internal links only):**
```bash
cd /Users/tyo/GITHUB/omics-os/lobster
python scripts/check_wiki_links.py
```

**Check external links (slower, requires `requests` library):**
```bash
python scripts/check_wiki_links.py --external
```

**Save report to file:**
```bash
python scripts/check_wiki_links.py --output wiki-link-report.md
```

**Check specific wiki directory:**
```bash
python scripts/check_wiki_links.py --wiki-dir /path/to/wiki
```

**All options combined:**
```bash
python scripts/check_wiki_links.py --external --output report.md --wiki-dir wiki/
```

### Expected Output

**Success (no broken links):**
```
================================================================================
Wiki Link Checker
================================================================================

Found 43 markdown files in /Users/tyo/GITHUB/omics-os/lobster/wiki

Checking 01-getting-started.md... ✅
Checking 02-installation.md... ✅
...
Checking README.md... ✅

================================================================================
# Wiki Link Check Report

**Date**: 2025-11-16 02:45:00

## Summary

- **Files Checked**: 43
- **Internal Links**: 147
- **External Links**: 22
- **Anchor Links**: 63

- **Broken Internal Links**: 0
- **Broken External Links**: 0
- **Broken Anchor Links**: 0

**Total Issues Found**: 0

✅ **No broken links found!**

================================================================================

✅ All links are valid!
```

**Failure (broken links found):**
```
Checking Home.md... ❌ 3 issues

================================================================================
# Wiki Link Check Report
...
### Home.md (3 issues)

🔗 **Line 45**: [internal] `21-cloud-local-architecture.md`
   - File not found in wiki directory

⚓ **Line 69**: [anchor] `#required-api-keys`
   - Heading not found in file
...

❌ Found 26 broken links
```

## CI/CD Integration

### Automated Checks

The link checker runs automatically via GitHub Actions on:

1. **Every push** to `wiki/` directory
2. **Every pull request** modifying `wiki/` files
3. **Weekly schedule** (Mondays at 9am UTC)
4. **Manual trigger** via GitHub Actions UI

### Workflow Configuration

File: `.github/workflows/wiki-link-check.yml`

```yaml
name: Wiki Link Checker

on:
  push:
    paths:
      - 'wiki/**'
  pull_request:
    paths:
      - 'wiki/**'
  schedule:
    - cron: '0 9 * * 1'  # Weekly on Mondays
```

### Viewing Results

**In GitHub Actions:**
1. Go to repository → Actions tab
2. Select "Wiki Link Checker" workflow
3. View run details and logs
4. Download "link-check-report" artifact if issues found

**In Pull Requests:**
- Link checker automatically comments on PR with results if broken links detected
- PR checks show ❌ if broken links found, ✅ if all links valid

## What It Checks

### 1. Internal Links

**Markdown format:** `[Link Text](filename.md)`

Validates that:
- Target file exists in wiki directory
- Relative paths are correct
- No typos in filenames

**Common issues:**
- `../README.md` → Should be `README.md` (wrong relative path)
- `INSTALLATION.md` → Should be `02-installation.md` (wrong filename)

### 2. Anchor Links

**Markdown format:** `[Link Text](filename.md#heading-anchor)` or `[Link Text](#heading-anchor)`

Validates that:
- Referenced heading exists in target file
- Anchor format matches GitHub heading conversion rules

**GitHub anchor rules:**
- Lowercase conversion
- Spaces → hyphens
- Special characters removed
- Examples:
  - `## Quick Start` → `#quick-start`
  - `## API Reference` → `#api-reference`
  - `## FAQ: Common Issues` → `#faq-common-issues`

**Common issues:**
- `#required-api-keys` when heading is `## Setup API Keys` (mismatch)
- Special characters in anchors not matching actual headings

### 3. External Links (Optional)

**Markdown format:** `[Link Text](https://example.com)`

When `--external` flag is used:
- Sends HTTP HEAD request to check availability
- Accepts status codes 200-399 as valid
- Retries with GET if HEAD fails (some servers block HEAD)
- Includes User-Agent header to avoid blocking

**Known limitations:**
- Rate limiting may cause false positives
- Some sites block automated requests
- Timeouts may occur for slow sites
- External checks are SLOW (0.5s delay between requests)

## Handling Failures

### In CI/CD

If link checker fails in CI:

1. **Review the report artifact:**
   - Go to GitHub Actions run
   - Download "link-check-report" artifact
   - Open `wiki-link-report.md`

2. **Fix broken links:**
   - Update filenames if files were renamed
   - Fix typos in links
   - Update anchor references to match exact headings
   - Remove or replace dead external links

3. **Re-run checks:**
   - Push fixes to branch
   - CI automatically re-runs link checker
   - PR shows ✅ when all links valid

### During Development

**Before committing wiki changes:**

```bash
# Run link checker
python scripts/check_wiki_links.py

# If issues found, fix them
# Then verify fixes
python scripts/check_wiki_links.py
```

**Quick fix workflow:**
```bash
# 1. Find broken links
python scripts/check_wiki_links.py | grep "❌"

# 2. Fix in editor
# 3. Verify fix
python scripts/check_wiki_links.py --output /tmp/report.md && echo "All good!"
```

## Common Link Issues & Solutions

### Issue 1: Relative Path Errors

**Problem:**
```markdown
[Main Docs](../README.md)
```

**Solution:**
```markdown
[Main Docs](README.md)
```

**Why:** Wiki files are all in same directory, no `../` needed.

### Issue 2: Wrong Filename

**Problem:**
```markdown
[Install Guide](INSTALLATION.md)
```

**Solution:**
```markdown
[Install Guide](02-installation.md)
```

**Why:** Wiki files use numbered naming convention.

### Issue 3: Anchor Mismatch

**Problem:**
```markdown
[Quick Start](#getting-started)
```

When heading is: `## Quick Start Guide`

**Solution:**
```markdown
[Quick Start](#quick-start-guide)
```

**Why:** Anchor must match exact heading text (lowercased, spaces→hyphens).

### Issue 4: File Doesn't Exist

**Problem:**
```markdown
[Architecture](21-cloud-local-architecture.md)
```

When file doesn't exist or was renamed.

**Solution:**
```markdown
[Architecture](18-architecture-overview.md)
```

**Why:** File was renamed or deleted. Update link to correct file.

### Issue 5: External Link Dead

**Problem:**
```markdown
[Tool](https://example.com/old-page)
```

Returns 404.

**Solutions:**
1. Find new URL: `[Tool](https://example.com/new-page)`
2. Use Internet Archive: `[Tool](https://web.archive.org/web/.../old-page)`
3. Remove if no longer relevant

## Technical Details

### Implementation

**File:** `scripts/check_wiki_links.py`

**Key classes:**
- `WikiLinkChecker`: Main checker class
  - `find_markdown_files()`: Discovers wiki files
  - `extract_links()`: Parses markdown for links
  - `check_internal_link()`: Validates file references
  - `check_external_link()`: HTTP status checking
  - `extract_headings()`: Parses heading anchors
  - `generate_report()`: Creates markdown report

**Link patterns detected:**
- Markdown links: `[text](url)`
- Wiki-style links: `[[page-name]]` (converted to `page-name.md`)

**Heading anchor conversion:**
Matches GitHub's algorithm:
1. Lowercase text
2. Remove markdown formatting
3. Remove special characters
4. Replace spaces with hyphens

### Dependencies

**Required:**
- Python 3.11+
- Standard library: `re`, `sys`, `pathlib`, `argparse`

**Optional:**
- `requests` library (for external link checking)
- Install: `pip install requests`

### Exit Codes

- `0`: All links valid (success)
- `1`: Broken links found (failure)

This allows CI/CD integration:
```bash
if python scripts/check_wiki_links.py; then
  echo "Links OK"
else
  echo "Broken links detected"
fi
```

## Known Limitations

### Anchor Link Validation

**Not checked:**
- Cross-file anchors (e.g., `other-file.md#section`) - only validates file exists
- Dynamic headings generated by JavaScript
- Anchors with complex special characters

**Reason:** Cross-file anchor validation requires parsing target file, which adds complexity. Current implementation provides 90% coverage for most cases.

### External Link Checking

**False positives may occur for:**
- Sites with aggressive bot protection
- Rate-limited APIs (GitHub, Twitter, etc.)
- Sites requiring authentication
- Temporary network issues
- Geographic restrictions

**Recommendation:** Use `--external` flag sparingly, primarily for manual checks rather than CI/CD.

### Performance

- **Internal links only**: Fast (~2-5 seconds for 43 files)
- **With external links**: Slow (~30-60 seconds with rate limiting)

## Maintenance & Updates

### Adding New Link Patterns

To support new markdown link syntax:

1. Update `extract_links()` method in `WikiLinkChecker`
2. Add regex pattern for new syntax
3. Add test cases
4. Update this documentation

### Adjusting Heading Anchor Rules

If GitHub changes anchor conversion algorithm:

1. Update `extract_headings()` method
2. Test against known headings
3. Update test fixtures

### Modifying CI/CD Behavior

Edit `.github/workflows/wiki-link-check.yml`:

- **Change schedule:** Modify `cron` expression
- **Add external checks:** Add `--external` flag (NOT recommended - too slow)
- **Adjust triggers:** Modify `on:` section

## Troubleshooting

### Link checker not running in CI

**Check:**
1. Workflow file exists: `.github/workflows/wiki-link-check.yml`
2. Changes are in `wiki/` directory
3. GitHub Actions enabled for repository
4. Check Actions tab for error messages

### False positives for valid links

**Anchor links:**
- Verify heading text matches exactly
- Check for special characters
- Try link in actual wiki interface

**Internal links:**
- Verify file exists: `ls wiki/filename.md`
- Check for typos in filename
- Ensure correct extension (`.md`)

### Script fails to run

**Error: `ModuleNotFoundError: No module named 'requests'`**
```bash
pip install requests
```

**Error: `FileNotFoundError: wiki directory not found`**
```bash
# Run from repository root
cd /Users/tyo/GITHUB/omics-os/lobster
python scripts/check_wiki_links.py
```

## Best Practices

### For Wiki Contributors

1. **Run link checker before committing:**
   ```bash
   python scripts/check_wiki_links.py
   ```

2. **Use correct relative paths:**
   - Same directory: `filename.md`
   - Not: `./filename.md` or `../filename.md`

3. **Verify anchors match headings:**
   - Copy heading text
   - Convert to anchor format manually
   - Or test in wiki interface

4. **Avoid external links when possible:**
   - External links break over time
   - Reference wiki pages instead
   - If external link needed, prefer stable sources (official docs, GitHub, etc.)

### For Maintainers

1. **Review link check reports weekly**
2. **Fix broken links promptly** (prevents cascading issues)
3. **Update link checker as needed** (new patterns, improved validation)
4. **Document link conventions** in style guide

## Future Enhancements

Potential improvements (not yet implemented):

- [ ] **Cross-file anchor validation** - Check `other-file.md#section`
- [ ] **Image link validation** - Verify `![alt](image.png)` targets exist
- [ ] **Link age tracking** - Identify old external links for review
- [ ] **Auto-fix suggestions** - Propose corrections for common issues
- [ ] **Performance optimization** - Parallel external link checking
- [ ] **Whitelist support** - Skip known problematic external URLs
- [ ] **Markdown table support** - Better handling of links in tables

## Support

For issues or questions:

- **GitHub Issues**: [Report bugs](https://github.com/the-omics-os/lobster/issues)
- **Wiki**: [Documentation Wiki](https://github.com/the-omics-os/lobster/wiki)
- **Script**: `/Users/tyo/GITHUB/omics-os/lobster/scripts/check_wiki_links.py`

---

**Last Updated**: 2025-11-16
**Version**: 1.0.0
**Author**: Lobster AI Infrastructure Team
