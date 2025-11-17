# Disabled GitHub Actions Workflows

This directory contains workflows that have been disabled to optimize GitHub Actions resource usage.

## Disabled Workflows

### Wiki Validation Workflows (Disabled: 2025-01-17)

These workflows validated wiki documentation but were disabled to reduce costs by ~40%:

- **wiki-code-test.yml** - Tests code examples in wiki docs
- **wiki-dashboard.yml** - Generates wiki health metrics
- **wiki-link-check.yml** - Checks for broken links
- **wiki-markdown-lint.yml** - Lints markdown formatting

**Manual Alternative:**
Run these checks manually before major documentation releases:

```bash
# Test wiki code examples
python scripts/test_wiki_code_examples.py

# Check wiki links
python scripts/check_wiki_links.py

# Generate wiki dashboard
python scripts/generate_wiki_dashboard.py

# Lint wiki markdown
python scripts/lint_wiki_markdown.py
```

**Re-enabling:** To re-enable any workflow, move it back to `.github/workflows/` directory.

## Deleted Workflows

- **sync-monitor.yml** - Redundant (sync-to-public.yml already handles synchronization)

## Active Workflows (9 workflows)

Critical workflows still running:
- ✅ ci-basic.yml
- ✅ docker.yml
- ✅ release.yml
- ✅ sync-to-public.yml
- ✅ pr-validation-basic.yml
- ✅ platform-tests.yml
- ✅ sync-wikis.yml
- ✅ dependency-updates.yml
- ✅ api-integration-tests.yml
