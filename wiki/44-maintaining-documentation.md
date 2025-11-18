# Maintaining Documentation - Wiki Maintenance Guide

This guide explains how to maintain the Lobster AI wiki documentation using the automated quality systems.

## Overview

The Lobster AI wiki uses a comprehensive automation infrastructure to maintain documentation quality:

- **Code Testing**: Validates all Python code examples in documentation
- **Link Checking**: Ensures all internal and external links are valid
- **Markdown Linting**: Enforces consistent markdown style
- **Health Dashboard**: Provides real-time documentation health metrics

## Automation Systems

### 1. Code Testing Framework

**Purpose**: Automatically test all Python code examples for syntax and import validity.

**Script**: `scripts/test_wiki_code_examples.py`

**What It Checks**:
- Syntax errors in code blocks
- Import availability
- Code completeness

**Code Categories**:
- **Executable**: Complete, runnable code (tested)
- **Template**: Code with placeholders like `your_service` (skipped)
- **Snippet**: Partial code requiring context (skipped)
- **Configuration**: YAML/JSON/TOML examples (skipped)
- **Bash**: Shell commands (skipped)

**Usage**:
```bash
# Run code tests
python scripts/test_wiki_code_examples.py

# With verbose output
python scripts/test_wiki_code_examples.py --verbose

# Save report
python scripts/test_wiki_code_examples.py --output report.md
```

**Marking Template Code**:
When writing code examples with placeholders, mark them as templates:

```python
# Template example
class YourService:
    def your_method(self):
        pass
```

Use clear placeholder names:
- `your_service`, `your_agent`, `your_modality`
- `YourService`, `YourAgent`, `YourClass`
- `your_parameter`, `your_config`

### 2. Link Checking System

**Purpose**: Validate all internal and external links in documentation.

**Script**: `scripts/check_wiki_links.py`

**What It Checks**:
- Internal links to other wiki pages
- Anchor links to headings
- External URLs (optional)

**Usage**:
```bash
# Check internal links only
python scripts/check_wiki_links.py

# Include external links (slower)
python scripts/check_wiki_links.py --external

# Save report
python scripts/check_wiki_links.py --output report.md
```

**Common Link Issues**:
- **Broken internal links**: File renamed or moved
- **Broken anchors**: Heading text changed
- **Broken external links**: Website moved or removed

**How to Fix**:
1. Update links to correct filenames
2. Update anchor links to match current headings
3. Replace or remove dead external links

### 3. Markdown Linting

**Purpose**: Enforce consistent markdown style across all documentation.

**Script**: `scripts/lint_wiki_markdown.py`

**What It Checks**:
- Heading hierarchy (no skipped levels: h1 → h2 → h3)
- Code block language tags (```python, ```bash)
- Bare URLs (should use [text](url) format)
- Trailing whitespace
- List consistency (same marker throughout list)
- Version tag formatting (v0.2+, not V2.3)
- Table formatting
- File path validity

**Usage**:
```bash
# Run linter
python scripts/lint_wiki_markdown.py

# With verbose output
python scripts/lint_wiki_markdown.py --verbose

# Save report
python scripts/lint_wiki_markdown.py --output report.md
```

**Issue Levels**:
- **Errors** 🔴: Critical issues (must fix)
- **Warnings** 🟡: Potential problems (should fix)
- **Info** 🔵: Style suggestions (optional)

**Common Issues and Fixes**:

| Issue | Problem | Fix |
|-------|---------|-----|
| heading-hierarchy | Skipped heading level | Use h2 after h1, h3 after h2 |
| code-language | Missing language tag | Add ```python or ```bash |
| bare-url | URL not formatted | Use [text](url) format |
| trailing-whitespace | Spaces at line end | Remove trailing spaces |
| list-consistency | Mixed list markers | Use same marker (-, *, +) |
| version-format | Inconsistent version | Use v0.2+ format |

### 4. Wiki Health Dashboard

**Purpose**: Visual dashboard showing comprehensive documentation health metrics.

**Script**: `scripts/generate_wiki_dashboard.py`

**What It Tracks**:
- Code accuracy percentage
- Link health status
- Markdown style score
- Version coverage
- Freshness (files not updated in 90+ days)
- Completeness (TODOs, placeholders)
- Overall health score (0-100)

**Usage**:
```bash
# Generate dashboard
python scripts/generate_wiki_dashboard.py

# Specify output location
python scripts/generate_wiki_dashboard.py --output wiki/WIKI_HEALTH_DASHBOARD.md
```

**Dashboard Location**: `wiki/WIKI_HEALTH_DASHBOARD.md`

**Health Scores**:
- **90-100**: 🟢 Excellent
- **75-89**: 🟡 Good
- **60-74**: 🟠 Fair
- **0-59**: 🔴 Needs Improvement

## GitHub Actions Integration

All automation systems run automatically via GitHub Actions:

### Code Testing Workflow

**File**: `.github/workflows/wiki-code-test.yml`

**Triggers**:
- Push to wiki files
- Pull requests affecting wiki
- Weekly on Tuesdays at 10am UTC
- Manual trigger

**What It Does**:
1. Tests all code examples in wiki
2. Uploads test report as artifact
3. Comments on PRs if tests fail
4. Fails CI if critical issues found

### Markdown Linting Workflow

**File**: `.github/workflows/wiki-markdown-lint.yml`

**Triggers**:
- Push to wiki files
- Pull requests affecting wiki
- Weekly on Wednesdays at 10am UTC
- Manual trigger

**What It Does**:
1. Lints all markdown files
2. Uploads lint report as artifact
3. Comments on PRs if errors found
4. Fails CI if critical errors exist

### Dashboard Update Workflow

**File**: `.github/workflows/wiki-dashboard.yml`

**Triggers**:
- Push to wiki or scripts
- Daily at midnight UTC
- Manual trigger

**What It Does**:
1. Generates fresh health dashboard
2. Commits and pushes dashboard updates
3. Uploads dashboard as artifact
4. Posts summary to workflow output

**Note**: Only runs on main/dev branches, not PRs.

## Local Development Setup

### Install Pre-commit Hooks

For local validation before committing:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

**Configured Hooks**:
- Wiki link checking
- Wiki code testing
- Wiki markdown linting
- Python formatting (black, isort)
- Trailing whitespace removal
- YAML/TOML validation

### Running Checks Locally

Before pushing documentation changes:

```bash
# 1. Test code examples
python scripts/test_wiki_code_examples.py --verbose

# 2. Check links
python scripts/check_wiki_links.py

# 3. Lint markdown
python scripts/lint_wiki_markdown.py --verbose

# 4. Generate dashboard
python scripts/generate_wiki_dashboard.py
```

## Writing High-Quality Documentation

### Code Examples Best Practices

1. **Keep Examples Complete**:
```python
# Good: Complete, runnable example
from lobster.core.data_manager_v2 import DataManagerV2

data_manager = DataManagerV2()
modality = data_manager.get_modality("my_dataset")
```

```python
# Bad: Incomplete snippet
modality = data_manager.get_modality(...)  # What is data_manager?
```

2. **Mark Template Code**:
```python
# Template example - replace with your values
class YourService:
    def your_method(self, your_parameter: str):
        # Your implementation here
        pass
```

3. **Include Necessary Imports**:
```python
# Good: All imports included
from typing import Dict, Any
import numpy as np
import anndata

def process_data(adata: anndata.AnnData) -> Dict[str, Any]:
    return {"cells": adata.n_obs}
```

4. **Add Type Hints**:
```python
# Good: Clear types
def analyze_modality(
    modality_name: str,
    threshold: float = 0.5
) -> Dict[str, Any]:
    pass
```

### Link Formatting

**Internal Links** (to other wiki pages):
```markdown
See [Creating Agents](09-creating-agents.md) for details.
```

**Anchor Links** (to headings):
```markdown
Jump to [Installation Steps](#installation-steps) below.
```

**External Links**:
```markdown
Visit [Scanpy Documentation](https://scanpy.readthedocs.io/) for more.
```

**Avoid Bare URLs**:
```markdown
<!-- Bad -->
https://github.com/the-omics-os/lobster

<!-- Good -->
[Lobster GitHub Repository](https://github.com/the-omics-os/lobster)
```

### Heading Hierarchy

Always follow proper heading progression:

```markdown
# Page Title (h1)

## Main Section (h2)

### Subsection (h3)

#### Detail (h4)

## Another Main Section (h2)
```

**Avoid**:
```markdown
# Page Title (h1)

### Skipped to h3 ❌ (should be h2)
```

### Version Tags

Use consistent version formatting:

```markdown
<!-- Good -->
Available in v0.2+
New in v0.2
Introduced in v0.2

<!-- Bad -->
Available in V2.3 (uppercase V)
New in version 2.4 (word "version")
Introduced in ver 2.2 (abbreviated "ver")
```

### Code Block Language Tags

Always specify the language:

```markdown
<!-- Good -->
```python
def example():
    pass
```

```bash
lobster chat
```

<!-- Bad - missing language tag -->
```
def example():
    pass
```
```

## Interpreting Dashboard Metrics

### Code Accuracy

**Metric**: Percentage of executable code blocks that pass syntax/import tests

**Target**: 95%+ (excellent), 85-94% (good), <85% (needs work)

**How to Improve**:
- Fix syntax errors in code examples
- Ensure imports are available
- Mark incomplete code as templates

### Link Health

**Metric**: Percentage of links that are valid

**Target**: 100% (perfect), 95-99% (excellent), <95% (needs work)

**How to Improve**:
- Fix broken internal links
- Update renamed files
- Remove or replace dead external links

### Markdown Style Score

**Metric**: Errors, warnings, and info suggestions from linter

**Target**: 0 errors (critical), <10 warnings (good), info is optional

**How to Improve**:
- Fix heading hierarchy issues
- Add language tags to code blocks
- Convert bare URLs to links
- Ensure consistent list formatting

### Version Coverage

**Metric**: Percentage of files with version tags

**Target**: 80%+ (excellent), 60-79% (good), <60% (needs work)

**How to Improve**:
- Add version tags to new features (v0.2+)
- Tag breaking changes
- Document migration requirements

### Freshness

**Metric**: Number of files not updated in 90+ days

**Target**: <5 stale files (excellent), 5-10 (good), >10 (review needed)

**How to Improve**:
- Review old documentation for accuracy
- Update deprecated information
- Add new examples and use cases

### Completeness

**Metric**: Number of files with TODOs/placeholders

**Target**: 0 (complete), 1-3 (good), >3 (needs work)

**How to Improve**:
- Complete TODO items
- Remove FIXME notes
- Fill in placeholder sections

## Troubleshooting

### Code Tests Failing

**Problem**: Code examples have syntax errors

**Solution**:
1. Run `python scripts/test_wiki_code_examples.py --verbose`
2. Check the error messages
3. Fix syntax errors in the indicated files/lines
4. Re-test locally before pushing

### Link Checker Failing

**Problem**: Broken internal links

**Solution**:
1. Run `python scripts/check_wiki_links.py`
2. Review the broken links report
3. Update links to correct filenames
4. Check for renamed or moved files

### Markdown Linter Errors

**Problem**: Style consistency issues

**Solution**:
1. Run `python scripts/lint_wiki_markdown.py --verbose`
2. Review the error/warning messages
3. Fix issues following the suggestions
4. Focus on errors first, then warnings

### Dashboard Not Updating

**Problem**: Dashboard seems out of date

**Solution**:
1. Check GitHub Actions status
2. Manually trigger dashboard workflow
3. Run `python scripts/generate_wiki_dashboard.py` locally
4. Verify scripts are up to date

## Contributing Guidelines

When updating documentation:

1. **Before Writing**:
   - Check existing documentation for similar topics
   - Review the style guide in this document
   - Plan your heading structure

2. **While Writing**:
   - Test code examples locally
   - Use proper markdown formatting
   - Add version tags for new features
   - Include cross-references to related topics

3. **Before Committing**:
   - Run local checks (code tests, link checker, linter)
   - Preview changes in markdown viewer
   - Check for TODOs or placeholders
   - Ensure examples are complete and tested

4. **After Pushing**:
   - Monitor GitHub Actions for failures
   - Review PR comments from automation
   - Check dashboard for impact on health score
   - Address any failures before merging

## Maintenance Schedule

### Daily
- Dashboard automatically updates (midnight UTC)

### Weekly
- Code tests run (Tuesday 10am UTC)
- Markdown linting runs (Wednesday 10am UTC)
- Link checking runs (Monday 9am UTC)

### On Every Wiki Change
- All checks run automatically in CI/CD
- Dashboard regenerates if on main/dev branch

### Manual Reviews (Recommended)
- Monthly: Review stale files
- Quarterly: Update examples with latest features
- Annually: Comprehensive documentation audit

## Additional Resources

- [Wiki Home](Home.md) - Main documentation index
- [Contributing Guide](08-developer-overview.md) - Developer guidelines
- [Architecture Overview](18-architecture-overview.md) - System design
- [Wiki Health Dashboard](WIKI_HEALTH_DASHBOARD.md) - Current metrics

## Contact

For questions about documentation maintenance:
- Open an issue on GitHub
- Check the [FAQ](29-faq.md)
- Review [Troubleshooting Guide](28-troubleshooting.md)

---

*This guide is maintained by the Lobster AI documentation team. Last updated: 2025-11-16*
