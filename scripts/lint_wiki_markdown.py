#!/usr/bin/env python3
"""
Wiki Markdown Linter
Enforces markdown style consistency across all wiki files.

Usage:
    python scripts/lint_wiki_markdown.py [--fix] [--output errors.txt]

Checks:
    - Heading hierarchy (no skipped levels)
    - Code block language tags
    - Link formatting (no bare URLs)
    - Table formatting
    - List consistency
    - Trailing whitespace
    - Version tag formatting
    - File path validation

Options:
    --fix       Automatically fix simple issues
    --output    Save errors to file
"""
import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import time


class LintLevel(Enum):
    """Severity level for lint issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class LintIssue:
    """Represents a linting issue."""
    file: str
    line: int
    level: LintLevel
    rule: str
    message: str
    suggestion: str = ""


@dataclass
class LintStats:
    """Linting statistics."""
    files_checked: int = 0
    total_issues: int = 0
    errors: int = 0
    warnings: int = 0
    info: int = 0
    fixed: int = 0


class WikiMarkdownLinter:
    """Lint markdown files for style consistency."""

    def __init__(self, wiki_dir: Path, fix: bool = False, verbose: bool = False):
        self.wiki_dir = wiki_dir
        self.fix = fix
        self.verbose = verbose
        self.issues: List[LintIssue] = []
        self.stats = LintStats()
        self.project_root = wiki_dir.parent  # lobster directory

    def find_markdown_files(self) -> List[Path]:
        """Find all .md files in wiki directory."""
        files = sorted(self.wiki_dir.glob("*.md"))
        # Exclude special/generated files
        exclude = {
            'PHASE2_CODE_VERIFICATION_REPORT.md',
            'PHASE2_VERIFICATION_SUMMARY.md',
            'PHASE2_FEATURES_DOCUMENTATION_REPORT.md',
            'PHASE2_UX_ENHANCEMENT_REPORT.md',
            'DOCUMENTATION_GAP_ANALYSIS_2025-11-16.md'
        }
        return [f for f in files if f.name not in exclude]

    def check_heading_hierarchy(self, lines: List[str], file_name: str) -> List[LintIssue]:
        """Check that heading levels don't skip (h1 -> h3 is invalid)."""
        issues = []
        prev_level = 0

        for i, line in enumerate(lines, 1):
            match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if match:
                current_level = len(match.group(1))
                heading_text = match.group(2)

                # Check for level skipping (e.g., h2 -> h4)
                if current_level > prev_level + 1 and prev_level > 0:
                    issues.append(LintIssue(
                        file=file_name,
                        line=i,
                        level=LintLevel.WARNING,
                        rule="heading-hierarchy",
                        message=f"Heading level skipped: h{prev_level} -> h{current_level}",
                        suggestion=f"Use h{prev_level + 1} instead of h{current_level}"
                    ))

                # Check for multiple h1 headings (often indicates bad structure)
                if current_level == 1 and prev_level == 1:
                    issues.append(LintIssue(
                        file=file_name,
                        line=i,
                        level=LintLevel.INFO,
                        rule="multiple-h1",
                        message="Multiple h1 headings found in file",
                        suggestion="Consider using h2 for subsections"
                    ))

                prev_level = current_level

        return issues

    def check_code_blocks(self, lines: List[str], file_name: str) -> List[LintIssue]:
        """Check code blocks have language tags."""
        issues = []
        in_code_block = False
        code_start_line = 0

        for i, line in enumerate(lines, 1):
            if line.startswith('```'):
                if not in_code_block:
                    # Starting code block
                    code_start_line = i
                    in_code_block = True

                    # Check if language tag is present
                    match = re.match(r'^```(\w+)?', line)
                    if match:
                        lang = match.group(1)
                        if not lang:
                            issues.append(LintIssue(
                                file=file_name,
                                line=i,
                                level=LintLevel.WARNING,
                                rule="code-language",
                                message="Code block missing language tag",
                                suggestion="Add language tag: ```python, ```bash, etc."
                            ))
                else:
                    # Ending code block
                    in_code_block = False

        return issues

    def check_bare_urls(self, lines: List[str], file_name: str) -> List[LintIssue]:
        """Check for bare URLs (should use [text](url) format)."""
        issues = []

        # Pattern for bare URLs not in link format
        bare_url_pattern = r'(?<!\()(https?://[^\s\)]+)(?!\))'

        for i, line in enumerate(lines, 1):
            # Skip code blocks
            if line.startswith('```') or line.startswith('    '):
                continue

            # Skip lines that already have proper link format
            if '[' in line and '](' in line:
                continue

            matches = re.finditer(bare_url_pattern, line)
            for match in matches:
                url = match.group(1)
                issues.append(LintIssue(
                    file=file_name,
                    line=i,
                    level=LintLevel.INFO,
                    rule="bare-url",
                    message=f"Bare URL found: {url[:50]}...",
                    suggestion=f"Use markdown link format: [text]({url})"
                ))

        return issues

    def check_trailing_whitespace(self, lines: List[str], file_name: str) -> List[LintIssue]:
        """Check for trailing whitespace."""
        issues = []

        for i, line in enumerate(lines, 1):
            if line.rstrip() != line:
                issues.append(LintIssue(
                    file=file_name,
                    line=i,
                    level=LintLevel.INFO,
                    rule="trailing-whitespace",
                    message="Trailing whitespace found",
                    suggestion="Remove trailing spaces/tabs"
                ))

        return issues

    def check_list_consistency(self, lines: List[str], file_name: str) -> List[LintIssue]:
        """Check list markers are consistent within sections."""
        issues = []
        in_list = False
        list_marker = None
        list_start_line = 0

        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()

            # Check if this is a list item
            dash_list = stripped.startswith('- ')
            asterisk_list = stripped.startswith('* ')
            plus_list = stripped.startswith('+ ')

            if dash_list or asterisk_list or plus_list:
                current_marker = stripped[0]

                if not in_list:
                    # Starting new list
                    in_list = True
                    list_marker = current_marker
                    list_start_line = i
                elif current_marker != list_marker:
                    # Inconsistent marker in same list
                    issues.append(LintIssue(
                        file=file_name,
                        line=i,
                        level=LintLevel.INFO,
                        rule="list-consistency",
                        message=f"Inconsistent list marker: using '{current_marker}' but list started with '{list_marker}'",
                        suggestion=f"Use '{list_marker}' for consistency"
                    ))

            elif in_list and stripped and not stripped.startswith((' ', '\t')):
                # End of list (non-empty, non-indented line)
                in_list = False
                list_marker = None

        return issues

    def check_version_tags(self, lines: List[str], file_name: str) -> List[LintIssue]:
        """Check version tag formatting consistency."""
        issues = []

        # Valid formats: v2.3+, v2.3, (v2.3+), **v2.3+**
        # Invalid: V2.3, version 2.3, ver 2.3
        invalid_patterns = [
            (r'\bV(\d+\.\d+)', 'Use lowercase: v{version}'),
            (r'\bversion\s+(\d+\.\d+)', 'Use format: v{version}'),
            (r'\bver\s+(\d+\.\d+)', 'Use format: v{version}')
        ]

        for i, line in enumerate(lines, 1):
            # Skip code blocks
            if line.startswith('```') or line.startswith('    '):
                continue

            for pattern, suggestion in invalid_patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    version = match.group(1)
                    issues.append(LintIssue(
                        file=file_name,
                        line=i,
                        level=LintLevel.INFO,
                        rule="version-format",
                        message=f"Inconsistent version format: {match.group(0)}",
                        suggestion=suggestion.format(version=version)
                    ))

        return issues

    def check_file_paths(self, lines: List[str], file_name: str) -> List[LintIssue]:
        """Check file paths in code examples exist."""
        issues = []

        # Pattern for file paths in code examples
        path_patterns = [
            r'lobster/[a-z_/]+\.py',
            r'scripts/[a-z_]+\.py',
            r'tests/[a-z_/]+\.py'
        ]

        for i, line in enumerate(lines, 1):
            # Only check in code blocks or code-related contexts
            if '```' in line or 'import' in line or 'from' in line:
                continue

            for pattern in path_patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    file_path = match.group(0)
                    full_path = self.project_root / file_path

                    if not full_path.exists():
                        issues.append(LintIssue(
                            file=file_name,
                            line=i,
                            level=LintLevel.WARNING,
                            rule="file-path",
                            message=f"Referenced file does not exist: {file_path}",
                            suggestion="Verify the file path is correct"
                        ))

        return issues

    def check_table_formatting(self, lines: List[str], file_name: str) -> List[LintIssue]:
        """Check table formatting consistency."""
        issues = []
        in_table = False
        table_start_line = 0

        for i, line in enumerate(lines, 1):
            # Check if this is a table row
            if '|' in line and line.strip().startswith('|'):
                if not in_table:
                    in_table = True
                    table_start_line = i

                # Check if this is a separator line
                if re.match(r'^\|\s*[-:]+\s*(\|\s*[-:]+\s*)*\|?\s*$', line):
                    # Verify previous line was header
                    if i == table_start_line:
                        issues.append(LintIssue(
                            file=file_name,
                            line=i,
                            level=LintLevel.ERROR,
                            rule="table-header",
                            message="Table separator without header",
                            suggestion="Add table header row above separator"
                        ))

            elif in_table and line.strip() and not line.strip().startswith('|'):
                # End of table
                in_table = False

        return issues

    def lint_file(self, file_path: Path) -> None:
        """Lint a single markdown file."""
        if self.verbose:
            print(f"Linting {file_path.name}...", end=' ')

        content = file_path.read_text(encoding='utf-8')
        lines = content.split('\n')

        file_issues = []

        # Run all checks
        file_issues.extend(self.check_heading_hierarchy(lines, file_path.name))
        file_issues.extend(self.check_code_blocks(lines, file_path.name))
        file_issues.extend(self.check_bare_urls(lines, file_path.name))
        file_issues.extend(self.check_trailing_whitespace(lines, file_path.name))
        file_issues.extend(self.check_list_consistency(lines, file_path.name))
        file_issues.extend(self.check_version_tags(lines, file_path.name))
        file_issues.extend(self.check_file_paths(lines, file_path.name))
        file_issues.extend(self.check_table_formatting(lines, file_path.name))

        self.issues.extend(file_issues)

        # Update statistics
        for issue in file_issues:
            self.stats.total_issues += 1
            if issue.level == LintLevel.ERROR:
                self.stats.errors += 1
            elif issue.level == LintLevel.WARNING:
                self.stats.warnings += 1
            else:
                self.stats.info += 1

        if self.verbose:
            if file_issues:
                error_count = sum(1 for i in file_issues if i.level == LintLevel.ERROR)
                warning_count = sum(1 for i in file_issues if i.level == LintLevel.WARNING)
                info_count = sum(1 for i in file_issues if i.level == LintLevel.INFO)
                print(f"❌ {len(file_issues)} issues (E:{error_count} W:{warning_count} I:{info_count})")
            else:
                print("✅")

        self.stats.files_checked += 1

    def lint_all(self) -> int:
        """Lint all markdown files and return error count."""
        files = self.find_markdown_files()
        print(f"Found {len(files)} markdown files in {self.wiki_dir}")
        print("Linting markdown files...\n")

        for file_path in files:
            self.lint_file(file_path)

        return self.stats.errors

    def generate_report(self) -> str:
        """Generate markdown report of lint issues."""
        report = [
            "# Wiki Markdown Lint Report",
            "",
            f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- **Files Checked**: {self.stats.files_checked}",
            f"- **Total Issues**: {self.stats.total_issues}",
            f"- **Errors**: {self.stats.errors} 🔴",
            f"- **Warnings**: {self.stats.warnings} 🟡",
            f"- **Info**: {self.stats.info} 🔵",
            ""
        ]

        if self.stats.fixed > 0:
            report.append(f"- **Auto-Fixed**: {self.stats.fixed}")
            report.append("")

        if not self.issues:
            report.append("✅ **All markdown files pass linting checks!**")
            return "\n".join(report)

        # Group issues by level and file
        by_level: Dict[LintLevel, List[LintIssue]] = {
            LintLevel.ERROR: [],
            LintLevel.WARNING: [],
            LintLevel.INFO: []
        }
        for issue in self.issues:
            by_level[issue.level].append(issue)

        # Report errors first
        if by_level[LintLevel.ERROR]:
            report.extend([
                "## Errors 🔴",
                "",
                f"Found {len(by_level[LintLevel.ERROR])} critical issues:",
                ""
            ])
            report.extend(self._format_issues(by_level[LintLevel.ERROR]))

        # Then warnings
        if by_level[LintLevel.WARNING]:
            report.extend([
                "## Warnings 🟡",
                "",
                f"Found {len(by_level[LintLevel.WARNING])} potential issues:",
                ""
            ])
            report.extend(self._format_issues(by_level[LintLevel.WARNING]))

        # Finally info
        if by_level[LintLevel.INFO]:
            report.extend([
                "## Info 🔵",
                "",
                f"Found {len(by_level[LintLevel.INFO])} style suggestions:",
                ""
            ])
            report.extend(self._format_issues(by_level[LintLevel.INFO]))

        # Add recommendations
        report.extend([
            "",
            "## Recommendations",
            "",
            "### Priority Fixes",
            "1. Fix all **errors** (critical issues) immediately",
            "2. Review and fix **warnings** (potential problems)",
            "3. Consider **info** items for consistency improvements",
            "",
            "### Common Fixes",
            "- **heading-hierarchy**: Ensure headings progress logically (h2 after h1, h3 after h2)",
            "- **code-language**: Add language tags to code blocks (```python, ```bash)",
            "- **bare-url**: Convert URLs to markdown link format [text](url)",
            "- **trailing-whitespace**: Remove spaces/tabs at end of lines",
            "- **list-consistency**: Use same marker (-, *, +) throughout a list",
            "- **version-format**: Use lowercase v prefix (v2.3+, not V2.3 or version 2.3)",
            ""
        ])

        return "\n".join(report)

    def _format_issues(self, issues: List[LintIssue]) -> List[str]:
        """Format list of issues for report."""
        lines = []

        # Group by file
        by_file: Dict[str, List[LintIssue]] = {}
        for issue in issues:
            by_file.setdefault(issue.file, []).append(issue)

        for file in sorted(by_file.keys()):
            file_issues = by_file[file]
            lines.append(f"### {file} ({len(file_issues)} issues)")
            lines.append("")

            for issue in file_issues:
                lines.append(f"**Line {issue.line}** [{issue.rule}]:")
                lines.append(f"- {issue.message}")
                if issue.suggestion:
                    lines.append(f"- 💡 *Suggestion: {issue.suggestion}*")
                lines.append("")

        return lines


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Lint markdown files in wiki for style consistency'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Automatically fix simple issues'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Save errors to file'
    )
    parser.add_argument(
        '--wiki-dir',
        type=str,
        help='Wiki directory path (default: ../wiki)'
    )

    args = parser.parse_args()

    # Determine wiki directory
    if args.wiki_dir:
        wiki_dir = Path(args.wiki_dir)
    else:
        wiki_dir = Path(__file__).parent.parent / "wiki"

    if not wiki_dir.exists():
        print(f"❌ Error: Wiki directory not found at {wiki_dir}")
        sys.exit(1)

    print("=" * 80)
    print("Wiki Markdown Linter")
    print("=" * 80)
    print()

    linter = WikiMarkdownLinter(wiki_dir, fix=args.fix, verbose=args.verbose)
    error_count = linter.lint_all()

    print("\n" + "=" * 80)
    report = linter.generate_report()
    print(report)
    print("=" * 80)

    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report)
        print(f"\n📄 Report saved to: {output_path}")

    # Summary
    if error_count > 0:
        print(f"\n❌ Found {error_count} errors, {linter.stats.warnings} warnings, {linter.stats.info} info")
        sys.exit(1)
    elif linter.stats.warnings > 0:
        print(f"\n⚠️  No errors, but found {linter.stats.warnings} warnings, {linter.stats.info} info")
        sys.exit(0)
    else:
        print(f"\n✅ All checks passed! ({linter.stats.info} style suggestions)")
        sys.exit(0)


if __name__ == "__main__":
    main()
