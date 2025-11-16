#!/usr/bin/env python3
"""
Wiki Health Dashboard Generator
Creates a comprehensive health dashboard for wiki documentation.

Usage:
    python scripts/generate_wiki_dashboard.py [--output wiki/WIKI_HEALTH_DASHBOARD.md]

Metrics:
    - Code accuracy (from test_wiki_code_examples.py)
    - Link health (from check_wiki_links.py)
    - Markdown style (from lint_wiki_markdown.py)
    - Version coverage
    - Freshness (last update dates)
    - Cross-references (orphaned pages)
    - Completeness (TODOs, placeholders)

Options:
    --output    Dashboard output file (default: wiki/WIKI_HEALTH_DASHBOARD.md)
"""
import re
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time


@dataclass
class FileMetrics:
    """Metrics for a single documentation file."""
    name: str
    size_bytes: int
    line_count: int
    word_count: int
    code_blocks: int
    last_modified: datetime
    version_tags: int
    internal_links: int
    external_links: int
    todos: int
    has_examples: bool


@dataclass
class DashboardMetrics:
    """Overall wiki health metrics."""
    # Code quality
    total_code_blocks: int = 0
    executable_code_blocks: int = 0
    code_tests_passed: int = 0
    code_tests_failed: int = 0
    code_accuracy: float = 0.0

    # Link health
    total_links: int = 0
    broken_links: int = 0
    link_health: float = 0.0

    # Markdown style
    lint_errors: int = 0
    lint_warnings: int = 0
    lint_info: int = 0

    # Content metrics
    total_files: int = 0
    total_words: int = 0
    total_lines: int = 0
    files_with_todos: int = 0
    stale_files: int = 0  # Not updated in 90+ days
    orphaned_pages: int = 0

    # Version coverage
    files_with_version_tags: int = 0
    version_coverage: float = 0.0

    # Overall health score
    health_score: int = 0
    health_status: str = "Unknown"


class WikiDashboardGenerator:
    """Generate comprehensive wiki health dashboard."""

    def __init__(self, wiki_dir: Path, project_root: Path):
        self.wiki_dir = wiki_dir
        self.project_root = project_root
        self.file_metrics: Dict[str, FileMetrics] = {}
        self.metrics = DashboardMetrics()

    def collect_file_metrics(self) -> None:
        """Collect basic metrics for each file."""
        files = self._find_markdown_files()
        print(f"Collecting metrics from {len(files)} files...")

        for file_path in files:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')

            # Count various elements
            code_blocks = len(re.findall(r'^```', content, re.MULTILINE))
            version_tags = len(re.findall(r'\bv\d+\.\d+[+]?', content))
            internal_links = len(re.findall(r'\[([^\]]+)\]\(([^)]+\.md)\)', content))
            external_links = len(re.findall(r'\[([^\]]+)\]\((https?://[^)]+)\)', content))
            todos = len(re.findall(r'\bTODO\b|\bFIXME\b|__PLACEHOLDER__', content, re.IGNORECASE))
            has_examples = 'example' in content.lower() or 'cookbook' in file_path.name.lower()

            # Count words (excluding code blocks)
            text_content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
            words = len(text_content.split())

            # Get last modified time
            stat = file_path.stat()
            last_modified = datetime.fromtimestamp(stat.st_mtime)

            self.file_metrics[file_path.name] = FileMetrics(
                name=file_path.name,
                size_bytes=stat.st_size,
                line_count=len(lines),
                word_count=words,
                code_blocks=code_blocks // 2,  # Divide by 2 (opening/closing)
                last_modified=last_modified,
                version_tags=version_tags,
                internal_links=internal_links,
                external_links=external_links,
                todos=todos,
                has_examples=has_examples
            )

        self.metrics.total_files = len(self.file_metrics)
        self.metrics.total_words = sum(f.word_count for f in self.file_metrics.values())
        self.metrics.total_lines = sum(f.line_count for f in self.file_metrics.values())
        self.metrics.files_with_todos = sum(1 for f in self.file_metrics.values() if f.todos > 0)

        # Calculate stale files (90+ days old)
        cutoff_date = datetime.now() - timedelta(days=90)
        self.metrics.stale_files = sum(
            1 for f in self.file_metrics.values()
            if f.last_modified < cutoff_date
        )

        # Calculate version coverage
        self.metrics.files_with_version_tags = sum(
            1 for f in self.file_metrics.values() if f.version_tags > 0
        )
        self.metrics.version_coverage = (
            self.metrics.files_with_version_tags / self.metrics.total_files * 100
            if self.metrics.total_files > 0 else 0
        )

    def run_code_tests(self) -> None:
        """Run code testing and collect metrics."""
        print("Running code tests...")
        try:
            script_path = self.project_root / "scripts" / "test_wiki_code_examples.py"
            result = subprocess.run(
                ["python", str(script_path), "--output", "/tmp/code_test_dashboard.txt"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )

            # Parse output for metrics
            output = result.stdout + result.stderr

            # Extract metrics from output
            match = re.search(r'Total Code Blocks[:\s]+(\d+)', output)
            if match:
                self.metrics.total_code_blocks = int(match.group(1))

            match = re.search(r'Executable[:\s]+(\d+)', output)
            if match:
                self.metrics.executable_code_blocks = int(match.group(1))

            match = re.search(r'Passed[:\s]+(\d+)', output)
            if match:
                self.metrics.code_tests_passed = int(match.group(1))

            match = re.search(r'Failed[:\s]+(\d+)', output)
            if match:
                self.metrics.code_tests_failed = int(match.group(1))

            # Calculate accuracy
            if self.metrics.executable_code_blocks > 0:
                self.metrics.code_accuracy = (
                    self.metrics.code_tests_passed / self.metrics.executable_code_blocks * 100
                )

        except subprocess.TimeoutExpired:
            print("⚠️  Code tests timed out")
        except Exception as e:
            print(f"⚠️  Code tests failed: {e}")

    def run_link_checks(self) -> None:
        """Run link checker and collect metrics."""
        print("Checking links...")
        try:
            script_path = self.project_root / "scripts" / "check_wiki_links.py"
            result = subprocess.run(
                ["python", str(script_path), "--output", "/tmp/link_check_dashboard.txt"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )

            # Parse output for metrics
            output = result.stdout + result.stderr

            # Extract link metrics
            internal_match = re.search(r'Internal Links[:\s]+(\d+)', output)
            if internal_match:
                self.metrics.total_links = int(internal_match.group(1))

            broken_match = re.search(r'Broken Internal Links[:\s]+(\d+)', output)
            if broken_match:
                self.metrics.broken_links = int(broken_match.group(1))

            # Calculate link health
            if self.metrics.total_links > 0:
                self.metrics.link_health = (
                    (self.metrics.total_links - self.metrics.broken_links)
                    / self.metrics.total_links * 100
                )

        except subprocess.TimeoutExpired:
            print("⚠️  Link checks timed out")
        except Exception as e:
            print(f"⚠️  Link checks failed: {e}")

    def run_markdown_lint(self) -> None:
        """Run markdown linter and collect metrics."""
        print("Linting markdown...")
        try:
            script_path = self.project_root / "scripts" / "lint_wiki_markdown.py"
            result = subprocess.run(
                ["python", str(script_path), "--output", "/tmp/markdown_lint_dashboard.txt"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )

            # Parse output for metrics
            output = result.stdout + result.stderr

            # Extract lint metrics
            errors_match = re.search(r'Errors[:\s]+(\d+)', output)
            if errors_match:
                self.metrics.lint_errors = int(errors_match.group(1))

            warnings_match = re.search(r'Warnings[:\s]+(\d+)', output)
            if warnings_match:
                self.metrics.lint_warnings = int(warnings_match.group(1))

            info_match = re.search(r'Info[:\s]+(\d+)', output)
            if info_match:
                self.metrics.lint_info = int(info_match.group(1))

        except subprocess.TimeoutExpired:
            print("⚠️  Markdown lint timed out")
        except Exception as e:
            print(f"⚠️  Markdown lint failed: {e}")

    def calculate_health_score(self) -> None:
        """Calculate overall health score (0-100)."""
        score = 0

        # Code accuracy (30 points)
        if self.metrics.executable_code_blocks > 0:
            score += (self.metrics.code_accuracy / 100) * 30
        else:
            score += 30  # No code = perfect code

        # Link health (25 points)
        if self.metrics.total_links > 0:
            score += (self.metrics.link_health / 100) * 25
        else:
            score += 25

        # Markdown style (20 points)
        # Deduct points for errors and warnings
        style_deduction = (self.metrics.lint_errors * 2) + (self.metrics.lint_warnings * 0.5)
        style_score = max(0, 20 - (style_deduction / 10))
        score += style_score

        # Freshness (10 points)
        freshness_ratio = 1 - (self.metrics.stale_files / self.metrics.total_files)
        score += freshness_ratio * 10

        # Completeness (10 points)
        completeness_ratio = 1 - (self.metrics.files_with_todos / max(1, self.metrics.total_files))
        score += completeness_ratio * 10

        # Version coverage (5 points)
        score += (self.metrics.version_coverage / 100) * 5

        self.metrics.health_score = int(score)

        # Determine status
        if score >= 90:
            self.metrics.health_status = "🟢 Excellent"
        elif score >= 75:
            self.metrics.health_status = "🟡 Good"
        elif score >= 60:
            self.metrics.health_status = "🟠 Fair"
        else:
            self.metrics.health_status = "🔴 Needs Improvement"

    def generate_dashboard(self) -> str:
        """Generate the dashboard markdown."""
        lines = [
            "# Wiki Health Dashboard",
            "",
            f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
            f"**Overall Health**: {self.metrics.health_status} ({self.metrics.health_score}/100)",
            "",
            "## Quick Metrics",
            "",
            "| Metric | Score | Status |",
            "|--------|-------|--------|",
            self._metric_row(
                "Code Accuracy",
                f"{self.metrics.code_accuracy:.1f}%",
                f"{self.metrics.code_tests_passed}/{self.metrics.executable_code_blocks}",
                self.metrics.code_accuracy
            ),
            self._metric_row(
                "Link Health",
                f"{self.metrics.link_health:.1f}%",
                f"{self.metrics.broken_links} broken",
                self.metrics.link_health
            ),
            self._metric_row(
                "Markdown Style",
                f"{self.metrics.lint_errors} errors, {self.metrics.lint_warnings} warnings",
                f"{self.metrics.lint_info} info",
                100 - (self.metrics.lint_errors * 10)
            ),
            self._metric_row(
                "Version Coverage",
                f"{self.metrics.version_coverage:.1f}%",
                f"{self.metrics.files_with_version_tags}/{self.metrics.total_files}",
                self.metrics.version_coverage
            ),
            self._metric_row(
                "Freshness",
                f"{self.metrics.stale_files} stale",
                "<90 days old",
                100 - (self.metrics.stale_files / max(1, self.metrics.total_files) * 100)
            ),
            self._metric_row(
                "Completeness",
                f"{self.metrics.files_with_todos} TODOs",
                "No placeholders",
                100 - (self.metrics.files_with_todos / max(1, self.metrics.total_files) * 100)
            ),
            "",
            "## Content Statistics",
            "",
            f"- **Total Files**: {self.metrics.total_files}",
            f"- **Total Words**: {self.metrics.total_words:,}",
            f"- **Total Lines**: {self.metrics.total_lines:,}",
            f"- **Code Blocks**: {self.metrics.total_code_blocks}",
            f"- **Internal Links**: {sum(f.internal_links for f in self.file_metrics.values())}",
            f"- **External Links**: {sum(f.external_links for f in self.file_metrics.values())}",
            "",
            "## Detailed Metrics",
            "",
            "### Code Quality",
            "",
            f"- Total code blocks: {self.metrics.total_code_blocks}",
            f"- Executable (tested): {self.metrics.executable_code_blocks}",
            f"- Tests passed: ✅ {self.metrics.code_tests_passed}",
            f"- Tests failed: ❌ {self.metrics.code_tests_failed}",
            f"- Accuracy: **{self.metrics.code_accuracy:.1f}%**",
            "",
            "### Link Health",
            "",
            f"- Total internal links: {self.metrics.total_links}",
            f"- Broken links: {self.metrics.broken_links}",
            f"- Health: **{self.metrics.link_health:.1f}%**",
            "",
            "### Markdown Style",
            "",
            f"- Errors (critical): 🔴 {self.metrics.lint_errors}",
            f"- Warnings (potential issues): 🟡 {self.metrics.lint_warnings}",
            f"- Info (style suggestions): 🔵 {self.metrics.lint_info}",
            "",
            "## File-Level Health",
            "",
            "Top files by size:",
            ""
        ]

        # Top 10 files by word count
        sorted_files = sorted(
            self.file_metrics.values(),
            key=lambda f: f.word_count,
            reverse=True
        )[:10]

        lines.append("| File | Words | Code Blocks | Version Tags | Last Updated |")
        lines.append("|------|-------|-------------|--------------|--------------|")
        for f in sorted_files:
            days_ago = (datetime.now() - f.last_modified).days
            age_str = f"{days_ago}d ago" if days_ago > 0 else "today"
            lines.append(
                f"| {f.name} | {f.word_count:,} | {f.code_blocks} | "
                f"{f.version_tags} | {age_str} |"
            )

        lines.extend([
            "",
            "## Recommendations",
            ""
        ])

        # Generate recommendations based on metrics
        if self.metrics.code_tests_failed > 0:
            lines.append(
                f"1. 🔴 **Fix {self.metrics.code_tests_failed} failing code examples** "
                "(run `python scripts/test_wiki_code_examples.py`)"
            )

        if self.metrics.broken_links > 0:
            lines.append(
                f"2. 🔴 **Fix {self.metrics.broken_links} broken links** "
                "(run `python scripts/check_wiki_links.py`)"
            )

        if self.metrics.lint_errors > 0:
            lines.append(
                f"3. 🔴 **Fix {self.metrics.lint_errors} markdown errors** "
                "(run `python scripts/lint_wiki_markdown.py`)"
            )

        if self.metrics.lint_warnings > 10:
            lines.append(
                f"4. 🟡 **Address {self.metrics.lint_warnings} markdown warnings** "
                "(style improvements)"
            )

        if self.metrics.stale_files > 5:
            lines.append(
                f"5. 🟡 **Review {self.metrics.stale_files} stale files** "
                "(not updated in 90+ days)"
            )

        if self.metrics.files_with_todos > 0:
            lines.append(
                f"6. 🟡 **Complete {self.metrics.files_with_todos} files with TODOs** "
                "(search for TODO/FIXME)"
            )

        if not any(c in ''.join(lines[-10:]) for c in ['🔴', '🟡']):
            lines.append("✅ **All critical issues resolved!** Wiki documentation is in excellent health.")

        lines.extend([
            "",
            "## Automation Status",
            "",
            "- ✅ **Link Checker**: Automated (GitHub Actions, weekly)",
            "- ✅ **Code Tester**: Automated (GitHub Actions, on wiki changes)",
            "- ✅ **Markdown Linter**: Automated (GitHub Actions, on wiki changes)",
            "- ✅ **Dashboard Update**: Automated (GitHub Actions, daily)",
            "",
            "## Maintenance Commands",
            "",
            "```bash",
            "# Run all checks locally",
            "python scripts/check_wiki_links.py",
            "python scripts/test_wiki_code_examples.py",
            "python scripts/lint_wiki_markdown.py",
            "python scripts/generate_wiki_dashboard.py",
            "```",
            "",
            "---",
            f"*Dashboard generated by automation system on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S UTC')}*"
        ])

        return "\n".join(lines)

    def _metric_row(self, name: str, score: str, detail: str, percentage: float) -> str:
        """Generate a metric table row with status emoji."""
        if percentage >= 95:
            status = "🟢 Excellent"
        elif percentage >= 85:
            status = "🟡 Good"
        elif percentage >= 70:
            status = "🟠 Fair"
        else:
            status = "🔴 Needs Work"

        return f"| {name} | {score} ({detail}) | {status} |"

    def _find_markdown_files(self) -> List[Path]:
        """Find all markdown files excluding generated reports."""
        files = sorted(self.wiki_dir.glob("*.md"))
        exclude = {
            'PHASE2_CODE_VERIFICATION_REPORT.md',
            'PHASE2_VERIFICATION_SUMMARY.md',
            'PHASE2_FEATURES_DOCUMENTATION_REPORT.md',
            'PHASE2_UX_ENHANCEMENT_REPORT.md',
            'DOCUMENTATION_GAP_ANALYSIS_2025-11-16.md',
            'WIKI_HEALTH_DASHBOARD.md'  # Exclude self
        }
        return [f for f in files if f.name not in exclude]


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate wiki health dashboard'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file (default: wiki/WIKI_HEALTH_DASHBOARD.md)'
    )
    parser.add_argument(
        '--wiki-dir',
        type=str,
        help='Wiki directory path (default: ../wiki)'
    )

    args = parser.parse_args()

    # Determine paths
    project_root = Path(__file__).parent.parent
    wiki_dir = Path(args.wiki_dir) if args.wiki_dir else project_root / "wiki"
    output_path = Path(args.output) if args.output else wiki_dir / "WIKI_HEALTH_DASHBOARD.md"

    if not wiki_dir.exists():
        print(f"❌ Error: Wiki directory not found at {wiki_dir}")
        sys.exit(1)

    print("=" * 80)
    print("Wiki Health Dashboard Generator")
    print("=" * 80)
    print()

    # Generate dashboard
    generator = WikiDashboardGenerator(wiki_dir, project_root)

    generator.collect_file_metrics()
    generator.run_code_tests()
    generator.run_link_checks()
    generator.run_markdown_lint()
    generator.calculate_health_score()

    dashboard = generator.generate_dashboard()

    # Save dashboard
    output_path.write_text(dashboard)
    print(f"\n✅ Dashboard generated: {output_path}")
    print(f"   Health Score: {generator.metrics.health_score}/100 ({generator.metrics.health_status})")

    sys.exit(0)


if __name__ == "__main__":
    main()
