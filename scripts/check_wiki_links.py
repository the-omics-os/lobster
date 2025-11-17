#!/usr/bin/env python3
"""
Wiki Link Checker
Validates all internal and external links in wiki markdown files.

Usage:
    python scripts/check_wiki_links.py [--external] [--output report.md]

Options:
    --external    Check external links (slower, may have false positives)
    --output      Save report to file
"""
import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from urllib.parse import urlparse, unquote
import time

# Optional: requests for external link checking
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class WikiLinkChecker:
    """Check links in wiki markdown files."""

    def __init__(self, wiki_dir: Path, check_external: bool = False):
        self.wiki_dir = wiki_dir
        self.check_external = check_external
        self.issues: List[Dict] = []
        self.stats = {
            'files_checked': 0,
            'internal_links': 0,
            'external_links': 0,
            'anchor_links': 0,
            'broken_internal': 0,
            'broken_external': 0,
            'broken_anchors': 0
        }

    def find_markdown_files(self) -> List[Path]:
        """Find all .md files in wiki directory."""
        files = sorted(self.wiki_dir.glob("*.md"))
        return [f for f in files if f.name != '.git']

    def extract_links(self, content: str, file_path: Path) -> Dict[str, List[Tuple[int, str]]]:
        """Extract all links from markdown content.

        Returns:
            Dict with 'internal' and 'external' links, each as list of (line_num, url) tuples
        """
        links = {'internal': [], 'external': [], 'anchors': []}

        # Markdown link pattern: [text](url)
        md_link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'

        # Wiki-style link pattern: [[page-name]]
        wiki_link_pattern = r'\[\[([^\]]+)\]\]'

        for line_num, line in enumerate(content.split('\n'), 1):
            # Find markdown links
            for match in re.finditer(md_link_pattern, line):
                url = match.group(2)

                # Skip mailto links
                if url.startswith('mailto:'):
                    continue

                # Skip image embeds (already covered by link check)
                if url.startswith('http://') or url.startswith('https://'):
                    links['external'].append((line_num, url))
                    self.stats['external_links'] += 1
                elif url.startswith('#'):
                    links['anchors'].append((line_num, url))
                    self.stats['anchor_links'] += 1
                else:
                    # Handle relative links and anchors
                    if '#' in url:
                        file_part, anchor = url.split('#', 1)
                        if file_part:
                            links['internal'].append((line_num, file_part))
                            self.stats['internal_links'] += 1
                        links['anchors'].append((line_num, f"#{anchor}"))
                        self.stats['anchor_links'] += 1
                    else:
                        links['internal'].append((line_num, url))
                        self.stats['internal_links'] += 1

            # Find wiki-style links
            for match in re.finditer(wiki_link_pattern, line):
                page_name = match.group(1)
                # Convert to .md filename (GitHub wiki style)
                filename = page_name.replace(' ', '-') + '.md'
                links['internal'].append((line_num, filename))
                self.stats['internal_links'] += 1

        return links

    def check_internal_link(self, link: str, source_file: Path) -> bool:
        """Check if internal link target exists."""
        # Decode URL-encoded characters
        link = unquote(link)

        # Handle relative paths
        if link.startswith('./'):
            link = link[2:]
        elif link.startswith('../'):
            # Wiki files should be in same directory
            return False

        target = self.wiki_dir / link
        return target.exists()

    def check_external_link(self, url: str, timeout: int = 10) -> bool:
        """Check if external link is accessible."""
        if not REQUESTS_AVAILABLE:
            return True  # Skip if requests not available

        try:
            # Use HEAD request for efficiency
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; WikiLinkChecker/1.0)'
            }
            response = requests.head(url, timeout=timeout, allow_redirects=True, headers=headers)

            # Some servers don't like HEAD requests, try GET
            if response.status_code == 405:
                response = requests.get(url, timeout=timeout, allow_redirects=True, headers=headers)

            return response.status_code < 400
        except requests.RequestException as e:
            # Try one more time with GET for stubborn servers
            try:
                response = requests.get(url, timeout=timeout, allow_redirects=True, headers=headers)
                return response.status_code < 400
            except:
                return False

    def extract_headings(self, content: str) -> set:
        """Extract all heading anchors from markdown content."""
        headings = set()
        heading_pattern = r'^#{1,6}\s+(.+)$'

        for line in content.split('\n'):
            match = re.match(heading_pattern, line)
            if match:
                # Convert heading to GitHub-style anchor
                heading_text = match.group(1)
                # Remove markdown formatting
                heading_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', heading_text)
                heading_text = re.sub(r'`([^`]+)`', r'\1', heading_text)
                heading_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', heading_text)
                heading_text = re.sub(r'\*([^*]+)\*', r'\1', heading_text)

                # Convert to anchor format
                anchor = heading_text.lower()
                anchor = re.sub(r'[^\w\s-]', '', anchor)
                anchor = re.sub(r'\s+', '-', anchor)
                headings.add(f"#{anchor}")

        return headings

    def check_file(self, file_path: Path) -> None:
        """Check all links in a single markdown file."""
        print(f"Checking {file_path.name}...", end=' ')

        content = file_path.read_text(encoding='utf-8')
        links = self.extract_links(content, file_path)

        # Extract headings for anchor validation
        headings = self.extract_headings(content)

        issues_in_file = 0

        # Check internal links
        for line_num, link in links['internal']:
            if not self.check_internal_link(link, file_path):
                self.issues.append({
                    'file': file_path.name,
                    'line': line_num,
                    'type': 'internal',
                    'link': link,
                    'issue': 'File not found in wiki directory'
                })
                self.stats['broken_internal'] += 1
                issues_in_file += 1

        # Check anchor links (same file)
        for line_num, anchor in links['anchors']:
            if anchor not in headings:
                self.issues.append({
                    'file': file_path.name,
                    'line': line_num,
                    'type': 'anchor',
                    'link': anchor,
                    'issue': 'Heading not found in file'
                })
                self.stats['broken_anchors'] += 1
                issues_in_file += 1

        # Check external links (if enabled)
        if self.check_external and REQUESTS_AVAILABLE:
            for line_num, url in links['external']:
                if not self.check_external_link(url):
                    self.issues.append({
                        'file': file_path.name,
                        'line': line_num,
                        'type': 'external',
                        'link': url,
                        'issue': 'URL not accessible (404 or timeout)'
                    })
                    self.stats['broken_external'] += 1
                    issues_in_file += 1
                time.sleep(0.5)  # Rate limiting

        if issues_in_file > 0:
            print(f"❌ {issues_in_file} issues")
        else:
            print("✅")

        self.stats['files_checked'] += 1

    def check_all(self) -> int:
        """Check all markdown files and return issue count."""
        files = self.find_markdown_files()
        print(f"Found {len(files)} markdown files in {self.wiki_dir}\n")

        if self.check_external and not REQUESTS_AVAILABLE:
            print("⚠️  Warning: 'requests' module not available, skipping external link checks\n")
            self.check_external = False

        for file_path in files:
            self.check_file(file_path)

        return len(self.issues)

    def generate_report(self) -> str:
        """Generate markdown report of issues."""
        report = [
            "# Wiki Link Check Report",
            "",
            f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- **Files Checked**: {self.stats['files_checked']}",
            f"- **Internal Links**: {self.stats['internal_links']}",
            f"- **External Links**: {self.stats['external_links']}",
            f"- **Anchor Links**: {self.stats['anchor_links']}",
            "",
            f"- **Broken Internal Links**: {self.stats['broken_internal']}",
            f"- **Broken External Links**: {self.stats['broken_external']}",
            f"- **Broken Anchor Links**: {self.stats['broken_anchors']}",
            "",
            f"**Total Issues Found**: {len(self.issues)}",
            ""
        ]

        if not self.issues:
            report.append("✅ **No broken links found!**")
            return "\n".join(report)

        report.extend([
            "## Issues by File",
            ""
        ])

        # Group by file
        by_file: Dict[str, List[Dict]] = {}
        for issue in self.issues:
            by_file.setdefault(issue['file'], []).append(issue)

        for file, file_issues in sorted(by_file.items()):
            report.append(f"### {file} ({len(file_issues)} issues)")
            report.append("")
            for issue in file_issues:
                icon = "🔗" if issue['type'] == 'internal' else "🌐" if issue['type'] == 'external' else "⚓"
                report.append(f"{icon} **Line {issue['line']}**: [{issue['type']}] `{issue['link']}`")
                report.append(f"   - {issue['issue']}")
                report.append("")

        # Add actionable recommendations
        report.extend([
            "## Recommendations",
            ""
        ])

        if self.stats['broken_internal'] > 0:
            report.append("### Internal Links")
            report.append("- Check for typos in filenames")
            report.append("- Verify files exist in wiki directory")
            report.append("- Update links if files were renamed")
            report.append("")

        if self.stats['broken_anchors'] > 0:
            report.append("### Anchor Links")
            report.append("- Verify heading text matches exactly")
            report.append("- Check for special characters in headings")
            report.append("- Ensure headings haven't been renamed")
            report.append("")

        if self.stats['broken_external'] > 0:
            report.append("### External Links")
            report.append("- Test manually to confirm breakage")
            report.append("- Update to new URL if site moved")
            report.append("- Use Internet Archive for dead sites")
            report.append("- Remove if no longer relevant")
            report.append("")

        return "\n".join(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Check links in wiki markdown files')
    parser.add_argument('--external', action='store_true',
                       help='Check external links (slower, requires requests module)')
    parser.add_argument('--output', type=str,
                       help='Save report to file')
    parser.add_argument('--wiki-dir', type=str,
                       help='Wiki directory path (default: ../wiki)')

    args = parser.parse_args()

    # Determine wiki directory
    if args.wiki_dir:
        wiki_dir = Path(args.wiki_dir)
    else:
        wiki_dir = Path(__file__).parent.parent / "wiki"

    if not wiki_dir.exists():
        print(f"❌ Error: Wiki directory not found at {wiki_dir}")
        sys.exit(1)

    print("="*80)
    print("Wiki Link Checker")
    print("="*80)
    print()

    checker = WikiLinkChecker(wiki_dir, check_external=args.external)
    issue_count = checker.check_all()

    print("\n" + "="*80)
    report = checker.generate_report()
    print(report)
    print("="*80)

    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report)
        print(f"\n📄 Report saved to: {output_path}")

    if issue_count > 0:
        print(f"\n❌ Found {issue_count} broken links")
        sys.exit(1)
    else:
        print("\n✅ All links are valid!")
        sys.exit(0)


if __name__ == "__main__":
    main()
