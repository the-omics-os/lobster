#!/usr/bin/env python3
"""
Wiki Code Examples Tester
Tests all Python code examples in wiki markdown files for syntax and import validity.

Usage:
    python scripts/test_wiki_code_examples.py [--verbose] [--output report.md]

Categories:
    - Executable: Can be run as-is (imports, simple operations)
    - Template: Contains placeholders (your_service, your_agent)
    - Snippet: Partial code requiring context
    - Configuration: YAML/JSON/TOML examples

Options:
    --verbose    Show detailed test output
    --output     Save report to file
"""
import re
import sys
import ast
import argparse
import importlib.util
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time


class CodeBlockType(Enum):
    """Type classification for code blocks."""
    EXECUTABLE = "executable"
    TEMPLATE = "template"
    SNIPPET = "snippet"
    CONFIGURATION = "configuration"
    BASH = "bash"
    UNKNOWN = "unknown"


@dataclass
class CodeBlock:
    """Represents a code block found in markdown."""
    file: str
    line: int
    language: str
    content: str
    type: CodeBlockType
    test_result: Optional[str] = None
    error_message: Optional[str] = None


class WikiCodeTester:
    """Test code examples in wiki markdown files."""

    # Template placeholders that indicate template code
    TEMPLATE_MARKERS = [
        'your_service', 'your_agent', 'your_modality', 'your_file',
        'your_dataset', 'your_analysis', 'your_parameter', 'your_config',
        'YourService', 'YourAgent', 'YourClass', 'your_function',
        '[your_', 'TODO:', 'FIXME:', '# Template example',
        '# Example template', '<<EOF', 'cat <<', '__PLACEHOLDER__'
    ]

    # Snippet indicators (partial code)
    SNIPPET_MARKERS = [
        '...', '# ...', '# (continued)', '# Add more', '# Implementation',
        '# Your code here', '# Fill in', '# Complete'
    ]

    # Configuration file markers
    CONFIG_MARKERS = [
        'yaml', 'toml', 'json', 'ini', 'cfg', 'conf'
    ]

    def __init__(self, wiki_dir: Path, verbose: bool = False):
        self.wiki_dir = wiki_dir
        self.verbose = verbose
        self.code_blocks: List[CodeBlock] = []
        self.stats = {
            'files_checked': 0,
            'total_blocks': 0,
            'executable': 0,
            'template': 0,
            'snippet': 0,
            'configuration': 0,
            'bash': 0,
            'unknown': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0
        }

    def find_markdown_files(self) -> List[Path]:
        """Find all .md files in wiki directory."""
        files = sorted(self.wiki_dir.glob("*.md"))
        # Exclude special files
        exclude = {
            'PHASE2_CODE_VERIFICATION_REPORT.md',
            'PHASE2_VERIFICATION_SUMMARY.md',
            'PHASE2_FEATURES_DOCUMENTATION_REPORT.md',
            'PHASE2_UX_ENHANCEMENT_REPORT.md',
            'DOCUMENTATION_GAP_ANALYSIS_2025-11-16.md'
        }
        return [f for f in files if f.name not in exclude]

    def extract_code_blocks(self, file_path: Path) -> List[CodeBlock]:
        """Extract all code blocks from a markdown file."""
        blocks = []
        content = file_path.read_text(encoding='utf-8')
        lines = content.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i]

            # Match code fence: ```language or ```
            match = re.match(r'^```(\w+)?', line)
            if match:
                start_line = i + 1
                language = match.group(1) or 'unknown'
                i += 1

                # Collect code until closing fence
                code_lines = []
                while i < len(lines) and not lines[i].startswith('```'):
                    code_lines.append(lines[i])
                    i += 1

                if code_lines:  # Only process non-empty blocks
                    code_content = '\n'.join(code_lines)
                    block_type = self._classify_code_block(code_content, language)

                    blocks.append(CodeBlock(
                        file=file_path.name,
                        line=start_line,
                        language=language,
                        content=code_content,
                        type=block_type
                    ))

            i += 1

        return blocks

    def _classify_code_block(self, code: str, language: str) -> CodeBlockType:
        """Classify code block type based on content and language."""
        # Check language tag
        if language.lower() in ['bash', 'sh', 'shell', 'console']:
            return CodeBlockType.BASH

        if language.lower() in self.CONFIG_MARKERS:
            return CodeBlockType.CONFIGURATION

        if language.lower() not in ['python', 'py', '']:
            return CodeBlockType.UNKNOWN

        # Check for template markers
        if any(marker in code for marker in self.TEMPLATE_MARKERS):
            return CodeBlockType.TEMPLATE

        # Check for snippet markers
        if any(marker in code for marker in self.SNIPPET_MARKERS):
            return CodeBlockType.SNIPPET

        # Check if it looks like complete, executable code
        try:
            ast.parse(code)
            # If it parses and has imports or function calls, likely executable
            if ('import' in code or 'from' in code) or \
               ('def ' in code or 'class ' in code) or \
               (len(code.strip().split('\n')) >= 5):
                return CodeBlockType.EXECUTABLE
            else:
                return CodeBlockType.SNIPPET
        except SyntaxError:
            # If it doesn't parse, check if it's a snippet or template
            if '...' in code or '# ...' in code:
                return CodeBlockType.SNIPPET
            return CodeBlockType.UNKNOWN

    def test_code_block(self, block: CodeBlock) -> Tuple[bool, Optional[str]]:
        """
        Test a code block for syntax and import validity.

        Returns:
            Tuple of (passed, error_message)
        """
        try:
            # Test 1: Syntax check
            ast.parse(block.content)

            # Test 2: Check imports
            tree = ast.parse(block.content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if not self._check_import(alias.name):
                            return False, f"Import not available: {alias.name}"

                elif isinstance(node, ast.ImportFrom):
                    module = node.module
                    if module and not self._check_import(module):
                        return False, f"Import not available: {module}"

            return True, None

        except SyntaxError as e:
            return False, f"Syntax error: {e.msg} (line {e.lineno})"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    def _check_import(self, module_name: str) -> bool:
        """Check if a module is available for import."""
        # Allow common standard library modules
        stdlib_modules = {
            'os', 'sys', 're', 'json', 'time', 'datetime', 'pathlib',
            'typing', 'dataclasses', 'collections', 'itertools', 'functools',
            'argparse', 'logging', 'unittest', 'pytest', 'ast', 'inspect'
        }

        # Allow project modules
        project_modules = {
            'lobster', 'anndata', 'scanpy', 'numpy', 'pandas', 'matplotlib',
            'plotly', 'langchain', 'langgraph', 'pydantic', 're'
        }

        base_module = module_name.split('.')[0]
        if base_module in stdlib_modules or base_module in project_modules:
            return True

        # Try actual import check (optional, can be slow)
        try:
            spec = importlib.util.find_spec(base_module)
            return spec is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    def test_file(self, file_path: Path) -> None:
        """Test all code blocks in a single markdown file."""
        if self.verbose:
            print(f"Testing {file_path.name}...", end=' ')

        blocks = self.extract_code_blocks(file_path)
        if not blocks:
            if self.verbose:
                print("(no code blocks)")
            return

        file_results = []
        for block in blocks:
            self.stats['total_blocks'] += 1
            self.stats[block.type.value] += 1

            # Only test executable Python blocks
            if block.type == CodeBlockType.EXECUTABLE:
                passed, error = self.test_code_block(block)
                block.test_result = "PASS" if passed else "FAIL"
                block.error_message = error

                if passed:
                    self.stats['passed'] += 1
                    file_results.append("✓")
                else:
                    self.stats['failed'] += 1
                    file_results.append("✗")
            else:
                self.stats['skipped'] += 1
                block.test_result = "SKIP"
                file_results.append("·")

            self.code_blocks.append(block)

        if self.verbose:
            result_str = ''.join(file_results)
            failed_count = file_results.count("✗")
            if failed_count > 0:
                print(f"[{result_str}] ❌ {failed_count} failed")
            else:
                print(f"[{result_str}] ✅")

        self.stats['files_checked'] += 1

    def test_all(self) -> int:
        """Test all markdown files and return failure count."""
        files = self.find_markdown_files()
        print(f"Found {len(files)} markdown files in {self.wiki_dir}")
        print(f"Testing code examples...\n")

        for file_path in files:
            self.test_file(file_path)

        return self.stats['failed']

    def generate_report(self) -> str:
        """Generate markdown report of test results."""
        report = [
            "# Wiki Code Examples Test Report",
            "",
            f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- **Files Checked**: {self.stats['files_checked']}",
            f"- **Total Code Blocks**: {self.stats['total_blocks']}",
            "",
            "### Code Block Categories",
            "",
            f"- **Executable**: {self.stats['executable']} (tested)",
            f"- **Template**: {self.stats['template']} (skipped - contains placeholders)",
            f"- **Snippet**: {self.stats['snippet']} (skipped - partial code)",
            f"- **Configuration**: {self.stats['configuration']} (skipped - YAML/JSON/TOML)",
            f"- **Bash/Shell**: {self.stats['bash']} (skipped - shell commands)",
            f"- **Unknown**: {self.stats['unknown']} (skipped)",
            "",
            "### Test Results",
            "",
            f"- ✅ **Passed**: {self.stats['passed']}",
            f"- ❌ **Failed**: {self.stats['failed']}",
            f"- ⊘ **Skipped**: {self.stats['skipped']}",
            ""
        ]

        # Calculate accuracy percentage
        testable = self.stats['executable']
        if testable > 0:
            accuracy = (self.stats['passed'] / testable) * 100
            report.append(f"**Accuracy**: {accuracy:.1f}% ({self.stats['passed']}/{testable})")
        else:
            report.append("**Accuracy**: N/A (no executable code blocks found)")

        report.append("")

        # Report failed tests
        failed_blocks = [b for b in self.code_blocks if b.test_result == "FAIL"]
        if failed_blocks:
            report.extend([
                "## Failed Tests",
                "",
                f"Found {len(failed_blocks)} code blocks with issues:",
                ""
            ])

            # Group by file
            by_file: Dict[str, List[CodeBlock]] = {}
            for block in failed_blocks:
                by_file.setdefault(block.file, []).append(block)

            for file, blocks in sorted(by_file.items()):
                report.append(f"### {file} ({len(blocks)} issues)")
                report.append("")
                for block in blocks:
                    report.append(f"**Line {block.line}** ({block.language}):")
                    report.append(f"```")
                    report.append(f"Error: {block.error_message}")
                    report.append(f"```")
                    report.append("")
        else:
            report.extend([
                "## Test Results",
                "",
                "✅ **All executable code blocks passed syntax and import checks!**",
                ""
            ])

        # Add statistics by file
        report.extend([
            "## Code Blocks by File",
            "",
            "| File | Total | Executable | Template | Snippet | Config | Bash | Passed | Failed |",
            "|------|-------|-----------|----------|---------|--------|------|--------|--------|"
        ])

        # Aggregate by file
        file_stats: Dict[str, Dict] = {}
        for block in self.code_blocks:
            if block.file not in file_stats:
                file_stats[block.file] = {
                    'total': 0,
                    'executable': 0,
                    'template': 0,
                    'snippet': 0,
                    'configuration': 0,
                    'bash': 0,
                    'unknown': 0,
                    'passed': 0,
                    'failed': 0
                }

            file_stats[block.file]['total'] += 1
            file_stats[block.file][block.type.value] += 1
            if block.test_result == "PASS":
                file_stats[block.file]['passed'] += 1
            elif block.test_result == "FAIL":
                file_stats[block.file]['failed'] += 1

        for file in sorted(file_stats.keys()):
            stats = file_stats[file]
            report.append(
                f"| {file} | {stats['total']} | {stats['executable']} | "
                f"{stats['template']} | {stats['snippet']} | {stats['configuration']} | "
                f"{stats['bash']} | {stats['passed']} | {stats['failed']} |"
            )

        report.extend([
            "",
            "## Recommendations",
            ""
        ])

        if self.stats['failed'] > 0:
            report.append("### Failed Code Blocks")
            report.append("- Review syntax errors and fix code examples")
            report.append("- Verify imports are available in the project")
            report.append("- Consider marking incomplete code with `# Template example`")
            report.append("")

        report.append("### Template Code")
        report.append("- Template code with placeholders is automatically skipped")
        report.append("- Mark template examples with `# Template example` comment")
        report.append("- Use clear placeholder names: `your_service`, `your_agent`, etc.")
        report.append("")

        report.append("### Best Practices")
        report.append("- Keep executable examples complete and runnable")
        report.append("- Add type hints for better code clarity")
        report.append("- Include necessary imports in code examples")
        report.append("- Test complex examples in actual environment")
        report.append("")

        return "\n".join(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Test Python code examples in wiki markdown files'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed test output'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Save report to file'
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
    print("Wiki Code Examples Tester")
    print("=" * 80)
    print()

    tester = WikiCodeTester(wiki_dir, verbose=args.verbose)
    failure_count = tester.test_all()

    print("\n" + "=" * 80)
    report = tester.generate_report()
    print(report)
    print("=" * 80)

    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report)
        print(f"\n📄 Report saved to: {output_path}")

    if failure_count > 0:
        print(f"\n❌ Found {failure_count} code blocks with issues")
        sys.exit(1)
    else:
        print(f"\n✅ All {tester.stats['executable']} executable code blocks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
