#!/usr/bin/env python3
"""
Sync premium files from lobster to lobster-custom-databiomix.

Auto-discovers all matching files and rewrites imports.
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from difflib import unified_diff
from pathlib import Path
from typing import Optional

# ============================================================================
# Configuration
# ============================================================================

# Root paths (relative to this script)
SCRIPT_DIR = Path(__file__).parent
LOBSTER_ROOT = SCRIPT_DIR.parent
DATABIOMIX_ROOT = LOBSTER_ROOT.parent / "lobster-custom-databiomix"

# Source package name
SRC_PKG = "lobster"
DST_PKG = "lobster_custom_databiomix"

# Import rewrite rules: (from_pattern, to_replacement)
# Only premium services that exist in both packages get rewritten
IMPORT_REWRITES = [
    # Premium metadata services (exist in custom package)
    (r"from lobster\.services\.metadata\.disease_standardization_service",
     f"from {DST_PKG}.services.metadata.disease_standardization_service"),
    (r"from lobster\.services\.metadata\.microbiome_filtering_service",
     f"from {DST_PKG}.services.metadata.microbiome_filtering_service"),
    (r"from lobster\.services\.metadata\.metadata_filtering_service",
     f"from {DST_PKG}.services.metadata.metadata_filtering_service"),
    (r"from lobster\.services\.metadata\.sample_mapping_service",
     f"from {DST_PKG}.services.metadata.sample_mapping_service"),
    (r"from lobster\.services\.metadata\.identifier_provenance_service",
     f"from {DST_PKG}.services.metadata.identifier_provenance_service"),
    # Agent factory function
    (r'factory_function="lobster\.agents\.metadata_assistant',
     f'factory_function="{DST_PKG}.agents.metadata_assistant'),
]

# Files that should NEVER have imports rewritten (they only use public lobster.*)
NO_REWRITE_FILES = {
    "microbiome_filtering_service.py",  # Only uses lobster.core (public)
    "agent_configs.py",  # Only uses lobster.config (public)
}


# ============================================================================
# ANSI Colors (clean, minimal)
# ============================================================================

class C:
    """ANSI color codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RED = "\033[31m"
    CYAN = "\033[36m"


def ok(msg: str) -> str:
    return f"{C.GREEN}✓{C.RESET} {msg}"


def warn(msg: str) -> str:
    return f"{C.YELLOW}!{C.RESET} {msg}"


def err(msg: str) -> str:
    return f"{C.RED}✗{C.RESET} {msg}"


def info(msg: str) -> str:
    return f"{C.BLUE}→{C.RESET} {msg}"


def dim(msg: str) -> str:
    return f"{C.DIM}{msg}{C.RESET}"


def bold(msg: str) -> str:
    return f"{C.BOLD}{msg}{C.RESET}"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class FileMapping:
    """A file to sync from lobster to databiomix."""
    src: Path
    dst: Path
    relative: str  # e.g., "agents/metadata_assistant.py"
    needs_rewrite: bool = True

    @property
    def name(self) -> str:
        return self.src.name


@dataclass
class SyncResult:
    """Result of syncing a single file."""
    file: FileMapping
    changed: bool = False
    additions: int = 0
    deletions: int = 0
    rewrites: int = 0
    error: Optional[str] = None


@dataclass
class SyncSummary:
    """Summary of all sync operations."""
    files_synced: int = 0
    files_unchanged: int = 0
    files_failed: int = 0
    total_additions: int = 0
    total_deletions: int = 0
    total_rewrites: int = 0
    results: list = field(default_factory=list)


# ============================================================================
# Core Logic
# ============================================================================

def discover_sync_files() -> list[FileMapping]:
    """
    Auto-discover files that exist in BOTH lobster and databiomix.

    Strategy: Scan databiomix for .py files, find matching source in lobster.
    """
    mappings = []

    # Directories to scan in databiomix (relative to package root)
    scan_dirs = [
        "agents",
        "services/metadata",
        "services/orchestration",
        "config",
    ]

    databiomix_pkg = DATABIOMIX_ROOT / DST_PKG
    lobster_pkg = LOBSTER_ROOT / SRC_PKG

    for scan_dir in scan_dirs:
        dst_dir = databiomix_pkg / scan_dir
        if not dst_dir.exists():
            continue

        for dst_file in dst_dir.glob("*.py"):
            if dst_file.name.startswith("_"):
                continue  # Skip __init__.py, etc.
            if dst_file.name.endswith(".backup"):
                continue

            # Find corresponding source file
            relative = str(dst_file.relative_to(databiomix_pkg))

            # Handle config rename: agent_configs.py <- premium_agent_configs.py
            if relative == "config/agent_configs.py":
                src_file = lobster_pkg / "config" / "premium_agent_configs.py"
            else:
                src_file = lobster_pkg / relative

            if src_file.exists():
                needs_rewrite = dst_file.name not in NO_REWRITE_FILES
                mappings.append(FileMapping(
                    src=src_file,
                    dst=dst_file,
                    relative=relative,
                    needs_rewrite=needs_rewrite,
                ))

    # Sort by relative path for consistent output
    return sorted(mappings, key=lambda m: m.relative)


def rewrite_imports(content: str, filename: str) -> tuple[str, int]:
    """
    Rewrite imports from lobster.* to lobster_custom_databiomix.*.

    Returns: (rewritten_content, number_of_rewrites)
    """
    if filename in NO_REWRITE_FILES:
        return content, 0

    rewrite_count = 0
    for pattern, replacement in IMPORT_REWRITES:
        new_content, count = re.subn(pattern, replacement, content)
        if count > 0:
            rewrite_count += count
            content = new_content

    return content, rewrite_count


def compute_diff(old: str, new: str, filename: str) -> tuple[list[str], int, int]:
    """Compute unified diff and count additions/deletions."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)

    diff_lines = list(unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
    ))

    additions = sum(1 for line in diff_lines if line.startswith('+') and not line.startswith('+++'))
    deletions = sum(1 for line in diff_lines if line.startswith('-') and not line.startswith('---'))

    return diff_lines, additions, deletions


def sync_file(mapping: FileMapping, dry_run: bool = False) -> SyncResult:
    """Sync a single file with import rewriting."""
    result = SyncResult(file=mapping)

    try:
        # Read source
        src_content = mapping.src.read_text()

        # Rewrite imports
        if mapping.needs_rewrite:
            new_content, result.rewrites = rewrite_imports(src_content, mapping.name)
        else:
            new_content = src_content
            result.rewrites = 0

        # Check if target exists and compare
        if mapping.dst.exists():
            dst_content = mapping.dst.read_text()
            _, result.additions, result.deletions = compute_diff(
                dst_content, new_content, mapping.name
            )
            result.changed = dst_content != new_content
        else:
            result.changed = True
            result.additions = len(new_content.splitlines())

        # Apply changes if not dry-run
        if not dry_run and result.changed:
            mapping.dst.parent.mkdir(parents=True, exist_ok=True)
            mapping.dst.write_text(new_content)

    except Exception as e:
        result.error = str(e)

    return result


def print_file_result(result: SyncResult, verbose: bool = False):
    """Print result for a single file."""
    rel = result.file.relative

    if result.error:
        print(f"  {err(rel)}")
        print(f"    {C.RED}Error: {result.error}{C.RESET}")
        return

    if not result.changed:
        print(f"  {ok(rel)} {dim('(no changes)')}")
        return

    # Build change summary
    parts = []
    if result.additions:
        parts.append(f"{C.GREEN}+{result.additions}{C.RESET}")
    if result.deletions:
        parts.append(f"{C.RED}-{result.deletions}{C.RESET}")
    if result.rewrites:
        parts.append(f"{C.CYAN}{result.rewrites} rewrites{C.RESET}")

    change_str = " ".join(parts) if parts else "modified"
    print(f"  {ok(rel)} [{change_str}]")


def run_sync(dry_run: bool = False, verbose: bool = False) -> SyncSummary:
    """Run the full sync operation."""
    summary = SyncSummary()

    # Validate paths
    if not LOBSTER_ROOT.exists():
        print(err(f"Lobster root not found: {LOBSTER_ROOT}"))
        sys.exit(1)

    if not DATABIOMIX_ROOT.exists():
        print(err(f"Databiomix root not found: {DATABIOMIX_ROOT}"))
        print(dim(f"  Expected at: {DATABIOMIX_ROOT}"))
        sys.exit(1)

    # Discover files
    mappings = discover_sync_files()

    if not mappings:
        print(warn("No files found to sync"))
        return summary

    # Header
    mode = f"{C.YELLOW}DRY RUN{C.RESET}" if dry_run else f"{C.GREEN}SYNC{C.RESET}"
    print()
    print(f"{bold('lobster')} → {bold('lobster-custom-databiomix')} [{mode}]")
    print(dim("─" * 60))
    print()

    # Process files
    for mapping in mappings:
        result = sync_file(mapping, dry_run=dry_run)
        summary.results.append(result)

        if result.error:
            summary.files_failed += 1
        elif result.changed:
            summary.files_synced += 1
            summary.total_additions += result.additions
            summary.total_deletions += result.deletions
            summary.total_rewrites += result.rewrites
        else:
            summary.files_unchanged += 1

        print_file_result(result, verbose=verbose)

    # Summary
    print()
    print(dim("─" * 60))

    total = len(mappings)

    if dry_run:
        if summary.files_synced > 0:
            print(f"{C.YELLOW}Would sync:{C.RESET} {summary.files_synced}/{total} files")
            print(f"  {C.GREEN}+{summary.total_additions}{C.RESET} / {C.RED}-{summary.total_deletions}{C.RESET} lines, {C.CYAN}{summary.total_rewrites}{C.RESET} import rewrites")
        else:
            print(ok("All files up to date"))
    else:
        if summary.files_synced > 0:
            print(ok(f"Synced {summary.files_synced}/{total} files"))
            print(f"  {C.GREEN}+{summary.total_additions}{C.RESET} / {C.RED}-{summary.total_deletions}{C.RESET} lines, {C.CYAN}{summary.total_rewrites}{C.RESET} import rewrites")
        else:
            print(ok("All files up to date"))

    if summary.files_unchanged > 0:
        print(dim(f"  {summary.files_unchanged} unchanged"))

    if summary.files_failed > 0:
        print(err(f"{summary.files_failed} files failed"))

    # Next steps
    if not dry_run and summary.files_synced > 0:
        print()
        print(bold("Next steps:"))
        print(f"  cd {DATABIOMIX_ROOT}")
        print("  git diff")
        print("  pytest tests/ -v")
        print("  git add . && git commit -m 'sync: update from lobster'")

    print()
    return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sync premium files from lobster to lobster-custom-databiomix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dry-run    Preview changes without applying
  %(prog)s              Apply changes
  %(prog)s -v           Verbose output
        """,
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview changes without applying",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List files that would be synced and exit",
    )

    args = parser.parse_args()

    if args.list:
        mappings = discover_sync_files()
        print(f"\n{bold('Files to sync:')} ({len(mappings)} files)\n")
        for m in mappings:
            src_name = m.src.name
            if src_name != m.dst.name:
                print(f"  {m.relative} {dim(f'← {src_name}')}")
            else:
                print(f"  {m.relative}")
        print()
        return

    summary = run_sync(dry_run=args.dry_run, verbose=args.verbose)

    # Exit with error code if any failures
    if summary.files_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
