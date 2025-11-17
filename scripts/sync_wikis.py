#!/usr/bin/env python3
"""
Wiki synchronization script for lobster documentation.
Syncs local wiki/ directory to local wiki repositories (lobster.wiki and lobster-local.wiki).
User manually pushes changes after review.
"""

import os
import shutil
from pathlib import Path
import argparse
from datetime import datetime


class WikiSync:
    """Handles wiki synchronization to local wiki repository directories."""

    def __init__(self, source_dir: Path, allowlist_file: Path = None):
        self.source_dir = source_dir
        self.allowlist_file = allowlist_file
        self.stats = {
            "files_synced": 0,
            "files_skipped": 0,
            "total_size": 0
        }

    def load_allowlist(self) -> set:
        """Load and parse allowlist patterns."""
        if not self.allowlist_file or not self.allowlist_file.exists():
            return None

        allowed_files = set()
        excluded_files = set()

        with open(self.allowlist_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Handle explicit exclusions (lines starting with !)
                    if line.startswith('!'):
                        excluded_files.add(line[1:])
                    else:
                        allowed_files.add(line)

        return allowed_files, excluded_files

    def is_allowed(self, filename: str, allowlist_data) -> bool:
        """Check if file is in allowlist and not explicitly excluded."""
        if allowlist_data is None:
            return True  # No allowlist means sync everything

        allowed_files, excluded_files = allowlist_data

        # Check explicit exclusions first
        if filename in excluded_files:
            return False

        # Check for data/** pattern exclusion
        if 'data/**' in excluded_files and filename.startswith('data/'):
            return False

        # Direct filename match
        if filename in allowed_files:
            return True

        # Check for data/** pattern
        if 'data/**' in allowed_files and filename.startswith('data/'):
            return True

        return False

    def sync_to_directory(self, target_dir: Path, use_allowlist: bool = False):
        """Sync wiki files to a local directory.

        Args:
            target_dir: Local wiki repository directory (e.g., /path/to/lobster.wiki)
            use_allowlist: If True, filter files using allowlist
        """
        print(f"\n{'='*60}")
        print(f"Syncing to: {target_dir}")
        print(f"Using allowlist: {use_allowlist}")
        print(f"{'='*60}\n")

        # Verify target directory exists and has .git
        if not target_dir.exists():
            print(f"❌ Error: Target directory '{target_dir}' does not exist")
            return False

        if not (target_dir / '.git').exists():
            print(f"❌ Error: '{target_dir}' is not a git repository (no .git directory)")
            return False

        # Load allowlist if needed
        allowlist_data = None
        if use_allowlist:
            allowlist_data = self.load_allowlist()
            if allowlist_data:
                allowed_files, excluded_files = allowlist_data
                print(f"Loaded {len(allowed_files)} allowed patterns, {len(excluded_files)} excluded patterns")

        # Remove all existing files (except .git)
        print("Cleaning existing wiki files...")
        for item in target_dir.iterdir():
            if item.name != '.git':
                if item.is_dir():
                    shutil.rmtree(item)
                    print(f"  🗑️  Removed directory: {item.name}/")
                else:
                    item.unlink()
                    print(f"  🗑️  Removed file: {item.name}")

        # Copy files from source
        print("\nCopying wiki files...")
        self.stats = {"files_synced": 0, "files_skipped": 0, "total_size": 0}

        for item in self.source_dir.iterdir():
            if item.name.startswith('.'):
                continue

            if item.is_file():
                if self.is_allowed(item.name, allowlist_data):
                    shutil.copy2(item, target_dir / item.name)
                    self.stats["files_synced"] += 1
                    self.stats["total_size"] += item.stat().st_size
                    print(f"  ✅ {item.name}")
                else:
                    self.stats["files_skipped"] += 1
                    print(f"  ⏭️  {item.name} (excluded)")
            elif item.is_dir() and not item.name.startswith('.'):
                # Handle directories (like data/)
                dir_name = item.name
                if self.is_allowed(f"{dir_name}/**", allowlist_data):
                    shutil.copytree(item, target_dir / dir_name)
                    # Count files in directory
                    for root, _, files in os.walk(item):
                        self.stats["files_synced"] += len(files)
                        for file in files:
                            file_path = Path(root) / file
                            self.stats["total_size"] += file_path.stat().st_size
                    print(f"  ✅ {dir_name}/ (directory)")
                else:
                    self.stats["files_skipped"] += 1
                    print(f"  ⏭️  {dir_name}/ (excluded)")

        # Print summary
        print(f"\n✅ Successfully synced to {target_dir}")
        print(f"\nSummary:")
        print(f"  Files synced: {self.stats['files_synced']}")
        if use_allowlist:
            print(f"  Files excluded: {self.stats['files_skipped']}")
        print(f"  Total size: {self.stats['total_size'] / 1024 / 1024:.2f} MB")

        return True


def main():
    parser = argparse.ArgumentParser(
        description='Sync wiki files to local wiki repository directories'
    )
    parser.add_argument(
        '--source',
        default='wiki',
        help='Source wiki directory (default: wiki)'
    )
    parser.add_argument(
        '--allowlist',
        default='scripts/wiki_public_allowlist.txt',
        help='Allowlist file for public wiki (default: scripts/wiki_public_allowlist.txt)'
    )
    parser.add_argument(
        '--private-wiki',
        default='/Users/tyo/GITHUB/omics-os/lobster.wiki',
        help='Private wiki repository path (default: /Users/tyo/GITHUB/omics-os/lobster.wiki)'
    )
    parser.add_argument(
        '--public-wiki',
        default='/Users/tyo/GITHUB/omics-os/lobster-local.wiki',
        help='Public wiki repository path (default: /Users/tyo/GITHUB/omics-os/lobster-local.wiki)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run (show what would be synced without making changes)'
    )
    parser.add_argument(
        '--skip-private',
        action='store_true',
        help='Skip syncing to private wiki'
    )
    parser.add_argument(
        '--skip-public',
        action='store_true',
        help='Skip syncing to public wiki'
    )

    args = parser.parse_args()

    # Resolve paths
    source_dir = Path(args.source).resolve()
    allowlist_file = Path(args.allowlist).resolve()
    private_wiki = Path(args.private_wiki).resolve()
    public_wiki = Path(args.public_wiki).resolve()

    if not source_dir.exists():
        print(f"❌ Error: Source directory '{source_dir}' does not exist")
        return 1

    if not source_dir.is_dir():
        print(f"❌ Error: '{source_dir}' is not a directory")
        return 1

    print(f"🚀 Starting wiki synchronization")
    print(f"Source: {source_dir}")
    print(f"Allowlist: {allowlist_file}")
    print(f"Private wiki: {private_wiki}")
    print(f"Public wiki: {public_wiki}")
    print()

    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made\n")
        syncer = WikiSync(source_dir, allowlist_file)

        # Show what would be synced to each repo
        print("📚 Files in source wiki:")
        all_files = []
        for item in source_dir.iterdir():
            if not item.name.startswith('.'):
                if item.is_file():
                    all_files.append(item.name)
                elif item.is_dir():
                    all_files.append(f"{item.name}/")

        for file in sorted(all_files):
            print(f"  - {file}")

        # Show private wiki sync (all files)
        print(f"\n📖 Private wiki ({private_wiki}) would receive:")
        print("  (All files - no filtering)")
        for file in sorted(all_files):
            print(f"  ✅ {file}")

        # Check allowlist for public wiki
        allowlist_data = syncer.load_allowlist()
        if allowlist_data:
            allowed_files, excluded_files = allowlist_data
            print(f"\n📋 Public wiki ({public_wiki}) would receive:")
            print(f"  ({len(allowed_files)} allowed patterns, {len(excluded_files)} excluded patterns)")

            excluded = []
            for file in sorted(all_files):
                if syncer.is_allowed(file.rstrip('/'), allowlist_data):
                    print(f"  ✅ {file}")
                else:
                    print(f"  ❌ {file} (excluded)")
                    excluded.append(file)

            if excluded:
                print(f"\n⚠️  {len(excluded)} files will be excluded from public wiki:")
                for file in excluded:
                    print(f"  - {file}")

        print("\n✅ Dry run complete. No changes were made.")
        print("\n💡 To apply changes, run without --dry-run flag")
        print("   After syncing, review and push from each wiki directory:")
        print(f"     cd {private_wiki} && git status && git add -A && git commit && git push")
        print(f"     cd {public_wiki} && git status && git add -A && git commit && git push")
        return 0

    # Create syncer
    syncer = WikiSync(source_dir, allowlist_file)

    # Sync to private wiki (no filtering)
    if not args.skip_private:
        success = syncer.sync_to_directory(
            private_wiki,
            use_allowlist=False
        )

        if not success:
            print("❌ Failed to sync to private wiki")
            return 1
    else:
        print("⏭️  Skipping private wiki sync")

    # Sync to public wiki (with filtering)
    if not args.skip_public:
        success = syncer.sync_to_directory(
            public_wiki,
            use_allowlist=True
        )

        if not success:
            print("❌ Failed to sync to public wiki")
            return 1
    else:
        print("⏭️  Skipping public wiki sync")

    # Print next steps
    print("\n🎉 Wiki synchronization completed successfully!\n")
    print("📝 Next steps:")
    print("   1. Review changes in each wiki directory:")
    if not args.skip_private:
        print(f"      cd {private_wiki} && git status")
    if not args.skip_public:
        print(f"      cd {public_wiki} && git status")
    print("\n   2. Commit and push changes:")
    if not args.skip_private:
        print(f"      cd {private_wiki}")
        print(f"      git add -A")
        print(f"      git commit -m 'Sync wiki from lobster repository'")
        print(f"      git push")
    if not args.skip_public:
        print(f"\n      cd {public_wiki}")
        print(f"      git add -A")
        print(f"      git commit -m 'Sync wiki from lobster repository (filtered)'")
        print(f"      git push")

    return 0


if __name__ == '__main__':
    exit(main())
