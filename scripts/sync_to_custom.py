#!/usr/bin/env python3
"""
Simple sync script for custom packages.
Copies PREMIUM agent code from lobster/ to lobster-custom-{package}/.

Usage:
    python scripts/sync_to_custom.py --package databiomix --dry-run
    python scripts/sync_to_custom.py --package databiomix
"""

import argparse
import shutil
from pathlib import Path
import fnmatch


class CustomPackageSync:
    """Simple file copier for custom packages."""

    def __init__(self, package_name: str, dry_run: bool = False):
        self.package_name = package_name
        self.dry_run = dry_run

        # Paths
        self.source_dir = Path(__file__).parent.parent  # lobster/
        self.target_dir = self.source_dir.parent / f"lobster-custom-{package_name}"
        self.target_package_dir = self.target_dir / f"lobster_custom_{package_name}"
        self.allowlist_file = self.source_dir / "scripts/custom_package_allowlist.txt"

        self.copied_files = []
        self.skipped_files = []

    def load_allowlist(self) -> tuple[list[str], list[str]]:
        """Load allowlist and return (includes, excludes)."""
        includes = []
        excludes = []

        with open(self.allowlist_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('!'):
                    excludes.append(line[1:])
                else:
                    includes.append(line)

        return includes, excludes

    def is_allowed(self, filepath: str, includes: list, excludes: list) -> bool:
        """Check if file matches allowlist patterns."""
        # Check excludes first
        for pattern in excludes:
            if fnmatch.fnmatch(filepath, pattern):
                return False

        # Check includes
        for pattern in includes:
            if fnmatch.fnmatch(filepath, pattern):
                return True

        return False

    def sync_files(self):
        """Sync files from lobster/ to custom package."""
        includes, excludes = self.load_allowlist()

        # Find all files in lobster/
        for source_file in self.source_dir.rglob("*"):
            if not source_file.is_file():
                continue

            # Get relative path from source_dir
            rel_path = source_file.relative_to(self.source_dir)
            rel_str = str(rel_path).replace("\\", "/")

            # Skip irrelevant files
            if "__pycache__" in rel_str or rel_str.endswith((".pyc", ".pyo")):
                continue

            # Check allowlist
            if not self.is_allowed(rel_str, includes, excludes):
                self.skipped_files.append(rel_str)
                continue

            # Determine target path
            # Map lobster/ paths to lobster_custom_{package}/ paths
            target_path = self.target_package_dir / rel_path

            if self.dry_run:
                print(f"[DRY-RUN] Would copy: {rel_str}")
                self.copied_files.append(rel_str)
            else:
                # Create parent directories
                target_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy file
                shutil.copy2(source_file, target_path)
                print(f"✓ Copied: {rel_str}")
                self.copied_files.append(rel_str)

        # Print summary
        print(f"\n{'='*60}")
        print(f"Sync Summary ({'DRY-RUN' if self.dry_run else 'COMPLETED'})")
        print(f"{'='*60}")
        print(f"Package: {self.package_name}")
        print(f"Source: {self.source_dir}")
        print(f"Target: {self.target_package_dir}")
        print(f"Files copied: {len(self.copied_files)}")
        print(f"Files skipped: {len(self.skipped_files)}")

        if not self.dry_run:
            print(f"\nNext steps:")
            print(f"1. cd {self.target_dir}")
            print(f"2. python -m build")
            print(f"3. aws s3 cp dist/*.whl s3://lobster-license-packages-649207544517/packages/lobster-custom-{self.package_name}/...")


def main():
    parser = argparse.ArgumentParser(
        description="Sync PREMIUM agents to custom package"
    )
    parser.add_argument(
        "--package",
        required=True,
        help="Package name (e.g., 'databiomix')"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without actually copying"
    )

    args = parser.parse_args()

    syncer = CustomPackageSync(args.package, args.dry_run)
    syncer.sync_files()


if __name__ == "__main__":
    main()
