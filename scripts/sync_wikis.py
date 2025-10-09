#!/usr/bin/env python3
"""
Wiki synchronization script for lobster documentation.
Syncs local wiki to both lobster.wiki (full) and lobster-local.wiki (filtered).
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import argparse
from datetime import datetime


class WikiSync:
    """Handles wiki synchronization to GitHub wiki repositories."""

    def __init__(self, source_dir: Path, allowlist_file: Path = None):
        self.source_dir = source_dir
        self.allowlist_file = allowlist_file
        self.stats = {
            "files_synced": 0,
            "files_skipped": 0,
            "total_size": 0
        }
        # Get PAT from environment for HTTPS authentication
        self.wiki_pat = os.environ.get('WIKI_SYNC_PAT', '')

    def inject_pat_token(self, url: str) -> str:
        """Inject PAT token into HTTPS URLs for authentication.

        Args:
            url: Git repository URL (SSH or HTTPS format)

        Returns:
            URL with PAT token injected if HTTPS, otherwise unchanged
        """
        # If no PAT token available, return URL as-is
        if not self.wiki_pat:
            return url

        # Only inject token into HTTPS URLs
        if url.startswith('https://github.com/'):
            # Format: https://TOKEN@github.com/org/repo.git
            return url.replace('https://github.com/', f'https://{self.wiki_pat}@github.com/')

        # Return SSH URLs unchanged
        return url
    
    def load_allowlist(self) -> set:
        """Load and parse allowlist patterns."""
        if not self.allowlist_file or not self.allowlist_file.exists():
            return None
        
        allowed_files = set()
        with open(self.allowlist_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    allowed_files.add(line)
        return allowed_files
    
    def is_allowed(self, filename: str, allowed_files: set) -> bool:
        """Check if file is in allowlist."""
        if allowed_files is None:
            return True  # No allowlist means sync everything
        
        # Direct filename match
        if filename in allowed_files:
            return True
        
        # Check for data/** pattern
        if 'data/**' in allowed_files and filename.startswith('data/'):
            return True
        
        return False
    
    def sync_to_repo(self, wiki_repo_url: str, branch: str = "master",
                     use_allowlist: bool = False, force: bool = False):
        """Sync wiki files to a GitHub wiki repository."""
        print(f"\n{'='*60}")
        print(f"Syncing to: {wiki_repo_url}")
        print(f"Using allowlist: {use_allowlist}")
        print(f"{'='*60}\n")

        # Inject PAT token into URL if using HTTPS
        authenticated_url = self.inject_pat_token(wiki_repo_url)

        # Load allowlist if needed
        allowed_files = None
        if use_allowlist:
            allowed_files = self.load_allowlist()
            if allowed_files:
                print(f"Loaded {len(allowed_files)} allowed patterns")

        # Create temporary directory for wiki clone
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wiki_path = temp_path / "wiki"

            # Clone wiki repository
            print(f"Cloning wiki repository...")
            try:
                subprocess.run(
                    ['git', 'clone', authenticated_url, str(wiki_path)],
                    check=True,
                    capture_output=True,
                    text=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Error cloning repository: {e.stderr}")
                return False

            # Configure git
            subprocess.run(['git', 'config', 'user.name', 'cewinharhar'], check=True, cwd=wiki_path)
            subprocess.run(['git', 'config', 'user.email', 'kevin.yar@outlook.com'], check=True, cwd=wiki_path)
            
            # Remove all existing files (except .git)
            print("Cleaning existing wiki files...")
            for item in wiki_path.iterdir():
                if item.name != '.git':
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
            
            # Copy files from source
            print("Copying wiki files...")
            self.stats = {"files_synced": 0, "files_skipped": 0, "total_size": 0}
            
            for item in self.source_dir.iterdir():
                if item.name.startswith('.'):
                    continue
                
                if item.is_file():
                    if self.is_allowed(item.name, allowed_files):
                        shutil.copy2(item, wiki_path / item.name)
                        self.stats["files_synced"] += 1
                        self.stats["total_size"] += item.stat().st_size
                        print(f"  ✅ {item.name}")
                    else:
                        self.stats["files_skipped"] += 1
                        print(f"  ⏭️  {item.name} (excluded)")
                elif item.is_dir() and not item.name.startswith('.'):
                    # Handle directories (like data/)
                    dir_name = item.name
                    if self.is_allowed(f"{dir_name}/**", allowed_files):
                        shutil.copytree(item, wiki_path / dir_name)
                        # Count files in directory
                        for root, _, files in os.walk(item):
                            self.stats["files_synced"] += len(files)
                            for file in files:
                                file_path = Path(root) / file
                                self.stats["total_size"] += file_path.stat().st_size
                        print(f"  ✅ {dir_name}/ (directory)")
            
            # Check if there are changes
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                cwd=wiki_path
            )
            
            if not result.stdout.strip():
                print("\n✅ Wiki is already up to date!")
                return True
            
            # Add all files
            subprocess.run(['git', 'add', '-A'], check=True, cwd=wiki_path)
            
            # Create commit
            commit_msg = f"Sync wiki from lobster repository\n\n"
            commit_msg += f"Timestamp: {datetime.now().isoformat()}\n"
            commit_msg += f"Files synced: {self.stats['files_synced']}\n"
            if use_allowlist:
                commit_msg += f"Files excluded: {self.stats['files_skipped']}\n"
            commit_msg += f"Total size: {self.stats['total_size'] / 1024 / 1024:.2f} MB"
            
            subprocess.run(['git', 'commit', '-m', commit_msg], check=True, cwd=wiki_path)
            
            # Update remote URL to use authenticated URL for push
            subprocess.run(
                ['git', 'remote', 'set-url', 'origin', authenticated_url],
                check=True,
                capture_output=True,
                cwd=wiki_path
            )

            # Push changes
            print("\nPushing changes...")
            push_cmd = ['git', 'push', 'origin', branch]
            if force:
                push_cmd.insert(2, '--force')

            try:
                subprocess.run(push_cmd, check=True, capture_output=True, text=True, cwd=wiki_path)
                print(f"\n✅ Successfully synced to {wiki_repo_url}")
            except subprocess.CalledProcessError as e:
                print(f"Error pushing changes: {e.stderr}")
                return False
            
            # Print summary
            print(f"\nSummary:")
            print(f"  Files synced: {self.stats['files_synced']}")
            if use_allowlist:
                print(f"  Files excluded: {self.stats['files_skipped']}")
            print(f"  Total size: {self.stats['total_size'] / 1024 / 1024:.2f} MB")
            
        return True


def main():
    parser = argparse.ArgumentParser(description='Sync wiki to GitHub wiki repositories')
    parser.add_argument(
        '--source', 
        default='wiki',
        help='Source wiki directory (default: wiki)'
    )
    parser.add_argument(
        '--allowlist',
        default='scripts/wiki_public_allowlist.txt',
        help='Allowlist file for public wiki'
    )
    parser.add_argument(
        '--lobster-wiki',
        default='https://github.com/the-omics-os/lobster.wiki.git',
        help='Lobster wiki repository URL (HTTPS or SSH format)'
    )
    parser.add_argument(
        '--lobster-local-wiki',
        default='https://github.com/the-omics-os/lobster-local.wiki.git',
        help='Lobster-local wiki repository URL (HTTPS or SSH format)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force push to wiki repositories'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run (show what would be synced)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    source_dir = Path(args.source).resolve()
    allowlist_file = Path(args.allowlist).resolve()
    
    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' does not exist")
        return 1
    
    if not source_dir.is_dir():
        print(f"Error: '{source_dir}' is not a directory")
        return 1
    
    print(f"Starting wiki synchronization")
    print(f"Source: {source_dir}")
    print(f"Allowlist: {allowlist_file}")
    print()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be pushed")
        syncer = WikiSync(source_dir, allowlist_file)
        
        # Show what would be synced to each repo
        print("\n📚 Files in source wiki:")
        all_files = []
        for item in source_dir.iterdir():
            if not item.name.startswith('.'):
                if item.is_file():
                    all_files.append(item.name)
                elif item.is_dir():
                    all_files.append(f"{item.name}/")
        
        for file in sorted(all_files):
            print(f"  - {file}")
        
        # Check allowlist
        allowed_files = syncer.load_allowlist()
        if allowed_files:
            print(f"\n📋 Allowed files for lobster-local.wiki ({len(allowed_files)} patterns):")
            excluded = []
            for file in sorted(all_files):
                if syncer.is_allowed(file.rstrip('/'), allowed_files):
                    print(f"  ✅ {file}")
                else:
                    print(f"  ❌ {file} (excluded)")
                    excluded.append(file)
            
            if excluded:
                print(f"\n⚠️  {len(excluded)} files will be excluded from lobster-local.wiki:")
                for file in excluded:
                    print(f"  - {file}")
        
        print("\n✅ Dry run complete. No changes were made.")
        return 0
    
    # Create syncer
    syncer = WikiSync(source_dir, allowlist_file)
    
    # Sync to full lobster.wiki (no filtering)
    success = syncer.sync_to_repo(
        args.lobster_wiki,
        use_allowlist=False,
        force=args.force
    )
    
    if not success:
        print("❌ Failed to sync to lobster.wiki")
        return 1
    
    # Sync to lobster-local.wiki (with filtering)
    success = syncer.sync_to_repo(
        args.lobster_local_wiki,
        use_allowlist=True,
        force=args.force
    )
    
    if not success:
        print("❌ Failed to sync to lobster-local.wiki")
        return 1
    
    print("\n🎉 Wiki synchronization completed successfully!")
    return 0


if __name__ == '__main__':
    exit(main())
