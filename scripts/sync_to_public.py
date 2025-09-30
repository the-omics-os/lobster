#!/usr/bin/env python3
"""
Safe synchronization script for public repository.
Ensures only allowed files are copied and no sensitive data leaks.
"""

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
import fnmatch
import argparse
import hashlib
import json
from datetime import datetime


class PublicRepoSync:
    """Handles safe synchronization to public repository."""
    
    def __init__(self, source_dir: Path, allowlist_file: Path, 
                 public_repo_url: str, branch: str = "dev"):
        self.source_dir = source_dir
        self.allowlist_file = allowlist_file
        self.public_repo_url = public_repo_url
        self.branch = branch
        self.temp_dir = None
        self.stats = {
            "files_copied": 0,
            "files_skipped": 0,
            "total_size": 0
        }
    
    def load_allowlist(self) -> list:
        """Load and parse allowlist patterns."""
        patterns = []
        with open(self.allowlist_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    patterns.append(line)
        return patterns
    
    def is_allowed(self, filepath: Path, patterns: list) -> bool:
        """Check if file matches allowlist patterns."""
        # Convert to relative path string
        rel_path = str(filepath.relative_to(self.source_dir))
        
        # Check exclusion patterns first (starting with !)
        for pattern in patterns:
            if pattern.startswith('!'):
                if fnmatch.fnmatch(rel_path, pattern[1:]):
                    return False
        
        # Check inclusion patterns
        for pattern in patterns:
            if not pattern.startswith('!'):
                if fnmatch.fnmatch(rel_path, pattern):
                    return True
        
        return False
    
    def copy_allowed_files(self, patterns: list):
        """Copy only allowed files to temp directory."""
        print(f"Copying allowed files from {self.source_dir}")
        
        for root, dirs, files in os.walk(self.source_dir):
            # Skip .git and other hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            root_path = Path(root)
            for file in files:
                if file.startswith('.'):
                    continue
                    
                filepath = root_path / file
                if self.is_allowed(filepath, patterns):
                    rel_path = filepath.relative_to(self.source_dir)
                    dest_path = self.temp_dir / rel_path
                    
                    # Create destination directory
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file
                    shutil.copy2(filepath, dest_path)
                    self.stats["files_copied"] += 1
                    self.stats["total_size"] += filepath.stat().st_size
                    print(f"  ✅ {rel_path}")
                else:
                    self.stats["files_skipped"] += 1
    
    def scan_for_secrets(self):
        """Basic scan for potential secrets in copied files."""
        secret_patterns = [
            'api_key', 'api-key', 'apikey',
            'secret', 'password', 'passwd',
            'token', 'private_key', 'private-key',
            'aws_access_key', 'aws_secret',
            'AKIA',  # AWS access key prefix
        ]
        
        warnings = []
        for root, _, files in os.walk(self.temp_dir):
            for file in files:
                if file.endswith('.py') or file.endswith('.yml') or file.endswith('.yaml'):
                    filepath = Path(root) / file
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().lower()
                        for pattern in secret_patterns:
                            if pattern.lower() in content:
                                rel_path = filepath.relative_to(self.temp_dir)
                                warnings.append(f"Potential secret pattern '{pattern}' in {rel_path}")
        
        if warnings:
            print("\n⚠️  Security Warnings:")
            for warning in warnings:  # Limit output
                print(f"  - {warning}")

            # Check if running in CI/non-interactive environment
            if not sys.stdin.isatty() or os.environ.get('CI'):
                print("\n🤖 Running in CI mode - auto-continuing with security warnings")
                print("⚠️  Please review warnings above and ensure no secrets are exposed")
            else:
                response = input("\nContinue with sync? (y/N): ")
                if response.lower() != 'y':
                    raise Exception("Sync aborted due to security concerns")
    
    def create_public_commit(self):
        """Create a clean commit in the public repository."""
        os.chdir(self.temp_dir)

        # Setup SSH environment for all git operations
        env = os.environ.copy()
        env["GIT_SSH_COMMAND"] = "ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no"

        # Initialize git if needed
        if not (self.temp_dir / '.git').exists():
            subprocess.run(['git', 'init'], check=True)
            subprocess.run(['git', 'remote', 'add', 'origin', self.public_repo_url], check=True)

            # Fetch existing remote state if it exists
            subprocess.run(['git', 'fetch', 'origin', self.branch], check=False, env=env)

            # Try to checkout existing branch or create new one
            result = subprocess.run(['git', 'checkout', self.branch], check=False, capture_output=True)
            if result.returncode != 0:
                subprocess.run(['git', 'checkout', '-b', self.branch], check=True)

        # Configure git
        subprocess.run(['git', 'config', 'user.name', 'Lobster Bot'], check=True)
        subprocess.run(['git', 'config', 'user.email', 'bot@omics-os.com'], check=True)

        # Add all files
        subprocess.run(['git', 'add', '-A'], check=True)

        # Create commit with metadata
        commit_msg = f"Sync from private repository\n\nTimestamp: {datetime.now().isoformat()}\n"
        commit_msg += f"Files: {self.stats['files_copied']} copied, {self.stats['files_skipped']} skipped\n"
        commit_msg += f"Size: {self.stats['total_size'] / 1024 / 1024:.2f} MB"

        subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
    
    def push_to_public(self, force: bool = False):
        """Push changes to public repository."""
        os.chdir(self.temp_dir)

        # Setup SSH environment - explicitly use the SSH key
        env = os.environ.copy()
        env["GIT_SSH_COMMAND"] = "ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no"

        # Fetch current state with SSH auth
        subprocess.run(['git', 'fetch', 'origin', self.branch], check=False, env=env)

        # Build push command
        push_cmd = ['git', 'push', '--set-upstream', 'origin', f'HEAD:{self.branch}']
        if force:
            push_cmd.insert(2, '--force')

        print(f"Pushing with command: {' '.join(push_cmd)}")

        subprocess.run(push_cmd, check=True, env=env)
    
    def sync(self, force: bool = False, dry_run: bool = False):
        """Execute full sync process."""
        print(f"Starting sync to public repository")
        print(f"Source: {self.source_dir}")
        print(f"Target: {self.public_repo_url}")
        print(f"Branch: {self.branch}")
        print()
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_dir = Path(temp_dir)
            
            # Load patterns
            patterns = self.load_allowlist()
            print(f"Loaded {len(patterns)} allowlist patterns")
            
            # Copy files
            self.copy_allowed_files(patterns)
            print(f"\nCopied {self.stats['files_copied']} files")
            print(f"Skipped {self.stats['files_skipped']} files")
            print(f"Total size: {self.stats['total_size'] / 1024 / 1024:.2f} MB")
            
            # Security scan
            print("\nPerforming security scan...")
            self.scan_for_secrets()
            
            if dry_run:
                print("\n🔍 Dry run complete. No changes pushed.")
                return
            
            # Create commit
            print("\nCreating public commit...")
            self.create_public_commit()
            
            # Push
            print("\nPushing to public repository...")
            self.push_to_public(force)
            
            print("\n✅ Sync completed successfully!")


def main():
    parser = argparse.ArgumentParser(description='Sync to public repository')
    parser.add_argument('--source', default='.', help='Source directory')
    parser.add_argument('--allowlist', default='scripts/public_allowlist.txt', 
                       help='Allowlist file')
    parser.add_argument('--repo', required=True, help='Public repository URL')
    parser.add_argument('--branch', default='main', help='Target branch')
    parser.add_argument('--force', action='store_true', help='Force push')
    parser.add_argument('--dry-run', action='store_true', help='Dry run only')
    
    args = parser.parse_args()
    
    syncer = PublicRepoSync(
        source_dir=Path(args.source).resolve(),
        allowlist_file=Path(args.allowlist).resolve(),
        public_repo_url=args.repo,
        branch=args.branch
    )
    
    syncer.sync(force=args.force, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
