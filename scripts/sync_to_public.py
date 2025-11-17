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
                 public_repo_url: str, branch: str = "main"):
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
        self.skipped_files = []  # Track skipped files for reporting
        self.copied_files = []   # Track copied files for reporting
        # Get git user info from environment
        self.git_user_name = os.environ.get('GIT_USER_NAME', 'omics-os-bot')
        self.git_user_email = os.environ.get('GIT_USER_EMAIL', 'bot@omics-os.com')

    @staticmethod
    def get_ssh_key_path() -> str:
        """
        Intelligently detect which SSH key to use.

        Priority:
        1. Environment variable SSH_KEY_PATH (for custom configs)
        2. Local deploy key (for local development)
        3. Default id_ed25519 (for GitHub Actions)

        Returns:
            str: Path to SSH private key
        """
        # Check environment variable first
        if 'SSH_KEY_PATH' in os.environ:
            key_path = os.path.expanduser(os.environ['SSH_KEY_PATH'])
            if os.path.exists(key_path):
                print(f"Using SSH key from SSH_KEY_PATH: {key_path}")
                return key_path

        # Check for local deploy key (development)
        local_deploy_key = os.path.expanduser('~/.ssh/omics-os-deploy-keys/lobster-local-deploy-key')
        if os.path.exists(local_deploy_key):
            print(f"Using local deploy key: {local_deploy_key}")
            return local_deploy_key

        # Default to id_ed25519 (GitHub Actions)
        default_key = os.path.expanduser('~/.ssh/id_ed25519')
        print(f"Using default SSH key: {default_key}")
        return default_key

    @staticmethod
    def is_relevant_file(filepath: Path) -> bool:
        """
        Check if a file is relevant (not cache/temporary).

        Returns False for:
        - Cache files (__pycache__, *.pyc, *.pyo, *.pyd)
        - Build artifacts (*.egg-info, dist/, build/)
        - IDE files (.vscode, .idea, *.swp)
        - OS files (.DS_Store, Thumbs.db)
        - Test artifacts (.pytest_cache, .coverage, htmlcov/)
        - Temporary files (*.tmp, *.bak, *~)
        """
        path_str = str(filepath)
        name = filepath.name

        # Cache and compiled files
        if any(x in path_str for x in ['__pycache__', '.pyc', '.pyo', '.pyd', '.egg-info']):
            return False

        # Build directories
        if any(x in path_str for x in ['/dist/', '/build/', '/.eggs/']):
            return False

        # IDE and editor files
        if any(x in path_str for x in ['.vscode', '.idea', '.swp', '.swo']):
            return False

        # OS files
        if name in ['.DS_Store', 'Thumbs.db', 'desktop.ini']:
            return False

        # Test artifacts
        if any(x in path_str for x in ['.pytest_cache', '.coverage', '/htmlcov/', '.tox']):
            return False

        # Temporary files
        if name.endswith(('.tmp', '.bak', '.log')) or name.endswith('~'):
            return False

        # Node modules, virtual environments
        if any(x in path_str for x in ['/node_modules/', '/.venv/', '/venv/', '/.env/']):
            return False

        return True

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
                # Skip hidden files except for specific allowed ones
                if file.startswith('.') and file not in ['.env.example', '.gitignore', '.pre-commit-config.yaml']:
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
                    # Track copied files for reporting
                    self.copied_files.append(str(rel_path))
                    print(f"  ✅ {rel_path}")
                else:
                    self.stats["files_skipped"] += 1
                    # Track relevant skipped files (exclude cache/temp)
                    if self.is_relevant_file(filepath):
                        rel_path = filepath.relative_to(self.source_dir)
                        self.skipped_files.append(str(rel_path))
    
    def report_skipped_files(self):
        """Display categorized list of skipped relevant files."""
        if not self.skipped_files:
            return

        print("\n" + "="*60)
        print("📋 EXCLUDED FILES (Relevant files not synced to public)")
        print("="*60)

        # Categorize skipped files
        categories = {
            'Premium Agents': [],
            'Premium Services': [],
            'Premium Schemas/Adapters': [],
            'API/Server Code': [],
            'Infrastructure': [],
            'Tests': [],
            'Internal Documentation': [],
            'Other': []
        }

        for filepath in sorted(self.skipped_files):
            # Categorize by path patterns
            if 'agents/' in filepath and any(x in filepath for x in ['proteomics', 'custom_feature']):
                categories['Premium Agents'].append(filepath)
            elif 'tools/' in filepath and 'proteomics' in filepath:
                categories['Premium Services'].append(filepath)
            elif any(x in filepath for x in ['schemas/proteomics', 'adapters/proteomics']):
                categories['Premium Schemas/Adapters'].append(filepath)
            elif any(x in filepath for x in ['api/', 'server/', 'websocket', 'streamlit']):
                categories['API/Server Code'].append(filepath)
            elif any(x in filepath for x in ['cdk/', 'infrastructure/', 'terraform/']):
                categories['Infrastructure'].append(filepath)
            elif 'tests/' in filepath or filepath.startswith('test_'):
                categories['Tests'].append(filepath)
            elif any(x in filepath for x in ['ALPHA', 'template', 'private', 'secret']):
                categories['Internal Documentation'].append(filepath)
            else:
                categories['Other'].append(filepath)

        # Display non-empty categories
        for category, files in categories.items():
            if files:
                print(f"\n📦 {category} ({len(files)} files):")
                for f in files[:10]:  # Show first 10
                    print(f"  ❌ {f}")
                if len(files) > 10:
                    print(f"  ... and {len(files) - 10} more")

        print(f"\n📊 Summary:")
        print(f"  • Total relevant files excluded: {len(self.skipped_files)}")
        print(f"  • Total files skipped (including cache): {self.stats['files_skipped']}")
        print("="*60)

    def report_copied_files(self):
        """Display categorized list of copied files (synced to public)."""
        if not self.copied_files:
            return

        print("\n" + "="*60)
        print("📦 INCLUDED FILES (Synced to public repository)")
        print("="*60)

        # Categorize copied files using same logic
        categories = {
            'Core Orchestration': [],
            'Research & Content': [],
            'Open-Core Agents': [],
            'Transcriptomics Services': [],
            'Data Management': [],
            'Visualization & ML': [],
            'Configuration': [],
            'Infrastructure': [],
            'Documentation': [],
            'Other': []
        }

        for filepath in sorted(self.copied_files):
            # Categorize by path patterns
            if 'agents/' in filepath:
                if any(x in filepath for x in ['supervisor', 'data_expert', 'metadata_assistant']):
                    categories['Core Orchestration'].append(filepath)
                elif any(x in filepath for x in ['research', 'content']):
                    categories['Research & Content'].append(filepath)
                elif any(x in filepath for x in ['singlecell', 'bulk_rnaseq', 'visualization', 'machine_learning', 'protein_structure']):
                    categories['Open-Core Agents'].append(filepath)
                else:
                    categories['Other'].append(filepath)
            elif 'tools/' in filepath:
                if any(x in filepath for x in ['providers/', 'content_access', 'docling', 'geo_']):
                    categories['Research & Content'].append(filepath)
                elif any(x in filepath for x in ['preprocessing', 'quality', 'clustering', 'differential', 'pseudobulk', 'bulk_rnaseq', 'enhanced_singlecell', 'concatenation']):
                    categories['Transcriptomics Services'].append(filepath)
                elif any(x in filepath for x in ['visualization', 'scvi', 'ml_', 'pymol', 'structure']):
                    categories['Visualization & ML'].append(filepath)
                else:
                    categories['Other'].append(filepath)
            elif 'core/' in filepath:
                if any(x in filepath for x in ['data_manager', 'provenance', 'download_queue', 'workspace', 'notebook']):
                    categories['Data Management'].append(filepath)
                elif any(x in filepath for x in ['schemas/transcriptomics', 'adapters/transcriptomics', 'adapters/pseudobulk', 'schemas/protein_structure', 'adapters/protein_structure']):
                    categories['Data Management'].append(filepath)
                else:
                    categories['Other'].append(filepath)
            elif 'config/' in filepath:
                categories['Configuration'].append(filepath)
            elif any(x in filepath for x in ['README', '.md', 'LICENSE', 'CONTRIBUTING']):
                categories['Documentation'].append(filepath)
            elif any(x in filepath for x in ['Makefile', 'setup.py', 'pyproject.toml', '.gitignore', '.pre-commit']):
                categories['Infrastructure'].append(filepath)
            else:
                categories['Other'].append(filepath)

        # Display non-empty categories
        for category, files in categories.items():
            if files:
                print(f"\n📂 {category} ({len(files)} files):")
                for f in files[:10]:  # Show first 10
                    print(f"  ✅ {f}")
                if len(files) > 10:
                    print(f"  ... and {len(files) - 10} more")

        print(f"\n📊 Summary:")
        print(f"  • Total files synced to public: {len(self.copied_files)}")
        print(f"  • Total size: {self.stats['total_size'] / 1024 / 1024:.2f} MB")
        print("="*60)

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
    
    def create_public_commit(self) -> bool:
        """Create a clean commit in the public repository.

        Returns:
            bool: True if commit was created, False if no changes to commit
        """
        os.chdir(self.temp_dir)

        print("\n" + "="*60)
        print("DEBUG: Creating public commit")
        print("="*60)

        # Configure git user
        print("\nDEBUG: Configuring git user...")
        subprocess.run(['git', 'config', 'user.name', self.git_user_name], check=True)
        subprocess.run(['git', 'config', 'user.email', self.git_user_email], check=True)

        # Add all files
        print("DEBUG: Adding all files to staging...")
        subprocess.run(['git', 'add', '-A'], check=True)

        # Check if there are any changes to commit
        print("\nDEBUG: Checking for changes...")
        status_result = subprocess.run(['git', 'status', '--porcelain'],
                                      capture_output=True, text=True, check=True)

        if not status_result.stdout.strip():
            print("⚠️  No changes detected - all files are identical to public repository")
            return False

        # Show what will be committed
        print("\nDEBUG: Files to be committed:")
        subprocess.run(['git', 'status', '--short'])

        # Create commit with metadata
        commit_msg = f"Sync from private repository\n\nTimestamp: {datetime.now().isoformat()}\n"
        commit_msg += f"Files: {self.stats['files_copied']} copied, {self.stats['files_skipped']} skipped\n"
        commit_msg += f"Size: {self.stats['total_size'] / 1024 / 1024:.2f} MB"

        print("\nDEBUG: Creating commit...")
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True)

        # Show the commit that was created
        print("\nDEBUG: Commit created:")
        subprocess.run(['git', 'log', '--oneline', '-n', '1'])

        return True
    
    def push_to_public(self, force: bool = False):
        """Push changes to public repository."""
        os.chdir(self.temp_dir)

        print("\n" + "="*60)
        print("DEBUG: Push to public repository")
        print(f"  Force parameter: {force} (type: {type(force)})")
        print("="*60)

        # Setup SSH environment - explicitly use the SSH key
        env = os.environ.copy()
        ssh_key = self.get_ssh_key_path()
        env["GIT_SSH_COMMAND"] = f"ssh -i {ssh_key} -o StrictHostKeyChecking=no"

        # Debug: Show current git state
        print("\nDEBUG: Current git status:")
        subprocess.run(['git', 'status', '--short'], env=env)

        print("\nDEBUG: Current branch:")
        subprocess.run(['git', 'branch', '--show-current'], env=env)

        print("\nDEBUG: Remote configuration:")
        subprocess.run(['git', 'remote', '-v'], env=env)

        print("\nDEBUG: Recent commits:")
        subprocess.run(['git', 'log', '--oneline', '-n', '3'], env=env)

        # Build push command - simplified since branch is already tracking remote
        push_cmd = ['git', 'push']
        if force:
            push_cmd.append('--force')
            print(f"\nDEBUG: Force flag is True, adding --force")
        else:
            print(f"\nDEBUG: Force flag is False, using normal push")

        print(f"\nDEBUG: Final push command: {push_cmd}")
        print(f"       Command string: {' '.join(push_cmd)}")

        # Execute push with detailed error capture
        print("\nExecuting push...")
        try:
            result = subprocess.run(push_cmd, capture_output=True, text=True,
                                  check=False, env=env)

            if result.returncode == 0:
                print("✅ Push successful!")
                if result.stdout:
                    print(f"Stdout: {result.stdout}")
            else:
                print(f"❌ Push failed with return code: {result.returncode}")
                print(f"Stderr: {result.stderr}")
                print(f"Stdout: {result.stdout}")

                # Provide helpful debugging info
                print("\nDEBUG: Checking divergence...")
                subprocess.run(['git', 'status', '-sb'], env=env)

                print("\nDEBUG: Comparing with origin...")
                subprocess.run(['git', 'log', '--oneline', f'origin/{self.branch}..HEAD'],
                             env=env)

                # Re-raise the error for proper handling
                result.check_returncode()

        except subprocess.CalledProcessError as e:
            print(f"\n❌ Git push failed: {e}")
            print("\nTroubleshooting suggestions:")
            print("1. Check if SSH key is properly configured")
            print("2. Verify repository permissions")
            print("3. Check if branch protection rules are blocking the push")
            print("4. Try running with --force flag if this is intentional replacement")
            raise
    
    def sync(self, force: bool = False, dry_run: bool = False):
        """Execute full sync process."""
        print(f"Starting sync to public repository")
        print(f"Source: {self.source_dir}")
        print(f"Target: {self.public_repo_url}")
        print(f"Branch: {self.branch}")
        print(f"DEBUG: Force push enabled: {force}")
        print(f"DEBUG: Dry run: {dry_run}")
        print()

        # Setup SSH environment for git operations
        env = os.environ.copy()
        ssh_key = self.get_ssh_key_path()
        env["GIT_SSH_COMMAND"] = f"ssh -i {ssh_key} -o StrictHostKeyChecking=no"

        # Create temp directory and clone existing repo to preserve history
        temp_dir = tempfile.mkdtemp()
        try:
            self.temp_dir = Path(temp_dir)

            print(f"Cloning existing public repository to preserve git history...")
            clone_result = subprocess.run(
                ['git', 'clone', '--branch', self.branch, self.public_repo_url, str(temp_dir)],
                capture_output=True, text=True, env=env
            )

            if clone_result.returncode != 0:
                # Handle first-time sync or empty repo
                print(f"  Clone failed (may be first sync): {clone_result.stderr}")
                print(f"  Initializing new repository...")
                subprocess.run(['git', 'init'], cwd=temp_dir, check=True)
                subprocess.run(['git', 'remote', 'add', 'origin', self.public_repo_url],
                             cwd=temp_dir, check=True)
                subprocess.run(['git', 'checkout', '-b', self.branch], cwd=temp_dir, check=True)
            else:
                print(f"  ✅ Cloned existing repository with history")

            # Clear all files except .git directory
            print("Clearing existing files (preserving .git)...")
            for item in self.temp_dir.iterdir():
                if item.name != '.git':
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()

            # Load patterns
            patterns = self.load_allowlist()
            print(f"Loaded {len(patterns)} allowlist patterns")

            # Copy files
            self.copy_allowed_files(patterns)
            print(f"\nCopied {self.stats['files_copied']} files")
            print(f"Skipped {self.stats['files_skipped']} files")
            print(f"Total size: {self.stats['total_size'] / 1024 / 1024:.2f} MB")

            # Report included files (synced to public)
            self.report_copied_files()

            # Report excluded relevant files
            self.report_skipped_files()

            # Security scan
            print("\nPerforming security scan...")
            self.scan_for_secrets()

            if dry_run:
                print("\n🔍 Dry run complete. No changes pushed.")
                return

            # Create commit
            print("\nCreating public commit...")
            has_changes = self.create_public_commit()

            if not has_changes:
                print("\n✅ Sync completed - repository is already up to date!")
                return

            # Push
            print("\nPushing to public repository...")
            print(f"DEBUG: Calling push_to_public with force={force}")
            self.push_to_public(force)

            print("\n✅ Sync completed successfully!")

        finally:
            # Clean up temp directory
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(temp_dir)


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

    # Debug: Print parsed arguments
    print("="*60)
    print("DEBUG: Parsed arguments:")
    print(f"  Source: {args.source}")
    print(f"  Allowlist: {args.allowlist}")
    print(f"  Repo: {args.repo}")
    print(f"  Branch: {args.branch}")
    print(f"  Force: {args.force} (type: {type(args.force)})")
    print(f"  Dry-run: {args.dry_run}")
    print("="*60)
    print()

    syncer = PublicRepoSync(
        source_dir=Path(args.source).resolve(),
        allowlist_file=Path(args.allowlist).resolve(),
        public_repo_url=args.repo,
        branch=args.branch
    )
    
    syncer.sync(force=args.force, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
