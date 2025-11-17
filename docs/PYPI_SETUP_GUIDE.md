# PyPI Publishing Setup Guide for lobster-ai

This guide explains how to set up PyPI publishing for the first time and how to release new versions.

## Table of Contents
- [First-Time Setup](#first-time-setup)
- [Creating a Release](#creating-a-release)
- [Testing Locally](#testing-locally)
- [Troubleshooting](#troubleshooting)

---

## First-Time Setup

### 1. Create PyPI Accounts

#### Production PyPI Account
1. Go to https://pypi.org/account/register/
2. Create account with **info@omics-os.com** email
3. Verify email address
4. Enable Two-Factor Authentication (2FA) - **Required for API tokens**

#### TestPyPI Account
1. Go to https://test.pypi.org/account/register/
2. Create account with same email
3. Verify email address
4. Enable 2FA

### 2. Reserve Package Name on PyPI

Before your first release, you need to reserve the `lobster-ai` name on PyPI:

```bash
# Build the package locally first
cd /path/to/lobster
python -m build

# Upload to PyPI manually (first time only)
python -m twine upload dist/*
```

You'll be prompted for PyPI username and password. After this first manual upload, the automated workflow will handle future releases.

### 3. Generate API Tokens

#### For Production PyPI
1. Log in to https://pypi.org
2. Go to Account Settings → API tokens
3. Click "Add API token"
   - **Token name**: `GitHub Actions - lobster-ai`
   - **Scope**: `Project: lobster-ai` (after first upload) or `Entire account` (for first upload)
4. **Copy the token** (starts with `pypi-`)
5. Store it securely - you won't see it again!

#### For TestPyPI
1. Log in to https://test.pypi.org
2. Follow same steps as above
3. Token name: `GitHub Actions - lobster-ai-test`

### 4. Add Tokens to GitHub Secrets

**Private Repository**: `the-omics-os/lobster`

1. Go to repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add both tokens:

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `PYPI_API_TOKEN` | `pypi-...` | Production PyPI token |
| `TEST_PYPI_API_TOKEN` | `pypi-...` | TestPyPI token |

### 5. Set Up GitHub Environments (Optional but Recommended)

For manual approval before production releases:

1. Go to repository Settings → Environments
2. Create three environments:

#### Environment: `testpypi`
- No protection rules needed
- Used for automatic TestPyPI publishing

#### Environment: `pypi-approval`
- Enable "Required reviewers"
- Add yourself as reviewer
- This creates a manual approval gate

#### Environment: `pypi`
- No additional rules needed
- Deployment happens after approval

### 6. Verify GitHub Secrets Are Set

Check that these secrets exist in your private repo:
- ✅ `PYPI_API_TOKEN`
- ✅ `TEST_PYPI_API_TOKEN`
- ✅ `PUBLIC_REPO_DEPLOY_KEY` (for syncing to lobster-local)

---

## Creating a Release

### Option 1: Automatic Release (Recommended)

1. **Ensure your code is ready**:
   ```bash
   cd lobster
   make test
   make lint
   ```

2. **Update version number**:
   ```bash
   # Edit these files:
   lobster/version.py
   pyproject.toml (tool.bumpversion.current_version)
   ```

3. **Commit version bump**:
   ```bash
   git add lobster/version.py pyproject.toml
   git commit -m "chore: bump version to 0.2.0"
   git push origin main
   ```

4. **Create and push git tag**:
   ```bash
   git tag -a v0.2.0 -m "Release version 0.2.0 - Alpha"
   git push origin v0.2.0
   ```

5. **Monitor the workflow**:
   - Go to https://github.com/the-omics-os/lobster/actions
   - Watch the "Publish to PyPI" workflow
   - It will:
     1. ✅ Sync code to lobster-local
     2. ✅ Build package from public code
     3. ✅ Publish to TestPyPI automatically
     4. ⏸️ Wait for your approval
     5. ✅ Publish to production PyPI after approval
     6. ✅ Create GitHub release

6. **Test TestPyPI installation**:
   ```bash
   # Create fresh virtual environment
   python -m venv test-env
   source test-env/bin/activate

   # Install from TestPyPI
   pip install --index-url https://test.pypi.org/simple/ \
               --extra-index-url https://pypi.org/simple \
               lobster-ai==0.2.0

   # Verify it works
   lobster --help
   ```

7. **Approve production release**:
   - If TestPyPI works, go to GitHub Actions
   - Click on the workflow run
   - Review and approve the deployment to `pypi-approval` environment
   - Workflow will automatically publish to production PyPI

### Option 2: Manual Release (Troubleshooting)

If automated workflow fails, you can publish manually:

```bash
# From private repo - build from lobster-local
cd /tmp
git clone git@github.com:the-omics-os/lobster-local.git
cd lobster-local

# Build package
python -m build

# Publish to TestPyPI first
python -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple \
            lobster-ai

# If OK, publish to production
python -m twine upload dist/*
```

---

## Testing Locally

### Test Package Build

```bash
cd lobster

# Install build tools
pip install build twine

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build package
python -m build

# Check package
twine check dist/*

# Inspect contents
tar -tzf dist/*.tar.gz | head -50
unzip -l dist/*.whl | head -50
```

### Test Local Installation

```bash
# Create test environment
python -m venv /tmp/test-lobster
source /tmp/test-lobster/bin/activate

# Install from local build
pip install dist/*.whl

# Test imports
python -c "import lobster; print(lobster.__version__)"
lobster --help

# Verify private code is NOT included
python -c "from lobster.agents import ms_proteomics_expert" 2>&1 | grep "ModuleNotFoundError"
# Should fail with ModuleNotFoundError

# Verify public code IS included
python -c "from lobster.agents import supervisor; print('OK')"
python -c "from lobster.agents import singlecell_expert; print('OK')"
```

### Verify Package Metadata

```bash
# Check package info
python -m twine check dist/*

# View metadata
python setup.py --name
python setup.py --version
python setup.py --author
python setup.py --classifiers
```

---

## Package Details

### What Gets Published

**From lobster-local repository ONLY** (synced via `scripts/sync_to_public.py`):
- ✅ Core engine code
- ✅ Open-source agents (single-cell, bulk RNA-seq, research)
- ✅ Transcriptomics services
- ✅ Data management (DataManagerV2, provenance, queues)
- ✅ Visualization services
- ✅ Cloud client (for upgrades)
- ✅ Configuration files
- ✅ README, LICENSE, documentation

### What Does NOT Get Published (Private)

**Excluded from PyPI package**:
- ❌ Proteomics agents (premium tier)
- ❌ Proteomics services (premium tier)
- ❌ Custom feature agent
- ❌ API/server code
- ❌ CDK infrastructure
- ❌ Tests
- ❌ Private documentation

This is enforced by:
1. `scripts/public_allowlist.txt` - controls what syncs to lobster-local
2. Workflow verification step - scans package for private code before publishing

---

## Troubleshooting

### Error: "Package name already exists"

The `lobster` name is taken on PyPI. We use `lobster-ai` instead:
- **PyPI package name**: `lobster-ai` (install with `pip install lobster-ai`)
- **Import name**: `lobster` (use in code as `import lobster`)

### Error: "Invalid or expired API token"

1. Regenerate tokens on PyPI/TestPyPI
2. Update GitHub secrets
3. Ensure secrets are named exactly:
   - `PYPI_API_TOKEN`
   - `TEST_PYPI_API_TOKEN`

### Error: "This filename has already been used"

PyPI doesn't allow re-uploading the same version. Options:
1. Bump version number (recommended)
2. Use `--skip-existing` flag (for manual uploads)
3. Delete the version on PyPI (only for testing)

### Private code detected in package

If the workflow fails with "Found private code pattern":
1. Check `scripts/public_allowlist.txt`
2. Verify sync completed correctly
3. Check package contents: `unzip -l dist/*.whl`
4. Ensure building from lobster-local, not private repo

### Workflow won't approve production release

Check GitHub environment settings:
1. Settings → Environments → `pypi-approval`
2. Ensure you're listed as required reviewer
3. Check Actions tab for approval button

---

## Release Checklist

Before creating a release:

- [ ] All tests pass: `make test`
- [ ] Code is formatted: `make format`
- [ ] Linting passes: `make lint`
- [ ] Version bumped in `lobster/version.py` and `pyproject.toml`
- [ ] CHANGELOG.md updated (if exists)
- [ ] Committed to main branch
- [ ] Tag created: `git tag -a v0.2.0 -m "Release 0.2.0"`
- [ ] Tag pushed: `git push origin v0.2.0`
- [ ] Monitor GitHub Actions workflow
- [ ] Test installation from TestPyPI
- [ ] Approve production release
- [ ] Verify installation from PyPI
- [ ] Announce release

---

## Security Notes

1. **Never commit API tokens** to git
2. **Use scoped tokens** (project-specific, not entire account)
3. **Enable 2FA** on PyPI accounts (required for tokens)
4. **Rotate tokens** if compromised
5. **Review packages** on TestPyPI before production
6. **Verify no secrets** in published package:
   ```bash
   unzip -l dist/*.whl | grep -i "secret\|key\|password"
   ```

---

## Links

- **Production PyPI**: https://pypi.org/project/lobster-ai/
- **TestPyPI**: https://test.pypi.org/project/lobster-ai/
- **Public Repo**: https://github.com/the-omics-os/lobster-local
- **PyPI Guide**: https://packaging.python.org/tutorials/packaging-projects/
- **Twine Docs**: https://twine.readthedocs.io/

---

## Support

If you encounter issues:
1. Check GitHub Actions logs
2. Review this guide's troubleshooting section
3. Test locally first with `python -m build`
4. Verify package contents before publishing
5. Contact team lead if still stuck

**Remember**: Always test on TestPyPI first! 🧪
