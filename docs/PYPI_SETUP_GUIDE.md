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

Before your first release, you need to reserve the `lobster-ai` name on PyPI.

**The first upload to PyPI automatically reserves the package name.** You have two options:

#### Option 1: Let the workflow handle it (Recommended)
Just create the tag and push. The workflow will attempt to publish. If the name isn't reserved yet, the first successful publish claims it.

#### Option 2: Manual first upload
```bash
# Build from public repository (CRITICAL - not private repo!)
cd /tmp
git clone https://github.com/the-omics-os/lobster.git
cd lobster
python -m build

# Upload to PyPI manually (first time only)
python -m twine upload dist/*
```

You'll be prompted for:
- **Username**: `__token__` (exactly this, with underscores)
- **Password**: Your PyPI API token starting with `pypi-...`

After this first upload, the package name is yours and the automated workflow will handle future releases.

---

## Authentication Methods

PyPI supports two authentication methods for automated publishing. Choose based on your security requirements and setup preferences.

### Option A: Trusted Publishing (Recommended - Secure & Modern) ⭐

**Why Trusted Publishing?**
- ✅ No long-lived secrets in GitHub (OIDC-based)
- ✅ PyPI mints short-lived tokens automatically
- ✅ Official PyPA recommendation for CI/CD
- ✅ Zero maintenance (no token rotation needed)
- ✅ More secure (tokens expire in minutes)

**Setup Steps:**

1. **Configure on PyPI** (one-time):
   - Log in to https://pypi.org (or https://test.pypi.org)
   - Navigate to your project: `lobster-ai`
   - Go to **Manage** → **Publishing**
   - Click **Add a new publisher**
   - Fill in GitHub details:
     - **Owner**: `the-omics-os`
     - **Repository name**: `lobster`
     - **Workflow name**: `publish-pypi.yml`
     - **Environment name**: Leave blank (or `pypi` if using GitHub Environments)
   - Click **Add**

2. **Repeat for TestPyPI**:
   - Log in to https://test.pypi.org
   - Same steps, but use environment name `testpypi`

3. **Verify workflow configuration** (already done):
   ```yaml
   permissions:
     id-token: write  # Required for OIDC
     contents: read
   ```

4. **No GitHub secrets needed!** The workflow will authenticate automatically.

**How it works:**
- GitHub Actions requests short-lived token from PyPI via OIDC
- PyPI verifies workflow identity (repo, workflow name, environment)
- PyPI mints token valid for ~15 minutes
- Workflow uses token to publish
- Token expires automatically

**Official Docs:**
- https://docs.pypi.org/trusted-publishers/
- https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect

---

### Option B: API Tokens (Legacy/Fallback Method)

**When to use:**
- You can't configure Trusted Publishers (e.g., no project access yet)
- You need to test locally with `twine upload`
- Temporary setup before migrating to Trusted Publishing

**Important:** As of January 1, 2024, **PyPI requires 2FA (Two-Factor Authentication) for all uploads**, whether using tokens or interactive login.

#### Generate API Tokens

### 3. Generate API Tokens (Option B Only)

#### For Production PyPI
1. Log in to https://pypi.org
2. Ensure 2FA is enabled (required since Jan 1, 2024)
3. Go to Account Settings → API tokens
4. Click "Add API token"
   - **Token name**: `GitHub Actions - lobster-ai`
   - **Scope**: `Project: lobster-ai` (after first upload) or `Entire account` (for first upload)
5. **Copy the token** (starts with `pypi-`)
6. Store it securely - you won't see it again!

#### For TestPyPI
1. Log in to https://test.pypi.org
2. Ensure 2FA is enabled
3. Follow same steps as above
4. Token name: `GitHub Actions - lobster-ai-test`

### 4. Add Tokens to GitHub Secrets (Option B Only)

**Private Repository**: `the-omics-os/lobster`

1. Go to repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add both tokens:

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `PYPI_API_TOKEN` | `pypi-...` | Production PyPI token |
| `TEST_PYPI_API_TOKEN` | `pypi-...` | TestPyPI token |

**Note:** The workflow supports both authentication methods:
- If Trusted Publishing is configured on PyPI, tokens are ignored (OIDC used)
- If Trusted Publishing is NOT configured, workflow falls back to tokens
- This provides zero-downtime migration path

#### Local Testing with .pypirc (Optional)

For manual `twine upload` testing, create `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-production-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-test-token-here
```

**Security:** Ensure this file has restricted permissions:
```bash
chmod 600 ~/.pypirc
```

Then upload with:
```bash
twine upload --repository testpypi dist/*  # TestPyPI
twine upload dist/*                        # Production PyPI
```

---

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

### 6. Verify Setup

**For Trusted Publishing (Option A):**
- ✅ Trusted Publisher configured on PyPI for `lobster-ai`
- ✅ Trusted Publisher configured on TestPyPI for `lobster-ai`
- ✅ Workflow has `id-token: write` permission (already set)

**For API Tokens (Option B):**
- ✅ `PYPI_API_TOKEN` secret exists in GitHub
- ✅ `TEST_PYPI_API_TOKEN` secret exists in GitHub

**Workflow is ready once secrets are configured.**

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
     1. ✅ Build package
     2. ✅ Publish to TestPyPI automatically
     3. ⏸️ Wait for your approval
     4. ✅ Publish to production PyPI after approval
     5. ✅ Create GitHub release

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
# Build from repository
cd /tmp
git clone git@github.com:the-omics-os/lobster.git
cd lobster

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

## Post-Release: User Experience

### For Users Installing via pip

When users run `pip install lobster-ai`, they will need to configure environment variables manually.

**What's included in the package:**
- ✅ `.env.example` template file
- ✅ README.md with configuration instructions

**User workflow:**
```bash
# 1. Install
pip install lobster-ai

# 2. Create config file (two options)

# Option A: Download template
curl -O https://raw.githubusercontent.com/the-omics-os/lobster/main/.env.example
mv .env.example .env

# Option B: Create manually
cat > .env << 'EOF'
# Required: Choose ONE LLM provider
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
# OR
AWS_BEDROCK_ACCESS_KEY=your-access-key
AWS_BEDROCK_SECRET_ACCESS_KEY=your-secret-key

# Optional: Enhanced literature search
NCBI_API_KEY=your-ncbi-key
NCBI_EMAIL=your.email@example.com

# Optional: Performance tuning
LOBSTER_PROFILE=production
LOBSTER_MAX_FILE_SIZE_MB=500
EOF

# 3. Edit config
nano .env

# 4. Run
lobster chat
```

**Error handling:**
If users forget to configure, they'll see:
```
❌ No LLM provider configured

Lobster AI requires API credentials to function.

Quick Setup:
1. Create a .env file in your current directory
2. Add ONE of the following:

   Option A - Claude API (Recommended for testing):
   ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

   Option B - AWS Bedrock (Recommended for production):
   AWS_BEDROCK_ACCESS_KEY=your-access-key
   AWS_BEDROCK_SECRET_ACCESS_KEY=your-secret-key

Get API Keys:
  • Claude API: https://console.anthropic.com/
  • AWS Bedrock: https://aws.amazon.com/bedrock/

For detailed setup instructions, see:
  https://github.com/the-omics-os/lobster/wiki/03-configuration

Tip: If you installed via pip, make sure to create a .env file in your current directory.
Tip: See README for installation instructions: https://github.com/the-omics-os/lobster
```

**Design rationale:**
- Manual configuration keeps package simple and maintainable
- No interactive CLI wizard reduces complexity
- Users can version-control their .env files
- Follows standard Python package patterns (e.g., Flask, Django)
- Clear error messages guide users to correct configuration

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
# Check package for PyPI compliance
python -m twine check dist/*

# View metadata (modern approach - no setup.py commands)
pip show lobster-ai                           # After installation
unzip -p dist/*.whl */METADATA | head -50    # Inspect wheel metadata
cat pyproject.toml | grep -E "^(name|version|description)"  # Source of truth
```

**Deprecated commands (do NOT use):**
```bash
# ❌ LEGACY - these invoke setup.py and are no longer recommended
python setup.py --name
python setup.py --version
python setup.py --author
```

---

## Package Details

### What Gets Published

**Published to PyPI**:
- ✅ Core engine code (all modular packages)
- ✅ All agent packages (lobster-transcriptomics, lobster-ml, etc.)
- ✅ Services (analysis, quality, visualization, ML, etc.)
- ✅ Data management (DataManagerV2, provenance, queues)
- ✅ CLI and configuration
- ✅ README, LICENSE, documentation

### Premium Features (Runtime Gated)

**Published to PyPI but gated at runtime**:
- 🔒 Premium agents (proteomics_expert, metadata_assistant, etc.)
- 🔒 Premium services (accessed via ComponentRegistry)
- 🔒 Enterprise features

**Architecture:**
```
lobster-ai + lobster-{domain} packages
         ↓
   Published to PyPI (PUBLIC)
         ↓
subscription_tiers.py defines tiers
         ↓
Runtime checks via ComponentRegistry
         ↓
TierRestrictedError for premium features without valid license
```

**Key change (Kraken architecture):** Premium code is public but requires license activation for use. No more separate public/private repositories.

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
1. Check `lobster/config/subscription_tiers.py` - defines tier requirements
2. Verify ComponentRegistry is checking tiers correctly
3. Check package contents: `unzip -l dist/*.whl`
4. Ensure pyproject.toml excludes sensitive files (kevin_notes/, etc.)

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

### Project Links
- **Production PyPI**: https://pypi.org/project/lobster-ai/
- **TestPyPI**: https://test.pypi.org/project/lobster-ai/
- **Public Repo**: https://github.com/the-omics-os/lobster

### Official PyPA Resources
- **Trusted Publishing Guide**: https://docs.pypi.org/trusted-publishers/
- **GitHub Actions Publishing**: https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/
- **Packaging Tutorial**: https://packaging.python.org/tutorials/packaging-projects/
- **PyPI Classifiers List**: https://pypi.org/classifiers/
- **Core Metadata Spec**: https://packaging.python.org/specifications/core-metadata/
- **Twine Documentation**: https://twine.readthedocs.io/

### Recommended Tools
- **Build**: `python -m build` (PEP 517 compliant)
- **Upload**: `python -m twine upload` (secure upload)
- **Check**: `python -m twine check` (validate before upload)

---

## Support

If you encounter issues:
1. Check GitHub Actions logs
2. Review this guide's troubleshooting section
3. Test locally first with `python -m build`
4. Verify package contents before publishing
5. Contact team lead if still stuck

**Remember**: Always test on TestPyPI first! 🧪
