# PyPI Release Setup Summary

## ✅ Completed Setup

All PyPI publishing infrastructure is now in place for **lobster-ai v0.2.0**.

### Files Created/Modified

1. **`pyproject.toml`** ✅
   - Package name: `lobster-ai` (import name remains `lobster`)
   - Version: `0.2.0`
   - Consistent with setup.py

2. **`lobster/version.py`** ✅
   - Version: `0.2.0`

3. **`MANIFEST.in`** ✅ NEW
   - Controls which files are included in distribution
   - Excludes tests, docs, development files

4. **`.github/workflows/publish-pypi.yml`** ✅ NEW
   - Complete automated publishing workflow
   - 7 stages: sync → build → testpypi → approve → pypi → release → summary

5. **`docs/PYPI_SETUP_GUIDE.md`** ✅ NEW
   - Complete guide for first-time PyPI setup
   - Token generation instructions
   - Testing procedures
   - Troubleshooting tips

---

## 🔒 Security Verification

### Private Repo Build (`lobster/`)
Building from private repository **INCLUDES** all premium code (as expected):
- ❌ ms_proteomics_expert.py (42 KB)
- ❌ affinity_proteomics_expert.py (42 KB)
- ❌ custom_feature_agent.py (99 KB)
- ❌ 6 proteomics services (~250 KB total)
- ❌ unified_agent_creation_template.md

### Public Repo Build (`lobster-local/`)
Building from public repository **EXCLUDES** all premium code (verified):
- ✅ NO proteomics agents
- ✅ NO custom feature agent
- ✅ NO proteomics services
- ✅ Only open-source agents included:
  - supervisor.py
  - research_agent.py
  - singlecell_expert.py
  - bulk_rnaseq_expert.py
  - data_expert.py
  - visualization_expert.py
  - machine_learning_expert.py
  - method_expert.py

**Conclusion**: The sync mechanism (`scripts/sync_to_public.py` + `scripts/public_allowlist.txt`) correctly filters private code. ✅

---

## 📋 Next Steps (Action Required)

### 1. Set Up PyPI Accounts (One-Time)

#### Create Accounts
- [ ] Production PyPI: https://pypi.org/account/register/
- [ ] TestPyPI: https://test.pypi.org/account/register/
- [ ] Enable 2FA on both accounts (required for ALL uploads since Jan 1, 2024)

---

## 🔐 Authentication Setup (Choose One Method)

### Method A: Trusted Publishing (OIDC) ⭐ RECOMMENDED

**Advantages:**
- ✅ No long-lived secrets in GitHub
- ✅ Zero maintenance (no token rotation)
- ✅ More secure (tokens auto-expire in ~15 minutes)
- ✅ Official PyPA recommendation

**Setup Steps:**

1. **Configure on PyPI:**
   - [ ] Log in to https://pypi.org
   - [ ] Go to project `lobster-ai` → Manage → Publishing
   - [ ] Click "Add a new publisher"
   - [ ] Fill in:
     - Owner: `the-omics-os`
     - Repository: `lobster`
     - Workflow: `publish-pypi.yml`
     - Environment: leave blank or `pypi`

2. **Configure on TestPyPI:**
   - [ ] Log in to https://test.pypi.org
   - [ ] Same steps as above
   - [ ] Environment: `testpypi`

3. **Done!** Workflow already has `id-token: write` permission configured.

**Official Docs:**
- https://docs.pypi.org/trusted-publishers/
- https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect

---

### Method B: API Tokens (Legacy/Fallback)

**When to use:** Can't configure Trusted Publishers yet, or need local testing.

#### Generate API Tokens
- [ ] PyPI: Account Settings → API tokens → "GitHub Actions - lobster-ai"
- [ ] TestPyPI: Account Settings → API tokens → "GitHub Actions - lobster-ai-test"

#### Add Tokens to GitHub Secrets
In private repo (`the-omics-os/lobster`):
- [ ] Settings → Secrets → Actions → New repository secret
- [ ] Add `PYPI_API_TOKEN` = `pypi-...`
- [ ] Add `TEST_PYPI_API_TOKEN` = `pypi-...`

**Note:** Workflow supports both methods. If Trusted Publishing is configured on PyPI, tokens are ignored.

### 2. Reserve Package Name on PyPI (First Release)

```bash
# Option A: Let workflow handle it (recommended)
# Just create the tag - workflow will fail first time, then retry after you claim the name

# Option B: Manual first upload
cd /tmp/lobster-local
python -m build
python -m twine upload dist/*  # Use PyPI credentials
```

### 3. Create Your First Release

```bash
cd /path/to/private/lobster

# 1. Ensure tests pass
make test

# 2. Commit version bump (already done)
git add pyproject.toml lobster/version.py
git commit -m "chore: bump version to 0.2.0"
git push origin main

# 3. Create and push tag
git tag -a v0.2.0 -m "Release version 0.2.0 - Alpha release"
git push origin v0.2.0
```

### 4. Monitor Workflow

1. Go to: https://github.com/the-omics-os/lobster/actions
2. Watch "Publish to PyPI" workflow
3. Stages:
   - ✅ Sync to lobster-local (automatic)
   - ✅ Build from public code (automatic)
   - ✅ Publish to TestPyPI (automatic)
   - ⏸️ **Wait for manual approval** (you approve)
   - ✅ Publish to PyPI (after approval)
   - ✅ Create GitHub release (automatic)

### 5. Test Installation

After TestPyPI publish:
```bash
# Test from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple \
            lobster-ai==0.2.0

# Verify
lobster --help
python -c "import lobster; print(lobster.__version__)"
```

After production PyPI publish:
```bash
# Test from production PyPI
pip install lobster-ai==0.2.0

# Verify
lobster --help
```

---

## 🔄 Release Workflow Diagram

```
Private Repo (lobster)
│
├─ Developer: git tag v0.2.0
├─ Developer: git push origin v0.2.0
│
└─ GitHub Actions Workflow
   │
   ├─ [1] Sync to Public
   │    └─ scripts/sync_to_public.py → lobster-local
   │
   ├─ [2] Build Package
   │    ├─ Clone lobster-local
   │    ├─ Build wheel + sdist
   │    └─ Verify no private code
   │
   ├─ [3] TestPyPI
   │    ├─ Publish to test.pypi.org
   │    └─ Test installation
   │
   ├─ [4] Manual Approval ⏸️
   │    └─ Reviewer approves in GitHub UI
   │
   ├─ [5] Production PyPI
   │    ├─ Publish to pypi.org
   │    └─ Test installation
   │
   └─ [6] GitHub Release
        └─ Attach wheels to release
```

---

## 📊 Package Details

| Metric | Value |
|--------|-------|
| **PyPI Package Name** | `lobster-ai` |
| **Import Name** | `lobster` |
| **Current Version** | `0.2.0` |
| **Package Size** | ~1.3 MB (tar.gz), ~978 KB (wheel) |
| **Python Requirement** | 3.11+ |
| **License** | AGPL-3.0-or-later |
| **Public Repo** | github.com/the-omics-os/lobster-local |

### Installation Commands

```bash
# Production (after release)
pip install lobster-ai

# Import in code
import lobster
print(lobster.__version__)

# Development
pip install git+https://github.com/the-omics-os/lobster-local.git@main

# Specific version
pip install lobster-ai==0.2.0
```

### Authentication Methods Supported

The workflow supports **both** authentication methods with automatic fallback:

```yaml
# Workflow configuration (already set up):
permissions:
  id-token: write  # Enables Trusted Publishing
  contents: read

# Publishing action:
- uses: pypa/gh-action-pypi-publish@release/v1
  with:
    password: ${{ secrets.PYPI_API_TOKEN || '' }}  # Fallback to token if needed
```

**How it works:**
1. If Trusted Publishing configured on PyPI → uses OIDC (secrets ignored)
2. If NOT configured → falls back to API token from secrets
3. Zero-downtime migration: configure Trusted Publishing anytime

---

## 🛡️ Safety Features

1. **Code Filtering**: Only public code from lobster-local gets published
2. **Verification Step**: Workflow scans package for private code patterns
3. **TestPyPI First**: Always test on test.pypi.org before production
4. **Manual Approval**: Human review required before production release
5. **Automated Testing**: Package installation verified before/after publishing

---

## 📚 Documentation

- **Setup Guide**: `docs/PYPI_SETUP_GUIDE.md`
- **Allowlist**: `scripts/public_allowlist.txt`
- **Sync Script**: `scripts/sync_to_public.py`
- **Workflow**: `.github/workflows/publish-pypi.yml`
- **Release Script**: `scripts/release.sh`

---

## 🎯 Success Criteria

After successful release:
- [ ] Package visible at https://pypi.org/project/lobster-ai/
- [ ] Installation works: `pip install lobster-ai`
- [ ] CLI works: `lobster --help`
- [ ] Import works: `python -c "import lobster"`
- [ ] Version correct: `lobster.__version__ == "0.2.0"`
- [ ] No premium code in package (verified via `pip show -f lobster-ai`)
- [ ] GitHub release created with wheels attached

---

## ⚠️ Important Notes

1. **Never build for PyPI from private repo** - always use lobster-local
2. **Always test on TestPyPI first** - production releases can't be deleted
3. **Version numbers can't be reused** - bump version for each release
4. **Keep API tokens secret** - rotate immediately if exposed
5. **Review packages before approval** - manual gate is critical for security

---

## 🆘 Troubleshooting

### Package name already exists
- Use `lobster-ai` (not `lobster`, which is taken)
- Already configured correctly in pyproject.toml

### Private code in package
- Verify building from lobster-local, not private repo
- Check `scripts/public_allowlist.txt` is up to date
- Review workflow build logs

### Workflow fails on sync
- Check `PUBLIC_REPO_DEPLOY_KEY` secret is set
- Verify SSH key has write access to lobster-local

### Manual approval not appearing
- Check GitHub Environments are configured
- Ensure you're added as required reviewer for `pypi-approval` environment

---

## 📞 Support

For issues:
1. Review this summary
2. Check `docs/PYPI_SETUP_GUIDE.md`
3. Inspect GitHub Actions logs
4. Test locally: `cd /tmp && git clone lobster-local && cd lobster-local && python -m build`

---

## 🔄 Modern Authentication Approach

**2024+ Best Practice:**

This setup follows the **latest PyPI security recommendations** as of 2024:

1. **Trusted Publishing (OIDC) as primary method**
   - No long-lived secrets
   - Automatic token management by PyPI
   - Official PyPA recommendation

2. **Backward compatible with API tokens**
   - Supports teams not yet using Trusted Publishing
   - Fallback mechanism in workflow
   - Easy migration path

3. **2FA required for ALL uploads**
   - PyPI requirement since January 1, 2024
   - Applies to both tokens and Trusted Publishing

4. **Modern metadata inspection**
   - Uses `pip show` and `pyproject.toml`
   - No deprecated `setup.py` commands

**References:**
- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions Publishing Guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [PyPI 2FA Announcement](https://blog.pypi.org/posts/2023-05-25-securing-pypi-with-2fa/)

---

**Status**: ✅ Ready for first release
**Version**: 0.2.0 (Alpha)
**Last Updated**: 2025-01-17
**Auth Method**: Trusted Publishing (OIDC) with API token fallback
