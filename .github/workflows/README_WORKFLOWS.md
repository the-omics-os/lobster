# GitHub Actions Workflows

This directory contains all CI/CD workflows for the Lobster project.

## Workflow Index

### 📦 Repository Synchronization

| Workflow | File | Trigger | Purpose |
|----------|------|---------|---------|
| **Sync to Public Repository via Github** | `sync-to-public.yml` | Push to `main` | Syncs FREE tier code to `lobster-local` (public repo) |
| **Custom Package - DataBioMix Sync** | `custom-package-databiomix-sync.yml` | Changes to DataBioMix files | Syncs metadata_assistant to `lobster-custom-databiomix` |

### 🚀 Publishing & Releases

| Workflow | File | Trigger | Purpose |
|----------|------|---------|---------|
| **Publish to PyPI** | `publish-pypi.yml` | Git tags (`v*`) | 7-stage pipeline: sync → build → TestPyPI → approval → PyPI → release |
| **Public Release** | `public-release.yml` | Release published | Publishes `lobster-local` releases |
| **Release** | `release.yml` | Tags | Creates GitHub releases |

### 🧪 Testing & Validation

| Workflow | File | Trigger | Purpose |
|----------|------|---------|---------|
| **CI - Basic** | `ci-basic.yml` | PRs + push | Fast validation (lint, format, quick tests) |
| **PR Validation - Basic** | `pr-validation-basic.yml` | Pull requests | PR-specific checks |
| **Platform Tests** | `platform-tests.yml` | Push to `main` | Cross-platform testing (Ubuntu, macOS, Windows) |
| **API Integration Tests** | `api-integration-tests.yml` | Scheduled/manual | Tests against live APIs (PubMed, GEO, etc.) |

### 🐳 Container Builds

| Workflow | File | Trigger | Purpose |
|----------|------|---------|---------|
| **Docker Build & Publish** | `docker.yml` | Tags + manual | Builds and publishes CLI Docker image |

### 🔄 Maintenance

| Workflow | File | Trigger | Purpose |
|----------|------|---------|---------|
| **Dependency Updates** | `dependency-updates.yml` | Scheduled | Automated dependency management |

---

## Custom Package Workflows (Pattern)

Custom packages follow this naming pattern:
```
custom-package-{customer}-sync.yml
```

### Adding a New Custom Package Workflow

1. **Copy template**:
   ```bash
   cp custom-package-databiomix-sync.yml custom-package-newcustomer-sync.yml
   ```

2. **Update configuration**:
   - Change workflow name: `name: Custom Package - NewCustomer Sync`
   - Update `paths:` to include customer-specific files
   - Change repository: `repository: the-omics-os/lobster-custom-newcustomer`
   - Update package name in sync command: `--package newcustomer`

3. **Test**:
   ```bash
   # Manual trigger via GitHub Actions UI
   # Or push a change to trigger paths
   ```

---

## Workflow Security

### Secrets Used

| Secret | Used By | Purpose |
|--------|---------|---------|
| `PUBLIC_REPO_DEPLOY_KEY` | `sync-to-public.yml` | SSH push to lobster-local |
| `AWS_ACCESS_KEY_ID` | Custom package workflows | S3 package upload |
| `AWS_SECRET_ACCESS_KEY` | Custom package workflows | S3 package upload |
| `PYPI_API_TOKEN` | `publish-pypi.yml` | Publish to PyPI |
| `GITHUB_TOKEN` | All workflows | Auto-provided by GitHub |

### Permissions

All sync workflows use:
```yaml
permissions:
  contents: write
```

This allows workflows to:
- Push to repositories
- Create commits
- Update files

---

## Manual Triggers

All sync workflows support manual execution via workflow_dispatch:

### Sync to Public (Manual)
```
GitHub Actions → sync-to-public.yml → Run workflow
Options:
  - dry_run: Preview changes
  - force_push: Force push (history reset)
```

### DataBioMix Sync (Manual)
```
GitHub Actions → custom-package-databiomix-sync.yml → Run workflow
Options:
  - dry_run: Preview changes without committing
```

---

## Troubleshooting

### Workflow Not Triggering

**Issue**: Workflow doesn't run after push

**Solutions**:
1. Check `paths:` section matches your changed files
2. Verify workflow file syntax: `yamllint .github/workflows/*.yml`
3. Check workflow is not in `disabled/` directory

### Sync Conflicts

**Issue**: Workflow fails with merge conflicts

**Solutions**:
1. Manually resolve conflicts in target repo
2. Re-run workflow via Actions UI

### Permission Errors

**Issue**: "Resource not accessible by integration"

**Solutions**:
1. Check `permissions:` in workflow file
2. Verify secrets are configured: Settings → Secrets and variables → Actions

---

## Validation

### Allowlist Validation (CI)

The `publish-pypi.yml` and `sync-to-public.yml` workflows validate:
```bash
python scripts/generate_allowlist.py --validate
```

This ensures `public_allowlist.txt` stays in sync with `subscription_tiers.py`.

### Manual Validation

```bash
# Validate workflow syntax
yamllint .github/workflows/*.yml

# Test sync scripts locally
python scripts/sync_to_custom.py --package databiomix --dry-run
python scripts/sync_to_public.py --repo git@github.com:the-omics-os/lobster-local.git --dry-run
```

---

## Monitoring

### Workflow Status

Check status: [GitHub Actions](https://github.com/the-omics-os/lobster/actions)

### Failed Workflows

Failed workflows will:
1. Send email notifications (if enabled)
2. Show red X on commit
3. Create step summary with error details

### Workflow Logs

Access via:
```
GitHub → Actions → Select workflow run → Click on job → View logs
```

---

## Best Practices

1. **Test locally first**: Always test sync scripts with `--dry-run` before pushing
2. **Small commits**: Keep commits focused to make sync workflows faster
3. **Manual trigger**: Use workflow_dispatch for one-off syncs
4. **Monitor first run**: Watch new workflows closely on first execution
5. **Document changes**: Update this README when adding new workflows

---

## Related Documentation

- **Sync Infrastructure**: `scripts/README_SYNC_INFRASTRUCTURE.md`
- **Premium Licensing**: `docs/PREMIUM_LICENSING.md`
- **PyPI Publishing**: `docs/PYPI_SETUP_GUIDE.md`
- **Custom Packages**: `scripts/README_SYNC_INFRASTRUCTURE.md#adding-a-new-custom-package`
