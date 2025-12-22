# Sync Infrastructure Documentation

This document describes the automated synchronization system for Lobster repositories and custom packages.

## Architecture Overview

```
lobster (private) ──────┐
                        │
                        ├─── sync_to_public.py ────────> lobster-local (public)
                        │     • Uses: public_allowlist.txt
                        │     • Workflow: .github/workflows/sync-to-public.yml
                        │
                        ├─── sync_to_custom.py ────────> lobster-custom-databiomix
                        │     • Uses: custom_package_allowlist_databiomix.txt
                        │     • Workflow: .github/workflows/sync-to-custom-databiomix.yml
                        │
                        └─── sync_wikis.py ────────────> Both wikis
                              • Uses: wiki_public_allowlist.txt
                              • Workflow: .github/workflows/sync-wikis.yml
```

## 1. Public Repository Sync (lobster → lobster-local)

### Purpose
Sync FREE tier code to public open-core distribution.

### Files
- **Script**: `scripts/sync_to_public.py`
- **Allowlist**: `scripts/public_allowlist.txt` (AUTO-GENERATED)
- **Generator**: `scripts/generate_allowlist.py`
- **Workflow**: `.github/workflows/sync-to-public.yml`

### Single Source of Truth
```
subscription_tiers.py (defines FREE/PREMIUM tiers)
         ↓
generate_allowlist.py (derives file-level allowlist)
         ↓
public_allowlist.txt (used by sync_to_public.py)
         ↓
lobster-local (public PyPI package)
```

### Usage
```bash
# Regenerate public allowlist after tier changes
python scripts/generate_allowlist.py --write

# Validate (CI)
python scripts/generate_allowlist.py --validate

# Sync to public repo
python scripts/sync_to_public.py --repo git@github.com:the-omics-os/lobster-local.git --dry-run
python scripts/sync_to_public.py --repo git@github.com:the-omics-os/lobster-local.git
```

### Automation
- **Trigger**: Push to `main` branch
- **Actions**: Sync code → Build wheel → Publish to PyPI (on tags)

---

## 2. Custom Package Sync (lobster → lobster-custom-*)

### Purpose
Sync PREMIUM/ENTERPRISE code to customer-specific packages.

### Files
- **Script**: `scripts/sync_to_custom.py`
- **Allowlists**:
  - `scripts/custom_package_allowlist.txt` (generic - all premium agents)
  - `scripts/custom_package_allowlist_databiomix.txt` (DataBioMix-specific)
- **Workflow**: `.github/workflows/sync-to-custom-databiomix.yml`

### Package-Specific Allowlists
The sync script automatically selects:
1. **Package-specific** (`custom_package_allowlist_{package}.txt`) if it exists
2. **Generic fallback** (`custom_package_allowlist.txt`) otherwise

This allows tailoring each customer package to their specific needs.

### DataBioMix Example
```bash
# Sync DataBioMix-specific files only
python scripts/sync_to_custom.py --package databiomix --dry-run
python scripts/sync_to_custom.py --package databiomix
```

**Files synced for DataBioMix**:
- `lobster/agents/metadata_assistant.py`
- `lobster/services/metadata/*` (sample_mapping, disease_standardization, microbiome_filtering)
- `lobster/services/orchestration/publication_processing_service.py`

**Files NOT synced** (not used by DataBioMix):
- ML agent (`machine_learning_expert.py`)
- Proteomics agent (`proteomics_expert.py`)
- Structure visualization agent
- All ML/proteomics/visualization services

### Automation (DataBioMix)
- **Trigger**: Changes to DataBioMix-relevant files
- **Actions**: Sync files → Commit → Push to `lobster-custom-databiomix`
- **Manual**: Developer runs build & S3 upload

---

## 3. Wiki Sync (lobster/wiki → Both wikis)

### Purpose
Keep documentation synchronized between private and public wikis.

### Files
- **Script**: `scripts/sync_wikis.py`
- **Allowlist**: `scripts/wiki_public_allowlist.txt` (filename matching)
- **Workflow**: `.github/workflows/sync-wikis.yml`

### Usage
```bash
python scripts/sync_wikis.py --dry-run
python scripts/sync_wikis.py
```

---

## File Reference

### Sync Scripts
| Script | Purpose | Allowlist Type |
|--------|---------|----------------|
| `sync_to_public.py` | lobster → lobster-local | Auto-generated from tiers |
| `sync_to_custom.py` | lobster → lobster-custom-* | Package-specific or generic |
| `sync_wikis.py` | lobster/wiki → both wikis | Filename patterns |

### Allowlist Files
| File | Purpose | Generated? |
|------|---------|------------|
| `public_allowlist.txt` | Public repo sync | ✅ Auto (from subscription_tiers.py) |
| `custom_package_allowlist.txt` | Generic custom sync | ❌ Manual |
| `custom_package_allowlist_databiomix.txt` | DataBioMix-specific | ❌ Manual |
| `wiki_public_allowlist.txt` | Public wiki sync | ❌ Manual |

### GitHub Actions Workflows
| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `sync-to-public.yml` | Push to main | Sync + publish lobster-local |
| `sync-to-custom-databiomix.yml` | Changes to DataBioMix files | Sync lobster-custom-databiomix |
| `sync-wikis.yml` | Wiki changes | Sync both wikis |
| `publish-pypi.yml` | Git tags | Publish lobster-ai to PyPI |

---

## Adding a New Custom Package

### 1. Create Package-Specific Allowlist
```bash
cp scripts/custom_package_allowlist.txt scripts/custom_package_allowlist_newcustomer.txt
# Edit to include only customer-specific agents/services
```

### 2. Create Workflow
```bash
cp .github/workflows/sync-to-custom-databiomix.yml .github/workflows/sync-to-custom-newcustomer.yml
# Update package name and trigger paths
```

### 3. Test Sync
```bash
python scripts/sync_to_custom.py --package newcustomer --dry-run
python scripts/sync_to_custom.py --package newcustomer
```

### 4. Build & Deploy
```bash
cd ../lobster-custom-newcustomer
python -m build
aws s3 cp dist/*.whl s3://lobster-license-packages-649207544517/newcustomer/
```

---

## Maintenance

### When to Regenerate public_allowlist.txt
Run after ANY of these changes:
```bash
# Changed subscription tiers
vim lobster/config/subscription_tiers.py
python scripts/generate_allowlist.py --write

# Added new premium agent
vim lobster/agents/new_premium_agent.py
# Update AGENT_FILE_MAPPING in generate_allowlist.py
python scripts/generate_allowlist.py --write

# Added new agent service dependencies
# Update AGENT_SERVICE_DEPENDENCIES in generate_allowlist.py
python scripts/generate_allowlist.py --write
```

### When to Update Custom Allowlists
Manually update `custom_package_allowlist_{package}.txt` when:
- Adding new files customer needs
- Customer requirements change
- New services added to existing agents

### CI Validation
Both sync-to-public and publish-pypi workflows validate:
```bash
python scripts/generate_allowlist.py --validate
```

This ensures public_allowlist.txt stays in sync with subscription_tiers.py.

---

## Security Considerations

### Access Control
- **Public repo**: Anyone (lobster-local on GitHub)
- **Custom packages**: Customer-specific (private repos)
- **S3 packages**: Pre-signed URLs during activation (1-hour expiry)

### Secrets Required
| Secret | Used By | Purpose |
|--------|---------|---------|
| `PUBLIC_REPO_DEPLOY_KEY` | sync-to-public | SSH push to lobster-local |
| `AWS_ACCESS_KEY_ID` | sync-to-custom-* | S3 package upload |
| `AWS_SECRET_ACCESS_KEY` | sync-to-custom-* | S3 package upload |
| `PYPI_API_TOKEN` | publish-pypi | Publish lobster-ai |

### Validation Gates
1. **Allowlist validation**: CI ensures allowlist matches source of truth
2. **Import tests**: CI verifies no premium imports in public package
3. **Manual approval**: publish-pypi requires approval before production

---

## Troubleshooting

### "Allowlist out of sync" CI Error
```bash
cd lobster
python scripts/generate_allowlist.py --write
git add scripts/public_allowlist.txt
git commit -m "Regenerate public allowlist"
```

### Custom Package Missing Files
```bash
# Check what would be synced
python scripts/sync_to_custom.py --package databiomix --dry-run

# Add missing patterns to allowlist
vim scripts/custom_package_allowlist_databiomix.txt

# Re-sync
python scripts/sync_to_custom.py --package databiomix
```

### Workflow Not Triggering
Check workflow file's `paths:` section matches the files you changed.

---

## References

- **Architecture**: `/Users/tyo/GITHUB/omics-os/README.md`
- **Premium Licensing**: `/Users/tyo/GITHUB/omics-os/docs/PREMIUM_LICENSING.md`
- **PyPI Publishing**: `/Users/tyo/GITHUB/omics-os/docs/PYPI_SETUP_GUIDE.md`
- **Subscription Tiers**: `lobster/config/subscription_tiers.py`
