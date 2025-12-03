# Deprecated Agents Cleanup - Implementation Checklist

**Start Date:** ____________
**Target Completion:** ____________
**Assigned To:** ____________

---

## Phase 0: Create Test Suite (BLOCKER)

**Estimated Time:** 1 week
**Priority:** P0 (Must complete before any other phase)

### Test File Creation
- [ ] Create directory: `mkdir -p tests/unit/agents/transcriptomics`
- [ ] Create `tests/unit/agents/transcriptomics/__init__.py`
- [ ] Create `tests/unit/agents/transcriptomics/test_transcriptomics_expert.py`
- [ ] Create `tests/unit/agents/transcriptomics/test_annotation_expert.py`
- [ ] Create `tests/unit/agents/transcriptomics/test_de_analysis_expert.py`
- [ ] Create `tests/unit/agents/transcriptomics/test_transcriptomics_integration.py`

### Test Coverage Requirements

#### test_transcriptomics_expert.py
- [ ] Test agent initialization
- [ ] Test `check_data_status` tool
- [ ] Test `assess_data_quality` tool
- [ ] Test `filter_and_normalize_modality` tool
- [ ] Test `cluster_modality` tool
- [ ] Test `subcluster_cells` tool
- [ ] Test `create_analysis_summary` tool
- [ ] Test data type auto-detection (single-cell vs bulk)
- [ ] Test delegation to annotation_expert
- [ ] Test delegation to de_analysis_expert
- [ ] Test error handling (ModalityNotFoundError)
- [ ] Test state management across tool calls

#### test_annotation_expert.py
- [ ] Test agent initialization
- [ ] Test `annotate_cell_types_automatically` tool
- [ ] Test `label_cluster_manually` tool
- [ ] Test `detect_debris_cells` tool
- [ ] Test `apply_annotation_template` tool
- [ ] Test error handling
- [ ] Test state preservation

#### test_de_analysis_expert.py
- [ ] Test agent initialization
- [ ] Test `perform_pseudobulk_aggregation` tool
- [ ] Test `perform_differential_expression` tool
- [ ] Test `build_design_formula` tool
- [ ] Test `perform_pathway_enrichment` tool
- [ ] Test bulk RNA-seq quantification import
- [ ] Test error handling
- [ ] Test state preservation

#### test_transcriptomics_integration.py
- [ ] Test full workflow: QC → clustering → annotation
- [ ] Test full workflow: QC → pseudobulk → DE
- [ ] Test parent-to-child delegation
- [ ] Test child agent completion handoff back to parent
- [ ] Test error propagation across agents
- [ ] Test state consistency

### Verification
- [ ] Run tests: `pytest tests/unit/agents/transcriptomics/ -v`
- [ ] Check coverage: `pytest --cov=lobster/agents/transcriptomics --cov-report=html tests/unit/agents/transcriptomics/`
- [ ] Verify coverage ≥80%
- [ ] All tests passing
- [ ] No flaky tests

### Commit
- [ ] `git add tests/unit/agents/transcriptomics/`
- [ ] `git commit -m "test: Add comprehensive test suite for transcriptomics architecture"`
- [ ] `git push origin feature/transcriptomics-tests`
- [ ] Create PR and get approval

**Sign-off:** ____________ Date: ____________

---

## Phase 1: Update Documentation

**Estimated Time:** 1 day
**Priority:** P1

### Wiki Updates

#### wiki/19-agent-system.md
- [ ] Open file: `vim wiki/19-agent-system.md`
- [ ] Mark `singlecell_expert` as DEPRECATED
- [ ] Mark `bulk_rnaseq_expert` as DEPRECATED
- [ ] Add migration notes pointing to `transcriptomics_expert`
- [ ] Update agent list to show new unified architecture
- [ ] Add "See Migration Guide" link
- [ ] Save and verify rendering

#### wiki/15-agents-api.md
- [ ] Open file: `vim wiki/15-agents-api.md`
- [ ] Replace examples using `singlecell_expert` with `transcriptomics_expert`
- [ ] Replace examples using `bulk_rnaseq_expert` with `transcriptomics_expert`
- [ ] Update code snippets
- [ ] Update import statements
- [ ] Save and verify rendering

#### wiki/30-glossary.md
- [ ] Open file: `vim wiki/30-glossary.md`
- [ ] Find: "lobster.agents.singlecell_expert.singlecell_expert"
- [ ] Replace with: "lobster.agents.transcriptomics.transcriptomics_expert.transcriptomics_expert"
- [ ] Add glossary entry for "Agent Unification"
- [ ] Save and verify rendering

#### wiki/34-architecture-diagram.md
- [ ] Open file: `vim wiki/34-architecture-diagram.md`
- [ ] Update architecture diagrams
- [ ] Replace old agent boxes with new unified architecture
- [ ] Add sub-agent relationships (annotation_expert, de_analysis_expert)
- [ ] Save and verify rendering

#### lobster/config/README_CONFIGURATION.md
- [ ] Open file: `vim lobster/config/README_CONFIGURATION.md`
- [ ] Remove lines referencing `singlecell_expert_agent` (line 31)
- [ ] Remove lines referencing `bulk_rnaseq_expert_agent` (line 32)
- [ ] Add entry for `transcriptomics_expert`
- [ ] Save and verify rendering

### Migration Guide Creation

#### wiki/41-migration-guides.md
- [ ] Open file: `vim wiki/41-migration-guides.md`
- [ ] Append new section: "## Transcriptomics Agent Migration (v2.6 → v2.7)"
- [ ] Add summary of changes
- [ ] Add code examples (before/after)
- [ ] Add table mapping old tools to new tools
- [ ] Add troubleshooting section
- [ ] Add "Why this change?" explanation
- [ ] Save and verify rendering

### Deprecation Warnings

#### lobster/agents/singlecell_expert.py
- [ ] Open file: `vim lobster/agents/singlecell_expert.py`
- [ ] Update deprecation warning (lines 15-20)
- [ ] Add link to migration guide: `wiki/41-migration-guides.md#transcriptomics-migration`
- [ ] Add version removal notice: "will be removed in v2.7"
- [ ] Save

#### lobster/agents/bulk_rnaseq_expert.py
- [ ] Open file: `vim lobster/agents/bulk_rnaseq_expert.py`
- [ ] Update deprecation warning (lines 14-19)
- [ ] Add link to migration guide: `wiki/41-migration-guides.md#transcriptomics-migration`
- [ ] Add version removal notice: "will be removed in v2.7"
- [ ] Save

### Verification
- [ ] Build documentation: `make docs` (if applicable)
- [ ] Check all links work
- [ ] Verify code snippets are correct
- [ ] Verify deprecation warnings show on import:
  ```bash
  python -c "from lobster.agents.singlecell_expert import singlecell_expert"
  # Should show warning
  ```

### Commit
- [ ] `git add wiki/ lobster/agents/singlecell_expert.py lobster/agents/bulk_rnaseq_expert.py lobster/config/README_CONFIGURATION.md`
- [ ] `git commit -m "docs: Update documentation for transcriptomics agent unification"`
- [ ] `git push origin feature/transcriptomics-docs`
- [ ] Create PR and get approval

**Sign-off:** ____________ Date: ____________

---

## Phase 2: Refactor Tests

**Estimated Time:** 1 week
**Priority:** P1

### Group A: Delete Agent-Specific Tests

- [ ] Verify services are independently tested:
  ```bash
  pytest tests/unit/services/analysis/ -v
  pytest tests/unit/services/quality/ -v
  ```
- [ ] Delete: `git rm tests/unit/agents/test_singlecell_expert.py`
- [ ] Delete: `git rm tests/unit/agents/test_bulk_quantification_communication.py`
- [ ] Commit: `git commit -m "test: Remove deprecated agent unit tests (services tested separately)"`

### Group B: Integration Tests

#### tests/integration/test_scvi_agent_handoff.py (8 imports)
- [ ] Open file
- [ ] Find: `from lobster.agents.singlecell_expert import singlecell_expert`
- [ ] Replace with: `from lobster.agents.transcriptomics.transcriptomics_expert import transcriptomics_expert`
- [ ] Update agent factory calls: `singlecell_expert(data_manager)` → `transcriptomics_expert(data_manager)`
- [ ] Update tool names if any direct calls
- [ ] Update assertions (agent names, response patterns)
- [ ] Run test: `pytest tests/integration/test_scvi_agent_handoff.py -xvs`
- [ ] Verify PASSING
- [ ] Save

#### tests/integration/test_agent_guided_formula_construction.py
- [ ] Open file
- [ ] Find: `from lobster.agents.singlecell_expert import singlecell_expert`
- [ ] Replace with: `from lobster.agents.transcriptomics.transcriptomics_expert import transcriptomics_expert`
- [ ] Update agent factory calls
- [ ] Update tool names if any direct calls
- [ ] Update assertions
- [ ] Run test: `pytest tests/integration/test_agent_guided_formula_construction.py -xvs`
- [ ] Verify PASSING
- [ ] Save

#### tests/integration/test_quantification_end_to_end.py
- [ ] Open file
- [ ] Find: `from lobster.agents.bulk_rnaseq_expert import bulk_rnaseq_expert`
- [ ] Replace with: `from lobster.agents.transcriptomics.transcriptomics_expert import transcriptomics_expert`
- [ ] Update agent factory calls
- [ ] Update tool names if any direct calls
- [ ] Update assertions
- [ ] Run test: `pytest tests/integration/test_quantification_end_to_end.py -xvs`
- [ ] Verify PASSING
- [ ] Save

#### tests/integration/test_scvi_handoff_flow.py
- [ ] Open file
- [ ] Update imports as above
- [ ] Update agent factory calls
- [ ] Run test: `pytest tests/integration/test_scvi_handoff_flow.py -xvs`
- [ ] Verify PASSING
- [ ] Save

### Group C: System Tests

#### tests/system/test_full_analysis_workflows.py
- [ ] Open file
- [ ] Find and replace both imports (singlecell_expert and bulk_rnaseq_expert)
- [ ] Replace with: `from lobster.agents.transcriptomics.transcriptomics_expert import transcriptomics_expert`
- [ ] Update all agent factory calls
- [ ] Update assertions
- [ ] Run test: `pytest tests/system/test_full_analysis_workflows.py -xvs`
- [ ] Verify PASSING
- [ ] Save

#### tests/system/test_error_recovery.py
- [ ] Open file
- [ ] Update imports (same pattern as above)
- [ ] Update agent factory calls
- [ ] Update assertions
- [ ] Run test: `pytest tests/system/test_error_recovery.py -xvs`
- [ ] Verify PASSING
- [ ] Save

### Group D: Performance Tests

#### tests/performance/test_large_dataset_processing.py
- [ ] Open file
- [ ] Update imports
- [ ] Update agent factory calls
- [ ] Run test: `pytest tests/performance/test_large_dataset_processing.py -xvs`
- [ ] Benchmark performance (should be similar)
- [ ] Verify PASSING
- [ ] Save

#### tests/performance/test_concurrent_agent_execution.py
- [ ] Open file
- [ ] Update imports
- [ ] Update agent factory calls
- [ ] Run test: `pytest tests/performance/test_concurrent_agent_execution.py -xvs`
- [ ] Benchmark performance (should be similar)
- [ ] Verify PASSING
- [ ] Save

### Group E: Other Tests

#### tests/test_scvi_integration_validation.py
- [ ] Open file
- [ ] Update imports
- [ ] Update agent factory calls
- [ ] Run test: `pytest tests/test_scvi_integration_validation.py -xvs`
- [ ] Verify PASSING
- [ ] Save

#### tests/test_visualization_expert.py
- [ ] Open file
- [ ] Check if it uses singlecell_expert (search for import)
- [ ] If yes: update imports
- [ ] Run test: `pytest tests/test_visualization_expert.py -xvs`
- [ ] Verify PASSING
- [ ] Save

#### tests/test_expert_handoffs.py
- [ ] Open file
- [ ] Check for old handoff tool names
- [ ] Update to new tool names: `handoff_to_transcriptomics_expert`
- [ ] Run test: `pytest tests/test_expert_handoffs.py -xvs`
- [ ] Verify PASSING
- [ ] Save

### Full Test Suite Verification
- [ ] Run all unit tests: `pytest tests/unit/ -v`
- [ ] Run all integration tests: `pytest tests/integration/ -v`
- [ ] Run all system tests: `pytest tests/system/ -v`
- [ ] Run all performance tests: `pytest tests/performance/ -v`
- [ ] Run full suite: `pytest tests/ -k "not manual" -v`
- [ ] Verify ALL PASSING
- [ ] Check for deprecation warnings (should only come from imports)
- [ ] Verify no performance regressions

### Commit
- [ ] `git add tests/`
- [ ] `git commit -m "test: Refactor tests to use transcriptomics_expert instead of deprecated agents"`
- [ ] `git push origin feature/transcriptomics-test-refactor`
- [ ] Create PR and get approval
- [ ] Merge PR

### Soak Period
- [ ] Allow 1 week for integration testing
- [ ] Monitor CI/CD pipeline
- [ ] Check for bug reports
- [ ] Verify no issues in production

**Sign-off:** ____________ Date: ____________

---

## Phase 3: Remove from Public Sync

**Estimated Time:** 1 day
**Priority:** P2
**Prerequisites:** Phase 2 complete + 1 week soak

### Update Public Allowlist

#### scripts/public_allowlist.txt
- [ ] Open file: `vim scripts/public_allowlist.txt`
- [ ] Find line 100: `lobster/agents/singlecell_expert.py`
- [ ] Delete or comment out line
- [ ] Verify line 125 still has: `!lobster/agents/bulk_rnaseq_expert.py` (already excluded)
- [ ] Save

### Dry Run Sync
- [ ] Run dry run: `python scripts/sync_to_public.py --repo git@github.com:the-omics-os/lobster-local.git --dry-run`
- [ ] Review changes carefully
- [ ] Verify singlecell_expert.py will be removed
- [ ] Verify transcriptomics/ directory will be synced
- [ ] Verify no other unexpected changes

### Actual Sync
- [ ] Run sync: `python scripts/sync_to_public.py --repo git@github.com:the-omics-os/lobster-local.git`
- [ ] Verify sync completed successfully
- [ ] Check public repo: https://github.com/the-omics-os/lobster-local
- [ ] Verify singlecell_expert.py removed
- [ ] Verify transcriptomics/ directory present

### Update Public Documentation

#### Create GitHub Issue in Public Repo
- [ ] Go to: https://github.com/the-omics-os/lobster-local/issues
- [ ] Click "New Issue"
- [ ] Title: `[DEPRECATION] singlecell_expert removed in favor of transcriptomics_expert`
- [ ] Body (template):
  ```markdown
  ## Summary
  Starting in v2.7.0 (planned release: [DATE]), `singlecell_expert` has been removed
  in favor of the unified `transcriptomics_expert` agent.

  ## What Changed
  - `singlecell_expert` → `transcriptomics_expert`
  - `bulk_rnaseq_expert` → `transcriptomics_expert` (already premium-only)
  - Single agent now handles both single-cell AND bulk RNA-seq

  ## Migration Guide
  See: [wiki/41-migration-guides.md#transcriptomics-migration](link)

  ## Timeline
  - **v2.6.x**: Deprecation warnings added
  - **v2.7.0 (in 1 month)**: Removal complete
  - **Support**: Available via [support channel]

  ## Benefits
  - 23% code reduction
  - Improved modularity
  - Unified API for all transcriptomics workflows

  ## Questions?
  Please comment below or contact [support email]
  ```
- [ ] Label: `breaking-change`, `deprecation`
- [ ] Create issue

### Update Public Wiki
- [ ] Navigate to public wiki: https://github.com/the-omics-os/lobster-local/wiki
- [ ] Verify migration guide is present and linked
- [ ] Add notice to main wiki page about deprecation
- [ ] Update quick start guide if needed

### Commit Allowlist Changes
- [ ] `git add scripts/public_allowlist.txt`
- [ ] `git commit -m "build: Remove singlecell_expert from public sync (deprecated)"`
- [ ] `git push origin main`

### Communication
- [ ] Post announcement to Discord/Slack (if applicable)
- [ ] Email notification to active users (if applicable)
- [ ] Update landing page documentation (if applicable)

**Sign-off:** ____________ Date: ____________

---

## Phase 4: Final Removal

**Estimated Time:** 1 day
**Priority:** P2
**Prerequisites:** Phase 3 complete + 1 month deprecation notice

### Pre-Removal Verification
- [ ] Verify deprecation notice posted for ≥1 month
- [ ] Check bug reports (should be ZERO related to transcriptomics_expert)
- [ ] Run full test suite: `pytest tests/ -k "not manual" -v` (should PASS)
- [ ] Verify coverage: `pytest --cov=lobster --cov-report=html`
- [ ] Verify agent registry has only transcriptomics_expert
- [ ] Verify NO production code imports old agents:
  ```bash
  grep -r "from lobster.agents.singlecell_expert" lobster/ --include="*.py" | grep -v test
  grep -r "from lobster.agents.bulk_rnaseq_expert" lobster/ --include="*.py" | grep -v test
  # Should return: NOTHING
  ```

### Remove Agent Files
- [ ] Delete: `git rm lobster/agents/singlecell_expert.py`
- [ ] Delete: `git rm lobster/agents/bulk_rnaseq_expert.py`
- [ ] Verify files removed: `git status`

### Check State Classes
- [ ] Open: `vim lobster/agents/state.py`
- [ ] Search for: `SingleCellExpertState`
- [ ] Search for: `BulkRNASeqExpertState`
- [ ] If found and unused elsewhere:
  - [ ] Remove state class definitions
  - [ ] Update imports
  - [ ] Save
- [ ] If used elsewhere: KEEP (document why)

### Update CHANGELOG.md
- [ ] Open: `vim CHANGELOG.md`
- [ ] Add new section for v2.7.0:
  ```markdown
  ## [2.7.0] - YYYY-MM-DD

  ### Removed
  - **BREAKING:** Removed deprecated `singlecell_expert` and `bulk_rnaseq_expert` agents
    - Use `transcriptomics_expert` instead (handles both single-cell and bulk RNA-seq)
    - See migration guide: wiki/41-migration-guides.md#transcriptomics-migration
    - Code reduction: 6692 lines removed (23% smaller)

  ### Improved
  - Unified transcriptomics architecture with better modularity
  - Sub-agents for specialized tasks (annotation, differential expression)
  - Cleaner delegation patterns

  ### Migration
  - Old agents emitted deprecation warnings since v2.6.0
  - All functionality preserved in `transcriptomics_expert`
  - Test suite fully migrated and passing
  ```
- [ ] Save

### Create Release Notes
- [ ] Create file: `docs/releases/v2.7.0.md`
- [ ] Write comprehensive release notes:
  - Summary of changes
  - Migration instructions
  - Breaking changes
  - New features
  - Bug fixes
  - Contributors
- [ ] Link from CHANGELOG.md

### Final Testing
- [ ] Run full test suite: `pytest tests/ -k "not manual" -v`
- [ ] Verify ALL PASSING
- [ ] Run linters: `make lint`
- [ ] Run type checks: `make type-check`
- [ ] Run coverage: `pytest --cov=lobster --cov-report=html`
- [ ] Verify NO import errors
- [ ] Test CLI: `lobster chat` (start and exit cleanly)
- [ ] Test query: `lobster query "list agents"`

### Commit and Release
- [ ] `git add -A`
- [ ] `git commit -m "BREAKING: Remove deprecated singlecell_expert and bulk_rnaseq_expert [v2.7.0]"`
- [ ] `git push origin main`
- [ ] Create git tag: `git tag -a v2.7.0 -m "Release v2.7.0: Unified transcriptomics architecture"`
- [ ] Push tag: `git push origin v2.7.0`
- [ ] Create GitHub release from tag
- [ ] Attach release notes
- [ ] Publish release

### Post-Release Monitoring
- [ ] Monitor CI/CD pipeline (should pass)
- [ ] Monitor bug reports (watch for issues)
- [ ] Check community feedback (Discord/Slack/GitHub)
- [ ] Monitor package downloads (PyPI)
- [ ] Verify documentation is updated

### Celebration 🎉
- [ ] Update team on completion
- [ ] Document lessons learned
- [ ] Archive cleanup plan documents

**Sign-off:** ____________ Date: ____________

---

## Emergency Rollback Procedure

### If Critical Issues Discovered
1. **STOP** - Do not proceed to next phase
2. Create hotfix branch: `git checkout -b hotfix/transcriptomics-issue`
3. Document issue in GitHub: Create issue with `critical` label
4. Assess impact:
   - Production down? → Immediate rollback
   - Tests failing? → Fix and re-test
   - Minor bug? → Fix in next release

### Rollback Commands (Phase 2-3 Only)
```bash
# Revert test changes
git checkout main -- tests/unit/agents/test_singlecell_expert.py
git checkout main -- tests/unit/agents/test_bulk_quantification_communication.py
git checkout main -- tests/integration/

# Revert public sync
git checkout main -- scripts/public_allowlist.txt
python scripts/sync_to_public.py --repo git@github.com:the-omics-os/lobster-local.git

# Revert commits
git revert <commit-hash>
```

### Rollback Commands (Phase 4 - Should NOT Happen)
```bash
# Restore agent files from last known good commit
git checkout <last-good-commit> -- lobster/agents/singlecell_expert.py
git checkout <last-good-commit> -- lobster/agents/bulk_rnaseq_expert.py

# Re-add to registry if needed (should NOT be necessary)
# Fix issue in transcriptomics_expert instead
```

---

## Sign-Off Summary

| Phase | Completion Date | Sign-Off | Notes |
|-------|----------------|----------|-------|
| **Phase 0: Tests** | ____________ | ____________ | |
| **Phase 1: Docs** | ____________ | ____________ | |
| **Phase 2: Refactor** | ____________ | ____________ | |
| **Phase 3: Public Sync** | ____________ | ____________ | |
| **Phase 4: Removal** | ____________ | ____________ | |

**Final Sign-Off:** ____________ Date: ____________

---

**Generated:** 2025-12-02
**Version:** 1.0
