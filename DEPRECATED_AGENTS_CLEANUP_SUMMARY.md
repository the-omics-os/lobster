# Deprecated Agents Cleanup - Executive Summary

**Status:** ⚠️ SAFE TO PROCEED - BUT CRITICAL TEST GAP MUST BE ADDRESSED FIRST

---

## Quick Facts

| Metric | Value |
|--------|-------|
| **Deprecated Files** | 2 files (285 KB, 6692 lines) |
| **Replacement Files** | 5 files (220 KB, 5201 lines) |
| **Production Dependencies** | 0 (SAFE) |
| **Test Dependencies** | 20 files (NEEDS REFACTORING) |
| **Documentation References** | 5 wiki pages (NEEDS UPDATE) |
| **Code Reduction** | 65 KB / 1491 lines (23% smaller) |

---

## Critical Finding: Test Coverage Gap 🔴

**BLOCKER:** New `transcriptomics_expert` architecture has **ZERO dedicated tests**

```bash
# Old agents: 712 lines of tests
tests/unit/agents/test_singlecell_expert.py              (399 lines)
tests/unit/agents/test_bulk_quantification_communication.py (313 lines)

# New agents: 0 lines of tests
find tests -name "*transcriptomics_expert*"  # Returns: nothing
```

**Why this matters:**
- Services are tested ✅
- Supervisor handoffs tested ✅
- But agent-level logic NOT tested ❌
  - Tool argument validation
  - Delegation to sub-agents
  - Error handling
  - State management

**Action Required:** Create test suite BEFORE any removal

---

## Risk Assessment

| Component | Status | Risk | Notes |
|-----------|--------|------|-------|
| Production Code | ✅ SAFE | 🟢 NONE | Zero imports found |
| Agent Registry | ✅ MIGRATED | 🟢 NONE | Already using transcriptomics_expert |
| Services | ✅ TESTED | 🟢 NONE | 85%+ coverage, shared across old/new |
| Agent Tests | ⚠️ NEW TESTS NEEDED | 🔴 HIGH | **BLOCKER** |
| Integration Tests | ⚠️ REFACTOR NEEDED | 🟡 MEDIUM | 20 files to update |
| Documentation | ⚠️ UPDATE NEEDED | 🟡 MEDIUM | 5 wiki pages outdated |
| Public Sync | ⚠️ NEEDS ATTENTION | 🟡 MEDIUM | singlecell_expert.py still public |

---

## What Can Break

### Won't Break (Safe Areas)
✅ Production code (no imports)
✅ CLI usage (supervisor routes automatically)
✅ Service layer (independently tested)
✅ Agent registry (already migrated)

### Could Break (Risk Areas)
⚠️ Test suite (20 files will fail)
⚠️ Tool argument validation (untested in new agents)
⚠️ Sub-agent delegation (untested)
⚠️ Error handling (untested)

---

## Recommended Phased Approach

### Phase 0: Create Tests (1 week) - **MUST DO FIRST**
- Create comprehensive test suite for transcriptomics architecture
- Target: 80%+ coverage
- Test delegation, error handling, state management

### Phase 1: Update Docs (1 day) - **Safe**
- Update 5 wiki pages
- Create migration guide
- Add stronger deprecation warnings

### Phase 2: Refactor Tests (1 week) - **Medium Risk**
- Delete 2 agent-specific test files
- Refactor 11 integration/system/performance tests
- Verify no regressions

### Phase 3: Remove from Public (1 day) - **Low Risk**
- Update public_allowlist.txt
- Sync to lobster-local
- Post deprecation notice

### Phase 4: Final Removal (1 day) - **Safe**
- Delete deprecated agent files
- Update CHANGELOG
- Release v2.7.0

**Total Timeline:** 2 weeks dev + 5 weeks soak/notice = **7 weeks end-to-end**

---

## Immediate Action Items

### This Week (CRITICAL)
1. **Create test suite for transcriptomics architecture**
   ```bash
   mkdir -p tests/unit/agents/transcriptomics
   # Create 4 test files (see full plan)
   ```
   **Owner:** Development team
   **Priority:** P0 (BLOCKER)

2. **Update documentation**
   - wiki/19-agent-system.md
   - wiki/15-agents-api.md
   - wiki/30-glossary.md
   - wiki/34-architecture-diagram.md
   - lobster/config/README_CONFIGURATION.md
   **Owner:** Tech writer
   **Priority:** P1

### Next Week
3. **Refactor integration tests** (11 files)
   **Owner:** QA team
   **Priority:** P1

### Next Month
4. **Update public repo** (remove from allowlist)
   **Owner:** DevOps
   **Priority:** P2

### 2+ Months
5. **Final removal** (delete files)
   **Owner:** Release manager
   **Priority:** P2

---

## Decision Matrix

### Should we proceed with cleanup?
**YES** - but Phase 0 is mandatory first

### Should we delete old tests immediately?
**NO** - refactor to new agents instead (preserve test scenarios)

### Should we remove from public repo now?
**NO** - wait until tests are passing and documentation is updated

### Should we keep deprecation warnings?
**YES** - strengthen them and add migration guide link

### What version number for removal?
**v2.7.0** (breaking change, but within major version)

---

## Key Metrics

### Lines of Code Removed
```
Old: 6692 lines across 2 files
New: 5201 lines across 5 files
Reduction: 1491 lines (23% smaller)
```

### Maintenance Benefit
- **Before:** 2 large monolithic agents (4000+ lines each)
- **After:** 1 parent + 2 sub-agents (modular, ~1500 lines each)
- **Benefit:** Easier to maintain, test, and extend

### Test Migration Effort
- Delete: 2 files (712 lines)
- Refactor: 11 files (~2000 lines)
- Create: 4 new files (~800 lines estimated)
- **Total effort:** ~2 weeks

---

## Open Questions for Decision Makers

1. **Who will create the Phase 0 test suite?** (BLOCKER)
   - Estimated effort: 1 week
   - Required coverage: 80%+
   - Priority: P0

2. **Should we extend deprecation notice beyond 2 months?**
   - Current plan: 2 months
   - Alternative: 3 months for enterprise users

3. **Should this be v2.7.0 or v3.0.0?**
   - Breaking change (removal of public imports)
   - But registry-based usage unchanged

4. **Do we need a blog post announcement?**
   - Could help with user communication
   - Explain benefits of unified architecture

---

## Success Criteria

**Phase 0 (Tests):**
- [ ] 4 new test files created
- [ ] 80%+ coverage for transcriptomics agents
- [ ] All tests passing
- [ ] No regressions

**Phase 1 (Docs):**
- [ ] 5 wiki pages updated
- [ ] Migration guide published
- [ ] Deprecation warnings strengthened

**Phase 2 (Test Refactoring):**
- [ ] 2 agent tests deleted
- [ ] 11 integration tests refactored
- [ ] Full test suite passing
- [ ] No performance regressions

**Phase 3 (Public Sync):**
- [ ] Public allowlist updated
- [ ] Changes synced to lobster-local
- [ ] GitHub issue created

**Phase 4 (Removal):**
- [ ] 2 files deleted (6692 lines)
- [ ] CHANGELOG updated
- [ ] No bug reports for 2+ weeks

---

## Bottom Line

### Can we remove these files now?
**NO** - Critical test gap must be addressed first

### What's the path forward?
1. Create comprehensive tests (1 week)
2. Update documentation (1 day)
3. Refactor existing tests (1 week)
4. Allow soak period (1 week)
5. Update public repo (1 day)
6. Allow deprecation notice (1 month)
7. Final removal (1 day)

### What's the biggest risk?
**Removing agents without testing new architecture** = potential regressions in production

### What's the biggest benefit?
**23% code reduction + better modularity** = easier maintenance and extension

---

**See full plan:** `DEPRECATED_AGENTS_CLEANUP_PLAN.md`

**Generated:** 2025-12-02
**Status:** Ready for Executive Review
